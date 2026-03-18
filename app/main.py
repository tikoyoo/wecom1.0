from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from sqlalchemy.orm import Session

from .admin import html_page, require_admin
from .config import settings
from .db import ChatMessage, Document, User, get_db, init_db
import json as _json

from .db import ExternalSendLog, ParentContact, ParentStudentBinding, StudentWeeklyMetric
from .hydro_service import get_weekly_students
from .llm import deepseek_chat
from .rag import RagIndex, add_document_with_chunks
from .wecom_api import send_text
from .wecom_crypto import WeComCrypto
from .wecom_external_api import add_msg_template_single, get_external_contact, list_external_contacts
from .wecom_kf_api import kf_send_text, kf_sync_msg
from .wecom_kf_xml import parse_kf_event_xml
from .wecom_xml import (
    build_encrypted_reply_xml,
    build_plain_text_reply,
    parse_encrypted_xml,
    parse_plain_xml,
)

logger = logging.getLogger("wecom-bot")


crypto: WeComCrypto | None = None
kf_crypto: WeComCrypto | None = None

rindex: RagIndex | None = None


def _rebuild_index(db: Session) -> None:
    global rindex
    rindex = RagIndex.from_db(db)


def _get_or_create_user(db: Session, wecom_user_id: str) -> User:
    u = db.query(User).filter(User.wecom_user_id == wecom_user_id).one_or_none()
    if u:
        return u
    u = User(wecom_user_id=wecom_user_id)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


async def answer_with_rag_and_memory(db: Session, wecom_user_id: str, question: str) -> str:
    user = _get_or_create_user(db, wecom_user_id)

    # memory
    turns = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(settings.memory_max_turns * 2)
        .all()
    )
    turns = list(reversed(turns))

    # rag
    hits = (rindex.search(question, settings.rag_top_k) if rindex else [])
    ctx_lines = []
    for i, (_chunk_id, text, _score) in enumerate(hits, start=1):
        ctx_lines.append(f"[{i}] {text.strip()}")
    ctx = "\n\n".join(ctx_lines)

    system = (
        "你是企业客服与咨询助手。优先根据【知识库】回答；如果知识库没有相关信息，明确说明并给出可执行的建议。"
        "回答要简洁、中文、结构清晰。若引用知识库，请在句末用 [1]/[2] 标注来源编号。"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if ctx:
        messages.append({"role": "system", "content": f"【知识库】\n{ctx}"})

    for t in turns:
        if t.role in ("user", "assistant"):
            messages.append({"role": t.role, "content": t.content})
    messages.append({"role": "user", "content": question})

    reply = await deepseek_chat(messages)

    db.add(ChatMessage(user_id=user.id, role="user", content=question))
    db.add(ChatMessage(user_id=user.id, role="assistant", content=reply))
    db.commit()
    return reply


async def _answer_and_push(wecom_user_id: str, question: str) -> None:
    try:
        with next(get_db()) as db:  # type: ignore[arg-type]
            reply = await answer_with_rag_and_memory(db, wecom_user_id, question)
        await send_text(touser=wecom_user_id, content=reply)
    except Exception:
        logger.exception("async push failed")
        try:
            await send_text(touser=wecom_user_id, content="抱歉，刚才处理超时或出错了。请稍后再试一次。")
        except Exception:
            logger.exception("fallback push failed")


async def _kf_answer_and_push(open_kfid: str, external_userid: str, question: str) -> None:
    try:
        with next(get_db()) as db:  # type: ignore[arg-type]
            reply = await answer_with_rag_and_memory(db, f"kf:{external_userid}", question)
        await kf_send_text(open_kfid=open_kfid, external_userid=external_userid, content=reply)
    except Exception:
        logger.exception("kf async push failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    init_db()
    with next(get_db()) as db:  # type: ignore[arg-type]
        _rebuild_index(db)
    global crypto, kf_crypto
    if settings.wecom_token and settings.wecom_encoding_aes_key and settings.wecom_corp_id:
        crypto = WeComCrypto(
            token=settings.wecom_token,
            encoding_aes_key=settings.wecom_encoding_aes_key,
            corp_id=settings.wecom_corp_id,
        )
    if settings.wecom_kf_token and settings.wecom_kf_encoding_aes_key and settings.wecom_corp_id:
        kf_crypto = WeComCrypto(
            token=settings.wecom_kf_token,
            encoding_aes_key=settings.wecom_kf_encoding_aes_key,
            corp_id=settings.wecom_corp_id,
        )
    yield


app = FastAPI(title="WeCom KB Bot", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/wecom/callback")
async def wecom_verify(msg_signature: str, timestamp: str, nonce: str, echostr: str):
    if crypto is None:
        return PlainTextResponse("wecom not configured", status_code=500)
    if not crypto.verify_signature(msg_signature, timestamp, nonce, echostr):
        return PlainTextResponse("invalid signature", status_code=403)
    try:
        plain = crypto.decrypt(echostr)
        return Response(content=plain, media_type="text/plain")
    except Exception as e:
        logger.exception("verify decrypt failed")
        return PlainTextResponse(f"decrypt failed: {e}", status_code=400)


@app.post("/wecom/callback")
async def wecom_callback(
    request: Request,
    msg_signature: str,
    timestamp: str,
    nonce: str,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
):
    if crypto is None:
        return PlainTextResponse("wecom not configured", status_code=500)
    body = await request.body()
    enc = parse_encrypted_xml(body)
    if not crypto.verify_signature(msg_signature, timestamp, nonce, enc.encrypt):
        return PlainTextResponse("invalid signature", status_code=403)

    plain_xml = crypto.decrypt(enc.encrypt)
    msg = parse_plain_xml(plain_xml)

    if msg.msg_type != "text" or not msg.content.strip():
        reply_text = "目前只支持文本咨询。请发送文字问题。"
    else:
        # WeCom requires a quick response; long LLM calls can time out.
        question = msg.content.strip()
        background.add_task(asyncio.run, _answer_and_push(msg.from_user_name, question))
        reply_text = "已收到，我正在查询并整理答案，稍后会再发一条消息给你。"

    plain_reply = build_plain_text_reply(to_user=msg.from_user_name, from_user=msg.to_user_name, content=reply_text)
    encrypt, signature, ts = crypto.encrypt(plain_reply, nonce=nonce, timestamp=timestamp)
    resp = build_encrypted_reply_xml(encrypt=encrypt, signature=signature, timestamp=ts, nonce=nonce)
    return Response(content=resp, media_type="application/xml")


@app.get("/wecom/kf/callback")
async def wecom_kf_verify(msg_signature: str, timestamp: str, nonce: str, echostr: str):
    if kf_crypto is None:
        return PlainTextResponse("wecom kf not configured", status_code=500)
    if not kf_crypto.verify_signature(msg_signature, timestamp, nonce, echostr):
        return PlainTextResponse("invalid signature", status_code=403)
    try:
        plain = kf_crypto.decrypt(echostr)
        return Response(content=plain, media_type="text/plain")
    except Exception as e:
        logger.exception("kf verify decrypt failed")
        return PlainTextResponse(f"decrypt failed: {e}", status_code=400)


@app.post("/wecom/kf/callback")
async def wecom_kf_callback(request: Request, msg_signature: str, timestamp: str, nonce: str, background: BackgroundTasks):
    if kf_crypto is None:
        return PlainTextResponse("wecom kf not configured", status_code=500)
    body = await request.body()
    enc = parse_encrypted_xml(body)
    if not kf_crypto.verify_signature(msg_signature, timestamp, nonce, enc.encrypt):
        return PlainTextResponse("invalid signature", status_code=403)

    plain_xml = kf_crypto.decrypt(enc.encrypt)
    ev = parse_kf_event_xml(plain_xml)

    # WeChat Customer Service new message notify: pull message content via sync_msg
    if ev.msg_type == "event" and ev.event == "kf_msg_or_event" and ev.token and ev.open_kfid:
        async def _pull_and_reply() -> None:
            data = await kf_sync_msg(token=ev.token, cursor="", limit=50)
            msgs = data.get("msg_list") or []
            # reply to latest text message
            for m in reversed(msgs):
                if m.get("msgtype") == "text" and m.get("external_userid") and m.get("text", {}).get("content"):
                    question = str(m["text"]["content"]).strip()
                    if question:
                        await _kf_answer_and_push(ev.open_kfid, m["external_userid"], question)
                    break

        background.add_task(asyncio.run, _pull_and_reply())

    # For event callbacks, replying "success" is sufficient.
    return PlainTextResponse("success")


# ---------------- Admin (very small) ----------------


@app.get("/admin/", response_class=HTMLResponse)
async def admin_home(_user: str = Depends(require_admin)):
    body = """
    <h2>管理后台</h2>
    <div class="card">
      <div><a href="/admin/docs">知识库管理</a></div>
      <div><a href="/admin/chats">会话记录</a></div>
      <div><a href="/admin/push">主动推送</a></div>
      <div><a href="/admin/parents">家长通讯录（客户联系）</a></div>
      <div><a href="/admin/bindings">家长-学生绑定</a></div>
      <div><a href="/admin/students">学生五维数据</a></div>
      <div><a href="/admin/reports">周报/群发</a></div>
    </div>
    """
    return html_page("Admin", body)


@app.get("/admin/docs", response_class=HTMLResponse)
async def admin_docs(db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    rows = "".join(
        f"<tr><td>{d.id}</td><td>{d.title}</td><td><code>{d.filename}</code></td><td>{d.created_at}</td></tr>" for d in docs
    )
    body = f"""
    <h2>知识库</h2>
    <div class="card">
      <form action="/admin/docs/upload" method="post" enctype="multipart/form-data">
        <div style="margin-bottom:10px">
          <label>标题</label>
          <input name="title" placeholder="例如：某机构课程咨询FAQ" />
        </div>
        <div style="margin-bottom:10px">
          <label>文件（txt/md/pdf/docx）</label>
          <input type="file" name="file" />
        </div>
        <button type="submit">上传入库</button>
      </form>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>ID</th><th>标题</th><th>文件名</th><th>时间</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Docs", body)


def _read_upload_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("gbk", errors="ignore")
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        import io

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        from docx import Document as DocxDocument
        import io

        doc = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("unsupported file type")


@app.post("/admin/docs/upload")
async def admin_docs_upload(
    title: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    data = await file.read()
    text = _read_upload_text(file.filename or "upload", data)
    if not title.strip():
        title = file.filename or "未命名文档"
    add_document_with_chunks(db, title=title.strip(), filename=file.filename or "upload", content=text)
    _rebuild_index(db)
    return Response(status_code=303, headers={"Location": "/admin/docs"})


@app.get("/admin/chats", response_class=HTMLResponse)
async def admin_chats(db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    msgs = db.query(ChatMessage).order_by(ChatMessage.created_at.desc()).limit(200).all()
    rows = "".join(
        f"<tr><td>{m.created_at}</td><td><code>{m.role}</code></td><td>{m.content[:200]}</td></tr>" for m in msgs
    )
    body = f"""
    <h2>最近会话（最多200条）</h2>
    <div class="card">
      <table>
        <thead><tr><th>时间</th><th>角色</th><th>内容(截断)</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Chats", body)


@app.get("/admin/push", response_class=HTMLResponse)
async def admin_push(_user: str = Depends(require_admin)):
    body = """
    <h2>主动推送（企业微信应用消息）</h2>
    <div class="card">
      <form action="/admin/push" method="post">
        <div style="margin-bottom:10px">
          <label>接收人 touser（企业微信 UserID，多个用 | 分隔）</label>
          <input name="touser" placeholder="zhangsan|lisi" />
        </div>
        <div style="margin-bottom:10px">
          <label>内容</label>
          <textarea name="content" rows="5" placeholder="要推送的消息"></textarea>
        </div>
        <button type="submit">发送</button>
      </form>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Push", body)


@app.post("/admin/push")
async def admin_push_post(
    touser: str = Form(...),
    content: str = Form(...),
    _user: str = Depends(require_admin),
):
    await send_text(touser=touser.strip(), content=content.strip())
    return Response(status_code=303, headers={"Location": "/admin/push"})


@app.get("/admin/parents", response_class=HTMLResponse)
async def admin_parents(db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    parents = db.query(ParentContact).order_by(ParentContact.updated_at.desc()).limit(200).all()
    rows = "".join(
        f"<tr><td>{p.external_userid[:10]}...</td><td>{p.name}</td><td>{p.remark}</td><td><code>{p.follow_userid}</code></td></tr>"
        for p in parents
    )
    body = f"""
    <h2>家长通讯录（客户联系）</h2>
    <div class="card">
      <form action="/admin/parents/sync" method="post">
        <div style="margin-bottom:10px">
          <label>跟进人 userid（sender）</label>
          <input name="follow_userid" placeholder="例如：YangShengPin" value="{settings.wecom_external_sender_id}" />
        </div>
        <button type="submit">同步外部联系人</button>
      </form>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>external_userid</th><th>名称</th><th>备注</th><th>跟进人</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Parents", body)


@app.post("/admin/parents/sync")
async def admin_parents_sync(
    follow_userid: str = Form(...),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    follow_userid = follow_userid.strip()
    ids = await list_external_contacts(follow_userid)
    for eid in ids:
        info = await get_external_contact(eid)
        ext = info.get("external_contact") or {}
        name = ext.get("name") or ""
        follow = (info.get("follow_user") or [{}])[0] if isinstance(info.get("follow_user"), list) else {}
        remark = follow.get("remark") or ""

        row = db.query(ParentContact).filter(ParentContact.external_userid == eid).one_or_none()
        if row is None:
            row = ParentContact(external_userid=eid, name=name, remark=remark, follow_userid=follow_userid)
            db.add(row)
        else:
            row.name = name
            row.remark = remark
            row.follow_userid = follow_userid
            row.updated_at = datetime.utcnow()
    db.commit()
    return Response(status_code=303, headers={"Location": "/admin/parents"})


@app.get("/admin/bindings", response_class=HTMLResponse)
async def admin_bindings(q: str = "", db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    q = (q or "").strip()
    parents_q = db.query(ParentContact)
    if q:
        like = f"%{q}%"
        parents_q = parents_q.filter((ParentContact.name.like(like)) | (ParentContact.remark.like(like)) | (ParentContact.external_userid.like(like)))
    parents = parents_q.order_by(ParentContact.updated_at.desc()).limit(300).all()

    # list existing bindings (latest first)
    bindings = db.query(ParentStudentBinding).order_by(ParentStudentBinding.created_at.desc()).limit(500).all()
    b_rows = "".join(
        f"<tr><td>{b.id}</td><td>{b.parent.name}</td><td><code>{b.parent.external_userid[:10]}...</code></td>"
        f"<td><code>{b.student_uid}</code></td><td>{b.created_at}</td>"
        f"<td><form action='/admin/bindings/delete' method='post' style='margin:0'>"
        f"<input type='hidden' name='binding_id' value='{b.id}'/>"
        f"<button class='secondary' type='submit'>解绑</button></form></td></tr>"
        for b in bindings
    )

    p_rows = ""
    for p in parents:
        p_rows += (
            "<tr>"
            f"<td>{p.name}</td>"
            f"<td>{p.remark}</td>"
            f"<td><code>{p.external_userid}</code></td>"
            f"<td><code>{p.follow_userid}</code></td>"
            "<td>"
            "<form action='/admin/bindings/set' method='post' style='display:flex;gap:8px;align-items:center;margin:0'>"
            f"<input type='hidden' name='external_userid' value='{p.external_userid}'/>"
            "<input name='student_uid' placeholder='填学生UID(可多个家长绑定同一UID)' style='width:220px'/>"
            "<button type='submit'>绑定</button>"
            "</form>"
            "</td>"
            "</tr>"
        )

    body = f"""
    <h2>家长-学生绑定</h2>
    <div class="card">
      <form method="get" action="/admin/bindings">
        <div style="display:flex;gap:10px;align-items:center">
          <input name="q" placeholder="按姓名/备注/external_userid 搜索" value="{q}" />
          <button type="submit">搜索</button>
          <a href="/admin/bindings" style="margin-left:6px">清空</a>
        </div>
      </form>
      <div style="margin-top:10px;color:#666;font-size:12px">
        说明：先在“家长通讯录”同步外部联系人，再在这里把家长绑定到 Hydro 学生 UID（支持一生多家长）。
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0">新增绑定（从家长列表）</h3>
      <table>
        <thead><tr><th>家长名称</th><th>备注</th><th>external_userid</th><th>跟进人</th><th>绑定到学生UID</th></tr></thead>
        <tbody>{p_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h3 style="margin-top:0">已有绑定（最多500条）</h3>
      <table>
        <thead><tr><th>ID</th><th>家长</th><th>external_userid</th><th>学生UID</th><th>时间</th><th>操作</th></tr></thead>
        <tbody>{b_rows}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Bindings", body)


@app.post("/admin/bindings/set")
async def admin_bindings_set(
    external_userid: str = Form(...),
    student_uid: str = Form(...),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    external_userid = external_userid.strip()
    student_uid = student_uid.strip()
    if not external_userid or not student_uid:
        raise HTTPException(status_code=400, detail="external_userid and student_uid required")

    parent = db.query(ParentContact).filter(ParentContact.external_userid == external_userid).one_or_none()
    if parent is None:
        raise HTTPException(status_code=404, detail="parent not found, sync first")

    exists = (
        db.query(ParentStudentBinding)
        .filter(ParentStudentBinding.parent_id == parent.id)
        .filter(ParentStudentBinding.student_uid == student_uid)
        .one_or_none()
    )
    if exists is None:
        db.add(ParentStudentBinding(parent_id=parent.id, student_uid=student_uid))
        db.commit()
    return Response(status_code=303, headers={"Location": "/admin/bindings"})


@app.post("/admin/bindings/delete")
async def admin_bindings_delete(
    binding_id: int = Form(...),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    row = db.query(ParentStudentBinding).filter(ParentStudentBinding.id == binding_id).one_or_none()
    if row is not None:
        db.delete(row)
        db.commit()
    return Response(status_code=303, headers={"Location": "/admin/bindings"})


@app.get("/admin/reports", response_class=HTMLResponse)
async def admin_reports(db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    latest = db.query(StudentWeeklyMetric.week_key).order_by(StudentWeeklyMetric.week_key.desc()).first()
    latest_week = latest[0] if latest else ""
    # quick list of groups from latest week (best-effort)
    group_set: set[str] = set()
    group_stats: dict[str, dict[str, int]] = {}
    if latest_week:
        rows = (
            db.query(
                StudentWeeklyMetric.student_uid,
                StudentWeeklyMetric.groups_json,
                StudentWeeklyMetric.hw_done,
                StudentWeeklyMetric.hw_total,
            )
            .filter(StudentWeeklyMetric.week_key == latest_week)
            .limit(5000)
            .all()
        )

        # precompute which student_uids have at least one parent binding
        bound_uids = {uid for (uid,) in db.query(ParentStudentBinding.student_uid).distinct().all()}

        for uid, gj, hw_done, hw_total in rows:
            try:
                gs = [g.strip() for g in _json.loads(gj or "[]") if isinstance(g, str) and g.strip()]
            except Exception:
                gs = []
            if not gs:
                gs = ["(未分组)"]
            for g in gs:
                group_set.add(g)
                st = group_stats.setdefault(g, {"students": 0, "bound_students": 0, "unfinished": 0})
                st["students"] += 1
                if str(uid) in bound_uids:
                    st["bound_students"] += 1
                if (hw_total or 0) > 0 and (hw_done or 0) < (hw_total or 0):
                    st["unfinished"] += 1
    group_options = "".join(f"<option value='{g}'></option>" for g in sorted(group_set))

    # group stats table
    stat_rows = ""
    for g in sorted(group_stats.keys()):
        st = group_stats[g]
        stat_rows += (
            f"<tr><td><code>{g}</code></td><td>{st['students']}</td><td>{st['bound_students']}</td><td>{st['unfinished']}</td>"
            f"<td><a href='/admin/reports?week={latest_week}&group={g}'>选择该分组</a></td></tr>"
        )

    logs = db.query(ExternalSendLog).order_by(ExternalSendLog.created_at.desc()).limit(50).all()
    log_rows = "".join(
        f"<tr><td>{l.created_at}</td><td><code>{l.week_key}</code></td><td><code>{l.sender_userid}</code></td>"
        f"<td>{l.group_filter or '-'}</td><td>{'是' if l.only_unfinished else '否'}</td>"
        f"<td><code>{l.student_uid}</code></td><td><code>{l.external_userid[:10]}...</code></td>"
        f"<td>{l.status}</td><td><code>{l.msgid}</code></td><td style='max-width:360px;white-space:pre-wrap'>{(l.error or '')[:120]}</td></tr>"
        for l in logs
    )

    body = f"""
    <h2>周报/群发</h2>
    <div class="card">
      <div style="color:#666;font-size:12px;margin-bottom:8px">
        提示：先在 <a href="/admin/reports/weekly/preview">预览</a> 看筛选效果，再群发；也可以在下表点“选择该分组”快速填充筛选条件。
      </div>
      <table>
        <thead><tr><th>分组/班级</th><th>学生数</th><th>已绑定学生数</th><th>未完成作业学生数</th><th>操作</th></tr></thead>
        <tbody>{stat_rows or "<tr><td colspan='5'>暂无统计（请先拉取本周数据）</td></tr>"}</tbody>
      </table>
    </div>
    <div class="card">
      <form action="/admin/reports/weekly/preview" method="post">
        <div style="margin-bottom:10px">
          <label>跟进人 sender（用于客户联系群发）</label>
          <input name="sender" value="{settings.wecom_external_sender_id}" />
        </div>
        <div style="margin-bottom:10px">
          <label>week_key（默认最新，例如：2026-W12）</label>
          <input name="week" value="{latest_week}" />
        </div>
        <div style="margin-bottom:10px">
          <label>分组/班级筛选（Hydro groups，留空表示全部）</label>
          <input name="group" list="group_list" placeholder="例如：四班" />
          <datalist id="group_list">{group_options}</datalist>
        </div>
        <div style="margin-bottom:10px">
          <label><input type="checkbox" name="only_unfinished" /> 仅未完成作业（done &lt; total 且 total&gt;0）</label>
        </div>
        <button type="submit">拉取本周数据（预览）</button>
      </form>
    </div>
    <div class="card">
      <form action="/admin/reports/weekly/send" method="post">
        <div style="margin-bottom:10px">
          <label>跟进人 sender</label>
          <input name="sender" value="{settings.wecom_external_sender_id}" />
        </div>
        <div style="margin-bottom:10px">
          <label>week_key</label>
          <input name="week" value="{latest_week}" />
        </div>
        <div style="margin-bottom:10px">
          <label>分组/班级筛选</label>
          <input name="group" list="group_list" placeholder="例如：四班" />
        </div>
        <div style="margin-bottom:10px">
          <label><input type="checkbox" name="only_unfinished" /> 仅未完成作业</label>
        </div>
        <button type="submit">一键群发本周周报（按绑定关系）</button>
      </form>
      <div style="margin-top:8px;color:#666;font-size:12px">
        注：群发会对每个已绑定家长创建一条 externalcontact 群发任务，并记录结果到“最近发送记录”。
      </div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">最近发送记录（最多50条）</h3>
      <table>
        <thead><tr>
          <th>时间</th><th>周</th><th>sender</th><th>分组</th><th>仅未完成</th>
          <th>UID</th><th>external_userid</th><th>状态</th><th>msgid</th><th>错误</th>
        </tr></thead>
        <tbody>{log_rows}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Reports", body)


@app.get("/admin/students", response_class=HTMLResponse)
async def admin_students(week: str = "", q: str = "", db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    week = (week or "").strip()
    q = (q or "").strip()

    # default to latest week_key
    if not week:
        latest = db.query(StudentWeeklyMetric.week_key).order_by(StudentWeeklyMetric.week_key.desc()).first()
        week = latest[0] if latest else ""

    rows_q = db.query(StudentWeeklyMetric)
    if week:
        rows_q = rows_q.filter(StudentWeeklyMetric.week_key == week)
    if q:
        like = f"%{q}%"
        rows_q = rows_q.filter((StudentWeeklyMetric.student_uid.like(like)) | (StudentWeeklyMetric.name.like(like)))
    rows = rows_q.order_by(StudentWeeklyMetric.rank.asc()).limit(500).all()

    tr = ""
    for r in rows:
        try:
            groups = ", ".join(_json.loads(r.groups_json or "[]"))
        except Exception:
            groups = ""
        tr += (
            f"<tr><td><code>{r.student_uid}</code></td><td>{r.name}</td><td>{r.rank}</td>"
            f"<td>{groups}</td><td>{r.hw_title}</td><td>{r.hw_done}/{r.hw_total}</td>"
            f"<td>{r.week_ac}</td><td>{r.week_submits}</td><td>{r.active_days}</td><td>{r.last_active}</td></tr>"
        )

    body = f"""
    <h2>学生五维数据（按周）</h2>
    <div class="card">
      <form method="get" action="/admin/students">
        <div style="display:flex;gap:10px;align-items:center">
          <input name="week" placeholder="例如：2026-W12" value="{week}" style="width:180px" />
          <input name="q" placeholder="按UID/姓名搜索" value="{q}" />
          <button type="submit">查询</button>
          <a href="/admin/students" style="margin-left:6px">最新</a>
        </div>
      </form>
      <div style="margin-top:10px;color:#666;font-size:12px">
        数据来源：Hydro 周数据拉取。每次你点击“周报预览/群发”都会刷新并写入这一周数据；后续会改成每周一自动更新。
      </div>
    </div>
    <div class="card">
      <table>
        <thead><tr>
          <th>UID</th><th>姓名</th><th>排名</th><th>分组</th><th>作业</th><th>作业进度</th>
          <th>本周AC</th><th>本周提交</th><th>活跃天数</th><th>最近活跃</th>
        </tr></thead>
        <tbody>{tr}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Students", body)


def _render_weekly_report(s: dict) -> str:
    hw = s.get("hw_info") or {}
    return (
        f"📊 {s.get('name','')} 的学习周报\n\n"
        f"👤 UID：{s.get('uid')}\n"
        f"📈 当前排名：第 {s.get('rank','-')} 名\n\n"
        f"📚 本周作业：{hw.get('title','无')}\n"
        f"✅ 完成情况：{hw.get('done',0)}/{hw.get('total',0)}\n\n"
        f"💡 本周AC：{(s.get('dim2') or {}).get('ac',0)} 题\n"
        f"📝 本周提交：{(s.get('dim2') or {}).get('submits',0)} 次\n"
        f"🔥 活跃天数：{(s.get('dim3') or {}).get('days',0)} 天（最近：{(s.get('dim3') or {}).get('last','无')}）\n\n"
        f"有问题随时联系我。"
    )


@app.post("/admin/reports/weekly/preview", response_class=HTMLResponse)
async def admin_weekly_preview(
    sender: str = Form(...),
    week: str = Form(""),
    group: str = Form(""),
    only_unfinished: str = Form(""),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    sender = sender.strip()
    week = (week or "").strip()
    group = (group or "").strip()
    only_unfinished_bool = bool(only_unfinished)

    # refresh from hydro and persist weekly metrics
    get_weekly_students(db, force_refresh=True)

    if not week:
        latest = db.query(StudentWeeklyMetric.week_key).order_by(StudentWeeklyMetric.week_key.desc()).first()
        week = latest[0] if latest else ""

    rows_q = db.query(StudentWeeklyMetric)
    if week:
        rows_q = rows_q.filter(StudentWeeklyMetric.week_key == week)
    rows = rows_q.order_by(StudentWeeklyMetric.rank.asc()).all()

    filtered: list[StudentWeeklyMetric] = []
    for r in rows:
        if group:
            try:
                gs = _json.loads(r.groups_json or "[]")
                if group not in gs:
                    continue
            except Exception:
                continue
        if only_unfinished_bool and (r.hw_total <= 0 or r.hw_done >= r.hw_total):
            continue
        filtered.append(r)

    body_rows = ""
    for r in filtered[:80]:
        body_rows += f"<tr><td><code>{r.student_uid}</code></td><td>{r.name}</td><td>{r.rank}</td><td>{r.hw_done}/{r.hw_total}</td><td>{r.week_ac}</td><td>{r.active_days}</td></tr>"

    bindings = db.query(ParentStudentBinding).all()
    uid_has_parent = {b.student_uid for b in bindings}
    will_send_students = [r for r in filtered if r.student_uid in uid_has_parent]
    will_send_parents = sum(1 for b in bindings if any(r.student_uid == b.student_uid for r in will_send_students))

    body = f"""
    <h2>本周数据预览（前50）</h2>
    <div class="card">
      <div><code>sender={sender}</code></div>
      <div><code>week={week or '-'}</code> <code>group={group or '-'}</code> <code>only_unfinished={'1' if only_unfinished_bool else '0'}</code></div>
      <div style="margin-top:6px;color:#666;font-size:12px">
        筛选后学生数：<b>{len(filtered)}</b>；其中已绑定可发送学生数：<b>{len(will_send_students)}</b>；预计发送家长数（按绑定条数）：<b>{will_send_parents}</b>
      </div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>UID</th><th>姓名</th><th>排名</th><th>作业</th><th>本周AC</th><th>活跃天数</th></tr></thead>
        <tbody>{body_rows}</tbody>
      </table>
    </div>
    <div><a href="/admin/reports">返回</a></div>
    """
    return html_page("Weekly Preview", body)


@app.post("/admin/reports/weekly/send", response_class=HTMLResponse)
async def admin_weekly_send(
    sender: str = Form(...),
    week: str = Form(""),
    group: str = Form(""),
    only_unfinished: str = Form(""),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    sender = sender.strip()
    if not sender:
        raise HTTPException(status_code=400, detail="sender required")

    week = (week or "").strip()
    group = (group or "").strip()
    only_unfinished_bool = bool(only_unfinished)

    # refresh from hydro and persist weekly metrics
    weekly = get_weekly_students(db, force_refresh=True)
    # infer week key if not provided
    if not week:
        latest = db.query(StudentWeeklyMetric.week_key).order_by(StudentWeeklyMetric.week_key.desc()).first()
        week = latest[0] if latest else ""

    # build dict for report rendering from raw hydro payload (keeps existing format)
    by_uid = {str(s.get("uid")): s for s in weekly if s.get("uid") is not None}

    bindings = db.query(ParentStudentBinding).all()
    ok = 0
    fail = 0
    errs: list[str] = []
    for b in bindings:
        s_raw = by_uid.get(b.student_uid)
        if not s_raw:
            continue

        # apply filters using metrics table (more reliable)
        m = (
            db.query(StudentWeeklyMetric)
            .filter(StudentWeeklyMetric.week_key == week)
            .filter(StudentWeeklyMetric.student_uid == b.student_uid)
            .one_or_none()
        )
        if m is None:
            continue
        if group:
            try:
                gs = _json.loads(m.groups_json or "[]")
                if group not in gs:
                    continue
            except Exception:
                continue
        if only_unfinished_bool and (m.hw_total <= 0 or m.hw_done >= m.hw_total):
            continue

        content = _render_weekly_report(s_raw)
        try:
            resp = await add_msg_template_single(external_userid=b.parent.external_userid, content=content, sender_userid=sender)
            db.add(
                ExternalSendLog(
                    week_key=week,
                    sender_userid=sender,
                    group_filter=group,
                    only_unfinished=1 if only_unfinished_bool else 0,
                    student_uid=b.student_uid,
                    external_userid=b.parent.external_userid,
                    status="ok",
                    msgid=str(resp.get("msgid") or resp.get("msg_id") or ""),
                    response_json=_json.dumps(resp, ensure_ascii=False),
                    error="",
                )
            )
            db.commit()
            ok += 1
        except Exception as e:
            db.add(
                ExternalSendLog(
                    week_key=week,
                    sender_userid=sender,
                    group_filter=group,
                    only_unfinished=1 if only_unfinished_bool else 0,
                    student_uid=b.student_uid,
                    external_userid=b.parent.external_userid,
                    status="fail",
                    msgid="",
                    response_json="",
                    error=str(e),
                )
            )
            db.commit()
            fail += 1
            errs.append(f"{b.parent.external_userid[:10]}... uid={b.student_uid}: {e}")

    err_html = "<br/>".join(errs[:20])
    body = f"""
    <h2>周报群发完成</h2>
    <div class="card">
      成功：<b>{ok}</b>，失败：<b>{fail}</b>
    </div>
    <div class="card">
      <div style="color:#666">失败示例（最多20条）</div>
      <div style="white-space:pre-wrap">{err_html or "无"}</div>
    </div>
    <div><a href="/admin/reports">返回</a></div>
    """
    return html_page("Weekly Send", body)

