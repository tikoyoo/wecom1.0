from __future__ import annotations

import html
import json
import logging
import re
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .admin import html_page, require_admin
from .config import settings
from .db import (
    BindingNameRequest,
    ChatMessage,
    Document,
    ParentStudentBinding,
    StudentRecord,
    User,
    get_db,
    init_db,
)
from .llm import deepseek_chat
from .rag import RagIndex, add_document_with_chunks
from .wecom_api import send_text
from .wecom_crypto import WeComCrypto
from .wecom_xml import (
    build_encrypted_reply_xml,
    build_plain_text_reply,
    parse_encrypted_xml,
    parse_plain_xml,
)

logger = logging.getLogger("wecom-bot")


crypto: WeComCrypto | None = None

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


async def answer_with_rag_and_memory(
    db: Session,
    wecom_user_id: str,
    question: str,
    *,
    extra_system: str | None = None,
) -> str:
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
    if extra_system:
        system = f"{system}\n\n{extra_system}"
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    init_db()
    with next(get_db()) as db:  # type: ignore[arg-type]
        _rebuild_index(db)
    global crypto
    if settings.wecom_token and settings.wecom_encoding_aes_key and settings.wecom_corp_id:
        crypto = WeComCrypto(
            token=settings.wecom_token,
            encoding_aes_key=settings.wecom_encoding_aes_key,
            corp_id=settings.wecom_corp_id,
        )
    yield


app = FastAPI(title="WeCom KB Bot", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/mini-status")
async def api_mini_status():
    """给小程序/运维自检：是否配置了微信 AppID/Secret（不返回 Secret）。"""
    aid = (settings.wx_mini_appid or "").strip()
    has_secret = bool((settings.wx_mini_secret or "").strip())
    return {
        "wx_mini_configured": bool(aid and has_secret),
        "appid": aid or None,
        "secret_configured": has_secret,
        "hint": "小程序工具里的 AppID 须与此处一致；改 .env 后必须重启后端进程。",
    }


def _norm_student_name(name: str) -> str:
    s = (name or "").strip()
    return re.sub(r"\s+", "", s)


class WxLoginIn(BaseModel):
    code: str = Field(min_length=1)


class BindByStudentNameIn(BaseModel):
    openid: str = Field(min_length=1)
    student_name: str = Field(min_length=1)
    parent_name: str = ""


class ChatIn(BaseModel):
    openid: str = Field(min_length=1)
    student_uid: str = Field(min_length=1)
    message: str = Field(min_length=1)


@app.post("/api/wx-login")
async def api_wx_login(body: WxLoginIn):
    if not settings.wx_mini_appid or not settings.wx_mini_secret:
        logger.warning("wx-login 拒绝: 未配置 WX_MINI_APPID / WX_MINI_SECRET")
        raise HTTPException(status_code=503, detail="微信小程序未配置（WX_MINI_APPID / WX_MINI_SECRET）")
    url = "https://api.weixin.qq.com/sns/jscode2session"
    params = {
        "appid": settings.wx_mini_appid.strip(),
        "secret": settings.wx_mini_secret.strip(),
        "js_code": body.code.strip(),
        "grant_type": "authorization_code",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            try:
                data = r.json()
            except Exception:
                logger.exception("jscode2session 响应非 JSON: %s", r.text[:200])
                raise HTTPException(status_code=502, detail="微信接口返回非 JSON，请稍后重试")
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.exception("请求 jscode2session 失败")
        raise HTTPException(status_code=502, detail=f"无法连接微信接口: {e}") from e

    errcode = data.get("errcode", 0) or 0
    if errcode:
        errmsg = data.get("errmsg") or f"wx err {errcode}"
        logger.warning("jscode2session errcode=%s errmsg=%s", errcode, errmsg)
        raise HTTPException(status_code=400, detail=f"{errmsg} (errcode={errcode})")
    openid = data.get("openid")
    if not openid:
        logger.warning("jscode2session 无 openid 字段: %s", data)
        raise HTTPException(status_code=400, detail="no openid in wx response")
    return {"openid": openid}


@app.get("/api/binding-status")
async def api_binding_status(openid: str = "", db: Session = Depends(get_db)):
    if not (openid or "").strip():
        return {"approved_students": []}
    rows = (
        db.query(ParentStudentBinding)
        .filter(ParentStudentBinding.openid == openid.strip())
        .order_by(ParentStudentBinding.created_at.asc())
        .all()
    )
    return {"approved_students": [r.student_uid for r in rows]}


@app.post("/api/bind-by-student-name")
async def api_bind_by_student_name(body: BindByStudentNameIn, db: Session = Depends(get_db)):
    openid = body.openid.strip()
    raw_name = body.student_name.strip()
    key = _norm_student_name(raw_name)
    if not openid or not key:
        raise HTTPException(status_code=400, detail="openid and student_name required")

    existing = (
        db.query(ParentStudentBinding).filter(ParentStudentBinding.openid == openid).order_by(ParentStudentBinding.id.asc()).first()
    )
    if existing:
        return {"status": "approved", "student_uid": existing.student_uid}

    matches = db.query(StudentRecord).filter(StudentRecord.name_key == key).all()
    if not matches:
        return {"status": "not_found"}

    if len(matches) == 1:
        uid = matches[0].student_uid
        db.add(ParentStudentBinding(openid=openid, student_uid=uid))
        db.commit()
        return {"status": "approved", "student_uid": uid}

    cand = [m.student_uid for m in matches]
    db.add(
        BindingNameRequest(
            openid=openid,
            student_name_submitted=raw_name,
            candidates_json=json.dumps(cand, ensure_ascii=False),
            status="pending",
        )
    )
    db.commit()
    return {"status": "pending", "candidates": cand}


@app.post("/api/chat")
async def api_chat(body: ChatIn, db: Session = Depends(get_db)):
    openid = body.openid.strip()
    student_uid = body.student_uid.strip()
    msg = body.message.strip()
    if not openid or not student_uid or not msg:
        raise HTTPException(status_code=400, detail="invalid body")

    ok = (
        db.query(ParentStudentBinding)
        .filter(ParentStudentBinding.openid == openid, ParentStudentBinding.student_uid == student_uid)
        .first()
    )
    if not ok:
        raise HTTPException(status_code=403, detail="not bound for this openid/student_uid")

    extra = f"【会话上下文】家长已通过小程序认证（openid）。当前绑定学生 student_uid={student_uid}。"
    reply = await answer_with_rag_and_memory(db, f"mp:{openid}", msg, extra_system=extra)
    return {"reply": reply}


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
async def wecom_callback(request: Request, msg_signature: str, timestamp: str, nonce: str, db: Session = Depends(get_db)):
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
        reply_text = await answer_with_rag_and_memory(db, msg.from_user_name, msg.content.strip())

    plain_reply = build_plain_text_reply(to_user=msg.from_user_name, from_user=msg.to_user_name, content=reply_text)
    encrypt, signature, ts = crypto.encrypt(plain_reply, nonce=nonce, timestamp=timestamp)
    resp = build_encrypted_reply_xml(encrypt=encrypt, signature=signature, timestamp=ts, nonce=nonce)
    return Response(content=resp, media_type="application/xml")


# ---------------- Admin (very small) ----------------


@app.get("/admin/", response_class=HTMLResponse)
async def admin_home(_user: str = Depends(require_admin)):
    body = """
    <h2>管理后台</h2>
    <div class="card">
      <div><a href="/admin/docs">知识库管理</a></div>
      <div><a href="/admin/chats">会话记录</a></div>
      <div><a href="/admin/students">学生名录（小程序姓名匹配）</a></div>
      <div><a href="/admin/binding-requests">待审核绑定</a></div>
      <div><a href="/admin/push">主动推送</a></div>
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


@app.get("/admin/students", response_class=HTMLResponse)
async def admin_students(db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    students = db.query(StudentRecord).order_by(StudentRecord.id.desc()).limit(500).all()
    rows = "".join(
        f"<tr><td>{s.id}</td><td><code>{html.escape(s.student_uid)}</code></td>"
        f"<td>{html.escape(s.display_name)}</td><td><code>{html.escape(s.name_key)}</code></td>"
        f"<td>{s.created_at}</td></tr>"
        for s in students
    )
    body = f"""
    <h2>学生名录</h2>
    <p>小程序家长发送<strong>孩子姓名</strong>时，会与这里的<strong>规范化姓名</strong>（去空白）做<strong>精确匹配</strong>。
    多条同名会进入「待审核绑定」。</p>
    <div class="card">
      <form action="/admin/students/add" method="post">
        <div style="margin-bottom:10px">
          <label>student_uid（Hydro UID 或业务唯一 ID）</label>
          <input name="student_uid" required placeholder="例如 1001" />
        </div>
        <div style="margin-bottom:10px">
          <label>姓名（显示名，匹配时会去掉所有空白）</label>
          <input name="display_name" required placeholder="例如 张 睿 宸" />
        </div>
        <button type="submit">添加</button>
      </form>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>ID</th><th>student_uid</th><th>姓名</th><th>name_key</th><th>时间</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Students", body)


@app.post("/admin/students/add")
async def admin_students_add(
    student_uid: str = Form(...),
    display_name: str = Form(...),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    uid = student_uid.strip()
    dn = display_name.strip()
    nk = _norm_student_name(dn)
    if not uid or not nk:
        return Response(status_code=303, headers={"Location": "/admin/students"})
    ex = db.query(StudentRecord).filter(StudentRecord.student_uid == uid).one_or_none()
    if ex:
        ex.display_name = dn
        ex.name_key = nk
    else:
        db.add(StudentRecord(student_uid=uid, display_name=dn, name_key=nk))
    db.commit()
    return Response(status_code=303, headers={"Location": "/admin/students"})


@app.get("/admin/binding-requests", response_class=HTMLResponse)
async def admin_binding_requests(db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    reqs = db.query(BindingNameRequest).order_by(BindingNameRequest.id.desc()).limit(200).all()
    blocks = []
    for br in reqs:
        try:
            cand = json.loads(br.candidates_json or "[]")
        except json.JSONDecodeError:
            cand = []
        cand_opts = "".join(
            f'<option value="{html.escape(c, quote=True)}">{html.escape(c)}</option>' for c in cand
        )
        st = html.escape(br.status)
        resolve_form = ""
        if br.status == "pending" and cand:
            resolve_form = f"""
              <form action="/admin/binding-requests/resolve" method="post" style="margin-top:10px">
                <input type="hidden" name="request_id" value="{br.id}" />
                <label>选择正确 student_uid</label>
                <select name="student_uid">{cand_opts}</select>
                <button type="submit" style="margin-left:8px">通过并绑定</button>
              </form>
            """
        blocks.append(
            f"""
            <div class="card">
              <div><strong>#{br.id}</strong> 状态：<code>{st}</code> 时间：{br.created_at}</div>
              <div>openid：<code>{html.escape(br.openid)}</code></div>
              <div>提交姓名：{html.escape(br.student_name_submitted)}</div>
              <div>候选 UID：{html.escape(json.dumps(cand, ensure_ascii=False))}</div>
              {resolve_form}
            </div>
            """
        )
    body = f"""
    <h2>待审核绑定（姓名多条匹配）</h2>
    <p>审核通过后，会为该 openid 写入 <code>parent_student_bindings</code>。</p>
    {"".join(blocks) if blocks else "<p>暂无记录。</p>"}
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Binding requests", body)


@app.post("/admin/binding-requests/resolve")
async def admin_binding_requests_resolve(
    request_id: int = Form(...),
    student_uid: str = Form(...),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    br = db.query(BindingNameRequest).filter(BindingNameRequest.id == request_id).one_or_none()
    if not br:
        return Response(status_code=303, headers={"Location": "/admin/binding-requests"})
    try:
        cand = json.loads(br.candidates_json or "[]")
    except json.JSONDecodeError:
        cand = []
    su = student_uid.strip()
    if su not in cand:
        return Response(status_code=303, headers={"Location": "/admin/binding-requests"})
    br.status = "approved"
    br.resolved_student_uid = su
    exb = (
        db.query(ParentStudentBinding)
        .filter(ParentStudentBinding.openid == br.openid, ParentStudentBinding.student_uid == su)
        .one_or_none()
    )
    if not exb:
        db.add(ParentStudentBinding(openid=br.openid, student_uid=su))
    db.commit()
    return Response(status_code=303, headers={"Location": "/admin/binding-requests"})

