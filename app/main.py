from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from sqlalchemy.orm import Session

from .admin import html_page, require_admin
from .config import settings
from .db import ChatMessage, Document, User, get_db, init_db
from .llm import deepseek_chat
from .rag import RagIndex, add_document_with_chunks
from .wecom_api import send_text
from .wecom_crypto import WeComCrypto
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

