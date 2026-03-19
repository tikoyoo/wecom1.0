from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .admin import html_page, require_admin
from .config import settings
from .db import (
    BindingNameRequest,
    ChatMessage,
    Document,
    ExternalContact,
    ExternalSendLog,
    ParentStudentBinding,
    StudentRecord,
    StudentWeeklyMetric,
    User,
    get_db,
    init_db,
)
from .hydro_service import get_today_students_stats, get_weekly_students
from .llm import deepseek_chat
from .rag import RagIndex, add_document_with_chunks
from .reports_service import render_weekly_report, send_weekly_reports
from .wecom_api import send_text
from .wecom_external_api import add_msg_template_single, get_external_contact, list_external_userids
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
weekly_scheduler_task: asyncio.Task | None = None
WECOM_REPLY_MAX_CHARS = 1800


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
    logger.info(
        "rag search: user=%s q=%s hits=%s",
        wecom_user_id,
        (question or "")[:120],
        len(hits),
    )
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

    # If KB hits exist, avoid old memory overriding retrieval-grounded answers.
    # This keeps WeCom and mini-program answers more consistent for same question.
    if not hits:
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
    global weekly_scheduler_task
    weekly_scheduler_task = asyncio.create_task(_weekly_scheduler_loop())
    yield
    if weekly_scheduler_task:
        weekly_scheduler_task.cancel()
        try:
            await weekly_scheduler_task
        except asyncio.CancelledError:
            pass


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


def _latest_week_key(db: Session) -> str:
    latest = db.query(StudentWeeklyMetric.week_key).order_by(StudentWeeklyMetric.week_key.desc()).first()
    return latest[0] if latest else ""


def _weekly_updates_dir() -> Path:
    p = Path(settings.data_dir) / "weekly_student_updates"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _weekly_schedule_file() -> Path:
    p = Path(settings.data_dir) / "weekly_report_schedule.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _weekly_template_file() -> Path:
    p = Path(settings.data_dir) / "weekly_report_template.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _default_weekly_template() -> str:
    return (
        "📊 {name} 的学习周报\n\n"
        "👤 HYDRO ID：{uid}\n"
        "📈 当前排名：第 {rank} 名\n\n"
        "📚 本周作业：{hw_title}\n"
        "✅ 完成情况：{hw_done}/{hw_total}\n\n"
        "💡 本周AC：{week_ac} 题\n"
        "📝 本周提交：{week_submits} 次\n"
        "🔥 活跃天数：{active_days} 天（最近：{last_active}）\n\n"
        "如需辅导建议，请直接回复本消息。"
    )


def _load_weekly_template() -> str:
    fp = _weekly_template_file()
    if not fp.exists():
        return _default_weekly_template()
    try:
        t = fp.read_text(encoding="utf-8").strip()
        return t or _default_weekly_template()
    except Exception:
        return _default_weekly_template()


def _save_weekly_template(text: str) -> None:
    fp = _weekly_template_file()
    content = (text or "").strip() or _default_weekly_template()
    fp.write_text(content, encoding="utf-8")


def _load_weekly_schedule() -> dict[str, object]:
    fp = _weekly_schedule_file()
    if not fp.exists():
        return {
            "enabled": False,
            "time_hhmm": settings.weekly_report_default_time or "07:30",
            "group": "",
            "only_unfinished": False,
            "last_run_date": "",
        }
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        return {
            "enabled": bool(data.get("enabled", False)),
            "time_hhmm": str(data.get("time_hhmm") or settings.weekly_report_default_time or "07:30"),
            "group": str(data.get("group") or ""),
            "only_unfinished": bool(data.get("only_unfinished", False)),
            "last_run_date": str(data.get("last_run_date") or ""),
        }
    except Exception:
        return {
            "enabled": False,
            "time_hhmm": settings.weekly_report_default_time or "07:30",
            "group": "",
            "only_unfinished": False,
            "last_run_date": "",
        }


def _save_weekly_schedule(data: dict[str, object]) -> None:
    fp = _weekly_schedule_file()
    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_op_ids() -> set[str]:
    raw = (settings.wecom_weekly_operator_ids or "").strip()
    if not raw:
        return set()
    return {x.strip().lower() for x in re.split(r"[,\s|]+", raw) if x.strip()}


def _guess_uid_from_text(*texts: str) -> str:
    for t in texts:
        s = (t or "").strip()
        if not s:
            continue
        # 常见写法：37 / id:37 / uid=abc_01
        m = re.search(r"(?:uid|id|学号|hydro)[\s:=：-]*([A-Za-z0-9_-]{1,32})", s, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        if re.fullmatch(r"[A-Za-z0-9_-]{1,32}", s):
            return s
    return ""


def _extract_json_object(text: str) -> dict[str, object] | None:
    s = (text or "").strip()
    if not s:
        return None
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _summarize_weekly_group_stats(db: Session, group: str = "") -> str:
    wk = _latest_week_key(db)
    if not wk:
        return "暂无周数据，请先同步 Hydro 周数据。"
    rows = db.query(StudentWeeklyMetric).filter(StudentWeeklyMetric.week_key == wk).all()
    if not rows:
        return "暂无周数据，请先同步 Hydro 周数据。"
    filt: list[StudentWeeklyMetric] = []
    for r in rows:
        if not group:
            filt.append(r)
            continue
        try:
            gs = json.loads(r.groups_json or "[]")
        except Exception:
            gs = []
        if group in gs:
            filt.append(r)
    if not filt:
        return f"周 {wk} 未找到班级 {group} 的学生数据。"
    stu = len(filt)
    sum_sub = sum(int(x.week_submits or 0) for x in filt)
    sum_ac = sum(int(x.week_ac or 0) for x in filt)
    avg_sub = round(sum_sub / stu, 2) if stu else 0
    avg_ac = round(sum_ac / stu, 2) if stu else 0
    return (
        f"周统计（{wk}）{('班级=' + group) if group else '全量'}\n"
        f"学生数：{stu}\n"
        f"总提交：{sum_sub}，总AC：{sum_ac}\n"
        f"人均提交：{avg_sub}，人均AC：{avg_ac}"
    )


def _summarize_today_stats(today_rows: list[dict], group: str = "") -> str:
    filt: list[dict] = []
    for r in today_rows:
        gs = [str(x) for x in (r.get("groups") or [])]
        if group and group not in gs:
            continue
        filt.append(r)
    if not filt:
        return f"今天未找到{('班级=' + group) if group else ''}学生数据。"
    stu = len(filt)
    sum_sub = sum(int(r.get("today_submits") or 0) for r in filt)
    sum_ac = sum(int(r.get("today_ac") or 0) for r in filt)
    top = sorted(filt, key=lambda x: (int(x.get("today_ac") or 0), int(x.get("today_submits") or 0)), reverse=True)[:5]
    top_lines = "\n".join(
        f"- {x.get('name','')}({x.get('uid','')}): 提交{int(x.get('today_submits') or 0)}，AC{int(x.get('today_ac') or 0)}"
        for x in top
    )
    return (
        f"今日统计 {('班级=' + group) if group else '全量'}\n"
        f"学生数：{stu}\n"
        f"总提交：{sum_sub}，总AC：{sum_ac}\n"
        f"Top5：\n{top_lines}"
    )


async def _send_custom_message_to_scope(
    db: Session,
    *,
    sender_userid: str,
    message: str,
    scope: str = "all",
    student_uid: str = "",
    group: str = "",
) -> tuple[int, int, int]:
    msg = (message or "").strip()
    if not msg:
        return (0, 0, 0)
    scope = (scope or "all").strip().lower()
    su = (student_uid or "").strip()
    gp = (group or "").strip()

    week = _latest_week_key(db)
    week_groups: dict[str, list[str]] = defaultdict(list)
    if week:
        wrows = db.query(StudentWeeklyMetric).filter(StudentWeeklyMetric.week_key == week).all()
        for w in wrows:
            try:
                gs = [str(x).strip() for x in json.loads(w.groups_json or "[]")]
            except Exception:
                gs = []
            week_groups[w.student_uid] = [x for x in gs if x]

    targets = db.query(ParentStudentBinding).all()
    ok = 0
    fail = 0
    skip = 0
    for b in targets:
        ext = (b.external_userid or "").strip()
        if not ext:
            skip += 1
            continue
        if scope == "student" and su and b.student_uid != su:
            continue
        if scope == "group" and gp and gp not in week_groups.get(b.student_uid, []):
            continue
        try:
            await add_msg_template_single(external_userid=ext, content=msg, sender_userid=sender_userid)
            ok += 1
        except Exception:
            fail += 1
    return (ok, fail, skip)


async def _parse_operator_intent(text: str) -> dict[str, object]:
    raw = (text or "").strip()
    quick = re.match(r"^发送今天做了多少道题给学生ID\s*([A-Za-z0-9_-]+)$", raw)
    if quick:
        return {
            "action": "send_today_student",
            "scope": "student",
            "student_uid": quick.group(1),
            "group": "",
            "message": "",
        }
    messages = [
        {
            "role": "system",
            "content": (
                "你是企业微信运营指令解析器。请把用户命令解析成 JSON，不要输出其它文字。\n"
                "字段：action, scope, student_uid, group, message。\n"
                "action 只允许：send_weekly, send_custom, send_today_student, stats_weekly, stats_today。\n"
                "scope 只允许：all, group, student, me。\n"
                "无法确定时：action=unknown，scope=all。"
            ),
        },
        {"role": "user", "content": raw},
    ]
    out = await deepseek_chat(messages, temperature=0.0)
    obj = _extract_json_object(out) or {}
    return {
        "action": str(obj.get("action") or "unknown").strip().lower(),
        "scope": str(obj.get("scope") or "all").strip().lower(),
        "student_uid": str(obj.get("student_uid") or "").strip(),
        "group": str(obj.get("group") or "").strip(),
        "message": str(obj.get("message") or "").strip(),
    }


def _looks_like_operator_command(cmd: str) -> bool:
    s = (cmd or "").strip()
    if not s:
        return False
    # 强约束：仅 #s 开头视为运维指令（例如：#s 统计今天CSP-J4班做题数据）
    return bool(re.match(r"^\s*[#＃]s(\s+|$)", s, flags=re.IGNORECASE))


async def _weekly_scheduler_loop() -> None:
    while True:
        try:
            sch = _load_weekly_schedule()
            if not sch.get("enabled"):
                await asyncio.sleep(15)
                continue
            hhmm = str(sch.get("time_hhmm") or "07:30")
            now = datetime.now()
            now_hhmm = now.strftime("%H:%M")
            today = now.strftime("%Y-%m-%d")
            if now_hhmm == hhmm and str(sch.get("last_run_date") or "") != today:
                with next(get_db()) as db:  # type: ignore[arg-type]
                    res = await send_weekly_reports(
                        db,
                        group=str(sch.get("group") or ""),
                        only_unfinished=bool(sch.get("only_unfinished", False)),
                        force_refresh=True,
                        template_text=_load_weekly_template(),
                    )
                    logger.info(
                        "weekly scheduler sent: week=%s group=%s ok=%s fail=%s skip=%s",
                        res.week_key,
                        res.group,
                        res.ok,
                        res.fail,
                        res.skip,
                    )
                sch["last_run_date"] = today
                _save_weekly_schedule(sch)
            await asyncio.sleep(20)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("weekly scheduler loop error")
            await asyncio.sleep(30)


async def _send_deferred_wecom_reply(from_user: str, text: str) -> None:
    """Compute answer asynchronously and push via app message."""
    try:
        with next(get_db()) as db:  # type: ignore[arg-type]
            reply = await answer_with_rag_and_memory(db, from_user, text)
        if len(reply) > WECOM_REPLY_MAX_CHARS:
            reply = reply[:WECOM_REPLY_MAX_CHARS] + "\n\n（内容较长，已截断）"
        await send_text(touser=from_user, content=reply)
        logger.info("deferred wecom reply sent: to=%s len=%s", from_user, len(reply))
    except Exception:
        logger.exception("deferred wecom reply failed: to=%s", from_user)


def _dump_weekly_snapshot_file(db: Session, week_key: str) -> dict[str, object]:
    wk = (week_key or "").strip()
    if not wk:
        return {"week_key": "", "rows": 0, "filename": ""}
    rows = db.query(StudentWeeklyMetric).filter(StudentWeeklyMetric.week_key == wk).all()
    payload: list[dict[str, object]] = []
    for r in rows:
        try:
            groups = json.loads(r.groups_json or "[]")
        except json.JSONDecodeError:
            groups = []
        payload.append(
            {
                "week_key": r.week_key,
                "student_uid": r.student_uid,
                "name": r.name,
                "rank": r.rank,
                "groups": groups,
                "hw_title": r.hw_title,
                "hw_done": r.hw_done,
                "hw_total": r.hw_total,
                "week_submits": r.week_submits,
                "week_ac": r.week_ac,
                "active_days": r.active_days,
                "last_active": r.last_active,
                "updated_at": str(r.updated_at),
            }
        )
    name = f"{wk}.json"
    out = _weekly_updates_dir() / name
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"week_key": wk, "rows": len(payload), "filename": name}


def _sync_student_records_from_weekly(
    db: Session,
    *,
    week_key: str = "",
    force_refresh: bool = False,
) -> dict[str, object]:
    refresh_error = ""
    if force_refresh:
        try:
            get_weekly_students(db, force_refresh=True)
        except Exception as e:
            refresh_error = str(e)

    wk = week_key.strip() or _latest_week_key(db)
    if not wk:
        return {"week_key": "", "updated": 0, "created": 0, "refresh_error": refresh_error}

    rows = db.query(StudentWeeklyMetric).filter(StudentWeeklyMetric.week_key == wk).all()
    created = 0
    updated = 0
    for m in rows:
        uid = (m.student_uid or "").strip()
        name = (m.name or "").strip()
        nk = _norm_student_name(name)
        if not uid or not nk:
            continue
        ex = db.query(StudentRecord).filter(StudentRecord.student_uid == uid).one_or_none()
        if ex:
            if ex.display_name != name or ex.name_key != nk:
                ex.display_name = name
                ex.name_key = nk
                updated += 1
        else:
            db.add(StudentRecord(student_uid=uid, display_name=name, name_key=nk))
            created += 1
    db.commit()
    snap = _dump_weekly_snapshot_file(db, wk)
    return {
        "week_key": wk,
        "updated": updated,
        "created": created,
        "refresh_error": refresh_error,
        "snapshot_file": snap.get("filename", ""),
        "snapshot_rows": snap.get("rows", 0),
    }


class WxLoginIn(BaseModel):
    code: str = Field(min_length=1)


class BindByStudentNameIn(BaseModel):
    openid: str = Field(min_length=1)
    student_name: str = Field(min_length=1)
    student_uid: str = Field(min_length=1)
    parent_name: str = ""


class ChatIn(BaseModel):
    openid: str = Field(min_length=1)
    student_uid: str = Field(min_length=1)
    message: str = Field(min_length=1)


async def _handle_weekly_command(db: Session, operator_id: str, text: str) -> str | None:
    ops = _parse_op_ids()
    op = (operator_id or "").strip().lower()
    if not ops or op not in ops:
        return None

    cmd = (text or "").strip()
    if not cmd.startswith("周报"):
        return None

    if cmd in ("周报", "周报 帮助", "周报 help"):
        return (
            "周报指令：\n"
            "1) 周报 状态\n"
            "2) 周报 发送 [班级=xxx]\n"
            "3) 周报 开启 07:30 [班级=xxx]\n"
            "4) 周报 时间 07:30\n"
            "5) 周报 关闭"
        )

    if cmd.startswith("周报 状态"):
        sch = _load_weekly_schedule()
        return (
            f"周报定时：{'开启' if sch.get('enabled') else '关闭'}\n"
            f"时间：{sch.get('time_hhmm')}\n"
            f"班级过滤：{sch.get('group') or '无（全量）'}\n"
            f"仅未完成：{'是' if sch.get('only_unfinished') else '否'}\n"
            f"最近执行日期：{sch.get('last_run_date') or '无'}"
        )

    if cmd.startswith("周报 时间"):
        m = re.search(r"(\d{1,2}:\d{2})", cmd)
        if not m:
            return "格式错误，示例：周报 时间 07:30"
        hhmm = m.group(1)
        sch = _load_weekly_schedule()
        sch["time_hhmm"] = hhmm
        _save_weekly_schedule(sch)
        return f"已更新周报时间：{hhmm}"

    if cmd.startswith("周报 开启"):
        m = re.search(r"(\d{1,2}:\d{2})", cmd)
        g = re.search(r"班级\s*=\s*([^\s]+)", cmd)
        sch = _load_weekly_schedule()
        if m:
            sch["time_hhmm"] = m.group(1)
        sch["enabled"] = True
        sch["group"] = g.group(1) if g else ""
        _save_weekly_schedule(sch)
        return f"周报定时已开启：{sch.get('time_hhmm')}，班级={sch.get('group') or '全量'}"

    if cmd.startswith("周报 关闭"):
        sch = _load_weekly_schedule()
        sch["enabled"] = False
        _save_weekly_schedule(sch)
        return "周报定时已关闭"

    if cmd.startswith("周报 发送"):
        g = re.search(r"班级\s*=\s*([^\s]+)", cmd)
        group = g.group(1) if g else ""
        try:
            res = await send_weekly_reports(db, group=group, force_refresh=True)
            return f"已执行周报发送：week={res.week_key} group={res.group or '全量'} ok={res.ok} fail={res.fail} skip={res.skip}"
        except Exception as e:
            return f"周报发送失败：{e}"

    return "未识别指令。发送“周报 帮助”查看用法。"


async def _handle_operator_ai_command(db: Session, operator_id: str, text: str) -> str | None:
    ops = _parse_op_ids()
    op = (operator_id or "").strip().lower()
    if not ops or op not in ops:
        if _looks_like_operator_command(text):
            logger.info(
                "operator command ignored: operator_id=%s not in whitelist=%s",
                operator_id,
                sorted(list(ops)),
            )
        return None

    cmd = (text or "").strip()
    if not cmd:
        return None
    # 仅处理 #s 开头的显式运维口令，避免影响普通咨询对话。
    if not _looks_like_operator_command(cmd):
        return None
    clean = re.sub(r"^\s*[#＃]s[\s,，:：-]*", "", cmd, flags=re.IGNORECASE).strip() or cmd
    logger.info("operator command accepted: operator_id=%s raw=%s clean=%s", operator_id, cmd, clean)

    sender_userid = (settings.wecom_external_sender_id or "").strip() or operator_id
    try:
        intent = await _parse_operator_intent(clean)
    except Exception as e:
        return f"指令解析失败：{e}"

    action = str(intent.get("action") or "unknown")
    scope = str(intent.get("scope") or "all")
    su = str(intent.get("student_uid") or "")
    gp = str(intent.get("group") or "")
    msg = str(intent.get("message") or "")

    if action == "send_weekly":
        if scope == "student" and su:
            week_raw = get_weekly_students(db, force_refresh=True)
            target_raw = None
            for r in week_raw:
                if str(r.get("uid") or "").strip() == su:
                    target_raw = r
                    break
            if not target_raw:
                return f"未找到 student_uid={su} 的周数据。"
            target = db.query(ParentStudentBinding).filter(ParentStudentBinding.student_uid == su).all()
            ok = 0
            fail = 0
            for b in target:
                ext = (b.external_userid or "").strip()
                if not ext:
                    continue
                try:
                    content = render_weekly_report(target_raw, template_text=_load_weekly_template())
                    await add_msg_template_single(external_userid=ext, content=content, sender_userid=sender_userid)
                    ok += 1
                except Exception:
                    fail += 1
            return f"已发送学生周报：student_uid={su}，ok={ok}，fail={fail}"
        res = await send_weekly_reports(
            db,
            sender=sender_userid,
            group=gp if scope == "group" else "",
            force_refresh=True,
            template_text=_load_weekly_template(),
        )
        return f"已执行周报发送：week={res.week_key} group={res.group or '全量'} ok={res.ok} fail={res.fail} skip={res.skip}"

    if action == "send_today_student" and su:
        today = get_today_students_stats()
        row = next((r for r in today if str(r.get("uid") or "").strip() == su), None)
        if not row:
            return f"未找到 student_uid={su} 今日数据。"
        msg_text = (
            f"今日做题数据：{row.get('name','')}（{su}）\n"
            f"提交：{int(row.get('today_submits') or 0)}，AC：{int(row.get('today_ac') or 0)}"
        )
        ok, fail, skip = await _send_custom_message_to_scope(
            db,
            sender_userid=sender_userid,
            message=msg_text,
            scope="student",
            student_uid=su,
        )
        return f"已发送今日数据：student_uid={su} ok={ok} fail={fail} skip={skip}"

    if action == "send_custom":
        ok, fail, skip = await _send_custom_message_to_scope(
            db,
            sender_userid=sender_userid,
            message=msg,
            scope=scope,
            student_uid=su,
            group=gp,
        )
        return f"自定义消息已发送：scope={scope} group={gp or '-'} student_uid={su or '-'} ok={ok} fail={fail} skip={skip}"

    if action == "stats_weekly":
        return _summarize_weekly_group_stats(db, group=gp if scope == "group" else "")

    if action == "stats_today":
        today = get_today_students_stats()
        return _summarize_today_stats(today, group=gp if scope == "group" else "")

    return (
        "未识别该运维指令。示例：\n"
        "1) Hi bot 发送五维数据周报给每个学生\n"
        "2) 发送今天做了多少道题给学生ID39\n"
        "3) 发送今天要登录电子协会考级网站http://www.dzxh.com注册报名考级\n"
        "4) 统计本周学生做题情况给我\n"
        "5) 统计今天CSP-J4班的做题数据给我"
    )


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
    raw_uid = body.student_uid.strip()
    key = _norm_student_name(raw_name)
    if not openid or not key or not raw_uid:
        raise HTTPException(status_code=400, detail="openid, student_name and student_uid required")

    existing = (
        db.query(ParentStudentBinding).filter(ParentStudentBinding.openid == openid).order_by(ParentStudentBinding.id.asc()).first()
    )
    if existing:
        return {"status": "approved", "student_uid": existing.student_uid}

    target = db.query(StudentRecord).filter(StudentRecord.student_uid == raw_uid).one_or_none()
    if target is None:
        # 名录为空/过期时，优先尝试从最近周数据自动回填一次
        _sync_student_records_from_weekly(db, force_refresh=False)
        target = db.query(StudentRecord).filter(StudentRecord.student_uid == raw_uid).one_or_none()

    # 不成功绑定统一进入待审核
    if target is None or target.name_key != key:
        db.add(
            BindingNameRequest(
                openid=openid,
                student_name_submitted=f"{raw_name}|{raw_uid}",
                candidates_json=json.dumps(([raw_uid] if raw_uid else []), ensure_ascii=False),
                status="pending",
            )
        )
        db.commit()
        return {"status": "pending"}

    db.add(ParentStudentBinding(openid=openid, student_uid=target.student_uid))
    db.commit()
    return {"status": "approved", "student_uid": target.student_uid}


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
        txt = msg.content.strip()
        logger.info("wecom text received: from=%s content=%s", msg.from_user_name, txt[:200])
        cmd_reply = await _handle_weekly_command(db, msg.from_user_name, txt)
        if cmd_reply is not None:
            reply_text = cmd_reply
        else:
            ai_cmd_reply = await _handle_operator_ai_command(db, msg.from_user_name, txt)
            if ai_cmd_reply is not None:
                reply_text = ai_cmd_reply
            else:
                # Always acknowledge immediately, then send full AI answer asynchronously.
                asyncio.create_task(_send_deferred_wecom_reply(msg.from_user_name, txt))
                reply_text = "已收到，正在整理详细回复，将稍后推送给你。"
                logger.info("wecom immediate ack + deferred reply: from=%s", msg.from_user_name)

    # WeCom callback text payload has practical size limits; overlong replies may be dropped and retried.
    if len(reply_text) > WECOM_REPLY_MAX_CHARS:
        logger.info(
            "wecom reply truncated: from=%s orig_len=%s max_len=%s",
            msg.from_user_name,
            len(reply_text),
            WECOM_REPLY_MAX_CHARS,
        )
        reply_text = reply_text[:WECOM_REPLY_MAX_CHARS] + "\n\n（内容较长，已截断）"
    else:
        logger.info("wecom reply length: from=%s len=%s", msg.from_user_name, len(reply_text))

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
      <div><a href="/admin/reports">周报发送</a></div>
      <div><a href="/admin/external-contacts">外部联系人同步与匹配</a></div>
      <div><a href="/admin/students">学生名录（小程序姓名匹配）</a></div>
      <div><a href="/admin/bindings">已匹配成功（家长绑定）</a></div>
      <div><a href="/admin/weekly-files">每周学生更新数据文件</a></div>
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


@app.get("/admin/reports", response_class=HTMLResponse)
async def admin_reports(db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    sch = _load_weekly_schedule()
    weekly_template = _load_weekly_template()
    latest_week = _latest_week_key(db)
    class_count: dict[str, int] = {}
    if latest_week:
        wrows = db.query(StudentWeeklyMetric).filter(StudentWeeklyMetric.week_key == latest_week).all()
        for w in wrows:
            try:
                groups = json.loads(w.groups_json or "[]")
            except Exception:
                groups = []
            for g in groups:
                gn = str(g).strip()
                if gn:
                    class_count[gn] = class_count.get(gn, 0) + 1
    class_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td><td><button type='button' onclick=\"fillGroup('{html.escape(k, quote=True)}')\">使用</button></td></tr>"
        for k, v in sorted(class_count.items(), key=lambda x: (-x[1], x[0]))
    )
    class_tip = (
        f"<div>最新周：<code>{html.escape(latest_week)}</code>，共 <code>{len(class_count)}</code> 个班级</div>"
        if latest_week
        else "<div>暂无周数据，请先在“学生名录”执行周同步。</div>"
    )
    logs = db.query(ExternalSendLog).order_by(ExternalSendLog.id.desc()).limit(50).all()
    rows = "".join(
        f"<tr><td>{l.created_at}</td><td>{html.escape(l.week_key)}</td><td>{html.escape(l.group_filter or '全量')}</td>"
        f"<td><code>{html.escape(l.student_uid)}</code></td><td><code>{html.escape(l.external_userid)}</code></td>"
        f"<td>{html.escape(l.status)}</td><td>{html.escape((l.error or '')[:120])}</td></tr>"
        for l in logs
    )
    body = f"""
    <h2>周报发送</h2>
    <div class="card">
      <div><strong>定时状态：</strong>{'开启' if sch.get('enabled') else '关闭'}</div>
      <div><strong>发送时间：</strong>{html.escape(str(sch.get('time_hhmm') or '07:30'))}</div>
      <div><strong>班级过滤：</strong>{html.escape(str(sch.get('group') or '无（全量）'))}</div>
      <div><strong>最近执行日期：</strong>{html.escape(str(sch.get('last_run_date') or '无'))}</div>
    </div>
    <div class="card">
      <h3>班级分类（来自 Hydro 最新周分组）</h3>
      {class_tip}
      <form action="/admin/reports/sync-hydro" method="post" style="margin:8px 0 12px 0">
        <button type="submit">同步 Hydro 分组并刷新班级</button>
      </form>
      <table>
        <thead><tr><th>班级</th><th>人数</th><th>快速填入</th></tr></thead>
        <tbody>{class_rows or '<tr><td colspan="3">暂无班级数据</td></tr>'}</tbody>
      </table>
    </div>
    <div class="card">
      <h3>周报发送模板（可修改）</h3>
      <div style="margin-bottom:8px;color:#666">
        可用变量：<code>{'{name}'}</code> <code>{'{uid}'}</code> <code>{'{rank}'}</code>
        <code>{'{hw_title}'}</code> <code>{'{hw_done}'}</code> <code>{'{hw_total}'}</code>
        <code>{'{week_ac}'}</code> <code>{'{week_submits}'}</code> <code>{'{active_days}'}</code>
        <code>{'{last_active}'}</code> <code>{'{groups}'}</code>
      </div>
      <form action="/admin/reports/template" method="post">
        <textarea name="template_text" rows="12" style="width:100%;font-family:monospace">{html.escape(weekly_template)}</textarea>
        <div style="margin-top:10px"><button type="submit">保存周报模板</button></div>
      </form>
    </div>
    <div class="card">
      <form action="/admin/reports/send-now" method="post">
        <div style="margin-bottom:10px"><label>班级（可选，留空=全量）</label><input id="send-group" name="group" placeholder="例如 高一1班" /></div>
        <div style="margin-bottom:10px"><label><input type="checkbox" name="only_unfinished" value="1" /> 仅发送作业未完成学生</label></div>
        <button type="submit">立即发送周报</button>
      </form>
    </div>
    <div class="card">
      <form action="/admin/reports/schedule" method="post">
        <div style="margin-bottom:10px"><label><input type="checkbox" name="enabled" value="1" {'checked' if sch.get('enabled') else ''}/> 开启定时发送</label></div>
        <div style="margin-bottom:10px"><label>时间（HH:MM）</label><input name="time_hhmm" value="{html.escape(str(sch.get('time_hhmm') or '07:30'))}" /></div>
        <div style="margin-bottom:10px"><label>默认班级（可选）</label><input id="schedule-group" name="group" value="{html.escape(str(sch.get('group') or ''), quote=True)}" /></div>
        <div style="margin-bottom:10px"><label><input type="checkbox" name="only_unfinished" value="1" {'checked' if sch.get('only_unfinished') else ''}/> 默认仅未完成</label></div>
        <button type="submit">保存定时配置</button>
      </form>
    </div>
    <div class="card">
      <h3>最近发送日志（50条）</h3>
      <table>
        <thead><tr><th>时间</th><th>周</th><th>班级</th><th>student_uid</th><th>external_userid</th><th>状态</th><th>错误</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    <script>
      function fillGroup(v) {{
        var a = document.getElementById('send-group');
        var b = document.getElementById('schedule-group');
        if (a) a.value = v;
        if (b) b.value = v;
      }}
    </script>
    """
    return html_page("Reports", body)


@app.get("/admin/external-contacts", response_class=HTMLResponse)
async def admin_external_contacts(db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    latest_week = _latest_week_key(db)
    group_map: dict[str, str] = {}
    if latest_week:
        week_rows = db.query(StudentWeeklyMetric).filter(StudentWeeklyMetric.week_key == latest_week).all()
        for w in week_rows:
            try:
                gs = json.loads(w.groups_json or "[]")
            except Exception:
                gs = []
            group_map[w.student_uid] = "、".join([str(x) for x in gs]) if gs else ""

    contacts = db.query(ExternalContact).order_by(ExternalContact.updated_at.desc()).limit(500).all()
    uid_name = {s.student_uid: s.display_name for s in db.query(StudentRecord).all()}
    rows = "".join(
        (
            f"<tr><td><code>{html.escape(c.external_userid)}</code></td>"
            f"<td>{html.escape(c.name)}</td>"
            f"<td><code>{html.escape(c.follow_userid)}</code></td>"
            f"<td>{html.escape(c.remark)}</td>"
            f"<td><code>{html.escape(c.student_uid_hint or '')}</code></td>"
            f"<td>{html.escape(uid_name.get(c.student_uid_hint, ''))}</td>"
            f"<td>{html.escape(group_map.get(c.student_uid_hint, ''))}</td>"
            f"<td>"
            f"<form action='/admin/external-contacts/link' method='post' style='display:flex;gap:6px;align-items:center'>"
            f"<input type='hidden' name='external_userid' value='{html.escape(c.external_userid, quote=True)}' />"
            f"<input name='student_uid' value='{html.escape(c.student_uid_hint or '', quote=True)}' style='width:120px' placeholder='HYDRO ID' />"
            f"<button type='submit'>保存匹配</button></form></td></tr>"
        )
        for c in contacts
    )
    body = f"""
    <h2>外部联系人同步与匹配</h2>
    <p>自动拉取 external_userid，优先从备注中识别 HYDRO ID；可手工修正 HYDRO ID 并保存。</p>
    <div class="card">
      <form action="/admin/external-contacts/sync" method="post">
        <div style="margin-bottom:10px">
          <label>跟进成员 userid（默认取 WECOM_EXTERNAL_SENDER_ID）</label>
          <input name="follower_userid" placeholder="例如 yangshengpin" />
        </div>
        <button type="submit">同步外部联系人并自动匹配</button>
      </form>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>external_userid</th><th>昵称</th><th>跟进人</th><th>备注</th><th>HYDRO ID</th><th>学生姓名</th><th>班级</th><th>操作</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("External contacts", body)


@app.post("/admin/external-contacts/sync")
async def admin_external_contacts_sync(
    follower_userid: str = Form(""),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    uid = (follower_userid or "").strip() or (settings.wecom_external_sender_id or "").strip()
    if not uid:
        return Response(status_code=303, headers={"Location": "/admin/external-contacts"})

    try:
        ex_ids = await list_external_userids(uid)
    except Exception as e:
        logger.warning("sync external contacts failed list: %s", e)
        return Response(status_code=303, headers={"Location": "/admin/external-contacts"})

    for exid in ex_ids:
        try:
            detail = await get_external_contact(exid)
        except Exception as e:
            logger.warning("sync external contact failed get %s: %s", exid, e)
            continue
        contact = detail.get("external_contact") or {}
        follow = (detail.get("follow_user") or [{}])[0] or {}
        name = str(contact.get("name") or "")
        remark = str(follow.get("remark") or "")
        hint = _guess_uid_from_text(remark, name)

        row = db.query(ExternalContact).filter(ExternalContact.external_userid == exid).one_or_none()
        if row is None:
            row = ExternalContact(external_userid=exid)
            db.add(row)
        row.name = name
        row.follow_userid = uid
        row.remark = remark
        row.student_uid_hint = hint
        row.updated_at = datetime.utcnow()

        # 自动绑定 external_userid -> student_uid（若 hint 命中学生名录）
        if hint and db.query(StudentRecord).filter(StudentRecord.student_uid == hint).one_or_none():
            bind = db.query(ParentStudentBinding).filter(ParentStudentBinding.external_userid == exid).one_or_none()
            if bind is None:
                bind = db.query(ParentStudentBinding).filter(ParentStudentBinding.student_uid == hint).order_by(ParentStudentBinding.id.asc()).first()
            if bind is None:
                bind = ParentStudentBinding(openid="", student_uid=hint, external_userid=exid)
                db.add(bind)
            else:
                bind.student_uid = hint
                bind.external_userid = exid
    db.commit()
    return Response(status_code=303, headers={"Location": "/admin/external-contacts"})


@app.post("/admin/external-contacts/link")
async def admin_external_contacts_link(
    external_userid: str = Form(...),
    student_uid: str = Form(...),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    exid = external_userid.strip()
    su = student_uid.strip()
    if not exid:
        return Response(status_code=303, headers={"Location": "/admin/external-contacts"})

    row = db.query(ExternalContact).filter(ExternalContact.external_userid == exid).one_or_none()
    if row:
        row.student_uid_hint = su
        row.updated_at = datetime.utcnow()

    if su and db.query(StudentRecord).filter(StudentRecord.student_uid == su).one_or_none():
        bind = db.query(ParentStudentBinding).filter(ParentStudentBinding.external_userid == exid).one_or_none()
        if bind is None:
            bind = db.query(ParentStudentBinding).filter(ParentStudentBinding.student_uid == su).order_by(ParentStudentBinding.id.asc()).first()
        if bind is None:
            bind = ParentStudentBinding(openid="", student_uid=su, external_userid=exid)
            db.add(bind)
        else:
            bind.student_uid = su
            bind.external_userid = exid
    db.commit()
    return Response(status_code=303, headers={"Location": "/admin/external-contacts"})


@app.post("/admin/reports/send-now")
async def admin_reports_send_now(
    group: str = Form(""),
    only_unfinished: str = Form(""),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    res = await send_weekly_reports(
        db,
        group=group.strip(),
        only_unfinished=(only_unfinished or "").strip() == "1",
        force_refresh=True,
        template_text=_load_weekly_template(),
    )
    logger.info("admin send now: week=%s group=%s ok=%s fail=%s skip=%s", res.week_key, res.group, res.ok, res.fail, res.skip)
    return Response(status_code=303, headers={"Location": "/admin/reports"})


@app.post("/admin/reports/schedule")
async def admin_reports_schedule(
    enabled: str = Form(""),
    time_hhmm: str = Form("07:30"),
    group: str = Form(""),
    only_unfinished: str = Form(""),
    _user: str = Depends(require_admin),
):
    hhmm = (time_hhmm or "07:30").strip()
    if not re.fullmatch(r"\d{1,2}:\d{2}", hhmm):
        hhmm = "07:30"
    sch = _load_weekly_schedule()
    sch["enabled"] = (enabled or "").strip() == "1"
    sch["time_hhmm"] = hhmm
    sch["group"] = group.strip()
    sch["only_unfinished"] = (only_unfinished or "").strip() == "1"
    _save_weekly_schedule(sch)
    return Response(status_code=303, headers={"Location": "/admin/reports"})


@app.post("/admin/reports/sync-hydro")
async def admin_reports_sync_hydro(
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    try:
        _sync_student_records_from_weekly(db, force_refresh=True)
    except Exception as e:
        logger.warning("reports sync hydro failed: %s", e)
    return Response(status_code=303, headers={"Location": "/admin/reports"})


@app.post("/admin/reports/template")
async def admin_reports_template(
    template_text: str = Form(""),
    _user: str = Depends(require_admin),
):
    _save_weekly_template(template_text)
    return Response(status_code=303, headers={"Location": "/admin/reports"})


@app.get("/admin/weekly-files", response_class=HTMLResponse)
async def admin_weekly_files(db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    latest_week = _latest_week_key(db)
    week_rows = (
        db.query(StudentWeeklyMetric.week_key)
        .distinct()
        .order_by(StudentWeeklyMetric.week_key.desc())
        .limit(30)
        .all()
    )
    week_opts = "".join(f'<option value="{html.escape(w[0])}">{html.escape(w[0])}</option>' for w in week_rows if w and w[0])

    files = sorted(_weekly_updates_dir().glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    rows = "".join(
        f"<tr><td><code>{html.escape(f.name)}</code></td><td>{f.stat().st_size}</td>"
        f"<td><a href='/admin/weekly-files/download?name={html.escape(f.name, quote=True)}'>下载</a></td></tr>"
        for f in files[:100]
    )
    body = f"""
    <h2>每周学生更新数据文件</h2>
    <p>目录：<code>{html.escape(str(_weekly_updates_dir()))}</code>。每周一个 JSON 文件（例如 <code>2026-W12.json</code>）。</p>
    <div class="card">
      <form action="/admin/weekly-files/generate" method="post">
        <div style="margin-bottom:10px">
          <label>week_key（可选，留空=最新）</label>
          <input name="week_key" placeholder="例如 2026-W12（默认最新 {html.escape(latest_week or '无')}）" />
        </div>
        <div style="margin-bottom:10px">
          <label>或从已有周选择</label>
          <select name="week_key_select">
            <option value="">（不选择）</option>
            {week_opts}
          </select>
        </div>
        <div style="margin-bottom:10px">
          <label><input type="checkbox" name="force_refresh" value="1" /> 先从 Hydro 刷新周数据后再写文件</label>
        </div>
        <button type="submit">生成/更新该周文件</button>
      </form>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>文件名</th><th>大小(bytes)</th><th>操作</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Weekly files", body)


@app.get("/admin/bindings", response_class=HTMLResponse)
async def admin_bindings(db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    binds = db.query(ParentStudentBinding).order_by(ParentStudentBinding.id.desc()).limit(500).all()
    uids = sorted({b.student_uid for b in binds if b.student_uid})
    stu_map: dict[str, str] = {}
    if uids:
        stus = db.query(StudentRecord).filter(StudentRecord.student_uid.in_(uids)).all()
        stu_map = {s.student_uid: s.display_name for s in stus}
    rows = "".join(
        (
            f"<tr><td>{b.id}</td>"
            f"<td><code>{html.escape(b.openid)}</code></td>"
            f"<td><code>{html.escape(b.student_uid)}</code></td>"
            f"<td><code>{html.escape(b.external_userid or '')}</code></td>"
            f"<td>{html.escape(stu_map.get(b.student_uid, ''))}</td>"
            f"<td>{b.created_at}</td>"
            f"<td>"
            f"<form action='/admin/bindings/update' method='post' style='display:flex;gap:6px;align-items:center'>"
            f"<input type='hidden' name='binding_id' value='{b.id}' />"
            f"<input name='student_uid' value='{html.escape(b.student_uid, quote=True)}' style='width:120px' />"
            f"<input name='external_userid' value='{html.escape((b.external_userid or ''), quote=True)}' style='width:150px' placeholder='外部联系人ID' />"
            f"<button type='submit'>保存</button>"
            f"</form>"
            f"<form action='/admin/bindings/delete' method='post' style='margin-top:6px'>"
            f"<input type='hidden' name='binding_id' value='{b.id}' />"
            f"<button type='submit' class='secondary'>删除绑定</button>"
            f"</form>"
            f"</td></tr>"
        )
        for b in binds
    )
    body = f"""
    <h2>已匹配成功（家长绑定）</h2>
    <p>显示最近 500 条 openid 与 student_uid 的绑定结果。</p>
    <div class="card">
      <table>
        <thead><tr><th>ID</th><th>openid</th><th>student_uid</th><th>external_userid</th><th>学生姓名</th><th>绑定时间</th><th>操作</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Bindings", body)


@app.post("/admin/bindings/update")
async def admin_bindings_update(
    binding_id: int = Form(...),
    student_uid: str = Form(...),
    external_userid: str = Form(""),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    b = db.query(ParentStudentBinding).filter(ParentStudentBinding.id == binding_id).one_or_none()
    if not b:
        return Response(status_code=303, headers={"Location": "/admin/bindings"})
    su = student_uid.strip()
    if not su:
        return Response(status_code=303, headers={"Location": "/admin/bindings"})
    b.student_uid = su
    b.external_userid = external_userid.strip()
    db.commit()
    return Response(status_code=303, headers={"Location": "/admin/bindings"})


@app.post("/admin/bindings/delete")
async def admin_bindings_delete(
    binding_id: int = Form(...),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    b = db.query(ParentStudentBinding).filter(ParentStudentBinding.id == binding_id).one_or_none()
    if b:
        db.delete(b)
        db.commit()
    return Response(status_code=303, headers={"Location": "/admin/bindings"})


@app.post("/admin/weekly-files/generate")
async def admin_weekly_files_generate(
    week_key: str = Form(""),
    week_key_select: str = Form(""),
    force_refresh: str = Form(""),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    wk = (week_key_select or week_key or "").strip()
    if (force_refresh or "").strip() == "1":
        try:
            get_weekly_students(db, force_refresh=True)
        except Exception as e:
            logger.warning("weekly file force refresh failed: %s", e)
    if not wk:
        wk = _latest_week_key(db)
    if wk:
        _dump_weekly_snapshot_file(db, wk)
    return Response(status_code=303, headers={"Location": "/admin/weekly-files"})


@app.get("/admin/weekly-files/download")
async def admin_weekly_files_download(name: str, _user: str = Depends(require_admin)):
    safe_name = Path(name).name
    if not safe_name.endswith(".json"):
        raise HTTPException(status_code=400, detail="invalid file")
    fp = _weekly_updates_dir() / safe_name
    if not fp.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path=str(fp), filename=safe_name, media_type="application/json")


@app.get("/admin/students", response_class=HTMLResponse)
async def admin_students(db: Session = Depends(get_db), _user: str = Depends(require_admin)):
    students = db.query(StudentRecord).order_by(StudentRecord.id.desc()).limit(500).all()
    latest_week = _latest_week_key(db)
    weekly_count = db.query(StudentWeeklyMetric).filter(StudentWeeklyMetric.week_key == latest_week).count() if latest_week else 0
    rows = "".join(
        f"<tr><td>{s.id}</td><td><code>{html.escape(s.student_uid)}</code></td>"
        f"<td>{html.escape(s.display_name)}</td><td><code>{html.escape(s.name_key)}</code></td>"
        f"<td>{s.created_at}</td></tr>"
        for s in students
    )
    body = f"""
    <h2>学生名录（只读）</h2>
    <p>小程序家长需发送<strong>孩子姓名 + HYDRO ID</strong>，系统会按这里的数据做绑定校验。名录只通过每周同步更新，不做手工增删改。</p>
    <div class="card">
      <div style="margin-bottom:10px">
        最新周数据：<code>{html.escape(latest_week or "无")}</code>，
        该周学生记录：<code>{weekly_count}</code>
      </div>
      <form action="/admin/students/sync-weekly" method="post">
        <div style="margin-bottom:10px">
          <label>week_key（可选，留空=最新）</label>
          <input name="week_key" placeholder="例如 2026-W12" />
        </div>
        <div style="margin-bottom:10px">
          <label><input type="checkbox" name="force_refresh" value="1" /> 先从 Hydro 拉取最新周数据（需配置 HYDRO_SSH_*）</label>
        </div>
        <button type="submit">从周数据同步到学生名录</button>
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


@app.post("/admin/students/sync-weekly")
async def admin_students_sync_weekly(
    week_key: str = Form(""),
    force_refresh: str = Form(""),
    db: Session = Depends(get_db),
    _user: str = Depends(require_admin),
):
    result = _sync_student_records_from_weekly(
        db,
        week_key=week_key,
        force_refresh=(force_refresh or "").strip() == "1",
    )
    wk = result.get("week_key") or "无"
    created = int(result.get("created") or 0)
    updated = int(result.get("updated") or 0)
    err = str(result.get("refresh_error") or "").strip()
    if err:
        logger.warning("sync weekly with refresh error: %s", err)
    tip = f"week={wk}, created={created}, updated={updated}"
    if err:
        tip = f"{tip}, refresh_error={err[:120]}"
    return Response(status_code=303, headers={"Location": f"/admin/students?sync={tip}"})


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

