from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import time
import hashlib
from urllib.parse import quote
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
from .hydro_service import (
    compute_today_stats_by_group,
    get_student_hydro_stats,
    get_today_students_stats,
    get_weekly_students,
)
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
daily_weekly_updates_task: asyncio.Task | None = None
WECOM_REPLY_MAX_CHARS = 1800

# 企业微信被动回复超时后会用相同 MsgId 重试回调，导致 #s / 周报发送 等副作用执行多次。
_wecom_side_effect_lock = asyncio.Lock()
_wecom_side_effect_msg_ids: dict[str, float] = {}
WECOM_SIDE_EFFECT_DEDUP_TTL_SEC = 600

# H5 学生统计查询频控：同一来源每小时最多查询 1 次。
H5_STUDENT_QUERY_LIMIT_SEC = 3600
_h5_student_query_lock = asyncio.Lock()
_h5_student_query_last_ts: dict[str, float] = {}
_h5_student_last_name_by_src: dict[str, str] = {}

EXAMS_DATA_DIR = Path(settings.data_dir) / "exams"
EXAMS_META_PATH = EXAMS_DATA_DIR / "exams.json"


def _wecom_text_has_side_effects(txt: str) -> bool:
    """会改状态或对外群发的文本，需要去重。"""
    t = (txt or "").strip()
    if _looks_like_operator_command(t):
        # @查询 和 @帮助 只读，不需要去重
        if t.startswith("@查询") or t in ("@帮助", "@help"):
            return False
        return True
    if not t.startswith("周报"):
        return False
    if t in ("周报", "周报 帮助", "周报 help") or t.startswith("周报 状态"):
        return False
    return True


async def _wecom_try_begin_side_effect(msg_id: str) -> bool:
    """
    同一 MsgId 在 TTL 内只允许执行一次副作用。
    返回 True 表示本次应执行；False 表示重复回调应跳过。
    """
    mid = (msg_id or "").strip()
    if not mid:
        return True
    async with _wecom_side_effect_lock:
        now = time.time()
        expired = [k for k, exp in _wecom_side_effect_msg_ids.items() if exp < now]
        for k in expired:
            del _wecom_side_effect_msg_ids[k]
        if mid in _wecom_side_effect_msg_ids:
            logger.info("wecom side-effect duplicate callback skipped msg_id=%s", mid)
            return False
        _wecom_side_effect_msg_ids[mid] = now + WECOM_SIDE_EFFECT_DEDUP_TTL_SEC
        if len(_wecom_side_effect_msg_ids) > 8000:
            for k in list(_wecom_side_effect_msg_ids.keys())[:2000]:
                _wecom_side_effect_msg_ids.pop(k, None)
        return True


def _h5_student_query_source_key(request: Request) -> str:
    # 优先取反向代理透传 IP；否则回退到直连客户端地址。
    xff = (request.headers.get("x-forwarded-for") or "").strip()
    if xff:
        return xff.split(",")[0].strip() or "unknown"
    client = request.client.host if request.client else ""
    return (client or "unknown").strip()


async def _h5_try_consume_query_quota(request: Request) -> tuple[bool, int]:
    src = _h5_student_query_source_key(request)
    now = time.time()
    async with _h5_student_query_lock:
        prev = _h5_student_query_last_ts.get(src)
        if prev is not None:
            remain = H5_STUDENT_QUERY_LIMIT_SEC - (now - prev)
            if remain > 0:
                return False, int(remain + 0.999)
        _h5_student_query_last_ts[src] = now
        if len(_h5_student_query_last_ts) > 20000:
            cutoff = now - H5_STUDENT_QUERY_LIMIT_SEC
            stale = [k for k, t in _h5_student_query_last_ts.items() if t < cutoff]
            for k in stale[:5000]:
                _h5_student_query_last_ts.pop(k, None)
        return True, 0


async def _h5_set_last_name_for_source(request: Request, name: str) -> None:
    nm = (name or "").strip()
    if not nm:
        return
    src = _h5_student_query_source_key(request)
    async with _h5_student_query_lock:
        _h5_student_last_name_by_src[src] = nm
        if len(_h5_student_last_name_by_src) > 20000:
            for k in list(_h5_student_last_name_by_src.keys())[:5000]:
                _h5_student_last_name_by_src.pop(k, None)


async def _h5_get_last_name_for_source(request: Request) -> str:
    src = _h5_student_query_source_key(request)
    async with _h5_student_query_lock:
        return (_h5_student_last_name_by_src.get(src) or "").strip()


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
    # 阶段1：输入清洗
    question = _clean_query(question)
    if not question:
        return "您好，请问有什么可以帮您？"

    user = _get_or_create_user(db, wecom_user_id)

    # 阶段2：意图识别 — 闲聊直接回复，不走 RAG
    if _is_chitchat(question):
        chitchat_replies = {
            "谢谢": "不客气，有其他问题随时告诉我！",
            "感谢": "不客气，有其他问题随时告诉我！",
            "再见": "再见，有需要随时联系！",
            "拜拜": "拜拜，有需要随时联系！",
            "bye": "Bye！有需要随时联系！",
        }
        for kw, reply in chitchat_replies.items():
            if kw in question.lower():
                return reply
        return "您好！请问有什么可以帮您？"

    # memory：取最近 N 轮历史
    turns = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(settings.memory_max_turns * 2)
        .all()
    )
    turns = list(reversed(turns))

    # 阶段3：知识检索（带相关性阈值）
    hits = (rindex.search(question, settings.rag_top_k, min_score=settings.rag_min_score) if rindex else [])
    logger.info(
        "rag search: user=%s q=%s hits=%s",
        wecom_user_id,
        question[:120],
        len(hits),
    )

    # 阶段4：AI 生成（严格 grounding prompt）
    system = (
        "你是编程教育机构的客服助手。\n"
        "【规则】\n"
        "1. 只能根据下方【知识库】内容回答，不得编造或推测知识库以外的信息。\n"
        "2. 如果知识库中没有相关内容，必须回复：'抱歉，我暂时没有这方面的信息，建议您直接联系老师确认。'\n"
        "3. 回答简洁、中文、结构清晰，不超过300字。\n"
        "4. 引用知识库内容时在句末标注 [来源N]。\n"
        "5. 不要重复用户的问题，直接给出答案。"
    )
    if extra_system:
        system = f"{system}\n\n{extra_system}"

    # 无 KB 命中时的处理
    if not hits:
        # 有历史上下文时走多轮对话（追问场景），无历史时直接兜底
        if not turns:
            reply = "抱歉，我暂时没有这方面的信息，建议您直接联系老师确认。"
            db.add(ChatMessage(user_id=user.id, role="user", content=question))
            db.add(ChatMessage(user_id=user.id, role="assistant", content=reply))
            db.commit()
            return reply
        # 有上下文时加约束后走 LLM
        system += "\n\n注意：当前知识库无相关内容，请基于对话上下文回答，如无法确定请如实说明，不要编造。"

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    if hits:
        ctx_lines = []
        for i, (_chunk_id, text, score) in enumerate(hits, start=1):
            ctx_lines.append(f"[来源{i}] {text.strip()}")
        ctx = "\n\n".join(ctx_lines)
        messages.append({"role": "system", "content": f"【知识库】\n{ctx}"})
        # 有 KB hits 时保留最近 2 轮历史作为上下文锚点（修复追问断裂问题）
        recent_turns = turns[-4:]
    else:
        # 无 KB hits 时保留全部历史记忆
        recent_turns = turns

    for t in recent_turns:
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
    global daily_weekly_updates_task
    daily_weekly_updates_task = asyncio.create_task(_daily_weekly_updates_loop())
    yield
    if weekly_scheduler_task:
        weekly_scheduler_task.cancel()
        try:
            await weekly_scheduler_task
        except asyncio.CancelledError:
            pass
    if daily_weekly_updates_task:
        daily_weekly_updates_task.cancel()
        try:
            await daily_weekly_updates_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="WeCom KB Bot", version="0.2.0", lifespan=lifespan)


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


# ── 阶段1：输入清洗 ──────────────────────────────────────────────────────────

def _clean_query(text: str) -> str:
    """去除控制字符，截断超长输入。"""
    t = (text or "").strip()
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", t)
    return t[:500]


# ── 阶段2：意图识别 ──────────────────────────────────────────────────────────

_CHITCHAT_PATTERNS = [
    r"^(你好|hi|hello|在吗|在不|哈喽|嗨|hey)[\s？?！!。~～]*$",
    r"^(谢谢|感谢|好的|收到|明白|了解|ok|okay|好)[\s！!。~～]*$",
    r"^(再见|拜拜|bye|goodbye|晚安|88)[\s！!。~～]*$",
    r"^(哈哈|哈哈哈|嗯嗯|嗯|哦|哦哦|噢|好的好的)[\s！!。~～]*$",
]


def _is_chitchat(text: str) -> bool:
    """判断是否为闲聊/问候，闲聊不走 RAG 检索。"""
    s = (text or "").strip()
    for p in _CHITCHAT_PATTERNS:
        if re.match(p, s, re.IGNORECASE):
            return True
    return False


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


def _daily_weekly_updates_schedule_file() -> Path:
    p = Path(settings.data_dir) / "daily_weekly_updates_schedule.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_daily_weekly_updates_schedule() -> dict[str, object]:
    fp = _daily_weekly_updates_schedule_file()
    if not fp.exists():
        return {
            "enabled": True,
            "time_hhmm": "02:00",
            "last_run_date": "",
        }
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        return {
            "enabled": bool(data.get("enabled", True)),
            "time_hhmm": str(data.get("time_hhmm") or "02:00"),
            "last_run_date": str(data.get("last_run_date") or ""),
        }
    except Exception:
        return {
            "enabled": True,
            "time_hhmm": "02:00",
            "last_run_date": "",
        }


def _save_daily_weekly_updates_schedule(data: dict[str, object]) -> None:
    fp = _daily_weekly_updates_schedule_file()
    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def _daily_weekly_updates_loop() -> None:
    """每天 02:00 从 Hydro 刷新周数据并生成“带日期后缀”的学生更新文件。"""
    while True:
        try:
            sch = _load_daily_weekly_updates_schedule()
            if not sch.get("enabled"):
                await asyncio.sleep(30)
                continue
            hhmm = str(sch.get("time_hhmm") or "02:00")
            now = datetime.now()
            now_hhmm = now.strftime("%H:%M")
            today = now.strftime("%Y-%m-%d")
            if now_hhmm == hhmm and str(sch.get("last_run_date") or "") != today:
                with next(get_db()) as db:  # type: ignore[arg-type]
                    # 刷新 Hydro 周指标并落盘快照
                    try:
                        get_weekly_students(db, force_refresh=True)
                    except Exception:
                        logger.exception("daily weekly updates hydro refresh failed")
                    wk = _latest_week_key(db)
                    if wk:
                        _dump_weekly_snapshot_file_with_suffix(db, wk, suffix_date=today)
                        logger.info("daily weekly updates generated file week=%s date=%s", wk, today)
                sch["last_run_date"] = today
                _save_daily_weekly_updates_schedule(sch)
            await asyncio.sleep(20)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("daily weekly updates loop error")
            await asyncio.sleep(30)


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


def _format_today_group_table(today_rows: list[dict], group: str) -> str:
    filt: list[dict] = []
    for r in today_rows:
        gs = [str(x) for x in (r.get("groups") or [])]
        if group and group not in gs:
            continue
        filt.append(r)

    if not filt:
        return f"今天未找到班级={group}的学生数据。"

    # Sort to match your screenshot style: by rank asc, then today_ac desc.
    def _key(x: dict) -> tuple[int, int, int]:
        try:
            rank = int(x.get("rank") or 999)
        except Exception:
            rank = 999
        try:
            ac = int(x.get("today_ac") or 0)
        except Exception:
            ac = 0
        try:
            sub = int(x.get("today_submits") or 0)
        except Exception:
            sub = 0
        return (rank, -ac, -sub)

    filt.sort(key=_key)
    # Avoid huge payloads.
    max_rows = 40
    shown = filt[:max_rows]
    total = len(filt)

    header = "学生UID | 姓名 | 排名 | 今日提交数 | 今日AC数 | 活跃天数 | 最后活跃时间"
    lines = [header]
    for r in shown:
        uid = str(r.get("uid") or "")
        name = str(r.get("name") or "")
        rank = int(r.get("rank") or 999)
        today_submits = int(r.get("today_submits") or 0)
        today_ac = int(r.get("today_ac") or 0)
        active_days = int(r.get("active_days") or 0)
        last_active = str(r.get("last_active_date") or "")
        lines.append(f"{uid} | {name} | {rank} | {today_submits} | {today_ac} | {active_days} | {last_active}")

    if total > max_rows:
        lines.append(f"... 共{total}人，已截断展示前{max_rows}人")
    return "\n".join(lines)


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
    # A) 今天班级ID：xxxx班级的做题数据给我（直接输出表格，不走 AI）
    m = re.search(
        r"今天.*班级ID\s*[:：]\s*([A-Za-z0-9_-]+)\s*班级.*?(做题|题目|提交|AC|数据)",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        return {
            "action": "stats_today_group_table",
            "scope": "group",
            "student_uid": "",
            "group": m.group(1),
            "message": "",
        }
    # Rule-first parsing for high-frequency command patterns.
    # 1) send weekly to one student
    m = re.search(r"(发送|推送).*(周报|五维).*(学生\s*ID|ID)\s*[:：]?\s*([A-Za-z0-9_-]+)", raw, flags=re.IGNORECASE)
    if m:
        return {
            "action": "send_weekly",
            "scope": "student",
            "student_uid": m.group(4),
            "group": "",
            "message": "",
        }
    # 2) stats today for one student (to operator)
    m = re.search(
        r"(统计|查看|查询|看下).*(今天|今日).*(学生\s*ID|ID)\s*[:：]?\s*([A-Za-z0-9_-]+).*?(做题|题目|提交|AC)?",
        raw,
        flags=re.IGNORECASE,
    )
    if m:
        return {
            "action": "stats_today_student",
            "scope": "student",
            "student_uid": m.group(4),
            "group": "",
            "message": "",
        }
    # 3) send today's done count to one student's parent
    m = re.search(r"(发送|推送).*(今天|今日).*(学生\s*ID|ID)\s*[:：]?\s*([A-Za-z0-9_-]+).*(做题|题目|提交|AC)", raw, flags=re.IGNORECASE)
    if m:
        return {
            "action": "send_today_student",
            "scope": "student",
            "student_uid": m.group(4),
            "group": "",
            "message": "",
        }
    # 4) class stats today/weekly
    m = re.search(r"(统计|查看|查询).*(今天|今日|本周).*(.+?班).*(做题|题目|提交|AC)", raw, flags=re.IGNORECASE)
    if m:
        return {
            "action": "stats_today" if m.group(2) in ("今天", "今日") else "stats_weekly",
            "scope": "group",
            "student_uid": "",
            "group": m.group(3).strip(),
            "message": "",
        }
    # 5) full stats today/weekly
    if re.search(r"(统计|查看|查询).*(今天|今日).*(做题|题目|提交|AC)", raw, flags=re.IGNORECASE):
        return {"action": "stats_today", "scope": "all", "student_uid": "", "group": "", "message": ""}
    if re.search(r"(统计|查看|查询).*(本周).*(做题|题目|提交|AC)", raw, flags=re.IGNORECASE):
        return {"action": "stats_weekly", "scope": "all", "student_uid": "", "group": "", "message": ""}

    messages = [
        {
            "role": "system",
            "content": (
                "你是企业微信运营指令解析器。请把用户命令解析成 JSON，不要输出其它文字。\n"
                "字段：action, scope, student_uid, group, message。\n"
                "action 只允许：send_weekly, send_custom, send_today_student, stats_weekly, stats_today, stats_today_student, stats_today_group_table。\n"
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


# ── 外部联系人消息处理 ────────────────────────────────────────────────────────

def _is_external_userid(user_id: str) -> bool:
    """企业微信外部联系人 ID 以 wo 或 wm 开头。"""
    s = (user_id or "").strip()
    return s.startswith("wo") or s.startswith("wm")


def _get_bound_students(db: Session, external_userid: str) -> list[str]:
    """返回该外部联系人绑定的所有 student_uid 列表。"""
    rows = (
        db.query(ParentStudentBinding)
        .filter(ParentStudentBinding.external_userid == external_userid)
        .all()
    )
    return [r.student_uid for r in rows if (r.student_uid or "").strip()]


_EXTERNAL_QUERY_TODAY_PATTERNS = [
    r"今[天日]",
    r"今日",
    r"查今",
]
_EXTERNAL_QUERY_WEEK_PATTERNS = [
    r"本周",
    r"这周",
    r"一周",
    r"周报",
    r"查周",
]
_EXTERNAL_QUERY_HW_PATTERNS = [
    r"作业",
    r"homework",
    r"hw",
]


def _match_external_intent(text: str) -> str:
    """识别外部用户查询意图：today / week / hw / unknown。"""
    t = (text or "").strip()
    for p in _EXTERNAL_QUERY_TODAY_PATTERNS:
        if re.search(p, t, re.IGNORECASE):
            return "today"
    for p in _EXTERNAL_QUERY_WEEK_PATTERNS:
        if re.search(p, t, re.IGNORECASE):
            return "week"
    for p in _EXTERNAL_QUERY_HW_PATTERNS:
        if re.search(p, t, re.IGNORECASE):
            return "hw"
    return "unknown"


def _format_external_today_reply(student_uid: str, db: Session) -> str:
    """生成单个学生今日做题回复文本。"""
    try:
        today_rows = get_today_students_stats()
    except Exception as e:
        return f"获取今日数据失败，请稍后再试。（{e}）"

    row = next((r for r in today_rows if str(r.get("uid") or "").strip() == student_uid), None)
    if not row:
        return "暂未查到今日数据，可能数据还未同步，请稍后再试。"

    name = str(row.get("name") or "")
    submits = int(row.get("today_submits") or 0)
    ac = int(row.get("today_ac") or 0)
    active_days = int(row.get("active_days") or 0)
    last_active = str(row.get("last_active_date") or "")

    return (
        f"📊 {name} 今日做题情况\n\n"
        f"✅ 今日 AC：{ac} 题\n"
        f"📝 今日提交：{submits} 次\n"
        f"🔥 累计活跃天数：{active_days} 天\n"
        f"📅 最近活跃：{last_active or '暂无记录'}"
    )


def _format_external_week_reply(student_uid: str, db: Session) -> str:
    """生成单个学生本周做题回复文本。"""
    wk = _latest_week_key(db)
    if not wk:
        return "暂无本周数据，请稍后再试。"

    row = db.query(StudentWeeklyMetric).filter(
        StudentWeeklyMetric.week_key == wk,
        StudentWeeklyMetric.student_uid == student_uid,
    ).one_or_none()

    if not row:
        return f"暂未查到 {wk} 周数据，可能数据还未同步，请稍后再试。"

    return (
        f"📊 {row.name} 本周做题情况（{wk}）\n\n"
        f"📈 当前排名：第 {row.rank} 名\n"
        f"💡 本周 AC：{row.week_ac} 题\n"
        f"📝 本周提交：{row.week_submits} 次\n"
        f"🔥 活跃天数：{row.active_days} 天\n"
        f"📅 最近活跃：{row.last_active or '暂无记录'}\n\n"
        f"📚 当前作业：{row.hw_title or '暂无'}\n"
        f"✅ 完成情况：{row.hw_done}/{row.hw_total}"
    )


def _format_external_hw_reply(student_uid: str, db: Session) -> str:
    """生成单个学生作业完成情况回复文本。"""
    wk = _latest_week_key(db)
    if not wk:
        return "暂无作业数据，请稍后再试。"

    row = db.query(StudentWeeklyMetric).filter(
        StudentWeeklyMetric.week_key == wk,
        StudentWeeklyMetric.student_uid == student_uid,
    ).one_or_none()

    if not row:
        return "暂未查到作业数据，可能数据还未同步，请稍后再试。"

    if not row.hw_title:
        return f"{row.name} 当前暂无进行中的作业。"

    done_emoji = "✅" if row.hw_done >= row.hw_total and row.hw_total > 0 else "⏳"
    return (
        f"📚 {row.name} 作业完成情况\n\n"
        f"作业：{row.hw_title}\n"
        f"{done_emoji} 完成：{row.hw_done}/{row.hw_total} 题\n"
        f"（数据周期：{wk}）"
    )


_EXTERNAL_HELP_TEXT = (
    "您好！我是学习助手，可以帮您查询孩子的学习情况。\n\n"
    "支持以下查询：\n"
    "• 发送「今日」或「今天」— 查看今日做题情况\n"
    "• 发送「本周」或「这周」— 查看本周做题情况\n"
    "• 发送「作业」— 查看当前作业完成情况\n\n"
    "其他问题也可以直接发送，我会尽力解答。"
)


async def _handle_external_message(db: Session, external_userid: str, text: str) -> str:
    """
    处理外部联系人（家长）发来的消息。
    返回回复文本，调用方负责通过 add_msg_template_single 发送。
    """
    txt = (text or "").strip()

    # 帮助指令
    if txt in ("帮助", "help", "?", "？", "菜单"):
        return _EXTERNAL_HELP_TEXT

    # 查找绑定的学生
    student_uids = _get_bound_students(db, external_userid)

    if not student_uids:
        # 未绑定：走知识库 AI 回复
        return await answer_with_rag_and_memory(db, f"ext:{external_userid}", txt)

    intent = _match_external_intent(txt)

    if intent == "unknown":
        # 非查询指令：走知识库 AI 回复，带学生上下文
        extra = f"【会话上下文】家长已绑定学生 student_uid={','.join(student_uids)}。"
        return await answer_with_rag_and_memory(db, f"ext:{external_userid}", txt, extra_system=extra)

    # 有绑定学生时，逐个生成查询结果
    parts: list[str] = []
    for uid in student_uids:
        if intent == "today":
            parts.append(_format_external_today_reply(uid, db))
        elif intent == "week":
            parts.append(_format_external_week_reply(uid, db))
        elif intent == "hw":
            parts.append(_format_external_hw_reply(uid, db))

    return "\n\n---\n\n".join(parts) if parts else "暂无数据，请稍后再试。"


_MASTER_AT_CMDS = ("@群发", "@查询", "@发送", "@帮助", "@help")


def _looks_like_operator_command(cmd: str) -> bool:
    s = (cmd or "").strip()
    if not s:
        return False
    # #s 开头视为运维指令
    if re.match(r"^\s*[#＃]s", s, flags=re.IGNORECASE):
        return True
    # @ 精确命令
    return any(s.startswith(c) for c in _MASTER_AT_CMDS)


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


def _dump_weekly_snapshot_file_with_suffix(db: Session, week_key: str, suffix_date: str) -> dict[str, object]:
    """和 _dump_weekly_snapshot_file 类似，但文件名带日期后缀，避免同一周覆盖。"""
    wk = (week_key or "").strip()
    sd = (suffix_date or "").strip()
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

    name = f"{wk}_{sd}.json" if sd else f"{wk}.json"
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


def _student_records_by_name(db: Session, name: str) -> list[StudentRecord]:
    key = _norm_student_name(name)
    if not key:
        return []
    return (
        db.query(StudentRecord)
        .filter(StudentRecord.name_key == key)
        .order_by(StudentRecord.student_uid.asc())
        .all()
    )


def _format_h5_student_stats_html(stats: dict, student_uid: str) -> str:
    """作业题 / 本周 / 今日 展示块（依赖 get_student_hydro_stats 返回结构）。"""
    name = html.escape(str(stats.get("name") or ""))
    uid_disp = html.escape(str(stats.get("uid") or student_uid))
    hw_title = html.escape(str(stats.get("hw_title") or ""))
    tasks = stats.get("hw_tasks") or []
    pills: list[str] = []
    for t in tasks:
        pid = html.escape(str(t.get("pid") or ""))
        cls = "task-ac" if t.get("ac") else "task-no"
        pills.append(f'<span class="{cls}">{pid}</span>')
    pills_html = " ".join(pills) if pills else "<span style='color:#6b7280'>当前无进行中作业或未分配到班级作业。</span>"

    week_pids = stats.get("week_ac_pids") or []
    week_txt = "、".join(html.escape(str(x)) for x in week_pids) if week_pids else "（无）"
    today_pids = stats.get("today_ac_pids") or []
    today_txt = "、".join(html.escape(str(x)) for x in today_pids) if today_pids else "（无）"

    extra_css = """
    <style>
    .task-ac { display:inline-block; margin:4px 6px 4px 0; padding:6px 10px; border-radius:8px;
      background:#ecfdf3; color:#166534; border:1px solid #bbf7d0; font-weight:600; }
    .task-no { display:inline-block; margin:4px 6px 4px 0; padding:6px 10px; border-radius:8px;
      background:#f3f4f6; color:#6b7280; border:1px solid #e5e7eb; }
    </style>
    """
    return f"""
    {extra_css}
    <h2>学生做题统计</h2>
    <p style="color:#374151">{name}　<code>{uid_disp}</code></p>
    <div class="card">
      <h3 style="margin-top:0">当前作业题</h3>
      <p style="color:#6b7280;font-size:14px">作业：{hw_title}</p>
      <p style="color:#6b7280;font-size:13px">绿色=已在 OJ 上 AC 该题；灰色=尚未 AC。</p>
      <div style="line-height:1.8">{pills_html}</div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">本周 AC</h3>
      <p>题数：<strong>{int(stats.get("week_ac_count") or 0)}</strong></p>
      <p style="word-break:break-all">题号：{week_txt}</p>
    </div>
    <div class="card">
      <h3 style="margin-top:0">今日</h3>
      <p>AC 题数：<strong>{int(stats.get("today_ac") or 0)}</strong>　提交次数：<strong>{int(stats.get("today_submits") or 0)}</strong></p>
      <p style="word-break:break-all">今日 AC 题号：{today_txt}</p>
    </div>
    <p style="color:#9ca3af;font-size:13px">数据来自 Hydro；今日/本周以 OJ 服务器本地日历为准。</p>
    """


def _list_groups(db: Session) -> str:
    wk = _latest_week_key(db)
    if not wk:
        return "暂无周数据，请先同步 Hydro 周数据。"
    rows = db.query(StudentWeeklyMetric).filter(StudentWeeklyMetric.week_key == wk).all()
    class_count: dict[str, int] = {}
    for r in rows:
        try:
            gs = json.loads(r.groups_json or "[]")
        except Exception:
            gs = []
        for g in gs:
            gn = str(g).strip()
            if gn:
                class_count[gn] = class_count.get(gn, 0) + 1
    if not class_count:
        return f"周 {wk} 暂无班级数据。"
    lines = [f"班级列表（{wk}，共 {len(class_count)} 个）："]
    for gn, cnt in sorted(class_count.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {gn}：{cnt} 人")
    return "\n".join(lines)


_MASTER_HELP_TEXT = (
    "主账号指令（精确命令）：\n"
    "\n"
    "【查询】\n"
    "@查询 今日              — 今日全量统计\n"
    "@查询 今日 班级=xxx     — 今日某班级统计\n"
    "@查询 今日 学生=uid     — 今日某学生数据\n"
    "@查询 本周              — 本周全量统计\n"
    "@查询 本周 班级=xxx     — 本周某班级统计\n"
    "@查询 班级列表          — 列出所有班级\n"
    "\n"
    "【群发】\n"
    "@群发 消息内容          — 全量群发给所有家长\n"
    "@群发 班级=xxx 消息内容 — 按班级群发\n"
    "\n"
    "【发送周报】\n"
    "@发送 周报              — 全量发送周报\n"
    "@发送 周报 班级=xxx     — 按班级发送周报\n"
    "@发送 周报 学生=uid     — 发送单个学生周报\n"
    "\n"
    "【其他】\n"
    "周报 帮助               — 周报定时指令\n"
    "#s 自然语言指令         — AI 解析运维指令（兜底）"
)


async def _handle_master_command(db: Session, operator_id: str, text: str) -> str | None:
    """处理主账号 @ 精确命令，优先级高于 AI 解析。"""
    ops = _parse_op_ids()
    op = (operator_id or "").strip().lower()
    if not ops or op not in ops:
        return None

    cmd = (text or "").strip()
    if not cmd:
        return None

    # 仅处理 @ 开头的精确命令
    if not any(cmd.startswith(c) for c in _MASTER_AT_CMDS):
        return None

    sender_userid = (settings.wecom_external_sender_id or "").strip() or operator_id

    # @帮助
    if cmd in ("@帮助", "@help"):
        return _MASTER_HELP_TEXT

    # @群发
    if cmd.startswith("@群发"):
        body = cmd[3:].strip()
        group = ""
        # 支持 班级=xxx 或 班级 xxx（空格）
        m = re.match(r"班级\s*[=＝]\s*([^\s]+)\s*([\s\S]*)", body)
        if not m:
            m = re.match(r"班级\s+([^\s]+)\s+([\s\S]+)", body)
        if m:
            group = m.group(1).strip()
            body = m.group(2).strip()
        if not body:
            return "格式：@群发 [班级=xxx] 消息内容\n示例：@群发 班级=CSP-J4 明天上课时间调整为下午3点"
        ok, fail, skip = await _send_custom_message_to_scope(
            db,
            sender_userid=sender_userid,
            message=body,
            scope="group" if group else "all",
            group=group,
        )
        scope_desc = f"班级={group}" if group else "全量"
        return f"群发完成（{scope_desc}）：成功 {ok}，失败 {fail}，跳过 {skip}"

    # @查询
    if cmd.startswith("@查询"):
        body = cmd[3:].strip()
        if not body:
            return "格式：@查询 今日/本周/班级列表 [班级=xxx] [学生=uid]"

        if body in ("班级列表", "班级"):
            return _list_groups(db)

        is_today = bool(re.match(r"今[天日]", body))
        is_week = "本周" in body

        gm = re.search(r"班级\s*[=＝]\s*([^\s]+)", body)
        sm = re.search(r"学生\s*[=＝]\s*([A-Za-z0-9_-]+)", body)
        group = gm.group(1).strip() if gm else ""
        student_uid = sm.group(1).strip() if sm else ""

        if is_today:
            today = get_today_students_stats()
            if student_uid:
                row = next((r for r in today if str(r.get("uid") or "").strip() == student_uid), None)
                if not row:
                    return f"未找到 student_uid={student_uid} 今日数据。"
                return (
                    f"学生今日数据：{row.get('name', '')}（{student_uid}）\n"
                    f"提交：{int(row.get('today_submits') or 0)}，AC：{int(row.get('today_ac') or 0)}"
                )
            return _summarize_today_stats(today, group=group)

        if is_week:
            return _summarize_weekly_group_stats(db, group=group)

        return "格式：@查询 今日/本周 [班级=xxx] [学生=uid]\n或：@查询 班级列表"

    # @发送
    if cmd.startswith("@发送"):
        body = cmd[3:].strip()
        if not body:
            return "格式：@发送 周报 [班级=xxx] [学生=uid]"

        if body.startswith("周报"):
            gm = re.search(r"班级\s*[=＝]\s*([^\s]+)", body)
            sm = re.search(r"学生\s*[=＝]\s*([A-Za-z0-9_-]+)", body)
            group = gm.group(1).strip() if gm else ""
            student_uid = sm.group(1).strip() if sm else ""

            if student_uid:
                week_raw = get_weekly_students(db, force_refresh=True)
                target_raw = next((r for r in week_raw if str(r.get("uid") or "").strip() == student_uid), None)
                if not target_raw:
                    return f"未找到 student_uid={student_uid} 的周数据。"
                targets = db.query(ParentStudentBinding).filter(ParentStudentBinding.student_uid == student_uid).all()
                ok = 0
                fail = 0
                for b in targets:
                    ext = (b.external_userid or "").strip()
                    if not ext:
                        continue
                    try:
                        content = render_weekly_report(target_raw, template_text=_load_weekly_template())
                        await add_msg_template_single(external_userid=ext, content=content, sender_userid=sender_userid)
                        ok += 1
                    except Exception:
                        fail += 1
                return f"已发送学生周报：student_uid={student_uid}，成功 {ok}，失败 {fail}"

            res = await send_weekly_reports(
                db,
                sender=sender_userid,
                group=group,
                force_refresh=True,
                template_text=_load_weekly_template(),
            )
            scope_desc = f"班级={res.group}" if res.group else "全量"
            return f"周报发送完成（{scope_desc}）：week={res.week_key} 成功 {res.ok}，失败 {res.fail}，跳过 {res.skip}"

        return "格式：@发送 周报 [班级=xxx] [学生=uid]"

    return None


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

    if action == "stats_today_group_table" and gp:
        today = get_today_students_stats()
        return _format_today_group_table(today, group=gp)

    if action == "stats_today_student" and su:
        today = get_today_students_stats()
        row = next((r for r in today if str(r.get("uid") or "").strip() == su), None)
        if not row:
            return f"未找到 student_uid={su} 今日数据。"
        return (
            f"学生今日做题数据：{row.get('name','')}（{su}）\n"
            f"提交：{int(row.get('today_submits') or 0)}，AC：{int(row.get('today_ac') or 0)}"
        )

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


class WxBotChatIn(BaseModel):
    sender: str = Field(min_length=1)  # 微信昵称/备注名
    message: str = Field(min_length=1)
    student_uid: str = ""  # 可选，指定学生；为空时按 sender 查绑定


@app.post("/api/wx-bot/chat")
async def api_wx_bot_chat(body: WxBotChatIn, db: Session = Depends(get_db)):
    """
    wxAuto 本地脚本调用的 API。
    接收 sender（微信昵称）+ message，走意图识别 + 数据查询 / 知识库 AI。
    """
    sender = body.sender.strip()
    msg = body.message.strip()
    student_uid = (body.student_uid or "").strip()

    # 闲聊直接回复
    if _is_chitchat(msg):
        chitchat_map = {
            "谢谢": "不客气，有其他问题随时告诉我！",
            "感谢": "不客气，有其他问题随时告诉我！",
            "再见": "再见，有需要随时联系！",
            "拜拜": "拜拜，有需要随时联系！",
        }
        for kw, reply in chitchat_map.items():
            if kw in msg:
                return {"reply": reply}
        return {"reply": "您好！请问有什么可以帮您？"}

    # 帮助
    if msg in ("帮助", "help", "?", "？", "菜单"):
        return {"reply": _EXTERNAL_HELP_TEXT}

    # 如果没有指定 student_uid，尝试通过 sender 名称查找绑定
    # 先查 ExternalContact 的 remark/name 匹配 sender
    bound_uids: list[str] = []
    if student_uid:
        bound_uids = [student_uid]
    else:
        # 方式1：通过 ExternalContact 的 name/remark 匹配
        contacts = db.query(ExternalContact).filter(
            (ExternalContact.name == sender) | (ExternalContact.remark.contains(sender))
        ).all()
        for c in contacts:
            if c.student_uid_hint:
                bound_uids.append(c.student_uid_hint)
        # 方式2：通过 StudentRecord 的 display_name 匹配（sender 可能是学生姓名）
        if not bound_uids:
            students = _student_records_by_name(db, sender)
            bound_uids = [s.student_uid for s in students]

    # 意图识别
    intent = _match_external_intent(msg)

    if intent != "unknown" and bound_uids:
        parts: list[str] = []
        for uid in bound_uids:
            if intent == "today":
                parts.append(_format_external_today_reply(uid, db))
            elif intent == "week":
                parts.append(_format_external_week_reply(uid, db))
            elif intent == "hw":
                parts.append(_format_external_hw_reply(uid, db))
        reply = "\n\n---\n\n".join(parts) if parts else "暂无数据，请稍后再试。"
        return {"reply": reply}

    # 其他问题走知识库 AI
    extra = ""
    if bound_uids:
        extra = f"【会话上下文】当前绑定学生 student_uid={','.join(bound_uids)}。"
    reply = await answer_with_rag_and_memory(db, f"wx:{sender}", msg, extra_system=extra or None)
    return {"reply": reply}
async def api_h5_student_stats_data(
    name: str = "",
    student_uid: str = "",
    db: Session = Depends(get_db),
):
    """JSON：按姓名（名录唯一匹配）或 student_uid 查询。"""
    name = (name or "").strip()
    student_uid = (student_uid or "").strip()
    if not student_uid and name:
        rows = _student_records_by_name(db, name)
        if len(rows) == 1:
            student_uid = rows[0].student_uid
        elif len(rows) == 0:
            raise HTTPException(status_code=404, detail="未找到该姓名，请核对或与名录同步一致")
        else:
            raise HTTPException(
                status_code=400,
                detail="存在重名，请在网页上点击具体 student_uid 或使用参数 student_uid",
            )
    if not student_uid:
        raise HTTPException(status_code=400, detail="请提供 name 或 student_uid")
    try:
        return get_student_hydro_stats(student_uid)
    except Exception as e:
        logger.exception("h5 student stats")
        raise HTTPException(status_code=502, detail=str(e)[:800]) from e


@app.get("/h5/student-stats", response_class=HTMLResponse)
async def h5_student_stats(
    request: Request,
    name: str = "",
    student_uid: str = "",
    db: Session = Depends(get_db),
):
    """输入姓名查询（名录匹配）；重名时点击链接带上 student_uid。"""
    name = (name or "").strip()
    student_uid = (student_uid or "").strip()
    remembered_name = await _h5_get_last_name_for_source(request)
    if not name and remembered_name:
        name = remembered_name

    form_html = f"""
    <h2>学生做题统计</h2>
    <div class="card">
      <form method="get" action="/h5/student-stats">
        <label>学生姓名</label>
        <input name="name" value="{html.escape(name)}" placeholder="与后台学生名录一致" style="max-width:280px" />
        <button type="submit" style="margin-left:8px">查询</button>
      </form>
      <p style="color:#6b7280;font-size:13px;margin-top:12px">姓名来自系统学生名录（<a href="/admin/students">管理</a>）。多人同名时请点选下方链接。</p>
    </div>
    """

    resolved_uid = ""
    if not student_uid and name:
        rows = _student_records_by_name(db, name)
        if len(rows) == 0:
            body = (
                form_html
                + '<div class="card" style="border-color:#fecaca;background:#fff1f2">未找到该姓名。请确认名录已同步，或尝试与班主任登记的写法完全一致。</div>'
            )
            return HTMLResponse(html_page("学生统计", body))
        if len(rows) > 1:
            lines = []
            for r in rows:
                q = quote(str(r.student_uid), safe="")
                lines.append(
                    f'<li><a href="/h5/student-stats?student_uid={q}">{html.escape(r.display_name)} '
                    f'<code>{html.escape(r.student_uid)}</code></a></li>'
                )
            body = (
                form_html
                + '<div class="card"><p><strong>同名学生</strong>，请选择：</p><ul>'
                + "\n".join(lines)
                + "</ul></div>"
            )
            return HTMLResponse(html_page("学生统计", body))
        resolved_uid = rows[0].student_uid
        await _h5_set_last_name_for_source(request, name)
    elif student_uid:
        resolved_uid = student_uid

    if not resolved_uid:
        return HTMLResponse(html_page("学生统计", form_html))

    ok, remain_sec = await _h5_try_consume_query_quota(request)
    if not ok:
        mm = remain_sec // 60
        ss = remain_sec % 60
        tip = f"查询过于频繁：同一来源每小时仅可查询 1 次，请在 {mm} 分 {ss} 秒后重试。"
        body = form_html + (
            '<div class="card" style="border-color:#fecaca;background:#fff1f2">'
            + html.escape(tip)
            + "</div>"
        )
        return HTMLResponse(html_page("学生统计", body))

    try:
        stats = get_student_hydro_stats(resolved_uid)
    except Exception as e:
        logger.exception("h5 get_student_hydro_stats")
        body = (
            form_html
            + "<h2>拉取 Hydro 数据失败</h2>"
            + '<div class="card"><pre style="white-space:pre-wrap">'
            + html.escape(str(e)[:2000])
            + "</pre></div>"
        )
        return HTMLResponse(html_page("学生统计", body))

    if stats.get("error"):
        body = (
            form_html
            + "<h2>数据异常</h2>"
            + '<div class="card"><pre style="white-space:pre-wrap">'
            + html.escape(json.dumps(stats, ensure_ascii=False, indent=2))
            + "</pre></div>"
        )
        return HTMLResponse(html_page("学生统计", body))

    if name:
        await _h5_set_last_name_for_source(request, name)
    body = form_html + _format_h5_student_stats_html(stats, resolved_uid)
    return HTMLResponse(html_page("学生统计", body))


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


async def _send_deferred_external_reply(external_userid: str, text: str) -> None:
    """异步处理外部联系人消息，通过主动推送回复。"""
    try:
        with next(get_db()) as db:  # type: ignore[arg-type]
            reply = await _handle_external_message(db, external_userid, text)
        if len(reply) > WECOM_REPLY_MAX_CHARS:
            reply = reply[:WECOM_REPLY_MAX_CHARS] + "\n\n（内容较长，已截断）"
        sender = (settings.wecom_external_sender_id or "").strip()
        if not sender:
            logger.warning("external reply skipped: WECOM_EXTERNAL_SENDER_ID not configured")
            return
        await add_msg_template_single(external_userid=external_userid, content=reply, sender_userid=sender)
        logger.info("external reply sent: to=%s len=%s", external_userid, len(reply))
    except Exception:
        logger.exception("external reply failed: to=%s", external_userid)


@app.post("/wecom/callback")
async def wecom_callback(request: Request, msg_signature: str, timestamp: str, nonce: str, db: Session = Depends(get_db)):
    if crypto is None:
        return PlainTextResponse("wecom not configured", status_code=500)
    body = await request.body()
    logger.info("callback raw: len=%s body_prefix=%s", len(body), body[:300])
    enc = parse_encrypted_xml(body)
    if not crypto.verify_signature(msg_signature, timestamp, nonce, enc.encrypt):
        logger.warning("callback signature verify failed")
        return PlainTextResponse("invalid signature", status_code=403)

    try:
        plain_xml = crypto.decrypt(enc.encrypt)
    except Exception as e:
        logger.exception("callback decrypt failed: %s", e)
        return PlainTextResponse("decrypt failed", status_code=400)

    logger.info("callback decrypted xml: %s", plain_xml[:500] if isinstance(plain_xml, (str, bytes)) else str(plain_xml)[:500])
    msg = parse_plain_xml(plain_xml)

    # 调试日志：记录所有进来的消息，方便排查外部联系人消息格式
    logger.info("callback msg: from=%s to=%s type=%s msg_id=%s", msg.from_user_name, msg.to_user_name, msg.msg_type, msg.msg_id)

    # 外部联系人消息：异步处理后主动推送，直接返回空响应
    if _is_external_userid(msg.from_user_name):
        if msg.msg_type == "text" and msg.content.strip():
            txt = msg.content.strip()
            logger.info("external msg received: from=%s msg_id=%s content=%s", msg.from_user_name, msg.msg_id, txt[:200])
            asyncio.create_task(_send_deferred_external_reply(msg.from_user_name, txt))
        return Response(content=b"", media_type="text/plain")

    # 内部员工消息：走原有逻辑
    if msg.msg_type != "text" or not msg.content.strip():
        reply_text = "目前只支持文本咨询。请发送文字问题。"
    else:
        txt = msg.content.strip()
        logger.info("wecom text received: from=%s msg_id=%s content=%s", msg.from_user_name, msg.msg_id, txt[:200])
        if _wecom_text_has_side_effects(txt) and not await _wecom_try_begin_side_effect(msg.msg_id):
            reply_text = "本指令已在首次回调中执行；企业微信重复回调已忽略，避免重复发送。"
        else:
            cmd_reply = await _handle_weekly_command(db, msg.from_user_name, txt)
            if cmd_reply is not None:
                reply_text = cmd_reply
            else:
                master_reply = await _handle_master_command(db, msg.from_user_name, txt)
                if master_reply is not None:
                    reply_text = master_reply
                else:
                    ai_cmd_reply = await _handle_operator_ai_command(db, msg.from_user_name, txt)
                    if ai_cmd_reply is not None:
                        reply_text = ai_cmd_reply
                    else:
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
      <div><a href="/admin/exams">考试系统（主页/试卷上传）</a></div>
      <div>考试系统入口：<code>/exam</code></div>
      <div><a href="/admin/chats">会话记录</a></div>
      <div><a href="/admin/reports">周报发送</a></div>
      <div><a href="/admin/external-contacts">外部联系人同步与匹配</a></div>
      <div><a href="/admin/students">学生名录（小程序姓名匹配）</a></div>
      <div><a href="/admin/bindings">已匹配成功（家长绑定）</a></div>
      <div><a href="/admin/weekly-files">每周学生更新数据文件</a></div>
      <div><a href="/admin/binding-requests">待审核绑定</a></div>
      <div><a href="/admin/push">主动推送</a></div>
      <div><a href="/admin/today-class-stats">今日各班级做题统计（Hydro）</a></div>
      <div>学生做题统计（输入姓名）：<code>/h5/student-stats</code></div>
    </div>
    """
    return html_page("Admin", body)


def _today_class_stats_cache_path() -> Path:
    return Path(settings.data_dir) / "today_class_stats_cache.json"


def _ensure_exams_store() -> None:
    EXAMS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not EXAMS_META_PATH.exists():
        EXAMS_META_PATH.write_text("[]", encoding="utf-8")


def _load_exams_meta() -> list[dict]:
    _ensure_exams_store()
    try:
        raw = json.loads(EXAMS_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        raw = []
    items: list[dict] = []
    for x in raw or []:
        if not isinstance(x, dict):
            continue
        eid = str(x.get("id") or "").strip()
        title = str(x.get("title") or "").strip()
        filename = Path(str(x.get("filename") or "")).name
        uploaded_at = str(x.get("uploaded_at") or "")
        if not eid or not filename:
            continue
        items.append(
            {
                "id": eid,
                "title": title or Path(filename).stem,
                "filename": filename,
                "uploaded_at": uploaded_at,
            }
        )
    return items


def _save_exams_meta(items: list[dict]) -> None:
    _ensure_exams_store()
    EXAMS_META_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_exam_filename(name: str) -> str:
    base = Path(name or "").name
    if not base:
        base = f"exam-{int(time.time())}.html"
    if not base.lower().endswith(".html"):
        base += ".html"
    return base


def _exam_id_for_filename(filename: str) -> str:
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]


def _inject_exam_redo_script(html_text: str, exam_id: str) -> str:
    marker = "__EXAM_REDO_INJECTED__"
    if marker in html_text:
        return html_text
    script = f"""
<script>
/* {marker} */
(function() {{
  var storageKey = "exam_submitted_" + {json.dumps(exam_id)};
  function readBtnText(btn) {{
    if (!btn) return "";
    if (btn.tagName === "INPUT") return String(btn.value || "");
    return String(btn.innerText || btn.textContent || "");
  }}
  function writeBtnText(btn, text) {{
    if (!btn) return;
    if (btn.tagName === "INPUT") btn.value = text;
    else btn.textContent = text;
  }}
  function findSubmitButton() {{
    var fixed = document.getElementById("btn-submit") || document.getElementById("submit-btn");
    if (fixed) return fixed;
    var btns = document.querySelectorAll("button, input[type='button'], input[type='submit']");
    for (var i = 0; i < btns.length; i++) {{
      var t = readBtnText(btns[i]).trim();
      if (t.indexOf("提交") >= 0 || t.indexOf("试卷") >= 0 || t.indexOf("试题") >= 0) return btns[i];
    }}
    return null;
  }}
  function applyRedoState() {{
    if (localStorage.getItem(storageKey) !== "1") return;
    var btn = findSubmitButton();
    if (!btn) return;
    btn.disabled = false;
    writeBtnText(btn, "重做试题");
    btn.onclick = function(e) {{
      e.preventDefault();
      localStorage.removeItem(storageKey);
      location.reload();
    }};
  }}
  function bindSubmitWatcher() {{
    document.addEventListener("click", function(e) {{
      var target = e.target && e.target.closest ? e.target.closest("button, input[type='button'], input[type='submit']") : null;
      if (!target) return;
      var txt = readBtnText(target).trim();
      if (txt.indexOf("重做试题") >= 0) return;
      if (txt.indexOf("提交") < 0 && txt.indexOf("试卷") < 0 && txt.indexOf("试题") < 0) return;
      setTimeout(function() {{
        localStorage.setItem(storageKey, "1");
        applyRedoState();
      }}, 10);
    }}, true);
  }}
  window.addEventListener("load", function() {{
    bindSubmitWatcher();
    applyRedoState();
    setTimeout(applyRedoState, 250);
  }});
}})();
</script>
"""
    lower = html_text.lower()
    pos = lower.rfind("</body>")
    if pos >= 0:
        return html_text[:pos] + script + "\n" + html_text[pos:]
    return html_text + "\n" + script


def _render_exam_home(exams: list[dict]) -> str:
    exams_json = json.dumps(exams, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>竞赛模拟考试系统</title>
  <style>
    :root {{ --sidebar-width: 280px; --primary-color: #2563eb; --bg-color: #f1f5f9; --text-main: #1e293b; }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg-color); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }}
    .sidebar {{ width: var(--sidebar-width); background: #1e293b; color: #fff; display: flex; flex-direction: column; box-shadow: 4px 0 10px rgba(0,0,0,0.1); }}
    .sidebar-header {{ padding: 2rem 1.5rem; border-bottom: 1px solid #334155; }}
    .sidebar-header h1 {{ font-size: 1.25rem; font-weight: 700; letter-spacing: 1px; }}
    .exam-list {{ flex: 1; padding: 1rem 0; overflow-y: auto; }}
    .exam-item {{ padding: 1rem 1.5rem; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 10px; border-left: 4px solid transparent; color: #94a3b8; }}
    .exam-item:hover {{ background: #334155; color: #f8fafc; }}
    .exam-item.active {{ background: #0f172a; color: #fff; border-left-color: var(--primary-color); }}
    .main-content {{ flex: 1; display: flex; flex-direction: column; background: #fff; position: relative; }}
    .top-bar {{ height: 60px; background: #fff; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; padding: 0 2rem; justify-content: space-between; }}
    #current-title {{ font-weight: 600; color: var(--text-main); }}
    .iframe-container {{ flex: 1; width: 100%; height: 100%; border: none; background: var(--bg-color); }}
    iframe {{ width: 100%; height: 100%; border: none; }}
    .welcome-screen {{ position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #fff; z-index: 10; text-align: center; }}
  </style>
</head>
<body>
  <div class="sidebar">
    <div class="sidebar-header">
      <h1>模拟考试系统</h1>
      <p style="font-size: 0.8rem; color: #64748b; margin-top: 5px;">Exam Simulation System</p>
    </div>
    <div class="exam-list" id="examList"></div>
  </div>
  <div class="main-content">
    <div class="top-bar">
      <div id="current-title">请选择试卷</div>
      <div style="font-size: 0.9rem; color: #64748b;">在线自测模式</div>
    </div>
    <div class="welcome-screen" id="welcome">
      <div style="font-size: 4rem; margin-bottom: 1rem;">📝</div>
      <h2>欢迎进入考试中心</h2>
      <p style="color: #64748b; margin-top: 10px;">从左侧列表点击试卷开始答题</p>
    </div>
    <div class="iframe-container">
      <iframe id="examFrame" src="about:blank"></iframe>
    </div>
  </div>
  <script>
    const exams = {exams_json};
    const examListEl = document.getElementById("examList");
    const examFrame = document.getElementById("examFrame");
    const welcomeScreen = document.getElementById("welcome");
    const currentTitleEl = document.getElementById("current-title");
    function loadExam(el, examId, title) {{
      document.querySelectorAll(".exam-item").forEach((item) => item.classList.remove("active"));
      if (el) el.classList.add("active");
      welcomeScreen.style.display = "none";
      currentTitleEl.innerText = title;
      examFrame.src = "/exam/paper/" + encodeURIComponent(examId);
    }}
    function init() {{
      if (!exams.length) {{
        examListEl.innerHTML = '<div style="padding:1rem 1.5rem;color:#cbd5e1">暂无试卷，请联系管理员在后台上传。</div>';
        return;
      }}
      examListEl.innerHTML = exams.map((exam) =>
        '<div class="exam-item" data-eid="' + exam.id + '"><span>📄</span><span>' + exam.title + "</span></div>"
      ).join("");
      document.querySelectorAll(".exam-item").forEach((item) => {{
        item.addEventListener("click", function() {{
          const eid = this.getAttribute("data-eid");
          const e = exams.find((x) => x.id === eid);
          if (!e) return;
          loadExam(this, e.id, e.title);
        }});
      }});
    }}
    init();
  </script>
</body>
</html>"""


@app.get("/admin/today-class-stats", response_class=HTMLResponse)
async def admin_today_class_stats(_user: str = Depends(require_admin)):
    cache_path = _today_class_stats_cache_path()
    updated_at = ""
    err_msg = ""
    groups: list = []
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            updated_at = str(data.get("updated_at") or "")
            groups = data.get("groups") or []
            err_msg = str(data.get("error") or "")
        except Exception as e:
            err_msg = f"缓存读取失败: {e}"

    parts: list[str] = []
    parts.append("<h2>今日各班级做题统计</h2>")
    parts.append(
        '<div class="card" style="margin-bottom:14px">'
        '<form action="/admin/today-class-stats/refresh" method="post" style="display:inline">'
        '<button type="submit">一键更新（从 Hydro 拉取）</button>'
        "</form>"
        '<span style="margin-left:12px;color:#666">数据来自 Hydro 远程查询，可能需数十秒。</span>'
        "</div>"
    )
    if updated_at:
        parts.append(f'<p style="color:#666">上次更新时间：<code>{html.escape(updated_at)}</code></p>')
    else:
        parts.append('<p style="color:#666">尚未拉取过数据，请点击「一键更新」。</p>')

    if err_msg:
        parts.append(
            '<div class="card" style="border-color:#fecaca;background:#fff1f2">'
            f"<strong>错误</strong><pre style='white-space:pre-wrap'>{html.escape(err_msg)}</pre></div>"
        )

    if not groups and not err_msg:
        parts.append("<p>暂无班级数据。</p>")
    elif groups:
        for g in groups:
            gname = html.escape(str(g.get("group") or ""))
            tac = g.get("total_ac")
            tsub = g.get("total_submits")
            cnt = g.get("student_count")
            parts.append(
                f'<div class="card" style="margin-bottom:16px">'
                f"<h3>{gname}</h3>"
                f"<p>班级合计：AC <strong>{tac}</strong>　提交 <strong>{tsub}</strong>　学生数 {cnt}</p>"
                "<table><thead><tr><th>姓名</th><th>今日 AC</th><th>今日提交</th><th>uid</th></tr></thead><tbody>"
            )
            for s in g.get("students") or []:
                nm = html.escape(str(s.get("name") or ""))
                uid = html.escape(str(s.get("uid") or ""))
                parts.append(
                    f"<tr><td>{nm}</td><td>{s.get('today_ac')}</td><td>{s.get('today_submits')}</td>"
                    f"<td><code>{uid}</code></td></tr>"
                )
            parts.append("</tbody></table></div>")

    body = "\n".join(parts)
    body += '<div style="margin-top:16px"><a href="/admin/">返回</a></div>'
    return html_page("今日班级统计", body)


@app.post("/admin/today-class-stats/refresh")
async def admin_today_class_stats_refresh(_user: str = Depends(require_admin)):
    cache_path = _today_class_stats_cache_path()
    old_groups: list = []
    if cache_path.exists():
        try:
            old = json.loads(cache_path.read_text(encoding="utf-8"))
            old_groups = old.get("groups") or []
        except Exception:
            pass
    try:
        rows = get_today_students_stats()
        groups = compute_today_stats_by_group(rows)
        payload = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "groups": groups,
            "error": None,
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.exception("today-class-stats refresh failed")
        payload = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "groups": old_groups,
            "error": str(e)[:4000],
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return Response(status_code=303, headers={"Location": "/admin/today-class-stats"})


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


@app.get("/exam", response_class=HTMLResponse)
@app.get("/exam/", response_class=HTMLResponse)
async def exam_home():
    exams = _load_exams_meta()
    return HTMLResponse(_render_exam_home(exams))


@app.get("/exam/paper/{exam_id}", response_class=HTMLResponse)
async def exam_paper(exam_id: str):
    eid = (exam_id or "").strip()
    exams = _load_exams_meta()
    row = next((x for x in exams if x.get("id") == eid), None)
    if not row:
        raise HTTPException(status_code=404, detail="exam not found")
    fp = EXAMS_DATA_DIR / Path(str(row.get("filename") or "")).name
    if not fp.exists():
        raise HTTPException(status_code=404, detail="exam file not found")
    raw = fp.read_text(encoding="utf-8", errors="ignore")
    return HTMLResponse(_inject_exam_redo_script(raw, eid))


@app.get("/admin/exams", response_class=HTMLResponse)
async def admin_exams(_user: str = Depends(require_admin)):
    exams = _load_exams_meta()
    rows = "".join(
        f"<tr>"
        f"<td>{idx + 1}</td>"
        f"<td>{html.escape(str(e.get('title') or ''))}</td>"
        f"<td><code>{html.escape(str(e.get('filename') or ''))}</code></td>"
        f"<td>{html.escape(str(e.get('uploaded_at') or ''))}</td>"
        f"<td>"
        f"<a href='/exam/paper/{html.escape(str(e.get('id') or ''), quote=True)}' target='_blank'>预览</a>"
        f"<form action='/admin/exams/delete' method='post' style='display:inline;margin-left:8px'>"
        f"<input type='hidden' name='exam_id' value='{html.escape(str(e.get('id') or ''), quote=True)}' />"
        f"<button type='submit' class='secondary'>删除</button></form>"
        f"</td>"
        f"</tr>"
        for idx, e in enumerate(exams)
    )
    body = f"""
    <h2>考试系统（主页与试卷管理）</h2>
    <div class="card">
      <div>考试主页入口：<a href="/exam" target="_blank"><code>/exam</code></a></div>
      <div style="margin-top:6px;color:#666">试卷要求：上传 HTML 文件即可，页面结构与初赛一/二/三保持一致即可自动接入。</div>
    </div>
    <div class="card">
      <form action="/admin/exams/upload" method="post" enctype="multipart/form-data">
        <div style="margin-bottom:10px">
          <label>试卷标题</label>
          <input name="title" placeholder="例如：初赛模拟测试（四）" />
        </div>
        <div style="margin-bottom:10px">
          <label>试卷文件（.html）</label>
          <input type="file" name="file" accept=".html,text/html" />
        </div>
        <button type="submit">上传并加入主页</button>
      </form>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>#</th><th>标题</th><th>文件名</th><th>上传时间</th><th>操作</th></tr></thead>
        <tbody>{rows or '<tr><td colspan="5">暂无试卷，请先上传。</td></tr>'}</tbody>
      </table>
    </div>
    <div><a href="/admin/">返回</a></div>
    """
    return html_page("Exams", body)


@app.post("/admin/exams/upload")
async def admin_exams_upload(
    title: str = Form(""),
    file: UploadFile = File(...),
    _user: str = Depends(require_admin),
):
    src_name = file.filename or "exam.html"
    safe_name = _safe_exam_filename(src_name)
    data = await file.read()
    if not safe_name.lower().endswith(".html"):
        raise HTTPException(status_code=400, detail="only html allowed")
    _ensure_exams_store()
    fp = EXAMS_DATA_DIR / safe_name
    fp.write_bytes(data)

    exams = _load_exams_meta()
    exam_id = _exam_id_for_filename(safe_name)
    exam_title = (title or "").strip() or Path(safe_name).stem
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existed = False
    for e in exams:
        if e.get("id") == exam_id:
            e["title"] = exam_title
            e["filename"] = safe_name
            e["uploaded_at"] = now
            existed = True
            break
    if not existed:
        exams.append(
            {
                "id": exam_id,
                "title": exam_title,
                "filename": safe_name,
                "uploaded_at": now,
            }
        )
    _save_exams_meta(exams)
    return Response(status_code=303, headers={"Location": "/admin/exams"})


@app.post("/admin/exams/delete")
async def admin_exams_delete(
    exam_id: str = Form(...),
    _user: str = Depends(require_admin),
):
    eid = (exam_id or "").strip()
    exams = _load_exams_meta()
    kept: list[dict] = []
    to_delete: list[str] = []
    for e in exams:
        if str(e.get("id") or "") == eid:
            to_delete.append(Path(str(e.get("filename") or "")).name)
            continue
        kept.append(e)
    for name in to_delete:
        fp = EXAMS_DATA_DIR / name
        if fp.exists():
            fp.unlink(missing_ok=True)
    _save_exams_meta(kept)
    return Response(status_code=303, headers={"Location": "/admin/exams"})


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
        f"<tr>"
        f"<td><code>{html.escape(f.name)}</code></td>"
        f"<td>{f.stat().st_size}</td>"
        f"<td>{datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}</td>"
        f"<td><a href='/admin/weekly-files/download?name={html.escape(f.name, quote=True)}'>下载</a></td>"
        f"</tr>"
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
        <thead><tr><th>文件名</th><th>大小(bytes)</th><th>更新时间</th><th>操作</th></tr></thead>
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
            f"<td><code>{html.escape(b.oa_openid or '')}</code></td>"
            f"<td><code>{html.escape(b.student_uid)}</code></td>"
            f"<td><code>{html.escape(b.external_userid or '')}</code></td>"
            f"<td>{html.escape(stu_map.get(b.student_uid, ''))}</td>"
            f"<td>{b.created_at}</td>"
            f"<td>"
            f"<form action='/admin/bindings/update' method='post' style='display:flex;gap:6px;align-items:center;flex-wrap:wrap'>"
            f"<input type='hidden' name='binding_id' value='{b.id}' />"
            f"<input name='student_uid' value='{html.escape(b.student_uid, quote=True)}' style='width:120px' />"
            f"<input name='oa_openid' value='{html.escape((b.oa_openid or ''), quote=True)}' style='width:180px' placeholder='服务号oa_openid' />"
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
    <p>显示最近 500 条绑定：小程序 openid、服务号 oa_openid 与 student_uid。</p>
    <div class="card">
      <table>
        <thead><tr><th>ID</th><th>openid(小程序)</th><th>oa_openid(服务号)</th><th>student_uid</th><th>external_userid</th><th>学生姓名</th><th>绑定时间</th><th>操作</th></tr></thead>
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
    oa_openid: str = Form(""),
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
    b.oa_openid = (oa_openid or "").strip()
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
    oid = (br.openid or "").strip()
    if oid.startswith("oa:"):
        oa = oid[3:]
        exb = (
            db.query(ParentStudentBinding)
            .filter(ParentStudentBinding.oa_openid == oa, ParentStudentBinding.student_uid == su)
            .one_or_none()
        )
        if not exb:
            db.add(ParentStudentBinding(openid="", oa_openid=oa, student_uid=su))
    else:
        exb = (
            db.query(ParentStudentBinding)
            .filter(ParentStudentBinding.openid == br.openid, ParentStudentBinding.student_uid == su)
            .one_or_none()
        )
        if not exb:
            db.add(ParentStudentBinding(openid=br.openid, student_uid=su))
    db.commit()
    return Response(status_code=303, headers={"Location": "/admin/binding-requests"})

