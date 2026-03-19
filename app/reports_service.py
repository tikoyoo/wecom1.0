from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from string import Formatter

from sqlalchemy.orm import Session

from .config import settings
from .db import ExternalSendLog, ParentStudentBinding, StudentWeeklyMetric
from .hydro_service import get_weekly_students
from .wecom_external_api import add_msg_template_single


def _safe_format(template: str, ctx: dict[str, object]) -> str:
    # Ignore unknown placeholders to avoid send failures from template typos.
    fields = {name for _, name, _, _ in Formatter().parse(template) if name}
    for k in fields:
        if k not in ctx:
            ctx[k] = ""
    return template.format(**ctx)


def render_weekly_report(raw: dict, template_text: str = "") -> str:
    hw = raw.get("hw_info") or {}
    dim2 = raw.get("dim2") or {}
    dim3 = raw.get("dim3") or {}
    ctx: dict[str, object] = {
        "name": raw.get("name", ""),
        "uid": raw.get("uid", ""),
        "rank": raw.get("rank", "-"),
        "hw_title": hw.get("title", "无"),
        "hw_done": hw.get("done", 0),
        "hw_total": hw.get("total", 0),
        "week_ac": dim2.get("ac", 0),
        "week_submits": dim2.get("submits", 0),
        "active_days": dim3.get("days", 0),
        "last_active": dim3.get("last", "无"),
        "groups": "、".join([str(x) for x in (raw.get("groups") or []) if str(x).strip()]),
    }
    if template_text.strip():
        return _safe_format(template_text, ctx)
    return (
        f"📊 {ctx['name']} 的学习周报\n\n"
        f"👤 HYDRO ID：{ctx['uid']}\n"
        f"📈 当前排名：第 {ctx['rank']} 名\n\n"
        f"📚 本周作业：{ctx['hw_title']}\n"
        f"✅ 完成情况：{ctx['hw_done']}/{ctx['hw_total']}\n\n"
        f"💡 本周AC：{ctx['week_ac']} 题\n"
        f"📝 本周提交：{ctx['week_submits']} 次\n"
        f"🔥 活跃天数：{ctx['active_days']} 天（最近：{ctx['last_active']}）\n\n"
        "如需辅导建议，请直接回复本消息。"
    )


@dataclass(frozen=True)
class WeeklySendResult:
    ok: int
    fail: int
    skip: int
    week_key: str
    sender: str
    group: str


def _latest_week_key(db: Session) -> str:
    latest = db.query(StudentWeeklyMetric.week_key).order_by(StudentWeeklyMetric.week_key.desc()).first()
    return latest[0] if latest else ""


def _passes_filters(m: StudentWeeklyMetric, group: str, only_unfinished: bool) -> bool:
    if group:
        try:
            gs = json.loads(m.groups_json or "[]")
            if group not in gs:
                return False
        except Exception:
            return False
    if only_unfinished and (m.hw_total <= 0 or m.hw_done >= m.hw_total):
        return False
    return True


async def send_weekly_reports(
    db: Session,
    *,
    sender: str = "",
    week_key: str = "",
    group: str = "",
    only_unfinished: bool = False,
    force_refresh: bool = True,
    template_text: str = "",
) -> WeeklySendResult:
    sender = (sender or "").strip() or (settings.wecom_external_sender_id or "").strip()
    if not sender:
        raise RuntimeError("sender required (set WECOM_EXTERNAL_SENDER_ID)")

    weekly_raw = get_weekly_students(db, force_refresh=force_refresh)
    by_uid_raw = {str(s.get("uid")): s for s in weekly_raw if s.get("uid") is not None}

    if not week_key:
        week_key = _latest_week_key(db)
    if not week_key:
        raise RuntimeError("no week data")

    bindings = db.query(ParentStudentBinding).all()
    ok = 0
    fail = 0
    skip = 0
    for b in bindings:
        ext = (b.external_userid or "").strip()
        if not ext:
            skip += 1
            db.add(
                ExternalSendLog(
                    created_at=datetime.utcnow(),
                    week_key=week_key,
                    sender_userid=sender,
                    group_filter=group,
                    only_unfinished=1 if only_unfinished else 0,
                    student_uid=b.student_uid,
                    external_userid=ext,
                    status="skip",
                    msgid="",
                    response_json="",
                    error="binding has no external_userid",
                )
            )
            db.commit()
            continue

        m = (
            db.query(StudentWeeklyMetric)
            .filter(StudentWeeklyMetric.week_key == week_key)
            .filter(StudentWeeklyMetric.student_uid == b.student_uid)
            .one_or_none()
        )
        if m is None or not _passes_filters(m, group, only_unfinished):
            skip += 1
            continue

        raw = by_uid_raw.get(b.student_uid)
        if not raw:
            skip += 1
            continue

        content = render_weekly_report(raw, template_text=template_text)
        try:
            resp = await add_msg_template_single(external_userid=ext, content=content, sender_userid=sender)
            db.add(
                ExternalSendLog(
                    created_at=datetime.utcnow(),
                    week_key=week_key,
                    sender_userid=sender,
                    group_filter=group,
                    only_unfinished=1 if only_unfinished else 0,
                    student_uid=b.student_uid,
                    external_userid=ext,
                    status="ok",
                    msgid=str(resp.get("msgid") or resp.get("msg_id") or ""),
                    response_json=json.dumps(resp, ensure_ascii=False),
                    error="",
                )
            )
            db.commit()
            ok += 1
        except Exception as e:
            db.add(
                ExternalSendLog(
                    created_at=datetime.utcnow(),
                    week_key=week_key,
                    sender_userid=sender,
                    group_filter=group,
                    only_unfinished=1 if only_unfinished else 0,
                    student_uid=b.student_uid,
                    external_userid=ext,
                    status="fail",
                    msgid="",
                    response_json="",
                    error=str(e),
                )
            )
            db.commit()
            fail += 1

    return WeeklySendResult(ok=ok, fail=fail, skip=skip, week_key=week_key, sender=sender, group=group)
