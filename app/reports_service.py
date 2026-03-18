from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from .config import settings
from .db import ExternalSendLog, ParentStudentBinding, StudentWeeklyMetric
from .hydro_service import get_weekly_students
from .wecom_external_api import add_msg_template_single


def render_weekly_report(raw: dict) -> str:
    hw = raw.get("hw_info") or {}
    return (
        f"📊 {raw.get('name','')} 的学习周报\n\n"
        f"👤 UID：{raw.get('uid')}\n"
        f"📈 当前排名：第 {raw.get('rank','-')} 名\n\n"
        f"📚 本周作业：{hw.get('title','无')}\n"
        f"✅ 完成情况：{hw.get('done',0)}/{hw.get('total',0)}\n\n"
        f"💡 本周AC：{(raw.get('dim2') or {}).get('ac',0)} 题\n"
        f"📝 本周提交：{(raw.get('dim2') or {}).get('submits',0)} 次\n"
        f"🔥 活跃天数：{(raw.get('dim3') or {}).get('days',0)} 天（最近：{(raw.get('dim3') or {}).get('last','无')}）\n\n"
        f"有问题随时联系我。"
    )


@dataclass(frozen=True)
class WeeklySendResult:
    ok: int
    fail: int
    week_key: str
    sender: str


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
    sender: str = "",
    week_key: str = "",
    group: str = "",
    only_unfinished: bool = False,
    force_refresh: bool = True,
) -> WeeklySendResult:
    sender = (sender or "").strip() or (settings.wecom_external_sender_id or "").strip()
    if not sender:
        raise RuntimeError("sender required (set WECOM_EXTERNAL_SENDER_ID)")

    # refresh from hydro and persist weekly metrics
    weekly_raw = get_weekly_students(db, force_refresh=force_refresh)
    by_uid_raw = {str(s.get("uid")): s for s in weekly_raw if s.get("uid") is not None}

    if not week_key:
        week_key = _latest_week_key(db)

    bindings = db.query(ParentStudentBinding).all()
    ok = 0
    fail = 0
    for b in bindings:
        m = (
            db.query(StudentWeeklyMetric)
            .filter(StudentWeeklyMetric.week_key == week_key)
            .filter(StudentWeeklyMetric.student_uid == b.student_uid)
            .one_or_none()
        )
        if m is None or not _passes_filters(m, group, only_unfinished):
            continue

        raw = by_uid_raw.get(b.student_uid)
        if not raw:
            continue

        content = render_weekly_report(raw)
        try:
            resp = await add_msg_template_single(external_userid=b.parent.external_userid, content=content, sender_userid=sender)
            db.add(
                ExternalSendLog(
                    created_at=datetime.utcnow(),
                    week_key=week_key,
                    sender_userid=sender,
                    group_filter=group,
                    only_unfinished=1 if only_unfinished else 0,
                    student_uid=b.student_uid,
                    external_userid=b.parent.external_userid,
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
                    external_userid=b.parent.external_userid,
                    status="fail",
                    msgid="",
                    response_json="",
                    error=str(e),
                )
            )
            db.commit()
            fail += 1

    return WeeklySendResult(ok=ok, fail=fail, week_key=week_key, sender=sender)

