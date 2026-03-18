from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from .config import settings
from .db import HydroCache, StudentWeeklyMetric


@dataclass(frozen=True)
class HydroStudentWeekly:
    uid: str
    name: str
    rank: int
    hw_title: str
    hw_done: int
    hw_total: int
    week_submits: int
    week_ac: int
    active_days: int
    last_active_date: str


HYDRO_WEEKLY_JS = r"""
try {
  const domain = 'system';
  const now = new Date();
  const weekAgo = new Date(); weekAgo.setDate(weekAgo.getDate() - 7);
  const weekObjId = ObjectId(Math.floor(weekAgo/1000).toString(16) + "0000000000000000");

  const users = db['domain.user'].find({ domainId: domain, nAccept: { $gt: 0 } }).toArray();

  const activeHwDocs = db.document.find({
      domainId: domain,
      docType: 30,
      rule: 'homework',
      endAt: { $gte: now }
  }).toArray();

  const results = users.map(u => {
    const uid = u.uid;
    const userGroups = db['user.group'].find({ domainId: domain, uids: uid }).toArray().map(g => g.name);
    const userHwDocs = activeHwDocs.filter(hw => hw.assign && hw.assign.some(groupName => userGroups.includes(groupName)));

    let allHwPids = [];
    let hwTitle = "无作业";
    if (userHwDocs.length > 0) {
      hwTitle = userHwDocs.map(hw => hw.title).join('、');
      userHwDocs.forEach(hw => {
        hw.pids.forEach(pid => {
          const pidStr = pid.toString();
          if (!allHwPids.includes(pidStr)) allHwPids.push(pidStr);
        });
      });
    }

    const weekAcPids = db.record.distinct("pid", {
      uid: uid,
      domainId: domain,
      status: 1,
      _id: { $gt: weekObjId }
    }).map(p => p.toString());

    const weekRecords = db.record.find({
      uid: uid,
      domainId: domain,
      _id: { $gt: weekObjId }
    }).sort({ _id: -1 }).toArray();

    const hwDone = allHwPids.filter(pid => weekAcPids.includes(pid)).length;
    const hwTotal = allHwPids.length;

    const activeDates = [...new Set(weekRecords.map(r =>
      new Date(r._id.getTimestamp()).toLocaleDateString('zh-CN')
    ))].sort().reverse();

    return {
      uid: uid,
      name: u.displayName || "未知",
      rank: u.rank || 999,
      groups: userGroups,
      hw_info: { title: hwTitle, done: hwDone, total: hwTotal },
      dim2: { submits: weekRecords.length, ac: weekAcPids.length },
      dim3: { days: activeDates.length, last: activeDates[0] || "无" }
    };
  });

  print("---JSON_START---");
  print(JSON.stringify(results));
  print("---JSON_END---");
} catch(e) {
  print("ERROR: " + e.message);
}
"""


def _run_remote_hydro_db(js: str) -> list[dict]:
    if not (settings.hydro_ssh_host and settings.hydro_ssh_user and settings.hydro_ssh_key_path):
        raise RuntimeError("Hydro SSH config missing (HYDRO_SSH_HOST/HYDRO_SSH_USER/HYDRO_SSH_KEY_PATH)")

    remote = (
        "sudo -i << 'EOF'\n"
        "hydrooj db << 'QUERY'\n"
        f"{js}\n"
        "QUERY\n"
        "EOF\n"
    )

    cmd = [
        "ssh",
        "-i",
        settings.hydro_ssh_key_path,
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{settings.hydro_ssh_user}@{settings.hydro_ssh_host}",
        remote,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(f"Hydro SSH failed (code={r.returncode}): {out[:800]}")

    if "---JSON_START---" not in out:
        raise RuntimeError(f"Hydro output missing JSON markers: {out[:800]}")

    blob = out.split("---JSON_START---", 1)[1].split("---JSON_END---", 1)[0].strip()
    return json.loads(blob)


def get_weekly_students(db: Session, force_refresh: bool = False) -> list[dict]:
    """
    Returns raw list of student weekly dicts from Hydro.
    Cached for up to 1 hour globally (simple first version).
    """
    now = datetime.now(timezone.utc)
    cache = db.query(HydroCache).filter(HydroCache.student_uid == "__ALL__").one_or_none()
    if (not force_refresh) and cache and cache.payload_json and cache.fetched_at and (now - cache.fetched_at.replace(tzinfo=timezone.utc)) < timedelta(hours=1):
        return json.loads(cache.payload_json)

    data = _run_remote_hydro_db(HYDRO_WEEKLY_JS)
    payload = json.dumps(data, ensure_ascii=False)
    if cache is None:
        cache = HydroCache(student_uid="__ALL__", payload_json=payload, fetched_at=now.replace(tzinfo=None))
        db.add(cache)
    else:
        cache.payload_json = payload
        cache.fetched_at = now.replace(tzinfo=None)
    db.commit()

    # persist into weekly metrics table
    iso = now.isocalendar()
    week_key = f"{iso.year}-W{iso.week:02d}"
    for s in data:
        uid = str(s.get("uid") or "").strip()
        if not uid:
            continue
        hw = s.get("hw_info") or {}
        dim2 = s.get("dim2") or {}
        dim3 = s.get("dim3") or {}
        groups = s.get("groups") or []
        row = (
            db.query(StudentWeeklyMetric)
            .filter(StudentWeeklyMetric.week_key == week_key)
            .filter(StudentWeeklyMetric.student_uid == uid)
            .one_or_none()
        )
        if row is None:
            row = StudentWeeklyMetric(week_key=week_key, student_uid=uid)
            db.add(row)
        row.name = str(s.get("name") or "")
        try:
            row.rank = int(s.get("rank") or 999)
        except Exception:
            row.rank = 999
        row.groups_json = json.dumps(groups, ensure_ascii=False)
        row.hw_title = str(hw.get("title") or "")
        row.hw_done = int(hw.get("done") or 0)
        row.hw_total = int(hw.get("total") or 0)
        row.week_submits = int(dim2.get("submits") or 0)
        row.week_ac = int(dim2.get("ac") or 0)
        row.active_days = int(dim3.get("days") or 0)
        row.last_active = str(dim3.get("last") or "")
        row.updated_at = datetime.utcnow()
    db.commit()
    return data

