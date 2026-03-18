from __future__ import annotations

import argparse
import asyncio

from .db import get_db, init_db
from .reports_service import send_weekly_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Send weekly reports via externalcontact.")
    parser.add_argument("--sender", default="", help="WeCom follow-up member userid (default from env).")
    parser.add_argument("--week", default="", help="Week key like 2026-W12 (default latest).")
    parser.add_argument("--group", default="", help="Group/class filter (default all).")
    parser.add_argument("--only-unfinished", action="store_true", help="Only students with unfinished homework.")
    parser.add_argument("--no-refresh", action="store_true", help="Do not refresh hydro, use cached metrics.")
    args = parser.parse_args()

    init_db()
    with next(get_db()) as db:  # type: ignore[arg-type]
        res = asyncio.run(
            send_weekly_reports(
                db,
                sender=args.sender,
                week_key=args.week,
                group=args.group,
                only_unfinished=args.only_unfinished,
                force_refresh=not args.no_refresh,
            )
        )
    print(f"OK: week={res.week_key} sender={res.sender} ok={res.ok} fail={res.fail}")


if __name__ == "__main__":
    main()

