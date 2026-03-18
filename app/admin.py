from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .config import settings

security = HTTPBasic()


def require_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    ok = credentials.username == settings.admin_username and credentials.password == settings.admin_password
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})
    return credentials.username


def html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 14px; }}
    input, textarea {{ width: 100%; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; }}
    button {{ padding: 10px 14px; border-radius: 8px; border: 1px solid #111827; background: #111827; color: white; cursor: pointer; }}
    button.secondary {{ background: white; color: #111827; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""

