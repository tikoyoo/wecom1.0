from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import settings


@dataclass
class KfTokenCache:
    access_token: str = ""
    expires_at: float = 0.0

    def valid(self) -> bool:
        return self.access_token and time.time() < (self.expires_at - 60)


kf_token_cache = KfTokenCache()


async def get_kf_access_token() -> str:
    """
    WeChat Customer Service uses a dedicated Secret in WeCom.
    """
    if kf_token_cache.valid():
        return kf_token_cache.access_token

    if not settings.wecom_kf_secret:
        raise RuntimeError("WECOM_KF_SECRET not configured")

    url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params={"corpid": settings.wecom_corp_id, "corpsecret": settings.wecom_kf_secret})
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"gettoken failed: {data}")
        kf_token_cache.access_token = data["access_token"]
        kf_token_cache.expires_at = time.time() + int(data.get("expires_in", 7200))
        return kf_token_cache.access_token


async def kf_sync_msg(token: str, cursor: str = "", limit: int = 50) -> dict[str, Any]:
    """
    Pull messages after receiving a kf_msg_or_event callback.
    """
    access_token = await get_kf_access_token()
    url = "https://qyapi.weixin.qq.com/cgi-bin/kf/sync_msg"
    payload = {"token": token, "cursor": cursor, "limit": limit}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, params={"access_token": access_token}, json=payload)
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"sync_msg failed: {data}")
        return data


async def kf_send_text(open_kfid: str, external_userid: str, content: str) -> dict[str, Any]:
    access_token = await get_kf_access_token()
    url = "https://qyapi.weixin.qq.com/cgi-bin/kf/send_msg"
    payload = {
        "touser": external_userid,
        "open_kfid": open_kfid,
        "msgtype": "text",
        "text": {"content": content},
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, params={"access_token": access_token}, json=payload)
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"send_kf failed: {data}")
        return data

