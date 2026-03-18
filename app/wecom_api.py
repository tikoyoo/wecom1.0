from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from .config import settings


@dataclass
class TokenCache:
    access_token: str = ""
    expires_at: float = 0.0

    def valid(self) -> bool:
        return self.access_token and time.time() < (self.expires_at - 60)


token_cache = TokenCache()


async def get_access_token() -> str:
    if token_cache.valid():
        return token_cache.access_token

    url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params={"corpid": settings.wecom_corp_id, "corpsecret": settings.wecom_corp_secret})
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"gettoken failed: {data}")
        token_cache.access_token = data["access_token"]
        token_cache.expires_at = time.time() + int(data.get("expires_in", 7200))
        return token_cache.access_token


async def send_text(touser: str, content: str) -> dict:
    access_token = await get_access_token()
    url = "https://qyapi.weixin.qq.com/cgi-bin/message/send"
    payload = {
        "touser": touser,
        "msgtype": "text",
        "agentid": settings.wecom_agent_id,
        "text": {"content": content},
        "safe": 0,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, params={"access_token": access_token}, json=payload)
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"send failed: {data}")
        return data

