from __future__ import annotations

import httpx

from .wecom_api import get_access_token


async def add_msg_template_single(external_userid: str, content: str, sender_userid: str) -> dict:
    """
    企业微信外部联系人群发（单个客户）。
    https://developer.work.weixin.qq.com/document/path/92135
    """
    access_token = await get_access_token()
    url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/add_msg_template"
    payload = {
        "chat_type": "single",
        "external_userid": [external_userid],
        "sender": sender_userid,
        "text": {"content": content},
        "allow_select": False,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, params={"access_token": access_token}, json=payload)
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"add_msg_template failed: {data}")
        return data
