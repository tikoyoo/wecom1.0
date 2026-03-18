from __future__ import annotations

from typing import Any

import httpx

from .config import settings
from .wecom_api import get_access_token


async def list_external_contacts(follow_userid: str) -> list[str]:
    access_token = await get_access_token()
    url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/list"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params={"access_token": access_token, "userid": follow_userid})
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"externalcontact list failed: {data}")
        return list(data.get("external_userid") or [])


async def get_external_contact(external_userid: str) -> dict[str, Any]:
    access_token = await get_access_token()
    url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/get"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params={"access_token": access_token, "external_userid": external_userid})
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"externalcontact get failed: {data}")
        return data


async def add_msg_template_single(external_userid: str, content: str, sender_userid: str) -> dict[str, Any]:
    """
    Create a single chat mass message task for one external contact.
    Depending on WeCom policy, it may be sent immediately or require confirmation.
    """
    access_token = await get_access_token()
    url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/add_msg_template"
    payload: dict[str, Any] = {
        "chat_type": "single",
        "external_userid": [external_userid],
        "sender": sender_userid,
        "text": {"content": content},
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, params={"access_token": access_token}, json=payload)
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"add_msg_template failed: {data}")
        return data

