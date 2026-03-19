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


async def list_external_userids(userid: str) -> list[str]:
    """拉取某跟进成员名下的外部联系人ID列表。"""
    access_token = await get_access_token()
    url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/list"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params={"access_token": access_token, "userid": userid})
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"externalcontact/list failed: {data}")
        return [str(x) for x in (data.get("external_userid") or []) if x]


async def get_external_contact(external_userid: str) -> dict:
    """获取外部联系人详情（含跟进备注等）。"""
    access_token = await get_access_token()
    url = "https://qyapi.weixin.qq.com/cgi-bin/externalcontact/get"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params={"access_token": access_token, "external_userid": external_userid})
        r.raise_for_status()
        data = r.json()
        if data.get("errcode") != 0:
            raise RuntimeError(f"externalcontact/get failed: {data}")
        return data
