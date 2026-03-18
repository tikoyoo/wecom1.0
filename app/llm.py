from __future__ import annotations

from typing import Any

import httpx

from .config import settings


async def deepseek_chat(messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    if not settings.deepseek_api_key:
        return "（未配置 DEEPSEEK_API_KEY，当前仅回声测试）"

    url = settings.deepseek_base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": settings.deepseek_chat_model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {settings.deepseek_api_key}"}
    async with httpx.AsyncClient(timeout=40) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

