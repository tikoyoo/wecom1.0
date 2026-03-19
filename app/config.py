from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 无论 uvicorn 工作目录在哪，都从「项目根目录」加载 .env（与 app/ 同级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.is_file() else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # WeCom
    wecom_corp_id: str = ""
    wecom_corp_secret: str = ""
    wecom_agent_id: int = 0
    wecom_token: str = ""
    wecom_encoding_aes_key: str = ""

    # DeepSeek (OpenAI-compatible)
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_api_key: str = ""
    deepseek_chat_model: str = "deepseek-chat"

    # RAG
    rag_top_k: int = 4
    rag_chunk_size: int = 900
    rag_chunk_overlap: int = 120

    # Memory
    memory_max_turns: int = 8

    # Admin
    admin_username: str = "admin"
    admin_password: str = "admin"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Storage
    data_dir: str = "./data"

    # 微信小程序（code2Session，用于小程序 openid）
    wx_mini_appid: str = ""
    wx_mini_secret: str = ""
    
    # Hydro 周数据拉取（用于家长姓名匹配基础数据）
    hydro_ssh_host: str = ""
    hydro_ssh_user: str = ""
    hydro_ssh_key_path: str = ""
    wecom_external_sender_id: str = ""

    @field_validator(
        "wx_mini_appid",
        "wx_mini_secret",
        "hydro_ssh_host",
        "hydro_ssh_user",
        "hydro_ssh_key_path",
        "wecom_external_sender_id",
        mode="before",
    )
    @classmethod
    def _strip_mini_secrets(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v).strip()


settings = Settings()

