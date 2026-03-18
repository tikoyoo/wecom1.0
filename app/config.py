from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # WeCom
    wecom_corp_id: str = ""
    wecom_corp_secret: str = ""
    wecom_agent_id: int = 0
    wecom_token: str = ""
    wecom_encoding_aes_key: str = ""

    # WeCom WeChat Customer Service (微信客服)
    # Uses a separate secret and a separate callback token/aes key
    wecom_kf_secret: str = ""
    wecom_kf_token: str = ""
    wecom_kf_encoding_aes_key: str = ""

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

    # Hydro (via SSH)
    hydro_ssh_host: str = ""
    hydro_ssh_user: str = ""
    hydro_ssh_key_path: str = ""

    # WeCom Customer Contact (externalcontact) for outbound reports
    # The follow-up member userid used as sender in add_msg_template
    wecom_external_sender_id: str = ""


settings = Settings()

