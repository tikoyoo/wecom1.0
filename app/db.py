from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import settings


Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
db_path = Path(settings.data_dir) / "wecom_bot.sqlite3"

engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    wecom_user_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="user")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(32))  # user/assistant/system
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    user: Mapped["User"] = relationship(back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(256))
    filename: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    idx: Mapped[int] = mapped_column(Integer)  # chunk index within document
    content: Mapped[str] = mapped_column(Text)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class StudentRecord(Base):
    """学生名录（姓名匹配用）；可从 Hydro 同步或后台手工维护。"""

    __tablename__ = "student_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_uid: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(256))
    name_key: Mapped[str] = mapped_column(String(256), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ParentStudentBinding(Base):
    """小程序 openid 与 Hydro/业务 student_uid 的绑定（已通过匹配或审核）。"""

    __tablename__ = "parent_student_bindings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    openid: Mapped[str] = mapped_column(String(128), index=True)
    # 微信服务号网页授权得到的 openid，与小程序 openid 不同，需单独绑定或后台填写
    oa_openid: Mapped[str] = mapped_column(String(128), default="", index=True)
    student_uid: Mapped[str] = mapped_column(String(128), index=True)
    external_userid: Mapped[str] = mapped_column(String(128), default="", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BindingNameRequest(Base):
    """姓名无法唯一匹配时的待审核记录。"""

    __tablename__ = "binding_name_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    openid: Mapped[str] = mapped_column(String(128), index=True)
    student_name_submitted: Mapped[str] = mapped_column(String(256))
    candidates_json: Mapped[str] = mapped_column(Text, default="[]")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    resolved_student_uid: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class HydroCache(Base):
    """Hydro 周数据缓存（按 key 存储，当前使用 __ALL__）。"""

    __tablename__ = "hydro_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_uid: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    payload_json: Mapped[str] = mapped_column(Text, default="[]")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StudentWeeklyMetric(Base):
    """学生每周5维数据（从 Hydro 拉取并落库）。"""

    __tablename__ = "student_weekly_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    week_key: Mapped[str] = mapped_column(String(32), index=True)  # 例如 2026-W12
    student_uid: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    rank: Mapped[int] = mapped_column(Integer, default=999)
    groups_json: Mapped[str] = mapped_column(Text, default="[]")
    hw_title: Mapped[str] = mapped_column(Text, default="")
    hw_done: Mapped[int] = mapped_column(Integer, default=0)
    hw_total: Mapped[int] = mapped_column(Integer, default=0)
    week_submits: Mapped[int] = mapped_column(Integer, default=0)
    week_ac: Mapped[int] = mapped_column(Integer, default=0)
    active_days: Mapped[int] = mapped_column(Integer, default=0)
    last_active: Mapped[str] = mapped_column(String(64), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ExternalSendLog(Base):
    """周报发送日志（外部联系人消息模板发送结果）。"""

    __tablename__ = "external_send_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    week_key: Mapped[str] = mapped_column(String(32), default="", index=True)
    sender_userid: Mapped[str] = mapped_column(String(128), default="")
    group_filter: Mapped[str] = mapped_column(String(128), default="")
    only_unfinished: Mapped[int] = mapped_column(Integer, default=0)
    student_uid: Mapped[str] = mapped_column(String(128), default="", index=True)
    external_userid: Mapped[str] = mapped_column(String(128), default="", index=True)
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok/fail/skip
    msgid: Mapped[str] = mapped_column(String(128), default="")
    response_json: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")


class ExternalContact(Base):
    """企业微信外部联系人快照（用于 external_userid 自动匹配 student_uid）。"""

    __tablename__ = "external_contacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_userid: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    follow_userid: Mapped[str] = mapped_column(String(128), default="", index=True)
    remark: Mapped[str] = mapped_column(String(256), default="")
    student_uid_hint: Mapped[str] = mapped_column(String(128), default="", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _sqlite_migrate_compat()


def _sqlite_migrate_compat() -> None:
    """兼容旧库：补齐新增列，避免线上升级时报 no such column / not null 错误。"""
    with engine.begin() as conn:
        # parent_student_bindings 可能是旧结构（含 parent_id，缺 openid/external_userid）
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(parent_student_bindings)")).fetchall()]
        if "openid" not in cols:
            conn.execute(text("ALTER TABLE parent_student_bindings ADD COLUMN openid VARCHAR(128)"))
        if "student_uid" not in cols:
            conn.execute(text("ALTER TABLE parent_student_bindings ADD COLUMN student_uid VARCHAR(128)"))
        if "external_userid" not in cols:
            conn.execute(text("ALTER TABLE parent_student_bindings ADD COLUMN external_userid VARCHAR(128) DEFAULT ''"))
        if "oa_openid" not in cols:
            conn.execute(text("ALTER TABLE parent_student_bindings ADD COLUMN oa_openid VARCHAR(128) DEFAULT ''"))
        if "created_at" not in cols:
            conn.execute(text("ALTER TABLE parent_student_bindings ADD COLUMN created_at DATETIME"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

