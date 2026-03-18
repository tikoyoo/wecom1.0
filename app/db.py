from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine
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


class ParentContact(Base):
    """
    External parent contact in WeCom Customer Contact (externalcontact).
    """

    __tablename__ = "parent_contacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_userid: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    remark: Mapped[str] = mapped_column(String(256), default="")
    follow_userid: Mapped[str] = mapped_column(String(128), default="", index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    bindings: Mapped[list["ParentStudentBinding"]] = relationship(back_populates="parent", cascade="all, delete-orphan")


class ParentStudentBinding(Base):
    """
    Manual binding between parent (external_userid) and Hydro student uid.
    Supports multiple parents per student.
    """

    __tablename__ = "parent_student_bindings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("parent_contacts.id"), index=True)
    student_uid: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    parent: Mapped["ParentContact"] = relationship(back_populates="bindings")


class HydroCache(Base):
    """
    Cached Hydro weekly stats per student uid to enforce rate limits.
    """

    __tablename__ = "hydro_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_uid: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    payload_json: Mapped[str] = mapped_column(Text, default="")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

