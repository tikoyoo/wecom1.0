from __future__ import annotations

from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from .config import settings
from .db import Chunk, Document


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def tokenize_zh(text: str) -> list[str]:
    return [t.strip() for t in jieba.lcut(text) if t.strip()]


@dataclass
class RagIndex:
    chunk_ids: list[int]
    chunk_texts: list[str]
    tokenized: list[list[str]]
    bm25: BM25Okapi

    @classmethod
    def from_db(cls, db: Session) -> "RagIndex":
        rows = db.query(Chunk).order_by(Chunk.id.asc()).all()
        chunk_ids = [c.id for c in rows]
        chunk_texts = [c.content for c in rows]
        tokenized = [tokenize_zh(t) for t in chunk_texts]
        bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([["空"]])
        return cls(chunk_ids=chunk_ids, chunk_texts=chunk_texts, tokenized=tokenized, bm25=bm25)

    def search(self, query: str, top_k: int) -> list[tuple[int, str, float]]:
        if not self.chunk_ids:
            return []
        q = tokenize_zh(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunk_ids[i], self.chunk_texts[i], float(scores[i])) for i in ranked]


def add_document_with_chunks(db: Session, title: str, filename: str, content: str) -> int:
    doc = Document(title=title, filename=filename)
    db.add(doc)
    db.flush()

    chunks = chunk_text(content, settings.rag_chunk_size, settings.rag_chunk_overlap)
    for idx, c in enumerate(chunks):
        db.add(Chunk(document_id=doc.id, idx=idx, content=c))
    db.commit()
    return doc.id

