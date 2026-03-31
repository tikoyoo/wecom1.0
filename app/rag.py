from __future__ import annotations

import re
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from .config import settings
from .db import Chunk, Document


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """原始硬切分块，作为 chunk_text_smart 的兜底。"""
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


def chunk_text_smart(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    语义感知分块：优先按段落（双换行）合并，段落过长再硬切。
    保证 FAQ 问答对、段落语义不被切断。
    """
    text = (text or "").strip()
    if not text:
        return []

    # 先按双换行切段落
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # 当前 buf + 新段落 在 chunk_size 内，合并
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # 段落本身超长则硬切
            if len(para) > chunk_size:
                chunks.extend(chunk_text(para, chunk_size, overlap))
                buf = ""
            else:
                buf = para

    if buf:
        chunks.append(buf)

    return chunks if chunks else chunk_text(text, chunk_size, overlap)


def tokenize_zh(text: str) -> list[str]:
    return [t.strip() for t in jieba.lcut(text) if t.strip()]


def _keyword_tokens(text: str) -> list[str]:
    toks = tokenize_zh(text)
    return [t for t in toks if len(t) >= 2]


def _expand_query_tokens(query: str) -> list[str]:
    """
    查询扩展：在 jieba 分词基础上，对 2-4 字词保留原词，
    同时保留单字高频词（数字/字母），提升召回率。
    """
    base = tokenize_zh(query)
    expanded: list[str] = []
    seen: set[str] = set()
    for t in base:
        if t not in seen:
            expanded.append(t)
            seen.add(t)
        # 对较长词拆出子词，覆盖部分同义/缩写写法
        if len(t) >= 4:
            sub = t[:len(t) // 2 * 2]  # 取前半部分
            if len(sub) >= 2 and sub not in seen:
                expanded.append(sub)
                seen.add(sub)
    return expanded


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

    def search(self, query: str, top_k: int, min_score: float = 0.0) -> list[tuple[int, str, float]]:
        if not self.chunk_ids:
            return []

        # 查询扩展后检索
        q = _expand_query_tokens(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # 应用相关性阈值过滤
        threshold = max(min_score, 0.0)
        picked = [i for i in ranked if float(scores[i]) >= threshold][:top_k]
        if picked:
            return [(self.chunk_ids[i], self.chunk_texts[i], float(scores[i])) for i in picked]

        # fallback: 阈值过滤后无结果，降级到关键词包含匹配
        kws = _keyword_tokens(query)
        if not kws:
            return []
        contains_rank: list[tuple[int, int]] = []
        for i, txt in enumerate(self.chunk_texts):
            hit = sum(1 for k in kws if k in txt)
            if hit > 0:
                contains_rank.append((i, hit))
        contains_rank.sort(key=lambda x: x[1], reverse=True)
        out_idx = [i for i, _ in contains_rank[:top_k]]
        return [(self.chunk_ids[i], self.chunk_texts[i], 0.0) for i in out_idx]


def add_document_with_chunks(db: Session, title: str, filename: str, content: str) -> int:
    doc = Document(title=title, filename=filename)
    db.add(doc)
    db.flush()

    # 使用语义感知分块替代硬切
    chunks = chunk_text_smart(content, settings.rag_chunk_size, settings.rag_chunk_overlap)
    for idx, c in enumerate(chunks):
        db.add(Chunk(document_id=doc.id, idx=idx, content=c))
    db.commit()
    return doc.id

