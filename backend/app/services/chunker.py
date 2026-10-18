"""
GraphRAG Backend - Text Chunker
Splits documents into overlapping semantic chunks for embedding and graph construction.
Uses token-aware splitting to respect LLM context limits.
"""

import re
import uuid
from typing import Dict, Any, List
from dataclasses import dataclass, field
from loguru import logger

from app.config import settings


@dataclass
class TextChunk:
    """Represents a single text chunk with metadata."""
    id: str
    text: str
    document_id: str
    document_name: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """
    Semantic text chunker with overlapping windows.
    Respects sentence/paragraph boundaries where possible.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk_document(
        self,
        text: str,
        document_id: str,
        document_name: str,
        extra_metadata: Dict[str, Any] = None,
        strategy: str = "semantic",
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> List[TextChunk]:
        """
        Split a document into overlapping text chunks using user-selected strategy:
        - 'semantic': Paragraph and sentence boundary aware window
        - 'recursive': Hierarchical markdown/section split
        - 'fixed': Exact token/word window sliding
        """
        if not text or not text.strip():
            logger.warning(f"Empty text for document {document_name}")
            return []

        active_chunk_size = chunk_size or self.chunk_size
        active_chunk_overlap = chunk_overlap or self.chunk_overlap
        text = self._clean_text(text)

        if strategy == "fixed":
            chunks = self._chunk_fixed(text, document_id, document_name, extra_metadata or {}, active_chunk_size, active_chunk_overlap)
        elif strategy == "recursive":
            chunks = self._chunk_recursive(text, document_id, document_name, extra_metadata or {}, active_chunk_size, active_chunk_overlap)
        else:
            # Default 'semantic'
            paragraphs = self._split_paragraphs(text, active_chunk_size)
            chunks = self._build_chunks(paragraphs, document_id, document_name, extra_metadata or {}, active_chunk_size, active_chunk_overlap)

        logger.info(f"Chunked '{document_name}' into {len(chunks)} chunks (strategy='{strategy}', size={active_chunk_size}, overlap={active_chunk_overlap})")
        return chunks

    def _chunk_fixed(
        self, text: str, document_id: str, document_name: str,
        extra_metadata: Dict[str, Any], chunk_size: int, chunk_overlap: int
    ) -> List[TextChunk]:
        words = text.split()
        chunks = []
        step = max(1, chunk_size - chunk_overlap)
        char_pos = 0
        idx = 0
        for i in range(0, len(words), step):
            slice_words = words[i:i + chunk_size]
            if not slice_words:
                break
            chunk_str = " ".join(slice_words)
            chunks.append(self._make_chunk([chunk_str], document_id, document_name, idx, char_pos, extra_metadata))
            char_pos += len(chunk_str)
            idx += 1
            if i + chunk_size >= len(words):
                break
        return chunks

    def _chunk_recursive(
        self, text: str, document_id: str, document_name: str,
        extra_metadata: Dict[str, Any], chunk_size: int, chunk_overlap: int
    ) -> List[TextChunk]:
        # Split on markdown headings first, then double newlines, then single newlines
        sections = re.split(r"(?m)(?=^#{1,4}\s+)", text)
        sub_sections = []
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            if self._estimate_tokens(sec) > chunk_size:
                sub_sections.extend(re.split(r"\n{2,}", sec))
            else:
                sub_sections.append(sec)
        return self._build_chunks(sub_sections, document_id, document_name, extra_metadata, chunk_size, chunk_overlap)

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and clean up artifacts."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse excessive whitespace (but keep paragraph breaks)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return text.strip()

    def _split_paragraphs(self, text: str, chunk_size: int = None) -> List[str]:
        """Split text into paragraph-level units, preserving tabular rows intact."""
        target_size = chunk_size or self.chunk_size
        paragraphs = re.split(r"\n{2,}", text)
        result = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Tabular data protection: never slice rows at decimals or emails
            if " | " in para and (":" in para or "Row " in para):
                sub_lines = [l.strip() for l in para.split("\n") if l.strip()]
                result.extend(sub_lines)
            elif self._estimate_tokens(para) > target_size * 1.5:
                sentences = self._split_sentences(para)
                result.extend(sentences)
            else:
                result.append(para)
        return result

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _build_chunks(
        self,
        paragraphs: List[str],
        document_id: str,
        document_name: str,
        extra_metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> List[TextChunk]:
        """
        Assemble paragraphs into chunks with sliding window overlap.
        """
        active_chunk_size = chunk_size or self.chunk_size
        active_chunk_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap

        chunks = []
        current_texts = []
        current_tokens = 0
        char_position = 0
        chunk_index = 0

        # Keep track of overlap buffer (last N tokens)
        overlap_buffer = []
        overlap_tokens = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)

            # If single paragraph exceeds chunk_size, hard-split it
            if para_tokens > active_chunk_size:
                # Save current chunk first
                if current_texts:
                    chunk = self._make_chunk(
                        current_texts, document_id, document_name,
                        chunk_index, char_position, extra_metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    char_position += len(" ".join(current_texts))
                    overlap_buffer, overlap_tokens = self._compute_overlap(current_texts, active_chunk_overlap)
                    current_texts = list(overlap_buffer)
                    current_tokens = overlap_tokens

                # Hard split the big paragraph
                words = para.split()
                word_chunk = []
                word_tokens = 0
                for word in words:
                    wt = self._estimate_tokens(word)
                    if word_tokens + wt > active_chunk_size and word_chunk:
                        sub_para = " ".join(word_chunk)
                        current_texts.append(sub_para)
                        chunk = self._make_chunk(
                            current_texts, document_id, document_name,
                            chunk_index, char_position, extra_metadata
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        char_position += len(sub_para)
                        overlap_buffer, overlap_tokens = self._compute_overlap([sub_para], active_chunk_overlap)
                        current_texts = list(overlap_buffer)
                        current_tokens = overlap_tokens
                        word_chunk = []
                        word_tokens = 0
                    word_chunk.append(word)
                    word_tokens += wt
                if word_chunk:
                    current_texts.append(" ".join(word_chunk))
                    current_tokens += word_tokens
                continue

            # Normal case: check if para fits in current chunk
            if current_tokens + para_tokens > active_chunk_size and current_texts:
                # Finalize current chunk
                chunk = self._make_chunk(
                    current_texts, document_id, document_name,
                    chunk_index, char_position, extra_metadata
                )
                chunks.append(chunk)
                chunk_index += 1
                char_position += len(" ".join(current_texts))

                # Start next chunk with overlap
                overlap_buffer, overlap_tokens = self._compute_overlap(current_texts, active_chunk_overlap)
                current_texts = list(overlap_buffer)
                current_tokens = overlap_tokens

            current_texts.append(para)
            current_tokens += para_tokens

        # Final chunk
        if current_texts:
            chunk = self._make_chunk(
                current_texts, document_id, document_name,
                chunk_index, char_position, extra_metadata
            )
            chunks.append(chunk)

        return chunks

    def _make_chunk(
        self,
        texts: List[str],
        document_id: str,
        document_name: str,
        index: int,
        char_start: int,
        extra_metadata: Dict[str, Any],
    ) -> TextChunk:
        text = "\n\n".join(texts)
        return TextChunk(
            id=str(uuid.uuid4()),
            text=text,
            document_id=document_id,
            document_name=document_name,
            chunk_index=index,
            char_start=char_start,
            char_end=char_start + len(text),
            token_count=self._estimate_tokens(text),
            metadata={**extra_metadata, "chunk_index": index},
        )

    def _compute_overlap(self, texts: List[str], chunk_overlap: int = None) -> tuple:
        """Extract the last N tokens worth of text for overlap."""
        target_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap
        if not texts or target_overlap <= 0:
            return [], 0
        all_text = " ".join(texts)
        words = all_text.split()
        overlap_words = words[-target_overlap:] if len(words) > target_overlap else words
        overlap_text = " ".join(overlap_words)
        return [overlap_text], self._estimate_tokens(overlap_text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Fast approximation: ~4 chars per token (standard GPT approximation).
        Using character-based estimation to avoid tiktoken overhead.
        """
        return max(1, len(text) // 4)


# Singleton
text_chunker = TextChunker()
