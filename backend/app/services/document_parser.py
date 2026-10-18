"""
GraphRAG Backend - Document Parsers
Handles parsing of PDF, DOCX, TXT, Markdown, CSV, JSON, and URLs.
Returns structured text content with metadata.
"""

import csv
import io
import json
import re
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger


class DocumentParser:
    """
    Unified document parser supporting multiple file formats.
    All parsers return (text: str, metadata: dict).
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown",
        ".csv", ".tsv", ".json", ".html", ".htm",
        ".xlsx", ".xls", ".xlsm",
    }

    @staticmethod
    def get_extension(filename: str) -> str:
        return Path(filename).suffix.lower()

    def parse(self, file_path: str, filename: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a document file and return (text, metadata).
        Dispatches to the appropriate parser based on file extension.
        """
        ext = self.get_extension(filename)
        logger.info(f"Parsing document: {filename} (type: {ext})")

        parsers = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".doc": self._parse_docx,
            ".txt": self._parse_txt,
            ".md": self._parse_markdown,
            ".markdown": self._parse_markdown,
            ".csv": self._parse_csv,
            ".tsv": self._parse_csv,
            ".json": self._parse_json,
            ".html": self._parse_html,
            ".htm": self._parse_html,
            ".xlsx": self._parse_excel,
            ".xls": self._parse_excel,
            ".xlsm": self._parse_excel,
        }

        parser_fn = parsers.get(ext, self._parse_txt)
        return parser_fn(file_path, filename)

    def _parse_pdf(self, file_path: str, filename: str) -> Tuple[str, Dict]:
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)

            full_text = "\n\n".join(pages)
            metadata = {
                "type": "pdf",
                "pages": len(reader.pages),
                "filename": filename,
                "char_count": len(full_text),
            }
            return full_text, metadata
        except Exception as e:
            logger.error(f"PDF parse error for {filename}: {e}")
            return "", {"type": "pdf", "error": str(e), "filename": filename}

    def _parse_docx(self, file_path: str, filename: str) -> Tuple[str, Dict]:
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            # Include table content
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)

            full_text = "\n\n".join(paragraphs)
            metadata = {
                "type": "docx",
                "filename": filename,
                "paragraphs": len(paragraphs),
                "char_count": len(full_text),
            }
            return full_text, metadata
        except Exception as e:
            logger.error(f"DOCX parse error for {filename}: {e}")
            return "", {"type": "docx", "error": str(e), "filename": filename}

    def _parse_txt(self, file_path: str, filename: str) -> Tuple[str, Dict]:
        try:
            import chardet
            with open(file_path, "rb") as f:
                raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding") or "utf-8"
            text = raw.decode(encoding, errors="replace")
            metadata = {
                "type": "txt",
                "filename": filename,
                "encoding": encoding,
                "char_count": len(text),
            }
            return text, metadata
        except Exception as e:
            logger.error(f"TXT parse error for {filename}: {e}")
            return "", {"type": "txt", "error": str(e), "filename": filename}

    def _parse_markdown(self, file_path: str, filename: str) -> Tuple[str, Dict]:
        try:
            import chardet
            with open(file_path, "rb") as f:
                raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding") or "utf-8"
            md_text = raw.decode(encoding, errors="replace")

            # Strip markdown syntax for cleaner text
            # Remove headers but keep content
            text = re.sub(r"#{1,6}\s+", "", md_text)
            # Remove bold/italic
            text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
            text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
            # Remove links but keep text
            text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
            # Remove code blocks (keep content)
            text = re.sub(r"```[\w]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
            # Remove inline code
            text = re.sub(r"`([^`]+)`", r"\1", text)

            metadata = {
                "type": "markdown",
                "filename": filename,
                "raw_length": len(md_text),
                "char_count": len(text),
            }
            return text, metadata
        except Exception as e:
            logger.error(f"Markdown parse error for {filename}: {e}")
            return "", {"type": "markdown", "error": str(e), "filename": filename}

    def _parse_csv(self, file_path: str, filename: str) -> Tuple[str, Dict]:
        try:
            import chardet
            with open(file_path, "rb") as f:
                raw = f.read()
            detected = chardet.detect(raw)
            encoding = detected.get("encoding") or "utf-8"
            text_io = io.StringIO(raw.decode(encoding, errors="replace"))

            reader = csv.DictReader(text_io)
            rows = list(reader)
            headers = reader.fieldnames or []

            # Convert CSV to readable text format
            lines = [f"CSV Data: {filename}"]
            lines.append(f"Columns: {', '.join(str(h) for h in headers)}")
            lines.append("")
            for i, row in enumerate(rows[:500]):  # Cap at 500 rows for context
                row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                lines.append(f"Row {i+1}: {row_text}")

            full_text = "\n".join(lines)
            metadata = {
                "type": "csv",
                "filename": filename,
                "columns": headers,
                "rows": len(rows),
                "char_count": len(full_text),
            }
            return full_text, metadata
        except Exception as e:
            logger.error(f"CSV parse error for {filename}: {e}")
            return "", {"type": "csv", "error": str(e), "filename": filename}

    def _parse_excel(self, file_path: str, filename: str) -> Tuple[str, Dict]:
        """
        Parse Excel spreadsheets (.xlsx, .xls, .xlsm).
        Iterates across all sheets and extracts rows as structured key-value statements.
        """
        try:
            import pandas as pd
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names

            lines = [f"Excel Spreadsheet: {filename}"]
            total_rows = 0

            for sheet in sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet)
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue

                lines.append(f"\n--- Sheet: {sheet} ({len(df)} rows, {len(df.columns)} columns) ---")
                headers = [str(c).strip() for c in df.columns]
                lines.append(f"Columns: {', '.join(headers)}")

                for i, row in df.iterrows():
                    row_parts = []
                    for col in df.columns:
                        val = row[col]
                        if pd.notna(val) and str(val).strip():
                            val_str = str(val).strip()
                            # Clean float representation like 1200.0 if integer
                            if isinstance(val, float) and val.is_integer():
                                val_str = str(int(val))
                            row_parts.append(f"{col}: {val_str}")
                    
                    if not row_parts:
                        continue

                    # Filter out aggregate summary/total rows (e.g., TOTAL / AVG)
                    first_val = row_parts[0].lower()
                    if "total" in first_val or "avg" in first_val or "average" in first_val:
                        continue

                    total_rows += 1
                    lines.append(f"Row {total_rows}: " + " | ".join(row_parts))

            full_text = "\n\n".join(lines)
            metadata = {
                "type": "excel",
                "filename": filename,
                "sheets": sheet_names,
                "total_rows": total_rows,
                "char_count": len(full_text),
            }
            logger.info(f"Parsed Excel document '{filename}': {len(sheet_names)} sheet(s), {total_rows} row(s)")
            return full_text, metadata
        except Exception as e:
            logger.error(f"Excel parse error for {filename}: {e}")
            return "", {"type": "excel", "error": str(e), "filename": filename}

    def _parse_json(self, file_path: str, filename: str) -> Tuple[str, Dict]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)

            # Convert JSON to readable text
            if isinstance(data, list):
                lines = [f"JSON Array Data: {filename} ({len(data)} items)"]
                for i, item in enumerate(data[:200]):
                    if isinstance(item, dict):
                        item_text = " | ".join(f"{k}: {v}" for k, v in item.items())
                        lines.append(f"Item {i+1}: {item_text}")
                    else:
                        lines.append(f"Item {i+1}: {str(item)}")
            elif isinstance(data, dict):
                lines = [f"JSON Object Data: {filename}"]
                lines.append(self._flatten_dict(data))
            else:
                lines = [str(data)]

            full_text = "\n".join(lines)
            metadata = {
                "type": "json",
                "filename": filename,
                "char_count": len(full_text),
                "structure": "array" if isinstance(data, list) else "object",
            }
            return full_text, metadata
        except Exception as e:
            logger.error(f"JSON parse error for {filename}: {e}")
            return "", {"type": "json", "error": str(e), "filename": filename}

    def _parse_html(self, file_path: str, filename: str) -> Tuple[str, Dict]:
        try:
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                html = f.read()

            soup = BeautifulSoup(html, "html.parser")
            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n")
            # Clean up whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

            metadata = {
                "type": "html",
                "filename": filename,
                "title": soup.title.string if soup.title else "",
                "char_count": len(text),
            }
            return text, metadata
        except Exception as e:
            logger.error(f"HTML parse error for {filename}: {e}")
            return "", {"type": "html", "error": str(e), "filename": filename}

    @staticmethod
    def _flatten_dict(d: dict, prefix: str = "", sep: str = ".") -> str:
        """Flatten nested dict to readable key-value pairs."""
        lines = []
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                lines.append(DocumentParser._flatten_dict(v, key, sep))
            elif isinstance(v, list):
                lines.append(f"{key}: [{', '.join(str(i) for i in v[:10])}]")
            else:
                lines.append(f"{key}: {v}")
        return "\n".join(lines)


# Singleton
document_parser = DocumentParser()
