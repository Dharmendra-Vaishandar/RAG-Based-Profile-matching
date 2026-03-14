from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import Settings, load_settings


LOGGER = logging.getLogger("resume_rag")


SECTION_ALIASES: Dict[str, List[str]] = {
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "career history",
    ],
    "education": [
        "education",
        "academic background",
        "academics",
        "qualifications",
    ],
    "skills": ["skills", "technical skills", "core skills", "competencies"],
    "projects": ["projects", "project experience", "key projects"],
    "certifications": ["certifications", "licenses", "certificates"],
    "summary": ["summary", "profile", "professional summary", "objective"],
}


DEGREE_PATTERN = re.compile(
    r"\b(bachelor|master|ph\.?d|b\.?tech|m\.?tech|b\.?sc|m\.?sc|mba|bca|mca)\b",
    re.IGNORECASE,
)


@dataclass
class ResumeMetadata:
    candidate_name: str
    resume_path: str
    skills: List[str]
    years_of_experience: float
    education: List[str]


class ResumeRAGIngestor:
    """
    Production-oriented ingestion pipeline for resumes.

    Pipeline stages:
    1) Load and parse PDF/DOCX files
    2) Split into logical sections (Education, Experience, Skills, Projects, etc.)
    3) Chunk section text with overlap
    4) Extract metadata fields for filtering
    5) Embed chunks and upsert to ChromaDB with metadata
    """

    def __init__(self, settings: Optional[Settings] = None, rebuild: bool = False) -> None:
        self.settings = settings or load_settings()
        self.embedder = SentenceTransformer(self.settings.embedding_model)
        self.client = chromadb.PersistentClient(path=self.settings.chroma_db_path)
        self.collection = self._get_or_create_collection(rebuild=rebuild)

    def _get_or_create_collection(self, rebuild: bool = False):
        if rebuild:
            try:
                self.client.delete_collection(self.settings.chroma_collection_name)
                LOGGER.info("Deleted existing collection '%s'.", self.settings.chroma_collection_name)
            except Exception:
                pass

        try:
            return self.client.get_collection(self.settings.chroma_collection_name)
        except Exception:
            return self.client.create_collection(
                name=self.settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    @staticmethod
    def load_resume_paths(resumes_dir: Path) -> List[Path]:
        supported_extensions = {".pdf", ".docx"}
        if not resumes_dir.exists():
            return []

        files: List[Path] = []
        for path in resumes_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in supported_extensions:
                files.append(path)
        return sorted(files)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[\t\r]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ ]{2,}", " ", text)
        return text.strip()

    def parse_resume_text(self, resume_path: Path) -> str:
        suffix = resume_path.suffix.lower()
        if suffix == ".pdf":
            text = self._parse_pdf(resume_path)
        elif suffix == ".docx":
            text = self._parse_docx(resume_path)
        else:
            raise ValueError(f"Unsupported file type: {resume_path}")
        return self._normalize_whitespace(text)

    @staticmethod
    def _parse_pdf(path: Path) -> str:
        reader = PdfReader(str(path))
        page_texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(page_texts)

    @staticmethod
    def _parse_docx(path: Path) -> str:
        document = Document(str(path))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        return "\n".join(paragraphs)

    @staticmethod
    def _clean_heading(line: str) -> str:
        return re.sub(r"[^a-zA-Z ]", "", line).strip().lower()

    def _detect_section_heading(self, line: str) -> Optional[str]:
        cleaned = self._clean_heading(line)
        if not cleaned:
            return None
        if len(cleaned.split()) > 5:
            return None

        for canonical, aliases in SECTION_ALIASES.items():
            if cleaned in aliases:
                return canonical
            if any(cleaned.startswith(alias) for alias in aliases):
                return canonical
        return None

    def split_into_sections(self, text: str) -> Dict[str, str]:
        """
        Splits resume text into logical sections by heading detection.

        Why this strategy:
        - Resume relevance is usually section-dependent.
        - Preserving section boundaries helps retrieval return better excerpts.
        """

        sections: Dict[str, List[str]] = defaultdict(list)
        current_section = "general"

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            maybe_section = self._detect_section_heading(line)
            if maybe_section:
                current_section = maybe_section
                continue

            sections[current_section].append(line)

        if not sections:
            return {"general": text}
        return {section: "\n".join(lines).strip() for section, lines in sections.items() if lines}

    def chunk_section_text(self, section_text: str) -> List[str]:
        """
        Creates overlapping chunks by paragraph-aware accumulation.

        Why this strategy:
        - Paragraph boundaries preserve semantics better than blind token slicing.
        - Overlap protects context continuity around chunk borders.
        """

        paragraphs = [part.strip() for part in re.split(r"\n{2,}", section_text) if part.strip()]
        if not paragraphs:
            return []

        chunk_size = max(50, self.settings.chunk_size_words)
        overlap = max(0, min(self.settings.chunk_overlap_words, chunk_size // 2))

        chunks: List[str] = []
        current_words: List[str] = []

        def flush_chunk() -> None:
            if current_words:
                chunks.append(" ".join(current_words).strip())

        for paragraph in paragraphs:
            paragraph_words = paragraph.split()
            if not paragraph_words:
                continue

            if len(paragraph_words) >= chunk_size:
                if current_words:
                    flush_chunk()
                    current_words = []

                step = max(1, chunk_size - overlap)
                for start in range(0, len(paragraph_words), step):
                    piece = paragraph_words[start : start + chunk_size]
                    if piece:
                        chunks.append(" ".join(piece))
                continue

            if len(current_words) + len(paragraph_words) <= chunk_size:
                current_words.extend(paragraph_words)
            else:
                flush_chunk()
                current_words = current_words[-overlap:] if overlap and current_words else []
                current_words.extend(paragraph_words)

        flush_chunk()
        return [chunk for chunk in chunks if chunk]

    def extract_metadata(self, text: str, resume_path: Path) -> ResumeMetadata:
        """
        Extracts key metadata fields for downstream filtering.

        Metadata strategy:
        - candidate_name: best-effort from first informative line, fallback to filename
        - skills: dictionary-based extraction from configurable skill vocabulary
        - years_of_experience: max year mention / date-range inference
        - education: degree line extraction
        """

        candidate_name = self._extract_candidate_name(text, resume_path)
        skills = self._extract_skills(text)
        years_of_experience = self._extract_years_of_experience(text)
        education = self._extract_education(text)
        return ResumeMetadata(
            candidate_name=candidate_name,
            resume_path=self._to_display_path(resume_path),
            skills=skills,
            years_of_experience=years_of_experience,
            education=education,
        )

    @staticmethod
    def _to_display_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()).as_posix())
        except Exception:
            return path.as_posix()

    @staticmethod
    def _is_valid_name_candidate(line: str) -> bool:
        if any(token in line.lower() for token in ["resume", "curriculum", "email", "phone", "linkedin", "github"]):
            return False
        words = re.findall(r"[A-Za-z][A-Za-z'\-]+", line)
        return 1 < len(words) <= 5

    def _extract_candidate_name(self, text: str, resume_path: Path) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:8]:
            if self._is_valid_name_candidate(line):
                letters_only = re.sub(r"[^A-Za-z'\- ]", "", line).strip()
                if letters_only:
                    return " ".join(part.capitalize() for part in letters_only.split())

        return resume_path.stem.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _skill_pattern(skill: str) -> re.Pattern[str]:
        escaped = re.escape(skill)
        if any(ch in skill for ch in ["+", "#", "."]):
            return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)

    def _extract_skills(self, text: str) -> List[str]:
        found: List[str] = []
        for skill in self.settings.skill_vocabulary:
            if self._skill_pattern(skill).search(text):
                found.append(skill)
        return sorted(set(found), key=str.lower)

    @staticmethod
    def _extract_year_mentions(text: str) -> List[float]:
        matches = re.findall(r"(\d{1,2}(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", text, flags=re.IGNORECASE)
        return [float(match) for match in matches]

    @staticmethod
    def _extract_date_range_years(text: str) -> List[float]:
        current_year = datetime.now().year
        ranges = re.findall(
            r"(19\d{2}|20\d{2})\s*(?:-|–|to)\s*(present|current|19\d{2}|20\d{2})",
            text,
            flags=re.IGNORECASE,
        )
        values: List[float] = []
        for start, end in ranges:
            start_year = int(start)
            if end.lower() in {"present", "current"}:
                end_year = current_year
            else:
                end_year = int(end)

            if start_year <= end_year:
                span = end_year - start_year
                if 0 <= span <= 50:
                    values.append(float(span))
        return values

    def _extract_years_of_experience(self, text: str) -> float:
        mentions = self._extract_year_mentions(text)
        ranges = self._extract_date_range_years(text)
        candidates = mentions + ranges
        if not candidates:
            return 0.0
        return round(max(candidates), 1)

    def _extract_education(self, text: str) -> List[str]:
        matches: List[str] = []
        for line in [line.strip() for line in text.splitlines() if line.strip()]:
            if DEGREE_PATTERN.search(line):
                matches.append(line[:160])
        if not matches:
            return []

        deduped: List[str] = []
        seen = set()
        for item in matches:
            key = item.lower()
            if key not in seen:
                deduped.append(item)
                seen.add(key)
        return deduped[:5]

    @staticmethod
    def _build_chunk_id(resume_path: str, section: str, chunk_index: int, chunk_text: str) -> str:
        raw = f"{resume_path}|{section}|{chunk_index}|{chunk_text[:150]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _to_chroma_metadata(self, resume_meta: ResumeMetadata, section: str, chunk_index: int) -> Dict[str, Any]:
        return {
            "candidate_name": resume_meta.candidate_name,
            "resume_path": resume_meta.resume_path,
            "section": section,
            "chunk_index": int(chunk_index),
            "years_of_experience": float(resume_meta.years_of_experience),
            "skills": "|".join(resume_meta.skills),
            "education": "|".join(resume_meta.education),
        }

    def _upsert_resume(self, resume_path: Path, text: str) -> int:
        resume_meta = self.extract_metadata(text, resume_path)
        sections = self.split_into_sections(text)

        chunk_ids: List[str] = []
        chunk_texts: List[str] = []
        chunk_metadatas: List[Dict[str, Any]] = []

        for section, section_text in sections.items():
            for idx, chunk_text in enumerate(self.chunk_section_text(section_text)):
                chunk_id = self._build_chunk_id(resume_meta.resume_path, section, idx, chunk_text)
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk_text)
                chunk_metadatas.append(self._to_chroma_metadata(resume_meta, section, idx))

        if not chunk_texts:
            return 0

        try:
            self.collection.delete(where={"resume_path": resume_meta.resume_path})
        except Exception:
            pass

        embeddings = self.embedder.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        ).tolist()

        self.collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=chunk_metadatas,
            documents=chunk_texts,
        )
        return len(chunk_texts)

    def ingest_directory(self, resumes_dir: Path) -> Dict[str, Any]:
        files = self.load_resume_paths(resumes_dir)
        report = {
            "resumes_directory": resumes_dir.as_posix(),
            "supported_files_found": len(files),
            "processed_resumes": 0,
            "skipped_resumes": 0,
            "ingested_chunks": 0,
            "errors": [],
        }

        for resume_path in files:
            try:
                text = self.parse_resume_text(resume_path)
                if not text.strip():
                    report["skipped_resumes"] += 1
                    continue

                chunk_count = self._upsert_resume(resume_path, text)
                if chunk_count == 0:
                    report["skipped_resumes"] += 1
                    continue

                report["processed_resumes"] += 1
                report["ingested_chunks"] += chunk_count
                LOGGER.info("Ingested %s (%d chunks)", resume_path.name, chunk_count)
            except Exception as exc:
                report["skipped_resumes"] += 1
                report["errors"].append({"resume": resume_path.as_posix(), "error": str(exc)})
                LOGGER.exception("Failed to ingest %s", resume_path)

        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resume RAG ingestion pipeline")
    parser.add_argument("--resumes-dir", type=str, default="resumes", help="Directory containing PDF/DOCX resumes")
    parser.add_argument("--db-path", type=str, default=None, help="Path to Chroma persistent DB directory")
    parser.add_argument("--collection", type=str, default=None, help="Chroma collection name")
    parser.add_argument("--embedding-model", type=str, default=None, help="SentenceTransformer model name")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in words")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in words")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the collection before ingest")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser


def _merge_cli_settings(base: Settings, args: argparse.Namespace) -> Settings:
    return Settings(
        embedding_model=args.embedding_model or base.embedding_model,
        chroma_db_path=args.db_path or base.chroma_db_path,
        chroma_collection_name=args.collection or base.chroma_collection_name,
        chunk_size_words=args.chunk_size if args.chunk_size is not None else base.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap if args.chunk_overlap is not None else base.chunk_overlap_words,
        default_top_k=base.default_top_k,
        initial_candidate_chunks=base.initial_candidate_chunks,
        skill_vocabulary=base.skill_vocabulary,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    settings = _merge_cli_settings(load_settings(), args)
    ingestor = ResumeRAGIngestor(settings=settings, rebuild=args.rebuild)
    report = ingestor.ingest_directory(Path(args.resumes_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
