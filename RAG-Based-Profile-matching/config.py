from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


DEFAULT_SKILL_VOCABULARY: List[str] = [
    "Python",
    "Java",
    "C++",
    "C#",
    "Go",
    "JavaScript",
    "TypeScript",
    "SQL",
    "PostgreSQL",
    "MySQL",
    "MongoDB",
    "Redis",
    "AWS",
    "Azure",
    "GCP",
    "Docker",
    "Kubernetes",
    "Terraform",
    "Git",
    "Linux",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "PyTorch",
    "TensorFlow",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "FastAPI",
    "Django",
    "Flask",
    "Spring Boot",
    "React",
    "Node.js",
    "Spark",
    "Hadoop",
    "Airflow",
    "Power BI",
]


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION", "resumes")
    chunk_size_words: int = int(os.getenv("CHUNK_SIZE_WORDS", "220"))
    chunk_overlap_words: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))
    default_top_k: int = int(os.getenv("TOP_K", "10"))
    initial_candidate_chunks: int = int(os.getenv("INITIAL_CANDIDATE_CHUNKS", "60"))
    skill_vocabulary: List[str] = field(default_factory=lambda: DEFAULT_SKILL_VOCABULARY)
    llm_provider: str = os.getenv("LLM_PROVIDER", "none")
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b")
    local_llm_base_url: str = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
    local_llm_timeout_seconds: int = int(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "40"))
    llm_reasoning_candidates: int = int(os.getenv("LLM_REASONING_CANDIDATES", "5"))
    hf_local_model: str = os.getenv("HF_LOCAL_MODEL", "google/flan-t5-small")
    hf_local_max_new_tokens: int = int(os.getenv("HF_LOCAL_MAX_NEW_TOKENS", "120"))


def load_settings() -> Settings:
    return Settings()
