from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

import chromadb
from sentence_transformers import SentenceTransformer

from config import Settings, load_settings


LOGGER = logging.getLogger("job_matcher")


@dataclass
class JobRequirements:
    required_skills: List[str] = field(default_factory=list)
    min_years: float = 0.0


@dataclass
class ChunkHit:
    distance: float
    section: str
    text: str


@dataclass
class CandidateAggregate:
    candidate_name: str
    resume_path: str
    skills: Set[str] = field(default_factory=set)
    years_of_experience: float = 0.0
    education: List[str] = field(default_factory=list)
    chunk_hits: List[ChunkHit] = field(default_factory=list)


class OllamaReasoner:
    def __init__(self, model: str, base_url: str, timeout_seconds: int = 40) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate_reasoning(
        self,
        job_description: str,
        candidate_name: str,
        resume_path: str,
        match_score: int,
        matched_skills: List[str],
        relevant_excerpts: List[str],
        evidence: Dict[str, Any],
    ) -> Optional[str]:
        payload = {
            "job_description": job_description,
            "candidate": {
                "candidate_name": candidate_name,
                "resume_path": resume_path,
                "match_score": match_score,
                "matched_skills": matched_skills,
                "relevant_excerpts": relevant_excerpts,
            },
            "scoring_evidence": evidence,
        }

        system_prompt = (
            "You are a strict resume-matching assistant. "
            "Use only provided evidence. Do not invent facts. "
            "Return a single compact JSON object with key 'reasoning'."
        )
        user_prompt = (
            "Generate concise 1-3 sentence reasoning for why the candidate matches or misses the role. "
            "Mention strongest matched skills, experience fit, and relevant sections. "
            "Output format exactly: {\"reasoning\": \"...\"}.\n\n"
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        request_payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        endpoint = f"{self.base_url}/api/chat"
        req = urllib_request.Request(
            endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
            response_json = json.loads(raw)
            content = str(response_json.get("message", {}).get("content", "")).strip()
            return self._extract_reasoning_text(content)
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ollama reasoning failed: %s", exc)
            return None

    @staticmethod
    def _extract_reasoning_text(content: str) -> Optional[str]:
        if not content:
            return None

        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        def parse_reasoning(candidate: str) -> Optional[str]:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                return None
            reasoning = parsed.get("reasoning") if isinstance(parsed, dict) else None
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()
            return None

        parsed_reasoning = parse_reasoning(cleaned)
        if parsed_reasoning:
            return parsed_reasoning

        object_match = re.search(r"\{[\s\S]*\}", cleaned)
        if object_match:
            parsed_reasoning = parse_reasoning(object_match.group(0))
            if parsed_reasoning:
                return parsed_reasoning

        return cleaned[:600]


class HFLocalReasoner:
    def __init__(self, model: str, max_new_tokens: int = 120) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self._generator = None

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return
        from transformers import pipeline

        self._generator = pipeline(
            "text2text-generation",
            model=self.model,
        )

    def generate_reasoning(
        self,
        job_description: str,
        candidate_name: str,
        resume_path: str,
        match_score: int,
        matched_skills: List[str],
        relevant_excerpts: List[str],
        evidence: Dict[str, Any],
    ) -> Optional[str]:
        try:
            self._ensure_generator()
            if self._generator is None:
                return None

            prompt = (
                "You are a resume screening assistant. "
                "Write 1-3 concise sentences explaining candidate-job match using only facts provided. "
                "Mention matched skills, experience fit, and strongest evidence sections.\n\n"
                f"Job Description: {job_description}\n"
                f"Candidate Name: {candidate_name}\n"
                f"Resume Path: {resume_path}\n"
                f"Match Score: {match_score}\n"
                f"Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}\n"
                f"Evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
                f"Relevant Excerpts: {json.dumps(relevant_excerpts[:2], ensure_ascii=False)}\n"
                "Reasoning:"
            )

            outputs = self._generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
            )
            if not outputs:
                return None

            generated = str(outputs[0].get("generated_text", "")).strip()
            generated = self._normalize_reasoning(generated)
            return generated
        except Exception as exc:
            LOGGER.warning("HF local reasoning failed: %s", exc)
            return None

    @staticmethod
    def _normalize_reasoning(generated: str) -> Optional[str]:
        if not generated:
            return None

        text = re.sub(r"\s+", " ", generated).strip()
        text = re.sub(r"^reasoning\s*:\s*", "", text, flags=re.IGNORECASE).strip()

        if text.startswith("[") and text.endswith("]"):
            return None
        if text.startswith("{") and text.endswith("}"):
            return None
        if len(text.split()) < 8:
            return None

        return text[:600]


class JobMatcher:
    """
    Semantic + hybrid retrieval engine for ranking resumes against a job description.

    Hybrid retrieval strategy:
    - semantic relevance from vector similarity
    - lexical skill overlap from extracted skills
    - optional hard filters (required skills, minimum years)
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()
        self.embedder = SentenceTransformer(self.settings.embedding_model)
        self.client = chromadb.PersistentClient(path=self.settings.chroma_db_path)
        self.collection = self.client.get_collection(self.settings.chroma_collection_name)

    def match(
        self,
        job_description: str,
        top_k: Optional[int] = None,
        initial_candidates: Optional[int] = None,
        required_skills: Optional[List[str]] = None,
        min_years: Optional[float] = None,
        use_llm_reasoning: bool = False,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_reasoning_candidates: Optional[int] = None,
        hf_max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        top_k = top_k or self.settings.default_top_k
        initial_candidates = initial_candidates or self.settings.initial_candidate_chunks

        inferred_requirements, jd_skills = self._infer_requirements(job_description)
        explicit_required = self._dedupe_skills(required_skills or [])

        hard_required_skills = self._dedupe_skills(explicit_required + inferred_requirements.required_skills)
        hard_min_years = (
            float(min_years)
            if min_years is not None
            else float(inferred_requirements.min_years)
        )

        query_embedding = self._embed_text(job_description)
        query_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k * 6, initial_candidates),
            include=["metadatas", "documents", "distances"],
        )

        candidates = self._aggregate_candidates(query_result)
        scoring_skills = self._dedupe_skills(jd_skills or hard_required_skills)

        ranked: List[Dict[str, Any]] = []
        for candidate in candidates.values():
            if not self._passes_filters(candidate, hard_required_skills, hard_min_years):
                continue

            scored = self._score_candidate(
                candidate=candidate,
                scoring_skills=scoring_skills,
                hard_min_years=hard_min_years,
            )
            ranked.append(scored)

        ranked.sort(key=lambda item: item["match_score"], reverse=True)

        if use_llm_reasoning:
            self._apply_llm_reasoning(
                job_description=job_description,
                ranked=ranked,
                top_k=top_k,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_reasoning_candidates=llm_reasoning_candidates,
                hf_max_new_tokens=hf_max_new_tokens,
            )

        for candidate in ranked:
            candidate.pop("_llm_evidence", None)

        return {
            "job_description": job_description,
            "top_matches": ranked[:top_k],
        }

    def _apply_llm_reasoning(
        self,
        job_description: str,
        ranked: List[Dict[str, Any]],
        top_k: int,
        llm_provider: Optional[str],
        llm_model: Optional[str],
        llm_base_url: Optional[str],
        llm_reasoning_candidates: Optional[int],
        hf_max_new_tokens: Optional[int],
    ) -> None:
        candidate_limit = llm_reasoning_candidates or self.settings.llm_reasoning_candidates
        candidate_limit = max(1, min(candidate_limit, top_k))
        to_enrich = ranked[:candidate_limit]
        if not to_enrich:
            return

        provider = (llm_provider or self.settings.llm_provider or "ollama").strip().lower()
        if provider == "none":
            provider = "ollama"

        if provider == "hf-local":
            reasoner = HFLocalReasoner(
                model=llm_model or self.settings.hf_local_model,
                max_new_tokens=hf_max_new_tokens or self.settings.hf_local_max_new_tokens,
            )
        else:
            reasoner = OllamaReasoner(
                model=llm_model or self.settings.local_llm_model,
                base_url=llm_base_url or self.settings.local_llm_base_url,
                timeout_seconds=self.settings.local_llm_timeout_seconds,
            )

        for candidate in to_enrich:
            reasoning = reasoner.generate_reasoning(
                job_description=job_description,
                candidate_name=str(candidate.get("candidate_name", "")),
                resume_path=str(candidate.get("resume_path", "")),
                match_score=int(candidate.get("match_score", 0)),
                matched_skills=list(candidate.get("matched_skills", []) or []),
                relevant_excerpts=list(candidate.get("relevant_excerpts", []) or []),
                evidence=dict(candidate.get("_llm_evidence", {}) or {}),
            )
            if reasoning:
                candidate["reasoning"] = reasoning

    def _embed_text(self, text: str) -> List[float]:
        return self.embedder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

    def _aggregate_candidates(self, query_result: Dict[str, Any]) -> Dict[str, CandidateAggregate]:
        metadatas = (query_result.get("metadatas") or [[]])[0]
        documents = (query_result.get("documents") or [[]])[0]
        distances = (query_result.get("distances") or [[]])[0]

        candidates: Dict[str, CandidateAggregate] = {}
        for metadata, document, distance in zip(metadatas, documents, distances):
            if not metadata:
                continue

            resume_path = str(metadata.get("resume_path", "")).strip()
            if not resume_path:
                continue

            candidate_name = str(metadata.get("candidate_name", "Unknown Candidate"))
            section = str(metadata.get("section", "general"))

            entry = candidates.get(resume_path)
            if not entry:
                entry = CandidateAggregate(
                    candidate_name=candidate_name,
                    resume_path=resume_path,
                    skills=set(self._split_pipe_values(metadata.get("skills"))),
                    years_of_experience=float(metadata.get("years_of_experience", 0.0) or 0.0),
                    education=self._split_pipe_values(metadata.get("education")),
                )
                candidates[resume_path] = entry

            entry.skills.update(self._split_pipe_values(metadata.get("skills")))
            entry.years_of_experience = max(
                entry.years_of_experience,
                float(metadata.get("years_of_experience", 0.0) or 0.0),
            )
            entry.chunk_hits.append(
                ChunkHit(
                    distance=float(distance or 0.0),
                    section=section,
                    text=(document or "").strip(),
                )
            )

        return candidates

    @staticmethod
    def _split_pipe_values(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        if not text:
            return []
        return [part.strip() for part in text.split("|") if part.strip()]

    def _passes_filters(
        self,
        candidate: CandidateAggregate,
        hard_required_skills: List[str],
        hard_min_years: float,
    ) -> bool:
        if hard_min_years > 0 and candidate.years_of_experience < hard_min_years:
            return False

        if hard_required_skills:
            candidate_skill_keys = {skill.lower() for skill in candidate.skills}
            required_keys = {skill.lower() for skill in hard_required_skills}
            if not required_keys.issubset(candidate_skill_keys):
                return False

        return True

    def _score_candidate(
        self,
        candidate: CandidateAggregate,
        scoring_skills: List[str],
        hard_min_years: float,
    ) -> Dict[str, Any]:
        semantic_similarity = self._semantic_score(candidate.chunk_hits)
        matched_skills = self._matched_skills(candidate.skills, scoring_skills)
        skill_overlap = self._skill_overlap_score(candidate.skills, scoring_skills)
        experience_score = self._experience_score(candidate.years_of_experience, hard_min_years)

        weighted_score = (
            0.55 * semantic_similarity
            + 0.30 * skill_overlap
            + 0.15 * experience_score
        )
        match_score = int(round(max(0.0, min(1.0, weighted_score)) * 100))

        excerpts = self._relevant_excerpts(candidate.chunk_hits, max_items=3)
        top_sections = self._top_sections(candidate.chunk_hits)
        reasoning = self._build_reasoning(
            matched_skills=matched_skills,
            candidate_years=candidate.years_of_experience,
            required_years=hard_min_years,
            semantic_similarity=semantic_similarity,
            top_sections=top_sections,
        )

        return {
            "candidate_name": candidate.candidate_name,
            "resume_path": candidate.resume_path,
            "match_score": match_score,
            "matched_skills": matched_skills,
            "relevant_excerpts": excerpts,
            "reasoning": reasoning,
            "_llm_evidence": {
                "semantic_similarity": round(semantic_similarity, 4),
                "skill_overlap": round(skill_overlap, 4),
                "experience_score": round(experience_score, 4),
                "candidate_years_of_experience": round(candidate.years_of_experience, 2),
                "required_years_of_experience": round(hard_min_years, 2),
                "top_sections": top_sections,
            },
        }

    @staticmethod
    def _semantic_score(chunk_hits: List[ChunkHit]) -> float:
        if not chunk_hits:
            return 0.0
        similarities = sorted((1.0 / (1.0 + hit.distance) for hit in chunk_hits), reverse=True)
        best = similarities[0]
        top_three = similarities[:3]
        avg_top = sum(top_three) / len(top_three)
        return max(0.0, min(1.0, 0.7 * best + 0.3 * avg_top))

    @staticmethod
    def _experience_score(candidate_years: float, required_years: float) -> float:
        if required_years <= 0:
            return min(candidate_years / 8.0, 1.0) if candidate_years > 0 else 0.5
        return max(0.0, min(candidate_years / required_years, 1.0))

    @staticmethod
    def _relevant_excerpts(chunk_hits: List[ChunkHit], max_items: int = 3) -> List[str]:
        ordered = sorted(chunk_hits, key=lambda item: item.distance)
        excerpts: List[str] = []
        seen = set()
        for hit in ordered:
            text = re.sub(r"\s+", " ", hit.text).strip()
            if not text:
                continue
            excerpt = text[:220] + ("..." if len(text) > 220 else "")
            if excerpt in seen:
                continue
            seen.add(excerpt)
            excerpts.append(excerpt)
            if len(excerpts) >= max_items:
                break
        return excerpts

    @staticmethod
    def _skill_overlap_score(candidate_skills: Set[str], target_skills: List[str]) -> float:
        if not target_skills:
            return 0.5
        candidate_keys = {skill.lower() for skill in candidate_skills}
        target_keys = {skill.lower() for skill in target_skills}
        if not target_keys:
            return 0.5
        overlap = len(candidate_keys.intersection(target_keys))
        return overlap / len(target_keys)

    @staticmethod
    def _matched_skills(candidate_skills: Set[str], target_skills: List[str]) -> List[str]:
        target_map = {skill.lower(): skill for skill in target_skills}
        matches = [target_map[skill.lower()] for skill in candidate_skills if skill.lower() in target_map]
        return sorted(set(matches), key=str.lower)

    def _build_reasoning(
        self,
        matched_skills: List[str],
        candidate_years: float,
        required_years: float,
        semantic_similarity: float,
        top_sections: List[str],
    ) -> str:
        reasons: List[str] = []
        if matched_skills:
            reasons.append(f"Matched critical skills: {', '.join(matched_skills[:6])}.")
        if required_years > 0:
            reasons.append(
                f"Experience fit: {candidate_years:.1f} years against {required_years:.1f} years required."
            )
        reasons.append(f"Semantic alignment: {semantic_similarity * 100:.1f}%.")
        if top_sections:
            reasons.append(f"Most relevant resume sections: {', '.join(top_sections)}.")
        return " ".join(reasons)

    @staticmethod
    def _top_sections(chunk_hits: List[ChunkHit], max_sections: int = 3) -> List[str]:
        sections: List[str] = []
        for hit in sorted(chunk_hits, key=lambda item: item.distance):
            if hit.section not in sections:
                sections.append(hit.section)
            if len(sections) >= max_sections:
                break
        return sections

    def _infer_requirements(self, job_description: str) -> Tuple[JobRequirements, List[str]]:
        all_skills = self._extract_skills(job_description)
        critical_skills = self._extract_critical_skills(job_description)
        min_years, skills_from_year_patterns = self._extract_year_requirements(job_description)

        required_skills = self._dedupe_skills(critical_skills + skills_from_year_patterns)
        requirements = JobRequirements(required_skills=required_skills, min_years=min_years)
        return requirements, all_skills

    def _extract_skills(self, text: str) -> List[str]:
        found: List[str] = []
        for skill in self.settings.skill_vocabulary:
            pattern = self._skill_pattern(skill)
            if pattern.search(text):
                found.append(skill)
        return self._dedupe_skills(found)

    def _extract_critical_skills(self, job_description: str) -> List[str]:
        sentences = re.split(r"[\n\.;]", job_description)
        critical_markers = re.compile(r"\b(must|required|mandatory|essential|need to have)\b", re.IGNORECASE)
        critical_skills: List[str] = []

        for sentence in sentences:
            if critical_markers.search(sentence):
                critical_skills.extend(self._extract_skills(sentence))

        return self._dedupe_skills(critical_skills)

    def _extract_year_requirements(self, text: str) -> Tuple[float, List[str]]:
        pattern = re.compile(
            r"(\d{1,2}(?:\.\d+)?)\s*\+?\s*years?(?:\s+of\s+experience)?(?:\s+in|\s+with)?\s*([A-Za-z0-9#\+\./\- ]{0,40})?",
            re.IGNORECASE,
        )
        years: List[float] = []
        skills: List[str] = []
        for match in pattern.finditer(text):
            years.append(float(match.group(1)))
            suffix = (match.group(2) or "").strip()
            if suffix:
                skills.extend(self._extract_skills(suffix))

        return (max(years) if years else 0.0, self._dedupe_skills(skills))

    @staticmethod
    def _skill_pattern(skill: str) -> re.Pattern[str]:
        escaped = re.escape(skill)
        if any(ch in skill for ch in ["+", "#", "."]):
            return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)

    def _dedupe_skills(self, skills: Iterable[str]) -> List[str]:
        canonical = {skill.lower(): skill for skill in self.settings.skill_vocabulary}
        deduped: Dict[str, str] = {}

        for skill in skills:
            normalized = str(skill).strip()
            if not normalized:
                continue
            key = normalized.lower()
            deduped[key] = canonical.get(key, normalized)

        return sorted(deduped.values(), key=str.lower)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic + hybrid job matcher for resume RAG")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive CLI mode for repeated job matching queries",
    )
    parser.add_argument("--job-description", type=str, default=None, help="Job description text")
    parser.add_argument("--job-file", type=str, default=None, help="Path to text file containing job description")
    parser.add_argument("--top-k", type=int, default=None, help="Number of top candidate resumes to return")
    parser.add_argument(
        "--initial-candidates",
        type=int,
        default=None,
        help="Number of top semantic chunk hits used before candidate-level aggregation",
    )
    parser.add_argument(
        "--required-skills",
        type=str,
        default=None,
        help="Comma-separated hard-required skills, e.g. 'Python,Machine Learning'",
    )
    parser.add_argument("--min-years", type=float, default=None, help="Hard minimum years of experience")
    parser.add_argument("--db-path", type=str, default=None, help="Path to Chroma persistent DB directory")
    parser.add_argument("--collection", type=str, default=None, help="Chroma collection name")
    parser.add_argument("--embedding-model", type=str, default=None, help="SentenceTransformer model name")
    parser.add_argument(
        "--use-llm-reasoning",
        action="store_true",
        help="Use local LLM to generate richer reasoning for top matches",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        choices=["ollama", "hf-local"],
        help="LLM provider: ollama (local server) or hf-local (in-process model)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Model name for selected provider, e.g. llama3.2:3b or google/flan-t5-small",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="Local Ollama base URL, e.g. http://localhost:11434",
    )
    parser.add_argument(
        "--llm-reasoning-candidates",
        type=int,
        default=None,
        help="How many top matches to enrich with LLM reasoning (defaults to env/config)",
    )
    parser.add_argument(
        "--hf-max-new-tokens",
        type=int,
        default=None,
        help="Max generated tokens when using hf-local provider",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser


def _read_job_description(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.job_description and args.job_description.strip():
        return args.job_description.strip()
    if args.job_file:
        path = Path(args.job_file)
        if not path.exists() or not path.is_file():
            parser.error(f"Job file does not exist: {args.job_file}")
        return path.read_text(encoding="utf-8").strip()
    parser.error("Provide either --job-description or --job-file")
    return ""


def _parse_required_skills(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _prompt_required_skills(default_skills: List[str]) -> List[str]:
    seed = ", ".join(default_skills)
    prompt = "Required skills (comma-separated, Enter to keep defaults): "
    if seed:
        prompt = f"Required skills (comma-separated, Enter for [{seed}]): "

    raw = input(prompt).strip()
    if not raw:
        return default_skills
    return [item.strip() for item in raw.split(",") if item.strip()]


def _prompt_min_years(default_min_years: Optional[float]) -> Optional[float]:
    default_text = ""
    if default_min_years is not None:
        default_text = f" [{default_min_years}]"

    raw = input(f"Minimum years of experience{default_text} (Enter to keep): ").strip()
    if not raw:
        return default_min_years

    try:
        return float(raw)
    except ValueError:
        print("Invalid number. Keeping previous/default min years.")
        return default_min_years


def _run_interactive_mode(matcher: JobMatcher, args: argparse.Namespace) -> None:
    print("Interactive Resume Matcher")
    print("Type a job description and press Enter. Type 'exit' to quit.")

    base_required_skills = _parse_required_skills(args.required_skills)
    base_min_years = args.min_years

    while True:
        jd = input("\nJob description > ").strip()
        if not jd:
            continue

        if jd.lower() in {"exit", "quit", ":q", "q"}:
            print("Exiting interactive mode.")
            break

        current_required_skills = _prompt_required_skills(base_required_skills)
        current_min_years = _prompt_min_years(base_min_years)

        result = matcher.match(
            job_description=jd,
            top_k=args.top_k,
            initial_candidates=args.initial_candidates,
            required_skills=current_required_skills,
            min_years=current_min_years,
            use_llm_reasoning=args.use_llm_reasoning,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_reasoning_candidates=args.llm_reasoning_candidates,
            hf_max_new_tokens=args.hf_max_new_tokens,
        )

        print(json.dumps(result, indent=2))

        update_defaults = input("Use current filters as new defaults? (y/N): ").strip().lower()
        if update_defaults in {"y", "yes"}:
            base_required_skills = current_required_skills
            base_min_years = current_min_years


def _merge_cli_settings(base: Settings, args: argparse.Namespace) -> Settings:
    return Settings(
        embedding_model=args.embedding_model or base.embedding_model,
        chroma_db_path=args.db_path or base.chroma_db_path,
        chroma_collection_name=args.collection or base.chroma_collection_name,
        chunk_size_words=base.chunk_size_words,
        chunk_overlap_words=base.chunk_overlap_words,
        default_top_k=base.default_top_k,
        initial_candidate_chunks=base.initial_candidate_chunks,
        skill_vocabulary=base.skill_vocabulary,
        llm_provider=base.llm_provider,
        local_llm_model=base.local_llm_model,
        local_llm_base_url=base.local_llm_base_url,
        local_llm_timeout_seconds=base.local_llm_timeout_seconds,
        llm_reasoning_candidates=base.llm_reasoning_candidates,
        hf_local_model=base.hf_local_model,
        hf_local_max_new_tokens=base.hf_local_max_new_tokens,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    settings = _merge_cli_settings(load_settings(), args)
    matcher = JobMatcher(settings=settings)

    if args.interactive:
        _run_interactive_mode(matcher=matcher, args=args)
        return

    job_description = _read_job_description(args, parser)
    required_skills = _parse_required_skills(args.required_skills)

    result = matcher.match(
        job_description=job_description,
        top_k=args.top_k,
        initial_candidates=args.initial_candidates,
        required_skills=required_skills,
        min_years=args.min_years,
        use_llm_reasoning=args.use_llm_reasoning,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_reasoning_candidates=args.llm_reasoning_candidates,
        hf_max_new_tokens=args.hf_max_new_tokens,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
