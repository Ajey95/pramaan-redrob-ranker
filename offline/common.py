from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field


REFERENCE_DATE = date(2026, 7, 2)
RRF_K = 60

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.\-]*")

REQUIRED_TERMS = [
    "embedding",
    "embeddings",
    "retrieval",
    "ranking",
    "ranker",
    "search",
    "recommendation",
    "recommender",
    "hybrid",
    "bm25",
    "vector",
    "faiss",
    "elasticsearch",
    "opensearch",
    "qdrant",
    "pinecone",
    "weaviate",
    "milvus",
    "llm",
    "fine-tuning",
    "finetuning",
    "lora",
    "qlora",
    "python",
    "ndcg",
    "mrr",
    "map",
    "ab",
    "a/b",
    "evaluation",
    "experiment",
    "production",
    "deployed",
    "shipped",
]

CORE_AI_TERMS = {
    "embedding",
    "embeddings",
    "retrieval",
    "ranking",
    "ranker",
    "search",
    "recommendation",
    "recommender",
    "nlp",
    "llm",
    "fine-tuning",
    "finetuning",
    "lora",
    "qlora",
    "python",
    "machine learning",
    "ml",
    "vector",
    "faiss",
    "elasticsearch",
    "opensearch",
    "qdrant",
    "pinecone",
    "weaviate",
    "milvus",
    "bm25",
    "hybrid search",
    "semantic search",
    "ndcg",
    "mrr",
    "map",
}

SYSTEM_EVIDENCE_TERMS = {
    "shipped",
    "deployed",
    "production",
    "scale",
    "users",
    "latency",
    "pipeline",
    "index",
    "refresh",
    "a/b",
    "experiment",
    "online",
    "offline benchmark",
    "metrics",
    "relevance",
}

TRENDY_FRAMEWORKS = {
    "langchain",
    "llamaindex",
    "openai",
    "anthropic",
    "gpt",
    "rag",
    "autogen",
    "crewai",
    "haystack",
}

CONSULTING_NAMES = {
    "tcs",
    "tata consultancy",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
}

MANAGEMENT_TITLE_TERMS = {
    "architect",
    "lead",
    "head",
    "manager",
    "director",
    "principal",
    "vp",
}

UNRELATED_TITLE_TERMS = {
    "marketing",
    "sales",
    "hr",
    "support",
    "operations",
    "project manager",
    "program manager",
    "business analyst",
    "qa",
    "accountant",
    "finance",
    "content",
    "designer",
    "frontend",
    "mechanical",
    "civil",
    "recruiter",
    "customer success",
}

NONCODING_TERMS = {
    "managed",
    "stakeholder",
    "roadmap",
    "program",
    "delivery",
    "governance",
    "mentored",
    "hiring",
    "budget",
}

CODING_TERMS = {
    "built",
    "implemented",
    "coded",
    "python",
    "api",
    "service",
    "pipeline",
    "deployed",
    "optimized",
}

RESEARCH_TERMS = {
    "research",
    "academic",
    "university",
    "lab",
    "publication",
    "thesis",
    "professor",
}

VISION_SPEECH_ROBOTICS_TERMS = {
    "computer vision",
    "image classification",
    "object detection",
    "speech",
    "asr",
    "tts",
    "robotics",
    "robot",
    "opencv",
}

NLP_IR_TERMS = {
    "nlp",
    "retrieval",
    "ranking",
    "search",
    "recommendation",
    "recommender",
    "semantic",
    "embedding",
    "bm25",
    "vector",
}

GOOD_LOCATIONS = {
    "pune",
    "noida",
    "hyderabad",
    "mumbai",
    "delhi",
    "gurgaon",
    "gurugram",
    "ncr",
    "bengaluru",
    "bangalore",
}


class JDStructured(BaseModel):
    role_title: str
    required_skills: list[str]
    nice_to_have_skills: list[str]
    hard_disqualifiers: list[str]
    heavy_penalties: list[str]
    ideal_profile: str
    preferred_locations: list[str]


class CacheManifest(BaseModel):
    feature_version: str
    source_candidates: str
    candidate_count: int
    jd_cache: str
    dense_backend: str = "unknown"
    embedding_model: str | None = None
    embeddings_cache: str | None = None
    created_at: str
    notes: list[str] = Field(default_factory=list)


class ReasoningCache(BaseModel):
    version: str
    reasonings: dict[str, str]


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".gz":
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"
    if path.suffix == ".json":
        with opener(path, mode, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON list")
        return data
    candidates = []
    with opener(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def iter_candidate_ids(path: str | Path) -> Iterable[str]:
    path = Path(path)
    if path.suffix == ".json":
        for candidate in load_candidates(path):
            yield candidate["candidate_id"]
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)["candidate_id"]


def write_json_validated(model: BaseModel, path: str | Path) -> None:
    Path(path).write_text(
        model.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def contains_any(text: str, terms: Iterable[str]) -> bool:
    low = str(text).lower()
    return any(term in low for term in terms)


def count_terms(text: str, terms: Iterable[str]) -> int:
    low = str(text).lower()
    return sum(1 for term in terms if term in low)


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_series(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def reciprocal_rank_fusion(rankings: list[list[int]], item_count: int) -> list[float]:
    scores = [0.0] * item_count
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (RRF_K + rank + 1)
    return scores


def bm25_scores(tokenized_docs: list[list[str]], query_tokens: list[str]) -> list[float]:
    if not tokenized_docs:
        return []
    n_docs = len(tokenized_docs)
    avgdl = sum(len(doc) for doc in tokenized_docs) / max(1, n_docs)
    doc_freq = Counter()
    doc_counts = []
    query_unique = set(query_tokens)
    for doc in tokenized_docs:
        counts = Counter(doc)
        doc_counts.append(counts)
        for term in query_unique:
            if counts.get(term, 0):
                doc_freq[term] += 1
    k1 = 1.5
    b = 0.75
    scores = []
    for counts, doc in zip(doc_counts, tokenized_docs):
        dl = max(1, len(doc))
        score = 0.0
        for term in query_unique:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        scores.append(float(score))
    return scores


def recency_days(value: str | None) -> int:
    if not value:
        return 999
    try:
        return (REFERENCE_DATE - datetime.strptime(value, "%Y-%m-%d").date()).days
    except ValueError:
        return 999


def seniority_level(title: str) -> int:
    low = title.lower()
    if "principal" in low or "staff" in low:
        return 4
    if "senior" in low or "sr" in low or "lead" in low:
        return 3
    if "engineer" in low or "scientist" in low:
        return 2
    return 1


def current_entry(candidate: dict[str, Any]) -> dict[str, Any]:
    history = candidate.get("career_history") or []
    for entry in history:
        if entry.get("is_current"):
            return entry
    return history[0] if history else {}


def career_chunks(candidate: dict[str, Any]) -> list[str]:
    chunks = []
    for idx, entry in enumerate(candidate.get("career_history", []), start=1):
        chunks.append(
            "\n".join(
                [
                    f"Role {idx}",
                    f"Company: {entry.get('company', '')}",
                    f"Title: {entry.get('title', '')}",
                    f"Industry: {entry.get('industry', '')}",
                    f"Duration months: {entry.get('duration_months', 0)}",
                    f"Current: {entry.get('is_current', False)}",
                    f"Description: {entry.get('description', '')}",
                ]
            )
        )
    return chunks


def skill_names(candidate: dict[str, Any]) -> list[str]:
    return [str(skill.get("name", "")).strip() for skill in candidate.get("skills", []) if skill.get("name")]


def format_pct(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    return f"{float(value) * 100:.0f}%"
