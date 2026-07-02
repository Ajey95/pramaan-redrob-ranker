from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from common import (
    CODING_TERMS,
    CONSULTING_NAMES,
    CORE_AI_TERMS,
    GOOD_LOCATIONS,
    MANAGEMENT_TITLE_TERMS,
    NLP_IR_TERMS,
    NONCODING_TERMS,
    REFERENCE_DATE,
    REQUIRED_TERMS,
    RESEARCH_TERMS,
    SYSTEM_EVIDENCE_TERMS,
    TRENDY_FRAMEWORKS,
    UNRELATED_TITLE_TERMS,
    VISION_SPEECH_ROBOTICS_TERMS,
    CacheManifest,
    bm25_scores,
    career_chunks,
    clip,
    contains_any,
    count_terms,
    current_entry,
    load_candidates,
    normalize_series,
    reciprocal_rank_fusion,
    recency_days,
    seniority_level,
    skill_names,
    tokenize,
    write_json_validated,
)


FEATURE_VERSION = "pramaan_v3_rules_rrf_minilm_2026_07_02"

JD_QUERY = (
    "Senior AI Engineer founding team production embeddings retrieval ranking hybrid search "
    "vector database BM25 semantic search recommendation recommender Python NDCG MRR MAP "
    "A/B testing evaluation LLM fine-tuning product engineering shipped deployed real users"
)


def _lower_join(values: list[str]) -> str:
    return " ".join(str(v).lower() for v in values if v)


def _company_is_named_consulting(company: str) -> bool:
    low = company.lower()
    return any(name in low for name in CONSULTING_NAMES)


def _industry_is_services(industry: str) -> bool:
    low = industry.lower()
    return "service" in low or "consult" in low or "outsourcing" in low


def compute_rule_features(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = candidate["profile"]
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate["redrob_signals"]
    current = current_entry(candidate)

    chunks = career_chunks(candidate)
    career_text = "\n".join(chunks)
    skills_text = _lower_join(skill_names(candidate))
    profile_text = " ".join(
        [
            profile.get("headline", ""),
            profile.get("summary", ""),
            profile.get("current_title", ""),
            profile.get("current_industry", ""),
        ]
    )
    all_text = f"{profile_text}\n{career_text}\n{skills_text}".lower()
    current_text = " ".join(
        [
            current.get("title", ""),
            current.get("industry", ""),
            current.get("description", ""),
        ]
    ).lower()
    prior_text = " ".join(entry.get("description", "") for entry in history[1:]).lower()

    ai_count = count_terms(all_text, CORE_AI_TERMS)
    career_ai_count = count_terms(career_text, CORE_AI_TERMS)
    system_count = count_terms(career_text, SYSTEM_EVIDENCE_TERMS)
    trendy_count = count_terms(f"{skills_text} {career_text}", TRENDY_FRAMEWORKS)
    nlp_ir_count = count_terms(all_text, NLP_IR_TERMS)
    vision_speech_count = count_terms(all_text, VISION_SPEECH_ROBOTICS_TERMS)

    companies = [entry.get("company", "") for entry in history]
    industries = [entry.get("industry", "") for entry in history]
    named_consulting_only = bool(history) and all(
        _company_is_named_consulting(company) or _industry_is_services(industry)
        for company, industry in zip(companies, industries)
    ) and not contains_any(career_text, SYSTEM_EVIDENCE_TERMS)

    researchish_entries = [
        contains_any(
            " ".join([entry.get("company", ""), entry.get("industry", ""), entry.get("title", ""), entry.get("description", "")]),
            RESEARCH_TERMS,
        )
        for entry in history
    ]
    pure_research_no_ship = bool(history) and all(researchish_entries) and not contains_any(career_text, SYSTEM_EVIDENCE_TERMS)

    recent_llm_only = (
        int(current.get("duration_months") or 0) < 12
        and contains_any(current_text + " " + skills_text, TRENDY_FRAMEWORKS)
        and not contains_any(prior_text, NLP_IR_TERMS | SYSTEM_EVIDENCE_TERMS)
    )

    title = profile.get("current_title", "")
    noncoding_manager = (
        contains_any(title, MANAGEMENT_TITLE_TERMS)
        and int(current.get("duration_months") or 0) > 18
        and contains_any(current_text, NONCODING_TERMS)
        and not contains_any(current_text, CODING_TERMS)
    )

    short_roles = [entry for entry in history if int(entry.get("duration_months") or 0) < 18]
    title_levels = [seniority_level(entry.get("title", "")) for entry in history]
    title_chaser = len(short_roles) >= 3 and max(title_levels or [0]) - min(title_levels or [0]) >= 2

    framework_enthusiast = trendy_count >= 4 and system_count <= 1 and career_ai_count <= 3
    cv_speech_primary = vision_speech_count >= 3 and nlp_ir_count == 0

    expert_zero = sum(
        1
        for skill in skills
        if skill.get("proficiency") == "expert" and int(skill.get("duration_months") or 0) <= 1
    )
    expert_count = sum(1 for skill in skills if skill.get("proficiency") == "expert")
    total_skill_months = sum(int(skill.get("duration_months") or 0) for skill in skills)
    years = float(profile.get("years_of_experience") or 0.0)
    skill_breadth_implausible = len(skills) >= 25 or expert_count > max(5, years * 1.2)
    duration_implausible = total_skill_months > max(180, years * 12 * 5)
    consistency_penalty = 1.0
    if expert_zero:
        consistency_penalty *= max(0.55, 1.0 - expert_zero * 0.08)
    if skill_breadth_implausible:
        consistency_penalty *= 0.75
    if duration_implausible:
        consistency_penalty *= 0.82

    productish = not _industry_is_services(profile.get("current_industry", "")) or contains_any(career_text, SYSTEM_EVIDENCE_TERMS)
    ranking_system_evidence = count_terms(career_text, {"ranking", "ranker", "search", "retrieval", "recommendation", "recommender", "matching"})
    product_ml_signal = clip((career_ai_count * 0.08) + (system_count * 0.08) + (0.2 if productish else 0.0))
    career_evidence_score = clip((ranking_system_evidence * 0.16) + (system_count * 0.07) + (career_ai_count * 0.04))
    jd_keyword_score = clip((ai_count * 0.045) + (trendy_count * 0.015))

    title_low = title.lower()
    title_score = 0.15
    if "ai" in title_low or "machine learning" in title_low or "ml" in title_low:
        title_score = 0.95
    elif "data scientist" in title_low or "data engineer" in title_low:
        title_score = 0.72
    elif contains_any(title_low, {"frontend", "qa", "cloud", "devops", "project manager", "business analyst"}):
        title_score = 0.24
    elif "backend" in title_low or "software" in title_low or "engineer" in title_low:
        title_score = 0.58
    elif contains_any(title_low, UNRELATED_TITLE_TERMS):
        title_score = 0.05

    experience_score = clip(1.0 - abs(years - 7.0) / 7.0)
    if 5.0 <= years <= 9.0:
        experience_score = max(experience_score, 0.88)
    elif 4.0 <= years < 5.0 or 9.0 < years <= 11.0:
        experience_score = max(experience_score, 0.62)

    location = profile.get("location", "")
    location_score = 0.25
    if contains_any(location, GOOD_LOCATIONS) or signals.get("willing_to_relocate"):
        location_score = 1.0
    elif profile.get("country", "").lower() == "india":
        location_score = 0.65

    jd_fit_gate_score = 1.0
    flags = []
    if pure_research_no_ship:
        jd_fit_gate_score *= 0.03
        flags.append("hard_pure_research_no_production")
    if named_consulting_only:
        jd_fit_gate_score *= 0.06
        flags.append("hard_consulting_services_only")
    if recent_llm_only:
        jd_fit_gate_score *= 0.55
        flags.append("penalty_recent_llm_only")
    if noncoding_manager:
        jd_fit_gate_score *= 0.45
        flags.append("penalty_non_coding_management")
    if title_chaser:
        jd_fit_gate_score *= 0.75
        flags.append("penalty_short_tenure_title_chasing")
    if framework_enthusiast:
        jd_fit_gate_score *= 0.70
        flags.append("penalty_framework_keywords_without_depth")
    if cv_speech_primary:
        jd_fit_gate_score *= 0.65
        flags.append("penalty_cv_speech_robotics_without_ir")
    if contains_any(title_low, UNRELATED_TITLE_TERMS) and career_evidence_score < 0.45:
        jd_fit_gate_score *= 0.18
        flags.append("penalty_unrelated_current_title")
    if title_score <= 0.15 and product_ml_signal < 0.25:
        jd_fit_gate_score *= 0.70
        flags.append("penalty_weak_product_ml_signal")
    if years > 15:
        jd_fit_gate_score *= 0.65
        flags.append("penalty_far_above_experience_band")
    elif years > 12:
        jd_fit_gate_score *= 0.78
        flags.append("penalty_above_experience_band")
    elif years < 4:
        jd_fit_gate_score *= 0.72
        flags.append("penalty_below_senior_experience_band")
    elif years < 5:
        jd_fit_gate_score *= 0.90
        flags.append("penalty_slightly_below_experience_band")

    inactive_days = recency_days(signals.get("last_active_date"))
    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    interview_rate = float(signals.get("interview_completion_rate") or 0.0)
    offer_rate = float(signals.get("offer_acceptance_rate") if signals.get("offer_acceptance_rate") is not None else -1)
    notice = int(signals.get("notice_period_days") or 0)
    github_score = float(signals.get("github_activity_score") if signals.get("github_activity_score") is not None else -1)

    reliability = 0.45 * response_rate + 0.35 * interview_rate + 0.20 * (0.55 if offer_rate < 0 else offer_rate)
    availability = 0.0
    availability += 0.35 if signals.get("open_to_work_flag") else 0.05
    availability += 0.35 * clip(1 - inactive_days / 180)
    availability += 0.30 * clip(1 - notice / 120)
    market = 0.5 * clip(float(signals.get("search_appearance_30d") or 0) / 60) + 0.5 * clip(float(signals.get("saved_by_recruiters_30d") or 0) / 12)
    github = 0.5 if github_score < 0 else clip(github_score / 100)
    hireability_raw = 0.45 * reliability + 0.35 * availability + 0.12 * market + 0.08 * github
    hireability_modifier = 0.70 + 0.55 * hireability_raw

    return {
        "candidate_id": candidate["candidate_id"],
        "current_title": title,
        "current_company": profile.get("current_company", ""),
        "current_industry": profile.get("current_industry", ""),
        "location": location,
        "country": profile.get("country", ""),
        "years_of_experience": years,
        "career_text": career_text,
        "skills_text": ", ".join(skill_names(candidate)),
        "semantic_doc": "\n".join([profile_text, career_text, career_text, career_text, skills_text]),
        "bm25_doc": "\n".join([career_text, career_text, profile_text, skills_text]),
        "jd_fit_gate_score": jd_fit_gate_score,
        "career_evidence_score": career_evidence_score,
        "jd_keyword_score": jd_keyword_score,
        "title_score": title_score,
        "experience_score": experience_score,
        "location_score": location_score,
        "product_ml_signal": product_ml_signal,
        "hireability_modifier": hireability_modifier,
        "consistency_modifier": consistency_penalty,
        "ai_term_count": ai_count,
        "career_ai_term_count": career_ai_count,
        "system_evidence_count": system_count,
        "ranking_system_evidence": ranking_system_evidence,
        "trendy_framework_count": trendy_count,
        "expert_zero_duration_count": expert_zero,
        "last_active_days": inactive_days,
        "recruiter_response_rate": response_rate,
        "interview_completion_rate": interview_rate,
        "offer_acceptance_rate": offer_rate,
        "notice_period_days": notice,
        "open_to_work_flag": bool(signals.get("open_to_work_flag")),
        "willing_to_relocate": bool(signals.get("willing_to_relocate")),
        "flags": flags,
        "evidence": {
            "current_duration_months": int(current.get("duration_months") or 0),
            "top_skills": skill_names(candidate)[:8],
            "career_excerpt": (career_text[:280]).replace("\n", " "),
        },
    }


def _tfidf_dense_scores(semantic_docs: list[str]) -> tuple[list[float], str, None]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=70000,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(semantic_docs + [JD_QUERY])
    scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel().tolist()
    return scores, "tfidf_fallback", None


def _sentence_transformer_scores(
    semantic_docs: list[str],
    model_name: str,
    batch_size: int,
    embeddings_out: str | None,
) -> tuple[list[float], str, str]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(f"sentence-transformers is unavailable: {exc}") from exc

    model = SentenceTransformer(model_name, device="cpu")
    candidate_embeddings = model.encode(
        semantic_docs,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    query_embedding = model.encode(
        [JD_QUERY],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")[0]
    scores = np.matmul(candidate_embeddings, query_embedding).astype("float32").tolist()
    if embeddings_out:
        path = Path(embeddings_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, candidate_embeddings)
    return scores, "sentence_transformers", model_name


def _dense_scores(
    semantic_docs: list[str],
    backend: str,
    model_name: str,
    batch_size: int,
    embeddings_out: str | None,
) -> tuple[list[float], str, str | None]:
    if backend == "tfidf":
        return _tfidf_dense_scores(semantic_docs)
    if backend in {"auto", "sentence-transformers"}:
        try:
            return _sentence_transformer_scores(semantic_docs, model_name, batch_size, embeddings_out)
        except Exception as exc:
            if backend == "sentence-transformers":
                raise
            print(f"Warning: dense embedding model unavailable; using TF-IDF fallback. {exc}")
            return _tfidf_dense_scores(semantic_docs)
    raise ValueError(f"Unknown dense backend: {backend}")


def _rank_positions(ranking: list[int]) -> list[int]:
    positions = [0] * len(ranking)
    for rank, idx in enumerate(ranking, start=1):
        positions[idx] = rank
    return positions


def build_features(
    candidates: list[dict[str, Any]],
    dense_backend: str = "auto",
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_batch_size: int = 256,
    embeddings_out: str | None = None,
) -> tuple[pd.DataFrame, str, str | None]:
    rows = [compute_rule_features(candidate) for candidate in candidates]

    semantic_docs = [row["semantic_doc"] for row in rows]
    dense_scores, actual_dense_backend, actual_embedding_model = _dense_scores(
        semantic_docs,
        dense_backend,
        embedding_model,
        embedding_batch_size,
        embeddings_out,
    )

    tokenized_docs = [tokenize(row["bm25_doc"]) for row in rows]
    bm25_raw = bm25_scores(tokenized_docs, tokenize(JD_QUERY + " " + " ".join(REQUIRED_TERMS)))
    dense_norm = normalize_series(dense_scores)
    bm25_norm = normalize_series(bm25_raw)
    dense_ranking = sorted(range(len(rows)), key=lambda idx: dense_scores[idx], reverse=True)
    bm25_ranking = sorted(range(len(rows)), key=lambda idx: bm25_raw[idx], reverse=True)
    dense_positions = _rank_positions(dense_ranking)
    bm25_positions = _rank_positions(bm25_ranking)
    rrf_raw = reciprocal_rank_fusion([dense_ranking, bm25_ranking], len(rows))
    rrf_norm = [score / (2.0 / 61.0) for score in rrf_raw]

    for idx, row in enumerate(rows):
        skill_match = (
            0.30 * rrf_norm[idx]
            + 0.30 * row["career_evidence_score"]
            + 0.10 * row["jd_keyword_score"]
            + 0.18 * row["title_score"]
            + 0.07 * row["experience_score"]
            + 0.05 * row["location_score"]
        )
        skill_match = clip(skill_match)
        final_score = (
            row["jd_fit_gate_score"]
            * skill_match
            * row["hireability_modifier"]
            * row["consistency_modifier"]
        )
        row["dense_score"] = float(dense_norm[idx])
        row["bm25_score"] = float(bm25_norm[idx])
        row["rrf_score"] = float(rrf_norm[idx])
        row["dense_rank"] = dense_positions[idx]
        row["bm25_rank"] = bm25_positions[idx]
        row["dense_backend"] = actual_dense_backend
        row["embedding_model"] = actual_embedding_model or ""
        row["skill_match_score"] = float(skill_match)
        row["final_score"] = float(final_score)
        row["flags_json"] = json.dumps(row.pop("flags"), ensure_ascii=False)
        row["evidence_json"] = json.dumps(row.pop("evidence"), ensure_ascii=False)
        row.pop("semantic_doc", None)
        row.pop("bm25_doc", None)

    return pd.DataFrame(rows), actual_dense_backend, actual_embedding_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="cache/features.parquet")
    parser.add_argument("--manifest", default="cache/feature_manifest.json")
    parser.add_argument("--preview", default="cache/preview_top.csv")
    parser.add_argument("--jd-cache", default="cache/jd_structured.json")
    parser.add_argument("--dense-backend", choices=["auto", "sentence-transformers", "tfidf"], default="auto")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--embeddings-out", default="cache/embeddings.npy")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    embeddings_out = args.embeddings_out if args.dense_backend != "tfidf" else None
    df, actual_dense_backend, actual_embedding_model = build_features(
        candidates,
        dense_backend=args.dense_backend,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        embeddings_out=embeddings_out,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.sort_values(["final_score", "candidate_id"], ascending=[False, True]).head(250).to_csv(args.preview, index=False)

    manifest = CacheManifest(
        feature_version=FEATURE_VERSION,
        source_candidates=str(args.candidates),
        candidate_count=len(candidates),
        jd_cache=str(args.jd_cache),
        dense_backend=actual_dense_backend,
        embedding_model=actual_embedding_model,
        embeddings_cache=embeddings_out if actual_dense_backend == "sentence_transformers" else None,
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=[
            "Career-history text is weighted above skills text.",
            "Dense retrieval and BM25 candidate rankings are fused with bis-compass RRF: 1/(60+rank+1).",
            "Dense retrieval prefers all-MiniLM-L6-v2 via sentence-transformers and records TF-IDF fallback if the local model is unavailable.",
            "rank.py consumes this cache and performs no model inference or network calls.",
        ],
    )
    write_json_validated(manifest, args.manifest)
    print(f"Wrote {len(df)} feature rows to {out}")
    print(f"Preview: {args.preview}")


if __name__ == "__main__":
    main()
