from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

try:
    from .common import CORE_AI_TERMS, ReasoningCache, format_pct, load_candidates, write_json_validated
except ImportError:  # pragma: no cover - supports direct script execution
    from common import CORE_AI_TERMS, ReasoningCache, format_pct, load_candidates, write_json_validated


def _candidate_map(path: str) -> dict[str, dict]:
    return {candidate["candidate_id"]: candidate for candidate in load_candidates(path)}


def _flags(row: pd.Series) -> list[str]:
    try:
        return json.loads(row.get("flags_json") or "[]")
    except json.JSONDecodeError:
        return []


def _evidence(row: pd.Series) -> dict:
    try:
        return json.loads(row.get("evidence_json") or "{}")
    except json.JSONDecodeError:
        return {}


def make_reason(rank: int, row: pd.Series, candidate: dict) -> str:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    skills = [skill.get("name", "") for skill in candidate.get("skills", [])]
    flags = _flags(row)
    evidence = _evidence(row)
    title = profile.get("current_title", "candidate")
    years = float(profile.get("years_of_experience") or 0)
    response = format_pct(signals.get("recruiter_response_rate"))
    active_days = int(row.get("last_active_days") or 999)
    notice = int(signals.get("notice_period_days") or 0)
    location = profile.get("location", "unknown location")

    if row["ranking_system_evidence"] >= 2:
        match = "career history shows shipped search/ranking/retrieval work"
    elif row["career_ai_term_count"] >= 5:
        match = "career descriptions show applied ML depth beyond the skills list"
    elif row["rrf_score"] >= 0.55:
        match = "hybrid retrieval placed the profile close to the JD language"
    else:
        match = "profile has adjacent AI/backend evidence but weaker direct ranking-system proof"

    relevant_skills = [
        s
        for s in skills
        if s and any(term in s.lower() for term in CORE_AI_TERMS)
    ]
    named_skills = (relevant_skills or [s for s in skills if s])[:4]
    skill_part = f"named skills include {', '.join(named_skills)}" if named_skills else "skills list is sparse"
    availability = (
        f"open to work with {response} recruiter response"
        if signals.get("open_to_work_flag")
        else f"not marked open to work, with {response} recruiter response"
    )
    if active_days <= 45:
        availability += f" and active {active_days} days ago"
    elif active_days >= 180:
        availability += f", but last active about {active_days} days ago"

    concern_bits = []
    if notice > 60:
        concern_bits.append(f"{notice}-day notice period")
    if flags:
        readable = flags[0].replace("hard_", "").replace("penalty_", "").replace("_", " ")
        concern_bits.append(readable)
    if row["consistency_modifier"] < 0.75:
        concern_bits.append("profile plausibility deductions")
    if row["career_evidence_score"] < 0.25:
        concern_bits.append("limited explicit shipped-ranking evidence")
    concern = ""
    if concern_bits:
        concern = " Concern: " + "; ".join(concern_bits[:2]) + "."

    tone = "Strong fit" if rank <= 20 else "Good fit" if rank <= 60 else "Borderline top-100 fit"
    return (
        f"{tone}: {title} with {years:.1f} years in {location}; {match}, and {skill_part}. "
        f"Hireability is {availability}; score reflects JD-fit {row['jd_fit_gate_score']:.2f} and skill-match {row['skill_match_score']:.2f}.{concern}"
    )


def make_llm_reason(rank: int, row: pd.Series, candidate: dict, model: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    deterministic = make_reason(rank, row, candidate)
    prompt = {
        "model": model,
        "max_tokens": 180,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Rewrite this candidate ranking reason into one or two concise, specific sentences. "
                    "Do not add facts. Preserve concerns and rank tone. Return text only.\n\n"
                    f"Rank: {rank}\n"
                    f"Candidate JSON: {json.dumps(candidate, ensure_ascii=False)[:4500]}\n"
                    f"Feature-backed reason: {deterministic}"
                ),
            }
        ],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(prompt).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = " ".join(part.get("text", "").strip() for part in payload.get("content", [])).strip()
    if len(text) < 40:
        raise RuntimeError("LLM reasoning response was too short")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="cache/features.parquet")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="cache/reasoning_cache.json")
    parser.add_argument("--top-n", type=int, default=150)
    parser.add_argument("--llm-provider", choices=["none", "anthropic"], default="none")
    parser.add_argument("--anthropic-model", default="claude-3-5-haiku-latest")
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    ranked = features.sort_values(["final_score", "candidate_id"], ascending=[False, True]).head(args.top_n)
    candidates = _candidate_map(args.candidates)
    reasonings = {}
    missing = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        cid = row["candidate_id"]
        candidate = candidates.get(cid)
        if candidate is None:
            missing.append(cid)
            continue
        if args.llm_provider == "anthropic":
            try:
                reasonings[cid] = make_llm_reason(rank, row, candidate, args.anthropic_model)
            except (RuntimeError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"Warning: reasoning LLM failed for {cid}; using deterministic fallback. {exc}")
                reasonings[cid] = make_reason(rank, row, candidate)
        else:
            reasonings[cid] = make_reason(rank, row, candidate)
    if missing:
        raise RuntimeError(f"Missing candidates for reasoning: {missing[:5]}")
    cache = ReasoningCache(version="deterministic_grounded_v1", reasonings=reasonings)
    write_json_validated(cache, args.out)
    print(f"Wrote {len(reasonings)} reasonings to {args.out}")


if __name__ == "__main__":
    main()
