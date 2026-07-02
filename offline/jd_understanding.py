from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

try:
    from .common import JDStructured, write_json_validated
except ImportError:  # pragma: no cover - supports direct script execution
    from common import JDStructured, write_json_validated


def build_structured_jd(jd_text: str) -> JDStructured:
    # Deterministic fallback parse. The JD is fixed; keeping this explicit makes
    # Stage 5 explanations easier than a brittle one-off LLM output.
    return JDStructured(
        role_title="Senior AI Engineer - Founding Team",
        required_skills=[
            "production embeddings-based retrieval",
            "hybrid search or vector database operations",
            "ranking evaluation with NDCG, MRR, MAP, or A/B testing",
            "strong Python production engineering",
            "shipping ML systems to real users",
        ],
        nice_to_have_skills=[
            "LLM fine-tuning",
            "learning-to-rank",
            "HR-tech or marketplace products",
            "distributed systems or inference optimization",
            "open-source AI/ML contributions",
        ],
        hard_disqualifiers=[
            "pure research career with no production deployment",
            "entire career at named consulting/services firms without product-company work",
        ],
        heavy_penalties=[
            "recent LangChain/OpenAI-only AI work under 12 months without earlier ML/IR depth",
            "architecture/leadership role longer than 18 months without hands-on production coding",
            "title-chasing short-tenure pattern",
            "computer vision, speech, or robotics primary background without NLP/IR exposure",
        ],
        ideal_profile=(
            "6-8 years total experience with 4-5 years applied ML/AI at product companies, "
            "including a shipped ranking, search, retrieval, or recommendation system at scale."
        ),
        preferred_locations=["Pune", "Noida", "Hyderabad", "Mumbai", "Delhi NCR"],
    )


def _anthropic_structured_jd(jd_text: str, model: str) -> JDStructured:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    prompt = {
        "model": model,
        "max_tokens": 1600,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Extract the fixed JD into strict JSON with keys role_title, "
                    "required_skills, nice_to_have_skills, hard_disqualifiers, "
                    "heavy_penalties, ideal_profile, preferred_locations. Return JSON only.\n\n"
                    + jd_text
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
    text = "".join(part.get("text", "") for part in payload.get("content", []))
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("Anthropic response did not contain JSON")
    return JDStructured.model_validate_json(text[start : end + 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd", default="data/job_description.md")
    parser.add_argument("--out", default="cache/jd_structured.json")
    parser.add_argument("--llm-provider", choices=["none", "anthropic"], default="none")
    parser.add_argument("--anthropic-model", default="claude-3-5-haiku-latest")
    args = parser.parse_args()

    jd_text = Path(args.jd).read_text(encoding="utf-8")
    if args.llm_provider == "anthropic":
        try:
            structured = _anthropic_structured_jd(jd_text, args.anthropic_model)
        except (RuntimeError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"Warning: offline JD LLM parse failed; using deterministic fallback. {exc}")
            structured = build_structured_jd(jd_text)
    else:
        structured = build_structured_jd(jd_text)
    write_json_validated(structured, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
