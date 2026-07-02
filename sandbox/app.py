from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.generate_reasoning import make_reason
from offline.precompute_features import build_features


def load_uploaded_candidates(raw: str) -> list[dict]:
    stripped = raw.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("JSON upload must be a candidate list")
        return data
    candidates = []
    for line in stripped.splitlines():
        if line.strip():
            candidates.append(json.loads(line))
    return candidates


def build_csv(ranked: pd.DataFrame, candidates_by_id: dict[str, dict]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        cid = row["candidate_id"]
        writer.writerow(
            [
                cid,
                rank,
                f"{float(row['final_score']):.12f}",
                make_reason(rank, row, candidates_by_id[cid]),
            ]
        )
    return out.getvalue()


st.set_page_config(page_title="Pramaan Redrob Ranker", layout="wide")
st.title("Pramaan Redrob Ranker")

st.write("Upload up to 100 Redrob candidate profiles as JSON or JSONL. The sandbox uses the same local scoring rules with TF-IDF dense fallback for fast hosted verification.")
uploaded = st.file_uploader("Candidate sample", type=["json", "jsonl"])

if uploaded is not None:
    raw = uploaded.read().decode("utf-8")
    candidates = load_uploaded_candidates(raw)
    if len(candidates) > 100:
        st.error("Sandbox accepts at most 100 candidates.")
    elif not candidates:
        st.warning("No candidates found.")
    else:
        with st.spinner("Ranking sample..."):
            features, dense_backend, _ = build_features(candidates, dense_backend="tfidf", embeddings_out=None)
            ranked = features.sort_values(["final_score", "candidate_id"], ascending=[False, True]).head(min(100, len(features)))
            candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
            csv_text = build_csv(ranked, candidates_by_id)
        st.caption(f"Dense backend: {dense_backend}. No network calls are made during ranking.")
        st.dataframe(ranked[["candidate_id", "current_title", "years_of_experience", "location", "final_score", "jd_fit_gate_score", "skill_match_score"]])
        st.download_button("Download CSV", csv_text, "sample_submission.csv", "text/csv")
