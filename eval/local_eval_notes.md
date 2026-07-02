# Local Evaluation Notes

The challenge has no public labels or live leaderboard, so QA is manual and methodology-driven.

## Checks Completed

- Confirmed the real `candidates.jsonl` has 100000 rows.
- Ran feature precompute on `sample_candidates.json` before full-pool scoring.
- Inspected the first sample ranking and found unrelated title keyword-stuffers ranking too high.
- Added unrelated-title and experience-band gates, then reran the sample and full pool.
- Inspected the final top 20 from `cache/preview_top.csv`; top candidates are ML/search/retrieval/ranking builders.
- Generated reasoning for the top 150 as a safety margin.
- Ran `rank.py` on the full candidate file and validated `submission.csv`.

## Worked Examples

Strong match: `CAND_0055905` has 8.1 years and career-history evidence of a large-scale semantic search migration from BM25 to hybrid retrieval, plus LLM/fine-tuning and ranking-evaluation work.

Keyword-stuffer rejection pattern: sample candidates with Marketing/HR/Mechanical/Operations titles and AI-looking skills were pushed down unless their career history showed shipped ranking, search, recommendation, or retrieval systems.

Plain-language surfaced pattern: the rules boost descriptions that mention search/discovery, recommendation systems, ranking layers, relevance labels, offline-online correlation, and A/B tests even when the skills list is less fashionable.

## Remaining Human Checks Before Portal Upload

- Fill contact/team fields in `submission_metadata.yaml`.
- Add the real GitHub repository URL.
- Add the live sandbox URL after hosting the Streamlit app or Docker image.
- Run one final reproduction in a no-network CPU-only environment.
