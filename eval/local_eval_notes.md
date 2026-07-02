# Local Evaluation Notes

The challenge has no public labels or live leaderboard, so QA is manual and methodology-driven.

## Checks Completed

- Confirmed the real `candidates.jsonl` has 100000 rows.
- Ran feature precompute on `sample_candidates.json` before full-pool scoring.
- Inspected the first sample ranking and found unrelated title keyword-stuffers ranking too high.
- Added unrelated-title and experience-band gates, then reran the sample and full pool.
- Attempted full-pool MiniLM and observed it was too slow for CPU-only offline iteration.
- Replaced full-pool MiniLM with a two-stage cascade: full-pool cheap recall followed by MiniLM on the top 1000 shortlist.
- Regenerated `features.parquet`, `embeddings.npy`, `reasoning_cache.json`, and `submission.csv` from the two-stage MiniLM cache.
- Inspected the final top 25 from `cache/preview_top.csv`; top candidates are ML/search/retrieval/ranking builders.
- Generated reasoning for the top 150 as a safety margin.
- Ran `rank.py` on the full candidate file in 9.55 seconds and validated `submission.csv`.

## Worked Examples

Strong match: `CAND_0079387` ranked first after two-stage MiniLM. It has 6.9 years, AI Engineer title, recommendation-system work, vector search/OpenSearch skills, and explicit shipped search/ranking/retrieval evidence.

Second strong match: `CAND_0041669` is a Recommendation Systems Engineer in Noida with career-history evidence around ranking layers and retrieval infrastructure.

Keyword-stuffer rejection pattern: sample candidates with Marketing/HR/Mechanical/Operations titles and AI-looking skills were pushed down unless their career history showed shipped ranking, search, recommendation, or retrieval systems.

Plain-language surfaced pattern: the rules boost descriptions that mention search/discovery, recommendation systems, ranking layers, relevance labels, offline-online correlation, and A/B tests even when the skills list is less fashionable.

## Timing Notes

- TF-IDF baseline full precompute: about 6.7 minutes.
- Full-pool MiniLM attempt: too slow on CPU, reached roughly 78 percent before timeout.
- Two-stage MiniLM full precompute with top-1000 shortlist: about 13.45 minutes offline.
- Timed `rank.py`: 9.55 seconds on the full 100k file.

## Remaining Human Checks Before Portal Upload

- Fill contact/team fields in `submission_metadata.yaml`.
- Add the real GitHub repository URL.
- Add the live sandbox URL after hosting the Streamlit app or Docker image.
- Run one final reproduction in a no-network CPU-only environment.
