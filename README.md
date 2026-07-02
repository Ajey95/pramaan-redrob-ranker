# Pramaan Redrob Ranker

Pramaan is a two-phase candidate ranker for the Redrob Senior AI Engineer JD. The offline phase can be slow: it parses the fixed JD, builds candidate features, runs dense retrieval plus BM25, applies JD Fit Gate rules, scores Redrob hireability, and caches reasonings. The timed `rank.py` phase is deliberately small: no model inference, no LLM/API calls, no GPU, and no network access.

Portal summary under 200 words:

Pramaan ranks Redrob candidates with an offline/online split. Offline scripts weight career-history evidence above skills keywords, because the JD explicitly warns that keyword stuffing is a trap. Each career entry is treated as a section-aware chunk containing company, title, dates, industry, and description. Dense retrieval prefers `all-MiniLM-L6-v2` through `sentence-transformers`, with an explicit TF-IDF fallback if the local model is unavailable; BM25 is fused with dense rankings using reciprocal-rank fusion. Final scoring multiplies JD Fit Gate score, skill-match score, hireability modifier, and profile-consistency modifier. The timed `rank.py` only loads cached features and deterministic reasonings, filters to the supplied candidate file, sorts deterministically, and writes the required 100-row CSV. Local validation passed with the organizer validator; the final timed ranking step ran in about 8.23 seconds on the full 100k file in the original local environment.

## Reproduce

```bash
python offline/jd_understanding.py --jd data/job_description.md --out cache/jd_structured.json
python offline/precompute_features.py --candidates ../India_runs_data_and_ai_challenge/candidates.jsonl --out cache/features.parquet --dense-backend sentence-transformers --embedding-model all-MiniLM-L6-v2 --embeddings-out cache/embeddings.npy
python offline/generate_reasoning.py --candidates ../India_runs_data_and_ai_challenge/candidates.jsonl --features cache/features.parquet --top-n 150
python rank.py --candidates ../India_runs_data_and_ai_challenge/candidates.jsonl --out submission.csv
python validate_submission.py submission.csv
```

If `sentence-transformers` or the local MiniLM model is unavailable, use `--dense-backend auto`; the manifest records whether it used `sentence_transformers` or `tfidf_fallback`.

## Architecture

```text
candidates.jsonl + job_description.md
  -> offline/jd_understanding.py -> cache/jd_structured.json
  -> offline/precompute_features.py
       career-history chunks
       dense MiniLM retrieval or explicit TF-IDF fallback
       BM25
       RRF = 1 / (60 + rank + 1)
       JD Fit Gate + profile consistency + hireability
       -> cache/features.parquet, cache/embeddings.npy, cache/preview_top.csv
  -> offline/generate_reasoning.py -> cache/reasoning_cache.json
  -> rank.py -> submission.csv
```

## JD Fit Gate Rules

- Pure research/academic career without shipped-system language gets near-zero score because the JD says Redrob will not move forward without production deployment.
- Consulting/services-only careers at named firms get near-zero score unless there is product-company or shipped-system evidence, matching the JD's explicit services-career warning.
- Recent LangChain/OpenAI-only AI work under 12 months is penalized unless earlier ML/IR depth exists, because the JD wants pre-LLM-era retrieval/ranking depth.
- Non-coding architecture/leadership roles longer than 18 months are penalized because the JD says this role writes code.
- Short-tenure title-chasing patterns are penalized because the JD wants someone likely to stay 3+ years.
- Framework keywords without shipped-system depth are penalized because the JD rejects framework enthusiasm over system thinking.
- CV/speech/robotics-primary profiles without NLP/IR evidence are penalized because the JD explicitly says that background is not the target.
- Far-outside experience bands are down-weighted, not excluded, because the JD treats 5-9 years as flexible but describes the ideal as roughly 6-8 years.

## Scoring

```text
final_score = jd_fit_gate_score * skill_match_score * hireability_modifier * consistency_modifier
```

`skill_match_score` combines dense/BM25 RRF, career evidence, JD keyword support, title fit, experience fit, and location fit. Career-history evidence has more weight than skills keywords. `hireability_modifier` uses recruiter response, interview completion, offer acceptance, open-to-work, recency, notice period, recruiter saves/search appearances, and GitHub activity as a multiplier, as requested by the Redrob signals guide.

## Manual QA Notes

Strong match example: `CAND_0055905` ranked first because the career history describes a large-scale semantic search migration, BM25-to-hybrid retrieval, LLM/fine-tuning work, evaluation harnesses, and 8.1 years of senior ML experience.

Rejected keyword-stuffer pattern: in the 50-candidate sample, unrelated Marketing/HR/Mechanical profiles with AI-looking skills initially ranked too high. The JD Fit Gate now down-weights unrelated current titles unless career-history evidence shows shipped ranking/retrieval work.

Plain-language match pattern: candidates with career descriptions like "search and discovery", "content recommendation", "ranking layer", "offline-online correlation", or "relevance evaluation" are boosted even when skills arrays omit fashionable terms like RAG.

## Reasoning

Reasoning is generated offline from cached feature breakdowns. Optional Anthropic calls are implemented for JD parsing and reasoning rewrite, but the default path is deterministic and grounded. A failed offline LLM call prints a warning and falls back to deterministic text; `rank.py` never calls an LLM.

## Runtime

- Full offline precompute on 100k with TF-IDF fallback: about 6.7 minutes in the original local run.
- Timed `rank.py` over full 100k: about 8.23 seconds in the original local run.
- `validate_submission.py submission.csv`: passed.

## Sandbox

Run the small-sample Streamlit demo:

```bash
streamlit run sandbox/app.py
```

The app accepts up to 100 candidates as JSON or JSONL and produces a ranked CSV. It uses the TF-IDF dense fallback for fast hosted execution.

## AI Tools Declaration

Codex/ChatGPT was used as an implementation assistant. The ranking step uses only local cached artifacts and makes no network or LLM calls.
