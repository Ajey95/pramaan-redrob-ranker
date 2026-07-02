# Sandbox Demo

This Streamlit app is a small-sample demo for Section 10.5 of the submission spec.
It accepts a JSON or JSONL file with up to 100 candidates and runs the local ranker end to end.

Run locally:

```bash
streamlit run sandbox/app.py
```

Streamlit Community Cloud settings:

```text
Repository: Ajey95/pramaan-redrob-ranker
Branch: master
Main file path: streamlit_app.py
```

The sandbox intentionally uses the fast local scoring path for hosted verification. The full submission cache uses the offline two-stage MiniLM pipeline:

1. BM25 + TF-IDF + JD Fit Gate full-pool recall.
2. `all-MiniLM-L6-v2` semantic scoring on a top-1000 shortlist.
3. Cached features consumed by `rank.py`.
