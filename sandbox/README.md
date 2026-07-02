# Sandbox Demo

This Streamlit app is a small-sample demo for Section 10.5 of the submission spec.
It accepts a JSON or JSONL file with up to 100 candidates and runs the local ranker end to end.

Run locally:

```bash
streamlit run sandbox/app.py
```

The sandbox intentionally uses the TF-IDF dense fallback for fast hosted execution.
The full offline pipeline can use `all-MiniLM-L6-v2` through `sentence-transformers`.
