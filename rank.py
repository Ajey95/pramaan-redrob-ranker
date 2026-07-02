from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from offline.common import iter_candidate_ids


def load_reasonings(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("reasonings", data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--features", default="cache/features.parquet")
    parser.add_argument("--reasoning-cache", default="cache/reasoning_cache.json")
    args = parser.parse_args()

    candidate_ids = set(iter_candidate_ids(args.candidates))
    features = pd.read_parquet(args.features)
    features = features[features["candidate_id"].isin(candidate_ids)].copy()
    if len(features) < 100:
        raise RuntimeError(f"Need at least 100 cached candidates, found {len(features)}")

    ranked = features.sort_values(["final_score", "candidate_id"], ascending=[False, True]).head(100)
    reasonings = load_reasonings(Path(args.reasoning_cache))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            cid = row["candidate_id"]
            raw_score = float(row["final_score"])
            score = max(0.0, raw_score - rank * 1e-12)
            reasoning = reasonings.get(
                cid,
                f"Ranked by cached JD-fit, hybrid retrieval, and hireability features; score {raw_score:.4f}.",
            )
            writer.writerow([cid, rank, f"{score:.12f}", reasoning])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
