from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

CID_RE = re.compile(r'"candidate_id"\s*:\s*"([^"]+)"')


def iter_candidate_ids(path: str | Path):
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for candidate in data:
            yield candidate["candidate_id"]
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            match = CID_RE.search(line)
            if match:
                yield match.group(1)
            else:
                yield json.loads(line)["candidate_id"]


def load_reasonings(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("reasonings", data)


def load_runtime_rankings(path: Path) -> list[tuple[str, float]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing runtime ranking cache: {path}. "
            "Run offline/precompute_features.py and export cache/runtime_rankings.csv before ranking."
        )
    rankings: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rankings.append((row["candidate_id"], float(row["final_score"])))
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ranking-cache", default="cache/runtime_rankings.csv")
    parser.add_argument("--reasoning-cache", default="cache/reasoning_cache.json")
    args = parser.parse_args()

    candidate_ids = set(iter_candidate_ids(args.candidates))
    ranked = [
        (cid, score)
        for cid, score in load_runtime_rankings(Path(args.ranking_cache))
        if cid in candidate_ids
    ][:100]
    if len(ranked) < 100:
        raise RuntimeError(f"Need at least 100 cached candidates, found {len(ranked)}")
    reasonings = load_reasonings(Path(args.reasoning_cache))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, raw_score) in enumerate(ranked, start=1):
            score = max(0.0, raw_score - rank * 1e-12)
            reasoning = reasonings.get(
                cid,
                f"Ranked by cached JD-fit, hybrid retrieval, and hireability features; score {raw_score:.4f}.",
            )
            writer.writerow([cid, rank, f"{score:.12f}", reasoning])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
