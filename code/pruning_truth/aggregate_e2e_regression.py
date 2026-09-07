from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate completed frozen E2E sample summaries")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries: list[dict[str, Any]] = []
    for input_dir in args.input:
        for path in sorted(input_dir.glob("*/summary.json")):
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        raise ValueError(f"No sample summaries found under {args.input}")
    payload = {
        "status": "completed",
        "source": [str(path.resolve()) for path in args.input],
        "samples_requested": len(summaries),
        "samples_completed": len(summaries),
        "failures": [],
        "dag_rate": sum(bool(item["dag"]["is_dag"]) for item in summaries) / len(summaries),
        "category_counts": dict(sorted(Counter(item["category"] for item in summaries).items())),
        "attachment_rate_mean": sum(item["attachment"]["attachment_rate"] for item in summaries) / len(summaries),
        "total_buds": sum(item["buds"] for item in summaries),
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
