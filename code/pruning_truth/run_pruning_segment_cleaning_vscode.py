from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHERRY_PYTHON = Path("D:/app/anna/envs/cherry/python.exe")
MANIFEST = PROJECT_ROOT / "01_data/03_processed/pruning_gnn_v1/manifests/current_pipeline_manifest.json"
OUTPUT = PROJECT_ROOT / "04_results/pruning_decision/pruning_segment_cleaning_v31_stable_20260731"
DEVICE = "cuda:0"
BUD_PATCH_BATCH = 16
RESUME = True
RETRY_FAILED = True
SAMPLE_IDS: list[str] = []


def main() -> int:
    if Path(sys.executable).resolve() != CHERRY_PYTHON.resolve():
        if not CHERRY_PYTHON.exists():
            print(f"[ERROR] Cherry Python not found: {CHERRY_PYTHON}", flush=True)
            return 2
        print(f"Switching to Cherry Python: {CHERRY_PYTHON}", flush=True)
        os.execv(str(CHERRY_PYTHON), [str(CHERRY_PYTHON), str(Path(__file__).resolve())])

    entry = Path(__file__).with_name("prepare_pruning_segment_cleaning.py")
    missing = [path for path in (entry, MANIFEST) if not path.exists()]
    if missing:
        for path in missing:
            print(f"[ERROR] Missing required file: {path}", flush=True)
        return 2

    arguments = [
        str(entry),
        "--manifest", str(MANIFEST),
        "--output", str(OUTPUT),
        "--device", DEVICE,
    ]
    os.environ["CHERRY_BUD_BATCH_SIZE"] = str(BUD_PATCH_BATCH)
    if RESUME:
        arguments.append("--resume")
    if RETRY_FAILED:
        arguments.append("--retry-failed")
    for sample_id in SAMPLE_IDS:
        arguments.extend(("--sample-id", sample_id))

    print("=" * 78, flush=True)
    print("V3.1 pruning atomic-segment preprocessing", flush=True)
    print(f"Python   : {sys.executable}", flush=True)
    print(f"Project  : {PROJECT_ROOT}", flush=True)
    print(f"Manifest : {MANIFEST}", flush=True)
    print(f"Output   : {OUTPUT}", flush=True)
    print(f"Device   : {DEVICE}", flush=True)
    print(f"Bud batch: {BUD_PATCH_BATCH}", flush=True)
    print(f"Samples  : {'all frozen samples' if not SAMPLE_IDS else ', '.join(SAMPLE_IDS)}", flush=True)
    print("Progress is also saved to output/progress.json", flush=True)
    print("Press Ctrl+C once to stop safely; run this file again to resume.", flush=True)
    print("=" * 78, flush=True)

    sys.path.insert(0, str(entry.parent))
    sys.argv = arguments
    from prepare_pruning_segment_cleaning import main as prepare_main

    return prepare_main()


if __name__ == "__main__":
    raise SystemExit(main())
