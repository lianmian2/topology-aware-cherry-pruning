from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_atomic_segment_graphs import ensure_dir, process_sample


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        deadline = time.monotonic() + 15.0
        while True:
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build atomic graphs for V3.1 global-quality warnings without rerunning GPU perception"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    auto_root = dataset / "auto_perception"
    graph_root = dataset / "atomic_graphs"
    annotation_root = dataset / "manual_annotations"
    warning_root = ensure_dir(dataset / "structural_warnings")
    report_root = ensure_dir(dataset / "exports" / "structural_warning_recovery_20260802")

    targets: list[tuple[str, Path, dict[str, Any]]] = []
    for sample_dir in sorted(path for path in graph_root.iterdir() if path.is_dir()):
        marker_path = sample_dir / ".complete.json"
        if not marker_path.exists():
            continue
        marker = read_json(marker_path)
        if marker.get("status") == "auto_excluded":
            targets.append((sample_dir.name, marker_path, marker))

    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    total = len(targets)
    print(f"Found {total} global structural warnings to recover for manual review.", flush=True)
    for index, (sample_id, marker_path, old_marker) in enumerate(targets, start=1):
        sample_started = time.perf_counter()
        annotation_path = annotation_root / f"{sample_id}.json"
        if annotation_path.exists():
            failed.append({"sample_id": sample_id, "error": "manual_annotation_already_exists"})
            print(f"[{index}/{total}] SKIP {sample_id}: manual annotation already exists", flush=True)
            continue
        try:
            graph_quality = process_sample(auto_root, graph_root, sample_id, None)
            structural_warning = old_marker.get("exclusion", {}) or {
                "reason": "legacy_auto_excluded",
                "sample_id": sample_id,
            }
            new_marker = {
                **old_marker,
                "status": "manual_quality_review",
                "legacy_status": "auto_excluded",
                "recovered_at": datetime.now().isoformat(timespec="seconds"),
                "recovery_policy": "global topology diagnostics are warnings; exclude only when a cut cannot be represented correctly",
                "structural_warning": structural_warning,
                "graph_quality": graph_quality,
            }
            new_marker.pop("exclusion", None)
            atomic_json(marker_path, new_marker)
            atomic_json(
                warning_root / f"{sample_id}.json",
                {
                    "schema_version": "1.0",
                    "sample_id": sample_id,
                    "status": "requires_manual_quality_review",
                    "warning": structural_warning,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            elapsed = time.perf_counter() - sample_started
            completed.append({"sample_id": sample_id, "elapsed_seconds": elapsed, **graph_quality})
            remaining = total - index
            mean = (time.perf_counter() - started) / max(index, 1)
            print(
                f"[{index}/{total}] OK {sample_id} | {elapsed:.1f}s | "
                f"geometry={graph_quality.get('geometry_valid')} | ETA={mean * remaining / 60:.1f} min",
                flush=True,
            )
        except Exception as exc:
            failed.append({"sample_id": sample_id, "error": repr(exc)})
            print(f"[{index}/{total}] FAILED {sample_id}: {exc!r}", flush=True)

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "global structural diagnostics downgraded from automatic exclusion to mandatory human warning review",
        "targets": total,
        "completed": len(completed),
        "failed": failed,
        "samples": completed,
    }
    atomic_json(report_root / "summary.json", report)
    print(json.dumps({key: report[key] for key in ("targets", "completed", "failed")}, ensure_ascii=False, indent=2))
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
