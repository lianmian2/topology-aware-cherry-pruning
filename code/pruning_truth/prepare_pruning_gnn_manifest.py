from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze complete pruning-truth samples into a current-pipeline manifest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_root = PROJECT_ROOT / "04_results/pruning_truth/audits/manual_cut_lines_20260720"
    annotation_root = PROJECT_ROOT / "04_results/pruning_truth/manual_cut_lines_20260720_raw"
    source_manifest = list(csv.DictReader((audit_root / "annotation_manifest.csv").open(encoding="utf-8-sig")))
    exclusions = list(csv.DictReader((audit_root / "excluded_samples.csv").open(encoding="utf-8-sig")))
    excluded = {row["sample_id"] for row in exclusions if row.get("sample_id")}
    defaults = json.loads((PROJECT_ROOT / "02_code/05_utils/pruning_truth/e2e_regression_manifest.json").read_text(encoding="utf-8"))["defaults"]
    accepted = []
    derived_rows = []
    for row in source_manifest:
        sample_id = row["sample_id"]
        tree_id, view = row["tree_id"], row["view"]
        image_rel = Path("01_data/01_raw/final_data") / tree_id / "before" / f"{view}.jpg"
        truth_path = annotation_root / sample_id / "cut_lines.json"
        reason = ""
        if sample_id in excluded:
            reason = "derived_exclusion_register"
        elif not (PROJECT_ROOT / image_rel).exists():
            reason = "missing_before_image"
        elif not truth_path.exists():
            reason = "missing_cut_truth"
        if reason:
            derived_rows.append({"sample_id": sample_id, "tree_id": tree_id, "view": view, "complete_by_user": False, "included": False, "reason": reason, "cut_truth_sha256": ""})
            continue
        digest = sha256_file(truth_path)
        accepted.append({"sample_id": f"{tree_id}_before_{view}", "image_path": image_rel.as_posix(), "category": "complete_manual_cut_truth", "selection_evidence": f"read-only cut truth {sample_id}; sha256={digest}"})
        derived_rows.append({"sample_id": sample_id, "tree_id": tree_id, "view": view, "complete_by_user": True, "included": True, "reason": "user_confirmed_complete_20260726", "cut_truth_sha256": digest})
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ensure_dir(output)
    run_manifest = {"schema_version": "1.0", "purpose": "pruning_gnn_current_auto_perception", "frozen_at": datetime.now().isoformat(timespec="seconds"), "defaults": defaults, "samples": accepted}
    (output / "current_pipeline_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "annotation_completeness.csv", derived_rows, ["sample_id", "tree_id", "view", "complete_by_user", "included", "reason", "cut_truth_sha256"])
    summary = {"source_manifest": str(audit_root / "annotation_manifest.csv"), "source_mode": "read_only", "user_confirmation": "all original-image cut annotations complete", "accepted_samples": len(accepted), "accepted_trees": len({item["sample_id"].split("_before_")[0] for item in accepted}), "excluded_samples": len(derived_rows) - len(accepted), "manifest_sha256": sha256_file(output / "current_pipeline_manifest.json")}
    (output / "manifest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
