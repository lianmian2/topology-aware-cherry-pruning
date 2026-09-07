from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_atomic_segment_graphs import GRAPH_SCHEMA_VERSION, ensure_dir, sha256_file


REQUIRED_ANNOTATABLE_ARTIFACTS = (
    "atomic_graph.json",
    "geometry_audit.json",
    "segment_overlay.jpg",
    "cut_label_overlay.jpg",
    "decision_graph.json",
)


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a completed pruning-segment manual-cleaning workspace")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p05": percentile(0.05),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def main() -> int:
    args = parse_args()
    dataset = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    dataset = dataset.resolve()
    output = args.output or dataset / "exports" / "pre_annotation_audit_20260801"
    output = output if output.is_absolute() else PROJECT_ROOT / output
    output = output.resolve()

    run_manifest_path = dataset / "run_manifest.json"
    progress_path = dataset / "progress.json"
    run_manifest = read_json(run_manifest_path)
    progress = read_json(progress_path)
    expected_rows = run_manifest.get("samples", [])
    expected = {str(item["sample_id"]): item for item in expected_rows}
    graph_root = dataset / "atomic_graphs"
    annotation_root = dataset / "manual_annotations"

    status_counts: Counter[str] = Counter()
    exclusion_reasons: Counter[str] = Counter()
    exclusion_details: Counter[str] = Counter()
    geometry_statuses: Counter[str] = Counter()
    candidate_types: Counter[str] = Counter()
    annotation_states: Counter[str] = Counter()
    operational: dict[str, list[float]] = {
        "total_seconds": [],
        "roi_seconds": [],
        "branch_seconds": [],
        "bud_seconds": [],
        "routing_seconds": [],
        "clip_seconds": [],
        "fusion_graph_feature_seconds": [],
        "peak_gpu_gb": [],
    }
    perception_counts: Counter[str] = Counter()
    cut_counts = Counter(total=0, annotatable=0, auto_excluded=0)
    segment_counts = Counter(total=0, candidate=0, context=0)
    signature_hashes: Counter[str] = Counter()
    sample_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    raw_hash_mismatches: list[dict[str, str]] = []
    annotatable_ids: list[str] = []
    auto_excluded_ids: list[str] = []
    warning_review_ids: list[str] = []
    warning_reasons: Counter[str] = Counter()

    for sample_id, manifest_row in sorted(expected.items()):
        sample_dir = graph_root / sample_id
        marker_path = sample_dir / ".complete.json"
        failed_path = sample_dir / ".failed.json"
        if failed_path.exists():
            invalid.append({"sample_id": sample_id, "reason": "stale_failure_marker"})
        if not marker_path.exists():
            invalid.append({"sample_id": sample_id, "reason": "missing_complete_marker"})
            status_counts["missing"] += 1
            continue
        try:
            marker = read_json(marker_path)
        except Exception as exc:
            invalid.append({"sample_id": sample_id, "reason": f"invalid_complete_marker:{exc!r}"})
            status_counts["invalid_marker"] += 1
            continue

        status = str(marker.get("status", "invalid"))
        status_counts[status] += 1
        signature = marker.get("signature", {})
        signature_hashes[str(signature.get("pipeline_config_sha256", "missing"))] += 1
        expected_signature = manifest_row.get("signature", {})
        for key in ("sample_id", "image_sha256", "cut_truth_sha256", "weights", "graph_schema_version"):
            if signature.get(key) != expected_signature.get(key):
                invalid.append({"sample_id": sample_id, "reason": f"marker_manifest_signature_mismatch:{key}"})

        perception = marker.get("perception_summary", {}) or {}
        timings = perception.get("timings", {}) or {}
        operational["total_seconds"].append(float(timings.get("total_seconds", marker.get("elapsed_seconds", 0.0))))
        operational["roi_seconds"].append(float(timings.get("roi_seconds", 0.0)))
        operational["branch_seconds"].append(float(timings.get("branch_seconds", 0.0)))
        operational["bud_seconds"].append(float(timings.get("bud_seconds", 0.0)))
        operational["routing_seconds"].append(
            float(timings.get("routing_geometry_seconds", 0.0))
            + float(timings.get("bud_flow_rerank_seconds", 0.0))
        )
        operational["clip_seconds"].append(float(timings.get("clip_seconds", 0.0)))
        operational["fusion_graph_feature_seconds"].append(float(timings.get("fusion_graph_feature_seconds", 0.0)))
        operational["peak_gpu_gb"].append(float(perception.get("peak_gpu_memory_bytes", 0.0)) / 1024 ** 3)
        bud_voting = perception.get("bud_voting", {}) or {}
        attachment = perception.get("attachment", {}) or {}
        dag = perception.get("dag", {}) or {}
        perception_counts["buds_detected"] += int(perception.get("buds", 0))
        perception_counts["bud_directions_reliable"] += int(bud_voting.get("directions_reliable", 0))
        perception_counts["bud_evidence_clusters"] += int(bud_voting.get("evidence_clusters", 0))
        perception_counts["bud_supported_root_splits"] += int(bud_voting.get("root_splits", 0))
        perception_counts["bud_supported_cycle_repairs"] += int(bud_voting.get("cycle_repairs", 0))
        perception_counts["buds_attached"] += int(attachment.get("attached", 0))
        perception_counts["attachments_component_aware"] += int(attachment.get("component_aware", 0))
        perception_counts["attachments_fallback"] += int(attachment.get("fallback", 0))
        perception_counts["dag_true"] += int(bool(dag.get("is_dag")))

        tree_id, view = sample_id.split("_before_", 1)
        raw_cut_path = PROJECT_ROOT / "04_results/pruning_truth/manual_cut_lines_20260720_raw" / f"{tree_id}_{view}" / "cut_lines.json"
        if not raw_cut_path.exists():
            raw_hash_mismatches.append({"sample_id": sample_id, "reason": "missing_raw_cut_truth"})
            raw_cut_count = 0
        else:
            observed_cut_hash = sha256_file(raw_cut_path)
            expected_cut_hash = str(signature.get("cut_truth_sha256", ""))
            if observed_cut_hash != expected_cut_hash:
                raw_hash_mismatches.append({"sample_id": sample_id, "reason": "raw_cut_truth_hash_mismatch"})
            raw_cut_count = len(read_json(raw_cut_path).get("cut_lines", []))
        cut_counts["total"] += raw_cut_count

        if status == "auto_excluded":
            auto_excluded_ids.append(sample_id)
            cut_counts["auto_excluded"] += raw_cut_count
            exclusion = marker.get("exclusion", {}) or {}
            reason = str(exclusion.get("reason", "auto_excluded"))
            exclusion_reasons[reason] += 1
            detailed = exclusion.get("binary_v31_quality", {}).get("exclusion_reasons", []) or []
            if not detailed:
                detailed = [reason]
            for item in detailed:
                exclusion_details[str(item)] += 1
            exclusion_rows.append({
                "sample_id": sample_id,
                "tree_id": tree_id,
                "cut_lines": raw_cut_count,
                "reason": reason,
                "detail": ";".join(map(str, detailed)),
            })
        elif status in {"complete", "manual_quality_review"}:
            annotatable_ids.append(sample_id)
            cut_counts["annotatable"] += raw_cut_count
            if status == "manual_quality_review":
                warning_review_ids.append(sample_id)
                warning = marker.get("structural_warning", {}) or {}
                warning_reasons[str(warning.get("reason", "structural_warning"))] += 1
            missing_artifacts = [name for name in REQUIRED_ANNOTATABLE_ARTIFACTS if not (sample_dir / name).exists()]
            if missing_artifacts:
                invalid.append({"sample_id": sample_id, "reason": "missing_artifacts:" + ";".join(missing_artifacts)})
                continue
            try:
                graph_path = sample_dir / "atomic_graph.json"
                graph = read_json(graph_path)
                audit = read_json(sample_dir / "geometry_audit.json")
            except Exception as exc:
                invalid.append({"sample_id": sample_id, "reason": f"invalid_graph_artifact:{exc!r}"})
                continue
            if graph.get("schema_version") != GRAPH_SCHEMA_VERSION:
                invalid.append({"sample_id": sample_id, "reason": "graph_schema_mismatch"})
            if not audit.get("valid") or audit.get("issues"):
                invalid.append({"sample_id": sample_id, "reason": "invalid_atomic_partition"})
            source = graph.get("source", {})
            for source_key in ("image", "skeleton", "attachments", "cut_truth"):
                source_path = Path(str(source.get(source_key, "")))
                if not source_path.exists():
                    invalid.append({"sample_id": sample_id, "reason": f"missing_graph_source:{source_key}"})
            if source.get("cut_truth_sha256") != signature.get("cut_truth_sha256"):
                invalid.append({"sample_id": sample_id, "reason": "graph_cut_truth_hash_mismatch"})
            mappings = graph.get("cut_mappings", [])
            if len(mappings) != raw_cut_count:
                invalid.append({"sample_id": sample_id, "reason": "cut_mapping_count_mismatch"})
            for mapping in mappings:
                geometry_statuses[str(mapping.get("status", "missing"))] += 1
            segments = graph.get("segment_nodes", [])
            segment_counts["total"] += len(segments)
            for segment in segments:
                if segment.get("is_candidate"):
                    segment_counts["candidate"] += 1
                    candidate_types[str(segment.get("candidate_type", "missing"))] += 1
                else:
                    segment_counts["context"] += 1
        else:
            invalid.append({"sample_id": sample_id, "reason": f"unexpected_marker_status:{status}"})

        annotation_path = annotation_root / f"{sample_id}.json"
        annotation_state = "not_required_auto_excluded" if status == "auto_excluded" else "not_started"
        annotation_hash_valid = ""
        if status in {"complete", "manual_quality_review"} and annotation_path.exists():
            try:
                annotation = read_json(annotation_path)
                annotation_state = str(annotation.get("status", "invalid"))
                graph_path = sample_dir / "atomic_graph.json"
                annotation_hash_valid = str(graph_path.exists() and annotation.get("graph_sha256") == sha256_file(graph_path)).lower()
                if annotation_hash_valid != "true":
                    invalid.append({"sample_id": sample_id, "reason": "manual_annotation_graph_hash_mismatch"})
            except Exception as exc:
                annotation_state = "invalid"
                invalid.append({"sample_id": sample_id, "reason": f"invalid_manual_annotation:{exc!r}"})
        annotation_states[annotation_state] += 1
        sample_rows.append({
            "sample_id": sample_id,
            "tree_id": tree_id,
            "marker_status": status,
            "cut_lines": raw_cut_count,
            "annotation_state": annotation_state,
            "annotation_graph_hash_valid": annotation_hash_valid,
        })

    unexpected_dirs = sorted(path.name for path in graph_root.iterdir() if path.is_dir() and path.name not in expected)
    if unexpected_dirs:
        invalid.extend({"sample_id": sample_id, "reason": "unexpected_sample_directory"} for sample_id in unexpected_dirs)

    trees_total = {sample_id.split("_before_")[0] for sample_id in expected}
    annotatable_trees = {sample_id.split("_before_")[0] for sample_id in annotatable_ids}
    excluded_trees = {sample_id.split("_before_")[0] for sample_id in auto_excluded_ids}
    manual_complete = annotation_states.get("accepted", 0) + annotation_states.get("excluded", 0)
    run_complete = (
        progress.get("status") == "complete"
        and len(expected) == sum(status_counts[status] for status in ("complete", "manual_quality_review", "auto_excluded"))
        and not invalid
        and not raw_hash_mismatches
    )
    summary = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "data_preparation",
        "evidence_boundary": "Structural and geometry audit only; not pruning accuracy, bud correctness, or botanical correctness.",
        "dataset": str(dataset),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "progress_sha256": sha256_file(progress_path),
        "source_cut_truth_mode": "read_only",
        "expected_samples": len(expected),
        "expected_trees": len(trees_total),
        "marker_status_counts": dict(sorted(status_counts.items())),
        "annotatable_samples": len(annotatable_ids),
        "annotatable_trees": len(annotatable_trees),
        "auto_excluded_samples": len(auto_excluded_ids),
        "manual_quality_review_samples": len(warning_review_ids),
        "manual_quality_warning_reasons": dict(sorted(warning_reasons.items())),
        "trees_with_at_least_one_auto_excluded_view": len(excluded_trees),
        "trees_with_no_annotatable_view": len(trees_total - annotatable_trees),
        "trees_with_all_manifest_views_annotatable": len(trees_total - excluded_trees),
        "trees_with_mixed_annotatable_and_auto_excluded_views": len(annotatable_trees & excluded_trees),
        "cut_lines": dict(cut_counts),
        "atomic_segments": dict(segment_counts),
        "candidate_segments_by_five_type": dict(sorted(candidate_types.items())),
        "automatic_geometry_statuses": dict(sorted(geometry_statuses.items())),
        "auto_exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "auto_exclusion_details": dict(sorted(exclusion_details.items())),
        "pipeline_config_hashes": dict(sorted(signature_hashes.items())),
        "operational_diagnostics": {key: distribution(values) for key, values in operational.items()},
        "automatic_perception_counts": dict(sorted(perception_counts.items())),
        "manual_annotation_states": dict(sorted(annotation_states.items())),
        "manual_reviews_complete": manual_complete,
        "manual_reviews_required": len(annotatable_ids),
        "invalid_samples": invalid,
        "raw_cut_truth_hash_issues": raw_hash_mismatches,
        "unexpected_sample_directories": unexpected_dirs,
        "ready_for_manual_annotation": run_complete,
        "ready_for_training_export": run_complete and manual_complete == len(annotatable_ids),
    }

    ensure_dir(output)
    atomic_json(output / "summary.json", summary)
    write_csv(output / "sample_audit.csv", sample_rows, [
        "sample_id", "tree_id", "marker_status", "cut_lines", "annotation_state", "annotation_graph_hash_valid",
    ])
    write_csv(output / "auto_exclusions.csv", exclusion_rows, ["sample_id", "tree_id", "cut_lines", "reason", "detail"])
    write_csv(output / "invalid_samples.csv", invalid, ["sample_id", "reason"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if run_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
