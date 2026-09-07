from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_atomic_segment_graphs import ensure_dir, sha256_file
from export_pruning_gnn_dataset import segment_features, tree_split, write_rows


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


def semantic_segment_type(segment: dict[str, Any]) -> str | None:
    left = set(map(str, segment.get("start_types", [])))
    right = set(map(str, segment.get("end_types", [])))
    if "root" in left:
        left.add("junction")
    if "root" in right:
        right.add("junction")
    if "bud" in left and "bud" in right:
        return "bud--bud"
    if ("bud" in left and "junction" in right) or ("bud" in right and "junction" in left):
        return "junction--bud"
    if ("bud" in left and "endpoint" in right) or ("bud" in right and "endpoint" in left):
        return "bud--endpoint"
    if ("junction" in left and "endpoint" in right) or ("junction" in right and "endpoint" in left):
        return "junction--endpoint"
    if "junction" in left and "junction" in right:
        return "junction--junction"
    return None


def effective_candidate_type(segment: dict[str, Any]) -> str | None:
    if segment.get("is_trunk_context"):
        return None
    return semantic_segment_type(segment)


def apply_manual_labels(graph: dict[str, Any], annotation: dict[str, Any]) -> tuple[dict[str, Any], Counter, Counter]:
    # The raw cut-truth file is immutable.  GUI add/delete operations are a
    # derived review overlay, applied to a copy of the graph at export time.
    deleted_cut_ids = {str(cut_id) for cut_id in annotation.get("deleted_cut_ids", [])}
    added_cut_lines = [dict(item) for item in annotation.get("added_cut_lines", []) if item.get("cut_id")]
    active_mappings = [
        dict(mapping) for mapping in graph.get("cut_mappings", [])
        if str(mapping.get("cut_id")) not in deleted_cut_ids
    ]
    active_ids = {str(mapping.get("cut_id")) for mapping in active_mappings}
    for cut in added_cut_lines:
        cut_id = str(cut["cut_id"])
        if cut_id in deleted_cut_ids:
            continue
        if cut_id in active_ids:
            raise ValueError(f"Duplicate derived cut id: {cut_id}")
        active_mappings.append({
            "cut_id": cut_id,
            "status": "manual_added",
            "suggested_segment_id": None,
            "intersected_segment_ids": [],
            "candidate_segment_ids": [],
            "intersection_clusters": [],
            "manual_added": True,
            "line": cut,
        })
        active_ids.add(cut_id)
    graph["cut_mappings"] = active_mappings
    assignments = annotation.get("assignments", [])
    cut_ids = {str(mapping["cut_id"]) for mapping in graph["cut_mappings"]}
    assignment_cut_ids = {str(item["cut_id"]) for item in assignments}
    if assignment_cut_ids != cut_ids:
        raise ValueError(f"Incomplete cut assignments: expected={sorted(cut_ids)} observed={sorted(assignment_cut_ids)}")
    segment_ids = [int(item["segment_id"]) for item in assignments]
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("Two cut lines use the same atomic segment")
    selected = set(segment_ids)
    assignment_by_segment = {int(item["segment_id"]): item for item in assignments}
    positives: Counter = Counter()
    negatives: Counter = Counter()
    for segment in graph["segment_nodes"]:
        segment_id = int(segment["id"])
        candidate_type = effective_candidate_type(segment)
        if segment_id in selected and candidate_type is None:
            # A human may correctly identify a pruning branch that the router
            # mislabeled as trunk/context.  Promote only that selected segment;
            # the remaining automatic context stays outside classification loss.
            candidate_type = semantic_segment_type(segment) or assignment_by_segment[segment_id].get("candidate_type")
        is_candidate = candidate_type is not None
        # Normalize the compatibility view before feature/tensor generation.
        segment["candidate_type"] = candidate_type
        segment["is_candidate"] = int(is_candidate)
        if segment_id in selected and not is_candidate:
            raise ValueError(f"Selected segment {segment_id} has no resolvable five-type endpoint semantics")
        segment["is_cut_segment"] = int(segment_id in selected)
        segment["label_mask"] = int(is_candidate)
        segment["cut_ids"] = [str(item["cut_id"]) for item in assignments if int(item["segment_id"]) == segment_id]
        if is_candidate:
            target = positives if segment_id in selected else negatives
            target[str(candidate_type)] += 1
    graph["manual_annotation"] = {
        "schema_version": annotation["schema_version"],
        "status": annotation["status"],
        "graph_sha256": annotation["graph_sha256"],
        "cut_truth_sha256": annotation["cut_truth_sha256"],
        "context_override_segment_ids": sorted(
            int(item["segment_id"]) for item in assignments if item.get("manual_context_override")
        ),
        "deleted_cut_ids": sorted(deleted_cut_ids),
        "added_cut_ids": sorted(str(item["cut_id"]) for item in added_cut_lines),
    }
    return graph, positives, negatives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze manually cleaned pruning atomic segments into GNN tensors")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    output = args.output or dataset / "exports" / "gnn_dataset"
    output = output if output.is_absolute() else PROJECT_ROOT / output
    run_manifest = read_json(dataset / "run_manifest.json")
    expected = [item["sample_id"] for item in run_manifest["samples"]]
    annotation_root = dataset / "manual_annotations"
    graph_root = dataset / "atomic_graphs"
    states: Counter = Counter()
    missing = []
    invalid = []
    accepted = []
    excluded_rows = []
    auto_excluded = []
    frozen_rows = []
    positive_counts: Counter = Counter()
    negative_counts: Counter = Counter()

    for sample_id in expected:
        annotation_path = annotation_root / f"{sample_id}.json"
        graph_path = graph_root / sample_id / "atomic_graph.json"
        complete_path = graph_root / sample_id / ".complete.json"
        audit_path = graph_root / sample_id / "geometry_audit.json"
        if not complete_path.exists():
            invalid.append({"sample_id": sample_id, "reason": "missing_complete_marker"})
            continue
        marker = read_json(complete_path)
        if marker.get("status") == "auto_excluded":
            exclusion = marker.get("exclusion", {}) or {}
            reason = exclusion.get("reason", "auto_excluded")
            auto_excluded.append(sample_id)
            states["auto_excluded"] += 1
            excluded_rows.append({"sample_id": sample_id, "reason": reason, "note": "automatic structural quality gate"})
            frozen_rows.append({
                "sample_id": sample_id,
                "status": "auto_excluded",
                "graph_sha256": "",
                "annotation_sha256": sha256_file(complete_path),
            })
            continue
        if marker.get("status") not in {"complete", "manual_quality_review"}:
            invalid.append({"sample_id": sample_id, "reason": "unexpected_complete_marker_status"})
            continue
        if not annotation_path.exists():
            missing.append(sample_id)
            states["missing"] += 1
            continue
        annotation = read_json(annotation_path)
        status = str(annotation.get("status", "invalid"))
        states[status] += 1
        if not graph_path.exists() or not complete_path.exists() or not audit_path.exists():
            invalid.append({"sample_id": sample_id, "reason": "missing_complete_graph_artifacts"})
            continue
        if not read_json(audit_path).get("valid"):
            invalid.append({"sample_id": sample_id, "reason": "graph_not_complete_or_geometry_invalid"})
            continue
        observed_graph_hash = sha256_file(graph_path)
        if annotation.get("graph_sha256") != observed_graph_hash:
            invalid.append({"sample_id": sample_id, "reason": "graph_hash_mismatch"})
            continue
        graph = read_json(graph_path)
        if annotation.get("cut_truth_sha256") != graph["source"]["cut_truth_sha256"]:
            invalid.append({"sample_id": sample_id, "reason": "cut_truth_hash_mismatch"})
            continue
        if status == "excluded":
            exclusion = annotation.get("exclusion") or {}
            excluded_rows.append({"sample_id": sample_id, "reason": exclusion.get("reason", "unknown"), "note": exclusion.get("note", "")})
            frozen_rows.append({"sample_id": sample_id, "status": status, "graph_sha256": observed_graph_hash, "annotation_sha256": sha256_file(annotation_path)})
            continue
        if status != "accepted":
            continue
        try:
            graph, positives, negatives = apply_manual_labels(graph, annotation)
        except Exception as exc:
            invalid.append({"sample_id": sample_id, "reason": repr(exc)})
            continue
        positive_counts.update(positives)
        negative_counts.update(negatives)
        graph_output = ensure_dir(output / "graphs" / sample_id)
        atomic_json(graph_output / "decision_graph.json", graph)
        x, edge_index, y, mask, feature_rows = segment_features(graph, Path(graph["source"]["image"]))
        torch.save({"sample_id": sample_id, "tree_id": sample_id.split("_before_")[0], "x": x, "edge_index": edge_index, "y": y, "label_mask": mask}, graph_output / "graph.pt")
        write_rows(graph_output / "segment_features.csv", feature_rows)
        accepted.append(sample_id)
        frozen_rows.append({"sample_id": sample_id, "status": status, "graph_sha256": observed_graph_hash, "annotation_sha256": sha256_file(annotation_path)})

    gate_complete = (
        not missing
        and not invalid
        and states.get("in_progress", 0) == 0
        and len(accepted) + (len(excluded_rows) - len(auto_excluded)) + len(auto_excluded) == len(expected)
    )
    audit = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "data_preparation" if gate_complete else "exploratory_partial",
        "expected_samples": len(expected),
        "state_counts": dict(states),
        "accepted_samples": len(accepted),
        "manually_excluded_samples": len(excluded_rows) - len(auto_excluded),
        "auto_excluded_samples": len(auto_excluded),
        "excluded_samples_total": len(excluded_rows),
        "manual_reviews_expected": len(expected) - len(auto_excluded),
        "missing_annotations": missing,
        "invalid_annotations": invalid,
        "positive_segments_by_type": dict(sorted(positive_counts.items())),
        "negative_segments_by_type": dict(sorted(negative_counts.items())),
        "freeze_gate_complete": gate_complete,
    }
    ensure_dir(output / "audit")
    atomic_json(output / "audit" / "summary.json", audit)
    write_csv(output / "audit" / "exclusions.csv", excluded_rows, ["sample_id", "reason", "note"])
    write_csv(output / "audit" / "frozen_manifest.csv", frozen_rows, ["sample_id", "status", "graph_sha256", "annotation_sha256"])
    write_csv(output / "audit" / "invalid_annotations.csv", invalid, ["sample_id", "reason"])
    if not gate_complete and not args.allow_partial:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 2

    splits = tree_split(sorted({sample_id.split("_before_")[0] for sample_id in accepted}))
    ensure_dir(output / "splits")
    for name, tree_ids in splits.items():
        atomic_json(output / "splits" / f"{name}_trees.json", {"tree_ids": tree_ids})
    audit["tree_splits"] = {name: len(tree_ids) for name, tree_ids in splits.items()}
    atomic_json(output / "audit" / "summary.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
