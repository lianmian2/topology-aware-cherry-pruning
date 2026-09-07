from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_atomic_segment_graphs import GRAPH_SCHEMA_VERSION, ensure_dir, process_sample, sha256_file
from run_e2e_regression import (
    BUD_DIRECTION_MIN_CONFIDENCE,
    ROUTER_CONFIG,
    ModelManager,
    run_sample,
    validate_manifest,
)
from mask_topology_routing import utils as routing_utils
from mask_topology_routing_binary_v3 import (
    audit_binary_topology_v3,
    load_binary_v3_config,
    refine_clipped_groups_v3,
)
from mask_topology_routing_binary_v3.thickness_routing import make_router_selector


# These two hashes were emitted by the same frozen V3.1 result pipeline before
# and after the Windows progress-writer fix.  Only the orchestration source
# changed; perception, routing, refinement, graph schema, weights and settings
# did not.  They are accepted once and migrated to the result-only signature.
LEGACY_V31_OPERATIONAL_ONLY_PIPELINE_HASHES = {
    "491c1582ff1ae7d9b20e32aaaa3bc4ad9ef79134c61ec17df41cd22969af9e3a",
    "ed3b60bc3b825d3110d68a7f4f8a76bc7c8c30cf8ce4648639d2090c8a0d290f",
}


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: Any, *, replace_timeout_seconds: float = 15.0) -> None:
    ensure_dir(path.parent)
    # A fixed ``progress.json.tmp`` is prone to collisions and BaiduSync/AV file
    # locks on Windows.  Use a unique same-directory temporary file so replace
    # remains atomic, then tolerate a short-lived lock on the destination.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        deadline = time.monotonic() + max(float(replace_timeout_seconds), 0.0)
        delay = 0.05
        while True:
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 1.7, 0.75)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def parse_sample(sample_id: str) -> tuple[str, str]:
    tree_id, view = sample_id.split("_before_", 1)
    return tree_id, view


def source_signature(sample: dict[str, Any], weight_signatures: dict[str, Any], pipeline_config_sha256: str) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    tree_id, view = parse_sample(sample_id)
    image_path = PROJECT_ROOT / sample["image_path"]
    truth_path = PROJECT_ROOT / "04_results/pruning_truth/manual_cut_lines_20260720_raw" / f"{tree_id}_{view}" / "cut_lines.json"
    return {
        "sample_id": sample_id,
        "image_sha256": sha256_file(image_path),
        "cut_truth_sha256": sha256_file(truth_path),
        "weights": weight_signatures,
        "pipeline_config_sha256": pipeline_config_sha256,
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
    }


def complete_matches(path: Path, signature: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if marker.get("status") not in {"complete", "auto_excluded", "manual_quality_review"}:
        return False
    stored = marker.get("signature")
    if stored == signature:
        return True
    if not isinstance(stored, dict):
        return False
    legacy_hash = stored.get("pipeline_config_sha256")
    if legacy_hash not in LEGACY_V31_OPERATIONAL_ONLY_PIPELINE_HASHES:
        return False
    stored_inputs = {key: value for key, value in stored.items() if key != "pipeline_config_sha256"}
    current_inputs = {key: value for key, value in signature.items() if key != "pipeline_config_sha256"}
    if stored_inputs != current_inputs:
        return False

    # Migrate only metadata.  The sample artifacts are untouched.
    marker["legacy_signature"] = stored
    marker["signature"] = signature
    marker["signature_migrated_at"] = datetime.now().isoformat(timespec="seconds")
    marker["signature_migration_reason"] = "operational_progress_writer_change_only"
    atomic_json(path, marker)
    return True


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger("pruning_segment_cleaning")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(ensure_dir(output / "logs") / "preprocess.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare automatic perception and atomic graphs for manual pruning-segment cleaning")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--sample-id", action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    auto_root = ensure_dir(output / "auto_perception")
    graph_root = ensure_dir(output / "atomic_graphs")
    for name in ("manual_annotations", "exclusions", "structural_warnings", "exports", "logs"):
        ensure_dir(output / name)
    (output / "README.md").write_text(
        "# 剪枝原子段人工清洗数据集\n\n"
        "本目录是派生数据准备产物，原始剪切线目录保持只读。\n\n"
        "- `auto_perception/`：ROI、枝条掩码、全图芽点、骨架分组和芽点附着。\n"
        "- `atomic_graphs/`：五类原子段图、几何审计、叠加图和完成/失败标记。\n"
        "- `manual_annotations/`：人工点击确认结果；自动建议不是真值。\n"
        "- `exports/`：全部人工审核完成后生成的 GNN 数据。\n"
        "- `progress.json`、`run_manifest.json`、`logs/`：续跑进度与来源审计。\n\n"
        "候选颜色：绿=分叉—芽，黄=芽—芽，蓝=芽—端点，橙=分叉—端点，紫=分叉—分叉，灰=主干/上下文。\n\n"
        "启动界面：`07_graphical_interface/run_pruning_segment_cleaning_gui.bat`。悬浮会加粗并放大线段，左键确认当前剪切线；无法表达的骨架粘连样本按 E 整图排除。\n",
        encoding="utf-8",
    )
    logger = setup_logger(output)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Manifest: %s", manifest_path)
    logger.info("Output: %s", output)
    logger.info("Device: %s | resume=%s | retry_failed=%s", args.device, args.resume, args.retry_failed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("CUDA device requested but torch.cuda.is_available() is false")
        return 2
    if args.device.startswith("cuda"):
        device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_index)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        logger.info(
            "CUDA confirmed: %s | total_memory=%.2f GB | bud_patch_batch=%s",
            torch.cuda.get_device_name(device_index),
            torch.cuda.get_device_properties(device_index).total_memory / 1024 ** 3,
            __import__("os").environ.get("CHERRY_BUD_BATCH_SIZE", "8"),
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binary_v3_config = load_binary_v3_config()
    thickness_rulebook = (
        PROJECT_ROOT
        / "04_results/mask_topology_routing/gt_mask_thickness_rulebook_20260731/gt_mask_thickness_rulebook.json"
    )
    if not thickness_rulebook.is_file():
        raise FileNotFoundError(thickness_rulebook)
    thickness_selector = make_router_selector(thickness_rulebook, binary_v3_config.rulebook_path)
    routing_utils._choose_trunk_path_on_skeleton = thickness_selector

    def v31_refiner(groups, processed_mask):
        return refine_clipped_groups_v3(groups, processed_mask, binary_v3_config)

    def v31_quality(groups, processed_mask, clipped_groups):
        return audit_binary_topology_v3(
            groups, processed_mask, binary_v3_config,
            baseline_groups=clipped_groups, include_global_checks=True,
        )
    validation = validate_manifest(manifest)
    atomic_json(output / "manifest_validation.json", validation)
    if not validation["valid"]:
        logger.error("Manifest or frozen model validation failed")
        return 2

    selected = set(args.sample_id or [])
    samples = [sample for sample in manifest["samples"] if not selected or sample["sample_id"] in selected]
    unknown = selected - {sample["sample_id"] for sample in samples}
    if unknown:
        raise ValueError(f"Unknown sample IDs: {sorted(unknown)}")

    weight_signatures = {}
    for stage in ("roi", "branch", "bud"):
        weight_path = PROJECT_ROOT / manifest["defaults"][f"{stage}_weight"]
        weight_signatures[stage] = {"path": str(weight_path), "sha256": sha256_file(weight_path)}
    result_pipeline_sources = {
        "e2e": SCRIPT_DIR / "run_e2e_regression.py",
        "routing": PROJECT_ROOT / "02_code" / "02_models" / "mask_topology_routing" / "utils.py",
        "bud_orientation": PROJECT_ROOT / "02_code" / "02_models" / "bud_skeleton_fusion" / "bud_orientation.py",
        "atomic_graph": SCRIPT_DIR / "build_atomic_segment_graphs.py",
        "binary_v3_config": PROJECT_ROOT / "02_code/02_models/mask_topology_routing_binary_v3/config.py",
        "binary_v3_refinement": PROJECT_ROOT / "02_code/02_models/mask_topology_routing_binary_v3/refinement.py",
        "binary_v3_quality": PROJECT_ROOT / "02_code/02_models/mask_topology_routing_binary_v3/quality.py",
        "thickness_routing": PROJECT_ROOT / "02_code/02_models/mask_topology_routing_binary_v3/thickness_routing.py",
    }
    result_pipeline_source_hashes = {name: sha256_file(path) for name, path in result_pipeline_sources.items()}
    operational_source_hashes = {
        "preparation": sha256_file(SCRIPT_DIR / "prepare_pruning_segment_cleaning.py"),
        "vscode_launcher": sha256_file(SCRIPT_DIR / "run_pruning_segment_cleaning_vscode.py"),
    }
    pipeline_configuration = {
        "defaults": manifest["defaults"],
        "router_config": ROUTER_CONFIG,
        "bud_direction_min_confidence": BUD_DIRECTION_MIN_CONFIDENCE,
        "routing_mode": "two_pass_geometry_then_bud_flow",
        "routing_release": "binary_v3.1_stable_thickness_budflow_v3_refinement",
        "hierarchy_partition_constraints": False,
        "shared_corridor_auto_rewrite": False,
        "binary_v3_rulebook_sha256": sha256_file(binary_v3_config.rulebook_path),
        "thickness_rulebook_sha256": sha256_file(thickness_rulebook),
        "bud_patch_batch": int(__import__("os").environ.get("CHERRY_BUD_BATCH_SIZE", "8")),
        "pipeline_source_hashes": result_pipeline_source_hashes,
    }
    pipeline_config_sha256 = stable_hash(pipeline_configuration)
    signatures = {sample["sample_id"]: source_signature(sample, weight_signatures, pipeline_config_sha256) for sample in samples}
    run_manifest = {
        "schema_version": "1.0",
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "purpose": "manual_pruning_segment_cleaning_data_preparation",
        "result_class": "data_preparation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "raw_cut_truth_mode": "read_only",
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "device": args.device,
        "defaults": manifest["defaults"],
        "pipeline_configuration": pipeline_configuration,
        "operational_source_hashes": operational_source_hashes,
        "samples": [{"sample_id": sample["sample_id"], "signature": signatures[sample["sample_id"]]} for sample in samples],
    }
    atomic_json(output / "run_manifest.json", run_manifest)

    manager = ModelManager()
    manager.device = args.device
    completed: list[str] = []
    skipped: list[str] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    processed_elapsed = 0.0
    total = len(samples)

    def save_progress(current: str | None = None) -> None:
        elapsed = time.perf_counter() - started
        done = len(completed) + len(skipped) + len(failures)
        processed = len(completed) + len(failures)
        remaining_to_process = max(total - len(skipped) - processed, 0)
        eta = processed_elapsed / processed * remaining_to_process if processed else None
        try:
            atomic_json(output / "progress.json", {
                "status": "complete" if done == total and not failures else "running" if done < total else "completed_with_failures",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "current_sample": current,
                "samples_total": total,
                "samples_completed": len(completed),
                "samples_skipped_resume": len(skipped),
                "samples_failed": len(failures),
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "completed": completed,
                "skipped": skipped,
                "failures": failures,
            })
        except PermissionError as exc:
            # progress.json is monitoring metadata.  A sync-client lock must
            # never discard an otherwise valid and expensive GPU sample.
            logger.warning("PROGRESS_WRITE_DEFERRED | %s", exc)

    save_progress()
    try:
        for index, sample in enumerate(samples, start=1):
            sample_id = sample["sample_id"]
            signature = signatures[sample_id]
            marker = graph_root / sample_id / ".complete.json"
            failure_marker = graph_root / sample_id / ".failed.json"
            if args.resume and complete_matches(marker, signature):
                skipped.append(sample_id)
                logger.info(
                    "PROGRESS %d/%d (%.1f%%) | resumed=%d | completed=%d | failed=%d | %s",
                    index, total, index / total * 100.0, len(skipped), len(completed), len(failures), sample_id,
                )
                save_progress()
                continue
            if args.resume and failure_marker.exists() and not args.retry_failed:
                failures.append({"sample_id": sample_id, "error": "previous_failure_not_retried"})
                logger.warning("[%d/%d] previous failure skipped %s; use --retry-failed", index, total, sample_id)
                save_progress()
                continue
            sample_started = time.perf_counter()
            logger.info("START %d/%d (%.1f%%) | %s", index, total, (index - 1) / total * 100.0, sample_id)
            save_progress(sample_id)
            try:
                perception = run_sample(
                    sample, manifest["defaults"], auto_root, manager,
                    postclip_refiner=v31_refiner,
                    postclip_quality_auditor=v31_quality,
                )
                routing_quality = perception.get("routing_quality", {})
                v31_quality_result = perception.get("postclip_quality") or {}
                if routing_quality.get("status") != "pass" or v31_quality_result.get("status") == "excluded":
                    # These diagnostics describe global routed-graph quality,
                    # not whether each manually annotated cut can be expressed
                    # by one local five-type segment.  Build the atomic graph
                    # and expose the sample with a warning; the annotator makes
                    # the final whole-view exclusion decision.
                    graph_quality = process_sample(auto_root, graph_root, sample_id, None)
                    elapsed = time.perf_counter() - sample_started
                    processed_elapsed += elapsed
                    warning = {
                        "schema_version": "1.0",
                        "sample_id": sample_id,
                        "status": "requires_manual_quality_review",
                        "reason": (
                            routing_quality.get("status")
                            if routing_quality.get("status") != "pass"
                            else "binary_v31_quality_excluded"
                        ),
                        "routing_quality": routing_quality,
                        "binary_v31_quality": v31_quality_result,
                        "source_cut_truth_mode": "read_only",
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    atomic_json(output / "structural_warnings" / f"{sample_id}.json", warning)
                    atomic_json(marker, {
                        "status": "manual_quality_review",
                        "completed_at": datetime.now().isoformat(timespec="seconds"),
                        "signature": signature,
                        "perception_summary": perception,
                        "structural_warning": warning,
                        "graph_quality": graph_quality,
                        "elapsed_seconds": elapsed,
                    })
                    if failure_marker.exists():
                        failure_marker.unlink()
                    completed.append(sample_id)
                    logger.warning("MANUAL_QUALITY_REVIEW %s | %s", sample_id, routing_quality)
                    save_progress()
                    continue
                graph_quality = process_sample(auto_root, graph_root, sample_id, None)
                elapsed = time.perf_counter() - sample_started
                processed_elapsed += elapsed
                atomic_json(marker, {
                    "status": "complete",
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                    "signature": signature,
                    "perception_summary": perception,
                    "graph_quality": graph_quality,
                    "elapsed_seconds": elapsed,
                })
                if failure_marker.exists():
                    failure_marker.unlink()
                completed.append(sample_id)
                processed = len(completed) + len(failures)
                remaining_to_process = max(total - len(skipped) - processed, 0)
                eta = processed_elapsed / processed * remaining_to_process
                logger.info(
                    "PROGRESS %d/%d (%.1f%%) | completed=%d | resumed=%d | failed=%d | sample=%s | sample_time=%s | elapsed=%s | ETA=%s",
                    index, total, index / total * 100.0, len(completed), len(skipped), len(failures), sample_id,
                    format_duration(elapsed), format_duration(time.perf_counter() - started), format_duration(eta),
                )
                logger.info(
                    "GPU %s | peak=%.2f GB | roi=%.1fs branch=%.1fs bud=%.1fs routing=%.1fs",
                    args.device,
                    float(perception.get("peak_gpu_memory_bytes", 0)) / 1024 ** 3,
                    float(perception.get("timings", {}).get("roi_seconds", 0.0)),
                    float(perception.get("timings", {}).get("branch_seconds", 0.0)),
                    float(perception.get("timings", {}).get("bud_seconds", 0.0)),
                    float(perception.get("timings", {}).get("routing_geometry_seconds", 0.0))
                    + float(perception.get("timings", {}).get("bud_flow_rerank_seconds", 0.0)),
                )
            except Exception as exc:
                processed_elapsed += time.perf_counter() - sample_started
                error = repr(exc)
                failures.append({"sample_id": sample_id, "error": error})
                atomic_json(failure_marker, {"status": "failed", "failed_at": datetime.now().isoformat(timespec="seconds"), "signature": signature, "error": error})
                logger.exception("[%d/%d] failed %s", index, total, sample_id)
            save_progress()
    except KeyboardInterrupt:
        save_progress()
        logger.warning("Interrupted; rerun with --resume")
        return 130

    save_progress()
    logger.info(
        "FINISHED | completed=%d | resumed=%d | failed=%d | total=%d | elapsed=%s",
        len(completed), len(skipped), len(failures), total, format_duration(time.perf_counter() - started),
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
