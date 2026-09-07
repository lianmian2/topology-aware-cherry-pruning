"""Build a local, non-destructive candidate public reproducibility package.

This script never alters a source image, annotation, model, or result.  It is
deliberately conservative: unresolved provenance is recorded as a blocker
rather than silently repaired or omitted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "04_results" / "public_reproducibility_preflight_20260907"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_jpeg_metadata(source: Path, destination: Path) -> None:
    """Copy JPEG entropy data unchanged while dropping EXIF/XMP/IPTC/comments."""
    data = source.read_bytes()
    if not data.startswith(b"\xff\xd8"):
        raise ValueError(f"Not a JPEG: {source}")
    result = bytearray(data[:2])
    pos = 2
    drop_markers = {0xE1, 0xED, 0xFE}  # APP1, APP13, COM
    while pos < len(data):
        if data[pos] != 0xFF:
            raise ValueError(f"Unexpected JPEG marker boundary in {source}")
        marker_start = pos
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            raise ValueError(f"Truncated JPEG marker in {source}")
        marker = data[pos]
        pos += 1
        if marker == 0xD9:  # EOI
            result.extend(data[marker_start:pos])
            break
        if marker == 0xDA:  # SOS; subsequent bytes are entropy-coded payload
            result.extend(data[marker_start:])
            break
        if marker in set(range(0xD0, 0xD8)) | {0x01}:
            result.extend(data[marker_start:pos])
            continue
        if pos + 2 > len(data):
            raise ValueError(f"Truncated JPEG segment in {source}")
        length = int.from_bytes(data[pos:pos + 2], "big")
        end = pos + length
        if length < 2 or end > len(data):
            raise ValueError(f"Invalid JPEG segment length in {source}")
        if marker not in drop_markers:
            result.extend(data[marker_start:end])
        pos = end
    else:
        raise ValueError(f"JPEG has no EOI marker: {source}")
    destination.write_bytes(result)


def source_image_path(sample_id: str) -> Path:
    parts = sample_id.split("_")
    if len(parts) != 5 or parts[0] != "tree" or parts[2] not in {"before", "after"}:
        raise ValueError(f"Unsupported final-data sample identifier: {sample_id}")
    return ROOT / "01_data" / "01_raw" / "final_data" / f"tree_{parts[1]}" / parts[2] / f"view_{parts[4]}.jpg"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_tree(source: Path, destination: Path) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in {"__pycache__", ".pytest_cache"} or name.endswith((".pyc", ".pyo"))}

    shutil.copytree(source, destination, ignore=ignore, dirs_exist_ok=False)


def copy_code_tree(source: Path, destination: Path) -> None:
    allowed = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".sh", ".ps1", ".bat"}
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        target = destination / path.relative_to(source)
        ensure_dir(target.parent)
        shutil.copy2(path, target)


def task_sources() -> tuple[list[dict[str, str]], dict[str, Path], dict[str, str]]:
    records: list[dict[str, str]] = []
    canonical_by_id: dict[str, Path] = {}
    tape_by_name: dict[str, str] = {}

    photo_dir = ROOT / "01_data" / "02_annotated" / "pre_lable_50" / "Photo"
    for image in sorted(photo_dir.glob("*.jpg")):
        sample_id = image.stem.split("_", 1)[1]
        records.append({"task": "roi_woody_and_bud", "source_id": sample_id, "source_path": str(image)})
        canonical_by_id[sample_id] = image

    topology_dir = ROOT / "01_data" / "02_annotated" / "skeleton_annotation"
    for annotation in sorted(topology_dir.glob("*_skeleton.json")):
        sample_id = annotation.stem.removesuffix("_skeleton")
        image = source_image_path(sample_id)
        records.append({"task": "topology", "source_id": sample_id, "source_path": str(image)})
        canonical_by_id.setdefault(sample_id, image)

    audit = ROOT / "04_results" / "pruning_decision" / "pruning_decision_closeout_20260809" / "features_v1" / "audit" / "per_sample.csv"
    with audit.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            image = source_image_path(sample_id)
            records.append({"task": "pruning_and_candidate_ranking", "source_id": sample_id, "source_path": str(image)})
            canonical_by_id.setdefault(sample_id, image)

    search_roots = [
        ROOT / "01_data" / "03_processed" / "train" / "tape",
        ROOT / "01_data" / "03_processed" / "train" / "trunk_extracted" / "Trunk",
        ROOT / "01_data" / "02_annotated" / "legacy_cloud_annotations" / "tree_003_after",
        ROOT / "01_data" / "03_processed" / "train" / "skeleton_prediction" / "images",
        ROOT / "01_data" / "03_processed" / "test" / "skeleton_prediction" / "images",
    ]
    image_index: dict[str, Path] = {}
    for directory in search_roots:
        for image in directory.rglob("*.jpg"):
            image_index.setdefault(image.name, image)
    tape_coco = load_json(ROOT / "01_data" / "03_processed" / "annotations" / "mmdet" / "train" / "Tape_merged_dedup.json")
    for item in tape_coco["images"]:
        name = item["file_name"]
        if name not in image_index:
            raise FileNotFoundError(f"Tape source image not found: {name}")
        image = image_index[name]
        records.append({"task": "tape_support", "source_id": name, "source_path": str(image)})
        tape_by_name[name] = str(image)
    return records, canonical_by_id, tape_by_name


def copy_bud_patches(destination: Path) -> dict[str, int]:
    source_root = ROOT / "01_data" / "compag" / "bud_patches"
    index = {path.name: path for path in source_root.rglob("*.jpg")}
    copied: dict[str, int] = {}
    for split, json_name in (("train", "train_bud_final.json"), ("test", "test_bud_final.json")):
        coco = load_json(ROOT / "01_data" / "compag" / "bud_annotations" / json_name)
        target = ensure_dir(destination / split)
        missing = []
        for image in coco["images"]:
            name = image["file_name"]
            path = index.get(name)
            if path is None:
                missing.append(name)
            else:
                shutil.copy2(path, target / name)
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} {split} bud patches; first: {missing[:3]}")
        write_json(destination.parents[2] / "annotations" / "bud" / json_name, coco)
        copied[split] = len(coco["images"])
    shutil.copy2(source_root / "manifest.json", destination.parents[2] / "annotations" / "bud" / "source_patch_manifest.json")
    return copied


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate package: {OUT}")
    ensure_dir(OUT)
    for relative in [
        "data/images_sanitized", "data/manifests", "data/annotations/roi_woody", "data/annotations/topology",
        "data/annotations/tape", "data/annotations/bud", "data/annotations/pruning", "data/derived/bud_patches",
        "data/splits", "checkpoints", "results", "code", "environment", "checksums", "docs",
    ]:
        ensure_dir(OUT / relative)

    records, canonical_by_id, tape_by_name = task_sources()
    by_hash: dict[str, dict[str, Any]] = {}
    for record in records:
        source = Path(record["source_path"])
        if not source.exists():
            raise FileNotFoundError(source)
        digest = sha256(source)
        item = by_hash.setdefault(digest, {"source": source, "records": []})
        item["records"].append(record)

    safe_by_source_id: dict[str, str] = {}
    safe_by_tape_name: dict[str, str] = {}
    membership_rows = []
    for digest, item in sorted(by_hash.items()):
        safe_name = f"rgb_{digest[:16]}.jpg"
        destination = OUT / "data" / "images_sanitized" / safe_name
        strip_jpeg_metadata(item["source"], destination)
        tasks = sorted({record["task"] for record in item["records"]})
        source_ids = sorted({record["source_id"] for record in item["records"]})
        for source_id in source_ids:
            safe_by_source_id[source_id] = safe_name
        for record in item["records"]:
            if record["task"] == "tape_support":
                safe_by_tape_name[record["source_id"]] = safe_name
        membership_rows.append({
            "sanitized_file": f"data/images_sanitized/{safe_name}",
            "sanitized_sha256": sha256(destination),
            "source_sha256": digest,
            "tasks": ";".join(tasks),
            "source_ids": ";".join(source_ids),
            "source_count": len(item["records"]),
        })

    with (OUT / "data" / "manifests" / "image_task_membership.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(membership_rows[0]))
        writer.writeheader()
        writer.writerows(membership_rows)

    roi_source = ROOT / "01_data" / "02_annotated" / "pre_lable_50" / "trunk_cleaned"
    for annotation in sorted(roi_source.glob("*.json")):
        payload = load_json(annotation)
        sample_id = annotation.stem.split("_", 1)[1] if "_" in annotation.stem else annotation.stem
        if sample_id in safe_by_source_id:
            payload["imagePath"] = f"../../images_sanitized/{safe_by_source_id[sample_id]}"
        write_json(OUT / "data" / "annotations" / "roi_woody" / annotation.name, payload)
    shutil.copy2(ROOT / "01_data" / "compag" / "roi_prepared" / "roi_branch_trunk_cleaned_dilate50_100_x10_parallel" / "manifest.json", OUT / "data" / "splits" / "roi_woody_augmentation_manifest.json")

    for annotation in sorted((ROOT / "01_data" / "02_annotated" / "skeleton_annotation").glob("*_skeleton.json")):
        payload = load_json(annotation)
        sample_id = annotation.stem.removesuffix("_skeleton")
        payload["image_path"] = f"../../images_sanitized/{safe_by_source_id[sample_id]}"
        write_json(OUT / "data" / "annotations" / "topology" / annotation.name, payload)
    shutil.copy2(ROOT / "04_results" / "mask_topology_routing" / "trunk_anchored_thinning_vs_routing_gt100_20260817" / "source_manifest.csv", OUT / "data" / "splits" / "topology_source_manifest.csv")

    tape_coco = load_json(ROOT / "01_data" / "03_processed" / "annotations" / "mmdet" / "train" / "Tape_merged_dedup.json")
    for image in tape_coco["images"]:
        image["file_name"] = f"../../images_sanitized/{safe_by_tape_name[image['file_name']]}"
    write_json(OUT / "data" / "annotations" / "tape" / "Tape_merged_dedup_rebased.json", tape_coco)
    shutil.copy2(ROOT / "01_data" / "03_processed" / "annotations" / "mmdet" / "train" / "Tape_merge_report.json", OUT / "data" / "annotations" / "tape" / "merge_report.json")

    bud_counts = copy_bud_patches(OUT / "data" / "derived" / "bud_patches")

    pruning_root = ROOT / "04_results" / "pruning_decision" / "pruning_decision_closeout_20260809"
    for source in [
        pruning_root / "features_v1" / "features.csv",
        pruning_root / "features_v1" / "feature_schema.json",
        pruning_root / "features_v1" / "audit" / "per_sample.csv",
        pruning_root / "features_v1" / "audit" / "summary.json",
        pruning_root / "tree_cv_v1" / "fold_manifest.json",
        pruning_root / "tree_cv_v1" / "metrics_by_fold.csv",
        pruning_root / "tree_cv_v1" / "out_of_fold_predictions.csv",
        pruning_root / "region_level_v2_strict_20260812" / "fold_manifest.json",
        pruning_root / "region_level_v2_strict_20260812" / "summary.json",
        pruning_root / "paper_result_summary.json",
    ]:
        destination = OUT / ("data/annotations/pruning" if "features_v1" in str(source) else "results/pruning") / source.name
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)
    copy_tree(pruning_root / "features_v1" / "graphs", OUT / "data" / "annotations" / "pruning" / "graphs")
    expert_source = pruning_root / "expert_review_analysis_20260822"
    if expert_source.exists():
        copy_tree(expert_source, OUT / "results" / "expert_review_deidentified")

    weights = [
        ROOT / "03_models" / "roi" / "epoch_12.pth",
        ROOT / "04_results" / "cloud_handoffs" / "WoodyDA-WoodyCAR_Ablation_Results_20260807_full" / "WoodyDA-WoodyCAR_Ablation_Results_20260807" / "runs" / "branchseg_csnet_v2_channel_only_channel_only_roi_20260807_s42" / "checkpoints" / "best_primary.pth",
        ROOT / "03_models" / "compag" / "bud_baseline_deloc_patch" / "best.pth",
        ROOT / "03_models" / "Tape_Segmentation_V2_outputs" / "best_model.pth",
        pruning_root / "baselines_v1" / "models" / "hgb_full.joblib",
        *sorted((pruning_root / "gnn_sage_branch_group_v1" / "checkpoints").glob("sage_full_seed*.pt")),
    ]
    checkpoint_manifest = []
    for weight in weights:
        target = OUT / "checkpoints" / weight.name
        if target.exists():
            target = OUT / "checkpoints" / f"{weight.parent.parent.name}_{weight.name}"
        shutil.copy2(weight, target)
        checkpoint_manifest.append({"file": target.relative_to(OUT).as_posix(), "sha256": sha256(target), "size_bytes": target.stat().st_size})
    write_json(OUT / "checkpoints" / "checkpoint_manifest.json", checkpoint_manifest)

    code_sources = [
        ROOT / "02_code" / "02_models" / "branch_seg_csnet_v2",
        ROOT / "02_code" / "02_models" / "mask_topology_routing",
        ROOT / "02_code" / "02_models" / "mask_topology_routing_binary_v3",
        ROOT / "02_code" / "02_models" / "bud_skeleton_fusion",
        ROOT / "02_code" / "02_models" / "tape_segmentation_v2",
        ROOT / "02_code" / "05_utils" / "pruning_truth",
        ROOT / "07_graphical_interface" / "unified_system",
    ]
    for source in code_sources:
        copy_code_tree(source, OUT / "code" / source.name)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    shutil.copy2(ROOT / "04_results" / "pruning_decision" / "runtime_profiling_woodyca_20260903" / "run_manifest.json", OUT / "environment" / "runtime_manifest.json")

    (OUT / "README.md").write_text(
        "# Candidate public reproducibility package\n\n"
        "This local package is a pre-publication candidate. It has not been uploaded or assigned a DOI. "
        "Images were copied without recompression while EXIF/XMP/IPTC/comment JPEG metadata segments were removed. "
        "See `docs/PREPUBLICATION_BLOCKERS.md` before public release.\n",
        encoding="utf-8",
    )
    (OUT / "docs" / "PREPUBLICATION_BLOCKERS.md").write_text(
        "# Items requiring resolution before publication\n\n"
        "- Confirm institutional and orchard-owner permission to release all sanitized images and annotations.\n"
        "- Select and approve separate code, data, and model-weight licences.\n"
        "- Review the tape-support tree-level split regenerated from seed 42 before treating it as the published split.\n"
        "- Match the final 185 pruning samples to portable manual cut-line source annotations and rebase their image paths.\n"
        "- Manually inspect sanitized images for faces, signage, vehicle plates, and location-revealing content.\n"
        "- Audit third-party pretrained-weight licences before distributing fine-tuned checkpoints.\n"
        "- Create GitHub and Zenodo releases, obtain a DOI, then update the manuscript Data Availability statement.\n",
        encoding="utf-8",
    )
    (OUT / "environment" / "REQUIREMENTS_TO_FREEZE.md").write_text(
        "The runtime manifest records PyTorch 2.1.0+cu118, CUDA 11.8 and the measured desktop hardware. "
        "A clean environment lockfile and smoke-test run remain required before publication.\n",
        encoding="utf-8",
    )

    checksums = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256(path)}  {path.relative_to(OUT).as_posix()}")
    (OUT / "checksums" / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    report = {
        "status": "local_prepublication_candidate",
        "unique_sanitized_rgb_images": len(by_hash),
        "source_image_records": len(records),
        "bud_patches": bud_counts,
        "model_checkpoints": len(checkpoint_manifest),
        "manual_pruning_source_annotation_subset": "pending matched portable-annotation export",
        "tape_split": "must be regenerated and independently audited from seed-42 algorithm before public release",
        "manuscript_data_availability": "not updated; DOI does not yet exist",
    }
    write_json(OUT / "preflight_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
