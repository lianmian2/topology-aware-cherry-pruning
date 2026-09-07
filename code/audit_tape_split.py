"""Regenerate and audit the documented seed-42 tape-support tree split."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_id(file_name: str) -> str:
    parts = file_name.split("_")
    if len(parts) < 3 or parts[0] != "tree":
        raise ValueError(f"Unsupported tape filename: {file_name}")
    return f"tree_{parts[1]}"


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    package = root / "04_results" / "public_reproducibility_preflight_20260907"
    coco_path = root / "01_data" / "03_processed" / "annotations" / "mmdet" / "train" / "Tape_merged_dedup.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    membership_path = package / "data" / "manifests" / "image_task_membership.csv"
    safe_by_source_name = {}
    with membership_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for source_name in row["source_ids"].split(";"):
                if source_name.endswith(".jpg"):
                    safe_by_source_name[source_name] = row["sanitized_file"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for image in coco["images"]:
        groups[tree_id(image["file_name"])].append(image)

    trees = sorted(groups)
    shuffled = trees.copy()
    random.Random(42).shuffle(shuffled)
    train_end = int(0.8 * len(shuffled))
    val_end = train_end + int(0.1 * len(shuffled))
    split_trees = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
    assignment = {tree: split for split, values in split_trees.items() for tree in values}
    overlap = sum(len(set(split_trees[a]) & set(split_trees[b])) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    if overlap:
        raise RuntimeError("Tree leakage detected while generating tape split")

    target = ensure_dir(package / "data" / "splits")
    rows = []
    for image in coco["images"]:
        name = image["file_name"]
        rows.append({"image_id": image["id"], "image_file": safe_by_source_name[name], "source_file": name, "tree_id": tree_id(name), "split": assignment[tree_id(name)]})
    with (target / "tape_tree_split_seed42.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "deterministically_regenerated_from_documented_code",
        "source": "02_code/02_models/tape_segmentation_v2/data_processing.py::build_split",
        "seed": 42,
        "tree_count": len(trees),
        "image_count": len(rows),
        "split_tree_counts": {split: len(values) for split, values in split_trees.items()},
        "split_image_counts": {split: sum(1 for row in rows if row["split"] == split) for split in split_trees},
        "tree_overlap_across_splits": overlap,
        "published_status": "candidate; compare with any recovered original training split artifact before final release",
    }
    (target / "tape_tree_split_seed42_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    shutil_target = package / "code" / Path(__file__).name
    shutil_target.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    preflight_path = package / "preflight_report.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["tape_split"] = "seed-42 tree-level split regenerated; original run membership still requires provenance comparison"
    preflight_path.write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sums = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{sha256(path)}  {path.relative_to(package).as_posix()}")
    (package / "checksums" / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
