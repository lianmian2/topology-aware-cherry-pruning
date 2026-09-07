from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


SAMPLE_PATTERN = re.compile(r"^(tree_\d{3})_(view_\d{2})$")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_numeric_pair(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def resolve_image_path(
    stored: Any,
    raw_images: Path,
    tree_id: str,
    stage: str,
    view: str,
) -> Path:
    if isinstance(stored, str) and stored:
        candidate = Path(stored)
        if candidate.exists():
            return candidate
    return raw_images / tree_id / stage / f"{view}.jpg"


def image_size(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def add_issue(
    issues: list[dict[str, str]],
    sample_id: str,
    severity: str,
    code: str,
    message: str,
) -> None:
    issues.append(
        {
            "sample_id": sample_id,
            "severity": severity,
            "code": code,
            "message": message,
        }
    )


def audit_sample(
    json_path: Path,
    annotations: Path,
    raw_images: Path,
    hash_images: bool,
    allow_unconfirmed: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    sample_id = json_path.parent.name
    match = SAMPLE_PATTERN.match(sample_id)
    directory_tree = match.group(1) if match else ""
    directory_view = match.group(2) if match else ""
    if not match:
        add_issue(issues, sample_id, "error", "invalid_sample_dir", sample_id)

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_issue(issues, sample_id, "error", "invalid_json", str(exc))
        return (
            {
                "sample_id": sample_id,
                "tree_id": directory_tree,
                "view": directory_view,
                "json_path": json_path.relative_to(annotations).as_posix(),
                "json_sha256": sha256_file(json_path),
                "status": "invalid",
                "num_cut_lines": 0,
                "num_confirmed": 0,
                "before_image": "",
                "after_image": "",
                "before_image_sha256": "",
                "after_image_sha256": "",
                "error_count": 1,
                "warning_count": 0,
            },
            issues,
        )

    tree_id = str(data.get("tree_id", ""))
    view = str(data.get("view", ""))
    stage = str(data.get("image_stage", ""))
    if tree_id != directory_tree:
        add_issue(issues, sample_id, "error", "tree_id_mismatch", f"{tree_id} != {directory_tree}")
    if view != directory_view:
        add_issue(issues, sample_id, "error", "view_mismatch", f"{view} != {directory_view}")
    if stage != "before":
        add_issue(issues, sample_id, "error", "invalid_stage", stage)

    before_path = resolve_image_path(data.get("image_path"), raw_images, tree_id, "before", view)
    after_path = resolve_image_path(data.get("reference_after_path"), raw_images, tree_id, "after", view)
    before_size = image_size(before_path)
    after_size = image_size(after_path)
    if before_size is None:
        add_issue(issues, sample_id, "error", "missing_or_invalid_before_image", str(before_path))
    if after_size is None:
        add_issue(issues, sample_id, "error", "missing_or_invalid_after_image", str(after_path))
    if before_size and after_size and before_size != after_size:
        add_issue(issues, sample_id, "warning", "image_size_mismatch", f"{before_size} != {after_size}")

    cut_lines = data.get("cut_lines")
    if not isinstance(cut_lines, list):
        add_issue(issues, sample_id, "error", "invalid_cut_lines", "cut_lines must be a list")
        cut_lines = []
    if not cut_lines:
        add_issue(issues, sample_id, "warning", "empty_annotation", "no cut lines")

    cut_ids: set[str] = set()
    confirmed_count = 0
    for index, line in enumerate(cut_lines):
        prefix = f"cut_lines[{index}]"
        if not isinstance(line, dict):
            add_issue(issues, sample_id, "error", "invalid_cut_line", prefix)
            continue
        cut_id = str(line.get("cut_id", ""))
        if not cut_id:
            add_issue(issues, sample_id, "error", "missing_cut_id", prefix)
        elif cut_id in cut_ids:
            add_issue(issues, sample_id, "error", "duplicate_cut_id", cut_id)
        cut_ids.add(cut_id)

        p0 = line.get("p0")
        p1 = line.get("p1")
        if not is_numeric_pair(p0) or not is_numeric_pair(p1):
            add_issue(issues, sample_id, "error", "invalid_endpoint", prefix)
            continue
        if float(p0[0]) == float(p1[0]) and float(p0[1]) == float(p1[1]):
            add_issue(issues, sample_id, "error", "zero_length_cut", cut_id)
        if before_size:
            width, height = before_size
            for point_name, point in (("p0", p0), ("p1", p1)):
                if not (0 <= float(point[0]) < width and 0 <= float(point[1]) < height):
                    add_issue(issues, sample_id, "error", "coordinate_out_of_bounds", f"{cut_id}.{point_name}")

        normal = line.get("normal_pruned_side")
        retained_normal = line.get("normal_retained_side")
        if not is_numeric_pair(normal):
            add_issue(issues, sample_id, "error", "invalid_pruned_normal", cut_id)
        elif not math.isclose(math.hypot(float(normal[0]), float(normal[1])), 1.0, abs_tol=1e-3):
            add_issue(issues, sample_id, "error", "non_unit_pruned_normal", cut_id)
        if not is_numeric_pair(retained_normal):
            add_issue(issues, sample_id, "error", "invalid_retained_normal", cut_id)
        elif is_numeric_pair(normal) and not (
            math.isclose(float(normal[0]), -float(retained_normal[0]), abs_tol=1e-3)
            and math.isclose(float(normal[1]), -float(retained_normal[1]), abs_tol=1e-3)
        ):
            add_issue(issues, sample_id, "error", "normal_direction_mismatch", cut_id)

        confirmed = line.get("confirmed")
        if confirmed is True:
            confirmed_count += 1
        elif confirmed is not False:
            add_issue(issues, sample_id, "error", "invalid_confirmed_value", cut_id)
        elif not allow_unconfirmed:
            add_issue(issues, sample_id, "error", "unconfirmed_cut", cut_id)

    error_count = sum(item["severity"] == "error" for item in issues)
    warning_count = sum(item["severity"] == "warning" for item in issues)
    status = "invalid" if error_count else "complete" if confirmed_count == len(cut_lines) else "incomplete"
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "tree_id": tree_id,
        "view": view,
        "json_path": json_path.relative_to(annotations).as_posix(),
        "json_sha256": sha256_file(json_path),
        "status": status,
        "num_cut_lines": len(cut_lines),
        "num_confirmed": confirmed_count,
        "before_image": str(before_path),
        "after_image": str(after_path),
        "before_image_sha256": sha256_file(before_path) if hash_images and before_size else "",
        "after_image_sha256": sha256_file(after_path) if hash_images and after_size else "",
        "error_count": error_count,
        "warning_count": warning_count,
    }
    return row, issues


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only audit for pruning cut-line annotations")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--raw-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--annotator", default="")
    parser.add_argument("--annotation-date", default="")
    parser.add_argument("--annotation-note", default="")
    parser.add_argument("--hash-images", action="store_true")
    parser.add_argument("--allow-unconfirmed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = args.annotations.resolve()
    raw_images = args.raw_images.resolve()
    output = args.output.resolve()
    if not annotations.exists():
        raise FileNotFoundError(annotations)
    if not raw_images.exists():
        raise FileNotFoundError(raw_images)
    if output == annotations or annotations in output.parents:
        raise ValueError("Audit output must be outside the source annotation directory")

    json_files = sorted(annotations.glob("*/cut_lines.json"))
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for json_path in json_files:
        row, sample_issues = audit_sample(
            json_path,
            annotations,
            raw_images,
            args.hash_images,
            args.allow_unconfirmed,
        )
        rows.append(row)
        issues.extend(sample_issues)

    sample_counts = Counter(row["sample_id"] for row in rows)
    for sample_id, count in sample_counts.items():
        if count > 1:
            add_issue(issues, sample_id, "error", "duplicate_sample", str(count))
    hash_counts = Counter(row["json_sha256"] for row in rows)
    for digest, count in hash_counts.items():
        if digest and count > 1:
            for row in rows:
                if row["json_sha256"] == digest:
                    add_issue(issues, row["sample_id"], "warning", "duplicate_json_content", digest)

    if args.expected_samples is not None and len(rows) != args.expected_samples:
        add_issue(
            issues,
            "__dataset__",
            "error",
            "unexpected_sample_count",
            f"{len(rows)} != {args.expected_samples}",
        )
    dataset_fingerprint = hashlib.sha256(
        "".join(f"{row['json_path']}\0{row['json_sha256']}\n" for row in rows).encode("utf-8")
    ).hexdigest()
    severity_counts = Counter(item["severity"] for item in issues)
    code_counts = Counter(item["code"] for item in issues)
    status_counts = Counter(row["status"] for row in rows)
    tree_ids = sorted({row["tree_id"] for row in rows if row["tree_id"]})
    views = sorted({row["view"] for row in rows if row["view"]})
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_annotations": str(annotations),
        "raw_images": str(raw_images),
        "source_mode": "read_only",
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "num_samples": len(rows),
        "num_trees": len(tree_ids),
        "trees": tree_ids,
        "views": views,
        "total_cut_lines": sum(int(row["num_cut_lines"]) for row in rows),
        "total_confirmed": sum(int(row["num_confirmed"]) for row in rows),
        "status_counts": dict(sorted(status_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "issue_counts": dict(sorted(code_counts.items())),
        "hash_images": bool(args.hash_images),
        "allow_unconfirmed": bool(args.allow_unconfirmed),
        "expected_samples": args.expected_samples,
        "annotator": args.annotator,
        "annotation_date": args.annotation_date,
        "annotation_note": args.annotation_note,
    }

    ensure_dir(output)
    write_csv(output / "annotation_manifest.csv", rows, list(rows[0].keys()) if rows else ["sample_id"])
    write_csv(
        output / "annotation_issues.csv",
        issues,
        ["sample_id", "severity", "code", "message"],
    )
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if severity_counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

