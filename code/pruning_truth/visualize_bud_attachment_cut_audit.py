"""Visual audit for predicted buds, their skeleton attachment, and manual cuts.

This is a data-preparation QC tool.  It never writes to raw cut-line truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from build_atomic_segment_graphs import PROJECT_ROOT, ensure_dir, process_sample


SEGMENT_COLORS = {
    "junction--bud": (70, 190, 70),       # green
    "bud--bud": (240, 200, 0),             # cyan-yellow
    "bud--endpoint": (255, 120, 0),        # blue
    "junction--endpoint": (0, 140, 255),  # orange
    "junction--junction": (190, 70, 190), # purple
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def put(canvas, text, xy, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def header(image, lines):
    band = 76 + 28 * max(0, len(lines) - 1)
    canvas = np.full((image.shape[0] + band, image.shape[1], 3), 28, dtype=np.uint8)
    canvas[band:] = image
    for index, line in enumerate(lines):
        put(canvas, line, (15, 32 + 28 * index), scale=0.68)
    return canvas


def draw_cuts(image, cuts, mapping_by_id):
    for cut in cuts:
        p0, p1 = tuple(map(int, cut["p0"])), tuple(map(int, cut["p1"]))
        status = mapping_by_id[str(cut["cut_id"])]["status"]
        color = {"exact_unique": (0, 220, 0), "non_intersection": (0, 0, 255), "multi_intersection": (0, 140, 255), "needs_landmark_review": (255, 0, 255)}.get(status, (255, 255, 255))
        cv2.line(image, p0, p1, color, 4, cv2.LINE_AA)
        center = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        put(image, f"C{cut['cut_id']}", (center[0] + 6, center[1] - 6), color, 0.55, 1)


def crop_with_context(image, center, size=520):
    x, y = map(int, center)
    half = size // 2
    left, top = max(0, x - half), max(0, y - half)
    right, bottom = min(image.shape[1], x + half), min(image.shape[0], y + half)
    return image[top:bottom, left:right]


def main():
    parser = argparse.ArgumentParser(description="Render visual QC panels for bud detection, attachment and cut mapping")
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    args = parser.parse_args()
    cases_root = args.cases_root if args.cases_root.is_absolute() else PROJECT_ROOT / args.cases_root
    output_root = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    ensure_dir(output_root)
    csv_rows = []

    for sample_id in args.sample_id:
        # Rebuild with strict three-type semantics before rendering.
        process_sample(cases_root, output_root / "strict_graphs", sample_id, None)
        graph = load(output_root / "strict_graphs" / sample_id / "decision_graph.json")
        tree_id, view = sample_id.split("_before_")
        image = cv2.imread(graph["source"]["image"])
        attachments = load(cases_root / sample_id / "attachments.json")
        cuts = load(PROJECT_ROOT / "04_results/pruning_truth/manual_cut_lines_20260720_raw" / f"{tree_id}_{view}" / "cut_lines.json")["cut_lines"]
        mapping_by_id = {str(item["cut_id"]): item for item in graph["cut_mappings"]}
        sample_output = ensure_dir(output_root / sample_id)

        attached = [item for item in attachments if item.get("skeleton_point") is not None]
        method_counts = Counter(item.get("attachment_method", "none") for item in attachments)
        status_counts = Counter(item["status"] for item in graph["cut_mappings"])

        detection = image.copy()
        for item in attachments:
            x, y = map(int, item["bud_centroid"])
            attached_now = item.get("skeleton_point") is not None
            color = (255, 100, 20) if attached_now else (0, 0, 255)
            cv2.circle(detection, (x, y), 7, color, 2, cv2.LINE_AA)
            put(detection, str(item["bud_index"]), (x + 7, y - 7), color, 0.35, 1)
        draw_cuts(detection, cuts, mapping_by_id)
        detection = header(detection, [
            f"{sample_id} | predicted buds: {len(attachments)}; attached: {len(attached)}; unattached: {len(attachments) - len(attached)}",
            "buds: blue=attached, red=unattached | cuts: green=unique, red=no skeleton, orange=multi skeleton, magenta=landmark review",
        ])
        cv2.imwrite(str(sample_output / "01_bud_detection_and_cuts.jpg"), detection)

        attachment_panel = image.copy()
        for segment in graph["segment_nodes"]:
            points = np.asarray(segment["polyline"], dtype=np.int32).reshape(-1, 1, 2)
            color = SEGMENT_COLORS.get(segment["candidate_type"], (130, 130, 130))
            cv2.polylines(attachment_panel, [points], False, color, 2, cv2.LINE_AA)
        for item in attachments:
            centroid = tuple(map(int, item["bud_centroid"]))
            target = item.get("skeleton_point")
            if target is None:
                cv2.circle(attachment_panel, centroid, 7, (0, 0, 255), 2, cv2.LINE_AA)
                continue
            target = tuple(map(int, target))
            color = (255, 220, 0) if item.get("attachment_method") == "component_snap" else (255, 0, 255)
            cv2.line(attachment_panel, centroid, target, color, 1, cv2.LINE_AA)
            cv2.circle(attachment_panel, centroid, 5, (255, 100, 20), -1, cv2.LINE_AA)
            cv2.drawMarker(attachment_panel, target, color, cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)
        draw_cuts(attachment_panel, cuts, mapping_by_id)
        attachment_panel = header(attachment_panel, [
            f"{sample_id} | five candidate segment types are colored; gray=context/root-related; attached buds={len(attached)}",
            "segments: green=J-B, yellow=B-B, blue=B-E, orange=J-E, purple=J-J",
            f"attachment: cyan=component-snap ({method_counts['component_snap']}), magenta=component/fallback ({method_counts['component'] + method_counts['fallback']}), red=none ({method_counts['none']})",
        ])
        cv2.imwrite(str(sample_output / "02_bud_attachment_skeleton_cuts.jpg"), attachment_panel)

        review_rows = []
        for cut in cuts:
            cut_id = str(cut["cut_id"])
            item = mapping_by_id[cut_id]
            p0, p1 = cut["p0"], cut["p1"]
            center = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            crop = crop_with_context(attachment_panel[76:], center)
            put(crop, f"C{cut_id}: {item['status']}", (12, 28), (255, 255, 255), 0.65, 2)
            safe_cut_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in cut_id)
            cv2.imwrite(str(sample_output / f"{safe_cut_id}_{item['status']}.jpg"), crop)
            row = {"sample_id": sample_id, "cut_id": cut_id, "status": item["status"], "intersected_segment_ids": ";".join(map(str, item["intersected_segment_ids"])), "candidate_segment_ids": ";".join(map(str, item["candidate_segment_ids"]))}
            review_rows.append(row)
            csv_rows.append(row)
        with (sample_output / "cut_mapping_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]) if review_rows else ["sample_id"])
            writer.writeheader(); writer.writerows(review_rows)
        (sample_output / "README.txt").write_text(
            "01: detector-centroid audit; 02: detector-to-skeleton attachment audit; cut_*.jpg: cut-local review.\n"
            "Status: exact_unique=usable automatic label; non_intersection=exclude as skeleton-untrusted; multi_intersection=manual review; needs_landmark_review=the cut touches a context/root-related segment, not one of the five supervised types; it is not evidence of a missed bud.\n",
            encoding="utf-8",
        )
        print(sample_id, dict(status_counts))
    with (output_root / "all_cut_mapping_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "cut_id", "status", "intersected_segment_ids", "candidate_segment_ids"])
        writer.writeheader(); writer.writerows(csv_rows)
    counts = Counter(row["status"] for row in csv_rows)
    (output_root / "README.md").write_text(
        "# 五类剪枝候选：可视化审计\\n\\n"
        "本目录是派生审计结果；原始剪切线标注未被修改。每个 `tree_*` 子目录包含：\\n\\n"
        "- `01_bud_detection_and_cuts.jpg`：原图、自动检测芽点和人工剪切线。蓝圈=已附着芽点；红圈=未附着芽点。\\n"
        "- `02_bud_attachment_skeleton_cuts.jpg`：原图、预测骨架、芽点到骨架的附着线、人工剪切线。骨架颜色：绿=分叉—芽(J-B)，黄=芽—芽(B-B)，蓝=芽—端(B-E)，橙=分叉—端(J-E)，紫=分叉—分叉(J-J)，灰=仅作上下文的主干/根相关或其他边。青色附着线=component-snap，品红附着线=component 或 fallback，红圈=未附着。\\n"
        "- `cut_*.jpg`：每条人工剪切线的局部裁剪。\\n"
        "- `cut_mapping_review.csv`：逐线几何映射。\\n\\n"
        "剪切线颜色：绿=`exact_unique`（唯一相交一条五类候选边）；红=`non_intersection`（与预测骨架零相交，应排除）；橙=`multi_intersection`（相交多条预测边，人工清洗）；品红=`needs_landmark_review`（仅相交上下文/主干相关边，不对芽点漏检作任何推断）。\\n\\n"
        f"本次试点汇总：{dict(counts)}。这些是数据准备与人工审计状态，不是剪枝模型性能。\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
