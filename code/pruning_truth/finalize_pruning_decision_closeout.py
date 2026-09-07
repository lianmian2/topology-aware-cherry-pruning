from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import sklearn
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize pruning decision evidence package")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    ensure_dir(root)
    key_artifacts = {
        "features": root / "features_v1" / "audit" / "summary.json",
        "baselines": root / "baselines_v1" / "selection.json",
        "sage_line": root / "gnn_sage_full_v1" / "summary.json",
        "gat_line": root / "gnn_gat_full_v1" / "summary.json",
        "sage_branch": root / "gnn_sage_branch_group_v1" / "summary.json",
        "cnn": root / "cnn_candidate_v1" / "summary.json",
        "recommendations": root / "recommendation_metrics_v1" / "summary.json",
        "tree_cv": root / "tree_cv_v1" / "summary.json",
        "expert_package": root / "expert_review_oof_v1" / "summary.json",
    }
    missing = [str(path) for path in key_artifacts.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing closeout artifacts: {missing}")
    summaries = {name: read_json(path) for name, path in key_artifacts.items()}
    code_paths = [
        PROJECT_ROOT / "02_code/05_utils/pruning_truth/prepare_pruning_decision_features.py",
        PROJECT_ROOT / "02_code/05_utils/pruning_truth/evaluate_pruning_segment_models.py",
        PROJECT_ROOT / "02_code/03_training/train_pruning_segment_gnn_v2.py",
        PROJECT_ROOT / "02_code/03_training/train_pruning_candidate_cnn.py",
        PROJECT_ROOT / "02_code/05_utils/pruning_truth/evaluate_pruning_recommendations.py",
        PROJECT_ROOT / "02_code/05_utils/pruning_truth/cross_validate_pruning_decision.py",
        PROJECT_ROOT / "02_code/05_utils/pruning_truth/build_expert_pruning_review_package.py",
    ]
    cv_by_model = {item["model"]: item for item in summaries["tree_cv"]["summaries"]}
    paper_summary = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "result_class": "model_evaluation",
        "primary_evaluation": "5-fold nested tree-level cross-validation after exploratory route freeze",
        "independent_unit": "tree",
        "data": {
            "trees": summaries["features"]["trees"],
            "views": summaries["features"]["accepted_views"],
            "candidate_segments": summaries["features"]["candidate_segments"],
            "positive_segments": summaries["features"]["positive_segments"],
            "feature_count": len(summaries["features"]["feature_columns"]),
        },
        "primary_cv": {
            "HGB-full": cv_by_model["HGB-full"],
            "GraphSAGE-branch": cv_by_model["GraphSAGE-branch"],
        },
        "local_cnn_exploratory": summaries["cnn"]["metrics_mean_std"],
        "selected_manuscript_route": (
            "Structure-aware pruning decision support: HGB-full as the stable decision head; "
            "branch-context GraphSAGE as a complementary topology-aware candidate ranker; "
            "local CNN as the no-global-structure comparator."
        ),
        "claim_boundary": (
            "The experiments demonstrate learnable pruning-decision signal and executable candidate ranking "
            "on accepted automatic-structure views. They do not establish universal horticultural correctness."
        ),
        "expert_validation_status": "blinded OOF package prepared; scoring pending and non-blocking",
    }
    atomic_json(root / "paper_result_summary.json", paper_summary)
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "computational_closeout_complete_expert_scoring_pending",
        "source_annotations_modified": False,
        "source_dataset": summaries["features"]["source_dataset"],
        "source_audit_sha256": summaries["features"]["source_audit_sha256"],
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in key_artifacts.items()
        },
        "code": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in code_paths},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "git": {
            "head": git_value("rev-parse", "HEAD"),
            "worktree_status_at_closeout": git_value("status", "--short"),
        },
        "evidence_classification": {
            "primary": ["tree_cv"],
            "supporting": ["features", "baselines", "recommendations", "expert_package"],
            "exploratory": ["sage_line", "gat_line", "sage_branch", "cnn"],
        },
    }
    atomic_json(root / "closeout_manifest.json", manifest)
    print(json.dumps(paper_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
