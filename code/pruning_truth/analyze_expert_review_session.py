from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SESSION = PROJECT_ROOT / "expert_review_sessions" / "gui_session_20260818"
DEFAULT_OUTPUT = PROJECT_ROOT / "expert_review_analysis_20260822"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
    }


def category_from_row(row: pd.Series) -> str:
    values = {
        "acceptable": int(row["acceptable_recommendation_0_1"]),
        "harmless": int(row["unnecessary_but_harmless_0_1"]),
        "harmful": int(row["harmful_or_unacceptable_0_1"]),
    }
    active = [key for key, value in values.items() if value == 1]
    return active[0] if len(active) == 1 else ""


def category_from_state(values: dict[str, Any]) -> str:
    category = str(values.get("category", "")).strip()
    return category if category in {"acceptable", "harmless", "harmful"} else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a blinded expert pruning-review session")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    session = resolve_path(args.session)
    output = ensure_dir(resolve_path(args.output))

    state = json.loads((session / "review_state.json").read_text(encoding="utf-8"))
    target_count = int(state.get("target_count", 20))
    manifest = pd.read_csv(session / "session_manifest.csv")
    manifest = manifest.loc[manifest.selection_order <= target_count].copy()
    scoring = pd.read_csv(session / "expert_scoring_sheet.csv")
    scoring["display_slot"] = pd.to_numeric(scoring["display_slot"], errors="coerce")
    scoring = scoring.loc[scoring.display_slot <= 3].copy()
    key = pd.read_csv(session / "blind_model_key.csv")
    scoring = scoring.merge(key, on=["sample_id", "panel", "display_slot"], how="left", validate="one_to_one")
    scoring = scoring.merge(manifest[["sample_id", "tree_id", "view", "selection_order"]], on="sample_id", how="inner", validate="many_to_one")
    scoring["category"] = scoring.apply(category_from_row, axis=1)

    # The GUI's current design randomizes all ten candidates in each panel and
    # asks the reviewer to score display slots 1-3.  The review_state is the
    # authoritative interaction log; the CSV is checked against it and then
    # used for the tabular merge/export because it carries the one-hot labels.
    state_rows = []
    for score_key, values in state.get("scores", {}).items():
        sample_id, panel, slot = score_key.split("|", 2)
        slot = int(slot)
        if slot <= 3:
            state_rows.append({"sample_id": sample_id, "panel": panel, "display_slot": slot, "category": category_from_state(values)})
    state_frame = pd.DataFrame(state_rows)
    state_reviewed = state_frame.loc[state_frame.display_slot <= 3].copy()
    state_lookup = state_reviewed.set_index(["sample_id", "panel", "display_slot"])["category"]
    csv_lookup = scoring.set_index(["sample_id", "panel", "display_slot"])["category"]
    shared_keys = state_lookup.index.intersection(csv_lookup.index)
    mismatched_keys = [key_item for key_item in shared_keys if state_lookup.loc[key_item] != csv_lookup.loc[key_item]]

    expected = target_count * 2 * 3
    rated = scoring.loc[scoring.category != ""].copy()
    invalid = scoring.loc[scoring.category == ""].copy()
    models = sorted(rated.model.dropna().unique().tolist())
    if len(models) != 2:
        raise ValueError(f"Expected two reviewed models, found {models}")
    if len(rated) != expected:
        raise ValueError(f"Reviewed candidate set is incomplete: {len(rated)}/{expected} rated rows")

    model_rows: list[dict[str, Any]] = []
    tree_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        subset = rated.loc[rated.model == model]
        tree_table = subset.groupby("tree_id", sort=True).agg(
            recommendations=("category", "size"),
            acceptable=("category", lambda values: int((values == "acceptable").sum())),
            harmless=("category", lambda values: int((values == "harmless").sum())),
            harmful=("category", lambda values: int((values == "harmful").sum())),
        ).reset_index()
        tree_table["acceptable_rate"] = tree_table.acceptable / tree_table.recommendations
        tree_table["harmless_rate"] = tree_table.harmless / tree_table.recommendations
        tree_table["harmful_rate"] = tree_table.harmful / tree_table.recommendations
        tree_table["non_harmful_rate"] = (tree_table.acceptable + tree_table.harmless) / tree_table.recommendations
        tree_table["three_reviewed_candidates_success"] = (tree_table.acceptable > 0).astype(int)
        tree_table["three_reviewed_candidates_non_harmful_success"] = ((tree_table.acceptable + tree_table.harmless) > 0).astype(int)
        tree_table["model"] = model
        tree_rows.extend(tree_table.to_dict("records"))
        model_rows.append(
            {
                "model": model,
                "recommendations": int(len(subset)),
                "tree_bundles": int(len(tree_table)),
                "acceptable": int((subset.category == "acceptable").sum()),
                "harmless": int((subset.category == "harmless").sum()),
                "harmful": int((subset.category == "harmful").sum()),
                "acceptable_rate": float((subset.category == "acceptable").mean()),
                "harmless_rate": float((subset.category == "harmless").mean()),
                "harmful_rate": float((subset.category == "harmful").mean()),
                "non_harmful_rate": float(subset.category.isin(["acceptable", "harmless"]).mean()),
                "three_reviewed_candidates_success": bootstrap_mean(tree_table.three_reviewed_candidates_success.to_numpy(), 20260822 + model_index),
                "three_reviewed_candidates_non_harmful_success": bootstrap_mean(tree_table.three_reviewed_candidates_non_harmful_success.to_numpy(), 20260832 + model_index),
                "tree_mean_acceptable_rate": bootstrap_mean(tree_table.acceptable_rate.to_numpy(), 20260842 + model_index),
                "tree_mean_harmful_rate": bootstrap_mean(tree_table.harmful_rate.to_numpy(), 20260852 + model_index),
            }
        )
        for rank in range(1, 11):
            rank_subset = subset.loc[subset.model_rank == rank]
            rank_rows.append(
                {
                    "model": model,
                    "model_rank": rank,
                    "recommendations": int(len(rank_subset)),
                    "acceptable": int((rank_subset.category == "acceptable").sum()),
                    "harmless": int((rank_subset.category == "harmless").sum()),
                    "harmful": int((rank_subset.category == "harmful").sum()),
                    "acceptable_rate": float((rank_subset.category == "acceptable").mean()),
                    "harmful_rate": float((rank_subset.category == "harmful").mean()),
                }
            )

    tree_frame = pd.DataFrame(tree_rows)
    model_frame = pd.DataFrame(model_rows)
    rank_frame = pd.DataFrame(rank_rows)
    overall_tree = rated.groupby("tree_id", sort=True).agg(
        recommendations=("category", "size"),
        acceptable=("category", lambda values: int((values == "acceptable").sum())),
        harmless=("category", lambda values: int((values == "harmless").sum())),
        harmful=("category", lambda values: int((values == "harmful").sum())),
    ).reset_index()
    overall_tree["acceptable_rate"] = overall_tree.acceptable / overall_tree.recommendations
    overall_tree["harmful_rate"] = overall_tree.harmful / overall_tree.recommendations
    overall_tree["at_least_one_acceptable"] = (overall_tree.acceptable > 0).astype(int)
    overall_tree["at_least_one_non_harmful"] = ((overall_tree.acceptable + overall_tree.harmless) > 0).astype(int)
    overall_tree["any_harmful"] = (overall_tree.harmful > 0).astype(int)
    pivot_success = tree_frame.pivot(index="tree_id", columns="model", values="three_reviewed_candidates_success")
    pivot_acceptable = tree_frame.pivot(index="tree_id", columns="model", values="acceptable_rate")
    pivot_harmful = tree_frame.pivot(index="tree_id", columns="model", values="harmful_rate")
    model_a, model_b = models
    common_tree_ids = pivot_acceptable.dropna().index
    paired = {"paired_tree_count": int(len(common_tree_ids)), "paired_tree_ids": [str(item) for item in common_tree_ids]}
    if len(common_tree_ids):
        paired.update(
            {
                "acceptable_rate_mean_difference_model_a_minus_model_b": bootstrap_mean((pivot_acceptable.loc[common_tree_ids, model_a] - pivot_acceptable.loc[common_tree_ids, model_b]).to_numpy(), 20260862),
                "harmful_rate_mean_difference_model_a_minus_model_b": bootstrap_mean((pivot_harmful.loc[common_tree_ids, model_a] - pivot_harmful.loc[common_tree_ids, model_b]).to_numpy(), 20260872),
                "three_reviewed_candidates_success_difference_model_a_minus_model_b": bootstrap_mean((pivot_success.loc[common_tree_ids, model_a] - pivot_success.loc[common_tree_ids, model_b]).to_numpy(), 20260882),
            }
        )

    total_counts = rated.category.value_counts().to_dict()
    tree_model_counts = tree_frame.groupby("tree_id")["model"].nunique()
    trees_with_both_models = int((tree_model_counts >= len(models)).sum())
    trees_with_one_model_only = int((tree_model_counts < len(models)).sum())
    summary = {
        "session_id": session.name,
        "source_artifacts": {
            "session_manifest": str(session / "session_manifest.csv"),
            "expert_scoring_sheet": str(session / "expert_scoring_sheet.csv"),
            "blind_model_key": str(session / "blind_model_key.csv"),
            "review_state": str(session / "review_state.json"),
        },
        "independent_unit": "tree",
        "trees": int(manifest.tree_id.nunique()),
        "views": int(manifest.sample_id.nunique()),
        "panels": int(manifest.sample_id.nunique() * 2),
        "models": models,
        "reviewed_display_slots_per_panel": 3,
        "candidate_sampling_scope": "Each panel contains one model's Top-10 list; the ten candidates are randomly permuted, and display slots 1-3 are reviewed.",
        "expected_reviewed_recommendations": expected,
        "rated_reviewed_recommendations": int(len(rated)),
        "invalid_or_unrated_reviewed_rows": int(len(invalid)),
        "observed_model_rank_counts": {str(int(rank)): int(count) for rank, count in rated.model_rank.value_counts().sort_index().items()},
        "observed_true_top3_rows": int((rated.model_rank <= 3).sum()),
        "observed_rank4_to10_rows": int((rated.model_rank >= 4).sum()),
        "model_recommendation_counts": {str(model): int(count) for model, count in rated.model.value_counts().to_dict().items()},
        "trees_with_both_models": trees_with_both_models,
        "trees_with_one_model_only": trees_with_one_model_only,
        "paired_model_comparison_valid": False,
        "model_comparison_exclusion_reason": (
            "This session used randomized unpaired channel assignment: "
            f"{trees_with_both_models} trees contained both model pools and "
            f"{trees_with_one_model_only} trees contained only one observed model pool. "
            "Pooled expert-judged outcomes are primary; model-specific rates are retained "
            "as sampling provenance/audit only, not as comparative evidence."
        ),
        "pooled_counts": {key: int(total_counts.get(key, 0)) for key in ("acceptable", "harmless", "harmful")},
        "pooled_rates": {key: float(total_counts.get(key, 0) / len(rated)) for key in ("acceptable", "harmless", "harmful")},
        "expert_judged_pruning_correctness_rate": float((rated.category == "acceptable").mean()),
        "expert_judged_non_harmful_rate": float(rated.category.isin(["acceptable", "harmless"]).mean()),
        "expert_judged_harmful_recommendation_rate": float((rated.category == "harmful").mean()),
        "label_semantics": {
            "acceptable": "The agronomist judged that this candidate location should be pruned under normal pruning practice; counted as an expert-judged correct pruning recommendation.",
            "harmless": "The agronomist judged that pruning this location was unnecessary but not harmful; not counted as correct pruning.",
            "harmful": "The agronomist judged that pruning this location was harmful or unacceptable; counted as a harmful recommendation.",
        },
        "tree_level_reviewed_set": {
            "trees_with_at_least_one_acceptable": int(overall_tree.at_least_one_acceptable.sum()),
            "trees_with_at_least_one_non_harmful": int(overall_tree.at_least_one_non_harmful.sum()),
            "trees_with_any_harmful": int(overall_tree.any_harmful.sum()),
            "mean_acceptable_rate": float(overall_tree.acceptable_rate.mean()),
            "mean_harmful_rate": float(overall_tree.harmful_rate.mean()),
        },
        "model_results": model_rows,
        "rank_results": rank_rows,
        "paired_tree_differences": paired,
        "confidence_filled_rows": int(rated.confidence_1_to_5.notna().sum()),
        "notes_filled_rows": int(rated.reason_or_note.fillna("").astype(str).str.strip().ne("").sum()),
        "state_csv_audit": {
            "state_reviewed_rows": int(len(state_reviewed)),
            "state_rated_rows": int((state_reviewed.category != "").sum()),
            "csv_reviewed_rows": int(len(scoring)),
            "csv_rated_rows": int(len(rated)),
            "shared_keys": int(len(shared_keys)),
            "category_mismatches": int(len(mismatched_keys)),
            "state_is_authoritative_interaction_log": True,
        },
        "interpretation_boundary": "single-expert, 20-tree audit of randomly sampled candidate outputs from two high-performing models; acceptable is operationalized as an expert-judged correct pruning recommendation for the sampled candidate, while the result is not universal botanical truth, complete-tree cut-budget coverage, or product-level deployment accuracy",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model_frame.to_csv(output / "model_results.csv", index=False, encoding="utf-8-sig")
    rank_frame.to_csv(output / "rank_results.csv", index=False, encoding="utf-8-sig")
    tree_frame.to_csv(output / "tree_results.csv", index=False, encoding="utf-8-sig")
    overall_tree.to_csv(output / "overall_tree_results.csv", index=False, encoding="utf-8-sig")
    rated.to_csv(output / "rated_reviewed_rows_unblinded.csv", index=False, encoding="utf-8-sig")
    invalid.to_csv(output / "unrated_or_invalid_reviewed_rows.csv", index=False, encoding="utf-8-sig")
    state_frame.to_csv(output / "review_state_slots_1_to_3.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
