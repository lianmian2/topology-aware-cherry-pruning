# Pruning Truth Utilities

`audit_annotations.py` audits a frozen cut-line annotation export without modifying its source files. The output directory must be outside the annotation directory.

```powershell
& 'D:/Stable Diffusion/sd-webui-aki-v4.9/python/python.exe' -B `
  '02_code/05_utils/pruning_truth/audit_annotations.py' `
  --annotations '04_results/pruning_truth/manual_cut_lines' `
  --raw-images '01_data/01_raw/final_data' `
  --output '04_results/pruning_truth/audits/final_v1' `
  --expected-samples 330 `
  --annotator 'AUTHOR_INPUT' `
  --annotation-date 'YYYY-MM-DD' `
  --annotation-note 'AUTHOR_INPUT' `
  --hash-images
```

Replace `330` with the frozen expected count. An exit code of 2 means the audit found errors; inspect `annotation_issues.csv`. Empty annotations are warnings because a valid view can contain no cut line. Unconfirmed cuts are errors unless `--allow-unconfirmed` is explicitly used for a progress audit.

The audit writes:

- `annotation_manifest.csv`
- `annotation_issues.csv`
- `audit_summary.json`

The source annotation directory is never written.

## Decision baselines

`evaluate_pruning_baselines.py` consumes a branch feature table following `feature_schema.json`. It excludes `uncertain`, groups every split by `tree_id`, selects thresholds only inside training trees and writes branch predictions plus fold- and tree-level metrics.

```powershell
& 'D:/Stable Diffusion/sd-webui-aki-v4.9/python/python.exe' -B `
  '02_code/05_utils/pruning_truth/evaluate_pruning_baselines.py' `
  --features '04_results/pruning_decision/features.csv' `
  --representation gt `
  --output '04_results/pruning_decision/gt_baselines_s42'
```

Run the same frozen splits for `--representation pred`. The `rule_prediction` column is required and must be created from a predeclared horticultural rule, not tuned on held-out trees. GNN evaluation is intentionally gated until these transparent baselines exist.

## End-to-end regression

`e2e_regression_manifest.json` freezes six evidence-selected cases. Validate paths and exact default-weight hashes without loading the models:

```powershell
& 'D:/app/anna/envs/cherry/python.exe' -B `
  '02_code/05_utils/pruning_truth/run_e2e_regression.py' `
  --output '04_results/pruning_decision/e2e_regression_20260717' `
  --validate-only
```

Remove `--validate-only` to run RGB → ROI → branch mask → routing → mask_clip → bud detection → attachment → DAG → branch feature export. The manifest includes a topology success, gap-bridge stress, multi-component trunk, low-routing, clip-heavy and dense-bud/ROI-edge candidate.
