# Cloud handoff: WoodyDA-Net internal ablation

## Objective

Run the three ROI-only internal ablations of **WoodyDA-Net** (Woody-structure Dual-Attention Network). This package does not run or require the public iMED-Lab CS-Net.

The reference WoodyDA-Net condition has already completed locally. The cloud only trains three matched controls that isolate the attention mechanism:

| Run | Model selector | Enabled attention |
|---|---|---|
| A1 | `csnet_v2_no_attention` | neither channel nor spatial |
| A2 | `csnet_v2_channel_only` | channel only |
| A3 | `csnet_v2_spatial_only` | spatial only |

## Upload these paths together

Keep the paths below unchanged beneath one cloud project root:

```text
02_code/02_models/branch_seg_csnet_v2/model.py
02_code/02_models/branch_seg_csnet_v2/loss.py
02_code/02_models/branch_seg_csnet_v2/CLOUD_HANDOFF.md
02_code/02_models/branch_seg_csnet_v2/requirements_cloud.txt
02_code/03_training/compag_experiments/common/
02_code/03_training/compag_experiments/reports/
02_code/03_training/compag_experiments/runners/train_branch_seg_binary.py
02_code/03_training/compag_experiments/configs/branch_seg/binary_cleaned_csnet_no_attention_roi_bcedice.yaml
02_code/03_training/compag_experiments/configs/branch_seg/binary_cleaned_csnet_channel_only_roi_bcedice.yaml
02_code/03_training/compag_experiments/configs/branch_seg/binary_cleaned_csnet_spatial_only_roi_bcedice.yaml
02_code/03_training/compag_experiments/scripts/run_woodyda_attention_ablations.sh
```

Also upload or mount the already prepared ROI train/test images and the existing read-only `trunk_cleaned` LabelMe annotations. Training must never modify the annotation directory.

## Cloud data contract

The three YAML files use these fields, which may be changed only to match the cloud locations:

```yaml
data.train_img_dir: ROI black-background training images
data.val_img_dir: ROI black-background validation images
data.labelme_dir: immutable trunk_cleaned LabelMe annotations
```

Image names ending in `_aug` resolve to the annotation whose name precedes `_aug`. Keep the existing train/test membership unchanged. Do not change model, loss, input size, seed, target or threshold fields.

## Fixed experimental contract

- Input: 1024 x 1024 normalized RGB ROI black-background image.
- Target: binary union of LabelMe `Trunk` and `Branch` polygons.
- Optimizer/loss: AdamW and BCE + soft Dice.
- Training: 100 epochs, physical batch size 4, AMP enabled, seed 42.
- Model selection: maximum validation IoU.
- Evaluation: sigmoid probability threshold 0.5.
- All variants retain the same U-shaped 64/128/256/512/1024 backbone and skip connections; only attention availability changes.

## Run command

From the cloud project root:

```bash
chmod +x 02_code/03_training/compag_experiments/scripts/run_woodyda_attention_ablations.sh
bash 02_code/03_training/compag_experiments/scripts/run_woodyda_attention_ablations.sh
```

To verify paths and code wiring before a full training session:

```bash
EPOCHS_ARG="--epochs 1" bash 02_code/03_training/compag_experiments/scripts/run_woodyda_attention_ablations.sh
```

Smoke-run metrics are not paper evidence. Each full run writes a timestamped experiment directory beneath `OUT` (default: `/root/autodl-tmp/outputs/compag_runs`) containing the resolved configuration, split record, checkpoints, epoch metrics, per-sample metrics, summary JSON and visualizations. Return all three full directories without renaming or deleting their contents.
