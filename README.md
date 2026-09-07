# Topology-aware structural representation from low-cost RGB images for sweet-cherry pruning candidate ranking

This repository contains the code accompanying a manuscript on RGB-based sweet-cherry structural representation and pruning-candidate ranking. The workflow is organized as ROI-conditioned woody and bud perception, tape-aware mask preparation, topology routing, bud–structure integration, and topology-informed pruning ranking.

## Repository scope

This is a code-only repository. It deliberately excludes RGB images, annotations, derived features, experiment outputs, and trained checkpoints. Those research artefacts are distributed under CC BY 4.0 through the corresponding [v1.0 GitHub Release](https://github.com/lianmian2/topology-aware-cherry-pruning/releases/tag/v1.0).

## Layout

- `code/branch_seg_csnet_v2/`: WoodyCA-Net training and evaluation utilities.
- `code/tape_segmentation_v2/`: physical-tape segmentation utilities.
- `code/mask_topology_routing/` and `code/mask_topology_routing_binary_v3/`: structural routing and refinement.
- `code/bud_skeleton_fusion/`: bud detection, orientation, and attachment.
- `code/pruning_truth/`: atomic-segment and topology-derived candidate construction, ranking, and audits.
- `code/unified_system/`: pipeline integration and graphical interfaces.
- `environment/requirements_core_pipeline.txt`: the paper-facing Python dependency set.
- `docs/REPRODUCTION_PROTOCOL.md`: required order for reproduction.

## Installation and use

Create a clean Python environment and install `environment/requirements_core_pipeline.txt`. CUDA-compatible PyTorch wheels should be selected for the available GPU and CUDA runtime. Download the `v1.0` Release assets for data paths, split manifests, model checkpoints, and supplied out-of-fold predictions.

The repository is released under the MIT License. Please cite the associated manuscript and the corresponding GitHub Release when using this code.
