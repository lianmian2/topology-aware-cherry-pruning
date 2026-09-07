"""Replace accidental code-side artifacts with a source-only code snapshot."""

from __future__ import annotations

import shutil
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_code_tree(source: Path, destination: Path) -> None:
    allowed = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".sh", ".ps1", ".bat"}
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            target = destination / path.relative_to(source)
            ensure_dir(target.parent)
            shutil.copy2(path, target)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    package = root / "04_results" / "public_reproducibility_preflight_20260907"
    target = package / "code"
    if target.exists():
        shutil.rmtree(target)
    ensure_dir(target)
    sources = [
        root / "02_code" / "02_models" / "branch_seg_csnet_v2",
        root / "02_code" / "02_models" / "mask_topology_routing",
        root / "02_code" / "02_models" / "mask_topology_routing_binary_v3",
        root / "02_code" / "02_models" / "bud_skeleton_fusion",
        root / "02_code" / "02_models" / "tape_segmentation_v2",
        root / "02_code" / "05_utils" / "pruning_truth",
        root / "07_graphical_interface" / "unified_system",
    ]
    for source in sources:
        copy_code_tree(source, target / source.name)
    shutil.copy2(root / "02_code" / "05_utils" / "public_release" / "build_preflight_package.py", target / "build_preflight_package.py")
    shutil.copy2(Path(__file__), target / Path(__file__).name)
    print(target)


if __name__ == "__main__":
    main()
