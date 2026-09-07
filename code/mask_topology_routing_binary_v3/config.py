from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V2_UTILS = PROJECT_ROOT / "02_code" / "02_models" / "mask_topology_routing" / "utils.py"
DEFAULT_RULEBOOK = (
    PROJECT_ROOT
    / "04_results"
    / "mask_topology_routing"
    / "gt_binary_topology_rulebook_20260731"
    / "gt_binary_topology_rulebook.json"
)
EXPECTED_V2_UTILS_SHA256 = "541fa8098f4edb961178cb3e1a21785d08e0dcd9a4a6e59c472ea39de0e13d9f"
SCHEMA_VERSION = "3.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BinaryV3Config:
    schema_version: str
    rulebook_path: Path
    rulebook_sha256: str
    rulebook: dict[str, Any]
    structural_tolerance_px: float = 24.0
    strict_tolerance_px: float = 5.0
    explicit_root_tolerance_px: float = 3.0
    repair_max_blank_px: float = 3.0
    enable_postclip_connectivity_repair: bool = True
    enable_internal_trunk_termination: bool = True
    enable_supported_root_family_merge: bool = True

    def distribution(self, section: str, field: str) -> dict[str, float]:
        value = self.rulebook["summary"][section][field]
        return {str(key): float(item) for key, item in value.items() if isinstance(item, (int, float))}

    def source_ref(self, section: str, field: str, statistic: str) -> dict[str, str]:
        return {
            "rulebook": str(self.rulebook_path),
            "field": f"summary.{section}.{field}.{statistic}",
        }


def load_binary_v3_config(rulebook_path: Path | None = None, verify_v2: bool = True) -> BinaryV3Config:
    selected = (rulebook_path or DEFAULT_RULEBOOK).resolve()
    if not selected.exists():
        raise FileNotFoundError(f"Binary v3 rulebook not found: {selected}")
    if verify_v2:
        observed = sha256_file(V2_UTILS)
        if observed != EXPECTED_V2_UTILS_SHA256:
            raise RuntimeError(
                "Frozen v2 routing source changed; refusing to run v3 wrapper. "
                f"expected={EXPECTED_V2_UTILS_SHA256}, observed={observed}"
            )
    rulebook = json.loads(selected.read_text(encoding="utf-8"))
    if int(rulebook.get("summary", {}).get("samples_total", 0)) != 100:
        raise ValueError("Binary v3 requires the complete 100-sample topology rulebook")
    if int(rulebook["summary"].get("samples_structurally_valid", 0)) != 99:
        raise ValueError("Expected 99 structurally valid samples plus one audited empty sample")
    return BinaryV3Config(
        schema_version=SCHEMA_VERSION,
        rulebook_path=selected,
        rulebook_sha256=sha256_file(selected),
        rulebook=rulebook,
    )
