from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "generate_learnable_delta_figure.py"
    spec = importlib.util.spec_from_file_location("generate_learnable_delta_figure", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_fixed_hr10_from_csv_supports_both_column_names(tmp_path: Path) -> None:
    mod = _load_script_module()

    csv_path = tmp_path / "fixed_metrics.csv"
    csv_path.write_text(
        "dataset,hr_at_10,fixed_hr10\n"
        "ML-1M,0.12,\n"
        "Beauty,,0.33\n",
        encoding="utf-8",
    )

    out = mod.load_fixed_hr10_from_csv(csv_path)
    assert out["ML-1M"] == 0.12
    assert out["Beauty"] == 0.33


def test_resolve_fixed_hr10_source_uses_explicit_csv(tmp_path: Path) -> None:
    mod = _load_script_module()

    csv_path = tmp_path / "fixed_metrics.csv"
    csv_path.write_text(
        "dataset,hr_at_10\n"
        "ML-1M,0.2\n"
        "Steam,0.8\n",
        encoding="utf-8",
    )

    fixed_map, fixed_source, warning_msg = mod.resolve_fixed_hr10_source(csv_path)
    assert fixed_source.startswith("csv:")
    assert warning_msg is None
    assert fixed_map["ML-1M"] == 0.2
    assert fixed_map["Steam"] == 0.8


def test_resolve_fixed_hr10_source_falls_back_to_constants_when_no_csv(tmp_path: Path) -> None:
    mod = _load_script_module()

    # Point default path to a non-existing temporary file for deterministic behavior.
    mod.DEFAULT_FIXED_METRICS_CSV = tmp_path / "missing_default.csv"

    fixed_map, fixed_source, warning_msg = mod.resolve_fixed_hr10_source(None)
    assert fixed_source == "table_constants"
    assert warning_msg is not None
    assert "built-in FIXED_HR10" in warning_msg
    assert fixed_map["ML-1M"] == mod.FIXED_HR10["ML-1M"]
