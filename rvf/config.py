"""Central configuration for the RV-forecasting project.

Every value here is *prespecified*: it encodes a decision made in the project
proposal before any test results were seen. Keeping these constants in one place
(rather than scattered as magic numbers) is a leakage/honesty control -- it makes
it auditable that thresholds, data fractions and the success criterion were fixed
in advance and not adjusted after seeing results.

The one thing you MUST set per machine/snapshot is DataConfig.raw_dir and the
snapshot metadata (download date + cleaning version). See configs/*.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Prespecified experimental constants (proposal sections 3.3, 5.1, 5.2)
# ---------------------------------------------------------------------------

# Nested, earliest-chronological target-training fractions (proposal 3.3).
# 0.0 is the zero-shot reference point for the pretrained model; it is undefined
# for a target-only model (handled downstream, not here).
TARGET_FRACTIONS: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00)

# Data-efficiency success criterion (proposal 5.2): smallest fraction f with
# E_m(f) <= TOLERANCE * E_ref. Prespecified, NOT an expected result.
EFFICIENCY_TOLERANCE: float = 1.05

# Primary forecast horizon in minutes (proposal 3.2). A 10-min target is an
# optional extension only if coverage proves adequate.
HORIZON_MINUTES: int = 30


@dataclass(frozen=True)
class SnapshotConfig:
    """Records WHICH data snapshot produced a result, for reproducibility.

    The HF Data Library is corrected/updated over time, and files downloaded
    before 2026-10-06 used a different dividend adjustment. A result is only
    interpretable alongside these fields.
    """
    download_date: str = "UNSET"          # ISO date the snapshot was downloaded
    cleaning_version: str = "clean"        # "clean" (primary) or "raw"
    dataset_doi: str = "10.5281/zenodo.19501604"
    # Guard against mixing the pre-2026-10-06 dividend-adjusted files.
    dividend_adjustment_regime: str = "post-2026-10-06-split-only"


@dataclass(frozen=True)
class SchemaConfig:
    """Column names in the raw 1-minute files. Configurable so the loader does
    not hard-code a schema the library might rename in a future version."""
    datetime: str = "datetime"   # bar timestamp, Eastern Time
    open: str = "Open"
    high: str = "High"
    low: str = "Low"
    close: str = "Close"
    volume: str = "Volume"
    source: str = "source"
    timezone: str = "America/New_York"


@dataclass(frozen=True)
class LiquidityConfig:
    """Screening thresholds (proposal 3.2). Values are placeholders to be FIXED
    after the exploratory audit and before model comparison. They are expressed
    against the library's daily quality variables."""
    max_gap_rate: float = 0.20            # reject stock-days with >20% missing minutes
    min_observed_bars: int = 300          # of ~390 regular-session minutes
    min_valid_days: int = 60              # a stock needs at least this many good days
    min_dollar_volume: float = 1e6        # Amihud-style liquidity floor


@dataclass(frozen=True)
class WindowConfig:
    """Window-construction rules (proposal 3.2)."""
    horizon_minutes: int = HORIZON_MINUTES
    feature_lookback_minutes: int = 30    # recent 1-min returns used as features
    # Minimum number N of valid consecutive-minute returns required in a window.
    # Chosen during data exploration; windows failing this are discarded (not filled).
    min_valid_returns: int = 20
    regular_open: str = "09:30"
    regular_close: str = "16:00"
    early_close: str = "13:00"            # bars at/after this are dropped on early-close days


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split + source/target partition (proposal 3.3)."""
    target_stock_fraction: float = 0.20   # ~80/20 source/target split
    train_frac: float = 0.60              # within each TARGET stock's timeline
    val_frac: float = 0.20
    test_frac: float = 0.20
    seed: int = 0                         # for the source/target assignment only


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"             # one *.parquet per ticker
    processed_dir: str = "data/processed"
    file_glob: str = "*.parquet"
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    fractions: tuple[float, ...] = TARGET_FRACTIONS
    efficiency_tolerance: float = EFFICIENCY_TOLERANCE
    seeds: tuple[int, ...] = (0, 1, 2)    # evaluation seeds (proposal 4.3/5.1)
    results_dir: str = "results"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: Any, override: dict[str, Any]) -> None:
    """Recursively set attributes on a dataclass from a nested dict."""
    for key, val in override.items():
        if not hasattr(base, key):
            raise KeyError(f"Unknown config key: {key!r} on {type(base).__name__}")
        cur = getattr(base, key)
        if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
            _merge(cur, val)
        elif isinstance(cur, tuple) and isinstance(val, list):
            setattr(base, key, tuple(val))
        else:
            setattr(base, key, val)


def load_config(path: str | None = None) -> Config:
    """Build a Config, optionally overlaying a YAML file over the defaults.

    Only keys present in the YAML override the prespecified defaults, so a
    config file is a small diff that documents exactly what a given run changed.
    Frozen sub-configs are rebuilt from merged dicts so immutability holds.
    """
    cfg = Config()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    # DataConfig holds frozen dataclasses; merge into a plain dict then rebuild.
    data_over = raw.pop("data", {}) or {}
    d = cfg.data
    sub = {
        "snapshot": SnapshotConfig(**{**asdict(d.snapshot), **data_over.pop("snapshot", {})}),
        "schema": SchemaConfig(**{**asdict(d.schema), **data_over.pop("schema", {})}),
        "liquidity": LiquidityConfig(**{**asdict(d.liquidity), **data_over.pop("liquidity", {})}),
        "window": WindowConfig(**{**asdict(d.window), **data_over.pop("window", {})}),
        "split": SplitConfig(**{**asdict(d.split), **data_over.pop("split", {})}),
    }
    cfg.data = DataConfig(**{**{k: v for k, v in asdict(d).items()
                               if k not in sub}, **data_over, **sub})
    _merge(cfg, raw)
    return cfg

