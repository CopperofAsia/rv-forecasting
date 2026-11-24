"""Data-efficient realized-volatility forecasting via cross-stock transfer learning.

DDA4220 project. This package currently implements Phase 0 (scaffold/config) and
Phase 1 (data layer): loading, liquidity audit, window/label construction,
chronological splits with source/target partition, feature scaling, and nested
target-data fractions. Models/training/eval come in later phases.
"""
from __future__ import annotations

from .config import Config, load_config

__all__ = ["Config", "load_config"]
