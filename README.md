# rv-forecasting

Data-efficient realized-volatility forecasting via cross-stock transfer learning
(DDA4220 project). The question: does pretraining a small MLP on *source* stocks
let a held-out *target* stock reach a good 30-minute realized-volatility forecast
with less of its own history than training from scratch?

See `docs/DDA4220_Project_Proposal.pdf` for the full design and
`docs/DATA.md` for how to obtain the dataset.

## Status

Implemented: **Phases 0–6**. The data layer (loading, liquidity audit,
window/label construction, chronological splits with a disjoint source/target
partition and shared leakage cutoff, feature scaling, nested fractions), the
baselines (persistence, linear, HAR), the target-only MLP, source pretraining +
fine-tuning (full and output-layer-only), the RMSPE / data-efficiency metrics,
and the experiment orchestrator with RMSPE-vs-fraction plots. Tuning the
liquidity thresholds on the real audit and writing up results remain.

## Layout

```
rvf/
  config.py            prespecified constants (fractions, 1.05x criterion, splits)
  data/
    calendar.py        session + early-close exclusions
    loader.py          load 1-min bars; per-stock liquidity audit + screening
    windows.py         feature windows + RV30 -> sigma30 labels (no look-ahead)
    splits.py          source/target partition, shared cutoff, scaler, fractions
    datamodule.py      assemble (X,y) arrays per stage; scaler fit on permitted rows
  models/
    baselines.py       persistence, linear regression, HAR-style
    mlp.py             small MLP, softplus nonnegative output
  training/
    trainer.py         Adam + prespecified LR/epoch grid, early stop, multi-seed
    transfer.py        source pretrain, fine-tune (full/head), zero-shot 0% point
  eval/
    metrics.py         RMSPE (+zero-label handling), D_m(f), efficiency threshold
    experiment.py      methods x fractions x targets x seeds comparison
    plots.py           RMSPE-vs-fraction curves
  utils/seeding.py     reproducible seeding
configs/example.yaml   per-run overrides (set the data snapshot here)
scripts/
  download_data.py        batch-download bars from the HF Data Library API
  make_synthetic_data.py  synthetic fixture to test without the real download
  build_dataset.py        Phase-1 driver: raw bars -> processed windows + reports
  run_experiments.py      Phase 2-6 driver: comparison -> results + plots
tests/
  test_data_layer.py      leakage / label / fraction checks
  test_models_eval.py     metrics / baselines / training smoke test
```

## Quick start (synthetic, no download needed)

```bash
pip install -r requirements.txt            # or: numpy pandas pyyaml pyarrow
python -I scripts/make_synthetic_data.py --out data/raw --n-tickers 8 --days 90 --format csv
python -I tests/test_data_layer.py         # all checks should pass
python -I scripts/build_dataset.py --config configs/example.yaml --out-format csv
```

With the real IEX data, download per `docs/DATA.md`, set
`data.snapshot.download_date` in the config, use `file_glob: "*.parquet"`, and
drop `--out-format csv`.

## Full run (real data)

```bash
# 1. download (see docs/DATA.md) -> data/raw/*.parquet
# 2. build the dataset (~2-3 min for ~200 tickers)
python -I scripts/build_dataset.py --config configs/example.yaml
# 3. inspect results/liquidity_audit.csv and results/discard_report.csv,
#    then FIX the liquidity thresholds in configs/example.yaml and rebuild.
# 4. quick plumbing check, then the full sweep:
python -I scripts/run_experiments.py --config configs/example.yaml --quick
python -I scripts/run_experiments.py --config configs/example.yaml
```

Outputs: `results/experiment_results.csv`, `results/efficiency_summary.csv`,
and `results/rmspe_curve_<TICKER>.png`.

### Report artifacts & secondary analyses

```bash
# aggregate tables + plots for the write-up (fast); add --source-count for the
# source-stock-count study (slower):
python -I scripts/run_analysis.py --config configs/example.yaml --source-count
```

### Raw-vs-Clean robustness (optional)

Download the SAME tickers with `--version raw` into `data/raw_version/`, then
build reusing the Clean partition so the comparison is apples-to-apples:

```bash
python -I scripts/download_data.py --top 200 --version raw --out data/raw_version
python -I scripts/build_dataset.py  --config configs/raw_version.yaml \
    --reuse-partition results/partition.json
python -I scripts/run_experiments.py --config configs/raw_version.yaml
python -I scripts/compare_versions.py     # writes results/report/clean_vs_raw.*
```

## Leakage controls (why the data layer is built this way)

- Features at decision time *t* use only bars at/before *t*; the label uses only
  the future *(t, t+30min]* window. The two never share a bar.
- Decision points step by the horizon, so feature windows never contain another
  window's future label.
- Source and target stock sets are disjoint. Source pretraining uses only bars
  strictly before the shared `val_start` cutoff, so nothing from the target
  evaluation period leaks in through another stock.
- Nested fractions take the earliest chronological rows, so each smaller budget
  is a prefix of the larger one; the same rows feed scratch and fine-tuning.
- The feature standardizer is fit only on rows permitted to the training stage.
