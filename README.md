# rv-forecasting

Data-efficient realized-volatility forecasting via cross-stock transfer learning
(DDA4220 project). The question: does pretraining a small MLP on *source* stocks
let a held-out *target* stock reach a good 30-minute realized-volatility forecast
with less of its own history than training from scratch?

See `docs/DDA4220_Project_Proposal.pdf` for the full design and
`docs/DATA.md` for how to obtain the dataset.

## Status

Implemented so far: **Phase 0 (scaffold/config)** and **Phase 1 (data layer)** —
loading, liquidity audit, window/label construction, chronological splits with a
disjoint source/target partition and shared leakage cutoff, feature scaling, and
nested target-data fractions. Models, training, and evaluation (Phases 2–5) are
next.

## Layout

```
rvf/
  config.py            prespecified constants (fractions, 1.05x criterion, splits)
  data/
    calendar.py        session + early-close exclusions
    loader.py          load 1-min bars; per-stock liquidity audit + screening
    windows.py         feature windows + RV30 -> sigma30 labels (no look-ahead)
    splits.py          source/target partition, shared cutoff, scaler, fractions
  utils/seeding.py     reproducible seeding
configs/example.yaml   per-run overrides (set the data snapshot here)
scripts/
  make_synthetic_data.py  synthetic fixture to test without the real download
  build_dataset.py        Phase-1 driver: raw bars -> processed windows + reports
tests/test_data_layer.py  leakage / label / fraction checks
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
