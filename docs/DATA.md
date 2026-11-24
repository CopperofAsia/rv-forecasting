# Getting the data

This project uses the **HF Data Library** 1-minute U.S. equity bars, built from
IEX Exchange HIST trade reports. The Hugging Face repo is **documentation only**;
the actual bars are downloaded from the library website after free registration.
Each team member must obtain the data under their own account — it cannot be
redistributed through this repo (IEX terms). The `data/` directory is gitignored.

- Dataset DOI: `10.5281/zenodo.19501604`
- Website: https://hfdatalibrary.com
- Docs / data dictionary: https://hfdatalibrary.com/pages/docs
- REST API: https://api.hfdatalibrary.com

## Step 1 — Register and agree to terms

Create a free account at https://hfdatalibrary.com and accept the IEX Historical
Data Terms of Use. The data is for coursework research only.

## Step 2 — Pick ONE shared snapshot for the whole team

The library is corrected and updated over time, and files downloaded **before
2026-10-06** carry a different dividend adjustment. Mixing the two corrupts
returns. So:

1. Agree on a single download date across the four of us.
2. Everyone downloads the **Clean**, **1-minute** version for the same tickers.
3. Record the date in `configs/*.yaml` under `data.snapshot.download_date`.

Do not re-download partway through the project.

## Step 3 — Download the 1-minute Clean bars

The data endpoints need a **verified email** and a **complete profile**
(institution, country, role) or they return `403`. The REST API serves each
ticker's **full history** as one Parquet file (no server-side date filtering —
you slice locally). Base URL `https://api.hfdatalibrary.com/v1`; key goes in the
`X-API-Key` header and expires every 30 days; limit 100 downloads/min.

### Option A — batch download with the helper script (recommended)

```bash
export HFDL_API_KEY="your-key-here"        # from the website, after registering
# specific tickers:
python -I scripts/download_data.py --tickers AAPL MSFT NVDA SPY
# or the first N of the whole 1,253-ticker universe (for the liquidity audit):
python -I scripts/download_data.py --top 200
```

It writes one `data/raw/<TICKER>.parquet` per ticker, skips files already
present (resumable), stays under the rate limit, and writes
`data/raw/_manifest.json` with the download date + version. Put that date into
`configs/example.yaml` under `data.snapshot.download_date`.

### Option B — single ticker by hand (curl)

```bash
curl -H "X-API-Key: $HFDL_API_KEY" -o data/raw/AAPL.parquet \
  "https://api.hfdatalibrary.com/v1/bars/AAPL?version=clean"
```

### Option C — website

Download the **1-minute**, **Clean**, **Parquet** files from the Download page
and drop them in `data/raw/`.

Either way each file has columns `datetime` (Eastern Time), `Open`, `High`,
`Low`, `Close`, `Volume`, `source` — the schema the loader expects (override in
`rvf/config.py:SchemaConfig` if the library renames anything). The loader strips
a `_clean`/`_raw` suffix from filenames, so `AAPL_clean.parquet` reads as `AAPL`.

Optionally grab the **daily academic variables** (`/variables/{ticker}`) or the
**quality report** (`/quality/{ticker}`) — they publish the gap-rate/observed-bar
indicators we screen on. The loader recomputes these itself, so neither is
required.

## Step 4 — Verify without the real data (optional)

You can exercise the whole pipeline on a synthetic fixture first:

```bash
python -I scripts/make_synthetic_data.py --out data/raw --n-tickers 8 --days 90 --format csv
python -I scripts/build_dataset.py --config configs/example.yaml --out-format csv
```

(Use `file_glob: "*.csv"` in the config for the synthetic CSVs; `"*.parquet"`
for the real download.)

## Step 5 — Build the dataset

```bash
python -I scripts/build_dataset.py --config configs/example.yaml
```

Outputs: `results/liquidity_audit.csv`, `results/discard_report.csv`,
`results/partition.json`, and `data/processed/windows.parquet`.

## Attribution (required if you share any derived data)

> Data provided for free by IEX. By accessing or using IEX Historical Data, you
> agree to the IEX Historical Data Terms of Use
> (https://www.iex.io/legal/hist-data-terms).

IEX is ~2–3% of consolidated U.S. volume, so results describe an IEX-based
research dataset, not full-market volatility.
