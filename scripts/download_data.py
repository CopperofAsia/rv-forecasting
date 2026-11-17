"""Batch-download 1-minute Clean bars from the HF Data Library REST API.

You must register (free) at https://hfdatalibrary.com/pages/download, verify
your email, and complete your profile (institution, country, role) -- otherwise
the data endpoints return 403. Then copy your API key and run:

    export HFDL_API_KEY="your-key-here"
    # a few named tickers:
    python -I scripts/download_data.py --tickers AAPL MSFT NVDA SPY
    # or the first N of the whole universe (good for the liquidity audit):
    python -I scripts/download_data.py --top 200

Each call fetches a ticker's FULL history as one Parquet file into data/raw/,
named <TICKER>.parquet so the loader reads the symbol cleanly. The script is
resumable (existing files are skipped) and stays under the 100 downloads/minute
limit. It also writes data/raw/_manifest.json recording the download date and
version -- keep this; the whole team must share one snapshot (files downloaded
before 2026-10-06 used a different dividend adjustment and must not be mixed).

Uses only `requests` (already in requirements-lock.txt). The API key is read
from the environment and never written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

import requests

BASE = "https://api.hfdatalibrary.com/v1"
MIN_INTERVAL = 0.65  # seconds between downloads -> < 100/min with margin


def list_universe(max_retries: int = 4) -> list[str]:
    """Public endpoint, no key needed: every available ticker. Retries on the
    flaky-network errors (this call is otherwise a single point of failure)."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(f"{BASE}/symbols", timeout=120)
            r.raise_for_status()
            return [s["ticker"] for s in r.json()["symbols"]]
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            wait = min(2 ** attempt, 30)
            print(f"  /symbols network error ({type(e).__name__}), "
                  f"retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)
    raise SystemExit("Could not reach /symbols after retries. Your connection to "
                     "api.hfdatalibrary.com is unstable -- try --tickers-from "
                     "data/raw/_manifest.json to reuse the Clean ticker list.")


def tickers_from_manifest(path: str) -> list[str]:
    """Reuse the exact ticker list from an existing download manifest (e.g. the
    Clean run's). Avoids the /symbols call and guarantees the same stock set."""
    with open(path) as fh:
        return list(json.load(fh)["tickers"])


def _fetch_to_file(ticker: str, path: str, version: str, key: str) -> None:
    """One streamed download attempt. Writes to a .part file and verifies the
    byte count against Content-Length before promoting it, so a connection that
    drops mid-stream can never leave a truncated .parquet in place."""
    tmp = path + ".part"
    with requests.get(f"{BASE}/bars/{ticker}", params={"version": version},
                      headers={"X-API-Key": key}, timeout=120, stream=True) as r:
        if r.status_code == 403:
            raise SystemExit("403 Forbidden: verify your email and complete your "
                             "profile (institution, country, role) on the website.")
        if r.status_code == 401:
            raise SystemExit("401 Unauthorized: HFDL_API_KEY missing/invalid/"
                             "expired (keys expire every 30 days -- regenerate).")
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", "5")))
            raise ConnectionError("rate limited")  # treated as retryable below
        r.raise_for_status()
        expected = int(r.headers.get("Content-Length", "0"))
        written = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                fh.write(chunk)
                written += len(chunk)
    if expected and written < expected:
        os.remove(tmp)
        raise ConnectionError(f"incomplete: {written}/{expected} bytes")
    os.replace(tmp, path)


def download_one(ticker: str, out_dir: str, version: str, key: str,
                 max_retries: int = 4) -> str:
    path = os.path.join(out_dir, f"{ticker}.parquet")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skip"
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            _fetch_to_file(ticker, path, version, key)
            return "ok"
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                ConnectionError) as e:
            last = e
            wait = min(2 ** attempt, 30)  # 2,4,8,16s backoff
            print(f"    {ticker}: network error ({type(e).__name__}), "
                  f"retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"{ticker}: failed after {max_retries} retries ({last})")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tickers", nargs="+", help="explicit ticker symbols")
    g.add_argument("--top", type=int, help="first N tickers of the universe")
    g.add_argument("--tickers-from", default=None,
                   help="reuse the ticker list from an existing manifest JSON "
                        "(e.g. data/raw/_manifest.json) -- skips the /symbols call "
                        "and guarantees the same stock set as that run")
    ap.add_argument("--version", choices=["clean", "raw"], default="clean")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    key = os.environ.get("HFDL_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set HFDL_API_KEY first:  export HFDL_API_KEY='...'")

    os.makedirs(args.out, exist_ok=True)
    if args.tickers:
        tickers = args.tickers
    elif args.tickers_from:
        tickers = tickers_from_manifest(args.tickers_from)
    else:
        tickers = list_universe()[: args.top]
    print(f"Downloading {len(tickers)} tickers (version={args.version}) -> {args.out}")

    counts = {"ok": 0, "skip": 0}
    for i, tk in enumerate(tickers, 1):
        status = download_one(tk, args.out, args.version, key)
        counts[status] += 1
        print(f"[{i}/{len(tickers)}] {tk}: {status}")
        if status == "ok":
            time.sleep(MIN_INTERVAL)

    manifest = {
        "download_date": date.today().isoformat(),
        "version": args.version,
        "dataset_doi": "10.5281/zenodo.19501604",
        "n_tickers": len(tickers),
        "tickers": sorted(tickers),
    }
    with open(os.path.join(args.out, "_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Done: {counts['ok']} downloaded, {counts['skip']} already present.")
    print(f"Manifest written. Put download_date={manifest['download_date']} in "
          f"your config's data.snapshot.")


if __name__ == "__main__":
    main()
