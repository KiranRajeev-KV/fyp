#!/usr/bin/env python3
"""
Download BTCUSDT 1h monthly kline CSVs from Binance Vision and unzip
into ppo/data/raw/.

Usage:
    python download_klines.py

Requires: requests
    pip install requests
"""

import io
import zipfile
from pathlib import Path
from datetime import date
import csv

import requests

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
START = (2017, 8)     # BTCUSDT 1h data on Binance starts here
END = (2026, 6)       # last complete month per your listing

OUT_DIR = Path("ppo/data/raw")
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}.zip"

# Binance switched some CSVs to include a header row at some point.
# We detect this per-file and normalize so every output CSV is headerless,
# OR you can flip NORMALIZE_HEADERS=False if your notebook expects headers.
NORMALIZE_TO_HEADERLESS = True
EXPECTED_COLUMNS = 12  # open_time, open, high, low, close, volume, close_time,
                       # quote_asset_volume, number_of_trades,
                       # taker_buy_base_volume, taker_buy_quote_volume, ignore


def month_range(start, end):
    y, m = start
    ey, em = end
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def has_header(first_row):
    # Header row has non-numeric first field (e.g. "open_time")
    try:
        float(first_row[0])
        return False
    except (ValueError, IndexError):
        return True


def process_zip_bytes(zbytes, out_path):
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        names = zf.namelist()
        csv_name = next((n for n in names if n.endswith(".csv")), None)
        if not csv_name:
            raise ValueError(f"No CSV found in zip, contents: {names}")
        raw = zf.read(csv_name).decode("utf-8")

    rows = list(csv.reader(io.StringIO(raw)))
    if not rows:
        raise ValueError("Empty CSV")

    header_present = has_header(rows[0])
    data_rows = rows[1:] if header_present else rows

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        if not NORMALIZE_TO_HEADERLESS and header_present:
            writer.writerow(rows[0])
        writer.writerows(data_rows)

    return header_present, len(data_rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []

    for year, month in month_range(START, END):
        url = BASE_URL.format(symbol=SYMBOL, interval=INTERVAL, year=year, month=month)
        out_name = f"{SYMBOL}-{INTERVAL}-{year}-{month:02d}.csv"
        out_path = OUT_DIR / out_name

        if out_path.exists():
            print(f"[skip] {out_name} already exists")
            continue

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                print(f"[404]  {url} (month not available yet, skipping)")
                failed.append((year, month, "404"))
                continue
            resp.raise_for_status()

            had_header, n_rows = process_zip_bytes(resp.content, out_path)
            flag = " (had header, normalized)" if had_header else ""
            print(f"[ok]   {out_name} -> {n_rows} rows{flag}")
            ok.append((year, month))

        except Exception as e:
            print(f"[fail] {year}-{month:02d}: {e}")
            failed.append((year, month, str(e)))

    print("\n--- Summary ---")
    print(f"Downloaded: {len(ok)} months")
    print(f"Failed/missing: {len(failed)} months")
    if failed:
        for y, m, reason in failed:
            print(f"  {y}-{m:02d}: {reason}")


if __name__ == "__main__":
    main()
