#!/usr/bin/env python3
"""
Crypto Warehouse - Weekly Update System

Features:
- Downloads and updates crypto OHLCV data from Yahoo Finance
- Stores one CSV per asset
- Maintains metadata for incremental updates
- Supports warehouse status reporting
"""

import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import yfinance as yf

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Path("crypto_data")
METADATA_FILE = DATA_DIR / "_crypto_metadata.json"
CRYPTO_JSON_FILE = Path("crypto_symbols.json")
UPDATE_INTERVAL_DAYS = 7

DEFAULT_CRYPTO_LIST = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Solana": "SOL",
}

KEEP_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


# =============================================================================
# HELPERS
# =============================================================================

def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json_file(path: Path, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as e:
        print(f"Warning: invalid JSON in {path}: {e}")
        return default


def save_json_file(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_crypto_list() -> Dict[str, str]:
    data = load_json_file(CRYPTO_JSON_FILE, None)

    if data is None:
        print(f"Warning: {CRYPTO_JSON_FILE} not found, using demo list.")
        return DEFAULT_CRYPTO_LIST.copy()

    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}

    if isinstance(data, list):
        out = {}
        for item in data:
            if isinstance(item, dict) and "name" in item and "symbol" in item:
                out[str(item["name"])] = str(item["symbol"])
        if out:
            return out

    print("Warning: crypto_symbols.json format not recognized, using demo list.")
    return DEFAULT_CRYPTO_LIST.copy()


def load_metadata() -> Dict[str, Dict[str, Any]]:
    meta = load_json_file(METADATA_FILE, {})
    return meta if isinstance(meta, dict) else {}


def save_metadata(meta: Dict[str, Dict[str, Any]]) -> None:
    save_json_file(METADATA_FILE, meta)


def normalize_yf_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if "-" in symbol:
        return symbol
    return f"{symbol}-USD"


def parse_date_safe(value: str) -> datetime:
    return datetime.fromisoformat(value)


def needs_update(symbol: str, meta: Dict[str, Dict[str, Any]]) -> bool:
    if symbol not in meta:
        return True

    try:
        last_update = parse_date_safe(meta[symbol]["last_update"])
    except Exception:
        return True

    return (datetime.now() - last_update).days >= UPDATE_INTERVAL_DAYS


def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # yfinance may return DatetimeIndex or Date index-like column after reset_index
    df.reset_index(inplace=True)

    date_col = None
    if "Date" in df.columns:
        date_col = "Date"
    elif "Datetime" in df.columns:
        date_col = "Datetime"
    elif "index" in df.columns:
        date_col = "index"

    if date_col is None:
        raise ValueError("Could not locate date column in downloaded data.")

    if date_col != "Date":
        df.rename(columns={date_col: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = df["Date"].dt.tz_convert(None).dt.strftime("%Y-%m-%d")

    # Keep only standard OHLCV columns if available
    available = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[available].copy()

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(6)

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64")

    df = df.dropna(subset=["Date"])
    df.sort_values("Date", inplace=True)
    df.drop_duplicates(subset="Date", keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# =============================================================================
# CORE UPDATE LOGIC
# =============================================================================

def fetch_and_merge(symbol: str, meta: Dict[str, Dict[str, Any]]) -> bool:
    csv_path = DATA_DIR / f"{symbol}.csv"
    is_first_time = (symbol not in meta) or (not csv_path.exists())

    yf_symbol = normalize_yf_symbol(symbol)

    try:
        ticker = yf.Ticker(yf_symbol)

        if is_first_time:
            print(f"  {symbol}: downloading full history...")
            df_new = ticker.history(period="max", auto_adjust=False)
        else:
            last_date = meta[symbol].get("last_date")
            if not last_date:
                print(f"  {symbol}: missing last_date metadata, downloading full history...")
                df_new = ticker.history(period="max", auto_adjust=False)
                is_first_time = True
            else:
                # Start from next day to avoid duplicate boundary rows
                start_date = (
                    datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                end_date = datetime.now().strftime("%Y-%m-%d")

                if start_date >= end_date:
                    print(f"  {symbol}: already up to date.")
                    return True

                df_new = ticker.history(start=start_date, end=end_date, auto_adjust=False)

        if df_new is None or df_new.empty:
            print(f"  {symbol}: no data found.")
            return False

        df_new = normalize_history_df(df_new)

        if csv_path.exists() and not is_first_time:
            df_old = pd.read_csv(csv_path)
            if not df_old.empty:
                df_old["Date"] = pd.to_datetime(df_old["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
                df_old = df_old.dropna(subset=["Date"])
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_combined = df_new
        else:
            df_combined = df_new

        if df_combined.empty:
            print(f"  {symbol}: combined dataset is empty.")
            return False

        df_combined["Date"] = pd.to_datetime(df_combined["Date"], errors="coerce")
        df_combined = df_combined.dropna(subset=["Date"]).copy()
        df_combined["Date"] = df_combined["Date"].dt.strftime("%Y-%m-%d")

        for col in ["Open", "High", "Low", "Close"]:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce")

        if "Volume" in df_combined.columns:
            df_combined["Volume"] = pd.to_numeric(df_combined["Volume"], errors="coerce").fillna(0)

        df_combined.drop_duplicates(subset="Date", keep="last", inplace=True)
        df_combined.sort_values("Date", inplace=True)
        df_combined.reset_index(drop=True, inplace=True)

        df_combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

        meta[symbol] = {
            "last_update": datetime.now().isoformat(),
            "first_date": str(df_combined["Date"].iloc[0]),
            "last_date": str(df_combined["Date"].iloc[-1]),
            "total_rows": int(len(df_combined)),
            "csv_file": str(csv_path),
            "yf_symbol": yf_symbol,
        }

        if is_first_time:
            print(f"  {symbol}: {len(df_combined):,} rows saved ✓")
        else:
            print(f"  {symbol}: +{len(df_new)} new rows (total: {len(df_combined):,}) ✓")

        return True

    except Exception as e:
        print(f"  {symbol}: failed — {e}")
        return False


def crypto_weekly_update() -> None:
    ensure_data_dir()

    cryptos = load_crypto_list()
    meta = load_metadata()
    symbols = list(dict.fromkeys(cryptos.values()))  # remove duplicates, preserve order
    to_update = [s for s in symbols if needs_update(s, meta)]

    print(f"\n{'=' * 55}")
    print("  Crypto Warehouse — Weekly Update System")
    print(f"{'=' * 55}")
    print(f"  Total Assets      : {len(symbols)}")
    print(f"  Need update       : {len(to_update)}")
    print(f"  Status            : {'READY' if to_update else 'UP TO DATE'}")
    print(f"{'=' * 55}\n")

    if not to_update:
        print("All crypto data is up to date.\n")
        return

    success = 0
    for i, symbol in enumerate(to_update, 1):
        print(f"[{i}/{len(to_update)}] Processing: {symbol}")
        if fetch_and_merge(symbol, meta):
            success += 1

        save_metadata(meta)

        if i < len(to_update):
            time.sleep(random.uniform(1.5, 3.0))

    print(f"\n{'=' * 55}")
    print(f"  Update Complete: {success}/{len(to_update)} coins updated.")
    print(f"{'=' * 55}\n")


# =============================================================================
# LOCAL LOADING
# =============================================================================

def load_local_crypto(symbol: str) -> pd.DataFrame:
    csv_path = DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No local data for {symbol}. Run crypto_weekly_update() first."
        )

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if df.empty:
        raise ValueError(f"Local file for {symbol} is empty.")

    return df


# =============================================================================
# STATUS REPORT
# =============================================================================

def warehouse_status() -> None:
    meta = load_metadata()
    if not meta:
        print("Warehouse is empty.")
        return

    print(f"\n{'=' * 78}")
    print(f"  {'Symbol':<12} {'From':<14} {'To':<14} {'Rows':<10} {'Updated'}")
    print(f"  {'-' * 74}")

    for sym, info in sorted(meta.items()):
        try:
            updated = datetime.fromisoformat(info["last_update"]).strftime("%Y-%m-%d")
        except Exception:
            updated = "unknown"

        first_date = str(info.get("first_date", "unknown"))
        last_date = str(info.get("last_date", "unknown"))
        total_rows = int(info.get("total_rows", 0))

        print(
            f"  {sym:<12} {first_date:<14} {last_date:<14} "
            f"{total_rows:<10,} {updated}"
        )

    print(f"{'=' * 78}\n")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ensure_data_dir()
    crypto_weekly_update()
    warehouse_status()


if __name__ == "__main__":
    main()
