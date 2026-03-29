import yfinance as yf
import pandas as pd
import json
import time
import random
from datetime import datetime
from pathlib import Path

DATA_DIR        = Path("data")
METADATA_FILE   = DATA_DIR / "_metadata.json"
UPDATE_INTERVAL = 7  # days between updates


def load_companies() -> dict:
    try:
        with open("companies.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: companies.json not found, using demo list.")
        return {"Apple": "AAPL", "Microsoft": "MSFT", "Tesla": "TSLA"}


def load_metadata() -> dict:
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_metadata(meta: dict):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def needs_update(symbol: str, meta: dict) -> bool:
    if symbol not in meta:
        return True
    last = datetime.fromisoformat(meta[symbol]["last_update"])
    return (datetime.now() - last).days >= UPDATE_INTERVAL


def fetch_and_merge(symbol: str, meta: dict) -> bool:
    csv_path      = DATA_DIR / f"{symbol}.csv"
    is_first_time = symbol not in meta or not csv_path.exists()

    try:
        ticker = yf.Ticker(symbol)

        if is_first_time:
            print(f"  {symbol}: downloading full history...")
            df_new = ticker.history(period="max")
        else:
            start_date = meta[symbol]["last_date"]
            end_date   = datetime.now().strftime("%Y-%m-%d")
            if start_date >= end_date:
                print(f"  {symbol}: already up to date.")
                return True
            df_new = ticker.history(start=start_date, end=end_date)

        if df_new.empty:
            print(f"  {symbol}: no data found.")
            return False

        # clean columns
        df_new.reset_index(inplace=True)
        date_col = "Date" if "Date" in df_new.columns else "Datetime"
        df_new.rename(columns={date_col: "Date"}, inplace=True)
        df_new["Date"] = pd.to_datetime(df_new["Date"]).dt.tz_localize(None)
        df_new["Date"] = df_new["Date"].dt.strftime("%Y-%m-%d")

        keep   = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df_new = df_new[[c for c in keep if c in df_new.columns]].copy()
        for col in ["Open", "High", "Low", "Close"]:
            if col in df_new.columns:
                df_new[col] = df_new[col].round(4)

        # merge with existing data
        if csv_path.exists() and not is_first_time:
            df_old = pd.read_csv(csv_path)
            df_old["Date"] = pd.to_datetime(df_old["Date"]).dt.strftime("%Y-%m-%d")
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.drop_duplicates(subset="Date", keep="last", inplace=True)
        df_combined.sort_values("Date", inplace=True)
        df_combined.reset_index(drop=True, inplace=True)

        df_combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

        meta[symbol] = {
            "last_update": datetime.now().isoformat(),
            "first_date" : df_combined["Date"].iloc[0],
            "last_date"  : df_combined["Date"].iloc[-1],
            "total_rows" : len(df_combined),
            "csv_file"   : str(csv_path)
        }

        if is_first_time:
            print(f"  {symbol}: {len(df_combined):,} rows saved "
                  f"({meta[symbol]['first_date']} to today) ✓")
        else:
            print(f"  {symbol}: +{len(df_new)} new rows "
                  f"(total: {len(df_combined):,}) ✓")
        return True

    except Exception as e:
        print(f"  {symbol}: failed — {e}")
        return False


def weekly_update():
    DATA_DIR.mkdir(exist_ok=True)
    companies  = load_companies()
    meta       = load_metadata()
    symbols    = list(companies.values())
    to_update  = [s for s in symbols if needs_update(s, meta)]

    print(f"\n{'='*50}")
    print(f"  Stock Warehouse — Weekly Update")
    print(f"{'='*50}")
    print(f"  Total symbols     : {len(symbols)}")
    print(f"  Need update       : {len(to_update)}")
    print(f"  Already up to date: {len(symbols) - len(to_update)}")
    print(f"{'='*50}\n")

    if not to_update:
        print("All data is up to date. Nothing to download.\n")
        return

    success = 0
    for i, symbol in enumerate(to_update, 1):
        print(f"[{i}/{len(to_update)}] {symbol}")
        if fetch_and_merge(symbol, meta):
            success += 1

        save_metadata(meta)  # save after each symbol

        if i < len(to_update):
            time.sleep(random.uniform(2, 4))

        if i % 20 == 0 and i < len(to_update):
            print("\n  Pausing for 1 minute...\n")
            time.sleep(60)

    print(f"\n{'='*50}")
    print(f"  Done: {success}/{len(to_update)} symbols updated.")
    print(f"  Next update in {UPDATE_INTERVAL} days.")
    print(f"{'='*50}\n")


def load_local(symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
    """Read a stock's local CSV — no internet needed."""
    csv_path = DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No local data for {symbol}. Run weekly_update() first.")
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def warehouse_status():
    """Print a summary of all locally stored data."""
    meta = load_metadata()
    if not meta:
        print("Warehouse is empty. Run weekly_update() first.")
        return

    total_rows = sum(v["total_rows"] for v in meta.values())
    print(f"\n{'='*66}")
    print(f"  Warehouse Status — {len(meta)} symbols — {total_rows:,} rows total")
    print(f"{'='*66}")
    print(f"  {'Symbol':<10} {'From':<14} {'To':<14} {'Rows':<10} {'Last Update'}")
    print(f"  {'-'*60}")
    for sym, info in sorted(meta.items()):
        updated = datetime.fromisoformat(info["last_update"]).strftime("%Y-%m-%d")
        print(f"  {sym:<10} {info['first_date']:<14} {info['last_date']:<14} "
              f"{info['total_rows']:<10,} {updated}")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    weekly_update()
    warehouse_status()
