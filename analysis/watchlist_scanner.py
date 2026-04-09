"""
analysis/watchlist_scanner.py — MarketLab Watchlist Scanner
Parallel scan of all tickers in companies.json using local warehouse data.

Usage (from main.py):
    from analysis.watchlist_scanner import scan_watchlist
    scan_watchlist(tickers=None, top_n=20, min_score=4, export=None)
"""
from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import pandas as pd
import pandas_ta as ta
from rich.console import Console
from rich.table import Table
from rich import box

from core.signals       import generate_signal, calculate_atr, calculate_adx
from core.sentiment     import analyze_sentiment
from core.stock_warehouse import load_local, load_companies

console = Console()

MAX_WORKERS  = 8
MIN_ROWS     = 200


# =============================================================================
# INDICATORS
# =============================================================================

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA50"]      = df["Close"].rolling(50).mean()
    df["MA200"]     = df["Close"].rolling(200).mean()
    df["RSI"]       = ta.rsi(df["Close"], length=14)
    df["BB_middle"] = df["Close"].rolling(20).mean()
    df["BB_std"]    = df["Close"].rolling(20).std()
    df["BB_upper"]  = df["BB_middle"] + df["BB_std"] * 2
    df["BB_lower"]  = df["BB_middle"] - df["BB_std"] * 2
    df["MACD"]      = (df["Close"].ewm(span=12, adjust=False).mean()
                       - df["Close"].ewm(span=26, adjust=False).mean())
    df["Signal"]    = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]
    df["L14"]       = df["Low"].rolling(14).min()
    df["H14"]       = df["High"].rolling(14).max()
    denom           = (df["H14"] - df["L14"]).replace(0, 1e-9)
    df["%K"]        = (df["Close"] - df["L14"]) / denom * 100
    df["%D"]        = df["%K"].rolling(3).mean()
    if "Volume" in df.columns:
        df["Volume_Avg"] = df["Volume"].rolling(20).mean()
    df.dropna(inplace=True)
    return df


# =============================================================================
# SINGLE TICKER SCAN
# =============================================================================

def _scan_ticker(symbol: str) -> dict | None:
    try:
        df = load_local(symbol)
        if df is None or len(df) < MIN_ROWS:
            return None

        # set Date as index
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        df = _add_indicators(df)
        if df.empty:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        price      = float(last["Close"])
        change_pct = (price - float(prev["Close"])) / float(prev["Close"]) * 100
        rsi        = float(last.get("RSI", 50))
        volume     = float(last.get("Volume", 0))
        vol_avg    = float(last.get("Volume_Avg", volume)) or 1
        vol_ratio  = round(volume / vol_avg, 2)

        atr = calculate_atr(df)
        adx = calculate_adx(df)

        metrics  = {"Sharpe Annualized": 0.5}
        sig_data = generate_signal(df, {}, metrics)

        if not isinstance(sig_data, dict):
            return None

        signal = str(sig_data.get("signal", "HOLD"))
        score  = int(sig_data.get("score",  0))

        # skip WAIT / ERROR
        if signal in ("WAIT", "ERROR"):
            return None

        # sentiment (fast — no heavy blocking)
        try:
            sent     = analyze_sentiment(symbol)
            sent_lbl = str(sent.get("label", "Neutral"))
        except Exception:
            sent_lbl = "N/A"

        return {
            "symbol"    : symbol,
            "price"     : round(price, 2),
            "change_pct": round(change_pct, 2),
            "signal"    : signal,
            "score"     : score,
            "rsi"       : round(rsi, 1),
            "adx"       : round(adx, 1),
            "atr_pct"   : round((atr / price) * 100, 2),
            "vol_ratio" : vol_ratio,
            "sentiment" : sent_lbl,
        }

    except Exception:
        return None


# =============================================================================
# DISPLAY
# =============================================================================

def _signal_style(signal: str) -> str:
    if "STRONG BUY"  in signal: return "bold green"
    if "BUY"         in signal: return "green"
    if "STRONG SELL" in signal: return "bold red"
    if "SELL"        in signal: return "red"
    return "dim"


def _print_results(results: list[dict], top_n: int, min_score: int) -> None:
    filtered = [r for r in results if r["score"] >= min_score]
    filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)[:top_n]

    if not filtered:
        console.print(f"\n  [yellow]No tickers with score ≥ {min_score}[/yellow]")
        return

    t = Table(
        title=f"Watchlist Scanner — Top {len(filtered)} (score ≥ {min_score})",
        box=box.SIMPLE_HEAVY,
        border_style="dim green",
        header_style="bold white",
        show_lines=False,
    )

    t.add_column("Symbol",    style="bold",    width=8)
    t.add_column("Price",     justify="right", width=9)
    t.add_column("Chg%",      justify="right", width=7)
    t.add_column("Signal",    width=13)
    t.add_column("Score",     justify="right", width=6)
    t.add_column("RSI",       justify="right", width=5)
    t.add_column("ADX",       justify="right", width=5)
    t.add_column("ATR%",      justify="right", width=6)
    t.add_column("Vol×",      justify="right", width=5)
    t.add_column("Sentiment", width=10)

    for r in filtered:
        chg_style  = "green" if r["change_pct"] >= 0 else "red"
        sig_style  = _signal_style(r["signal"])
        sent_style = (
            "green" if r["sentiment"] == "Positive"
            else ("red" if r["sentiment"] == "Negative" else "dim")
        )

        t.add_row(
            r["symbol"],
            f"${r['price']:,.2f}",
            f"[{chg_style}]{r['change_pct']:+.2f}%[/{chg_style}]",
            f"[{sig_style}]{r['signal']}[/{sig_style}]",
            f"{r['score']:+d}",
            f"{r['rsi']:.1f}",
            f"{r['adx']:.1f}",
            f"{r['atr_pct']:.2f}%",
            f"{r['vol_ratio']:.1f}x",
            f"[{sent_style}]{r['sentiment'][:8]}[/{sent_style}]",
        )

    console.print()
    console.print(t)

    buy_n  = sum(1 for r in filtered if "BUY"  in r["signal"])
    sell_n = sum(1 for r in filtered if "SELL" in r["signal"])
    console.print(
        f"  [dim]Shown: {len(filtered)}  |  "
        f"BUY [/dim][green]{buy_n}[/green]"
        f"[dim]  |  SELL [/dim][red]{sell_n}[/red]"
    )


# =============================================================================
# EXPORT
# =============================================================================

def _export_csv(results: list[dict], path: str) -> None:
    if not results:
        return
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        console.print(f"  [dim]Exported → {path}[/dim]")
    except Exception as e:
        console.print(f"  [red]Export failed: {e}[/red]")


# =============================================================================
# MAIN ENTRY
# =============================================================================

def scan_watchlist(
    tickers  : Optional[list[str]] = None,
    top_n    : int = 20,
    min_score: int = 4,
    export   : Optional[str] = None,
) -> list[dict]:
    """
    Scan tickers in parallel using local warehouse data.

    Args:
        tickers  : list of symbols — None = all from companies.json
        top_n    : max results to display
        min_score: minimum signal score to include
        export   : CSV file path to save results (None = skip)

    Returns:
        list of result dicts sorted by score descending
    """
    if tickers is None:
        companies = load_companies()
        tickers   = list(companies.values())

    total = len(tickers)
    console.print(f"\n  [dim]Scanning {total} tickers "
                  f"(workers={MAX_WORKERS}, min_score={min_score})…[/dim]")

    results  : list[dict] = []
    errors   : int        = 0
    done     : int        = 0
    start_ts = datetime.now()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_scan_ticker, sym): sym for sym in tickers}
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception:
                errors += 1

            # progress every 25 tickers
            if done % 25 == 0 or done == total:
                pct = done / total * 100
                console.print(
                    f"  [dim]Progress: {done}/{total} ({pct:.0f}%) "
                    f"— found {len(results)} signals[/dim]",
                    end="\r",
                )

    elapsed = (datetime.now() - start_ts).total_seconds()
    console.print(
        f"\n  [dim]Done in {elapsed:.1f}s — "
        f"{len(results)} signals / {errors} errors[/dim]"
    )

    # sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    _print_results(results, top_n, min_score)

    if export:
        _export_csv(results, export)

    return results