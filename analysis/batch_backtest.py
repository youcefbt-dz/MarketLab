"""
batch_backtest.py — MarketLab Batch Backtest Runner
Silently runs all AUTO_PLANS using the local warehouse to rapidly fill
backtest_history.json for ML training.

Usage:
    python batch_backtest.py              # runs all 8 plans (~40 backtests)
    python batch_backtest.py --plan 1     # runs only plan #1
    python batch_backtest.py --dry-run    # shows what would run, no execution

After completion, backtest_history.json will have enough records for
ml_predictor.py to train meaningfully (target: 50+ records).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from analysis.backtest import (
    AUTO_PLANS,
    COMMISSION,
    SLIPPAGE,
    compute_benchmark,
    compute_metrics,
    fetch_and_prepare,
    fetch_market_returns,
    run_backtest,
    validate_ticker,
    ensure_results_dir,
)
from analysis.backtest_logger import log_backtest_run, get_reliability_report, BacktestLogger

BAR      = "═" * 58
THIN_BAR = "─" * 58


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _current_count() -> int:
    logger = BacktestLogger()
    return logger.get_runs_count()


def _print_plan_table() -> None:
    print(f"\n  {THIN_BAR}")
    print("  AUTO_PLANS available:")
    print(f"  {THIN_BAR}")
    for i, plan in enumerate(AUTO_PLANS, 1):
        t_str = ", ".join(plan["tickers"])
        print(f"  [{i:>2}]  {plan['label']:<38} {plan['start']}  {t_str}")
    print(f"  {THIN_BAR}\n")


def _build_run_queue(
    selected_plans: list[dict],
) -> list[tuple[str, str, str, str]]:
    """
    Returns list of (ticker, start, end, label) tuples to process.
    Only includes tickers present in the warehouse.
    """
    end   = datetime.today().strftime("%Y-%m-%d")
    queue = []
    for plan in selected_plans:
        for ticker in plan["tickers"]:
            if validate_ticker(ticker):
                queue.append((ticker, plan["start"], end, plan["label"]))
            else:
                print(f"  ⚠  {ticker} not in warehouse — skipped.")
    return queue


# ─── CORE RUNNER ──────────────────────────────────────────────────────────────

def run_batch(
    plans       : list[dict],
    initial_cash: float = 10_000,
    annual_rf   : float = 0.04,
    verbose     : bool  = True,
) -> dict:
    """
    Runs all (ticker, period) combinations in `plans` sequentially,
    logs every result to backtest_history.json, and returns a summary dict.
    """
    ensure_results_dir()
    queue      = _build_run_queue(plans)
    total      = len(queue)
    done       = 0
    passed     = 0
    failed     = 0
    skipped    = 0
    start_time = time.time()

    print(f"\n{BAR}")
    print(f"  MarketLab  ·  Batch Backtest Runner")
    print(f"{BAR}")
    print(f"  Plans    : {len(plans)}")
    print(f"  Tickers  : {total} runs queued")
    print(f"  Capital  : ${initial_cash:,.0f}  |  RF: {annual_rf*100:.1f}%")
    print(f"  History  : {_current_count()} records before this batch")
    print(f"{BAR}\n")

    # Group by (start, end) so we fetch market returns once per period
    periods: dict[tuple, list] = {}
    for ticker, start, end, label in queue:
        key = (start, end)
        periods.setdefault(key, []).append((ticker, label))

    for (start, end), items in periods.items():

        print(f"\n  Period: {start} → {end}  ({len(items)} tickers)")
        print(f"  {THIN_BAR}")

        mkt_returns = fetch_market_returns(start, end)

        for ticker, label in items:
            done += 1
            prefix = f"  [{done:>3}/{total}]  {ticker:<6}  {label}"

            df = fetch_and_prepare(ticker, start, end)
            if df is None:
                print(f"{prefix}  →  ⚠  skipped (insufficient data)")
                skipped += 1
                continue

            try:
                result    = run_backtest(ticker, df, mkt_returns, initial_cash, annual_rf)
                metrics   = compute_metrics(result)
                benchmark = compute_benchmark(ticker, start, end, initial_cash)

                record = log_backtest_run(
                    ticker    = ticker,
                    start     = start,
                    end       = end,
                    metrics   = metrics,
                    benchmark = benchmark,
                    settings  = {
                        "initial_cash": initial_cash,
                        "commission"  : COMMISSION,
                        "slippage"    : SLIPPAGE,
                    },
                    trades = result["trades"],
                )

                verdict = "PASS ✅" if metrics["passed"] else "FAIL ❌"
                ret_str = f"{metrics['total_return']:+.1f}%"
                wr_str  = f"WR {metrics['win_rate']:.0f}%"
                sh_str  = f"Sharpe {metrics['sharpe']:.2f}"

                print(f"{prefix}  →  {verdict}  "
                      f"Return {ret_str}  {wr_str}  {sh_str}  "
                      f"Trades {metrics['total_trades']}")

                if metrics["passed"]:
                    passed += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"{prefix}  →  ✗  error: {e}")
                skipped += 1

    # ── Invalidate ML cache ───────────────────────────────────────────────────
    try:
        from analysis.ml_predictor import invalidate_cache
        invalidate_cache()
    except ImportError:
        pass

    elapsed = time.time() - start_time
    total_runs_now = _current_count()

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{BAR}")
    print(f"  Batch Complete")
    print(f"{BAR}")
    print(f"  Runs completed   : {done - skipped}")
    print(f"  Passed           : {passed}")
    print(f"  Failed           : {failed}")
    print(f"  Skipped          : {skipped}")
    print(f"  Total in history : {total_runs_now}")
    print(f"  Elapsed          : {elapsed:.0f}s  ({elapsed/max(done-skipped,1):.1f}s/run)")
    print(f"{BAR}")

    # ── Reliability snapshot ──────────────────────────────────────────────────
    if total_runs_now >= 5:
        try:
            report  = get_reliability_report()
            overall = report["overall"]
            print(f"\n  Reliability Score : {overall['score']}/100 — {overall['label']}")
            print(f"  Pass Rate         : {overall['raw']['pass_rate']}%")
            print(f"  Avg Win Rate      : {overall['raw']['avg_win_rate']}%")
            print(f"  Trend             : {report['trend']}")
        except Exception:
            pass

    ml_ready = total_runs_now >= 20
    print(f"\n  ML Predictor ready : {'✅ Yes' if ml_ready else f'⚠  Need {20 - total_runs_now} more runs'}")
    print(f"{BAR}\n")

    return {
        "done"           : done - skipped,
        "passed"         : passed,
        "failed"         : failed,
        "skipped"        : skipped,
        "total_in_history": total_runs_now,
        "ml_ready"       : ml_ready,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MarketLab Batch Backtest Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--plan",
        type=int,
        default=0,
        metavar="N",
        help="Run only plan N (1-based). Default: run all plans.",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=10_000,
        metavar="AMOUNT",
        help="Initial capital per backtest. Default: 10000",
    )
    parser.add_argument(
        "--rf",
        type=float,
        default=4.0,
        metavar="PCT",
        help="Annual risk-free rate %%. Default: 4.0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available plans and exit.",
    )

    args = parser.parse_args()

    if args.list:
        _print_plan_table()
        sys.exit(0)

    # Select plans
    if args.plan == 0:
        selected = AUTO_PLANS
    elif 1 <= args.plan <= len(AUTO_PLANS):
        selected = [AUTO_PLANS[args.plan - 1]]
    else:
        print(f"  ✗  Plan {args.plan} does not exist. Use --list to see options.")
        sys.exit(1)

    if args.dry_run:
        _print_plan_table()
        end   = datetime.today().strftime("%Y-%m-%d")
        queue = _build_run_queue(selected)
        print(f"  Dry run — {len(queue)} runs would be executed:\n")
        for ticker, start, end_d, label in queue:
            status = "✔" if validate_ticker(ticker) else "✗ (not in warehouse)"
            print(f"    {status}  {ticker:<6}  {start} → {end_d}  [{label}]")
        print()
        sys.exit(0)

    run_batch(
        plans        = selected,
        initial_cash = args.cash,
        annual_rf    = args.rf / 100,
    )


if __name__ == "__main__":
    main()
