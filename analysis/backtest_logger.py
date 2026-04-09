
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# ── Optional Streamlit (لا يُكسر الكود إن لم يكن مثبتاً) ──────────────────
try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

HISTORY_FILE   = "backtest_history.json"   # الملف الرئيسي للسجل
PASS_WIN_RATE  = 50.0    # الحد الأدنى لـ Win Rate لاعتبار الـ run ناجحاً
PASS_PF        = 1.3     # الحد الأدنى لـ Profit Factor
PASS_RETURN    = 0.0     # الحد الأدنى لـ Total Return %


# ─── JSON ENCODER ─────────────────────────────────────────────────────────────

class _JSONEncoder(json.JSONEncoder):
    """يتعامل مع numpy types التي لا يدعمها json الافتراضي."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 1: BacktestLogger
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestLogger:
    """
    يحفظ كل Backtest run في backtest_history.json بصيغة قائمة من الـ records.

    كل record يحتوي على:
        run_id        — معرّف فريد لكل run
        timestamp     — وقت التشغيل
        ticker        — رمز السهم
        period        — الفترة الزمنية (start → end)
        settings      — الإعدادات المستخدمة
        metrics       — نتائج الأداء (win_rate, sharpe, ...)
        benchmark_bh  — أداء Buy & Hold للمقارنة
        market_regime — Bull / Bear / Sideways بناءً على أداء السوق
        trade_summary — ملخص الصفقات (عدد، أنواع، أسباب الخروج)
        passed        — هل اجتاز الـ run معايير النجاح؟
    """

    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file = history_file
        self._history: list[dict] = self._load()

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []

    def _save(self) -> None:
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2, cls=_JSONEncoder)

    # ── Market Regime Detection ────────────────────────────────────────────────

    @staticmethod
    def _detect_market_regime(benchmark_return: float) -> str:
        """
        يحدد الـ Market Regime بناءً على أداء Buy & Hold للسهم نفسه.
        Bull     : > +10%
        Bear     : < -10%
        Sideways : بينهما
        """
        if benchmark_return > 10:
            return "Bull"
        elif benchmark_return < -10:
            return "Bear"
        else:
            return "Sideways"

    # ── Build Record ──────────────────────────────────────────────────────────

    def _build_record(
        self,
        ticker:    str,
        start:     str,
        end:       str,
        metrics:   dict,
        benchmark: float,
        settings:  dict,
        trades:    list[dict],
    ) -> dict:

        passed = (
            metrics.get("win_rate",      0) >= PASS_WIN_RATE and
            metrics.get("profit_factor", 0) >= PASS_PF       and
            metrics.get("total_return",  0) >  PASS_RETURN
        )

        # ── Trade summary (ملخص فقط، لا نحفظ كل الصفقات) ─────────────────────
        exit_reasons  = {}
        signal_counts = {}
        for t in trades:
            r = t.get("exit_reason", "Unknown")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1
            s = t.get("signal", "Unknown")
            signal_counts[s] = signal_counts.get(s, 0) + 1

        trade_summary = {
            "total_trades":  len(trades),
            "wins":          sum(1 for t in trades if t.get("result") == "WIN"),
            "losses":        sum(1 for t in trades if t.get("result") == "LOSS"),
            "exit_reasons":  exit_reasons,
            "signal_counts": signal_counts,
            "avg_score":     round(
                np.mean([t.get("score", 0) for t in trades]), 2
            ) if trades else 0,
        }

        return {
            "run_id":    str(uuid.uuid4())[:8],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker":    ticker,
            "period": {
                "start": start,
                "end":   end,
                "days":  (
                    datetime.strptime(end[:10],   "%Y-%m-%d") -
                    datetime.strptime(start[:10], "%Y-%m-%d")
                ).days,
            },
            "settings": {
                "initial_cash": settings.get("initial_cash", 10000),
                "commission":   settings.get("commission",   0.001),
                "slippage":     settings.get("slippage",     0.001),
            },
            "metrics": {
                "total_return":           round(metrics.get("total_return",           0), 2),
                "win_rate":               round(metrics.get("win_rate",               0), 2),
                "profit_factor":          round(metrics.get("profit_factor",          0), 2),
                "sharpe":                 round(metrics.get("sharpe",                 0), 2),
                "max_drawdown":           round(metrics.get("max_drawdown",           0), 2),
                "avg_r_multiple":         round(metrics.get("avg_r_multiple",         0), 2),
                "max_consecutive_losses": int(metrics.get("max_consecutive_losses",   0)),
            },
            "benchmark_bh":   round(benchmark, 2),
            "beat_benchmark": metrics.get("total_return", 0) > benchmark,
            "market_regime":  self._detect_market_regime(benchmark),
            "trade_summary":  trade_summary,
            "passed":         passed,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def log(
        self,
        ticker:    str,
        start:     str,
        end:       str,
        metrics:   dict,
        benchmark: float,
        settings:  dict,
        trades:    list[dict],
    ) -> dict:
        """يضيف run جديد إلى السجل ويحفظه. يُرجع الـ record المحفوظ."""
        record = self._build_record(
            ticker, start, end, metrics, benchmark, settings, trades
        )
        self._history.append(record)
        self._save()
        print(f"  📦 Logger: Run saved → [{record['run_id']}] {ticker} | "
              f"{'✅ PASS' if record['passed'] else '❌ FAIL'} | "
              f"Total runs: {len(self._history)}")
        return record

    def get_all(self) -> list[dict]:
        return self._history

    def get_runs_count(self) -> int:
        return len(self._history)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS 2: ReliabilityCalculator
# ═══════════════════════════════════════════════════════════════════════════════

class ReliabilityCalculator:
    """
    يحسب Reliability Score للنظام بناءً على السجل المتراكم.

    الـ Score مُركّب من:
        Pass Rate           (40%) — نسبة الـ runs الناجحة
        Avg Win Rate        (25%) — متوسط نسبة الصفقات الرابحة
        Avg Profit Factor   (20%) — متوسط الـ Profit Factor
        Beat Benchmark Rate (15%) — كم مرة تفوّق على Buy & Hold
    """

    WEIGHTS = {
        "pass_rate":      0.40,
        "avg_win_rate":   0.25,
        "avg_pf":         0.20,
        "beat_benchmark": 0.15,
    }

    WIN_RATE_TARGET = 65.0   # 65% win rate = score 1.0
    PF_TARGET       = 2.5    # PF 2.5      = score 1.0

    def __init__(self, history: list[dict]):
        self.history = history
        self.df      = pd.DataFrame(history) if history else pd.DataFrame()

    def _normalize(self, value: float, target: float) -> float:
        return min(value / target, 1.0) if target > 0 else 0.0

    def _extract_metric(self, field: str) -> pd.Series:
        return self.df["metrics"].apply(
            lambda m: m.get(field, 0) if isinstance(m, dict) else 0
        )

    # ── Overall Score ─────────────────────────────────────────────────────────

    def overall_score(self) -> dict:
        if self.df.empty:
            return {"score": 0, "label": "N/A", "runs": 0, "breakdown": {}, "raw": {}}

        n          = len(self.df)
        pass_rate  = self.df["passed"].mean()
        avg_wr     = self._extract_metric("win_rate").mean()
        avg_pf     = self._extract_metric("profit_factor").mean()
        beat_rate  = self.df["beat_benchmark"].mean() if "beat_benchmark" in self.df.columns else 0.0

        components = {
            "pass_rate":      pass_rate,
            "avg_win_rate":   self._normalize(avg_wr, self.WIN_RATE_TARGET),
            "avg_pf":         self._normalize(avg_pf, self.PF_TARGET),
            "beat_benchmark": beat_rate,
        }

        score = round(
            sum(components[k] * self.WEIGHTS[k] for k in self.WEIGHTS) * 100, 1
        )

        return {
            "score": score,
            "label": self._score_label(score),
            "runs":  n,
            "raw": {
                "pass_rate":      round(pass_rate * 100, 1),
                "avg_win_rate":   round(avg_wr,          1),
                "avg_pf":         round(avg_pf,          2),
                "beat_benchmark": round(beat_rate * 100, 1),
            },
            "breakdown": {k: round(v * 100, 1) for k, v in components.items()},
        }

    @staticmethod
    def _score_label(score: float) -> str:
        if score >= 80: return "🟢 Highly Reliable"
        if score >= 60: return "🟡 Moderately Reliable"
        if score >= 40: return "🟠 Needs More Data"
        return "🔴 Unreliable"

    # ── Per Ticker ────────────────────────────────────────────────────────────

    def per_ticker(self) -> dict:
        if self.df.empty:
            return {}

        results = {}
        for ticker, group in self.df.groupby("ticker"):
            n         = len(group)
            pass_rate = group["passed"].mean()
            avg_wr    = group["metrics"].apply(lambda m: m.get("win_rate", 0)).mean()
            avg_pf    = group["metrics"].apply(lambda m: m.get("profit_factor", 0)).mean()
            beat_rate = group["beat_benchmark"].mean() if "beat_benchmark" in group.columns else 0.0

            score = (
                pass_rate                                      * self.WEIGHTS["pass_rate"]      +
                self._normalize(avg_wr, self.WIN_RATE_TARGET) * self.WEIGHTS["avg_win_rate"]    +
                self._normalize(avg_pf, self.PF_TARGET)       * self.WEIGHTS["avg_pf"]          +
                beat_rate                                      * self.WEIGHTS["beat_benchmark"]
            ) * 100

            results[ticker] = {
                "score":     round(score, 1),
                "label":     self._score_label(score),
                "runs":      n,
                "pass_rate": round(pass_rate * 100, 1),
                "avg_wr":    round(avg_wr, 1),
                "avg_pf":    round(avg_pf, 2),
                "beat_bh":   round(beat_rate * 100, 1),
            }
        return results

    # ── Per Market Regime ─────────────────────────────────────────────────────

    def per_regime(self) -> dict:
        if self.df.empty or "market_regime" not in self.df.columns:
            return {}

        results = {}
        for regime, group in self.df.groupby("market_regime"):
            results[regime] = {
                "runs":       len(group),
                "pass_rate":  round(group["passed"].mean() * 100, 1),
                "avg_return": round(
                    group["metrics"].apply(lambda m: m.get("total_return", 0)).mean(), 2
                ),
            }
        return results

    # ── Trend ─────────────────────────────────────────────────────────────────

    def trend(self, window: int = 5) -> str:
        if len(self.df) < window * 2:
            return "⚪ Not enough data for trend analysis"

        recent = self.df.tail(window)["passed"].mean()
        older  = self.df.iloc[:-window].tail(window)["passed"].mean()
        delta  = recent - older

        if delta > 0.1:
            return f"📈 Improving  (+{delta*100:.0f}% pass rate in last {window} runs)"
        elif delta < -0.1:
            return f"📉 Declining  ({delta*100:.0f}% pass rate in last {window} runs)"
        else:
            return f"➡️  Stable  (±{abs(delta)*100:.0f}% pass rate in last {window} runs)"


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def log_backtest_run(
    ticker:       str,
    start:        str,
    end:          str,
    metrics:      dict,
    benchmark:    float,
    settings:     dict,
    trades:       list[dict],
    history_file: str = HISTORY_FILE,
) -> dict:
    
    logger = BacktestLogger(history_file)
    return logger.log(ticker, start, end, metrics, benchmark, settings, trades)


def get_reliability_report(history_file: str = HISTORY_FILE) -> dict:
    """يُرجع تقرير الموثوقية الكامل من السجل المتراكم."""
    logger = BacktestLogger(history_file)
    calc   = ReliabilityCalculator(logger.get_all())
    return {
        "overall":    calc.overall_score(),
        "per_ticker": calc.per_ticker(),
        "per_regime": calc.per_regime(),
        "trend":      calc.trend(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def display_dashboard(history_file: str = HISTORY_FILE) -> None:
    """
    صفحة Streamlit كاملة.
    أضفها في streamlit_app.py:
        from backtest_logger import display_dashboard
        display_dashboard()
    """
    if not _STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.title("📦 MarketLab — Black Box Logger")
    st.caption("سجل تاريخي لكل Backtest run · يُحدَّث تلقائياً بعد كل اختبار")

    logger  = BacktestLogger(history_file)
    history = logger.get_all()

    if not history:
        st.warning("لا يوجد سجل بعد. شغّل Backtest أولاً لتبدأ البيانات بالتراكم.")
        return

    calc = ReliabilityCalculator(history)
    report = {
        "overall":    calc.overall_score(),
        "per_ticker": calc.per_ticker(),
        "per_regime": calc.per_regime(),
        "trend":      calc.trend(),
    }
    overall = report["overall"]

    # ── Overall Score ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Overall System Reliability")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reliability Score", f"{overall['score']}/100")
    c2.metric("Status",            overall["label"])
    c3.metric("Total Runs",        overall["runs"])
    c4.metric("Pass Rate",         f"{overall['raw']['pass_rate']}%")

    score = overall["score"]
    color = "#2ecc71" if score >= 60 else "#e67e22" if score >= 40 else "#e74c3c"
    st.markdown(f"""
    <div style="background:#1e1e2e;border-radius:10px;padding:8px;margin:8px 0">
      <div style="background:{color};width:{score}%;height:28px;border-radius:8px;
                  display:flex;align-items:center;justify-content:center;
                  color:white;font-weight:bold;font-size:15px">
        {score} / 100
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.info(f"**Trend:** {report['trend']}")

    # ── Score Breakdown ───────────────────────────────────────────────────────
    with st.expander("📊 Score Breakdown (كيف يُحسب الـ Score؟)"):
        bd = overall["breakdown"]
        st.dataframe(pd.DataFrame([
            {"Component": "Pass Rate",        "Contribution": f"{bd['pass_rate']}%",      "Weight": "40%"},
            {"Component": "Avg Win Rate",     "Contribution": f"{bd['avg_win_rate']}%",   "Weight": "25%"},
            {"Component": "Avg Prof. Factor", "Contribution": f"{bd['avg_pf']}%",         "Weight": "20%"},
            {"Component": "Beat Benchmark",   "Contribution": f"{bd['beat_benchmark']}%", "Weight": "15%"},
        ]), use_container_width=True, hide_index=True)

        raw = overall["raw"]
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Avg Win Rate | **{raw['avg_win_rate']}%** |
        | Avg Profit Factor | **{raw['avg_pf']}** |
        | Beat Benchmark Rate | **{raw['beat_benchmark']}%** |
        """)

    # ── Per Ticker ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Reliability per Ticker")

    per_ticker = report["per_ticker"]
    if per_ticker:
        st.dataframe(pd.DataFrame([
            {
                "Ticker":    t,
                "Score":     v["score"],
                "Status":    v["label"],
                "Runs":      v["runs"],
                "Pass Rate": f"{v['pass_rate']}%",
                "Avg WR":    f"{v['avg_wr']}%",
                "Avg PF":    v["avg_pf"],
                "Beat B&H":  f"{v['beat_bh']}%",
            }
            for t, v in sorted(per_ticker.items(), key=lambda x: -x[1]["score"])
        ]), use_container_width=True, hide_index=True)

    # ── Per Market Regime ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🌡️ Performance per Market Regime")

    per_regime = report["per_regime"]
    if per_regime:
        st.dataframe(pd.DataFrame([
            {
                "Regime":     regime,
                "Runs":       v["runs"],
                "Pass Rate":  f"{v['pass_rate']}%",
                "Avg Return": f"{v['avg_return']}%",
            }
            for regime, v in per_regime.items()
        ]), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough runs for regime analysis.")

    # ── Full History Table ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🗃️ Full Run History")

    rows = []
    for r in reversed(history):
        m = r.get("metrics", {})
        rows.append({
            "ID":       r["run_id"],
            "Date":     r["timestamp"][:10],
            "Ticker":   r["ticker"],
            "Period":   f"{r['period']['start']} → {r['period']['end']}",
            "Regime":   r.get("market_regime", "—"),
            "Return":   f"{m.get('total_return', 0):+.1f}%",
            "Win Rate": f"{m.get('win_rate', 0):.1f}%",
            "Sharpe":   m.get("sharpe", 0),
            "Max DD":   f"{m.get('max_drawdown', 0):.1f}%",
            "Trades":   r.get("trade_summary", {}).get("total_trades", 0),
            "Passed":   "✅" if r["passed"] else "❌",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── ML Dataset Export ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 ML Dataset Export")
    st.caption("تصدير البيانات المتراكمة كـ dataset جاهز للتدريب مستقبلاً")

    if st.button("📥 Export as JSON Dataset"):
        ml_records = []
        for r in history:
            m  = r.get("metrics", {})
            ts = r.get("trade_summary", {})
            ml_records.append({
                # Features (المدخلات)
                "ticker":        r["ticker"],
                "period_days":   r["period"]["days"],
                "market_regime": r.get("market_regime", "Unknown"),
                "initial_cash":  r["settings"]["initial_cash"],
                "win_rate":      m.get("win_rate", 0),
                "profit_factor": m.get("profit_factor", 0),
                "sharpe":        m.get("sharpe", 0),
                "max_drawdown":  m.get("max_drawdown", 0),
                "avg_r_mult":    m.get("avg_r_multiple", 0),
                "max_cons_loss": m.get("max_consecutive_losses", 0),
                "total_trades":  ts.get("total_trades", 0),
                "avg_score":     ts.get("avg_score", 0),
                "benchmark_bh":  r.get("benchmark_bh", 0),
                "beat_benchmark": int(r.get("beat_benchmark", False)),
                # Labels (المخرجات للتدريب)
                "passed":        int(r["passed"]),
                "total_return":  m.get("total_return", 0),
            })

        st.download_button(
            label     = "⬇️ Download ml_dataset.json",
            data      = json.dumps(ml_records, indent=2, cls=_JSONEncoder),
            file_name = "ml_dataset.json",
            mime      = "application/json",
        )
        st.success(f"✅ {len(ml_records)} records ready for ML training!")

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    with st.expander("🔍 View Raw JSON (last 5 runs)"):
        st.json(history[-5:] if len(history) > 5 else history)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO — python backtest_logger.py
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("backtest_logger.py — Demo Mode")
    print("Injecting 3 fake runs …\n")

    fake_settings = {"initial_cash": 10000, "commission": 0.001, "slippage": 0.001}
    fake_trades   = [
        {"result": "WIN",  "signal": "STRONG BUY", "score": 12,
         "exit_reason": "Hit TP", "entry_price": 150, "stop_loss": 142, "position_pct": 35},
        {"result": "LOSS", "signal": "BUY",        "score": 7,
         "exit_reason": "Hit SL", "entry_price": 148, "stop_loss": 140, "position_pct": 22},
        {"result": "WIN",  "signal": "BUY",        "score": 8,
         "exit_reason": "Hit TP", "entry_price": 155, "stop_loss": 147, "position_pct": 22},
    ]

    fake_runs = [
        ("AAPL", "2021-01-01", "2022-01-01",
         {"win_rate": 68, "profit_factor": 1.8, "sharpe": 1.4, "total_return": 22.5,
          "max_drawdown": -8.2, "avg_r_multiple": 1.9, "max_consecutive_losses": 2},
         15.3),
        ("TSLA", "2021-01-01", "2022-01-01",
         {"win_rate": 42, "profit_factor": 0.9, "sharpe": 0.3, "total_return": -5.1,
          "max_drawdown": -18.7, "avg_r_multiple": 0.7, "max_consecutive_losses": 5},
         38.1),
        ("AAPL", "2022-01-01", "2023-01-01",
         {"win_rate": 61, "profit_factor": 1.5, "sharpe": 1.1, "total_return": 12.3,
          "max_drawdown": -10.1, "avg_r_multiple": 1.4, "max_consecutive_losses": 3},
         -19.4),
    ]

    for ticker, start, end, metrics, benchmark in fake_runs:
        log_backtest_run(ticker, start, end, metrics, benchmark, fake_settings, fake_trades)

    print("\n─── Reliability Report ───")
    report  = get_reliability_report()
    overall = report["overall"]

    print(f"\nOverall Score : {overall['score']}/100 — {overall['label']}")
    print(f"Pass Rate     : {overall['raw']['pass_rate']}%")
    print(f"Avg Win Rate  : {overall['raw']['avg_win_rate']}%")
    print(f"Beat B&H      : {overall['raw']['beat_benchmark']}%")
    print(f"Trend         : {report['trend']}")

    print("\nPer Ticker:")
    for t, v in report["per_ticker"].items():
        print(f"  {t}: {v['score']}/100 — {v['label']} ({v['runs']} runs)")

    print("\nPer Regime:")
    for r, v in report["per_regime"].items():
        print(f"  {r}: Pass {v['pass_rate']}% | Avg Return {v['avg_return']}%")

    print(f"\n✅ History saved to: {HISTORY_FILE}")
    print("   View dashboard: add display_dashboard() to your Streamlit app.")
