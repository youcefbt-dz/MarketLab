from __future__ import annotations
import json
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

HISTORY_FILE = "backtest_history.json"
# FIX 2+minor: raised from 10 → 20 so n_minority≥2 is meaningful for SMOTE/CV
MIN_SAMPLES  = 20
RANDOM_STATE = 42
REGIME_MAP   = {"Bull": 1, "Sideways": 0, "Bear": -1}

# ── quality_trade thresholds ──────────────────────────────────────────────────
QUALITY_SHARPE   =  0.5
QUALITY_DRAWDOWN = -12.0
QUALITY_WINRATE  =  50.0

# FIX 3: demo detection — flag used in purge_history, not a raw days check
DEMO_PERIOD_DAYS = 365          # exact value Backtester uses for demo runs
DEMO_TICKER      = "__DEMO__"   # optional: set a sentinel ticker in demo runs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data Purge
# ═══════════════════════════════════════════════════════════════════════════════

def _is_demo_run(r: dict) -> bool:
    """
    FIX 3: A run is a demo only when BOTH conditions hold:
      - period.days == 365  (the demo default)
      - ticker is the sentinel OR run_id starts with 'demo'
    A real 1-year backtest on a normal ticker is NOT a demo.
    """
    days       = r.get("period", {}).get("days", 0)
    ticker     = r.get("ticker", "")
    run_id     = r.get("run_id", "")
    is_365     = days == DEMO_PERIOD_DAYS
    is_sentinel = ticker == DEMO_TICKER or str(run_id).lower().startswith("demo")
    return is_365 and is_sentinel


def purge_history(data: list[dict]) -> tuple[list[dict], dict]:
    seen          = {}   # key → index in result (keep latest)
    removed_demo  = 0
    removed_dupe  = 0
    result        = []

    for r in data:
        if _is_demo_run(r):
            removed_demo += 1
            continue

        key = (r["ticker"], r["period"]["start"], r["period"]["days"])

        if key in seen:
            result[seen[key]] = r
            removed_dupe += 1
        else:
            seen[key] = len(result)
            result.append(r)

    stats = {
        "original"    : len(data),
        "removed_demo": removed_demo,
        "removed_dupe": removed_dupe,
        "clean"       : len(result),
    }
    return result, stats


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MLDataset  (Feature Engineering + quality_trade label)
# ═══════════════════════════════════════════════════════════════════════════════

# FIX 4: safe divisor helpers
def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return numerator/denominator, or fallback when denominator is ~0."""
    return numerator / denominator if abs(denominator) > 1e-9 else fallback


def _safe_drawdown_div(total_return: float, max_dd: float) -> float:
    """
    risk_adjusted_return = total_return / |max_dd|.
    When max_dd == 0 (no losses), return total_return directly
    so the feature stays meaningful instead of becoming ±∞.
    """
    return _safe_div(total_return, abs(max_dd), fallback=total_return)


class MLDataset:
    """
    Converts backtest_history.json to a DataFrame ready for ML.

    Features (16):
        Core     : win_rate, profit_factor, sharpe, max_drawdown,
                   avg_r_multiple, max_consecutive_losses
        Activity : total_trades, avg_score, total_return
        Context  : period_days, market_regime_enc
        Exit     : sl_rate, tp_rate
        Derived  : win_loss_ratio, trade_efficiency, risk_adjusted_return

    Target:
        quality_trade = 1  if:
            sharpe > 0.5  AND  max_drawdown > -12%  AND  win_rate > 50%
        quality_trade = 0  otherwise
    """

    FEATURE_COLS = [
        "win_rate", "profit_factor", "sharpe", "max_drawdown",
        "avg_r_multiple", "max_consecutive_losses",
        "total_trades", "avg_score", "total_return",
        "period_days", "market_regime_enc",
        "sl_rate", "tp_rate",
        "win_loss_ratio", "trade_efficiency", "risk_adjusted_return",
    ]

    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file = history_file
        self.df: Optional[pd.DataFrame] = None
        self.purge_stats: dict = {}

    def load(self) -> "MLDataset":
        if not os.path.exists(self.history_file):
            raise FileNotFoundError(
                f"Not found: {self.history_file}\n"
                "Run Backtests first to accumulate data."
            )

        with open(self.history_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        data, self.purge_stats = purge_history(raw)

        if len(data) < MIN_SAMPLES:
            raise ValueError(
                f"Only {len(data)} records after purge — "
                f"need at least {MIN_SAMPLES}."
            )

        rows = []
        for r in data:
            m  = r.get("metrics", {})
            ts = r.get("trade_summary", {})
            er = ts.get("exit_reasons", {})

            total_trades = ts.get("total_trades", 1) or 1
            wins         = ts.get("wins",   0)
            losses       = ts.get("losses", 0)
            total_return = m.get("total_return", 0)
            max_dd       = m.get("max_drawdown", 0)   # FIX 4: keep 0, handle in helper

            sl_hits = er.get("Hit SL", 0) + er.get("Hit SL (Gap Down)", 0)
            tp_hits = er.get("Hit TP", 0) + er.get("Hit TP (Gap Up)",   0)

            sharpe   = m.get("sharpe",       0)
            drawdown = m.get("max_drawdown", 0)
            win_rate = m.get("win_rate",     0)

            quality = int(
                sharpe   >  QUALITY_SHARPE   and
                drawdown >  QUALITY_DRAWDOWN  and
                win_rate >  QUALITY_WINRATE
            )

            rows.append({
                "win_rate"              : win_rate,
                "profit_factor"         : m.get("profit_factor",          0),
                "sharpe"                : sharpe,
                "max_drawdown"          : drawdown,
                "avg_r_multiple"        : m.get("avg_r_multiple",         0),
                "max_consecutive_losses": m.get("max_consecutive_losses", 0),
                "total_trades"          : total_trades,
                "avg_score"             : ts.get("avg_score", 0),
                "total_return"          : total_return,
                "period_days"           : r.get("period", {}).get("days", 0),
                "market_regime_enc"     : REGIME_MAP.get(
                                            r.get("market_regime", "Sideways"), 0),
                "sl_rate"               : round(_safe_div(sl_hits, total_trades), 4),
                "tp_rate"               : round(_safe_div(tp_hits, total_trades), 4),
                "win_loss_ratio"        : round(_safe_div(wins, losses + 1),      4),
                "trade_efficiency"      : round(_safe_div(total_return, total_trades), 4),
                # FIX 4: no more ±∞ when max_dd == 0
                "risk_adjusted_return"  : round(_safe_drawdown_div(total_return, max_dd), 4),
                "ticker"                : r.get("ticker", ""),
                "run_id"                : r.get("run_id", ""),
                "quality_trade"         : quality,
            })

        self.df = pd.DataFrame(rows)
        return self

    def get_X_y(self) -> tuple[pd.DataFrame, pd.Series]:
        if self.df is None:
            self.load()
        return self.df[self.FEATURE_COLS].copy(), self.df["quality_trade"]

    def summary(self) -> None:
        if self.df is None:
            self.load()

        ps = self.purge_stats
        n  = len(self.df)
        nq = self.df["quality_trade"].sum()

        print(f"\n{'═'*55}")
        print(f"  ML Dataset — Data Purge Report")
        print(f"{'═'*55}")
        print(f"  Original records   : {ps.get('original', '?')}")
        print(f"  Removed demo runs  : -{ps.get('removed_demo', 0)}")
        print(f"  Removed duplicates : -{ps.get('removed_dupe', 0)}")
        print(f"  ─────────────────────────────")
        print(f"  Clean records      : {n}")
        print(f"{'═'*55}")
        print(f"  quality_trade = 1  : {nq}  ({nq/n*100:.1f}%)")
        print(f"  quality_trade = 0  : {n-nq}  ({(n-nq)/n*100:.1f}%)")
        print(f"  Features           : {len(self.FEATURE_COLS)}")
        print(f"{'═'*55}")
        print(f"\n  quality_trade thresholds:")
        print(f"    Sharpe      > {QUALITY_SHARPE}")
        print(f"    Max Drawdown> {QUALITY_DRAWDOWN}%")
        print(f"    Win Rate    > {QUALITY_WINRATE}%")
        print(f"{'═'*55}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BacktestPredictor  (SMOTE + 3 Models)
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestPredictor:

    def _make_models(self, n_minority: int) -> dict:
        k     = max(1, min(5, n_minority - 1))
        smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)

        return {
            "RandomForest": ImbPipeline([
                ("smote", smote),
                ("clf",   RandomForestClassifier(
                    n_estimators=300,
                    max_depth=4,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )),
            ]),
            "GradientBoosting": ImbPipeline([
                ("smote", smote),
                ("clf",   GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=RANDOM_STATE,
                )),
            ]),
            "LogisticRegression": ImbPipeline([
                ("smote",  smote),
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                )),
            ]),
        }

    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file  = history_file
        self.dataset       = MLDataset(history_file)
        self.best_model    = None
        self.best_name     = None
        self.best_cv_score = 0.0
        self.feature_cols  = MLDataset.FEATURE_COLS
        self._trained      = False
        self._models       = {}
        # FIX 1: store unfitted clone for evaluate() to avoid data leakage
        self._best_pipeline_unfitted = None

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, verbose: bool = True) -> "BacktestPredictor":
        self.dataset.load()
        X, y = self.dataset.get_X_y()

        n_minority = int(y.sum())
        n_majority = int(len(y) - n_minority)

        if verbose:
            self.dataset.summary()
            print(f"  SMOTE: minority={n_minority}, majority={n_majority}")
            print(f"  Training 3 models...\n")

        self._models = self._make_models(n_minority)
        n_splits     = min(5, n_minority)
        cv           = StratifiedKFold(
                           n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        results = {}
        for name, pipeline in self._models.items():
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
            results[name] = {"mean": scores.mean(), "std": scores.std()}
            if verbose:
                print(f"  {name:<25} AUC = {scores.mean():.3f} ± {scores.std():.3f}")

        self.best_name     = max(results, key=lambda k: results[k]["mean"])
        self.best_cv_score = results[self.best_name]["mean"]

        # FIX 1: keep an unfitted clone BEFORE fitting on full data
        self._best_pipeline_unfitted = clone(self._models[self.best_name])

        self.best_model = self._models[self.best_name]
        self.best_model.fit(X, y)
        self._trained = True

        if verbose:
            print(f"\n  ✅ Best model : {self.best_name}")
            print(f"  CV AUC        : {self.best_cv_score:.3f}")
            print(f"{'═'*55}\n")

        return self

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        if not self._trained:
            self.train(verbose=False)

        X, y = self.dataset.get_X_y()

        n_minority = int(y.sum())
        n_splits   = min(5, n_minority)
        cv         = StratifiedKFold(
                         n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        # FIX 1: use the unfitted clone — no data leakage
        unfitted = self._best_pipeline_unfitted
        y_pred_cv = cross_val_predict(unfitted, X, y, cv=cv)
        y_prob_cv = cross_val_predict(
                        unfitted, X, y, cv=cv, method="predict_proba")[:, 1]

        auc_cv = roc_auc_score(y, y_prob_cv) if len(set(y)) > 1 else float("nan")
        cm_cv  = confusion_matrix(y, y_pred_cv)

        print(f"\n{'═'*55}")
        print(f"  Evaluation — {self.best_name}  (CV={n_splits}-fold)")
        print(f"{'═'*55}")
        print(f"  CV AUC          : {auc_cv:.3f}")
        print(f"  Best CV AUC     : {self.best_cv_score:.3f}")
        print(f"\n  Confusion Matrix (CV, rows=actual, cols=predicted):")
        print(f"               Non-Quality  Quality")
        for i, row in enumerate(cm_cv):
            label = "  Non-Quality " if i == 0 else "  Quality     "
            print(f"  {label}  {row[0]:>4}         {row[1]:>4}")

        quality_recall = (
            cm_cv[1][1] / (cm_cv[1][0] + cm_cv[1][1])
            if (cm_cv[1][0] + cm_cv[1][1]) > 0 else 0
        )
        print(f"\n  Quality Recall  : {quality_recall:.1%}")
        print(f"\n{classification_report(y, y_pred_cv, target_names=['Non-Quality','Quality'])}")
        print(f"{'═'*55}\n")

        return {
            "cv_auc"        : round(auc_cv, 3),
            "quality_recall": round(quality_recall, 3),
            "model"         : self.best_name,
        }

    # ── Feature Importance ────────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        if not self._trained:
            self.train(verbose=False)

        clf = self.best_model.named_steps["clf"]

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            print("  Model does not support feature importances.")
            return pd.DataFrame()

        df_imp = (
            pd.DataFrame({
                "feature"   : self.feature_cols,
                "importance": importances,
            })
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        df_imp["rank"]           = df_imp.index + 1
        df_imp["importance_pct"] = (
            df_imp["importance"] / df_imp["importance"].sum() * 100
        ).round(1)

        print(f"\n{'═'*55}")
        print(f"  Feature Importance — Top {top_n}  ({self.best_name})")
        print(f"{'═'*55}")
        for _, row in df_imp.iterrows():
            bar = "█" * int(row["importance_pct"] / 2)
            print(f"  {int(row['rank']):>2}. {row['feature']:<28} "
                  f"{row['importance_pct']:>5.1f}%  {bar}")
        print(f"{'═'*55}\n")

        return df_imp

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, metrics: dict) -> dict:
        if not self._trained:
            self.train(verbose=False)

        row = {}
        for col in self.feature_cols:
            if col == "market_regime_enc":
                raw    = metrics.get("market_regime",
                         metrics.get("market_regime_enc", "Sideways"))
                row[col] = REGIME_MAP.get(raw, raw) if isinstance(raw, str) else raw
            else:
                row[col] = metrics.get(col, 0)

        X_new = pd.DataFrame([row])[self.feature_cols]
        prob  = float(self.best_model.predict_proba(X_new)[0][1])

        prediction = "QUALITY"  if prob >= 0.5 else "STANDARD"
        confidence = (
            "High"   if prob >= 0.75 or prob <= 0.25 else
            "Medium" if prob >= 0.60 or prob <= 0.40 else
            "Low"
        )

        thresholds = {
            f"sharpe > {QUALITY_SHARPE}"      : metrics.get("sharpe",       0) > QUALITY_SHARPE,
            f"drawdown > {QUALITY_DRAWDOWN}%" : metrics.get("max_drawdown", 0) > QUALITY_DRAWDOWN,
            f"win_rate > {QUALITY_WINRATE}%"  : metrics.get("win_rate",     0) > QUALITY_WINRATE,
        }

        return {
            "probability": round(prob, 4),
            "prediction" : prediction,
            "confidence" : confidence,
            "model"      : self.best_name,
            "thresholds" : thresholds,
        }

    # ── Full Report ───────────────────────────────────────────────────────────

    def full_report(self) -> None:
        self.train(verbose=True)
        self.evaluate()
        self.feature_importance()


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def train_predictor(history_file: str = HISTORY_FILE) -> BacktestPredictor:
    """Train and return a ready predictor."""
    predictor = BacktestPredictor(history_file)
    predictor.train()
    return predictor


# FIX 2: module-level cache — train once, reuse across calls
_predictor_cache: dict[str, BacktestPredictor] = {}


def predict_quality(
    metrics: dict,
    history_file: str = HISTORY_FILE,
) -> dict:
    """
    Predict run quality after a backtest completes.
    The predictor is trained once per history_file and cached in memory.

    Example:
        from ml_predictor import predict_quality

        result = predict_quality({
            "win_rate": 64.3, "profit_factor": 2.6, "sharpe": 0.6,
            "max_drawdown": -10.09, "total_return": 28.7,
            ...
        })
        print(result["prediction"])   # 'QUALITY' or 'STANDARD'
        print(result["probability"])  # 0.0 — 1.0
    """
    global _predictor_cache
    if history_file not in _predictor_cache:
        predictor = BacktestPredictor(history_file)
        predictor.train(verbose=False)
        _predictor_cache[history_file] = predictor
    return _predictor_cache[history_file].predict(metrics)


def invalidate_cache(history_file: str = HISTORY_FILE) -> None:
    """
    Call this after new backtest runs are appended to history_file
    so the next predict_quality() call retrains on fresh data.
    """
    _predictor_cache.pop(history_file, None)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO — python ml_predictor.py
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  MarketLab — ML Predictor  v2.1")
    print("═"*55)

    predictor = BacktestPredictor()
    predictor.full_report()

    print("  Sample Predictions")
    print("═"*55)

    test_cases = [
        {
            "label":    "AAPL  — expected: QUALITY",
            "expected": "QUALITY",
            "metrics": {
                "win_rate": 66.7, "profit_factor": 4.22, "sharpe": 0.69,
                "max_drawdown": -7.85,  "avg_r_multiple": 0.02,
                "max_consecutive_losses": 2, "total_trades": 12,
                "avg_score": 6.0, "total_return": 32.73,
                "period_days": 3374, "market_regime": "Bull",
                "sl_rate": 0.33, "tp_rate": 0.58,
                "win_loss_ratio": 2.0, "trade_efficiency": 2.73,
                "risk_adjusted_return": 4.17,
            },
        },
        {
            "label":    "TSLA  — expected: STANDARD",
            "expected": "STANDARD",
            "metrics": {
                "win_rate": 31.6, "profit_factor": 1.52, "sharpe": 0.38,
                "max_drawdown": -14.88, "avg_r_multiple": -0.01,
                "max_consecutive_losses": 3, "total_trades": 19,
                "avg_score": 5.84, "total_return": 19.73,
                "period_days": 3374, "market_regime": "Bull",
                "sl_rate": 0.68, "tp_rate": 0.32,
                "win_loss_ratio": 0.46, "trade_efficiency": 1.04,
                "risk_adjusted_return": 1.33,
            },
        },
        {
            "label":    "MSFT  — expected: STANDARD",
            "expected": "STANDARD",
            "metrics": {
                "win_rate": 46.2, "profit_factor": 1.61, "sharpe": 0.34,
                "max_drawdown": -11.22, "avg_r_multiple": 0.0,
                "max_consecutive_losses": 3, "total_trades": 13,
                "avg_score": 6.15, "total_return": 13.25,
                "period_days": 3374, "market_regime": "Bull",
                "sl_rate": 0.54, "tp_rate": 0.46,
                "win_loss_ratio": 0.86, "trade_efficiency": 1.02,
                "risk_adjusted_return": 1.18,
            },
        },
        {
            "label":    "SPOT  — expected: QUALITY",
            "expected": "QUALITY",
            "metrics": {
                "win_rate": 63.6, "profit_factor": 4.83, "sharpe": 0.85,
                "max_drawdown": -8.51, "avg_r_multiple": 0.06,
                "max_consecutive_losses": 2, "total_trades": 11,
                "avg_score": 5.91, "total_return": 62.36,
                "period_days": 3741, "market_regime": "Bull",
                "sl_rate": 0.27, "tp_rate": 0.64,
                "win_loss_ratio": 1.75, "trade_efficiency": 5.67,
                "risk_adjusted_return": 7.33,
            },
        },
        {
            "label":    "HON   — expected: STANDARD",
            "expected": "STANDARD",
            "metrics": {
                "win_rate": 20.0, "profit_factor": 0.67, "sharpe": -0.15,
                "max_drawdown": -13.58, "avg_r_multiple": 0.0,
                "max_consecutive_losses": 6, "total_trades": 10,
                "avg_score": 5.9, "total_return": -6.49,
                "period_days": 4107, "market_regime": "Bull",
                "sl_rate": 0.80, "tp_rate": 0.10,
                "win_loss_ratio": 0.25, "trade_efficiency": -0.65,
                "risk_adjusted_return": 0.48,
            },
        },
    ]

    correct = 0
    for case in test_cases:
        result  = predictor.predict(case["metrics"])
        match   = result["prediction"] == case["expected"]
        correct += int(match)
        icon    = "✅" if match else "❌"

        print(f"\n  {icon} {case['label']}")
        print(f"     Prediction   : {result['prediction']}  "
              f"(expected: {case['expected']})")
        print(f"     Probability  : {result['probability']:.1%}")
        print(f"     Confidence   : {result['confidence']}")

        checks = result["thresholds"]
        status = "  ".join(f"{'✓' if v else '✗'} {k}" for k, v in checks.items())
        print(f"     Checks       : {status}")

    print(f"\n{'═'*55}")
    print(f"  Sample accuracy: {correct}/{len(test_cases)}")
    print(f"{'═'*55}")
    print("  Usage in backtest.py:")
    print("    from ml_predictor import predict_quality, invalidate_cache")
    print("    result = predict_quality(metrics_dict)")
    print("    invalidate_cache()  # call after appending new runs")
    print(f"{'═'*55}\n")
