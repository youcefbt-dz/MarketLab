"""
ml_predictor.py — MarketLab ML Predictor  v2.2
Predicts whether a backtest run is "QUALITY" based on accumulated history.

Changes from v2.1:
  - Added XGBoost as 4th model (replaces GradientBoosting when data ≥ 50)
  - GradientBoosting kept as fallback when xgboost not installed
  - All other logic unchanged
"""
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

# XGBoost — optional, graceful fallback to GradientBoosting
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

warnings.filterwarnings("ignore")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

HISTORY_FILE = "backtest_history.json"
MIN_SAMPLES  = 20
RANDOM_STATE = 42
REGIME_MAP   = {"Bull": 1, "Sideways": 0, "Bear": -1}

QUALITY_SHARPE   =  0.5
QUALITY_DRAWDOWN = -12.0
QUALITY_WINRATE  =  50.0

DEMO_PERIOD_DAYS = 365
DEMO_TICKER      = "__DEMO__"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data Purge
# ═══════════════════════════════════════════════════════════════════════════════

def _is_demo_run(r: dict) -> bool:
    days        = r.get("period", {}).get("days", 0)
    ticker      = r.get("ticker", "")
    run_id      = r.get("run_id", "")
    is_365      = days == DEMO_PERIOD_DAYS
    is_sentinel = ticker == DEMO_TICKER or str(run_id).lower().startswith("demo")
    return is_365 and is_sentinel


def purge_history(data: list[dict]) -> tuple[list[dict], dict]:
    seen          = {}
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

    return result, {
        "original"    : len(data),
        "removed_demo": removed_demo,
        "removed_dupe": removed_dupe,
        "clean"       : len(result),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MLDataset
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    return numerator / denominator if abs(denominator) > 1e-9 else fallback


def _safe_drawdown_div(total_return: float, max_dd: float) -> float:
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
        quality_trade = 1  if sharpe > 0.5 AND drawdown > -12% AND win_rate > 50%
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
                f"need at least {MIN_SAMPLES}. "
                f"Run batch_backtest.py to generate more data."
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
            max_dd       = m.get("max_drawdown", 0)

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
# STEP 3 — BacktestPredictor  (SMOTE + 4 Models)
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestPredictor:

    def _make_models(self, n_minority: int, n_total: int) -> dict:
        k     = max(1, min(5, n_minority - 1))
        smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)

        models = {
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

        # XGBoost — preferred over GradientBoosting when available
        if _HAS_XGB:
            models["XGBoost"] = ImbPipeline([
                ("smote", smote),
                ("clf",   XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    verbosity=0,
                )),
            ])
        else:
            # Fallback — GradientBoosting from sklearn
            models["GradientBoosting"] = ImbPipeline([
                ("smote", smote),
                ("clf",   GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=RANDOM_STATE,
                )),
            ])

        return models

    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file            = history_file
        self.dataset                 = MLDataset(history_file)
        self.best_model              = None
        self.best_name               = None
        self.best_cv_score           = 0.0
        self.feature_cols            = MLDataset.FEATURE_COLS
        self._trained                = False
        self._models                 = {}
        self._best_pipeline_unfitted = None

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, verbose: bool = True) -> "BacktestPredictor":
        self.dataset.load()
        X, y = self.dataset.get_X_y()

        n_minority = int(y.sum())
        n_majority = int(len(y) - n_minority)
        n_total    = len(y)

        if verbose:
            self.dataset.summary()
            xgb_status = "✅ XGBoost" if _HAS_XGB else "⚠  xgboost not installed — using GradientBoosting"
            print(f"  {xgb_status}")
            print(f"  SMOTE: minority={n_minority}, majority={n_majority}")
            print(f"  Training {3} models...\n")

        self._models = self._make_models(n_minority, n_total)
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

        unfitted  = self._best_pipeline_unfitted
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
                raw      = metrics.get("market_regime",
                           metrics.get("market_regime_enc", "Sideways"))
                row[col] = REGIME_MAP.get(raw, raw) if isinstance(raw, str) else raw
            else:
                row[col] = metrics.get(col, 0)

        X_new = pd.DataFrame([row])[self.feature_cols]
        prob  = float(self.best_model.predict_proba(X_new)[0][1])

        prediction = "QUALITY" if prob >= 0.5 else "STANDARD"
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
    predictor = BacktestPredictor(history_file)
    predictor.train()
    return predictor


_predictor_cache: dict[str, BacktestPredictor] = {}


def predict_quality(
    metrics: dict,
    history_file: str = HISTORY_FILE,
) -> dict:
    """
    Predict run quality after a backtest completes.
    Trained once per history_file and cached in memory.
    """
    global _predictor_cache
    if history_file not in _predictor_cache:
        predictor = BacktestPredictor(history_file)
        predictor.train(verbose=False)
        _predictor_cache[history_file] = predictor
    return _predictor_cache[history_file].predict(metrics)


def invalidate_cache(history_file: str = HISTORY_FILE) -> None:
    """Call after new backtest runs are appended so next predict_quality() retrains."""
    _predictor_cache.pop(history_file, None)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO — python ml_predictor.py
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  MarketLab — ML Predictor  v2.2")
    xgb_note = "XGBoost ✅" if _HAS_XGB else "XGBoost ⚠ not installed"
    print(f"  {xgb_note}")
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
        print(f"     Model        : {result['model']}")

        checks = result["thresholds"]
        status = "  ".join(f"{'✓' if v else '✗'} {k}" for k, v in checks.items())
        print(f"     Checks       : {status}")

    print(f"\n{'═'*55}")
    print(f"  Sample accuracy: {correct}/{len(test_cases)}")
    print(f"{'═'*55}")
    print("  Usage:")
    print("    from ml_predictor import predict_quality, invalidate_cache")
    print("    result = predict_quality(metrics_dict)")
    print("    invalidate_cache()  # after appending new runs")
    print(f"{'═'*55}\n")
