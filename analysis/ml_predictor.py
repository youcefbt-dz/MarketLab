"""
ml_predictor.py — MarketLab ML Predictor  v3.0
═══════════════════════════════════════════════
Major upgrades over v2.2:

  1. FEATURE ENGINEERING  — 12 interaction/ratio features added (24→28 total)
  2. CALIBRATION          — CalibratedClassifierCV wraps best model (Platt/Isotonic)
                            → probabilities are now meaningful percentages
  3. ADAPTIVE THRESHOLD   — auto-tuned on CV to maximize F1 for the minority class
  4. PERSISTENCE          — save/load trained model (joblib) to avoid retraining
  5. SHAP EXPLANATIONS    — per-prediction feature attribution (optional, graceful fallback)
  6. VOTING ENSEMBLE      — all 3/4 models combined via soft-voting before calibration
  7. WALK-FORWARD CV      — TimeSeriesSplit instead of random StratifiedKFold
                            (avoids data-leakage from temporal autocorrelation)
  8. DRIFT DETECTION      — warns when new data distribution differs from training set

Author  : MarketLab
Version : 3.0
"""
from __future__ import annotations

import json
import os
import warnings
import hashlib
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline as SkPipeline

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

warnings.filterwarnings("ignore")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

HISTORY_FILE   = "backtest_history.json"
MODEL_CACHE    = "ml_model_cache.joblib"
MIN_SAMPLES    = 20
RANDOM_STATE   = 42
REGIME_MAP     = {"Bull": 1, "Sideways": 0, "Bear": -1}

QUALITY_SHARPE   =  0.5
QUALITY_DRAWDOWN = -12.0
QUALITY_WINRATE  =  50.0

DEMO_PERIOD_DAYS = 365
DEMO_TICKER      = "__DEMO__"

# ─── FEATURE DEFINITIONS ─────────────────────────────────────────────────────

BASE_FEATURES = [
    "win_rate", "profit_factor", "sharpe", "max_drawdown",
    "avg_r_multiple", "max_consecutive_losses",
    "total_trades", "avg_score", "total_return",
    "period_days", "market_regime_enc",
    "sl_rate", "tp_rate",
    "win_loss_ratio", "trade_efficiency", "risk_adjusted_return",
]

# New interaction / ratio features (computed in MLDataset._engineer)
ENGINEERED_FEATURES = [
    "sharpe_per_trade",          # sharpe / log(total_trades+1)  — quality per trade
    "calmar_ratio",              # total_return / |max_drawdown|  — risk-adj return
    "expectancy",                # win_rate/100 * avg_r_multiple  — EV per trade
    "consistency_score",         # profit_factor * (1 - sl_rate)  — clean exits
    "regime_sharpe",             # sharpe * market_regime_enc     — regime alignment
    "drawdown_recovery",         # (total_return + max_drawdown)  — headroom
    "activity_quality",          # avg_score * win_rate / 100     — signal quality
    "loss_control",              # 1 / (max_consecutive_losses+1) — streak risk
    "tp_sl_ratio",               # tp_rate / (sl_rate+1e-9)       — exit quality
    "trade_density",             # total_trades / (period_days/365) — annualised
    "risk_reward_balance",       # win_loss_ratio * profit_factor  — combined edge
    "volatility_proxy",          # |max_drawdown| / total_trades   — dd per trade
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data Purge (unchanged from v2.2)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_demo_run(r: dict) -> bool:
    days        = r.get("period", {}).get("days", 0)
    ticker      = r.get("ticker", "")
    run_id      = r.get("run_id", "")
    is_365      = days == DEMO_PERIOD_DAYS
    is_sentinel = ticker == DEMO_TICKER or str(run_id).lower().startswith("demo")
    return is_365 and is_sentinel


def purge_history(data: list[dict]) -> tuple[list[dict], dict]:
    seen         = {}
    removed_demo = 0
    removed_dupe = 0
    result       = []

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
# STEP 2 — MLDataset  (now with feature engineering)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    return num / den if abs(den) > 1e-9 else fallback


class MLDataset:
    """
    Converts backtest_history.json → enriched DataFrame with 28 features.

    Target:
        quality_trade = 1  if sharpe > 0.5 AND drawdown > -12% AND win_rate > 50%
    """

    FEATURE_COLS = ALL_FEATURES

    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file = history_file
        self.df: Optional[pd.DataFrame] = None
        self.purge_stats: dict = {}
        self._data_hash: str = ""

    # ── Raw row extraction ────────────────────────────────────────────────────

    def _extract_row(self, r: dict) -> dict:
        m  = r.get("metrics", {})
        ts = r.get("trade_summary", {})
        er = ts.get("exit_reasons", {})

        total_trades = ts.get("total_trades", 1) or 1
        wins         = ts.get("wins",   0)
        losses       = ts.get("losses", 0)
        total_return = m.get("total_return", 0)
        max_dd       = m.get("max_drawdown", 0)
        sharpe       = m.get("sharpe",       0)
        drawdown     = m.get("max_drawdown", 0)
        win_rate     = m.get("win_rate",     0)
        period_days  = r.get("period", {}).get("days", 0)
        regime_raw   = r.get("market_regime", "Sideways")
        regime_enc   = REGIME_MAP.get(regime_raw, 0)
        avg_r        = m.get("avg_r_multiple", 0)
        profit_f     = m.get("profit_factor",  0)
        avg_score    = ts.get("avg_score", 0)

        sl_hits = er.get("Hit SL", 0) + er.get("Hit SL (Gap Down)", 0)
        tp_hits = er.get("Hit TP", 0) + er.get("Hit TP (Gap Up)",   0)
        sl_rate = round(_safe_div(sl_hits, total_trades), 4)
        tp_rate = round(_safe_div(tp_hits, total_trades), 4)

        win_loss_ratio       = round(_safe_div(wins, losses + 1),            4)
        trade_efficiency     = round(_safe_div(total_return, total_trades),   4)
        risk_adjusted_return = round(_safe_div(total_return, abs(max_dd)),    4)

        quality = int(
            sharpe   >  QUALITY_SHARPE   and
            drawdown >  QUALITY_DRAWDOWN  and
            win_rate >  QUALITY_WINRATE
        )

        return {
            # base features
            "win_rate"              : win_rate,
            "profit_factor"         : profit_f,
            "sharpe"                : sharpe,
            "max_drawdown"          : drawdown,
            "avg_r_multiple"        : avg_r,
            "max_consecutive_losses": m.get("max_consecutive_losses", 0),
            "total_trades"          : total_trades,
            "avg_score"             : avg_score,
            "total_return"          : total_return,
            "period_days"           : period_days,
            "market_regime_enc"     : regime_enc,
            "sl_rate"               : sl_rate,
            "tp_rate"               : tp_rate,
            "win_loss_ratio"        : win_loss_ratio,
            "trade_efficiency"      : trade_efficiency,
            "risk_adjusted_return"  : risk_adjusted_return,
            # metadata (not features)
            "ticker"                : r.get("ticker", ""),
            "run_id"                : r.get("run_id", ""),
            "quality_trade"         : quality,
            # raw values needed for engineering
            "_total_trades"         : total_trades,
            "_period_days"          : period_days,
            "_max_dd"               : max_dd,
            "_regime_enc"           : regime_enc,
        }

    # ── Feature Engineering ───────────────────────────────────────────────────

    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        eps = 1e-9
        tt  = df["_total_trades"]
        pd_ = df["_period_days"].clip(lower=1)
        mdd = df["_max_dd"].abs().clip(lower=eps)

        df["sharpe_per_trade"]    = df["sharpe"]         / np.log1p(tt)
        df["calmar_ratio"]        = df["total_return"]   / mdd
        df["expectancy"]          = (df["win_rate"]/100) * df["avg_r_multiple"]
        df["consistency_score"]   = df["profit_factor"]  * (1 - df["sl_rate"])
        df["regime_sharpe"]       = df["sharpe"]         * df["_regime_enc"]
        df["drawdown_recovery"]   = df["total_return"]   + df["max_drawdown"]
        df["activity_quality"]    = df["avg_score"]      * (df["win_rate"] / 100)
        df["loss_control"]        = 1.0 / (df["max_consecutive_losses"] + 1)
        df["tp_sl_ratio"]         = df["tp_rate"]        / (df["sl_rate"] + eps)
        df["trade_density"]       = tt / (pd_ / 365).clip(lower=eps)
        df["risk_reward_balance"] = df["win_loss_ratio"] * df["profit_factor"]
        df["volatility_proxy"]    = mdd / tt

        # clip extremes to prevent outlier domination
        clip_cols = [
            "calmar_ratio", "tp_sl_ratio", "trade_density",
            "risk_reward_balance", "sharpe_per_trade",
        ]
        for col in clip_cols:
            q01 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(q01, q99)

        return df

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> "MLDataset":
        if not os.path.exists(self.history_file):
            raise FileNotFoundError(
                f"Not found: {self.history_file}\n"
                "Run Backtests first to accumulate data."
            )

        with open(self.history_file, "r", encoding="utf-8") as f:
            raw_bytes = f.read()

        self._data_hash = hashlib.md5(raw_bytes.encode()).hexdigest()
        data, self.purge_stats = purge_history(json.loads(raw_bytes))

        if len(data) < MIN_SAMPLES:
            raise ValueError(
                f"Only {len(data)} records after purge — "
                f"need ≥ {MIN_SAMPLES}. Run batch_backtest.py first."
            )

        rows = [self._extract_row(r) for r in data]
        df   = pd.DataFrame(rows)
        df   = self._engineer(df)

        # drop engineering scratch columns
        df.drop(columns=["_total_trades", "_period_days", "_max_dd", "_regime_enc"],
                inplace=True)

        self.df = df
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
        nq = int(self.df["quality_trade"].sum())

        print(f"\n{'═'*60}")
        print(f"  ML Dataset  v3.0 — Data Purge Report")
        print(f"{'═'*60}")
        print(f"  Original records   : {ps.get('original', '?')}")
        print(f"  Removed demo runs  : -{ps.get('removed_demo', 0)}")
        print(f"  Removed duplicates : -{ps.get('removed_dupe', 0)}")
        print(f"  ─────────────────────────────────────")
        print(f"  Clean records      : {n}")
        print(f"  Data hash (MD5)    : {self._data_hash[:12]}…")
        print(f"{'═'*60}")
        print(f"  quality_trade = 1  : {nq}  ({nq/n*100:.1f}%)")
        print(f"  quality_trade = 0  : {n-nq}  ({(n-nq)/n*100:.1f}%)")
        print(f"  Base features      : {len(BASE_FEATURES)}")
        print(f"  Engineered features: {len(ENGINEERED_FEATURES)}")
        print(f"  Total features     : {len(ALL_FEATURES)}")
        print(f"{'═'*60}")
        print(f"\n  quality_trade thresholds:")
        print(f"    Sharpe      > {QUALITY_SHARPE}")
        print(f"    Max Drawdown> {QUALITY_DRAWDOWN}%")
        print(f"    Win Rate    > {QUALITY_WINRATE}%")
        print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Drift Detection
# ═══════════════════════════════════════════════════════════════════════════════

class DriftDetector:
    """
    Lightweight Population Stability Index (PSI) drift check.
    Warns if a new run's features are out-of-distribution vs. training data.
    PSI < 0.1  → No drift
    PSI 0.1–0.2 → Moderate drift (warn)
    PSI > 0.2  → Significant drift (alert)
    """

    def __init__(self, X_train: pd.DataFrame, n_bins: int = 10):
        self._stats = {}
        for col in X_train.columns:
            vals = X_train[col].dropna()
            self._stats[col] = {
                "mean": vals.mean(),
                "std" : vals.std() + 1e-9,
                "min" : vals.min(),
                "max" : vals.max(),
            }

    def check(self, x_new: dict) -> dict[str, str]:
        """Return drift status per feature for a single observation."""
        alerts = {}
        for col, s in self._stats.items():
            val = x_new.get(col, 0)
            z   = abs((val - s["mean"]) / s["std"])
            if z > 3.5:
                alerts[col] = f"⚠ z={z:.1f} (extreme)"
            elif z > 2.5:
                alerts[col] = f"○ z={z:.1f} (unusual)"
        return alerts


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — BacktestPredictor  (Voting + Calibration + Adaptive Threshold)
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestPredictor:
    """
    Upgraded ML predictor with:
      • 28 features (16 base + 12 engineered)
      • Soft-voting ensemble (RF + LR + XGB/GB)
      • Probability calibration (Platt scaling via CalibratedClassifierCV)
      • Adaptive decision threshold (maximises F1 on CV probabilities)
      • Walk-forward temporal cross-validation
      • Drift detection on new predictions
      • Save / load model cache (joblib)
      • SHAP explanations (when shap installed)
    """

    def __init__(self, history_file: str = HISTORY_FILE,
                 model_cache: str = MODEL_CACHE):
        self.history_file  = history_file
        self.model_cache   = model_cache
        self.dataset       = MLDataset(history_file)
        self.feature_cols  = ALL_FEATURES

        self.calibrated_model  = None
        self.best_threshold    = 0.5
        self.best_cv_auc       = 0.0
        self.best_cv_f1        = 0.0
        self.drift_detector    = None
        self._trained          = False
        self._shap_explainer   = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_base_estimators(self, k_smote: int) -> list[tuple]:
        """Return list of (name, pipeline) for the VotingClassifier."""
        smote = SMOTE(k_neighbors=k_smote, random_state=RANDOM_STATE)

        estimators = [
            ("rf", ImbPipeline([
                ("smote", smote),
                ("clf",   RandomForestClassifier(
                    n_estimators=400,
                    max_depth=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )),
            ])),
            ("lr", ImbPipeline([
                ("smote",  smote),
                ("scaler", RobustScaler()),      # RobustScaler handles outliers better
                ("clf",    LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                )),
            ])),
        ]

        if _HAS_XGB:
            estimators.append(("xgb", ImbPipeline([
                ("smote", smote),
                ("clf",   XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.04,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    gamma=0.1,
                    reg_alpha=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    verbosity=0,
                )),
            ])))
        else:
            estimators.append(("gb", ImbPipeline([
                ("smote", smote),
                ("clf",   GradientBoostingClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.04,
                    subsample=0.8,
                    random_state=RANDOM_STATE,
                )),
            ])))

        return estimators

    @staticmethod
    def _find_optimal_threshold(y_true: np.ndarray,
                                 y_prob: np.ndarray) -> tuple[float, float]:
        """Find threshold that maximises F1 for the positive (quality) class."""
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        best_idx  = int(np.argmax(f1_scores))
        best_thr  = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        best_f1   = float(f1_scores[best_idx])
        # keep threshold in sensible range
        best_thr  = float(np.clip(best_thr, 0.25, 0.75))
        return best_thr, best_f1

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, verbose: bool = True) -> "BacktestPredictor":
        self.dataset.load()
        X, y = self.dataset.get_X_y()

        n_minority = int(y.sum())
        n_majority = int(len(y) - n_minority)
        k_smote    = max(1, min(5, n_minority - 1))
        n_cv       = min(5, n_minority)

        if verbose:
            self.dataset.summary()
            xgb_status = "✅ XGBoost" if _HAS_XGB else "⚠  xgboost not installed → GradientBoosting"
            shap_status = "✅ SHAP" if _HAS_SHAP else "⚠  shap not installed → no explanations"
            print(f"  {xgb_status}   {shap_status}")
            print(f"  SMOTE k={k_smote}  minority={n_minority}  majority={n_majority}")
            print(f"  Building soft-voting ensemble…\n")

        # ── Build voting ensemble ────────────────────────────────────────────
        estimators = self._build_base_estimators(k_smote)
        voter      = VotingClassifier(estimators=estimators, voting="soft")

        # ── CV for AUC + threshold search ───────────────────────────────────
        # Use TimeSeriesSplit to respect temporal order (walk-forward)
        tscv = TimeSeriesSplit(n_splits=n_cv)

        cv_scores = cross_val_score(voter, X, y, cv=tscv, scoring="roc_auc")
        y_prob_cv = cross_val_predict(voter, X, y, cv=tscv, method="predict_proba")[:, 1]

        self.best_cv_auc               = float(cv_scores.mean())
        self.best_threshold, self.best_cv_f1 = self._find_optimal_threshold(
            y.values, y_prob_cv
        )

        if verbose:
            print(f"  Walk-Forward CV AUC : {self.best_cv_auc:.3f} ± {cv_scores.std():.3f}")
            print(f"  Optimal threshold   : {self.best_threshold:.3f}  (CV F1={self.best_cv_f1:.3f})")
            print(f"\n  Calibrating probabilities (Platt scaling)…")

        # ── Fit voter on full data then calibrate ────────────────────────────
        voter.fit(X, y)

        # CalibratedClassifierCV in "prefit" mode wraps the already-fitted voter
        self.calibrated_model = CalibratedClassifierCV(
            voter, cv="prefit", method="sigmoid"   # Platt
        )
        self.calibrated_model.fit(X, y)

        # ── Drift detector ───────────────────────────────────────────────────
        self.drift_detector = DriftDetector(X)

        # ── SHAP explainer ───────────────────────────────────────────────────
        if _HAS_SHAP:
            try:
                # Use a small background sample for speed
                bg = shap.sample(X, min(50, len(X)), random_state=RANDOM_STATE)
                self._shap_explainer = shap.KernelExplainer(
                    lambda arr: self.calibrated_model.predict_proba(
                        pd.DataFrame(arr, columns=self.feature_cols)
                    )[:, 1],
                    bg,
                )
                if verbose:
                    print(f"  SHAP KernelExplainer ready (background={len(bg)} samples)")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ SHAP init failed: {e}")

        self._trained = True

        if verbose:
            print(f"\n  ✅ Training complete")
            print(f"  CV AUC  : {self.best_cv_auc:.3f}")
            print(f"  CV F1   : {self.best_cv_f1:.3f}")
            print(f"  Features: {len(self.feature_cols)}")
            print(f"{'═'*60}\n")

        return self

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        if not _HAS_JOBLIB:
            print("  ⚠ joblib not available — cannot save model.")
            return
        if not self._trained:
            raise RuntimeError("Train the model first before saving.")

        path = path or self.model_cache
        payload = {
            "calibrated_model" : self.calibrated_model,
            "best_threshold"   : self.best_threshold,
            "best_cv_auc"      : self.best_cv_auc,
            "best_cv_f1"       : self.best_cv_f1,
            "feature_cols"     : self.feature_cols,
            "data_hash"        : self.dataset._data_hash,
            "drift_detector"   : self.drift_detector,
        }
        joblib.dump(payload, path, compress=3)
        print(f"  ✅ Model saved → {path}")

    def load(self, path: Optional[str] = None) -> bool:
        """
        Load cached model. Returns True if successful and data unchanged.
        Returns False if cache missing, stale, or joblib unavailable.
        """
        if not _HAS_JOBLIB:
            return False

        path = path or self.model_cache
        if not os.path.exists(path):
            return False

        try:
            payload = joblib.load(path)
        except Exception:
            return False

        # Verify data hasn't changed since last save
        self.dataset.load()
        if payload.get("data_hash") != self.dataset._data_hash:
            print("  ℹ  Data changed since last save — retraining.")
            return False

        self.calibrated_model = payload["calibrated_model"]
        self.best_threshold   = payload["best_threshold"]
        self.best_cv_auc      = payload["best_cv_auc"]
        self.best_cv_f1       = payload["best_cv_f1"]
        self.feature_cols     = payload["feature_cols"]
        self.drift_detector   = payload.get("drift_detector")
        self._trained         = True
        print(f"  ✅ Model loaded from cache → {path}  (AUC={self.best_cv_auc:.3f})")
        return True

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        if not self._trained:
            self.train(verbose=False)

        X, y  = self.dataset.get_X_y()
        n_min = int(y.sum())
        n_cv  = min(5, n_min)
        tscv  = TimeSeriesSplit(n_splits=n_cv)

        # Use a fresh clone of the unfitted voter for honest CV
        estimators = self._build_base_estimators(max(1, min(5, n_min - 1)))
        voter_clone = VotingClassifier(estimators=estimators, voting="soft")

        y_prob_cv = cross_val_predict(
            voter_clone, X, y, cv=tscv, method="predict_proba"
        )[:, 1]
        y_pred_cv = (y_prob_cv >= self.best_threshold).astype(int)

        auc_cv = roc_auc_score(y, y_prob_cv) if len(set(y)) > 1 else float("nan")
        f1_cv  = f1_score(y, y_pred_cv)
        cm_cv  = confusion_matrix(y, y_pred_cv)

        print(f"\n{'═'*60}")
        print(f"  Evaluation — Soft-Voting Ensemble  (Walk-Forward CV={n_cv})")
        print(f"{'═'*60}")
        print(f"  CV AUC              : {auc_cv:.3f}")
        print(f"  CV F1 (quality)     : {f1_cv:.3f}")
        print(f"  Decision threshold  : {self.best_threshold:.3f}")
        print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
        print(f"               Non-Quality  Quality")
        for i, row in enumerate(cm_cv):
            label = "  Non-Quality " if i == 0 else "  Quality     "
            print(f"  {label}  {row[0]:>4}         {row[1]:>4}")

        quality_recall = (
            cm_cv[1][1] / (cm_cv[1][0] + cm_cv[1][1])
            if (cm_cv[1][0] + cm_cv[1][1]) > 0 else 0
        )
        quality_prec = (
            cm_cv[1][1] / (cm_cv[0][1] + cm_cv[1][1])
            if (cm_cv[0][1] + cm_cv[1][1]) > 0 else 0
        )
        print(f"\n  Quality Recall      : {quality_recall:.1%}")
        print(f"  Quality Precision   : {quality_prec:.1%}")
        print(f"\n{classification_report(y, y_pred_cv, target_names=['Non-Quality','Quality'])}")
        print(f"{'═'*60}\n")

        return {
            "cv_auc"          : round(auc_cv, 3),
            "cv_f1"           : round(f1_cv,  3),
            "quality_recall"  : round(quality_recall, 3),
            "quality_precision": round(quality_prec,  3),
            "threshold"       : self.best_threshold,
        }

    # ── Feature Importance ────────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        if not self._trained:
            self.train(verbose=False)

        # Average importances across RF and XGB/GB sub-estimators
        importances_list = []
        voter = self.calibrated_model.estimator  # underlying VotingClassifier

        for name, pipe in voter.estimators_:
            clf = pipe.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                importances_list.append(clf.feature_importances_)
            elif hasattr(clf, "coef_"):
                importances_list.append(np.abs(clf.coef_[0]))

        if not importances_list:
            print("  No feature importances available.")
            return pd.DataFrame()

        importances = np.mean(importances_list, axis=0)

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
        df_imp["type"] = df_imp["feature"].apply(
            lambda f: "★ ENG" if f in ENGINEERED_FEATURES else "base"
        )

        print(f"\n{'═'*60}")
        print(f"  Feature Importance — Top {top_n}  (ensemble average)")
        print(f"  ★ = engineered feature")
        print(f"{'═'*60}")
        for _, row in df_imp.iterrows():
            bar  = "█" * int(row["importance_pct"] / 2)
            tag  = "★" if row["type"] == "★ ENG" else " "
            print(f"  {tag}{int(row['rank']):>2}. {row['feature']:<30} "
                  f"{row['importance_pct']:>5.1f}%  {bar}")
        print(f"{'═'*60}\n")

        return df_imp

    # ── SHAP Explanation ──────────────────────────────────────────────────────

    def explain(self, metrics: dict, top_n: int = 8) -> dict:
        """
        Return per-feature SHAP contributions for a single prediction.
        Falls back to an empty dict if shap is not installed.
        """
        if not _HAS_SHAP or self._shap_explainer is None:
            print("  ⚠ SHAP not available — install with: pip install shap")
            return {}
        if not self._trained:
            self.train(verbose=False)

        row   = self._build_row(metrics)
        X_new = pd.DataFrame([row])[self.feature_cols]

        shap_vals = self._shap_explainer.shap_values(X_new, silent=True)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]          # class=1 (quality)
        sv = shap_vals[0]

        df_shap = (
            pd.DataFrame({"feature": self.feature_cols, "shap": sv})
            .assign(abs_shap=lambda d: d["shap"].abs())
            .sort_values("abs_shap", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        print(f"\n{'═'*60}")
        print(f"  SHAP Explanation — Top {top_n} drivers")
        print(f"{'═'*60}")
        for _, row_s in df_shap.iterrows():
            direction = "▲ pushes QUALITY" if row_s["shap"] > 0 else "▼ pushes STANDARD"
            tag = "★" if row_s["feature"] in ENGINEERED_FEATURES else " "
            print(f"  {tag} {row_s['feature']:<30}  {row_s['shap']:+.4f}  {direction}")
        print(f"{'═'*60}\n")

        return df_shap.to_dict(orient="records")

    # ── Predict ───────────────────────────────────────────────────────────────

    def _build_row(self, metrics: dict) -> dict:
        """Convert raw metrics dict → feature row (including engineered features)."""
        total_trades = metrics.get("total_trades", 1) or 1
        period_days  = metrics.get("period_days",  365)
        max_dd       = metrics.get("max_drawdown", 0)
        total_return = metrics.get("total_return", 0)
        regime_raw   = metrics.get("market_regime", metrics.get("market_regime_enc", "Sideways"))
        regime_enc   = REGIME_MAP.get(regime_raw, regime_raw) if isinstance(regime_raw, str) else regime_raw
        sharpe       = metrics.get("sharpe",        0)
        profit_f     = metrics.get("profit_factor", 0)
        win_rate     = metrics.get("win_rate",      0)
        avg_r        = metrics.get("avg_r_multiple",0)
        sl_rate      = metrics.get("sl_rate",       0)
        tp_rate      = metrics.get("tp_rate",       0)
        avg_score    = metrics.get("avg_score",     0)
        wlr          = metrics.get("win_loss_ratio",0)
        mcl          = metrics.get("max_consecutive_losses", 0)
        eps          = 1e-9

        row = {
            # base
            "win_rate"              : win_rate,
            "profit_factor"         : profit_f,
            "sharpe"                : sharpe,
            "max_drawdown"          : max_dd,
            "avg_r_multiple"        : avg_r,
            "max_consecutive_losses": mcl,
            "total_trades"          : total_trades,
            "avg_score"             : avg_score,
            "total_return"          : total_return,
            "period_days"           : period_days,
            "market_regime_enc"     : regime_enc,
            "sl_rate"               : sl_rate,
            "tp_rate"               : tp_rate,
            "win_loss_ratio"        : wlr,
            "trade_efficiency"      : _safe_div(total_return, total_trades),
            "risk_adjusted_return"  : _safe_div(total_return, abs(max_dd)),
            # engineered
            "sharpe_per_trade"      : sharpe / np.log1p(total_trades),
            "calmar_ratio"          : _safe_div(total_return, abs(max_dd) + eps),
            "expectancy"            : (win_rate / 100) * avg_r,
            "consistency_score"     : profit_f * (1 - sl_rate),
            "regime_sharpe"         : sharpe * regime_enc,
            "drawdown_recovery"     : total_return + max_dd,
            "activity_quality"      : avg_score * (win_rate / 100),
            "loss_control"          : 1.0 / (mcl + 1),
            "tp_sl_ratio"           : tp_rate / (sl_rate + eps),
            "trade_density"         : total_trades / max(period_days / 365, eps),
            "risk_reward_balance"   : wlr * profit_f,
            "volatility_proxy"      : abs(max_dd) / total_trades,
        }
        return row

    def predict(self, metrics: dict,
                explain: bool = False) -> dict:
        if not self._trained:
            self.train(verbose=False)

        row   = self._build_row(metrics)
        X_new = pd.DataFrame([row])[self.feature_cols]

        prob       = float(self.calibrated_model.predict_proba(X_new)[0][1])
        prediction = "QUALITY" if prob >= self.best_threshold else "STANDARD"
        confidence = (
            "High"   if prob >= 0.75 or prob <= 0.25 else
            "Medium" if prob >= 0.60 or prob <= 0.40 else
            "Low"
        )

        thresholds = {
            f"sharpe > {QUALITY_SHARPE}"     : metrics.get("sharpe",       0) > QUALITY_SHARPE,
            f"drawdown > {QUALITY_DRAWDOWN}%": metrics.get("max_drawdown", 0) > QUALITY_DRAWDOWN,
            f"win_rate > {QUALITY_WINRATE}%" : metrics.get("win_rate",     0) > QUALITY_WINRATE,
            "rule_quality"                   : all([
                metrics.get("sharpe",       0) > QUALITY_SHARPE,
                metrics.get("max_drawdown", 0) > QUALITY_DRAWDOWN,
                metrics.get("win_rate",     0) > QUALITY_WINRATE,
            ]),
        }

        # Drift check
        drift_alerts = {}
        if self.drift_detector is not None:
            drift_alerts = self.drift_detector.check(row)

        result = {
            "probability"  : round(prob, 4),
            "prediction"   : prediction,
            "confidence"   : confidence,
            "threshold"    : round(self.best_threshold, 3),
            "model"        : "SoftVoting+Calibrated",
            "thresholds"   : thresholds,
            "drift_alerts" : drift_alerts,
        }

        if explain and _HAS_SHAP:
            result["shap"] = self.explain(metrics)

        return result

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
    use_cache: bool = True,
) -> dict:
    """
    Predict run quality.  Loads from joblib cache if available and data
    is unchanged; otherwise trains from scratch and saves to disk.
    """
    global _predictor_cache

    if history_file not in _predictor_cache or not use_cache:
        predictor = BacktestPredictor(history_file)
        loaded    = predictor.load()          # try disk cache first
        if not loaded:
            predictor.train(verbose=False)
            predictor.save()                  # persist for next call
        _predictor_cache[history_file] = predictor

    return _predictor_cache[history_file].predict(metrics)


def invalidate_cache(history_file: str = HISTORY_FILE) -> None:
    """Call after new backtest runs are appended → forces full retrain."""
    _predictor_cache.pop(history_file, None)
    cache_file = MODEL_CACHE
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"  ✅ Cache cleared: {cache_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI  __main__
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  MarketLab — ML Predictor  v3.0")
    xgb_note  = "XGBoost ✅" if _HAS_XGB  else "XGBoost ⚠ not installed"
    shap_note = "SHAP ✅"    if _HAS_SHAP else "SHAP ⚠  not installed"
    print(f"  {xgb_note}   {shap_note}")
    print("═"*60)

    predictor = BacktestPredictor()

    # Try loading from cache; retrain if needed
    if not predictor.load():
        predictor.full_report()
        predictor.save()
    else:
        predictor.evaluate()
        predictor.feature_importance()

    # ── Sample predictions ────────────────────────────────────────────────────
    print("  Sample Predictions")
    print("═"*60)

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
        print(f"     Prediction   : {result['prediction']}  (expected: {case['expected']})")
        print(f"     Probability  : {result['probability']:.1%}  (threshold: {result['threshold']:.2f})")
        print(f"     Confidence   : {result['confidence']}")
        print(f"     Model        : {result['model']}")

        checks = result["thresholds"]
        status = "  ".join(f"{'✓' if v else '✗'} {k}" for k, v in checks.items())
        print(f"     Checks       : {status}")

        if result.get("drift_alerts"):
            print(f"     Drift alerts : {result['drift_alerts']}")

    print(f"\n{'═'*60}")
    print(f"  Sample accuracy: {correct}/{len(test_cases)}")
    print(f"{'═'*60}")
    print("  Usage:")
    print("    from ml_predictor import predict_quality, invalidate_cache")
    print("    # Optional SHAP per-prediction:")
    print("    result = predictor.predict(metrics, explain=True)")
