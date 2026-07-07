"""Train the rich-poker-2 "forest" candidate model.

Forest-diverse stack (LightGBM + RandomForest + ExtraTrees + scaled logistic
regression) with a logistic meta-learner and hand-ngram side features over an
extended chunk-feature space (q25/q75 aggregates plus a showdown-rate signal).
Holdout and recency windows are derived dynamically from the freshest
benchmark source dates so the daily auto-retrain never goes stale. The
candidate artifact is written to artifacts/candidate.joblib;
training/promote_if_better.py decides deployment.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poker44_ml.calibration import BlendedQuantileCalibrator
from poker44_ml.hand_ngram import HandNgramEnsemble, hand_ngram_doc
from poker44_ml.stacked import StackedEnsemble
from poker44_ml.variant import VARIANT
from training.build_dataset import load_benchmark_examples
from training.fetch_benchmark import DATASET_PATH
from training.train_hand_ngram import _fit_calibration as fit_hgram_calibration
from training.train_model_v2 import (
    _apply_score_remap_np,
    _enrich_metrics,
    _logit_shift,
    _select_score_remap_for_validator_reward,
)

SEED = int(VARIANT["seed"])
N_JOBS = 3
MAX_FPR = 0.10
MAX_VALIDATOR_FPR = 0.08
BATCH_BOTS = 50
BATCH_HUMANS = 50
HUMAN_FINAL_MAX = 0.30
OUT_PATH = REPO_ROOT / "artifacts" / "candidate.joblib"


def git_output(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return ""


def build_base_models() -> list:
    return [
        lgb.LGBMClassifier(
            objective="binary", n_estimators=1000, num_leaves=31,
            learning_rate=0.025, min_child_samples=5, subsample=0.8,
            colsample_bytree=0.9, reg_alpha=0.05, reg_lambda=0.5,
            random_state=SEED, n_jobs=N_JOBS, verbosity=-1,
        ),
        RandomForestClassifier(
            n_estimators=700, max_depth=16, min_samples_leaf=2,
            max_features="sqrt", random_state=SEED, n_jobs=N_JOBS,
        ),
        ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=4,
            max_features="log2", random_state=SEED, n_jobs=N_JOBS,
        ),
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, max_iter=2000, random_state=SEED),
        ),
    ]


def fit_with_weights(model, x, y, sw):
    try:
        model.fit(x, y, sample_weight=sw)
    except (TypeError, ValueError):
        # sklearn Pipelines reject a bare sample_weight (ValueError on recent
        # versions); route it to the final step instead.
        step_name = model.steps[-1][0]
        model.fit(x, y, **{f"{step_name}__sample_weight": sw})


def build_hand_ngram_model(train_ex: list[dict], test_ex: list[dict]) -> HandNgramEnsemble:
    min_token = int(VARIANT["hgram_min_token"])
    lgb_weight = float(VARIANT["hgram_lgb_weight"])
    counter: Counter = Counter()
    for ex in train_ex + test_ex:
        for hand in ex.get("chunk") or []:
            if isinstance(hand, dict):
                counter.update(hand_ngram_doc(hand))
    vocab = {key: idx for idx, key in enumerate(k for k, c in counter.items() if c >= min_token)}

    def ex_to_row(ex: dict) -> np.ndarray | None:
        items: list = []
        for hand in ex.get("chunk") or []:
            if isinstance(hand, dict):
                items.extend(hand_ngram_doc(hand).items())
        if not items:
            return None
        row = np.zeros(len(vocab), dtype=np.float32)
        for key, value in items:
            column = vocab.get(key)
            if column is not None:
                row[column] += value
        return row

    train_rows, train_labels = [], []
    for ex in train_ex:
        row = ex_to_row(ex)
        if row is not None:
            train_rows.append(row)
            train_labels.append(int(ex.get("label", 0)))
    x_tr = np.array(train_rows)
    y_tr = np.array(train_labels, dtype=int)
    sw = np.where(y_tr == 0, float(VARIANT["human_weight"]), 1.0).astype(np.float64)
    sw /= sw.mean()

    lgb_h = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=31, learning_rate=0.05,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=N_JOBS, verbosity=-1,
    )
    lgb_h.fit(x_tr, y_tr, sample_weight=sw)
    lr_h = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=500, C=0.5, random_state=SEED),
    )
    lr_h.fit(x_tr, y_tr, logisticregression__sample_weight=sw)

    hold_raw, hold_labels = [], []
    for ex in test_ex:
        row = ex_to_row(ex)
        if row is None:
            continue
        x = row.reshape(1, -1)
        prob = (
            lgb_weight * lgb_h.predict_proba(x)[:, 1]
            + (1.0 - lgb_weight) * lr_h.predict_proba(x)[:, 1]
        )
        hold_raw.append(float(prob[0]))
        hold_labels.append(int(ex.get("label", 0)))

    center, scale = fit_hgram_calibration(
        np.array(hold_raw), np.array(hold_labels), score_low=0.04, score_high=0.49,
    )
    return HandNgramEnsemble(
        vocab, lgb_h, lr_h, lgb_weight=lgb_weight,
        stretch_center=center, stretch_scale=scale,
    )


def hgram_chunk_features(hgram_model: HandNgramEnsemble, chunk: list) -> dict[str, float]:
    hands = [h for h in (chunk or []) if isinstance(h, dict)]
    if not hands:
        return {"hgram_mean": 0.0, "hgram_max": 0.0, "hgram_std": 0.0}
    probs = hgram_model._hand_probs(hands)
    return {
        "hgram_mean": float(np.mean(probs)),
        "hgram_max": float(np.max(probs)),
        "hgram_std": float(np.std(probs)) if len(probs) > 1 else 0.0,
    }


def validator_reward(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    metrics = _enrich_metrics(
        np.asarray(y_true, dtype=int).tolist(),
        np.asarray(scores, dtype=float).tolist(),
    )
    return (
        float(metrics.get("validator_reward", 0.0)),
        float(metrics.get("pr_auc", 0.0)),
        float(metrics.get("validator_bot_recall", 0.0)),
        float(metrics.get("validator_fpr", 1.0)),
    )


def pipeline_scores(raw: np.ndarray, score_remap: dict | None, bias: float) -> np.ndarray:
    remapped = _apply_score_remap_np(np.asarray(raw, dtype=float), score_remap or {})
    return _logit_shift(remapped, bias, 1.0)


def main() -> int:
    t0 = time.time()
    print(f"=== {VARIANT['name']} training ({VARIANT['framework']}) ===", flush=True)

    examples = load_benchmark_examples(DATASET_PATH, miner_visible=True)
    dates = sorted({e["source_date"] for e in examples if e.get("source_date")})
    latest = date.fromisoformat(dates[-1])
    holdout_dates = set(dates[-int(VARIANT["holdout_days"]):])
    recent_cutoff = (latest - timedelta(days=int(VARIANT["recent_days"]) - 1)).isoformat()
    recent_dates = {d for d in dates if d >= recent_cutoff}
    print(
        f"Loaded {len(examples)} examples across {len(dates)} dates "
        f"(latest={latest}) holdout={sorted(holdout_dates)} "
        f"recent_window={sorted(recent_dates)[:1]}..{dates[-1]}",
        flush=True,
    )

    train_ex = [e for e in examples if e["source_date"] not in holdout_dates]
    test_ex = [e for e in examples if e["source_date"] in holdout_dates]
    print(f"Train={len(train_ex)} Holdout={len(test_ex)}", flush=True)

    print("Training hand-ngram side model...", flush=True)
    hgram_model = build_hand_ngram_model(train_ex, test_ex)
    for ex in examples:
        ex["features"].update(hgram_chunk_features(hgram_model, ex.get("chunk")))

    feat_names = sorted(examples[0]["features"].keys())

    def featurize(exs: list[dict]) -> np.ndarray:
        return np.array(
            [[float(e["features"].get(n, 0)) for n in feat_names] for e in exs],
            dtype=np.float32,
        )

    def sample_weights_for(exs: list[dict]) -> np.ndarray:
        sw = np.ones(len(exs), dtype=np.float64)
        for i, e in enumerate(exs):
            if e.get("label") == 0:
                sw[i] = float(VARIANT["human_weight"])
            if e.get("source_date") in recent_dates:
                sw[i] *= float(VARIANT["recency_boost"])
        return sw / sw.mean()

    x_train = featurize(train_ex)
    y_train = np.array([e["label"] for e in train_ex])
    x_test = featurize(test_ex)
    y_test = np.array([e["label"] for e in test_ex])
    sw_train = sample_weights_for(train_ex)

    n_folds = int(VARIANT["cv_folds"])
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    base_count = len(build_base_models())
    oof_cols = np.zeros((len(y_train), base_count))
    print(f"{n_folds}-fold CV over {base_count} base learners:", flush=True)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(x_train, y_train)):
        fold_models = build_base_models()
        for column, model in enumerate(fold_models):
            fit_with_weights(model, x_train[tr_idx], y_train[tr_idx], sw_train[tr_idx])
            oof_cols[va_idx, column] = model.predict_proba(x_train[va_idx])[:, 1]
        fold_ap = average_precision_score(y_train[va_idx], oof_cols[va_idx].mean(axis=1))
        print(f"  Fold {fold + 1}: mean-blend AP={fold_ap:.4f}", flush=True)

    meta = LogisticRegression(C=float(VARIANT["meta_c"]), max_iter=1000, random_state=SEED)
    meta.fit(oof_cols, y_train, sample_weight=sw_train)
    oof_blend = meta.predict_proba(oof_cols)[:, 1]
    print(f"OOF stacked AP={average_precision_score(y_train, oof_blend):.4f}", flush=True)

    calibrator = BlendedQuantileCalibrator(blend=float(VARIANT["quantile_blend"])).fit(oof_blend)
    oof_calibrated = calibrator.transform(oof_blend)

    _, cal_hold = train_test_split(
        list(range(len(train_ex))), test_size=0.25, random_state=SEED, stratify=y_train,
    )
    cal_idx = np.array(cal_hold)
    print("Tuning score_remap on calibration split...", flush=True)
    score_remap, remap_metrics = _select_score_remap_for_validator_reward(
        y_train[cal_idx],
        oof_calibrated[cal_idx],
        target_fpr=0.04,
        max_validator_fpr=MAX_VALIDATOR_FPR,
        calibration_objective="ap_first",
        temperature_grid=[0.12, 0.18, 0.25, 0.35, 0.50, 0.65, 0.85, 1.0, 1.25],
        prefer_smooth_remap=True,
    )
    print(
        f"  cal remap={score_remap} ap={remap_metrics.get('pr_auc', 0):.4f} "
        f"recall={remap_metrics.get('validator_bot_recall', 0):.3f} "
        f"fpr={remap_metrics.get('validator_fpr', 0):.3f}",
        flush=True,
    )

    x_all = featurize(examples)
    y_all = np.array([e["label"] for e in examples])
    sw_all = sample_weights_for(examples)
    prod_models = build_base_models()
    for model in prod_models:
        fit_with_weights(model, x_all, y_all, sw_all)

    raw_test = meta.predict_proba(
        np.column_stack([m.predict_proba(x_test)[:, 1] for m in prod_models])
    )[:, 1]
    raw_test = calibrator.transform(raw_test)
    print(f"Holdout raw AP={average_precision_score(y_test, raw_test):.4f}", flush=True)

    print("Re-tuning remap on holdout distribution...", flush=True)
    hold_remap, _ = _select_score_remap_for_validator_reward(
        y_test,
        raw_test,
        target_fpr=0.05,
        max_validator_fpr=MAX_FPR,
        calibration_objective="ap_first",
        temperature_grid=[0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.85, 1.0, 1.25, 1.5],
        prefer_smooth_remap=False,
    )
    if hold_remap:
        final = pipeline_scores(raw_test, hold_remap, 0.0)
        rew, _, recall, fpr = validator_reward(y_test, final)
        if recall >= 0.5 and fpr < MAX_FPR:
            score_remap = hold_remap
            print(f"  holdout remap={score_remap} reward={rew:.4f} recall={recall:.3f}", flush=True)
        else:
            print(f"  holdout remap failed guard (recall={recall:.3f}); keeping cal remap", flush=True)

    bot_raw = raw_test[y_test == 1]
    hum_raw = raw_test[y_test == 0]
    rng = np.random.default_rng(SEED)
    mixed_raw = np.concatenate([
        rng.choice(hum_raw, BATCH_HUMANS, replace=True) if len(hum_raw) else hum_raw,
        rng.choice(bot_raw, BATCH_BOTS, replace=True) if len(bot_raw) else bot_raw,
    ])
    mixed_labels = np.array([0] * BATCH_HUMANS + [1] * BATCH_BOTS)
    scenarios = [("holdout", raw_test, y_test), ("mixed50", mixed_raw, mixed_labels)]

    print("Tuning logit bias (recall>=0.5, fpr<10%, human_max guard)...", flush=True)
    best = {"reward": -1.0, "bias": 0.0}
    for bias in np.arange(-0.3, 1.5, 0.05):
        ok = True
        holdout_reward = 0.0
        for name, raw, labels in scenarios:
            final = pipeline_scores(raw, score_remap, float(bias))
            rew, _, recall, fpr = validator_reward(labels, final)
            human_max = float(final[labels == 0].max()) if np.any(labels == 0) else 0.0
            if fpr >= MAX_FPR or recall < 0.5 or human_max > HUMAN_FINAL_MAX:
                ok = False
                break
            if name == "holdout":
                holdout_reward = rew
        if ok and holdout_reward > best["reward"]:
            best = {"reward": holdout_reward, "bias": float(bias)}
    optimal_bias = best["bias"] if best["reward"] >= 0 else 0.0
    print(f"  Selected bias={optimal_bias:.2f} holdout_reward={best['reward']:.4f}", flush=True)

    final_test = pipeline_scores(raw_test, score_remap, optimal_bias)
    rew, ap, recall, fpr = validator_reward(y_test, final_test)
    print(
        f"Final holdout: reward={rew:.4f} AP={ap:.4f} recall={recall:.3f} FPR={fpr:.3f} "
        f"human_max={float(final_test[y_test == 0].max()) if np.any(y_test == 0) else 0:.4f}",
        flush=True,
    )

    model_name = os.getenv("POKER44_MODEL_NAME", "").strip() or str(VARIANT["name"])
    model_version = latest.isoformat().replace("-", ".")
    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "framework": str(VARIANT["framework"]),
        "variant": str(VARIANT["key"]),
        "variant_description": str(VARIANT["description"]),
        "artifact_filename": OUT_PATH.name,
        "repo_url": git_output("config", "--get", "remote.origin.url"),
        "repo_commit": git_output("rev-parse", "HEAD"),
        "benchmark_rows": float(len(y_all)),
        "train_latest_date": dates[-1],
        "train_total_examples": int(len(y_all)),
        "holdout_source_dates": sorted(holdout_dates),
        "recent_source_dates": sorted(recent_dates),
        "human_weight_multiplier": float(VARIANT["human_weight"]),
        "recency_boost_multiplier": float(VARIANT["recency_boost"]),
        "quantile_calibration_blend": float(VARIANT["quantile_blend"]),
        "cv_folds": n_folds,
        "seed": SEED,
        "score_remap": dict(score_remap) if score_remap else {},
        "score_logit_bias": float(optimal_bias),
        "score_logit_temperature": 1.0,
        "holdout_ap": float(ap),
        "holdout_bot_recall": float(recall),
        "holdout_fpr": float(fpr),
        "holdout_reward": float(rew),
        "hgram_stretch_center": float(hgram_model.stretch_center or 0),
        "hgram_stretch_scale": float(hgram_model.stretch_scale or 0),
        "stack_meta_weights": meta.coef_.tolist(),
        "ensemble_combiner": "stacking_logreg+blended_quantile+score_remap+score_logit",
    }

    artifact = {
        "models": [
            StackedEnsemble(base_models=prod_models, meta_model=meta, calibrator=calibrator)
        ],
        "model_weights": [1.0],
        "feature_names": feat_names,
        "metadata": metadata,
        "model_name": model_name,
        "model_version": model_version,
        "hand_ngram_model": hgram_model,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUT_PATH, compress=3)
    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"Saved candidate: {OUT_PATH}")
    print(f"SHA256: {sha}")
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
