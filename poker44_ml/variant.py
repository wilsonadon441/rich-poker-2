"""Variant identity for this miner deployment.

rich-poker-2 runs the "forest" variant: a forest-diverse stack
(LightGBM + RandomForest + ExtraTrees + scaled logistic regression) over an
extended chunk-feature space (q25/q75 aggregates plus a showdown-rate signal)
with a hand-ngram side model, calibrated holdout-first against the validator
reward.
"""
from __future__ import annotations

VARIANT: dict = {
    "key": "forest",
    "name": "rich-poker-2",
    "framework": "stack-lgb-rf-et-lr-hgram-forest-r2",
    "seed": 2029,
    "cv_folds": 6,
    "human_weight": 30.0,
    "recency_boost": 5.0,
    "recent_days": 10,
    "holdout_days": 3,
    "quantile_blend": 0.25,
    "hgram_lgb_weight": 0.5,
    "hgram_min_token": 30,
    "meta_c": 0.6,
    "description": (
        "Forest-diverse stacked ensemble (LightGBM, RandomForest, ExtraTrees, "
        "scaled logistic regression) with hand-ngram side features over an "
        "extended chunk-feature space with q25/q75 aggregates and a "
        "showdown-rate signal."
    ),
}
