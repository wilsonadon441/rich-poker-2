"""Promote the freshly trained candidate artifact if it beats the deployed one.

Both artifacts are scored with the full runtime inference path (Poker44Model)
on an evaluation slice built from the newest benchmark source dates, and
compared on the validator reward. The candidate must also clear the validator
safety guards (FPR < 10%, bot recall >= 50%) before it can be promoted.

Exit codes: 0 = promoted, 3 = kept deployed model, 1 = error.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poker44.score.scoring import reward as validator_reward
from poker44_ml.inference import Poker44Model
from training.build_dataset import load_benchmark_examples
from training.fetch_benchmark import DATASET_PATH

EVAL_HOLDOUT_DAYS = 2
MAX_FPR = 0.10
MIN_RECALL = 0.50
# Small tolerance in the candidate's favor: on a tie we prefer the model
# trained on the freshest benchmark data.
PROMOTE_TOLERANCE = 0.002

CANDIDATE_PATH = REPO_ROOT / "artifacts" / "candidate.joblib"


def eval_model(model_path: Path, chunks: list, labels: np.ndarray) -> dict:
    model = Poker44Model(model_path)
    scores = np.asarray(model.predict_chunk_scores(chunks), dtype=float)
    rew, details = validator_reward(scores, labels)
    return {
        "reward": float(rew),
        "fpr": float(details.get("fpr", 1.0)),
        "recall": float(details.get("bot_recall", 0.0)),
        "ap": float(details.get("ap_score", 0.0)),
    }


def main() -> int:
    deploy_path = Path(
        os.getenv("POKER44_MODEL_PATH", str(REPO_ROOT / "models" / "deploy.joblib"))
    )
    if not CANDIDATE_PATH.exists():
        print(f"ERROR: candidate artifact missing at {CANDIDATE_PATH}", flush=True)
        return 1

    examples = load_benchmark_examples(DATASET_PATH, miner_visible=True)
    dates = sorted({e["source_date"] for e in examples if e.get("source_date")})
    eval_dates = set(dates[-EVAL_HOLDOUT_DAYS:])
    eval_ex = [e for e in examples if e.get("source_date") in eval_dates]
    chunks = [e["chunk"] for e in eval_ex]
    labels = np.asarray([int(e["label"]) for e in eval_ex], dtype=int)
    print(
        f"Eval slice: dates={sorted(eval_dates)} chunks={len(chunks)} "
        f"bots={int(labels.sum())} humans={int((labels == 0).sum())}",
        flush=True,
    )

    candidate = eval_model(CANDIDATE_PATH, chunks, labels)
    print(f"Candidate: {candidate}", flush=True)

    if candidate["fpr"] >= MAX_FPR or candidate["recall"] < MIN_RECALL:
        print(
            f"KEPT: candidate fails safety guards (fpr={candidate['fpr']:.3f} "
            f"recall={candidate['recall']:.3f}); keeping deployed model.",
            flush=True,
        )
        return 3

    if deploy_path.exists():
        try:
            deployed = eval_model(deploy_path, chunks, labels)
            print(f"Deployed:  {deployed}", flush=True)
        except Exception as err:  # noqa: BLE001 - corrupt artifact must not block promotion
            print(f"Deployed model unusable ({err}); promoting candidate.", flush=True)
            deployed = {"reward": -1.0}
        if candidate["reward"] + PROMOTE_TOLERANCE < deployed["reward"]:
            print(
                f"KEPT: deployed reward {deployed['reward']:.4f} beats candidate "
                f"{candidate['reward']:.4f}; keeping deployed model.",
                flush=True,
            )
            return 3

    deploy_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = deploy_path.with_suffix(".tmp")
    shutil.copyfile(CANDIDATE_PATH, tmp_path)
    tmp_path.replace(deploy_path)
    print(f"PROMOTED: candidate deployed to {deploy_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
