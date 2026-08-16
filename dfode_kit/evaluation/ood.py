from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiagonalOODEnvelope:
    state_mean: np.ndarray
    state_std: np.ndarray
    log_dt_mean: float
    log_dt_std: float
    threshold: float

    def score(self, states: np.ndarray, dt: np.ndarray) -> np.ndarray:
        state = np.asarray(states, dtype=np.float64)
        intervals = np.asarray(dt, dtype=np.float64).reshape(-1)
        normalized_state = np.abs(
            (state - self.state_mean[None, :]) / self.state_std[None, :]
        )
        normalized_dt = np.abs(
            (np.log(np.maximum(intervals, 1.0e-300)) - self.log_dt_mean)
            / self.log_dt_std
        )
        return np.maximum(np.max(normalized_state, axis=1), normalized_dt)

    def is_ood(self, states: np.ndarray, dt: np.ndarray) -> np.ndarray:
        return self.score(states, dt) > self.threshold

    def to_dict(self) -> dict:
        return {
            "schema": "dfode.diagonal-ood-envelope",
            "schema_version": 1,
            "score": "max_abs_training_z_score",
            "threshold": float(self.threshold),
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "log_dt_mean": float(self.log_dt_mean),
            "log_dt_std": float(self.log_dt_std),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DiagonalOODEnvelope":
        if payload.get("schema") != "dfode.diagonal-ood-envelope":
            raise ValueError("unsupported OOD envelope schema")
        return cls(
            state_mean=np.asarray(payload["state_mean"], dtype=np.float64),
            state_std=np.asarray(payload["state_std"], dtype=np.float64),
            log_dt_mean=float(payload["log_dt_mean"]),
            log_dt_std=float(payload["log_dt_std"]),
            threshold=float(payload["threshold"]),
        )


def envelope_from_checkpoint(
    checkpoint: dict,
    states: np.ndarray,
    dt: np.ndarray,
    *,
    quantile: float = 0.999,
) -> tuple[DiagonalOODEnvelope, dict]:
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be in (0.5, 1.0)")
    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float64)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float64)
    log_dt_mean = float(np.asarray(checkpoint["log_dt_mean"]).reshape(-1)[0])
    log_dt_std = float(np.asarray(checkpoint["log_dt_std"]).reshape(-1)[0])
    state_std = np.where(state_std > 0.0, state_std, 1.0)
    log_dt_std = log_dt_std if log_dt_std > 1.0e-12 else 1.0
    provisional = DiagonalOODEnvelope(
        state_mean, state_std, log_dt_mean, log_dt_std, np.inf
    )
    scores = provisional.score(states, dt)
    threshold = float(np.quantile(scores, quantile))
    envelope = DiagonalOODEnvelope(
        state_mean, state_std, log_dt_mean, log_dt_std, threshold
    )
    summary = {
        "sample_count": int(scores.size),
        "threshold_quantile": float(quantile),
        "score_p50": float(np.quantile(scores, 0.50)),
        "score_p90": float(np.quantile(scores, 0.90)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "score_max": float(np.max(scores)),
    }
    return envelope, summary
