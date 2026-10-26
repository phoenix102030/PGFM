from typing import Dict, List, Sequence

import numpy as np


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))


def crps_deterministic(pred: np.ndarray, truth: np.ndarray) -> float:
    """CRPS for a deterministic point forecast, equal to absolute error."""
    return mae(pred, truth)


def bias(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(pred - truth))


def corr(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_1d = np.asarray(pred, dtype=np.float64).reshape(-1)
    truth_1d = np.asarray(truth, dtype=np.float64).reshape(-1)
    if pred_1d.size == 0 or truth_1d.size == 0:
        return float("nan")
    pred_std = float(np.std(pred_1d))
    truth_std = float(np.std(truth_1d))
    if pred_std < 1e-12 or truth_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred_1d, truth_1d)[0, 1])


def mape_eps(pred: np.ndarray, truth: np.ndarray, eps: float = 1.0) -> float:
    denom = np.maximum(np.abs(truth), float(eps))
    return float(100.0 * np.mean(np.abs(pred - truth) / denom))


def smape(pred: np.ndarray, truth: np.ndarray, eps: float = 1.0) -> float:
    denom = np.abs(pred) + np.abs(truth) + float(eps)
    return float(100.0 * np.mean(2.0 * np.abs(pred - truth) / denom))


def nmae(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.mean(np.abs(truth)))
    if denom < 1e-12:
        return float("nan")
    return float(mae(pred, truth) / denom)


def nrmse(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.std(np.asarray(truth, dtype=np.float64)))
    if denom < 1e-12:
        return float("nan")
    return float(rmse(pred, truth) / denom)


def point_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    percentage_eps_tecu: float = 1.0,
) -> Dict[str, float]:
    out = {
        "rmse": rmse(pred, truth),
        "mae": mae(pred, truth),
        "crps": crps_deterministic(pred, truth),
        "bias": bias(pred, truth),
        "corr": corr(pred, truth),
        "mape_eps": mape_eps(pred, truth, eps=percentage_eps_tecu),
        "smape": smape(pred, truth, eps=percentage_eps_tecu),
        "nmae": nmae(pred, truth),
        "nrmse": nrmse(pred, truth),
    }
    return out


def per_lead_metrics(
    pred_blocks: Sequence[np.ndarray],
    truth_blocks: Sequence[np.ndarray],
    horizon: int,
    percentage_eps_tecu: float = 1.0,
) -> List[Dict[str, float]]:
    pred_b = np.stack(pred_blocks, axis=0)
    truth_b = np.stack(truth_blocks, axis=0)
    rows: List[Dict[str, float]] = []
    for step in range(int(horizon)):
        row = point_metrics(
            pred_b[:, step],
            truth_b[:, step],
            percentage_eps_tecu=percentage_eps_tecu,
        )
        row = {"lead": int(step + 1), **row}
        rows.append(row)
    return rows
