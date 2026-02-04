"""
evaluation.py

Evaluation utilities for sparse mobility tensors (U x T x L) stored as 2D matrices
of shape (U, T*L).

Inputs (all 2D, same shape):
    WH        : continuous predictions (scores)
    input_mx  : simulated smartphone app matrix (0/1, with missingness)
    gt_mx     : ground truth matrix (0/1, no missing)
    zero_mx   : 0/1 mask where GT is known to be 0 (for zero-check)

Main entry:
    conduct_evaluation(...)
        -> returns (df_pred, df_impute)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import entropy
basepath = "C:\\Users\\Yiran\\OneDrive - UW\\Simulation\\2ndstage_test\\"

EPS = 1e-8


# ============================
# helpers: 2D <-> 3D reshape
# ============================

def to_3d(mx_2d: np.ndarray, U: int, T: int, L: int) -> np.ndarray:
    """
    Reshape 2D matrix (U, T*L) to 3D (U, T, L).
    """
    return mx_2d.reshape(U, T, L)


# ============================
# simple conversions
# ============================

def threshold_to_01(WH: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert continuous WH scores to binary predictions using a threshold.
    """
    WH_01 = np.zeros_like(WH)
    WH_01[WH >= threshold] = 1
    return WH_01


def build_impute_mx(pred_01: np.ndarray, input_mx: np.ndarray) -> np.ndarray:
    """
    Final imputed matrix: OR of input_mx and pred_01.
    """
    return np.where((pred_01 == 1) | (input_mx == 1), 1, 0)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute RMSE between two same-shaped arrays.
    Independent of scikit-learn's 'squared' kw.
    """
    diff = a - b
    return float(np.sqrt(np.mean(diff ** 2)))


def get_rmse_orig_gt(WH: np.ndarray, input_mx: np.ndarray, gt_mx: np.ndarray):
    """
    RMSE vs smartphone matrix and vs ground truth.
    """
    rmse_orig = np.round(rmse(input_mx, WH), 6)
    rmse_gt   = np.round(rmse(gt_mx,   WH), 6)
    return rmse_orig, rmse_gt


def get_accuracy_score_flat(WH: np.ndarray,
                            input_mx: np.ndarray,
                            gt_mx: np.ndarray):
    """
    Accuracy vs smartphone matrix and vs ground truth, flattened.
    """
    accuracy_ori = np.round(accuracy_score(input_mx.flatten(), WH.flatten()), 6)
    accuracy_gt  = np.round(accuracy_score(gt_mx.flatten(),   WH.flatten()), 6)
    return accuracy_ori, accuracy_gt


def zero_matrix_eval(WH: np.ndarray, zero_mx: np.ndarray) -> float:
    """
    Fraction of certain zeros that are incorrectly predicted as 1.
    zero_mx: 1 where GT is certainly 0 (or equivalent constraint region).
    """
    certain_zeros    = np.count_nonzero(zero_mx)
    pred_wrong_zeros = np.count_nonzero(WH * zero_mx)
    return np.round(pred_wrong_zeros / (certain_zeros + EPS), 6)


# ============================
# core accuracy metrics
# ============================

def prf_global(pred_3d: np.ndarray, gt_3d: np.ndarray):
    """
    Global precision / recall / F1 vs ground truth (flattened).
    """
    y_true = gt_3d.flatten()
    y_pred = pred_3d.flatten()
    return (
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true,    y_pred, zero_division=0),
        f1_score(y_true,        y_pred, zero_division=0),
    )


def accuracy_missing_only(pred_3d: np.ndarray,
                          gt_3d: np.ndarray,
                          diff_3d: np.ndarray):
    """
    Accuracy restricted to originally missing positives (diff_3d == 1).
    """
    mask = diff_3d.astype(bool)
    y_true = gt_3d[mask]
    y_pred = pred_3d[mask]
    if y_true.size == 0:
        return np.nan
    return accuracy_score(y_true, y_pred)


def user_jaccard(pred_3d: np.ndarray, gt_3d: np.ndarray) -> np.ndarray:
    """
    Jaccard similarity per user on (time, zone) support.
    """
    U = pred_3d.shape[0]
    out = np.zeros(U)
    for u in range(U):
        pred_pos = set(zip(*np.where(pred_3d[u] == 1)))
        gt_pos   = set(zip(*np.where(gt_3d[u]   == 1)))
        if len(pred_pos) == 0 and len(gt_pos) == 0:
            out[u] = 1.0
        else:
            inter = len(pred_pos & gt_pos)
            union = len(pred_pos | gt_pos)
            out[u] = inter / union if union > 0 else 0
    return out


# ============================
# coverage metrics (temporal, spatial, joint)
# ============================

def temporal_coverage(gt_3d: np.ndarray, X_3d: np.ndarray):
    """
    Coverage over time: total activity per time step.
    """
    C_gt = gt_3d.sum(axis=(0, 2))  # (T,)
    C_X  = X_3d.sum(axis=(0, 2))   # (T,)
    cov = C_X / (C_gt + EPS)
    return C_gt, C_X, cov


def spatial_coverage(gt_3d: np.ndarray, X_3d: np.ndarray):
    """
    Coverage over zones: total activity per zone.
    """
    Z_gt = gt_3d.sum(axis=(0, 1))  # (L,)
    Z_X  = X_3d.sum(axis=(0, 1))   # (L,)
    cov = Z_X / (Z_gt + EPS)
    return Z_gt, Z_X, cov


def temporal_coverage_error(gt_3d: np.ndarray, X_3d: np.ndarray) -> float:
    """
    Mean absolute deviation of temporal coverage ratio from 1.
    """
    _, _, cov = temporal_coverage(gt_3d, X_3d)
    return float(np.mean(np.abs(cov - 1.0)))


def spatial_coverage_error(gt_3d: np.ndarray, X_3d: np.ndarray) -> float:
    """
    Mean absolute deviation of spatial coverage ratio from 1.
    """
    _, _, cov = spatial_coverage(gt_3d, X_3d)
    return float(np.mean(np.abs(cov - 1.0)))


def joint_coverage_error(gt_3d: np.ndarray, X_3d: np.ndarray) -> float:
    """
    Normalized Frobenius norm error between (time,zone) heatmaps.
    """
    H_gt = gt_3d.sum(axis=0)  # (T, L)
    H_X  = X_3d.sum(axis=0)
    num = np.linalg.norm(H_X - H_gt)
    den = np.linalg.norm(H_gt) + EPS
    return float(num / den)


# Missing-only versions

def temporal_coverage_missing(gt_3d: np.ndarray,
                              X_3d: np.ndarray,
                              diff_3d: np.ndarray):
    """
    Temporal coverage restricted to originally missing positives.
    """
    gt_mask = gt_3d * diff_3d
    X_mask  = X_3d  * diff_3d
    C_gt = gt_mask.sum(axis=(0, 2))
    C_X  = X_mask.sum(axis=(0, 2))
    cov  = C_X / (C_gt + EPS)
    return C_gt, C_X, cov


def spatial_coverage_missing(gt_3d: np.ndarray,
                             X_3d: np.ndarray,
                             diff_3d: np.ndarray):
    """
    Spatial coverage restricted to originally missing positives.
    """
    gt_mask = gt_3d * diff_3d
    X_mask  = X_3d  * diff_3d
    Z_gt = gt_mask.sum(axis=(0, 1))
    Z_X  = X_mask.sum(axis=(0, 1))
    cov  = Z_X / (Z_gt + EPS)
    return Z_gt, Z_X, cov


# ============================
# structural JS divergence
# ============================

def temporal_zone_profiles(m_3d: np.ndarray):
    """
    Marginal profiles over time and zones.
    """
    time_prof = m_3d.sum(axis=(0, 2))  # (T,)
    zone_prof = m_3d.sum(axis=(0, 1))  # (L,)
    return time_prof, zone_prof


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen–Shannon divergence between two non-negative vectors.
    """
    p = p.astype(float); q = q.astype(float)
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def structural_divergence(gt_3d: np.ndarray, X_3d: np.ndarray):
    """
    JS divergence between temporal and spatial profiles.
    """
    tp_X, zp_X   = temporal_zone_profiles(X_3d)
    tp_gt, zp_gt = temporal_zone_profiles(gt_3d)
    return js_divergence(tp_X, tp_gt), js_divergence(zp_X, zp_gt)


# ============================
# Core evaluation for pred_mx (thresholded WH)
# ============================

def evaluate_pred_matrix(pred_mx: np.ndarray,
                         input_mx: np.ndarray,
                         gt_mx: np.ndarray,
                         zero_mx: np.ndarray,
                         diff_mx: np.ndarray,
                         num_missing: int,
                         non_zero_count_gt: int,
                         threshold: float,
                         U: int, T: int, L: int):
    """
    Evaluate a binary prediction matrix (2D) against input & GT, and
    compute global + structural metrics.
    """
    # 2D metrics
    missing_rate = 1.0 - (
        np.count_nonzero(pred_mx * gt_mx) / (non_zero_count_gt + EPS)
    )

    fill_rate = np.count_nonzero(pred_mx * diff_mx) / (num_missing + EPS)

    rmse_orig, rmse_gt = get_rmse_orig_gt(pred_mx, input_mx, gt_mx)
    acc_ori, acc_gt    = get_accuracy_score_flat(pred_mx, input_mx, gt_mx)
    zero_check         = zero_matrix_eval(pred_mx, zero_mx)
    captured_rate      = (
        np.count_nonzero((pred_mx == 1) & (gt_mx == 1)) /
        (non_zero_count_gt + EPS)
    )

    # reshape to 3D
    pred_3d  = to_3d(pred_mx,  U, T, L)
    gt_3d    = to_3d(gt_mx,    U, T, L)
    diff_3d  = to_3d(diff_mx,  U, T, L)

    # new metrics (3D)
    precision_gt, recall_gt, f1_gt = prf_global(pred_3d, gt_3d)
    acc_missing = accuracy_missing_only(pred_3d, gt_3d, diff_3d)

    jacc = user_jaccard(pred_3d, gt_3d)
    jacc_mean = float(np.nanmean(jacc))
    jacc_med  = float(np.nanmedian(jacc))

    temp_cov_err  = temporal_coverage_error(gt_3d, pred_3d)
    spat_cov_err  = spatial_coverage_error(gt_3d, pred_3d)
    joint_cov_err = joint_coverage_error(gt_3d, pred_3d)

    _, _, temp_cov_missing = temporal_coverage_missing(gt_3d, pred_3d, diff_3d)
    _, _, spat_cov_missing = spatial_coverage_missing(gt_3d, pred_3d, diff_3d)
    temp_cov_missing_err = float(np.mean(np.abs(temp_cov_missing - 1.0)))
    spat_cov_missing_err = float(np.mean(np.abs(spat_cov_missing - 1.0)))

    js_time, js_zone = structural_divergence(gt_3d, pred_3d)

    return {
        "threshold": threshold,
        # original-style metrics
        "missing_rate": missing_rate,
        "fill_rate": fill_rate,
        "rmse_orig": rmse_orig,
        "rmse_gt": rmse_gt,
        "accuracy_smartphone_flat": acc_ori,
        "accuracy_gt_flat": acc_gt,
        "zero_check": zero_check,
        "correctly_captured_rate_gt": captured_rate,
        # global imputation metrics
        "precision_gt": precision_gt,
        "recall_gt": recall_gt,
        "f1_gt": f1_gt,
        "accuracy_missing_only": acc_missing,
        "user_jaccard_mean": jacc_mean,
        "user_jaccard_median": jacc_med,
        # coverage metrics
        "temp_coverage_error": temp_cov_err,
        "spat_coverage_error": spat_cov_err,
        "joint_coverage_error": joint_cov_err,
        "temp_coverage_missing_error": temp_cov_missing_err,
        "spat_coverage_missing_error": spat_cov_missing_err,
        # structural consistency
        "js_time": js_time,
        "js_zone": js_zone,
    }


# ============================
# Evaluation for impute_mx
# ============================

def evaluate_impute_matrix(pred_01: np.ndarray,
                           input_mx: np.ndarray,
                           gt_mx: np.ndarray,
                           zero_mx: np.ndarray,
                           diff_mx: np.ndarray,
                           num_missing: int,
                           non_zero_count_gt: int,
                           threshold: float,
                           U: int, T: int, L: int):
    """
    Evaluate the final imputed matrix:
        impute_mx = OR(input_mx, pred_01)
    using the same metric machinery as evaluate_pred_matrix.
    """
    impute = build_impute_mx(pred_01, input_mx)
    return evaluate_pred_matrix(
        impute,
        input_mx,
        gt_mx,
        zero_mx,
        diff_mx,
        num_missing,
        non_zero_count_gt,
        threshold,
        U, T, L,
    )


# ============================
# Final top-level evaluation loop
# ============================

def conduct_evaluation(
    WH: np.ndarray,
    input_mx: np.ndarray,
    gt_mx: np.ndarray,
    zero_mx: np.ndarray,
    U: int,
    T: int,
    L: int,
    thresholds=None,
    save_prefix: str | None = None,
    save_parquet: bool = False,
):
    """
    Run evaluation across thresholds.

    Parameters
    ----------
    WH : 2D array (U, T*L)
        Continuous scores from model.
    input_mx : 2D array (U, T*L)
        Smartphone/app-based observed binary matrix.
    gt_mx : 2D array (U, T*L)
        Ground truth binary matrix (no missingness).
    zero_mx : 2D array (U, T*L)
        Mask where GT is certainly 0 (for zero-check).
    U, T, L : int
        Shape parameters such that WH.shape == (U, T*L).
    thresholds : iterable of float or None
        Thresholds to sweep. If None, auto from 0.01 to max(WH) - 0.01.
    save_prefix : str or None
        If not None, results are saved as:
            <prefix>_pred.csv
            <prefix>_impute.csv
        (And optionally Parquet.)
    save_parquet : bool
        If True, also save Parquet files.

    Returns
    -------
    df_pred : pd.DataFrame
        Metrics for thresholded predictions (pred_mx).
    df_impute : pd.DataFrame
        Metrics for final imputed matrix (impute_mx = input OR pred).
    """
    # Precompute missingness mask and counts once
    diff_mx = ((gt_mx == 1) & (input_mx == 0)).astype(int)
    num_missing = int(np.count_nonzero(diff_mx))
    non_zero_count_gt = int(np.count_nonzero(gt_mx))

    if thresholds is None:
        thresholds = np.arange(0.01, np.round(WH.max(), 2) - 0.01, 0.01)

    pred_records = []
    imp_records  = []

    for th in thresholds:
        th = float(np.round(th, 2))
        pred_01 = threshold_to_01(WH, th)

        pred_records.append(
            evaluate_pred_matrix(
                pred_01,
                input_mx,
                gt_mx,
                zero_mx,
                diff_mx,
                num_missing,
                non_zero_count_gt,
                th,
                U, T, L,
            )
        )

        imp_records.append(
            evaluate_impute_matrix(
                pred_01,
                input_mx,
                gt_mx,
                zero_mx,
                diff_mx,
                num_missing,
                non_zero_count_gt,
                th,
                U, T, L,
            )
        )

    df_pred   = pd.DataFrame(pred_records)
    df_impute = pd.DataFrame(imp_records)

    # Optional saving
    if save_prefix is not None:
        pred_path   = f"{basepath}{save_prefix}_pred.csv"
        impute_path = f"{basepath}{save_prefix}_impute.csv"

        df_pred.to_csv(pred_path, index=False)
        df_impute.to_csv(impute_path, index=False)

        if save_parquet:
            df_pred.to_parquet(f"{basepath}{save_prefix}_pred.parquet", index=False)
            df_impute.to_parquet(f"{basepath}{save_prefix}_impute.parquet", index=False)

        print(f"Saved CSV results to:\n  {pred_path}\n  {impute_path}")
        if save_parquet:
            print(f"Saved Parquet results to:\n  {basepath}{save_prefix}_pred.parquet\n  {basepath}{save_prefix}_impute.parquet")

    #return df_pred, df_impute
