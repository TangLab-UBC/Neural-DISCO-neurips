import re

import numpy as np
import pandas as pd


ANGLE_THRESH_DEG = 30.0
HARD_MATCH_FIELDS = ["trial_version", "num_targets", "num_barriers", "active_target"]
SOFT_MATCH_FIELDS = ["delay", "rt"]
PAIRING_STRATEGY = "context_matched"


def parse_target_pos(value):
    nums = [int(x) for x in re.findall(r"-?\d+", str(value))]
    if len(nums) == 0 or len(nums) % 2 != 0:
        return None
    return np.array(nums, dtype=float).reshape(-1, 2)


def active_target_xy(row):
    arr = parse_target_pos(row["target_pos"])
    if arr is None:
        return np.array([np.nan, np.nan])
    idx = int(row["active_target"]) if not pd.isna(row["active_target"]) else 0
    idx = max(0, min(idx, len(arr) - 1))
    return arr[idx]


def angle_diff(a, b):
    diff = abs(a - b)
    return min(diff, 360.0 - diff)


def _with_target_directions(trial_info):
    xy = np.vstack([active_target_xy(row) for _, row in trial_info.iterrows()])
    target_dir_deg = (np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) + 360.0) % 360.0
    trial_info = trial_info.copy()
    trial_info["target_dir_deg"] = target_dir_deg
    return trial_info


def _standardized_soft_values(trial_info):
    soft_vals = trial_info[SOFT_MATCH_FIELDS].astype(float).to_numpy()
    soft_mean = np.nanmean(soft_vals, axis=0)
    soft_std = np.nanstd(soft_vals, axis=0)
    soft_std[~np.isfinite(soft_std) | (soft_std == 0)] = 1.0
    soft_vals = (soft_vals - soft_mean) / soft_std
    soft_vals[~np.isfinite(soft_vals)] = 0.0
    return soft_vals


def _direction_candidates(trial_info, idx, angle_thresh_deg):
    src_angle = trial_info.loc[idx, "target_dir_deg"]
    candidates = []
    for cand in trial_info.index.to_numpy():
        if cand == idx:
            continue
        cand_angle = trial_info.loc[cand, "target_dir_deg"]
        if angle_diff(src_angle, cand_angle) < angle_thresh_deg:
            continue
        candidates.append(int(cand))
    return candidates


def _build_context_matched_pairs(trial_info, angle_thresh_deg):
    soft_vals = _standardized_soft_values(trial_info)
    pairs = []
    for _, group in trial_info.groupby(HARD_MATCH_FIELDS, dropna=False, sort=False):
        idxs = group.index.to_numpy()
        if len(idxs) < 2:
            continue
        for idx in idxs:
            candidates = [cand for cand in _direction_candidates(trial_info, idx, angle_thresh_deg) if cand in idxs]
            if not candidates:
                continue
            dists = np.linalg.norm(soft_vals[candidates] - soft_vals[idx], axis=1)
            target_idx = int(candidates[int(np.argmin(dists))])
            pairs.append((int(idx), target_idx))
    return pairs


def build_relaxed_pairs(trial_info, angle_thresh_deg=ANGLE_THRESH_DEG):
    """Pair source/target trials for MC Maze counterfactual evaluation.

    Uses the paper's ``context_matched`` pairing: each source trial is paired
    with the trial in the same task context (``HARD_MATCH_FIELDS``) whose
    target direction differs by at least ``angle_thresh_deg`` and whose
    standardized delay/RT is nearest. The pairing is deterministic.
    """
    trial_info = _with_target_directions(trial_info)
    return _build_context_matched_pairs(trial_info, angle_thresh_deg)
