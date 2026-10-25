import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

current_dir = os.getcwd()
sys.path.append(current_dir)

from scripts.cfact.mc_maze_mlp_decode_xy import (  # noqa: E402
    _cv_folds,
    decoder_predict,
    fit_decoder,
    _ci95,
    _normalize_between_controls,
)
from src.plotting import COUNTERFACTUAL_COLORS, COUNTERFACTUAL_SUMMARY_COLORS  # noqa: E402
from src.eval.mc_maze_windows import window_condition_target  # noqa: E402


def pooled_r2(true_arrays, pred_arrays):
    true_cat = np.concatenate(true_arrays, axis=0)
    pred_cat = np.concatenate(pred_arrays, axis=0)
    from sklearn.metrics import r2_score

    return float(r2_score(true_cat, pred_cat))


DISPLAY_CONTEXT_FIELDS = ("trial_version", "num_targets", "num_barriers", "active_target")
LOCKED_DISPLAY_FAMILY = {
    "source_bin": 3,
    "target_bin": 7,
    "position_pair": {
        "source_xy": [-108, 81],
        "target_xy": [131, -55],
    },
}

SUMMARY_R2_CLIP_FLOOR = -0.18
TRAJECTORY_GRID_MIN_ANGLE_GAP_DEG = 120.0


def _clip_summary_bars(values, yerr=None, floor=SUMMARY_R2_CLIP_FLOOR):
    values = np.asarray(values, dtype=float)
    plot_values = values.copy()
    clipped = np.isfinite(values) & (values < floor)
    plot_values[clipped] = floor
    if yerr is None:
        plot_yerr = None
    else:
        plot_yerr = np.asarray(yerr, dtype=float).copy()
        plot_yerr[clipped] = 0.0
    return plot_values, plot_yerr, clipped


def _annotate_clipped_summary_bars(ax, xs, raw_values, plot_values, clipped, yerr=None):
    if not np.any(clipped):
        return
    raw_values = np.asarray(raw_values, dtype=float)
    plot_values = np.asarray(plot_values, dtype=float)
    yerr = np.zeros_like(raw_values) if yerr is None else np.asarray(yerr, dtype=float)
    for xpos, raw_value, plotted_value, err, is_clipped in zip(xs, raw_values, plot_values, yerr, clipped):
        if not is_clipped:
            continue
        ax.plot(xpos, plotted_value, marker="v", color="#555555", markersize=4.5, clip_on=False, zorder=5)
        label = f"{raw_value:.1f}"
        if err > 0.05:
            label = f"{raw_value:.1f}±{err:.1f}"
        ax.annotate(
            label,
            xy=(xpos, plotted_value),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=6.5,
            color="#444444",
            clip_on=False,
        )


def _trajectory_angle(path):
    disp = _align_path(path)
    delta = disp[-1] - disp[0]
    if np.allclose(delta, 0):
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def _angle_distance(a, b):
    diff = abs(a - b)
    return min(diff, 2 * np.pi - diff)


def _trajectory_shape_distance(path_a, path_b, n_points=80):
    path_a = resample_path(_align_path(path_a), n_points=n_points)
    path_b = resample_path(_align_path(path_b), n_points=n_points)
    stack = np.concatenate([path_a, path_b], axis=0)
    scale = max(float(np.ptp(stack[:, 0])), float(np.ptp(stack[:, 1])), 1e-6)
    return float(np.mean(np.linalg.norm(path_a - path_b, axis=1)) / scale)


def _trajectory_curvature(path):
    path = _align_path(path)
    if len(path) < 3:
        return 0.0
    deltas = np.diff(path, axis=0)
    angles = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
    return float(np.sum(np.abs(np.diff(angles))))


def _real_pair_shape_distance(row_a, row_b, behaviour):
    arrays_a = row_a["arrays"][behaviour]
    arrays_b = row_b["arrays"][behaviour]
    return (
        _trajectory_shape_distance(arrays_a["source_true"], arrays_b["source_true"])
        + _trajectory_shape_distance(arrays_a["target_true"], arrays_b["target_true"])
    )


def resample_path(path, n_points=120):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"Expected path shape (T, 2), got {path.shape}")
    if len(path) == 0:
        return np.zeros((n_points, 2), dtype=float)
    if len(path) == 1:
        return np.repeat(path, n_points, axis=0)

    old_t = np.linspace(0.0, 1.0, len(path))
    new_t = np.linspace(0.0, 1.0, n_points)
    return np.stack([np.interp(new_t, old_t, path[:, dim]) for dim in range(path.shape[1])], axis=1)


def _angle_to_bin(angle, n_bins=8):
    wrapped = angle % (2 * np.pi)
    return int(np.floor((wrapped / (2 * np.pi)) * n_bins)) % n_bins


def build_families(per_pair, behaviour, n_bins=8):
    families = {}
    for row in per_pair:
        arrays = row["arrays"][behaviour]
        source_bin = _angle_to_bin(_trajectory_angle(arrays["source_true"]), n_bins=n_bins)
        target_bin = _angle_to_bin(_trajectory_angle(arrays["target_true"]), n_bins=n_bins)
        families.setdefault((source_bin, target_bin), []).append(row)
    return families


def summarize_families(families, behaviour):
    summary = []
    for (source_bin, target_bin), rows in families.items():
        cf_target = np.asarray(
            [row["behaviours"][behaviour]["counterfactual_decoded_vs_target_r2"] for row in rows],
            dtype=float,
        )
        target_real = np.asarray(
            [row["behaviours"][behaviour]["decoder_r2_on_target_real"] for row in rows],
            dtype=float,
        )
        delta = np.asarray(
            [row["behaviours"][behaviour]["target_minus_source"] for row in rows],
            dtype=float,
        )
        angle_gap = abs(target_bin - source_bin)
        angle_gap = min(angle_gap, 8 - angle_gap)
        summary.append(
            {
                "source_bin": int(source_bin),
                "target_bin": int(target_bin),
                "count": int(len(rows)),
                "mean_cf_to_target_r2": float(cf_target.mean()),
                "median_cf_to_target_r2": float(np.median(cf_target)),
                "mean_target_real_r2": float(target_real.mean()),
                "median_target_real_r2": float(np.median(target_real)),
                "mean_target_minus_source": float(delta.mean()),
                "median_target_minus_source": float(np.median(delta)),
                "angle_gap_bins": int(angle_gap),
            }
        )
    return summary


def choose_family(family_summary, min_pairs=10):
    eligible = [row for row in family_summary if row["count"] >= min_pairs and row["source_bin"] != row["target_bin"]]
    if not eligible:
        eligible = [row for row in family_summary if row["source_bin"] != row["target_bin"]] or list(family_summary)

    def family_score(row):
        return (
            0.015 * min(row["count"], 20)
            + 0.03 * row["angle_gap_bins"]
        )

    ranked = sorted(eligible, key=family_score, reverse=True)
    return ranked[0], ranked


def _normalize_group_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return "__nan__"
    return value


def _context_group_key(row):
    info = row["trial_info_source"]
    return tuple(_normalize_group_value(info.get(field)) for field in DISPLAY_CONTEXT_FIELDS)


def _parse_target_pos_string(value):
    nums = [int(x) for x in re.findall(r"-?\d+", str(value))]
    if len(nums) == 0 or len(nums) % 2 != 0:
        return None
    return np.asarray(nums, dtype=float).reshape(-1, 2)


def _active_target_xy_from_info(info):
    arr = _parse_target_pos_string(info.get("target_pos"))
    if arr is None:
        return None
    idx = int(info.get("active_target", 0)) if info.get("active_target", 0) is not None else 0
    idx = max(0, min(idx, len(arr) - 1))
    return tuple(int(v) for v in arr[idx].tolist())


def _position_pair_key(row):
    source_xy = _active_target_xy_from_info(row["trial_info_source"])
    target_xy = _active_target_xy_from_info(row["trial_info_target"])
    return source_xy, target_xy


def _display_condition_key(info):
    """Condition identity for repeated MC Maze trajectory exemplars."""
    fields = (
        "maze_id",
        "trial_version",
        "num_targets",
        "num_barriers",
        "active_target",
        "target_pos",
        "barrier_pos",
    )
    return tuple(_normalize_group_value(info.get(field)) for field in fields)


def _transformation_group_key(row):
    return (
        _display_condition_key(row["trial_info_source"]),
        _display_condition_key(row["trial_info_target"]),
    )


def _path_endpoint_angle(path):
    disp = _align_path(path)
    delta = disp[-1] - disp[0]
    if np.allclose(delta, 0):
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def _circular_std(angles):
    if len(angles) == 0:
        return float("inf")
    angles = np.asarray(angles, dtype=float)
    c = np.mean(np.cos(angles))
    s = np.mean(np.sin(angles))
    r = max(float(np.sqrt(c * c + s * s)), 1e-8)
    return float(np.sqrt(max(0.0, -2.0 * np.log(r))))


def _candidate_metrics(rows, behaviour, source_bin, target_bin, context=None):
    cf_vals = np.asarray(
        [row["behaviours"][behaviour]["counterfactual_decoded_vs_target_r2"] for row in rows],
        dtype=float,
    )
    target_vals = np.asarray(
        [row["behaviours"][behaviour]["decoder_r2_on_target_real"] for row in rows],
        dtype=float,
    )
    delta_vals = np.asarray(
        [row["behaviours"][behaviour]["target_minus_source"] for row in rows],
        dtype=float,
    )
    source_angles = [_path_endpoint_angle(row["arrays"][behaviour]["source_true"]) for row in rows]
    target_angles = [_path_endpoint_angle(row["arrays"][behaviour]["target_true"]) for row in rows]
    mean_std = float(
        np.mean(
            [
                _circular_std(source_angles),
                _circular_std(target_angles),
            ]
        )
    )
    angle_gap = abs(target_bin - source_bin)
    angle_gap = min(angle_gap, 8 - angle_gap)
    return {
        "source_bin": int(source_bin),
        "target_bin": int(target_bin),
        "count": int(len(rows)),
        "median_cf_to_target_r2": float(np.median(cf_vals)),
        "median_target_real_r2": float(np.median(target_vals)),
        "median_target_minus_source": float(np.median(delta_vals)),
        "angle_gap_bins": int(angle_gap),
        "mean_endpoint_angle_std": mean_std,
        "used_context_grouping": context is not None,
        "context": context,
    }


def _display_candidate_score(meta):
    count_score = 0.30 * min(meta["count"], 8)
    angle_gap_score = 0.10 * meta["angle_gap_bins"]
    tightness_penalty = 0.75 * meta["mean_endpoint_angle_std"]
    context_bonus = 0.30 if meta["used_context_grouping"] else 0.0
    position_bonus = 0.45 if meta.get("used_position_grouping") else 0.0
    return float(
        count_score
        + angle_gap_score
        + context_bonus
        + position_bonus
        - tightness_penalty
    )


def _matches_locked_display(meta, locked_display):
    if locked_display is None:
        return False
    if int(meta["source_bin"]) != int(locked_display["source_bin"]):
        return False
    if int(meta["target_bin"]) != int(locked_display["target_bin"]):
        return False
    locked_pair = locked_display.get("position_pair")
    if locked_pair is None:
        return True
    if not meta.get("used_position_grouping"):
        return False
    pos_pair = meta.get("position_pair") or {}
    return (
        list(pos_pair.get("source_xy", [])) == list(locked_pair.get("source_xy", []))
        and list(pos_pair.get("target_xy", [])) == list(locked_pair.get("target_xy", []))
    )


def choose_display_subset(
    families,
    behaviour,
    min_family_pairs=10,
    min_context_pairs=4,
    min_position_pairs=4,
    locked_display=None,
):
    candidates = []
    for (source_bin, target_bin), family_rows in families.items():
        if source_bin == target_bin or len(family_rows) < min_family_pairs:
            continue
        family_meta = _candidate_metrics(family_rows, behaviour, source_bin, target_bin, context=None)
        family_meta["used_position_grouping"] = False
        family_meta["position_pair"] = None
        candidates.append((family_rows, family_meta))

        groups = {}
        for row in family_rows:
            groups.setdefault(_context_group_key(row), []).append(row)
        for key, rows in groups.items():
            if len(rows) < min_context_pairs:
                continue
            context = {field: value for field, value in zip(DISPLAY_CONTEXT_FIELDS, key)}
            subgroup_meta = _candidate_metrics(rows, behaviour, source_bin, target_bin, context=context)
            subgroup_meta["context_group_count"] = int(len(groups))
            subgroup_meta["used_position_grouping"] = False
            subgroup_meta["position_pair"] = None
            candidates.append((rows, subgroup_meta))

        position_groups = {}
        for row in family_rows:
            pos_key = _position_pair_key(row)
            if pos_key[0] is None or pos_key[1] is None:
                continue
            position_groups.setdefault(pos_key, []).append(row)
        for pos_key, rows in position_groups.items():
            if len(rows) < min_position_pairs:
                continue
            pos_meta = _candidate_metrics(rows, behaviour, source_bin, target_bin, context=None)
            pos_meta["used_position_grouping"] = True
            pos_meta["position_pair"] = {
                "source_xy": list(pos_key[0]),
                "target_xy": list(pos_key[1]),
            }
            pos_meta["position_group_count"] = int(len(position_groups))
            candidates.append((rows, pos_meta))

    if not candidates:
        fallback_family, _ = choose_family(summarize_families(families, behaviour), min_pairs=3)
        fallback_rows = families[(fallback_family["source_bin"], fallback_family["target_bin"])]
        fallback_meta = _candidate_metrics(
            fallback_rows,
            behaviour,
            fallback_family["source_bin"],
            fallback_family["target_bin"],
            context=None,
        )
        fallback_meta["used_position_grouping"] = False
        fallback_meta["position_pair"] = None
        return fallback_rows, fallback_meta, [fallback_meta]

    position_candidates = [item for item in candidates if item[1].get("used_position_grouping")]
    if position_candidates:
        ranked = sorted(position_candidates, key=lambda item: _display_candidate_score(item[1]), reverse=True)
    else:
        ranked = sorted(candidates, key=lambda item: _display_candidate_score(item[1]), reverse=True)
    top_rows, top_meta = ranked[0]
    if locked_display is not None:
        locked_display_found = False
        for rows, meta in ranked:
            if _matches_locked_display(meta, locked_display):
                top_rows, top_meta = rows, meta
                locked_display_found = True
                break
        top_meta = dict(top_meta)
        top_meta["locked_display"] = locked_display
        top_meta["locked_display_found"] = locked_display_found
    ranked_meta = [meta for _, meta in ranked[:10]]
    return top_rows, top_meta, ranked_meta


def choose_secondary_display_subset(
    families,
    behaviour,
    primary_meta,
    min_position_pairs=4,
):
    primary_pair = primary_meta.get("position_pair") or {}
    strict_candidates = []
    relaxed_candidates = []
    for (source_bin, target_bin), family_rows in families.items():
        if source_bin == target_bin:
            continue
        position_groups = {}
        for row in family_rows:
            pos_key = _position_pair_key(row)
            if pos_key[0] is None or pos_key[1] is None:
                continue
            position_groups.setdefault(pos_key, []).append(row)
        for pos_key, rows in position_groups.items():
            if len(rows) < min_position_pairs:
                continue
            position_pair = {
                "source_xy": list(pos_key[0]),
                "target_xy": list(pos_key[1]),
            }
            if (
                position_pair["source_xy"] == list(primary_pair.get("source_xy", []))
                and position_pair["target_xy"] == list(primary_pair.get("target_xy", []))
            ):
                continue

            source_path = _mean_path(_family_paths(rows, behaviour, "source"))
            target_path = _mean_path(_family_paths(rows, behaviour, "target"))
            source_end = source_path[-1]
            target_end = target_path[-1]
            if source_end[0] <= 0 or source_end[1] <= 0:
                continue
            if target_end[0] >= 0 or target_end[1] >= 0:
                continue

            meta = _candidate_metrics(rows, behaviour, source_bin, target_bin, context=None)
            meta["used_position_grouping"] = True
            meta["position_pair"] = position_pair
            meta["position_group_count"] = int(len(position_groups))
            mostly_quadrant_clean = (
                np.percentile(source_path[:, 1], 20) >= -1.5
                and np.percentile(target_path[:, 1], 80) <= 1.5
            )
            vertical_separation = min(source_end[1], -target_end[1])
            horizontal_separation = min(source_end[0], -target_end[0])
            quadrant_fill = vertical_separation + horizontal_separation
            score = (
                0.03 * min(meta["count"], 8)
                + 0.055 * quadrant_fill
                - 0.40 * meta["mean_endpoint_angle_std"]
            )
            candidate = (score, rows, meta)
            if mostly_quadrant_clean:
                strict_candidates.append(candidate)
            else:
                relaxed_candidates.append(candidate)

    candidates = strict_candidates or relaxed_candidates
    if not candidates:
        return None, None
    _, rows, meta = max(candidates, key=lambda item: item[0])
    return rows, meta


def choose_trajectory_grid_rows(pair_rows, display_behaviour, n_examples=4, selection_mode="deterministic", seed=0):
    """Choose compact visual examples from coherent transformation groups.

    Each mini-panel should show repeated trials of one source condition mapped
    to one target condition. Diversity is encouraged across panels, not within
    the background traces of a panel. Selection uses metadata and real
    trajectories only, never counterfactual decoding scores.
    """
    groups = {}
    for row in pair_rows:
        pos_key = _position_pair_key(row)
        if pos_key[0] is None or pos_key[1] is None:
            continue
        groups.setdefault(_transformation_group_key(row), []).append(row)

    exact_groups = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        rep = min(
            rows,
            key=lambda row: np.mean([_real_pair_shape_distance(row, other, display_behaviour) for other in rows]),
        )
        arrays = rep["arrays"][display_behaviour]
        source_angle = _trajectory_angle(arrays["source_true"])
        target_angle = _trajectory_angle(arrays["target_true"])
        angle_gap = _angle_distance(source_angle, target_angle)
        source_xy, target_xy = _position_pair_key(rep)
        if source_xy == target_xy:
            continue
        exact_groups.append((rows, rep, source_angle, target_angle, angle_gap))

    min_angle_gap = np.deg2rad(TRAJECTORY_GRID_MIN_ANGLE_GAP_DEG)
    candidate_groups = [group for group in exact_groups if group[4] >= min_angle_gap]

    if len(candidate_groups) < n_examples:
        raise RuntimeError(
            "Could not find enough exact MC Maze transformation groups for trajectory grid: "
            f"found {len(candidate_groups)} with source-target direction gap >= "
            f"{TRAJECTORY_GRID_MIN_ANGLE_GAP_DEG:.0f} deg, need {n_examples}."
        )

    candidates = []
    for rows, rep, source_angle, target_angle, angle_gap in candidate_groups:
        arrays = rep["arrays"][display_behaviour]
        source_curvature = _trajectory_curvature(arrays["source_true"])
        target_curvature = _trajectory_curvature(arrays["target_true"])
        curvature_score = 0.5 * (
            min(source_curvature, 8.0) / 8.0
            + min(target_curvature, 8.0) / 8.0
        )
        source_bin = _angle_to_bin(source_angle)
        target_bin = _angle_to_bin(target_angle)
        candidates.append({
            "candidate_rank": len(candidates),
            "rows": rows,
            "row": rep,
            "source_angle": source_angle,
            "target_angle": target_angle,
            "angle_gap": angle_gap,
            "source_bin": source_bin,
            "target_bin": target_bin,
            "source_curvature": source_curvature,
            "target_curvature": target_curvature,
            "curvature_score": curvature_score,
            "count": len(rows),
        })

    if selection_mode not in {"deterministic", "seeded"}:
        raise ValueError(f"Unknown trajectory selection mode {selection_mode!r}")

    if selection_mode == "seeded":
        candidates = sorted(
            candidates,
            key=lambda item: (
                item["source_bin"],
                item["target_bin"],
                _position_pair_key(item["row"]),
                item["candidate_rank"],
            ),
        )
        rng = np.random.default_rng(seed)
        selected = []
        remaining = list(candidates)
        while len(selected) < n_examples and remaining:
            used_target_bins = {item["target_bin"] for item in selected}
            used_source_bins = {item["source_bin"] for item in selected}
            pool = [
                item
                for item in remaining
                if item["target_bin"] not in used_target_bins and item["source_bin"] not in used_source_bins
            ]
            if len(pool) < n_examples - len(selected):
                pool = [item for item in remaining if item["target_bin"] not in used_target_bins]
            if len(pool) < n_examples - len(selected):
                pool = remaining
            pick = pool[int(rng.integers(0, len(pool)))]
            remaining.remove(pick)
            selected.append(pick)
        return [item["row"] for item in selected[:n_examples]]

    curvature_values = np.asarray([item["curvature_score"] for item in candidates], dtype=float)
    curvature_targets = np.quantile(
        curvature_values,
        (np.arange(n_examples, dtype=float) + 0.5) / n_examples,
    )
    if n_examples == 4:
        curvature_targets = curvature_targets[[0, 3, 1, 2]]

    selected = []
    while len(selected) < n_examples and candidates:
        target_curvature = float(curvature_targets[len(selected)])
        used_target_bins = {
            _angle_to_bin(_trajectory_angle(row["arrays"][display_behaviour]["target_true"]))
            for row in selected
        }
        used_source_bins = {
            _angle_to_bin(_trajectory_angle(row["arrays"][display_behaviour]["source_true"]))
            for row in selected
        }

        def display_score(item):
            angle_gap_score = item["angle_gap"] / np.pi
            curvature_match = 1.0 - abs(item["curvature_score"] - target_curvature)
            count_score = min(item["count"], 8) / 8.0
            penalty = 0.0
            penalty += 1.10 if item["target_bin"] in used_target_bins else 0.0
            penalty += 0.35 if item["source_bin"] in used_source_bins else 0.0
            for row in selected:
                penalty += 0.70 if set(_position_pair_key(item["row"])) == set(_position_pair_key(row)) else 0.0
            if selected:
                target_div = min(
                    _angle_distance(
                        item["target_angle"],
                        _trajectory_angle(row["arrays"][display_behaviour]["target_true"]),
                    )
                    / np.pi
                    for row in selected
                )
                source_div = min(
                    _angle_distance(
                        item["source_angle"],
                        _trajectory_angle(row["arrays"][display_behaviour]["source_true"]),
                    )
                    / np.pi
                    for row in selected
                )
            else:
                target_div = 0.0
                source_div = 0.0
            return float(
                0.70 * angle_gap_score
                + 0.75 * curvature_match
                + 0.20 * count_score
                + 0.25 * target_div
                + 0.12 * source_div
                - penalty
            )

        best_item = max(candidates, key=lambda item: (display_score(item), -item["candidate_rank"]))
        candidates.remove(best_item)
        selected.append(best_item["row"])
    return selected[:n_examples]


def _display_background_family_rows(reference_row, candidate_rows, behaviour, max_rows=5):
    ref_transform = _transformation_group_key(reference_row)
    exact_transform = [
        row
        for row in candidate_rows
        if _transformation_group_key(row) == ref_transform
    ]
    if len(exact_transform) >= 2:
        def transform_distance(row):
            ref_arrays = reference_row["arrays"][behaviour]
            row_arrays = row["arrays"][behaviour]
            return (
                _trajectory_shape_distance(row_arrays["source_true"], ref_arrays["source_true"])
                + _trajectory_shape_distance(row_arrays["target_true"], ref_arrays["target_true"])
            )

        exact_transform = sorted(exact_transform, key=transform_distance)
        if reference_row not in exact_transform:
            exact_transform.insert(0, reference_row)
        return exact_transform[:max_rows]
    return [reference_row]


def summarize_transform_cv(
    train_dataset,
    per_pair,
    behaviours,
    input_window,
    output_window,
    spike_target_mode=None,
    behaviour_target_mode=None,
    decoder_type="cebra",
    hidden_dim=256,
    cv_folds=5,
    cv_repeats=5,
    decoder_kwargs=None,
):
    decoder_kwargs = decoder_kwargs or {}
    if decoder_type == "cebra" or cv_folds <= 1:
        rng = np.random.default_rng(decoder_kwargs.get("seed", 0))
        summary = {}
        for behaviour in behaviours:
            pair_arrays = [row["arrays"][behaviour] for row in per_pair]

            def pooled_metric(true_key, pred_key, rows):
                return pooled_r2([row[true_key] for row in rows], [row[pred_key] for row in rows])

            source_to_source = pooled_metric("source_true", "source_decoded", pair_arrays)
            source_to_target = pooled_metric("target_true", "source_decoded", pair_arrays)
            target_to_target = pooled_metric("target_true", "target_decoded", pair_arrays)
            cf_to_source = pooled_metric("source_true", "cf_decoded", pair_arrays)
            cf_to_target = pooled_metric("target_true", "cf_decoded", pair_arrays)

            source_to_source_boot = []
            source_to_target_boot = []
            target_to_target_boot = []
            cf_to_source_boot = []
            cf_to_target_boot = []
            n_bootstrap = decoder_kwargs.get("summary_bootstrap_samples", 500)
            for _ in range(n_bootstrap):
                indices = rng.integers(0, len(pair_arrays), size=len(pair_arrays))
                rows = [pair_arrays[idx] for idx in indices]
                source_to_source_boot.append(pooled_metric("source_true", "source_decoded", rows))
                source_to_target_boot.append(pooled_metric("target_true", "source_decoded", rows))
                target_to_target_boot.append(pooled_metric("target_true", "target_decoded", rows))
                cf_to_source_boot.append(pooled_metric("source_true", "cf_decoded", rows))
                cf_to_target_boot.append(pooled_metric("target_true", "cf_decoded", rows))
            cf_gap_closed_boot = _normalize_between_controls(
                np.asarray(cf_to_target_boot, dtype=float),
                np.asarray(source_to_target_boot, dtype=float),
                np.asarray(target_to_target_boot, dtype=float),
            )
            cf_gap_closed_boot = cf_gap_closed_boot[np.isfinite(cf_gap_closed_boot)]
            target_gain_boot = np.asarray(target_to_target_boot, dtype=float) - np.asarray(source_to_target_boot, dtype=float)
            cf_gain_boot = np.asarray(cf_to_target_boot, dtype=float) - np.asarray(source_to_target_boot, dtype=float)

            summary[behaviour] = {
                "source_to_source_r2": float(source_to_source),
                "source_to_source_ci95": _ci95(source_to_source_boot),
                "source_to_target_r2": float(source_to_target),
                "source_to_target_ci95": _ci95(source_to_target_boot),
                "target_to_target_r2": float(target_to_target),
                "target_to_target_ci95": _ci95(target_to_target_boot),
                "cf_to_source_r2": float(cf_to_source),
                "cf_to_source_ci95": _ci95(cf_to_source_boot),
                "cf_to_target_r2": float(cf_to_target),
                "cf_to_target_ci95": _ci95(cf_to_target_boot),
                "target_gain_to_target": float(target_to_target - source_to_target),
                "target_gain_to_target_ci95": _ci95(target_gain_boot),
                "cf_gain_to_target": float(cf_to_target - source_to_target),
                "cf_gain_to_target_ci95": _ci95(cf_gain_boot),
                "cf_gap_closed": float(
                    _normalize_between_controls(cf_to_target, source_to_target, target_to_target)
                ),
                "cf_gap_closed_ci95": _ci95(cf_gap_closed_boot),
                "summary_method": "bootstrap_over_pairs",
                "bootstrap_samples": int(n_bootstrap),
            }
        return summary
    train_spikes = train_dataset.spikes.data.squeeze(-1).numpy()
    train_behaviour = {
        behaviour: window_condition_target(
            train_dataset.data_loader.load_behaviour(behaviour),
            input_window=input_window,
            output_window=output_window,
            delay_factor=0,
            target_mode=(behaviour_target_mode or spike_target_mode or "future"),
        )
        for behaviour in behaviours
    }

    summary = {}
    for behaviour in behaviours:
        pair_arrays = [row["arrays"][behaviour] for row in per_pair]
        source_spikes = [row["spikes"]["source"] for row in per_pair]
        target_spikes = [row["spikes"]["target"] for row in per_pair]
        cf_spikes = [row["spikes"]["counterfactual"] for row in per_pair]

        source_to_source = []
        source_to_target = []
        target_to_target = []
        cf_to_source = []
        cf_to_target = []

        total_runs = 0
        for repeat_idx in range(cv_repeats):
            folds = _cv_folds(len(train_spikes), cv_folds, seed=repeat_idx)
            for fold_idx, _ in enumerate(folds):
                train_idx = np.concatenate([folds[i] for i in range(len(folds)) if i != fold_idx])
                decoder = fit_decoder(
                    train_spikes[train_idx],
                    train_behaviour[behaviour][train_idx],
                    decoder_type=decoder_type,
                    hidden_dim=hidden_dim,
                    decoder_kwargs=decoder_kwargs,
                )
                source_pred = [decoder_predict(decoder, x) for x in source_spikes]
                target_pred = [decoder_predict(decoder, x) for x in target_spikes]
                cf_pred = [decoder_predict(decoder, x) for x in cf_spikes]

                source_to_source.append(pooled_r2([row["source_true"] for row in pair_arrays], source_pred))
                source_to_target.append(pooled_r2([row["target_true"] for row in pair_arrays], source_pred))
                target_to_target.append(pooled_r2([row["target_true"] for row in pair_arrays], target_pred))
                cf_to_source.append(pooled_r2([row["source_true"] for row in pair_arrays], cf_pred))
                cf_to_target.append(pooled_r2([row["target_true"] for row in pair_arrays], cf_pred))
                total_runs += 1

        def pack(values):
            arr = np.asarray(values, dtype=float)
            ci95 = 1.96 * arr.std(ddof=1) if len(arr) > 1 else 0.0
            return float(arr.mean()), float(ci95)

        s2s, s2s_ci = pack(source_to_source)
        s2t, s2t_ci = pack(source_to_target)
        t2t, t2t_ci = pack(target_to_target)
        c2s, c2s_ci = pack(cf_to_source)
        c2t, c2t_ci = pack(cf_to_target)
        target_gain = np.asarray(target_to_target, dtype=float) - np.asarray(source_to_target, dtype=float)
        cf_gain = np.asarray(cf_to_target, dtype=float) - np.asarray(source_to_target, dtype=float)
        cf_gap_closed = _normalize_between_controls(
            np.asarray(cf_to_target, dtype=float),
            np.asarray(source_to_target, dtype=float),
            np.asarray(target_to_target, dtype=float),
        )
        cf_gap_closed = np.asarray(cf_gap_closed, dtype=float)
        cf_gap_closed_valid = cf_gap_closed[np.isfinite(cf_gap_closed)]
        gap_mean = float(np.nanmean(cf_gap_closed))
        gap_ci = float(1.96 * cf_gap_closed_valid.std(ddof=1)) if len(cf_gap_closed_valid) > 1 else 0.0
        target_gain_ci = float(1.96 * target_gain.std(ddof=1)) if len(target_gain) > 1 else 0.0
        cf_gain_ci = float(1.96 * cf_gain.std(ddof=1)) if len(cf_gain) > 1 else 0.0
        summary[behaviour] = {
            "source_to_source_r2": s2s,
            "source_to_source_ci95": s2s_ci,
            "source_to_target_r2": s2t,
            "source_to_target_ci95": s2t_ci,
            "target_to_target_r2": t2t,
            "target_to_target_ci95": t2t_ci,
            "cf_to_source_r2": c2s,
            "cf_to_source_ci95": c2s_ci,
            "cf_to_target_r2": c2t,
            "cf_to_target_ci95": c2t_ci,
            "target_gain_to_target": float(target_gain.mean()),
            "target_gain_to_target_ci95": target_gain_ci,
            "cf_gain_to_target": float(cf_gain.mean()),
            "cf_gain_to_target_ci95": cf_gain_ci,
            "cf_gap_closed": gap_mean,
            "cf_gap_closed_ci95": gap_ci,
            "cv_folds": int(cv_folds),
            "cv_repeats": int(cv_repeats),
            "cv_total_runs": int(total_runs),
        }
    return summary


def _align_path(path):
    return path - path[0:1]


def _mean_path(paths, n_points=120):
    return np.mean(np.stack([resample_path(path, n_points=n_points) for path in paths], axis=0), axis=0)


def _iter_family_groups(family_rows):
    if not family_rows:
        return []
    if isinstance(family_rows[0], list):
        return family_rows
    return [family_rows]


def _family_paths(family_rows, behaviour, kind):
    key_map = {
        "source": "source_true",
        "target": "target_true",
        "counterfactual": "cf_decoded",
    }
    key = key_map[kind]
    return [_align_path(row["arrays"][behaviour][key]) for row in family_rows]


def _shared_family_bounds(family_rows, behaviour):
    all_paths = []
    groups = _iter_family_groups(family_rows)
    for group_rows in groups:
        for kind in ("source", "target", "counterfactual"):
            all_paths.extend(_family_paths(group_rows, behaviour, kind))
            all_paths.append(_mean_path(_family_paths(group_rows, behaviour, kind)))

    stack = np.concatenate(all_paths, axis=0)
    x_low, y_low = np.nanmin(stack, axis=0)
    x_high, y_high = np.nanmax(stack, axis=0)
    if np.isclose(x_low, x_high):
        x_low -= 0.5
        x_high += 0.5
    if np.isclose(y_low, y_high):
        y_low -= 0.5
        y_high += 0.5
    return (float(x_low), float(x_high)), (float(y_low), float(y_high))


def _trajectory_artist_clearance_points():
    marker_radius = np.sqrt(32.0) / 2.0
    line_radius = 2.9 / 2.0
    return marker_radius + line_radius + 2.0


def _expand_bounds_by_artist_size(ax, xlim, ylim, clearance_points=None):
    clearance_points = _trajectory_artist_clearance_points() if clearance_points is None else clearance_points
    fig = ax.figure
    fig.canvas.draw()
    clearance_px = clearance_points * fig.dpi / 72.0
    inv = ax.transData.inverted()
    origin_px = ax.transData.transform((xlim[0], ylim[0]))
    pad_x = abs(inv.transform(origin_px + np.array([clearance_px, 0.0]))[0] - xlim[0])
    pad_y = abs(inv.transform(origin_px + np.array([0.0, clearance_px]))[1] - ylim[0])
    return (xlim[0] - pad_x, xlim[1] + pad_x), (ylim[0] - pad_y, ylim[1] + pad_y)


def plot_family_overlay_panel(ax, family_rows, behaviour, xlim, ylim, panel_mode="all_plus_mean"):
    style = {
        "source": {
            "mean_color": COUNTERFACTUAL_COLORS["source_decode"],
            "trace_color": COUNTERFACTUAL_COLORS["source_light"],
            "linestyle": "-",
            "label": "Source",
        },
        "target": {
            "mean_color": COUNTERFACTUAL_COLORS["target_true"],
            "trace_color": COUNTERFACTUAL_COLORS["target_light"],
            "linestyle": "-",
            "label": "Target",
        },
        "counterfactual": {
            "mean_color": COUNTERFACTUAL_COLORS["counterfactual"],
            "trace_color": COUNTERFACTUAL_COLORS["counterfactual_light"],
            "linestyle": "--",
            "label": "Counterfactual",
        },
    }
    for group_rows in _iter_family_groups(family_rows):
        for kind in ("source", "target", "counterfactual"):
            paths = _family_paths(group_rows, behaviour, kind)
            if panel_mode == "all_plus_mean":
                for path in paths:
                    ax.plot(
                        path[:, 0],
                        path[:, 1],
                        color=style[kind]["trace_color"],
                        linestyle=style[kind]["linestyle"],
                        linewidth=0.9,
                        alpha=0.12 if kind != "counterfactual" else 0.10,
                        zorder=1,
                    )
            mean_path = _mean_path(paths)
            ax.plot(
                mean_path[:, 0],
                mean_path[:, 1],
                color=style[kind]["mean_color"],
                linestyle=style[kind]["linestyle"],
                linewidth=2.8 if kind != "counterfactual" else 2.9,
                zorder=3,
                label=style[kind]["label"],
            )
    ax.scatter([0.0], [0.0], color="#555555", s=24, zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _plot_compact_mc_summary_plain(
    fig,
    subplot_spec,
    behaviour_labels,
    source_means,
    source_err,
    cf_means,
    cf_err,
    target_means,
    baseline_means=None,
    baseline_err=None,
    neds_means=None,
    neds_err=None,
    ablation_means=None,
    ablation_err=None,
):
    ax = fig.add_subplot(subplot_spec)
    x = np.arange(len(behaviour_labels))
    has_baseline = baseline_means is not None
    has_neds = neds_means is not None
    has_ablation = ablation_means is not None
    n_bars = 2 + int(has_baseline) + int(has_neds) + int(has_ablation)
    width = 0.15 if n_bars >= 5 else 0.18 if n_bars == 4 else 0.24 if n_bars == 3 else 0.32
    offsets = np.linspace(-0.5 * (n_bars - 1) * width, 0.5 * (n_bars - 1) * width, n_bars)
    offset_idx = 0
    source_x = x + offsets[offset_idx]
    offset_idx += 1
    baseline_x = x + offsets[offset_idx] if has_baseline else None
    offset_idx += int(has_baseline)
    neds_x = x + offsets[offset_idx] if has_neds else None
    offset_idx += int(has_neds)
    ablation_x = x + offsets[offset_idx] if has_ablation else None
    offset_idx += int(has_ablation)
    cf_x = x + offsets[offset_idx]

    source_values, source_plot_err, source_clipped = _clip_summary_bars(source_means, source_err)
    cf_values, cf_plot_err, cf_clipped = _clip_summary_bars(cf_means, cf_err)
    if has_baseline:
        baseline_values, baseline_plot_err, baseline_clipped = _clip_summary_bars(baseline_means, baseline_err)
    else:
        baseline_values = baseline_plot_err = baseline_clipped = None
    if has_neds:
        neds_values, neds_plot_err, neds_clipped = _clip_summary_bars(neds_means, neds_err)
    else:
        neds_values = neds_plot_err = neds_clipped = None
    if has_ablation:
        ablation_values, ablation_plot_err, ablation_clipped = _clip_summary_bars(ablation_means, ablation_err)
    else:
        ablation_values = ablation_plot_err = ablation_clipped = None

    ax.bar(
        source_x,
        source_values,
        width,
        yerr=source_plot_err,
        color=COUNTERFACTUAL_SUMMARY_COLORS["negative_control"],
        alpha=0.96,
        capsize=3,
    )
    if has_baseline:
        ax.bar(
            baseline_x,
            baseline_values,
            width,
            yerr=baseline_plot_err,
            color=COUNTERFACTUAL_SUMMARY_COLORS["baseline"],
            alpha=0.96,
            capsize=3,
        )
    if has_neds:
        ax.bar(
            neds_x,
            neds_values,
            width,
            yerr=neds_plot_err,
            color=COUNTERFACTUAL_SUMMARY_COLORS["neds_baseline"],
            alpha=0.96,
            capsize=3,
        )
    if has_ablation:
        ax.bar(
            ablation_x,
            ablation_values,
            width,
            yerr=ablation_plot_err,
            color=COUNTERFACTUAL_SUMMARY_COLORS["ablation"],
            alpha=0.96,
            capsize=3,
        )
    ax.bar(
        cf_x,
        cf_values,
        width,
        yerr=cf_plot_err,
        color=COUNTERFACTUAL_SUMMARY_COLORS["main_effect"],
        alpha=0.96,
        capsize=3,
    )
    _annotate_clipped_summary_bars(ax, source_x, source_means, source_values, source_clipped, source_err)
    if has_baseline:
        _annotate_clipped_summary_bars(
            ax, baseline_x, baseline_means, baseline_values, baseline_clipped, baseline_err
        )
    if has_neds:
        _annotate_clipped_summary_bars(ax, neds_x, neds_means, neds_values, neds_clipped, neds_err)
    if has_ablation:
        _annotate_clipped_summary_bars(
            ax, ablation_x, ablation_means, ablation_values, ablation_clipped, ablation_err
        )
    _annotate_clipped_summary_bars(ax, cf_x, cf_means, cf_values, cf_clipped, cf_err)
    for idx, ceiling in enumerate(target_means):
        left = x[idx] + offsets[0] - 0.5 * width
        right = x[idx] + offsets[-1] + 0.5 * width
        ax.hlines(
            ceiling,
            xmin=left,
            xmax=right,
            color=COUNTERFACTUAL_SUMMARY_COLORS["positive_control"],
            linestyle=":",
            linewidth=1.8,
        )

    all_vals = []
    any_clipped = bool(np.any(source_clipped) or np.any(cf_clipped))
    for mean, err in list(zip(source_values, source_plot_err)) + list(zip(cf_values, cf_plot_err)):
        all_vals.extend([mean - err, mean + err])
    if has_baseline:
        any_clipped = any_clipped or bool(np.any(baseline_clipped))
        baseline_plot_err = baseline_plot_err if baseline_plot_err is not None else [0.0 for _ in baseline_values]
        for mean, err in zip(baseline_values, baseline_plot_err):
            all_vals.extend([mean - err, mean + err])
    if has_neds:
        any_clipped = any_clipped or bool(np.any(neds_clipped))
        neds_plot_err = neds_plot_err if neds_plot_err is not None else [0.0 for _ in neds_values]
        for mean, err in zip(neds_values, neds_plot_err):
            all_vals.extend([mean - err, mean + err])
    if has_ablation:
        any_clipped = any_clipped or bool(np.any(ablation_clipped))
        ablation_plot_err = ablation_plot_err if ablation_plot_err is not None else [0.0 for _ in ablation_values]
        for mean, err in zip(ablation_values, ablation_plot_err):
            all_vals.extend([mean - err, mean + err])
    all_vals.extend(target_means)
    low = min(all_vals)
    high = max(all_vals)
    pad = max(0.08, 0.10 * max(high - low, 1e-6))
    if any_clipped:
        ax.set_ylim(SUMMARY_R2_CLIP_FLOOR - 0.34, high + pad)
    else:
        ax.set_ylim(low - pad, high + pad)

    ax.set_xticks(x)
    ax.set_xticklabels(behaviour_labels, rotation=0)
    ax.grid(axis="y", color=COUNTERFACTUAL_COLORS["grid"], linewidth=0.8, alpha=0.8)
    ax.axhline(0.0, color=COUNTERFACTUAL_COLORS["axis"], linewidth=1.0, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def _trajectory_legend_handles():
    return [
        Line2D([0], [0], color=COUNTERFACTUAL_COLORS["source_decode"], linewidth=2.6, label="Source"),
        Line2D([0], [0], color=COUNTERFACTUAL_COLORS["target_true"], linewidth=2.6, label="Target"),
        Line2D([0], [0], color=COUNTERFACTUAL_COLORS["counterfactual"], linewidth=2.6, linestyle="--", label="Neural-DISCO"),
    ]


def _summary_legend_handles(include_pivae=False, include_neds=False, include_ablation=False):
    handles = [
        Line2D(
            [0],
            [0],
            color=COUNTERFACTUAL_SUMMARY_COLORS["positive_control"],
            linestyle=":",
            linewidth=1.8,
            label="Target-spike decode",
        ),
        Patch(facecolor=COUNTERFACTUAL_SUMMARY_COLORS["negative_control"], edgecolor="none", label="Source spikes"),
    ]
    if include_pivae:
        handles.append(
            Patch(facecolor=COUNTERFACTUAL_SUMMARY_COLORS["baseline"], edgecolor="none", label="pi-VAE")
        )
    if include_neds:
        handles.append(
            Patch(facecolor=COUNTERFACTUAL_SUMMARY_COLORS["neds_baseline"], edgecolor="none", label="NEDS")
        )
    if include_ablation:
        handles.append(Patch(facecolor=COUNTERFACTUAL_SUMMARY_COLORS["ablation"], edgecolor="none", label="w/o reg."))
    handles.append(
        Patch(facecolor=COUNTERFACTUAL_SUMMARY_COLORS["main_effect"], edgecolor="none", label="Neural-DISCO")
    )
    return handles


def _summary_inputs(cfg, summary, pivae_summary=None, neds_summary=None, ablation_summary=None):
    label_map = {
        "hand_pos": "Hand position",
        "hand_vel": "Hand velocity",
        "eye_pos": "Eye position",
        "cursor_pos": "Cursor position",
    }
    behaviour_labels = [label_map.get(b, b) for b in cfg.behaviours]
    source_means = [summary[b]["source_to_target_r2"] for b in cfg.behaviours]
    source_err = [summary[b]["source_to_target_r2_sem"] for b in cfg.behaviours]
    cf_means = [summary[b]["cf_to_target_r2"] for b in cfg.behaviours]
    cf_err = [summary[b]["cf_to_target_r2_sem"] for b in cfg.behaviours]
    target_means = [summary[b]["target_to_target_r2"] for b in cfg.behaviours]

    pivae_means = None
    pivae_err = None
    if pivae_summary is not None:
        pivae_means = []
        pivae_err = []
        for behaviour in cfg.behaviours:
            value = pivae_summary[behaviour]["pivae_to_target_r2"]
            if isinstance(value, dict):
                pivae_means.append(float(value["mean"]))
                pivae_err.append(float(value.get("sem", 0.0)))
            else:
                pivae_means.append(float(value))
                pivae_err.append(float(pivae_summary[behaviour].get("pivae_to_target_r2_sem", 0.0)))

    neds_means = None
    neds_err = None
    if neds_summary is not None:
        neds_means = []
        neds_err = []
        for behaviour in cfg.behaviours:
            value = neds_summary[behaviour]["neds_to_target_r2"]
            if isinstance(value, dict):
                neds_means.append(float(value["mean"]))
                neds_err.append(float(value.get("sem", 0.0)))
            else:
                neds_means.append(float(value))
                neds_err.append(float(neds_summary[behaviour].get("neds_to_target_r2_sem", 0.0)))

    ablation_means = None
    ablation_err = None
    if ablation_summary is not None:
        ablation_means = [float(ablation_summary[b]["cf_to_target_r2"]) for b in cfg.behaviours]
        ablation_err = [float(ablation_summary[b].get("cf_to_target_r2_sem", 0.0)) for b in cfg.behaviours]

    return {
        "behaviour_labels": behaviour_labels,
        "source_means": source_means,
        "source_err": source_err,
        "cf_means": cf_means,
        "cf_err": cf_err,
        "target_means": target_means,
        "pivae_means": pivae_means,
        "pivae_err": pivae_err,
        "neds_means": neds_means,
        "neds_err": neds_err,
        "ablation_means": ablation_means,
        "ablation_err": ablation_err,
    }


def plot_pair_with_background_panel(ax, row, background_rows, behaviour, xlim, ylim):
    style = {
        "source": {
            "main_color": COUNTERFACTUAL_COLORS["source_decode"],
            "trace_color": COUNTERFACTUAL_COLORS["source_light"],
            "key": "source_true",
            "linestyle": "-",
        },
        "target": {
            "main_color": COUNTERFACTUAL_COLORS["target_true"],
            "trace_color": COUNTERFACTUAL_COLORS["target_light"],
            "key": "target_true",
            "linestyle": "-",
        },
        "counterfactual": {
            "main_color": COUNTERFACTUAL_COLORS["counterfactual"],
            "trace_color": COUNTERFACTUAL_COLORS["counterfactual_light"],
            "key": "cf_decoded",
            "linestyle": "--",
        },
    }
    for kind, params in style.items():
        for bg_row in background_rows:
            path = _align_path(bg_row["arrays"][behaviour][params["key"]])
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=params["trace_color"],
                linestyle=params["linestyle"],
                linewidth=0.85,
                alpha=0.18 if kind != "counterfactual" else 0.14,
                zorder=1,
            )
    for kind, params in style.items():
        path = _align_path(row["arrays"][behaviour][params["key"]])
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=params["main_color"],
            linestyle=params["linestyle"],
            linewidth=2.7 if kind != "counterfactual" else 2.9,
            zorder=3,
        )
        ax.scatter([path[-1, 0]], [path[-1, 1]], color=params["main_color"], s=32, zorder=4)
    ax.scatter([0.0], [0.0], color="#555555", s=24, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _draw_physical_space_axes(ax, occupied_paths=None):
    length = 0.095
    candidates = [
        (0.055, 0.070),
        (0.785, 0.070),
        (0.055, 0.790),
        (0.785, 0.790),
    ]
    occupied = []
    if occupied_paths:
        to_axes = ax.transAxes.inverted().transform
        for path in occupied_paths:
            path = np.asarray(path, dtype=float)
            if path.ndim == 2 and path.shape[1] == 2:
                occupied.append(to_axes(ax.transData.transform(path)))
    if occupied:
        occupied = np.concatenate(occupied, axis=0)

        def occupancy_score(candidate):
            x0, y0 = candidate
            x1 = x0 + length + 0.10
            y1 = y0 + length + 0.10
            inside = (
                (occupied[:, 0] >= x0 - 0.035)
                & (occupied[:, 0] <= x1)
                & (occupied[:, 1] >= y0 - 0.035)
                & (occupied[:, 1] <= y1)
            )
            center = np.asarray([x0 + 0.5 * length, y0 + 0.5 * length])
            min_dist = float(np.min(np.linalg.norm(occupied - center, axis=1)))
            return int(np.sum(inside)), -min_dist

        origin = min(candidates, key=occupancy_score)
    else:
        origin = candidates[0]
    arrowprops = dict(
        arrowstyle="-|>",
        lw=0.9,
        color="#444444",
        mutation_scale=7.0,
        shrinkA=0.0,
        shrinkB=0.0,
    )
    ax.annotate(
        "",
        xy=(origin[0] + length, origin[1]),
        xytext=origin,
        xycoords="axes fraction",
        arrowprops=arrowprops,
        annotation_clip=False,
        zorder=10,
    )
    ax.annotate(
        "",
        xy=(origin[0], origin[1] + length),
        xytext=origin,
        xycoords="axes fraction",
        arrowprops=arrowprops,
        annotation_clip=False,
        zorder=10,
    )
    ax.text(
        origin[0] + length + 0.018,
        origin[1] - 0.006,
        "x",
        transform=ax.transAxes,
        fontsize=7.0,
        color="#444444",
        ha="left",
        va="center",
    )
    ax.text(
        origin[0] - 0.006,
        origin[1] + length + 0.02,
        "y",
        transform=ax.transAxes,
        fontsize=7.0,
        color="#444444",
        ha="center",
        va="bottom",
    )


def plot_trajectory_grid(
    cfg,
    pair_rows,
    out_path,
    panel_mode="all_plus_mean",
    background_rows=None,
    selection_mode="deterministic",
    selection_seed=0,
):
    display_behaviour = "hand_pos" if "hand_pos" in cfg.behaviours else cfg.behaviours[0]
    rows = pair_rows[:4]
    if len(rows) < 4:
        raise RuntimeError(f"Trajectory grid requires 4 rows, got {len(rows)}")
    background_groups = [
        _display_background_family_rows(row, background_rows or rows, display_behaviour)
        for row in rows
    ]
    xlim, ylim = _shared_family_bounds(background_groups, display_behaviour)
    fig = plt.figure(figsize=(5.2, 4.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.16, 1.0, 1.0], hspace=0.08, wspace=0.04)

    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    ax_header.legend(
        handles=_trajectory_legend_handles(),
        frameon=False,
        fontsize=8.8,
        ncol=3,
        loc="center",
        handlelength=1.8,
        columnspacing=1.1,
    )

    axes = []
    for idx, row in enumerate(rows):
        ax = fig.add_subplot(gs[1 + idx // 2, idx % 2])
        plot_pair_with_background_panel(
            ax,
            row,
            background_groups[idx],
            display_behaviour,
            xlim=xlim,
            ylim=ylim,
        )
        axes.append(ax)

    if axes:
        xlim, ylim = _expand_bounds_by_artist_size(axes[0], xlim, ylim)
        for ax in axes:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        axis_paths = []
        for bg_row in background_groups[0]:
            arrays = bg_row["arrays"][display_behaviour]
            axis_paths.extend([arrays["source_true"], arrays["target_true"], arrays["cf_decoded"]])
        _draw_physical_space_axes(axes[0], axis_paths)

    fig.subplots_adjust(top=0.965, bottom=0.025, left=0.025, right=0.985)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    if selection_mode == "seeded":
        selection_description = "seeded sample from eligible transformations"
    else:
        selection_description = "deterministic curvature-quantile selection"
    selected_transformations = []
    for row in rows:
        arrays = row["arrays"][display_behaviour]
        source_angle = _trajectory_angle(arrays["source_true"])
        target_angle = _trajectory_angle(arrays["target_true"])
        source_curvature = _trajectory_curvature(arrays["source_true"])
        target_curvature = _trajectory_curvature(arrays["target_true"])
        source_xy, target_xy = _position_pair_key(row)
        selected_transformations.append(
            {
                "source_bin": int(_angle_to_bin(source_angle)),
                "target_bin": int(_angle_to_bin(target_angle)),
                "source_target_angle_gap_deg": float(np.degrees(_angle_distance(source_angle, target_angle))),
                "source_curvature": float(source_curvature),
                "target_curvature": float(target_curvature),
                "source_xy": list(source_xy) if source_xy is not None else None,
                "target_xy": list(target_xy) if target_xy is not None else None,
            }
        )
    return {
        "display_behaviour": display_behaviour,
        "num_examples": int(len(rows)),
        "selection_rule": (
            "exact repeated source->target condition groups; "
            f"source-target reach direction gap >= {TRAJECTORY_GRID_MIN_ANGLE_GAP_DEG:.0f} deg; "
            + selection_description
        ),
        "selection_mode": selection_mode,
        "selection_seed": int(selection_seed),
        "selected_transformations": selected_transformations,
    }


def plot_summary_figure(cfg, summary, out_path, pivae_summary=None, neds_summary=None, ablation_summary=None):
    values = _summary_inputs(
        cfg,
        summary,
        pivae_summary=pivae_summary,
        neds_summary=neds_summary,
        ablation_summary=ablation_summary,
    )
    fig = plt.figure(figsize=(5.4, 2.55))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.25, 1.0], hspace=0.02)
    ax_header = fig.add_subplot(gs[0, 0])
    ax_header.axis("off")
    ax_header.legend(
        handles=_summary_legend_handles(
            include_pivae=pivae_summary is not None,
            include_neds=neds_summary is not None,
            include_ablation=ablation_summary is not None,
        ),
        frameon=False,
        fontsize=8.0,
        ncol=3,
        loc="center",
        handlelength=1.7,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    ax_summary = _plot_compact_mc_summary_plain(
        fig,
        gs[1, 0],
        values["behaviour_labels"],
        values["source_means"],
        values["source_err"],
        values["cf_means"],
        values["cf_err"],
        values["target_means"],
        baseline_means=values["pivae_means"],
        baseline_err=values["pivae_err"],
        neds_means=values["neds_means"],
        neds_err=values["neds_err"],
        ablation_means=values["ablation_means"],
        ablation_err=values["ablation_err"],
    )
    ax_summary.set_ylabel(r"Decode $R^2$ to target")
    fig.subplots_adjust(top=0.98, bottom=0.18, left=0.13, right=0.995)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"behaviours": list(cfg.behaviours)}


def plot_figure(
    cfg,
    family_rows,
    family_meta,
    context_meta,
    summary,
    out_path,
    panel_mode="all_plus_mean",
    pivae_summary=None,
    neds_summary=None,
    ablation_summary=None,
):
    display_behaviour = "hand_pos" if "hand_pos" in cfg.behaviours else cfg.behaviours[0]

    fig = plt.figure(figsize=(5.15, 5.75))
    gs = fig.add_gridspec(
        4,
        1,
        height_ratios=[0.08, 1.70, 0.13, 0.68],
        hspace=0.02,
    )

    ax_header = fig.add_subplot(gs[0, 0])
    ax_header.axis("off")
    ax_header.legend(
        handles=_trajectory_legend_handles(),
        frameon=False,
        fontsize=8.6,
        ncol=3,
        loc="center",
        handlelength=1.7,
        columnspacing=1.0,
    )

    xlim, ylim = _shared_family_bounds(family_rows, display_behaviour)
    ax_top = fig.add_subplot(gs[1, 0])
    plot_family_overlay_panel(
        ax_top,
        family_rows,
        display_behaviour,
        xlim=xlim,
        ylim=ylim,
        panel_mode=panel_mode,
    )
    ax_top.set_title("")
    ax_top.text(-0.035, 0.98, "A", transform=ax_top.transAxes, fontsize=13.2, fontweight="bold")

    ax_summary_header = fig.add_subplot(gs[2, 0])
    ax_summary_header.axis("off")
    ax_summary_header.legend(
        handles=_summary_legend_handles(
            include_pivae=pivae_summary is not None,
            include_neds=neds_summary is not None,
            include_ablation=ablation_summary is not None,
        ),
        frameon=False,
        fontsize=7.8,
        ncol=3,
        loc="center",
        handlelength=1.8,
        columnspacing=0.9,
        borderaxespad=0.0,
    )

    values = _summary_inputs(
        cfg,
        summary,
        pivae_summary=pivae_summary,
        neds_summary=neds_summary,
        ablation_summary=ablation_summary,
    )
    ax_summary = _plot_compact_mc_summary_plain(
        fig,
        gs[3, 0],
        values["behaviour_labels"],
        values["source_means"],
        values["source_err"],
        values["cf_means"],
        values["cf_err"],
        values["target_means"],
        baseline_means=values["pivae_means"],
        baseline_err=values["pivae_err"],
        neds_means=values["neds_means"],
        neds_err=values["neds_err"],
        ablation_means=values["ablation_means"],
        ablation_err=values["ablation_err"],
    )
    ax_summary.set_ylabel(r"Decode $R^2$ to target")
    ax_summary.text(-0.09, 1.04, "B", transform=ax_summary.transAxes, fontsize=13.0, fontweight="bold")
    fig.subplots_adjust(top=0.975, bottom=0.105, left=0.145, right=0.99)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "display_behaviour": display_behaviour,
        "display_family": {
            "source_bin": int(family_meta["source_bin"]),
            "target_bin": int(family_meta["target_bin"]),
            "count": int(len(_iter_family_groups(family_rows)[0])) if _iter_family_groups(family_rows) else 0,
            "num_display_families": int(len(_iter_family_groups(family_rows))),
        },
        "display_context": context_meta,
    }
