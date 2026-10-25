import argparse
import glob
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
# Import this in scripts to access self-made packages (ie. src.utils)
current_dir = os.getcwd()
sys.path.append(current_dir)

import pandas as pd
from src.plotting.behaviour_style import behaviours_to_colours, get_disp_name
from src.utils import get_config_subpath, get_cv_subdirs, load_config, get_result_filepath, get_log_dir

plt.rcParams['font.size'] = 5
def extract_auc_by_behaviour(auc_csv_path):
    df = pd.read_csv(auc_csv_path, index_col=0)

    # Keep only numeric columns and drop accidental unnamed columns.
    numeric_cols = [
        c for c in df.columns if c and not str(c).lower().startswith("unnamed")
    ]
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    shared = [label for label in numeric_df.index if label in numeric_df.columns]
    if shared:
        diag_values = {
            behaviour: float(numeric_df.loc[behaviour, behaviour]) for behaviour in shared
        }
        if all(pd.notna(v) for v in diag_values.values()):
            return diag_values

    first_row = numeric_df.iloc[0]
    return {behaviour: float(value) for behaviour, value in first_row.items() if pd.notna(value)}


def load_auc_values_for_config(cfg, log_dir):
    """Load per-behaviour AUC values for a config, handling CV folds when present."""
    if cfg.get("cv_folds", 1) > 1:
        fold_values = {}
        for fold_dir in get_cv_subdirs(cfg, log_dir):
            auc_csv = get_result_filepath(fold_dir, "AUC_scores.csv")
            for behaviour, auc in extract_auc_by_behaviour(auc_csv).items():
                fold_values.setdefault(behaviour, []).append(float(auc))
        return fold_values
    auc_csv = get_result_filepath(log_dir, "AUC_scores.csv")
    return {behaviour: [float(auc)] for behaviour, auc in extract_auc_by_behaviour(auc_csv).items()}


def gather_results(config_dir):
    config_paths = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    rows = []
    for config_path in config_paths:
        cfg = load_config(filepath=config_path)

        l2_val = cfg.get("l2_unknown")

        log_dir = get_log_dir(cfg, config_path)
        auc_values = load_auc_values_for_config(cfg, log_dir)
        for behaviour, auc_list in auc_values.items():
            # Skip orientation_corr column
            if behaviour == "orientation_corr":
                continue

            auc_array = pd.Series(auc_list, dtype=float)
            rows.append(
                {
                    "config": config_path,
                    "l2_unknown": float(l2_val),
                    "behaviour": behaviour,
                    "auc_mean": float(auc_array.mean()),
                    "auc_stderr": float(auc_array.std(ddof=1) / (len(auc_array) ** 0.5)) if len(auc_array) > 1 else 0.0,
                    "n_folds": int(len(auc_array)),
                }
            )

    results_df = pd.DataFrame(rows)
    return results_df


def plot_results(results_df, output_png):
    plt.figure(figsize=(1.8,2))

    colors = behaviours_to_colours(results_df["behaviour"].unique())
    behaviours = list(results_df["behaviour"].unique())
    styles = ["-", "--", ":", "-."]
    style_map = {b: styles[i % len(styles)] for i, b in enumerate(behaviours)}

    for (behaviour, group), color in zip(results_df.groupby("behaviour"), colors):
        group = group.sort_values("l2_unknown")
        linestyle = style_map[behaviour]

        # Plot main line
        plt.plot(
            group["l2_unknown"],
            group["auc_mean"],
            label=get_disp_name(behaviour),
            color=color,
            linewidth=0.8,
            linestyle=linestyle,
        )

        # Plot shaded band
        plt.fill_between(
            group["l2_unknown"],
            group["auc_mean"] - group["auc_stderr"],
            group["auc_mean"] + group["auc_stderr"],
            alpha=0.2,
            color=color,
        )

    lambdas = sorted(results_df["l2_unknown"].unique())
    positive = [lam for lam in lambdas if lam > 0]
    plt.xscale("symlog", linthresh=min(positive, default=1e-5))
    plt.xticks(lambdas)
    plt.gca().xaxis.set_minor_locator(NullLocator())
    plt.xlabel(r"$\lambda$")
    plt.ylabel("AUC")
    plt.title(r"AUC vs $\lambda$")
    plt.ylim(bottom=0.5)

    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_png, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot AUC vs l2_unknown for configs in an ablation directory."
    )
    parser.add_argument(
        "--config-dir",
        default="configs/mc_maze/l2_attr_cv",
        help="Directory containing YAML configs.",
    )
    args = parser.parse_args()
    output_stem = f"logs/{get_config_subpath(args.config_dir)}_auc_vs_l2"
    args.output_png = f"{output_stem}.png"
    args.output_csv = f"{output_stem}.csv"

    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    results_df = gather_results(args.config_dir)

    if results_df.empty:
        raise RuntimeError(
            "No AUC results found. Ensure each config has a matching logs/<subdirs>-<stem>/fold_k/results/AUC_scores.csv"
        )

    results_df = results_df.sort_values(["behaviour", "l2_unknown"]).reset_index(drop=True)
    results_df.to_csv(args.output_csv, index=False)
    plot_results(results_df, args.output_png)

    print(f"Saved plot: {args.output_png}")
    print(f"Saved table: {args.output_csv}")

if __name__ == "__main__":
    main()
