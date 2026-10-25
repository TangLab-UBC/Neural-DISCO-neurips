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

from src.utils import get_cv_subdirs, load_config, get_result_filepath, get_log_dir
from src.plotting.behaviour_style import get_behaviour_colour
plt.rcParams['font.size'] = 5

def extract_orientation_corr(csv_path):
    """Extract the single orientation_corr value from a CSV file."""
    df = pd.read_csv(csv_path, index_col=0)
    if "orientation_corr" in df.columns:
        val = df["orientation_corr"].iloc[0]
        return float(val) if pd.notna(val) else None
    return None


def load_corr_values_for_config(cfg, log_dir):
    """Load orientation_corr values for a config, handling CV folds when present."""
    fold_values = []
    if cfg.get("cv_folds", 1) > 1:
        for fold_dir in get_cv_subdirs(cfg, log_dir):
            corr_csv = get_result_filepath(fold_dir, "AUC_scores.csv")
            corr = extract_orientation_corr(corr_csv)
            if corr is not None:
                fold_values.append(corr)
    else:
        corr_csv = get_result_filepath(log_dir, "AUC_scores.csv")
        corr = extract_orientation_corr(corr_csv)
        if corr is not None:
            fold_values.append(corr)

    return fold_values


def gather_results(config_dir):
    config_paths = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    rows = []
    for config_path in config_paths:
        cfg = load_config(filepath=config_path)

        l2_val = cfg.get("l2_unknown")
        log_dir = get_log_dir(cfg, config_path)
        corr_values = load_corr_values_for_config(cfg, log_dir)

        if corr_values:
            corr_array = pd.Series(corr_values, dtype=float)
            rows.append(
                {
                    "l2_unknown": float(l2_val),
                    "corr_mean": float(corr_array.mean()),
                    "corr_stderr": float(corr_array.std(ddof=1) / (len(corr_array) ** 0.5)) if len(corr_array) > 1 else 0.0,
                    "n_folds": int(len(corr_array)),
                }
            )

    results_df = pd.DataFrame(rows)
    return results_df


def plot_results(results_df, output_png):
    plt.figure(figsize=(1.8,2))


    results_df = results_df.sort_values("l2_unknown")
    color = get_behaviour_colour('orientation_corr')

    # Plot main line
    plt.plot(
        results_df["l2_unknown"],
        results_df["corr_mean"],
        color=color,
        label='Orientation',
        linewidth=2,
    )

    # Plot shaded band
    plt.fill_between(
        results_df["l2_unknown"],
        results_df["corr_mean"] - results_df["corr_stderr"],
        results_df["corr_mean"] + results_df["corr_stderr"],
        alpha=0.2,
        color=color,
    )

    lambdas = sorted(results_df["l2_unknown"].unique())
    positive = [lam for lam in lambdas if lam > 0]
    plt.xscale("symlog", linthresh=min(positive, default=1e-5))
    plt.xticks(lambdas)
    plt.gca().xaxis.set_minor_locator(NullLocator())
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Pearson r")
    plt.title(r"Orientation Selectivity Index vs $\lambda$")

    plt.legend(loc="best", framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_png, dpi=300)
    plt.close()


FIXED_ARGS = {
    "output_png": "logs/allen_vc_l2_ablation_corr_vs_l2.png",
    "output_csv": "logs/allen_vc_l2_ablation_corr_vs_l2.csv",
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot orientation_corr vs l2_unknown for configs in an ablation directory."
    )
    parser.add_argument(
        "--config-dir",
        default="configs/allen_vc/l2_ablation_attr_cv",
        help="Directory containing YAML configs.",
    )
    args = parser.parse_args()
    for key, value in FIXED_ARGS.items():
        setattr(args, key, value)

    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    results_df = gather_results(args.config_dir)

    if results_df.empty:
        raise RuntimeError(
            "No correlation results found. Ensure each config has a matching logs/<subdirs>-<stem>/fold_k/results/AUC_scores.csv with orientation_corr column"
        )

    results_df = results_df.sort_values("l2_unknown").reset_index(drop=True)
    results_df.to_csv(args.output_csv, index=False)
    plot_results(results_df, args.output_png)

    print(f"Saved plot: {args.output_png}")
    print(f"Saved table: {args.output_csv}")

if __name__ == "__main__":
    main()
