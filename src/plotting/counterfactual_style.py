COUNTERFACTUAL_COLORS = {
    "source_true": "#4c78a8",
    "source_decode": "#4c78a8",
    "target_true": "#111111",
    "target_decode": "#222222",
    "counterfactual": "#f58518",
    "ablation": "#8ac926",
    "source_to_target": "#ff595e",

    "pi-vae": "green",
    "neds": "pink",


    "counterfactual_light": "#f3c6a6",
    "source_light": "#bfd3f3",
    "target_light": "#b6b6b6",
    "grid": "#d9d9d9",
    "axis": "#555555",
}

COUNTERFACTUAL_SUMMARY_COLORS = {
    "negative_control": COUNTERFACTUAL_COLORS["source_decode"],
    "baseline": "#2ca02c",
    "neds_baseline": "#cc79a7",
    "ablation": "#9a9a9a",
    "main_effect": COUNTERFACTUAL_COLORS["counterfactual"],
    "positive_control": COUNTERFACTUAL_COLORS["target_decode"],
}
