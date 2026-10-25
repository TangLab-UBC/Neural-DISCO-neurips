import os
import sys

# Import this in scripts to access self-made packages (ie. src.utils)
current_dir = os.getcwd()
sys.path.append(current_dir)

import torch
import numpy as np
from src.utils import (
    get_log_dir,
    load_config,
    get_custom_filepath,
    get_result_filepath,
    load_datamodule_from_config,
    get_visualization_filepath,
    get_cv_subdirs
)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from src.plotting.behaviour_style import get_behaviour_colour

def plot_confusion_matrix(log_dir, loaded_attributions, y_true, auc, attribute):

    fpr, tpr, thresholds = roc_curve(y_true, loaded_attributions)

    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    y_pred = (loaded_attributions >= best_threshold).int()

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues')

    plt.title(f'Confusion Matrix AUC: {round(auc, 3)}')

    save_path = get_custom_filepath(log_dir,
                                    file_name=attribute,
                                    folder_name_1="roc_confusion_matrix",
                                    )
    plt.savefig(save_path)
    plt.close('all')

def auc_custom(log_dir, num_neurons):

    load_path = get_result_filepath(log_dir, 'attribution_scores.pt')
    loaded_attributions = torch.load(load_path, map_location=torch.device('cpu'))

    auc_scores = {}
    # position
    y_true = np.zeros(len(loaded_attributions["pos"]))
    y_true[0:num_neurons] = True
    auc = roc_auc_score(y_true, loaded_attributions["pos"])
    auc_scores["Position"] = round(auc, 3)
    plot_confusion_matrix(log_dir, loaded_attributions["pos"], y_true, auc, "Position")

    # head direction
    y_true = np.zeros(len(loaded_attributions["head_direction"]))
    y_true[num_neurons: num_neurons*2] = True
    auc = roc_auc_score(y_true, loaded_attributions["head_direction"])
    auc_scores["Head Direction"] = round(auc, 3)
    plot_confusion_matrix(log_dir, loaded_attributions["head_direction"], y_true, auc, "Head_Direction")

    # speed
    speed_attr_label = 'speed'
    if 'speed' not in loaded_attributions.keys():
      speed_attr_label = 'unknown'

    y_true = np.zeros(len(loaded_attributions[speed_attr_label]))
    y_true[num_neurons*2: num_neurons*3] = True
    auc = roc_auc_score(y_true, loaded_attributions[speed_attr_label])
    auc_scores["Speed"] = round(auc, 3)
    plot_confusion_matrix(log_dir, loaded_attributions[speed_attr_label], y_true, auc, "Speed")

    auc_scores['Behaviour'] = ["Position", "Head Direction", "Speed"]
    results_df = pd.DataFrame(auc_scores)
    results_df = results_df.set_index("Behaviour")
    csv_path = get_result_filepath(log_dir, "AUC_scores.csv")
    results_df.to_csv(csv_path)

def auc_synthetic(log_dir, num_neurons, num_synthetic):

    load_path = get_result_filepath(log_dir, 'attribution_scores.pt')
    loaded_attributions = torch.load(load_path, map_location=torch.device('cpu'))

    auc_scores = {}
    # recorded neurons are positives; synthetic controls are negatives
    y_true = np.zeros(num_neurons + num_synthetic)
    y_true[0:num_neurons] = True
    behaviours = ['hand_pos', 'hand_vel', 'eye_pos']
    total_behaviour_attr = torch.zeros(num_neurons + num_synthetic)
    for behaviour in behaviours:
      auc = roc_auc_score(y_true, loaded_attributions[behaviour])
      auc_scores[behaviour] = round(auc, 3)
      total_behaviour_attr += loaded_attributions[behaviour]
      plot_confusion_matrix(log_dir, loaded_attributions[behaviour], y_true, auc, behaviour)

    # now we try doing an attribution to all the behaviours:
    total_auc = roc_auc_score(y_true, total_behaviour_attr)
    auc_scores['all_behav'] = total_auc
    behaviours = ['all_behav']

    plot_confusion_matrix(log_dir, total_behaviour_attr, y_true, total_auc, 'all_behav')
    auc_scores['Behaviour'] = behaviours

    results_df = pd.DataFrame(auc_scores)
    results_df = results_df.set_index("Behaviour")
    csv_path = get_result_filepath(log_dir, "AUC_scores.csv")
    results_df.to_csv(csv_path)

def auc_allen(cfg, log_dir):
  load_path = get_result_filepath(log_dir, 'attribution_scores.pt')
  loaded_attributions = torch.load(load_path, map_location=torch.device('cpu'))
  datamodule = load_datamodule_from_config(cfg)
  unit_analysis_metrics = datamodule.dataset('train').get_unit_metrics()
  auc_scores = {}

  # running pvals
  y_true = unit_analysis_metrics['run_pval_dg'] < 0.05
  behaviour = 'speed'
  auc = roc_auc_score(y_true, loaded_attributions[behaviour])
  auc_scores[behaviour] = round(auc, 3)
  plot_confusion_matrix(log_dir, loaded_attributions[behaviour], y_true, auc, behaviour)

  plt.figure(figsize=(6, 5))
  colour = get_behaviour_colour('orientation_corr')
  plt.scatter(unit_analysis_metrics['g_osi_dg'], loaded_attributions['orientation'], s=10, alpha=0.7, color=colour)
  corr = np.corrcoef(loaded_attributions['orientation'], unit_analysis_metrics['g_osi_dg'])[0, 1]
  plt.xlabel("gOSI")
  plt.ylabel("Attribution")
  plt.title(f"Attribution vs Orientation Selectivity, corr: {round(corr, 3)}")
  auc_scores['orientation_corr'] = corr

  plt.tight_layout()
  path = get_visualization_filepath(log_dir, "g_osi_dg_vs_ori_attr_scatter.png")
  plt.savefig(path)
  print(f"saved to {path}")
  plt.close()

  auc_scores['Behaviour'] = ['speed', 'orientation_corr']
  results_df = pd.DataFrame(auc_scores)
  results_df = results_df.set_index("Behaviour")
  csv_path = get_result_filepath(log_dir, "AUC_scores.csv")
  results_df.to_csv(csv_path)

  print(f"saved to {csv_path}")

def plot_all():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    log_dir = get_log_dir(cfg, config_filepath)

    if 'cv_folds' in cfg and cfg.cv_folds > 1:
      cv_subdirs = get_cv_subdirs(cfg, log_dir)
      for fold_dir in cv_subdirs:
        plot_all_for_logdir(fold_dir, cfg)
    else:
      plot_all_for_logdir(log_dir, cfg)

def plot_all_for_logdir(log_dir, cfg):
  if cfg.dataset == 'custom_rat_in_a_box.pkl':
    auc_custom(log_dir, num_neurons=cfg.num_cells)
  elif cfg.dataset == 'mc_maze':
    auc_synthetic(log_dir, num_neurons=cfg.num_neurons, num_synthetic=cfg.num_synth_neurons)
  elif cfg.dataset == 'allen_vc':
    auc_allen(cfg, log_dir)
if __name__ == "__main__":
    plot_all()
