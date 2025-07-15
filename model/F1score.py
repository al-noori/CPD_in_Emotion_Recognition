from joblib import delayed, Parallel

import path
from best_psi import best_psi
from best_threshold import best_threshold

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ruptures import Pelt
from scipy.signal import find_peaks
from ruptures import metrics
import joblib
def mcpd(Y, win_size=40, alpha=1):
    """
    Multiple Change Point Detection (MCPD) via model
    """
    pscore, _, _ = best_psi(Y, win_size)
    threshold, _ = best_threshold(pscore, alpha)
    indices = find_peaks(pscore, height=threshold)[0] #Non-Maximum Suppression
    #np.where(pscore > threshold)[0]
    return indices, pscore, threshold

def pelt_cpd(Y, model="rbf", pen=10):
    """
    Change point detection using PELT
    """
    algo = Pelt(model=model).fit(Y)
    cp_indices = algo.predict(pen=pen)
    # Remove the last point as ruptures returns length as last cp
    return np.array(cp_indices[:-1])


gt_gsr = {}
pred_gsr = {}
gt_bvp = {}
pred_bvp = {}

F1_scores_gsr=[]
F1_scores_bvp=[]

def process_participant(participant_id):
    part_path = os.path.join(path.DATA_PATH, participant_id)

    bvp_path = os.path.join(part_path, 'BVP.csv')
    gsr_path = os.path.join(part_path, 'GSR.csv')

    bvp = pd.read_csv(bvp_path)
    gsr = pd.read_csv(gsr_path)

    gsr_valid = gsr[gsr['shortNTPTime'].notna()].reset_index(drop=True)
    bvp_valid = bvp[bvp['shortNTPTime'].notna()].reset_index(drop=True)

    gsr_gt_changepoints = gsr_valid.index[gsr_valid['emotion_HRI'].notna()].tolist()
    bvp_gt_changepoints = bvp_valid.index[bvp_valid['emotion_HRI'].notna()].tolist()

    Y_gsr = gsr_valid['GSR_clean'].values
    Y_bvp = bvp_valid['BVP_clean'].values

    cp_indices_gsr, _, _ = mcpd(Y_gsr, win_size=50, alpha=2)
    cp_indices_bvp, _, _ = mcpd(Y_bvp, win_size=150, alpha=2)

    f1_gsr = metrics.precision_recall(cp_indices_gsr, gsr_gt_changepoints, margin=20)
    f1_bvp = metrics.precision_recall(cp_indices_bvp, bvp_gt_changepoints, margin=320)

    return {
        'participant_id': participant_id,
        'gt_gsr': gsr_gt_changepoints,
        'gt_bvp': bvp_gt_changepoints,
        'pred_gsr': cp_indices_gsr.tolist(),
        'pred_bvp': cp_indices_bvp.tolist(),
        'f1_gsr': f1_gsr,
        'f1_bvp': f1_bvp
    }

# Get participant directories
participants = [
    pid for pid in os.listdir(path.DATA_PATH)
    if os.path.isdir(os.path.join(path.DATA_PATH, pid))
]

# Run in parallel
results = Parallel(n_jobs=-1)(delayed(process_participant)(pid) for pid in participants)

# Collect results
for res in results:
    pid = res['participant_id']
    gt_gsr[pid] = res['gt_gsr']
    gt_bvp[pid] = res['gt_bvp']
    pred_gsr[pid] = res['pred_gsr']
    pred_bvp[pid] = res['pred_bvp']
    F1_scores_gsr.append(res['f1_gsr'])
    F1_scores_bvp.append(res['f1_bvp'])

# Example data
participant_ids = np.arange(1,len(F1_scores_gsr)+1)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(participant_ids, F1_scores_gsr, color='blue', s=80)
plt.scatter(participant_ids, F1_scores_bvp, color='orange', s=80)

plt.xlabel('Participant ID')
plt.ylabel('F1')
plt.ylim(0, 1.05)  # F1 scores range from 0 to 1
plt.grid(True)
plt.xticks(participant_ids)
plt.tight_layout()

plt.savefig(path.PLOTS_PATH / 'F1_scores_participants.png')

'''
IDK GSR only:
  Precision: 0.029, Recall: 0.607, F1: 0.055
IDK BVP only:
  Precision: 0.002, Recall: 0.767, F1: 0.003

PELT GSR only:
  Precision: 0.289, Recall: 0.484, F1: 0.362
PELT BVP only:
  Precision: 0.026, Recall: 0.983, F1: 0.052
  
  with local maxima
IDK GSR only:
  Precision: 0.194, Recall: 0.493, F1: 0.279
IDK BVP only:
  Precision: 0.062, Recall: 0.406, F1: 0.107

PELT GSR only:
  Precision: 0.146, Recall: 0.491, F1: 0.225
PELT BVP only:
  Precision: 0.015, Recall: 0.989, F1: 0.029
'''
