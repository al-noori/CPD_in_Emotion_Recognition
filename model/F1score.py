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

    Y_gsr = gsr[['GSR_clean', 'GSR_tonic', 'GSR_phasic', 'GSR_avg', 'GSR_std']].values[gsr['shortNTPTime'].notna()]
    Y_bvp = bvp[['BVP_clean', 'BVP_rate', 'BVP_avg', 'BVP_std']].values[bvp['shortNTPTime'].notna()]
    cp_indices_gsr, _, _ = mcpd(Y_gsr, win_size=10, alpha=2)
    cp_indices_bvp, _, _ = mcpd(Y_bvp, win_size=150, alpha=1)
    cp_indices_bvp = np.concatenate(([0], cp_indices_bvp, [len(Y_bvp) - 1]))
    cp_indices_gsr = np.concatenate(([0], cp_indices_gsr, [len(Y_gsr) - 1]))

    # Calculate F1 scores
    p1, r1 = metrics.precision_recall(gsr_gt_changepoints, cp_indices_gsr, margin=20)
    p2, r2 = metrics.precision_recall(bvp_gt_changepoints, cp_indices_bvp, margin=320)
    f1_gsr = 2 * (p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0
    f1_bvp = 2 * (p2 * r2) / (p2 + r2) if (p2 + r2) > 0 else 0

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

print(participant_ids.shape, F1_scores_gsr.shape, F1_scores_bvp.shape)
# Plot
plt.figure(figsize=(8, 8))
plt.scatter(participant_ids, F1_scores_gsr, color='blue', s=80)
plt.scatter(participant_ids, F1_scores_bvp, color='orange', s=80)

plt.xlabel('Participant ID')
plt.ylabel('F1')
plt.ylim(0, 1.05)  # F1 scores range from 0 to 1
plt.grid(True)
plt.xticks(participant_ids)
plt.tight_layout()

plt.savefig(path.PLOTS_PATH / 'F1_scores_participants.png')

