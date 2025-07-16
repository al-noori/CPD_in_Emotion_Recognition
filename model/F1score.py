# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from joblib import delayed, Parallel
from ruptures import metrics
import path

from best_psi import best_psi
from best_threshold import best_threshold

# change point detection function
def mcpd(Y, win_size=40, alpha=1):
    pscore, _, _ = best_psi(Y, win_size)
    threshold, _ = best_threshold(pscore, alpha)
    indices = find_peaks(pscore, height=threshold)[0]
    return indices, pscore, threshold

# compute true positives, false positives, false negatives
def compute_tp_fp_fn(pred, gt, margin):
    matched_gt = set()
    tp = 0
    for p in pred:
        for g in gt:
            if abs(p - g) <= margin and g not in matched_gt:
                tp += 1
                matched_gt.add(g)
                break
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn

# compute mean absolute error
def compute_mae(pred, gt):
    errors = []
    for p in pred:
        if len(gt) == 0:
            continue
        nearest_gt = min(gt, key=lambda x: abs(x - p))
        errors.append(abs(p - nearest_gt))
    return np.mean(errors) if errors else 0

# per-participant processing function
def process_participant(participant_id, emotion="HPVLA"):
    part_path = os.path.join(path.DATA_PATH, participant_id)
    bvp_path = os.path.join(part_path, 'BVP.csv')
    gsr_path = os.path.join(part_path, 'GSR.csv')

    bvp = pd.read_csv(bvp_path)
    gsr = pd.read_csv(gsr_path)

    gsr_valid = gsr[gsr['shortNTPTime'].notna()].reset_index(drop=True)
    bvp_valid = bvp[bvp['shortNTPTime'].notna()].reset_index(drop=True)

    true_gsr = gsr_valid.index[gsr_valid['emotion_HRI'].notna()].tolist()
    true_bvp = bvp_valid.index[bvp_valid['emotion_HRI'].notna()].tolist()

    Y_gsr = gsr[['GSR_clean', 'GSR_tonic', 'GSR_phasic', 'GSR_avg', 'GSR_std']].values[gsr['shortNTPTime'].notna()]
    Y_bvp = bvp[['BVP_clean', 'BVP_rate', 'BVP_avg', 'BVP_std']].values[bvp['shortNTPTime'].notna()]

    pred_gsr, _, _ = mcpd(Y_gsr, win_size=50, alpha=0.5)
    pred_bvp, _, _ = mcpd(Y_bvp, win_size=200, alpha=2)

    pred_bvp = np.unique(np.concatenate(([0], pred_bvp, [len(Y_bvp) - 1])))
    pred_gsr = np.unique(np.concatenate(([0], pred_gsr, [len(Y_gsr) - 1])))
    true_gsr = np.unique(np.concatenate(([0], true_gsr, [len(Y_gsr) - 1])))
    true_bvp = np.unique(np.concatenate(([0], true_bvp, [len(Y_bvp) - 1])))

    p1, r1 = metrics.precision_recall(true_gsr, pred_gsr, margin=40)
    p2, r2 = metrics.precision_recall(true_bvp, pred_bvp, margin=640)
    f1_gsr = 2 * (p1 * r1) / (p1 + r1) if (p1 + r1) > 0 else 0
    f1_bvp = 2 * (p2 * r2) / (p2 + r2) if (p2 + r2) > 0 else 0

    return {
        'participant_id': participant_id,
        'gt_gsr': true_gsr,
        'gt_bvp': true_bvp,
        'pred_gsr': pred_gsr.tolist(),
        'pred_bvp': pred_bvp.tolist(),
        'f1_gsr': f1_gsr,
        'f1_bvp': f1_bvp
    }

# get list of participants
participants = [
    pid for pid in os.listdir(path.DATA_PATH)
    if os.path.isdir(os.path.join(path.DATA_PATH, pid))
]

# process all participants in parallel
results = Parallel(n_jobs=-1)(delayed(process_participant)(pid) for pid in participants)

# collect f1 scores
F1_scores_gsr = []
F1_scores_bvp = []

for res in results:
    F1_scores_gsr.append(res['f1_gsr'])
    F1_scores_bvp.append(res['f1_bvp'])

# initialize tp/fp/fn counters
total_tp_gsr = total_fp_gsr = total_fn_gsr = 0
total_tp_bvp = total_fp_bvp = total_fn_bvp = 0

# compute total tp, fp, fn
for res in results:
    tp, fp, fn = compute_tp_fp_fn(res['pred_gsr'], res['gt_gsr'], margin=40)
    total_tp_gsr += tp
    total_fp_gsr += fp
    total_fn_gsr += fn

    tp, fp, fn = compute_tp_fp_fn(res['pred_bvp'], res['gt_bvp'], margin=640)
    total_tp_bvp += tp
    total_fp_bvp += fp
    total_fn_bvp += fn

# compute micro precision, recall, f1
precision_gsr = total_tp_gsr / (total_tp_gsr + total_fp_gsr + 1e-10)
recall_gsr = total_tp_gsr / (total_tp_gsr + total_fn_gsr + 1e-10)
micro_f1_gsr = 2 * precision_gsr * recall_gsr / (precision_gsr + recall_gsr + 1e-10)

precision_bvp = total_tp_bvp / (total_tp_bvp + total_fp_bvp + 1e-10)
recall_bvp = total_tp_bvp / (total_tp_bvp + total_fn_bvp + 1e-10)
micro_f1_bvp = 2 * precision_bvp * recall_bvp / (precision_bvp + recall_bvp + 1e-10)

# compute mae for each participant
all_errors_gsr = []
all_errors_bvp = []

for res in results:
    for p in res['pred_gsr']:
        if len(res['gt_gsr']) == 0:
            continue
        nearest_gt = min(res['gt_gsr'], key=lambda x: abs(x - p))
        all_errors_gsr.append(abs(p - nearest_gt))

    for p in res['pred_bvp']:
        if len(res['gt_bvp']) == 0:
            continue
        nearest_gt = min(res['gt_bvp'], key=lambda x: abs(x - p))
        all_errors_bvp.append(abs(p - nearest_gt))
mae_gsr = np.mean(all_errors_gsr)
mae_bvp = np.mean(all_errors_bvp)

# print results
participant_ids = np.arange(1, len(F1_scores_gsr) + 1)
print("macro f1 (gsr):", np.mean(F1_scores_gsr))
print("macro f1 (bvp):", np.mean(F1_scores_bvp))
print("micro f1 (gsr):", micro_f1_gsr)
print("micro f1 (bvp):", micro_f1_bvp)
print("recall (gsr):", recall_gsr)
print("recall (bvp):", recall_bvp)
print("precision (gsr):", precision_gsr)
print("precision (bvp):", precision_bvp)
print("mean absolute error (gsr):", mae_gsr)
print("mean absolute error (bvp):", mae_bvp)

# plot f1 scores
plt.figure(figsize=(8, 8))
plt.scatter(participant_ids, F1_scores_gsr, color='blue', s=80, label='GSR')
plt.scatter(participant_ids, F1_scores_bvp, color='orange', s=80, label='BVP')

plt.xlabel('participant id')
plt.ylabel('f1 score')
plt.ylim(0, 1.05)
plt.grid(True)
plt.xticks(participant_ids)
plt.legend()
plt.tight_layout()

plt.savefig(path.PLOTS_PATH / 'F1_scores_participants.png')
