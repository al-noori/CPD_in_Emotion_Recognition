from best_psi import best_psi
from best_threshold import best_threshold
from point_score import point_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def mcpd(Y, win_size = 40, alpha = 1):
    """
    Multiple Change Point Detection (MCPD) via iCID
    """
    pscore, _, _ = best_psi(Y, win_size)
    threshold, _ = best_threshold(pscore, alpha)
    return np.where(pscore > threshold)[0], pscore, threshold


def compute_f1_by_participant(gt_dict, pred_dict, M=5):
    total_TP = 0
    total_pred = 0
    recall_sum = 0.0
    K = len(gt_dict)

    for k, T_k in gt_dict.items():
        X_k = pred_dict.get(k, [])
        matched = set()
        tp_k = 0

        # Greedy match within participant k
        for x in X_k:
            for idx, tau in enumerate(T_k):
                if idx in matched:
                    continue
                if abs(x - tau) <= M:
                    tp_k += 1
                    matched.add(idx)
                    break

        total_TP += tp_k
        total_pred += len(X_k)
        recall_sum += (tp_k / len(T_k)) if T_k else 0.0

    precision = total_TP / total_pred if total_pred > 0 else 0.0
    recall = recall_sum / K
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

dataset_dir = r"C:\Users\A. Lowejatan Noori\Desktop\ComTech1\AFFECT-HRI\anonymized-23-10-2023"

gt_gsr = {}
pred_gsr = {}
gt_bvp = {}
pred_bvp = {}

for participant_id in os.listdir(dataset_dir):
    part_path = os.path.join(dataset_dir, participant_id)

    # Paths to CSVs
    bvp_path = os.path.join(part_path, 'BVP.csv')
    gsr_path = os.path.join(part_path, 'GSR.csv')

    bvp = pd.read_csv(bvp_path)
    gsr = pd.read_csv(gsr_path)

    bvp_gt_changepoints =  bvp.index[bvp['emotion_HRI'].notna()].tolist()
    gsr_gt_changepoints = gsr.index[gsr['emotion_HRI'].notna()].tolist()

    gt_gsr[participant_id] = gsr_gt_changepoints
    gt_bvp[participant_id] = bvp_gt_changepoints

    Y = gsr['GSR_clean'].values[gsr['shortNTPTime'].notna()]
    Y2 = bvp['BVP_clean'].values[bvp['shortNTPTime'].notna()]

    print(participant_id)

    cp_indices_gsr, pscore, threshold = mcpd(Y.reshape(-1, 1), win_size=60, alpha=1)
    cp_indices_bvp, pscore2, threshold2 = mcpd(Y2.reshape(-1, 1), win_size=300, alpha=2)

    pred_gsr[participant_id] = cp_indices_gsr
    pred_bvp[participant_id] = cp_indices_bvp

P_gsr, R_gsr, F1_gsr = compute_f1_by_participant(gt_gsr, pred_gsr, M=12)
P_bvp, R_bvp, F1_bvp = compute_f1_by_participant(gt_bvp, pred_bvp, M=32)

print("GSR only:")
print(f"  Precision: {P_gsr:.3f}, Recall: {R_gsr:.3f}, F1: {F1_gsr:.3f}")
print("BVP only:")
print(f"  Precision: {P_bvp:.3f}, Recall: {R_bvp:.3f}, F1: {F1_bvp:.3f}")


