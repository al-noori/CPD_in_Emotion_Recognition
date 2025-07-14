from best_psi import best_psi
from best_threshold import best_threshold

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ruptures import Pelt
from scipy.signal import find_peaks
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
pred_gsr_pelt = {}
gt_bvp = {}
pred_bvp = {}
pred_bvp_pelt = {}

for participant_id in os.listdir(dataset_dir):
    if not os.path.isdir(os.path.join(dataset_dir, participant_id)):
        continue
    print(participant_id)
    part_path = os.path.join(dataset_dir, participant_id)

    # Paths to CSVs
    bvp_path = os.path.join(part_path, 'BVP.csv')
    gsr_path = os.path.join(part_path, 'GSR.csv')

    bvp = pd.read_csv(bvp_path)
    gsr = pd.read_csv(gsr_path)

    # Filter and reset index so everything aligns with prediction inputs
    gsr_valid = gsr[gsr['shortNTPTime'].notna()].reset_index(drop=True)
    bvp_valid = bvp[bvp['shortNTPTime'].notna()].reset_index(drop=True)

    # Get ground-truth changepoints by relative index
    gsr_gt_changepoints = gsr_valid.index[gsr_valid['emotion_HRI'].notna()].tolist()
    bvp_gt_changepoints = bvp_valid.index[bvp_valid['emotion_HRI'].notna()].tolist()

    # Store in dictionary
    gt_gsr[participant_id] = gsr_gt_changepoints
    gt_bvp[participant_id] = bvp_gt_changepoints

    # Extract signal arrays for detection
    Y_gsr = gsr_valid['GSR_clean'].values
    Y_bvp = bvp_valid['BVP_clean'].values

    # MCPD predictions
    cp_indices_gsr, pscore, threshold = mcpd(Y_gsr, win_size=50, alpha=1)
    cp_indices_bvp, pscore2, threshold2 = mcpd(Y_bvp, win_size=300, alpha=2)

    # PELT predictions (adjust penalty as needed)
    cp_indices_gsr_pelt = pelt_cpd(Y_gsr, model="rbf", pen=10)
    cp_indices_bvp_pelt = pelt_cpd(Y_bvp, model="rbf", pen=10)

    # Store predictions
    pred_gsr[participant_id] = cp_indices_gsr.tolist()
    pred_bvp[participant_id] = cp_indices_bvp.tolist()
    pred_gsr_pelt[participant_id] = cp_indices_gsr_pelt.tolist()
    pred_bvp_pelt[participant_id] = cp_indices_bvp_pelt.tolist()

    # Compute final F1 scores for MCPD
    P_gsr, R_gsr, F1_gsr = compute_f1_by_participant(gt_gsr, pred_gsr, M=40)
    P_bvp, R_bvp, F1_bvp = compute_f1_by_participant(gt_bvp, pred_bvp, M=640)

    # Compute final F1 scores for PELT
    P_gsr_pelt, R_gsr_pelt, F1_gsr_pelt = compute_f1_by_participant(gt_gsr, pred_gsr_pelt, M=40)
    P_bvp_pelt, R_bvp_pelt, F1_bvp_pelt = compute_f1_by_participant(gt_bvp, pred_bvp_pelt, M=640)

    print("IDK GSR only:")
    print(f"  Precision: {P_gsr:.3f}, Recall: {R_gsr:.3f}, F1: {F1_gsr:.3f}")
    print("IDK BVP only:")
    print(f"  Precision: {P_bvp:.3f}, Recall: {R_bvp:.3f}, F1: {F1_bvp:.3f}")

    print("\nPELT GSR only:")
    print(f"  Precision: {P_gsr_pelt:.3f}, Recall: {R_gsr_pelt:.3f}, F1: {F1_gsr_pelt:.3f}")
    print("PELT BVP only:")
    print(f"  Precision: {P_bvp_pelt:.3f}, Recall: {R_bvp_pelt:.3f}, F1: {F1_bvp_pelt:.3f}")

# Compute final F1 scores for MCPD
P_gsr, R_gsr, F1_gsr = compute_f1_by_participant(gt_gsr, pred_gsr, M=40)
P_bvp, R_bvp, F1_bvp = compute_f1_by_participant(gt_bvp, pred_bvp, M=640)

# Compute final F1 scores for PELT
P_gsr_pelt, R_gsr_pelt, F1_gsr_pelt = compute_f1_by_participant(gt_gsr, pred_gsr_pelt, M=40)
P_bvp_pelt, R_bvp_pelt, F1_bvp_pelt = compute_f1_by_participant(gt_bvp, pred_bvp_pelt, M=640)

print("IDK GSR only:")
print(f"  Precision: {P_gsr:.3f}, Recall: {R_gsr:.3f}, F1: {F1_gsr:.3f}")
print("IDK BVP only:")
print(f"  Precision: {P_bvp:.3f}, Recall: {R_bvp:.3f}, F1: {F1_bvp:.3f}")

print("\nPELT GSR only:")
print(f"  Precision: {P_gsr_pelt:.3f}, Recall: {R_gsr_pelt:.3f}, F1: {F1_gsr_pelt:.3f}")
print("PELT BVP only:")
print(f"  Precision: {P_bvp_pelt:.3f}, Recall: {R_bvp_pelt:.3f}, F1: {F1_bvp_pelt:.3f}")


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
