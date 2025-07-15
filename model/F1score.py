from joblib import Parallel, delayed
import os
import numpy as np
import pandas as pd
from ruptures import metrics
from scipy.signal import find_peaks
from best_psi import best_psi
from best_threshold import best_threshold
import path

def mcpd(Y, win_size=40, alpha=1):
    """
    Multiple Change Point Detection (MCPD) via custom model
    """
    pscore, _, _ = best_psi(Y, win_size)
    threshold, _ = best_threshold(pscore, alpha)
    indices = find_peaks(pscore, height=threshold)[0]
    return indices, pscore, threshold


def evaluate_participant_f1(pid, signal_type, alpha, win_size, feature_cols, margin):
    """
    Evaluate F1 score for one participant with given parameters.
    """
    try:
        part_path = os.path.join(path.DATA_PATH, pid)
        df = pd.read_csv(os.path.join(part_path, f'{signal_type}.csv'))
        df_valid = df[df['shortNTPTime'].notna()].reset_index(drop=True)
        Y = df_valid[feature_cols].values
        gt_cp = df_valid.index[df_valid['emotion_HRI'].notna()].tolist()

        pred_cp, _, _ = mcpd(Y, win_size=win_size, alpha=alpha)
        pred_cp = np.concatenate(([0], pred_cp, [len(Y) - 1]))

        p, r = metrics.precision_recall(gt_cp, pred_cp, margin=margin)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        return f1
    except Exception as e:
        print(f"Error processing {pid}: {e}")
        return 0


def global_grid_search(participants, signal_type='GSR'):
    """
    Run grid search across all participants to find the best alpha and win_size globally.
    """
    if signal_type == 'GSR':
        alpha_range = [0, 1, 2, 3]
        win_size_range = [10, 50, 100]
        margin = 20
        feature_cols = ['GSR_clean', 'GSR_tonic', 'GSR_phasic', 'GSR_avg', 'GSR_std']
    elif signal_type == 'BVP':
        alpha_range = [0, 1, 2, 3]
        win_size_range = [100, 200, 300]
        margin = 320
        feature_cols = ['BVP_clean', 'BVP_rate', 'BVP_avg', 'BVP_std']
    else:
        raise ValueError("signal_type must be either 'GSR' or 'BVP'")

    best_mean_f1 = -1
    best_params = None

    for alpha in alpha_range:
        for win_size in win_size_range:
            print(f"→ Testing {signal_type}: alpha={alpha}, win_size={win_size}")

            f1_scores = Parallel(n_jobs=-1)(
                delayed(evaluate_participant_f1)(
                    pid, signal_type, alpha, win_size, feature_cols, margin
                )
                for pid in participants
            )

            mean_f1 = np.mean(f1_scores)
            print(f"   → Mean F1: {mean_f1:.4f}")

            if mean_f1 > best_mean_f1:
                best_mean_f1 = mean_f1
                best_params = (alpha, win_size)

    print(f"\n✅ Best {signal_type} parameters: alpha={best_params[0]}, win_size={best_params[1]} → Mean F1: {best_mean_f1:.4f}")
    return best_params, best_mean_f1


# Main execution
if __name__ == "__main__":
    participants = [
        pid for pid in os.listdir(path.DATA_PATH)
        if os.path.isdir(os.path.join(path.DATA_PATH, pid))
    ]

    best_gsr_params, best_gsr_f1 = global_grid_search(participants, signal_type='GSR')
    best_bvp_params, best_bvp_f1 = global_grid_search(participants, signal_type='BVP')



'''
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

'''