import os
import pandas as pd


def merge(folder):
    bvp_path = os.path.join(folder, "BVP.csv")
    gsr_path = os.path.join(folder, "GSR.csv")
    gt_path = os.path.join(folder, "ground_truth.csv")

    bvp = pd.read_csv(bvp_path)
    gsr = pd.read_csv(gsr_path)
    gt = pd.read_csv(gt_path)

    start_ntp = gt.loc[gt['TAG'] == 'HRI_start']['NTPTime'].min() - 10000
    end_ntp = gt.loc[gt['label_emotion']== 'at_goodbye']['NTPTime'].max() + 10000

    gt = gt[['NTPTime','TAG','emotion_HRI']]
    bvp = bvp[['NTPTime', 'BVP', 'BVP_clean']]
    gsr = gsr[['NTPTime', 'GSR', 'GSR_clean']]

    bvp = pd.merge_ordered(bvp, gt, on='NTPTime')
    gsr = pd.merge_ordered(gsr, gt, on='NTPTime')

    bvp['shortNTPTime'] = bvp['NTPTime'].where((bvp['NTPTime'] >= start_ntp) & (bvp['NTPTime'] <= end_ntp))
    gsr['shortNTPTime'] = gsr['NTPTime'].where((gsr['NTPTime'] >= start_ntp) & (gsr['NTPTime'] <= end_ntp))

    # fill forward emotion_HRI for 0.5 to 4 seconds (avg. 2.25 s)
    fill_ms = 2250
    bvp_ms = 1000 / 64  # BVP sampling rate
    gsr_ms = 1000 / 4   # GSR sampling rate

    ffill_bvp = int(fill_ms / bvp_ms)  # BVP samples to fill
    ffill_gsr = int(fill_ms / gsr_ms)  # GSR samples to fill

    bvp['emotion_HRI'] = bvp['emotion_HRI'].ffill(limit=ffill_bvp)
    gsr['emotion_HRI'] = gsr['emotion_HRI'].ffill(limit=ffill_gsr)
    gsr.to_csv(gsr_path,index=False)
    bvp.to_csv(bvp_path,index=False)


parent_folder = r"C:\Users\A. Lowejatan Noori\Desktop\ComTech1\AFFECT-HRI\anonymized-23-10-2023"

for subfolder_name in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, subfolder_name)
    if os.path.isdir(subfolder_path):
        merge(subfolder_path)