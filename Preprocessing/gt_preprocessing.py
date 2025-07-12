import os
import pandas as pd
parent_folder = r"C:\Users\A. Lowejatan Noori\Desktop\ComTech1\AFFECT-HRI\anonymized-23-10-2023"
valid = []
for name in os.listdir(parent_folder):
    if (name != "gesture_speech.csv") and (name != "questionnaire.csv"):
        valid.append(name)

questionnaire = pd.read_csv(os.path.join(parent_folder, "questionnaire.csv"))
questionnaire = questionnaire[questionnaire['code'].isin(valid)]
questionnaire.to_csv(os.path.join(parent_folder, "questionnaire.csv"), index=False)

entry_column_map = {
    'at_drillrecommendation': ('at_drillrecommendation_valence', 'at_drillrecommendation_arousal'),
    'at_customeraccount_name': ('at_customeraccount_name_valence', 'at_customeraccount_name_arousal'),
    'at_customeraccount_consent': ('at_customeraccount_consent_valence', 'at_customeraccount_consent_arousal'),
    'at_mold_remover_handover': ('at_moldremover_handover_valence', 'at_moldremover_handover_arousal'),
    'at_goodbye': ('at_good-bye_valence', 'at_good-bye_arousal'),
}

for subfolder_name in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, subfolder_name)

    if os.path.isdir(subfolder_path):
        csv_path = os.path.join(subfolder_path, 'ground_truth.csv')
        df = pd.read_csv(csv_path)

        row = questionnaire[questionnaire['code'] == subfolder_name]
        row = row.iloc[0]
        for idx, data_row in df.iterrows():
            label = data_row['label_emotion']
            for entry, (val_col, arousal_col) in entry_column_map.items():
                if label == entry or label == "mood_pre" or label == "mood_post" :
                    df.at[idx, 'emotion_valence'] = row[val_col]
                    df.at[idx, 'emotion_arousal'] = row[arousal_col]

                    if not pd.isna(label) and row[val_col] <= 2 and row[arousal_col] <= 2:
                        df.at[idx, 'emotion'] = 'HNVLA'
                    elif not pd.isna(label) and row[val_col] <= 2 and row[arousal_col] >= 4:
                        df.at[idx, 'emotion'] = 'HNVHA'
                    elif not pd.isna(label) and row[val_col] >= 4 and row[arousal_col] >= 4:
                        df.at[idx, 'emotion'] = 'HPVHA'
                    elif not pd.isna(label) and row[val_col] >= 4 and row[arousal_col] <= 2:
                        df.at[idx, 'emotion'] = 'HPVLA'
                    elif not pd.isna(label):
                        df.at[idx, 'emotion'] = 'NEUTRAL'


        target_entries =  list(entry_column_map.keys())


        # Create a mask where label_emotion is in target_entries
        mask = df['label_emotion'].isin(target_entries)

        labels = df['label_emotion'].where(mask)

        # When label_emotion changes compared to previous row, mark True; else False
        change = labels != labels.shift()

        # Cumulative sum of changes to assign group numbers only to consecutive identical labels
        group_ids = change.cumsum()

        # Assign group IDs only to rows where mask is True; others get NaN
        df['group'] = group_ids.where(mask)

        df['emotion_HRI'] = df['emotion'].mask( df['group'].duplicated(), '')
        df.loc[df['label_emotion'].isin(['mood_pre', 'mood_post']), 'emotion_HRI'] = ''
        df.drop(columns=['group'])

        df.to_csv(csv_path, index=False)