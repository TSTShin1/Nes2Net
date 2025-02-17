import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

def read_scores(file_path):
    scores_df = pd.read_csv(file_path, sep=' ', header=None, names=['filename', 'score'])
    return scores_df

def read_labels(file_path):
    labels_df = pd.read_csv(file_path, sep=' ', header=None, names=['datasetname', 'index', 'filename', '-', 'atkID', 'label'])
    return labels_df

def calculate_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    eer = (fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))] + (1 - tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])) / 2
    return eer

def calculate_mindcf(scores, labels, prior=0.5, cost_fn=1.0):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    min_dcf = np.inf
    for t in thresholds:
        fnr = 1 - tpr
        fpr_i = fpr[thresholds == t]
        fnr_i = fnr[thresholds == t]
        dcf = prior * fpr_i + (1 - prior) * fnr_i
        min_dcf = min(min_dcf, dcf)
    return min_dcf

def main(scores_file, labels_file):
    scores_df = read_scores(scores_file)
    labels_df = read_labels(labels_file)
    
    # Merge scores with labels
    merged_df = pd.merge(scores_df, labels_df[['filename', 'label', 'atkID', 'datasetname']], on='filename')
    
    # Convert labels to binary (deepfake=1, bonafide=0)
    merged_df['label'] = merged_df['label'].apply(lambda x: 0 if x == 'deepfake' else 1)
    
    filtered_df_exA14 = merged_df[merged_df['atkID'] != 'A14']
    filtered_df_exACE = merged_df[merged_df['datasetname'] != 'acesinger']
    filtered_df_exACE_A14 = filtered_df_exACE[filtered_df_exACE['atkID'] != 'A14']

    print('---------------------------------------------------------')

    # Calculate EER and minDCF for each datasetname
    for datasetname in merged_df['datasetname'].unique():
        if datasetname != 'acesinger':
            dataset_df = merged_df[merged_df['datasetname'] == datasetname]
            eer_dataset = calculate_eer(dataset_df['score'], dataset_df['label']) * 100
            minDCF_dataset = calculate_mindcf(dataset_df['score'], dataset_df['label'])
            print(f'dataset {datasetname} - EER: {eer_dataset:.4f}%  minDCF: {minDCF_dataset[0]:.6f}')
    print('---------------------------------------------------------')

    # Exclude atkID A14
    eer_filtered = calculate_eer(filtered_df_exA14['score'], filtered_df_exA14['label']) * 100
    minDCF_filtered = calculate_mindcf(filtered_df_exA14['score'], filtered_df_exA14['label'])
    print('excluding A14 only, #: {}'.format(len(filtered_df_exA14)))
    print(f'- EER: {eer_filtered:.4f}%  minDCF: {minDCF_filtered[0]:.6f}')
    print('---------------------------------------------------------')


    print('excluding both acesinger and A14, #: {}'.format(len(filtered_df_exACE_A14)))
    eer_filtered = calculate_eer(filtered_df_exACE_A14['score'], filtered_df_exACE_A14['label']) * 100
    minDCF_filtered = calculate_mindcf(filtered_df_exACE_A14['score'], filtered_df_exACE_A14['label'])
    print(f'- EER: {eer_filtered:.4f}%  minDCF: {minDCF_filtered[0]:.6f}')

    # Calculate EER and minDCF for each atkID
    for atkID in ["A09", "A10", "A11", "A12", "A13"]:
        atkID_df = filtered_df_exACE_A14[(filtered_df_exACE_A14['atkID'] == atkID) | (filtered_df_exACE_A14['label'] == 1)]
        # print(atkID_df)
        # x = input('debug')
        eer_atkID = calculate_eer(atkID_df['score'], atkID_df['label']) * 100
        minDCF_atkID = calculate_mindcf(atkID_df['score'], atkID_df['label'])
        print(f'(atkID {atkID}) - EER: {eer_atkID:.4f}%  minDCF: {minDCF_atkID[0]:.6f}')
    print('---------------------------------------------------------')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate EER and minDCF from scores and labels.")
    parser.add_argument('--path', type=str, required=True, help="Path to the scores file.")
    parser.add_argument('--labels_file', type=str, default='/home/tianchi/data/SVDD2024/test.txt', help="Path to the labels file.")

    args = parser.parse_args()

    main(args.path, args.labels_file)
