import os
import sys
import json
import traceback
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score


def calc_metric(dataset_path: Path, result_path: Path):
    with open(dataset_path, 'r') as f:
        gt_df = pd.DataFrame(json.load(f))
        gt_df = gt_df[['id', 'label']]

    with open(result_path, 'r') as f:
        res_df = pd.DataFrame(json.load(f))
        if 'label' in res_df.columns:
            res_df = res_df.drop('label', axis=1)
        res_df = res_df.merge(gt_df, on='id', how='right')
        res_df['score'] = res_df['score'].fillna(0)
        gt_score = res_df['label']
        pred_score = res_df['score']

        gt_mean = gt_score.mean()
        pred_mean = pred_score.mean()
        err_rate = abs(gt_mean - pred_mean)

        try:
            auc_value = roc_auc_score(gt_score, pred_score).item()
        except:
            auc_value = 0

        try:
            pearsonr_value = pearsonr(gt_score, pred_score).statistic.item()
        except:
            pearsonr_value = 0

        report = {
            'auc': round(auc_value, 4),
            'mean': round(pred_mean, 4),
            'gt_mean': round(gt_mean, 4),
            'err_rate': round(err_rate, 4),
            'pearsonr': round(pearsonr_value, 4),
        }

    stats_path = result_path.parent / 'stats.json'
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            report.update(stats)
        
    return report

if __name__ == '__main__':
    exp_name = sys.argv[1]
    dataset = exp_name.split('|')[1] + '.json'
    report = calc_metric(
        Path(os.path.dirname(__file__)) / '..' / 'data' / 'final' / dataset,
        Path(os.path.dirname(__file__)) / '..' / 'output' / exp_name / 'score.json',
    )
    for metric, val in report.items():
        print('%s: %.04f' % (metric, val))