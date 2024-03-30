#!/usr/bin/env python3
import pandas as pd
import json
import ast
import scipy
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os.path, time, datetime
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm


# Given a directory or single file, this script will print the metrics for the predictions in the files.
# Files has to be csv files with the following format:
# (cell_id, drug_id, true_value, predicted_value)

DIGITS = 3

METRICS = {'RMSE': root_mean_squared_error,
           'R2': r2_score,
           'Spearman': lambda x, y: spearmanr(x, y)[0],
           'Pearson': lambda x, y: pearsonr(x, y)[0]}



def process_prediction(path, save_metrics=False):
    '''
    Process the prediction file(s) and return the metrics for the predictions

    Args:
        path (str): Path to the file or directory with the predictions
        save_metrics (bool): If True, the metrics will be saved in a file (useful to compute distribution)
    '''



    global_perf = {metric: [] for metric in METRICS}
    fixed_drug_perf = {metric: [] for metric in METRICS}
    fixed_cell_perf = {metric: [] for metric in METRICS}


    # Check if dir or file
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)]
    else:
        files = [path]


    # Loop through all files
    for i, f_path in enumerate(tqdm(files)):
        # print(f"Loding file: {f_path}")
        df = pd.read_csv(f_path)

        cells = df["cell"].values
        drugs = df["drug"].values
        y_test = df["true_value"].values
        y_hat_test = df["predicted_value"].values

        # Global metrics
        # print(f"Computing Global metrics for {f_path}")
        for metric in METRICS:
            global_perf[metric].append(METRICS[metric](y_test, y_hat_test))



        # Fixed drug metrics (use df to get the drugs preds)
        # print(f"Computing Fixed drug metrics for {f_path}")
        df_drugs = df.groupby("drug")
        fixed_drug = {metric: [] for metric in METRICS}
        for drug, df_drug in df_drugs:
            y_test_drug = df_drug["true_value"].values
            y_hat_test_drug = df_drug["predicted_value"].values
            if len(y_test_drug) < 2:
                continue
            for metric in METRICS:
                fixed_drug[metric].append(METRICS[metric](y_test_drug, y_hat_test_drug))
        for metric in METRICS:
            fixed_drug_perf[metric].append(np.nanmean(fixed_drug[metric]))

        # Fixed cell metrics (use df to get the cells preds)
        # print(f"Computing Fixed cell metrics for {f_path}")
        df_cells = df.groupby("cell")
        fixed_cell = {metric: [] for metric in METRICS}
        for cell, df_cell in df_cells:
            y_test_cell = df_cell["true_value"].values
            y_hat_test_cell = df_cell["predicted_value"].values
            if len(y_test_cell) < 2:
                continue
            for metric in METRICS:
                fixed_cell[metric].append(METRICS[metric](y_test_cell, y_hat_test_cell))
        for metric in METRICS:
            fixed_cell_perf[metric].append(np.nanmean(fixed_cell[metric]))

    # Print metrics for this file averaging all the runs,
    print(f"Metrics for {path.split('.')[0]}")
    print("Number of runs: ", len(global_perf["RMSE"]))

    print("Global metrics:")
    for metric in METRICS:
        print(f"\t{metric}: {round(np.mean(global_perf[metric]),DIGITS)}, std: {round(np.std(global_perf[metric]),DIGITS)}")

    print("\nFixed_drug metrics:")
    for metric in METRICS:
        print(f"\t{metric}: {round(np.mean(fixed_drug_perf[metric]),DIGITS)}, std: {round(np.std(fixed_drug_perf[metric]),DIGITS)}")

    print("\nFixed_cell metrics:")
    for metric in METRICS:
        print(f"\t{metric}: {round(np.mean(fixed_cell_perf[metric]),DIGITS)}, std: {round(np.std(fixed_cell_perf[metric]),DIGITS)}")
    print("\n\n")


    if save_metrics:
        save_path = f"{path.split('.')[0]}_metrics.json"
        with open(save_path, 'w') as f:
            json.dump({"global": global_perf, "fixed_drug": fixed_drug_perf, "fixed_cell": fixed_cell_perf}, f)
        print(f"Metrics saved in {save_path}")
    else:
        print("Metrics not saved")


def multi_process_prediction(paths, save_metrics=False):
    '''
    Process the prediction file(s) and return the metrics for the predictions

    Args:
        paths (str): Path to the files or directories with the predictions
        save_metrics (bool): If True, the metrics will be saved in a file (useful to compute distribution)
    '''
    for path in paths:
        process_prediction(path, save_metrics)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze predictions')
    parser.add_argument('path', type=str, help='Path to the file or directory containing the predictions')
    parser.add_argument('--save_metrics', action='store_true', help='If True, the metrics will be saved in a file')
    args = parser.parse_args()
    process_prediction(args.path, args.save_metrics)
