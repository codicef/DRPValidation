#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
import sys
import argparse


def convert(path, conv_path):
    for f_name in tqdm(os.listdir(path)):
        obj = pickle.load(open(f"{path}/{f_name}", "rb"))

        test_p = obj[1]

        if len(test_p[0]) > 2:
            cells, drugs = list(zip(*test_p[0]))
        else:
            cells, drugs = test_p[0]
        true_values = test_p[1]
        predicted_values = test_p[2]

        df = pd.DataFrame({
            'cell': cells,
            'drug': drugs,
            'true_value': true_values,
            'predicted_value': predicted_values
        })
        if not os.path.exists(conv_path):
            os.makedirs(conv_path)

        df.to_csv(f"{conv_path}/{f_name}.csv", index=False)


if __name__ == "__main__":
    # Usage: python convert.py path conv_path
    parser = argparse.ArgumentParser(description='Convert pickle files to csv files')
    parser.add_argument('path', type=str, help='Path to the directory containing the pickle files')
    parser.add_argument('conv_path', type=str, help='Path to the directory where the csv files will be saved')
    args = parser.parse_args()
    convert(args.path, args.conv_path)
