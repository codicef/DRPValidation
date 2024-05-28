# DRPValidation
Drug Response Predictiors Validation Pipeline

## Overview
This tool evaluates the performance of Drug Response Predictors. It processes CSV files containing predictions for drugs and cell lines, calculating metrics such as RMSE, R^2, Spearman, and Pearson correlations.

The analysis follow the validation pipeline proposed in our paper ([link]), it analyse the performance globally, by drug and by cell line.

## Requirements
Requires Python 3 and libraries: pandas, numpy, scipy, scikit-learn. Install with pip:
```bash
pip3 install pandas numpy scipy scikit-learn
```

## Usage
```bash
python3 pipeline.py <path_to_predictions> [--save_metrics]
```
- `<path_to_predictions>`: Path to your CSV file(s).
- `--save_metrics`: Optional. Saves metrics to JSON.

### Example
```bash
python pipeline.py predictions/ --save_metrics
```
Processes all files in `predictions/` directory and saves metrics.

## License
This project is licensed under the MIT License.
