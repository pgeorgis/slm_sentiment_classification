"""Inference evaluation and visualization tools."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score as calculate_f1

# NB: original IMDB dataset has 0 as positive label and 1 as negative label;
# this is reversed to be more intuitive when loading the dataset
BINARY_LABEL_MAP = {
    "positive": 1,
    "negative": 0,
}


def binary_eval(ref, hyp, label_map=BINARY_LABEL_MAP):
    """Return either TP [true positive], FP [false positive], TN [true negative], FN [false negative] based on binary classification 0 and 1."""
    if ref not in {0, 1} and not label_map.get(ref):
        raise ValueError(f"Invalid binary reference '{ref}'")
    if hyp not in {0, 1} and not label_map.get(hyp):
        raise ValueError(f"Invalid binary prediction '{hyp}'")
    hyp = BINARY_LABEL_MAP.get(hyp, hyp)
    if ref == 1 and hyp == 1:
        return "TP"
    elif ref == 0 and hyp == 0:
        return "TN"
    elif hyp == 1:
        return "FP"
    else:  # hyp == 0
        return "FN"


def create_tfpn_histogram_by_wordcount(results_df: pd.DataFrame,
                                       categorical_column: str,
                                       show_plot: bool=False,
                                       outfile: str=None):
    """Create histogram of TP/TN/FP/FN results according to word count of film review text."""
    unique_categories = sorted(results_df[categorical_column].unique())
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=results_df,
        x='wordcount',
        hue=categorical_column,
        hue_order=unique_categories,
        kde=False,
        bins=10,
        palette='Set2',
        multiple="stack"
    )
    plt.title(f"Classification Result by Word Count of Film Review")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    if outfile:
        outdir = os.path.dirname(outfile)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(outfile)
    if show_plot:
        plt.show()


def plot_confusion_matrix(results_df: pd.DataFrame,
                          results_column: str,
                          outfile: str = None,
                          show_plot: bool=False):
    """Plot a confusion matrix."""
    results = results_df[results_column].value_counts(normalize=False)
    results = results.to_dict()
    confusion_matrix = np.array(
        [
            [results.get("TN", 0), results.get("FP", 0)],
            [results.get("FN", 0), results.get("TP", 0)]
        ]
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    if show_plot:
        plt.show()
    if outfile:
        outdir = os.path.dirname(outfile)
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(outfile)
