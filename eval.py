"""Inference evaluation tools."""
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