"""Initializes constants."""
from utils import get_git_commit_hash

COMMIT_HASH = get_git_commit_hash()

RANDOM_SEED = 32

# IMDB dataset field labels
IMDB_REVIEW_TEXT_FIELD = "review"
IMDB_REVIEW_LABEL_FIELD = "label"
IMDB_INDEX_FIELD = "index"
IMDB_POSITIVE_LABEL = 1
IMDB_NEGATIVE_LABEL = 0
# NB: original IMDB dataset has 0 as positive label and 1 as negative label;
# this is reversed to be more intuitive when loading the dataset
BINARY_LABEL_MAP = {
    "positive": IMDB_POSITIVE_LABEL,
    "negative": IMDB_NEGATIVE_LABEL,
}
