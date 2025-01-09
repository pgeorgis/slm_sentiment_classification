"""Initializes constants."""
import logging

from slm_models import get_qwen05B, get_qwen15B
from utils import get_git_commit_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COMMIT_HASH = get_git_commit_hash()

RANDOM_SEED = 32
FEWSHOT_EXAMPLE_N = 3

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


# Initialize SLM models
logger.info("Initializing SLM Qwen models...")
qwen_05B = get_qwen05B()
qwen_15B = get_qwen15B()
DEFAULT_MODELS = [qwen_05B, qwen_15B]
