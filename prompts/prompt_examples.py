"""Example texts for use in prompts."""
# pylama:ignore=E501
import random
from functools import lru_cache

from pandas import DataFrame

from constants import IMDB_INDEX_FIELD, IMDB_REVIEW_TEXT_FIELD, RANDOM_SEED

EXAMPLE_POSITIVE_REVIEW = """"The Lost Journey" is a masterful piece of storytelling that captivates the audience from start to finish. The film beautifully balances breathtaking visuals with a deeply emotional narrative. The protagonist’s transformation is both inspiring and relatable, as they confront their fears and discover their true strength. The cinematography is stunning, with sweeping shots of the wilderness that immerse you in the story’s world. The score is hauntingly beautiful, adding layers of depth to the already compelling plot. This is a movie that stays with you long after the credits roll, sparking conversations about resilience and self-discovery. Highly recommended for anyone seeking a heartfelt and visually stunning cinematic experience."""

EXAMPLE_NEGATIVE_REVIEW = """"The Lost Journey" tries hard to be profound but falls flat with its overly sentimental and predictable storyline. The pacing is excruciatingly slow, with long, drawn-out scenes that add little to the narrative. While the cinematography is visually impressive, it feels like style over substance, as the film struggles to deliver any real emotional impact. The protagonist’s journey feels forced and lacks authenticity, making it difficult to connect with their struggles. The dialogue is often cliché, and the supporting characters are one-dimensional. Despite its ambition, the film ultimately feels hollow and fails to live up to its potential. Save your time and skip this one."""


@lru_cache(maxsize=None)
def select_indices(n: int, max_n: int, seed=RANDOM_SEED):
    """Select N indices from a range from 0 until max_n."""
    random.seed(seed)
    return random.sample(range(max_n), n)


@lru_cache(maxsize=None)
def select_example_indices(indices: tuple[int], n: int):
    """Select N example indices."""
    selected_indices = select_indices(n=n, max_n=len(indices) + 1)
    return [indices[i] for i in selected_indices]


def select_review_examples(example_pool: DataFrame, n_examples: int):
    """Select N film review examples from by index field."""
    indices = sorted(example_pool[IMDB_INDEX_FIELD].to_list())
    selected_example_indices = select_example_indices(tuple(indices), n_examples)
    selected_examples = example_pool[example_pool[IMDB_INDEX_FIELD].isin(selected_example_indices)]
    return selected_examples[IMDB_REVIEW_TEXT_FIELD].to_list()
