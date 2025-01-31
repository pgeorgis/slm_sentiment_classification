"""Example texts for use in prompts."""
# pylama:ignore=E501
import random
from functools import lru_cache

from pandas import DataFrame

from constants import IMDB_INDEX_FIELD, IMDB_REVIEW_TEXT_FIELD, RANDOM_SEED
from embedding import (EMBEDDING_ID_MAP_FILE, EMBEDDING_INDEX_FILE,
                       EMBEDDING_MODEL, load_faiss_index,
                       retrieve_most_similar_texts)
from load_imdb_data import split_positive_negative_reviews

EMBEDDING_INDEX, EMBEDDING_ID_MAP = load_faiss_index(EMBEDDING_INDEX_FILE, EMBEDDING_ID_MAP_FILE)

EXAMPLE_REVIEW_001 = """I was really impressed with this film. 
The writing was fantastic, and the characters were all rich, and simple. 
It's very easy to get emotionally attached to all of them. 
The creators of this movie really hit the nail right on the head when it comes to creating real life characters, and getting the viewer sucked right into their world. 
Further, the music is terrific. 
They employed some independents to do the score, and some of the soundtrack, and they do a fantastic job adding to the movie.
If you have a chance to catch this movie in a small theater or at a film festival (like I did), I highly recommend that you go see it.
Also, on a personal note, Paget Brewster is beautiful in this movie. That's reason enough to go check it out."""

EXAMPLE_REVIEW_002 = """The beginning was decent; the ending was alright. 
Sadly, this movie suffers from complete and utter shallowness of the characters, unrealistic confrontations/fight scenes, lack of anyone intelligent outside of the shuttle.
This makes for an awful middle screenplay.
Stuff to look for: overly obvious foreshadowing, fast-healing cuts, overly smoky fires, fun seatbelts, delayed reactions.
I did give it a 4, not a 0, because the start of the movie had some nice elements of happiness and basic character development.
The relationship between the main, dark-haired girl and her fiancée is touched upon briefly, and the placement of the blond friend's impact on that relationship is present, though awkwardly so.
The business discovered at the end is becoming more mainstream and decently done, though, as another commenter pointed out, not unexpected.  ~viper~"""

EXAMPLE_REVIEW_003 = """This film is absolute trash -- in the best way possible. It’s so unapologetically camp that it borders on iconic.
The writing and acting are hilariously bad; I was laughing so hard I nearly cried.
Honestly, the plot was so predictable and dull that you could zone out completely and still not miss a thing.
And yet... I’d watch it again in a heartbeat. It’s the perfect movie to throw on with friends, just to roast every absurd, cringe-worthy moment.
If you love a good "so-bad-it's-good" experience, this is the ultimate guilty pleasure."""

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


def select_positive_and_negative_examples(example_pool, n_examples):
    """Select N examples of both positive and negative reviews."""
    positive_example_pool, negative_example_pool = split_positive_negative_reviews(example_pool)
    selected_positive_examples = select_review_examples(positive_example_pool, n_examples)
    selected_negative_examples = select_review_examples(negative_example_pool, n_examples)
    return selected_positive_examples, selected_negative_examples


def select_positive_and_negative_reviews_by_embedding_similarity(review_text: str,
                                                                 example_pool: DataFrame,
                                                                 n_examples: int = 1,
                                                                 ):
    """Select N most similar positive and negative examples to a given review text using
    embedding similarity search wrt training set review texts."""
    positive_reviews, negative_reviews = split_positive_negative_reviews(example_pool)
    positive_indices = set(positive_reviews[IMDB_INDEX_FIELD].to_list())
    negative_indices = set(negative_reviews[IMDB_INDEX_FIELD].to_list())
    selected_positive_reviews = retrieve_most_similar_texts(
        query_text=review_text,
        embedding_model=EMBEDDING_MODEL,
        index=EMBEDDING_INDEX,
        id_map=EMBEDDING_ID_MAP,
        top_n=n_examples,
        valid_ids=positive_indices,
    )
    selected_negative_reviews = retrieve_most_similar_texts(
        query_text=review_text,
        embedding_model=EMBEDDING_MODEL,
        index=EMBEDDING_INDEX,
        id_map=EMBEDDING_ID_MAP,
        top_n=n_examples,
        valid_ids=negative_indices,
    )
    # Extract indices
    selected_positive_indices = [idx for idx, _ in selected_positive_reviews]
    selected_negative_indices = [idx for idx, _ in selected_negative_reviews]
    
    # Map indices to review texts
    selected_positive_reviews = positive_reviews.loc[
        positive_reviews[IMDB_INDEX_FIELD].isin(selected_positive_indices), 
        IMDB_REVIEW_TEXT_FIELD
    ].tolist()
    selected_negative_reviews = negative_reviews.loc[
        negative_reviews[IMDB_INDEX_FIELD].isin(selected_negative_indices), 
        IMDB_REVIEW_TEXT_FIELD
    ].tolist()
    
    # Return lists of review texts
    return selected_positive_reviews, selected_negative_reviews

