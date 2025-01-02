"""Load and sample IMBD dataset."""

import numpy as np
import pandas as pd
import re

RANDOM_SEED = 32

VALID_SPLITS = {"train", "test"}

HTML_TAG_REGEX = re.compile(r"\s*<[^>]+>\s*")

def remove_html_tags(text: str):
    """Preprocess HTML tags out of a text."""
    return HTML_TAG_REGEX.sub(" ", text)


def get_wordcount(text: str):
    """Count number of whitespace-separated words in a text."""
    return len(text.split())


def reverse_binary_label(label):
    """Reverse binary labels 0 and 1."""
    label = int(label)
    return abs(label - 1)


def load_imdb(split: str="train"):
    """Load a particular split (train or test) of IMDB dataset."""
    # Check split label is valid
    if split not in VALID_SPLITS:
        raise ValueError(f"Unexpected split value '{split}', expected one of {VALID_SPLITS}")

    # Load to dataframe
    df = pd.read_json(f"hf://datasets/ajaykarthick/imdb-movie-reviews/{split}.jsonl", lines=True)

    # Preprocess by removing HTML tags from review texts
    df["review"] = df["review"].apply(remove_html_tags)
    
    # Reverse labels: IMDB dataset originally has 0 as positive and 1 as negative
    # Instead use 1 as positive and 0 as negative as this is more intuitive
    df["label"] = df["label"].apply(reverse_binary_label)

    return df


def sample_from_imdb(imdb_df: pd.DataFrame, examples_per_class: int=100, length_brackets: int=10):
    """Sample equally from IMDB dataset by label and proportionately by text length."""
    # Add indices
    selected_indices = {0: set(), 1: set()}
    imdb_df["index"] = range(len(imdb_df))
    # Calculate text lengths and sort into brackets
    imdb_df["wordcount"] = imdb_df["review"].apply(get_wordcount)
    max_length = imdb_df["wordcount"].max()
    length_bins = np.linspace(0, max_length, length_brackets + 1)
    imdb_df["length_bracket"] = pd.cut(imdb_df["wordcount"], bins=length_bins, labels=False, include_lowest=True)

    # Draw N examples from each label/class, proportionate to text length bracket frequencies
    for label in [0, 1]:
        # Filter reviews of the current label
        label_df = imdb_df[imdb_df["label"] == label]
        # Count the frequency of each length bracket in this label
        length_bracket_counts = label_df["length_bracket"].value_counts(normalize=True)

        # Sample based on the frequency distribution of length brackets
        for bracket, freq in length_bracket_counts.items():
            bracket_df = label_df[label_df["length_bracket"] == bracket]

            # Adjust the number of samples for this bracket, according to its frequency
            bracket_sample_size = int(examples_per_class * freq)

            # Sample reviews from this bracket and add their indices to sample_data
            sampled_reviews = bracket_df.sample(n=bracket_sample_size, replace=False, random_state=RANDOM_SEED)
            selected_indices[label].update(sampled_reviews["index"].tolist())

            # Stop once the minimum required examples are collected for each class
            if len(selected_indices[0]) >= examples_per_class and len(selected_indices[1]) >= examples_per_class:
                break
    
    # Filter dataframe to only selected indices
    all_selected_indices = selected_indices[0].union(selected_indices[1])
    imdb_sample = imdb_df[imdb_df["index"].isin(all_selected_indices)]
    
    return imdb_sample
