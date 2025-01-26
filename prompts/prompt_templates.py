"""Prompt template functions."""
# pylama:ignore=E501
from pandas import DataFrame

from prompts.prompt_examples import (
    select_positive_and_negative_examples,
    select_positive_and_negative_reviews_by_embedding_similarity)
from prompts.prompt_texts import (CHAIN_OF_THOUGHT_INSTRUCTIONS_BASE,
                                  CHAIN_OF_THOUGHT_TRADITIONAL_BASE,
                                  CHAIN_OF_THOUGHT_WITH_LIKELIHOOD_TO_REWATCH,
                                  CHAIN_OF_THOUGHT_WITH_NUMERIC_RATINGS_BASE,
                                  RETURN_FORMAT, ZEROSHOT_PROMPT_BASE)


def zeroshot_review_classification(review_text: str):
    """Assembles zero-shot prompt for binary film review classification."""
    prompt = f"""{ZEROSHOT_PROMPT_BASE}

```
{review_text}
```
"""
    return prompt


def fewshot_review_classification(review_text: str,
                                  example_pool: DataFrame,
                                  n_examples: int = 1,
                                  selection_method: str = "random",
                                  ):
    """Assembles few-shot prompt for binary film review classification."""
    if selection_method not in {"random", "embedding_similarity"}:
        raise ValueError(f"Unexpected selection method '{selection_method}'")

    # Select N examples of both positive and negative reviews
    # Faiss embedding similarity-based example selection
    if selection_method == "embedding_similarity":
        selected_positive_examples, selected_negative_examples = select_positive_and_negative_reviews_by_embedding_similarity(
            review_text=review_text,
            example_pool=example_pool,
            n_examples=n_examples,
        )
    else:  # Random example selection
        selected_positive_examples, selected_negative_examples = select_positive_and_negative_examples(
            example_pool=example_pool,
            n_examples=n_examples,
        )
    
    # Initialize prompt text
    prompt = ""

    # Add examples
    for i, example in enumerate(selected_positive_examples):
        prompt += f"""\nFilm Review:\n```\n{example}\n```\nSentiment: positive\n\n"""
    for j, example in enumerate(selected_negative_examples):
        prompt += f"""\nFilm Review:```\n{example}\n```\nSentiment: negative\n\n"""

    # Add current review to classify
    prompt += f"\nFilm Review:```\n{review_text}\n```Sentiment: "

    return prompt


def fewshot_review_classification_with_similar_examples(review_text: str,
                                                        example_pool: DataFrame,
                                                        n_examples: int = 1,
                                                        ):
    """Construct fewshot prompt with most similar positive and negative train examplesto review in question."""
    return fewshot_review_classification(
        review_text=review_text,
        example_pool=example_pool,
        n_examples=n_examples,
        selection_method="embedding_similarity",
    )


def chain_of_thought_instructions_prompt(review_text: str):
    """Assembles an instruction-based chain-of-thought style prompt for film review binary classification."""
    prompt = f"""{CHAIN_OF_THOUGHT_INSTRUCTIONS_BASE}

```
{review_text}
```
"""
    return prompt

def chain_of_thought_traditional_prompt(review_text: str):
    """Assembles a traditional chain-of-thought style prompt for film review binary classification."""
    prompt = f"""{CHAIN_OF_THOUGHT_TRADITIONAL_BASE}

```
{review_text}
```
A: Let's think step by step.
"""
    return prompt


def chain_of_thought_with_numeric_ratings_prompt(review_text: str):
    """Assembles a chain-of-thought style prompt for obtaining numeric ratings of film reviews."""
    prompt = f"""{CHAIN_OF_THOUGHT_WITH_NUMERIC_RATINGS_BASE}

```
{review_text}
```
A: """
    return prompt


def chain_of_thought_with_likelihood_to_rewatch_prompt(review_text: str):
    """Assembles a chain-of-thought style prompt for obtaining likelihood that the author would choose to rewatch the film."""
    prompt = f"""{CHAIN_OF_THOUGHT_WITH_LIKELIHOOD_TO_REWATCH}

```
{review_text}
```
A: """
    return prompt


def extract_key_phrases_prompt(review_text: str):
    """Constructs a prompt to extract key words and phrases from a film review."""
    prompt = f"""Carefully read the following film review and identify any keywords or key phrases which reveal the author's attitude toward the film.
In particular, look for keywords and phrases which reveal:
- the author's opinion about the film
- the author's emotional reaction to the film, or how the film made the author feel
- criticism or praise of the film

Return a list of relevant keywords or key phrases from the film review. No further explanation is needed.

```
{review_text}
```
"""
    return prompt


def keyword_sentiment_analysis_prompt(key_phrases: str):
    """Constructs a prompt to perform keyword-based sentiment analysis of a film review."""
    main_prompt = f"""Carefully read the following keywords and/or key phrases taken from a film review and decide whether the overall review is positive or negative.
{RETURN_FORMAT}

```
{key_phrases}
```
"""
    return main_prompt


def rating_based_sentiment_analysis_prompt(review_text: str):
    """Constructs a prompt to perform sentiment analysis based on a film review's estimated rating."""
    prompt = f"""Carefully read the following film review and estimate the rating on a scale from 1 to 10 that the author would give to the film or to their overall experience.
Return only the estimated rating as an integer. No further explanation is needed.

```
{review_text}
```
"""
    return prompt
