"""Prompt template functions."""
# pylama:ignore=E501
from pandas import DataFrame

from constants import (IMDB_NEGATIVE_LABEL, IMDB_POSITIVE_LABEL,
                       IMDB_REVIEW_LABEL_FIELD)
from prompts.prompt_examples import select_review_examples
from prompts.prompt_texts import (CHAIN_OF_THOUGHT_PROMPT_BASE,
                                  CHAIN_OF_THOUGHT_V2_PROMPT_BASE,
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
                                  ):
    """Assembles few-shot prompt for binary film review classification."""

    # Select N examples of both positive and negative reviews
    positive_example_pool = example_pool[example_pool[IMDB_REVIEW_LABEL_FIELD] == IMDB_POSITIVE_LABEL]
    negative_example_pool = example_pool[example_pool[IMDB_REVIEW_LABEL_FIELD] == IMDB_NEGATIVE_LABEL]
    selected_positive_examples = select_review_examples(positive_example_pool, n_examples)
    selected_negative_examples = select_review_examples(negative_example_pool, n_examples)

    # Start by introducing examples
    prompt = f"Carefully study the following {n_examples * 2} examples of film reviews with their respective classifications as either positive or negative.\n\n"
    for i, example in enumerate(selected_positive_examples):
        prompt += f"""*** Example film review #{i + 1} ***\n```\n{example}\n```\nClassification: "positive"\n\n"""
    for j, example in enumerate(selected_negative_examples):
        prompt += f"""*** Example film review #{i + j + 2} ***\n```\n{example}\n```\nClassification: "negative"\n\n"""

    # Add zero-shot prompt instructions
    prompt += zeroshot_review_classification(review_text)

    return prompt


def chain_of_thought_prompt(review_text: str):
    """Assembles a chain-of-thought style prompt for film review binary classification."""
    prompt = f"""{CHAIN_OF_THOUGHT_PROMPT_BASE}

```
{review_text}
```
"""
    return prompt

def chain_of_thought_v2_prompt(review_text: str):
    """Assembles a chain-of-thought style prompt for film review binary classification."""
    prompt = f"""{CHAIN_OF_THOUGHT_V2_PROMPT_BASE}

```
{review_text}
```
A: Let's think step by step.
"""
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
