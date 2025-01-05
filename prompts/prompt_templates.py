"""Prompt template functions."""
from prompts.prompt_examples import EXAMPLE_POSITIVE_REVIEW, EXAMPLE_NEGATIVE_REVIEW
from prompts.prompt_texts import ZEROSHOT_PROMPT_BASE, CHAIN_OF_THOUGHT_PROMPT_BASE


def zeroshot_review_classification(review_text):
    """Assembles zero-shot prompt for binary film review classification."""
    prompt = f"""{ZEROSHOT_PROMPT_BASE}

```
{review_text}
```
"""
    return prompt


def oneshot_review_classification(review_text):
    """Assembles one-shot prompt for binary film review classification."""
    prompt = f"""{ZEROSHOT_PROMPT_BASE}

Below are 2 examples of reviews with their classification.
Example review #1:
```
{EXAMPLE_POSITIVE_REVIEW}
```
Classification: "positive"


Example review #2:
```
{EXAMPLE_NEGATIVE_REVIEW}
```
Classification: "negative"


----REVIEW TO BE CLASSIFIED----
```
{review_text}
```
"""
    return prompt


def chain_of_thought_prompt(review_text):
    """Assembles a chain-of-thought style prompt for film review binary classification."""
    prompt = f"""{CHAIN_OF_THOUGHT_PROMPT_BASE}

```
{review_text}
```
"""
    return prompt

# To be added:
# - one/few-shot prompt
# - chain-of-thought
# - multiple steps: identify key words/phrases, then sentiment analysis from those