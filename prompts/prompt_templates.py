"""Prompt template functions."""
from prompts.prompt_examples import EXAMPLE_POSITIVE_REVIEW, EXAMPLE_NEGATIVE_REVIEW
from prompts.prompt_texts import ZEROSHOT_PROMPT_BASE, CHAIN_OF_THOUGHT_PROMPT_BASE, RETURN_FORMAT

def zeroshot_review_classification(review_text: str):
    """Assembles zero-shot prompt for binary film review classification."""
    prompt = f"""{ZEROSHOT_PROMPT_BASE}

```
{review_text}
```
"""
    return prompt


def oneshot_review_classification(review_text: str):
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


def chain_of_thought_prompt(review_text: str):
    """Assembles a chain-of-thought style prompt for film review binary classification."""
    prompt = f"""{CHAIN_OF_THOUGHT_PROMPT_BASE}

```
{review_text}
```
"""
    return prompt


def extract_key_phrases_prompt(review_text: str):
    """Constructs a prompt to extract key words and phrases from a film review."""
    prompt = f"""Read the following film review and identify any keywords or key phrases which reveal the author's attitude toward the film.
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
    main_prompt = f"""Read the following keywords and/or key phrases taken from a film review and decide whether the overall review is positive or negative.
{RETURN_FORMAT}
    
```
{key_phrases}
```
"""
    return main_prompt
