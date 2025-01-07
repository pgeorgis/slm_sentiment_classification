"""Prompt template functions."""
from prompts.prompt_examples import EXAMPLE_POSITIVE_REVIEW, EXAMPLE_NEGATIVE_REVIEW
from prompts.prompt_texts import ZEROSHOT_PROMPT_BASE, CHAIN_OF_THOUGHT_PROMPT_BASE, RETURN_FORMAT
from prompts.system_messages import FILM_REVIEW_SUMMARIZER
from query_slm import Prompt, query_slm
from slm_models import qwen_15B

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


def extract_key_phrases_prompt(review_text):
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


def keyword_sentiment_analysis_prompt(review_text, model=qwen_15B):
    """Constructs a prompt to perform two-step sentiment analysis of a film review by first extracting key words and phrases."""
    # Embedded SLM prompt/call to first retrieve keywords/key phrases from film review
    keyword_prompt = Prompt(
        template=extract_key_phrases_prompt,
        system_message=FILM_REVIEW_SUMMARIZER,
        prompt_id="keyword extraction",
    )
    keyword_prompt.generate_prompt(review_text=review_text)
    key_phrases, _ = query_slm(
        model=model,
        prompt=keyword_prompt,
        max_tokens=200,
    )
    
    # Then construct prompt to classify sentiment of just the keywords/key phrases
    main_prompt = f"""Read the following keywords and/or key phrases taken from a film review and decide whether the overall review is positive or negative.
{RETURN_FORMAT}
    
```
{key_phrases}
```
"""
    return main_prompt
