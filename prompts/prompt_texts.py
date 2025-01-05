"""Text components of prompts shared across multiple prompts."""

ZEROSHOT_PROMPT_BASE = """Carefully read the following film review and decide whether the overall review is positive or negative.
Return only the labels "positive" or "negative". No further explanation is needed."""

CHAIN_OF_THOUGHT_PROMPT_BASE = """Carefully read the following film review and decide whether the overall review is positive or negative.
Let's think step-by-step:
- Identify the key words or phrases which reveal the author's attitude toward the film.
- Determine whether these attitudes are generally positive (e.g. impressed, pleased, moved, excited) or generally negative (e.g. bored, disgusted, disappointed, confused).
- Confirm that this positive or negative sentiment matches the overall tone of the review.
- Return only the labels "positive" or "negative". No further explanation is needed."""