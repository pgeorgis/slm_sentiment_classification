"""Text components of prompts shared across multiple prompts."""
# pylama:ignore=E501
from prompts.prompt_examples import (EXAMPLE_REVIEW_001, EXAMPLE_REVIEW_002,
                                     EXAMPLE_REVIEW_003)

RETURN_FORMAT = """Return only the labels "positive" or "negative". No further explanation is needed."""

ZEROSHOT_PROMPT_BASE = f"""Carefully read the following film review and decide whether the overall review is positive or negative.
{RETURN_FORMAT}"""

CHAIN_OF_THOUGHT_PROMPT_BASE = f"""Carefully read the following film review and decide whether the overall review is positive or negative.
Let's think step-by-step:
- Identify the key words or phrases which reveal the author's attitude toward the film.
- Determine whether these attitudes are generally positive (e.g. impressed, pleased, moved, excited) or generally negative (e.g. bored, disgusted, disappointed, confused).
- Confirm that this positive or negative sentiment matches the overall tone of the review.
{RETURN_FORMAT}"""

FILM_REVIEW_TIPS = """Consider the following when analyzing film reviews:
- What emotions does the author express toward the film? For example, is the author impressed, pleased, moved, or excited (positive emotions)? Or is the author bored, disgusted, disappointed, or confused (negative emotions)?
- Consider the author's description of the film's writing, plot, acting, cinematography, and other elements. Does the author praise or criticize these aspects? How does this contribute to the overall sentiment of the review?
- Did the author enjoy the film overrall, even if they had some criticisms? This can indicate a generally positive sentiment. Remember that a film does not need to be high quality to be enjoyable.
- Consider the author's tone. Does the author use sarcasm or hyperbole for comedic effect? If so, consider how this affects the overall attitude of the review.
- Would the author recommend the film to others, even if not to everyone? This can indicate a positive sentiment.
- Does the author mention an explicit star rating out of 10? Ratings >5 are positive and <5 are negative.
"""

CHAIN_OF_THOUGHT_V2_PROMPT_BASE = f"""{FILM_REVIEW_TIPS}

Q: Is the overall sentiment of the following film review positive or negative?

```
{EXAMPLE_REVIEW_001}
```
A: Let's think step by step. The author of the film review expresses praise of film's writing, characters, and music.
The author found the characters realistic and the plot engaging, and they recommend seeing the movie.
The overall sentiment of the review is positive.


Q: Is the overall sentiment of the following film review positive or negative?
```
{EXAMPLE_REVIEW_002}
```
A: Let's think step by step. Although the author appreciated the start of the film, they criticized the film's shallow characters, unrealistic scenes, and vapid plot.
The author found the screenplay disappointing and pointed out specific examples of poor writing and cinematography.
The overall sentiment of the review is negative.


Q: Is the overall sentiment of the following film review positive or negative?
```
{EXAMPLE_REVIEW_003}
```
A: Let's think step by step. Even though the author pointed out the film's poor writing, acting, and predictable plot, they enjoyed the film for its campy, absurd qualities.
The author found the movie entertaining for precisely these flaws and would watch it again with friends for a laugh.
The overall sentiment of the review is positive.


Q: Is the overall sentiment of the following film review positive or negative?
"""

 #TODO add tips back?
CHAIN_OF_THOUGHT_WITH_NUMERIC_RATINGS_BASE = f"""

Q: How would the author of the following review rate the film they watched on a scale from 1 to 10?Return your estimate in json format with "analysis" and "ratings" sections.

```
{EXAMPLE_REVIEW_001}
```
A:
{{
  "analysis": "The author of the film review expresses praise of film's writing, characters, and music. The author found the characters realistic and the plot engaging, and they recommend seeing the movie.",
  "rating": 10
}}


Q: How would the author of the following review rate the film they watched on a scale from 1 to 10? Return your estimate in json format with "analysis" and "ratings" sections.
```
{EXAMPLE_REVIEW_002}
```
A: 
{{
  "analysis": "Although the author appreciated the start of the film, they criticized the film's shallow characters, unrealistic scenes, and vapid plot. The author found the screenplay disappointing and pointed out specific examples of poor writing and cinematography.",
  "rating": 4
}}

Q: How would the author of the following review rate the film they watched on a scale from 1 to 10? Return your estimate in json format with "analysis" and "ratings" sections.
```
{EXAMPLE_REVIEW_003}
```
A: 
{{
  "analysis": "Even though the author pointed out the film's poor writing, acting, and predictable plot, they enjoyed the film for its campy, absurd qualities. The author found the movie entertaining for precisely these flaws and would watch it again with friends for a laugh.",
  "rating": 7
}}

Q: How would the author of the following review rate the film they watched on a scale from 1 to 10? Return your estimate in json format with "analysis" and "ratings" sections.
"""
