"""Text components of prompts shared across multiple prompts."""
# pylama:ignore=E501

RETURN_FORMAT = """Return only the labels "positive" or "negative". No further explanation is needed."""

ZEROSHOT_PROMPT_BASE = f"""Carefully read the following film review and decide whether the overall review is positive or negative.
{RETURN_FORMAT}"""

CHAIN_OF_THOUGHT_PROMPT_BASE = f"""Carefully read the following film review and decide whether the overall review is positive or negative.
Let's think step-by-step:
- Identify the key words or phrases which reveal the author's attitude toward the film.
- Determine whether these attitudes are generally positive (e.g. impressed, pleased, moved, excited) or generally negative (e.g. bored, disgusted, disappointed, confused).
- Confirm that this positive or negative sentiment matches the overall tone of the review.
{RETURN_FORMAT}"""

CHAIN_OF_THOUGHT_V2_PROMPT_BASE = f"""
Q: Is the overall sentiment of the following film review positive or negative?

```
I was really impressed with this film. 
The writing was fantastic, and the characters were all rich, and simple. 
It's very easy to get emotionally attached to all of them. 
The creators of this movie really hit the nail right on the head when it comes to creating real life characters, and getting the viewer sucked right into their world. 
Further, the music is terrific. 
They employed some independents to do the score, and some of the soundtrack, and they do a fantastic job adding to the movie.
If you have a chance to catch this movie in a small theater or at a film festival (like I did), I highly recommend that you go see it.
Also, on a personal note, Paget Brewster is beautiful in this movie. That's reason enough to go check it out.
```
A: The author of the film review expresses praise of film's writing, characters, and music.
The author found the characters realistic and the plot engaging, and they recommend seeing the movie.
The overall sentiment of the review is positive.


Q: Is the overall sentiment of the following film review positive or negative?
```
The beginning was decent; the ending was alright. 
Sadly, this movie suffers from complete and utter shallowness of the characters, unrealistic confrontations/fight scenes, lack of anyone intelligent outside of the shuttle.
This makes for an awful middle screenplay.
Stuff to look for: overly obvious foreshadowing, fast-healing cuts, overly smoky fires, fun seatbelts, delayed reactions.
I did give it a 4, not a 0, because the start of the movie had some nice elements of happiness and basic character development.
The relationship between the main, dark-haired girl and her fiancée is touched upon briefly, and the placement of the blond friend's impact on that relationship is present, though awkwardly so.
The business discovered at the end is becoming more mainstream and decently done, though, as another commenter pointed out, not unexpected.  ~viper~
```
A: Although the author appreciated the start of the film, they criticized the film's shallow characters, unrealistic scenes, and vapid plot.
The author found the screenplay disappointing and pointed out specific examples of poor writing and cinematography.
The overall sentiment of the review is negative.


Q: Is the overall sentiment of the following film review positive or negative?
"""
