from prompts.prompt_templates import (
    chain_of_thought_instructions_prompt, chain_of_thought_traditional_prompt,
    chain_of_thought_with_likelihood_to_rewatch_prompt,
    chain_of_thought_with_numeric_ratings_prompt,
    fewshot_review_classification,
    fewshot_review_classification_with_similar_examples,
    keyword_sentiment_analysis_prompt, rating_based_sentiment_analysis_prompt,
    zeroshot_review_classification)

PROMPT_METHODS = {
    "zeroshot": zeroshot_review_classification,
    "fewshot": fewshot_review_classification,
    "fewshot-with-most-similar-examples": fewshot_review_classification_with_similar_examples,
    "chain-of-thought-instructions": chain_of_thought_instructions_prompt,
    "chain-of-thought-traditional": chain_of_thought_traditional_prompt,
    "chain-of-thought-with-numeric-rating": chain_of_thought_with_numeric_ratings_prompt,
    "chain-of-thought-rewatch-likelihood": chain_of_thought_with_likelihood_to_rewatch_prompt,
    "rating-based_sentiment-analysis": rating_based_sentiment_analysis_prompt,
    "keyword-based_sentiment_analysis": keyword_sentiment_analysis_prompt,
}

QUANTITATIVE_PROMPT_METHODS = {
    rating_based_sentiment_analysis_prompt,
    chain_of_thought_with_numeric_ratings_prompt,
    chain_of_thought_with_likelihood_to_rewatch_prompt,
}

FEWSHOT_PROMPT_METHODS = {
    fewshot_review_classification,
    fewshot_review_classification_with_similar_examples,
}
