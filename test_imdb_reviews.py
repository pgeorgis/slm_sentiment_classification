import argparse
import os
import re
from collections import defaultdict
from functools import lru_cache
from statistics import mean
from typing import Callable, Union

import pandas as pd
from llama_cpp import Llama
from sklearn.metrics import f1_score as calculate_f1

from constants import (COMMIT_HASH, DEFAULT_MODELS, FEWSHOT_EXAMPLE_N,
                       IMDB_REVIEW_LABEL_FIELD, IMDB_REVIEW_TEXT_FIELD, logger,
                       qwen_15B)
from eval import (BINARY_LABEL_MAP, binary_eval,
                  create_tfpn_histogram_by_wordcount, plot_confusion_matrix,
                  plot_f1_bar_graph, plot_f1_latency_scatterplot)
from load_imdb_data import load_imdb, sample_from_imdb
from prompts.prompt_templates import (
    chain_of_thought_prompt, chain_of_thought_v2_prompt,
    chain_of_thought_with_likelihood_to_rewatch_prompt,
    chain_of_thought_with_numeric_ratings_prompt, extract_key_phrases_prompt,
    fewshot_review_classification,
    fewshot_review_classification_with_similar_examples,
    keyword_sentiment_analysis_prompt, rating_based_sentiment_analysis_prompt,
    zeroshot_review_classification)
from prompts.system_messages import (FILM_REVIEW_CLASSIFIER,
                                     FILM_REVIEW_SUMMARIZER)
from query_slm import Prompt, query_slm
from utils import create_datestamp, create_timestamp

VALID_REVIEW_LABELS = {"positive", "negative"}
VALID_REVIEW_REGEX = re.compile("|".join(VALID_REVIEW_LABELS))

PROMPT_METHODS = {
    "zeroshot": zeroshot_review_classification,
    "fewshot": fewshot_review_classification,
    "fewshot_by_similarity": fewshot_review_classification_with_similar_examples,
    "chain-of-thought": chain_of_thought_prompt,
    "keyword-based_sentiment_analysis": keyword_sentiment_analysis_prompt,
    "chain-of-thought-v2": chain_of_thought_v2_prompt,
    "chain-of-thought-with-numeric-rating": chain_of_thought_with_numeric_ratings_prompt,
    "chain-of-thought-rewatch-likelihood": chain_of_thought_with_likelihood_to_rewatch_prompt,
    "rating-based-sentiment-analysis": rating_based_sentiment_analysis_prompt,
}


def create_run_outdir(test_label=None):
    """Create an output directory for a single test run."""
    today = create_datestamp()
    timestamp = create_timestamp()
    if test_label is None:
        test_label = ""
    results_outdir = os.path.abspath("results")
    run_outdir = os.path.join(results_outdir, today, test_label, COMMIT_HASH, timestamp)
    os.makedirs(run_outdir, exist_ok=True)
    return run_outdir


@lru_cache(maxsize=None)
def extract_review_keywords(review_text: str,
                            model: Llama = qwen_15B
                            ):
    """Extract or retrieve keywords/key phrases from a film review text."""
    keyword_prompt = Prompt(
        template=extract_key_phrases_prompt,
        system_message=FILM_REVIEW_SUMMARIZER,
        prompt_id="keyword extraction",
    )
    keyword_prompt.generate_prompt(review_text=review_text)
    key_phrases, details = query_slm(
        model=model,
        prompt=keyword_prompt,
        max_tokens=200,
    )
    return key_phrases, details


def binary_classify_rating(rating: int, positive_threshold: int = 5):
    """Classify a numeric rating as 'positive' or 'negative'."""
    if rating >= positive_threshold:
        return "positive"
    else:
        return "negative"


def extract_rating_from_json(prediction_json: str):
    """
    Extracts the rating value from a JSON string.

    This function takes a JSON string, removes any leading or trailing 
    markdown code block delimiters (```json and ```), parses the string 
    into a dictionary, and then attempts to extract the value associated 
    with the keys "rating" or "ratings".

    Args:
        prediction_json (str): A JSON string potentially containing a 
                               "rating" or "ratings" key.

    Returns:
        int or float or None: The value associated with the "rating" or 
                              "ratings" key if found, otherwise None.
    """
    prediction_json = re.sub(r'\n', r' ', prediction_json, re.DOTALL)
    prediction_json = re.sub(r'\s+', r' ', prediction_json, re.DOTALL)
    prediction = re.search(r'(ratings?|likelihood.*)":\s*"?(\-?\d+)', prediction_json)
    if prediction:
        return prediction.group(2)
    return None


def postprocess_predicted_rating(predicted_rating: str, from_json=False, positive_threshold: int = 5) -> str|None:
    """Postprocess a predicted rating to classify as 'positive' or 'negative'.

    Args:
        predicted_rating (str): Predicted rating of the film review on a scale from 1 to 10.
        from_json (bool, optional): Whether to extract the predicted rating from a json object or from raw text.
        positive_threshold (int, optional): Threshold for positive rating. Defaults to 5.

    Returns:
        str|None: 'positive' or 'negative' classification of the review. None is returned if no numeric rating is found.
    """
    if from_json:
        match_rating = extract_rating_from_json(predicted_rating)
    else:
        match_rating = re.search(r"\d+", predicted_rating)
        if match_rating:
            match_rating = match_rating.group()
    if match_rating:
        numeric_rating = int(match_rating)
        binary_rating = binary_classify_rating(numeric_rating, positive_threshold)
        return binary_rating, numeric_rating
    else:
        logger.warning(f"Unable to extract rating from prediction: {predicted_rating}")
        return None, None


def classify_imdb_review(review_text: str,
                         model: Llama,
                         prompt_template: Union[Callable, str],
                         prompt_label: str = None,
                         example_pool: pd.DataFrame = None,
                         system_message: str = FILM_REVIEW_CLASSIFIER,
                         model_params: dict = None):
    """Classify an IMDB review as a 'positive' or 'negative' review."""
    prompt = Prompt(
        prompt_template,
        system_message=system_message,
        prompt_id=prompt_label,
    )
    # Check if using keyword-based sentiment analysis; if so, first extract keywords
    use_keywords = prompt_template == keyword_sentiment_analysis_prompt
    if use_keywords:
        key_phrases, key_phrases_call_details = extract_review_keywords(review_text)
        prompt.generate_prompt(key_phrases=key_phrases)
    elif prompt_template in {
        fewshot_review_classification,
        fewshot_review_classification_with_similar_examples
    }:
        prompt.generate_prompt(
            review_text=review_text,
            example_pool=example_pool,
            n_examples=FEWSHOT_EXAMPLE_N
        )
    else:
        prompt.generate_prompt(review_text=review_text)
    if not model_params:
        model_params = {}
    if prompt_template in {
        rating_based_sentiment_analysis_prompt,
        chain_of_thought_with_numeric_ratings_prompt,
        chain_of_thought_with_likelihood_to_rewatch_prompt,
    }:
        if prompt_template == rating_based_sentiment_analysis_prompt:
            from_json = False
        else:
            from_json = True
        ratings = []
        n_calls = 3
        details = defaultdict(lambda: 0)
        for i in range(n_calls):
            rating_model_params = {"temperature": 0.4}
            prediction, details_i = query_slm(model, prompt, **rating_model_params)
            prediction, rating = postprocess_predicted_rating(str(prediction), from_json=from_json)
            if rating is not None:
                ratings.append(rating)
            # Get token usage and latency from each call
            for field in details_i:
                if field in {"prompt_tokens", "completion_tokens", "total_tokens", "latency"}:
                    details[field] += details_i[field]
                else:
                    details[field] = details_i[field]
        # Get average of token usage and latency
        for field in {"prompt_tokens", "completion_tokens", "total_tokens", "latency"}:
            details[field] /= n_calls
        if len(ratings) > 0:
            rating = mean(ratings)
            prediction = binary_classify_rating(rating)
        else:
            rating, prediction = None, None
    else:
        rating = None
        prediction, details = query_slm(model, prompt, **model_params)
        prediction = prediction.strip().lower()
        match_prediction = VALID_REVIEW_REGEX.search(prediction)
        if not match_prediction:
            logger.warning(f"Unexpected response: {prediction}")
            prediction = None
        else:
            prediction = match_prediction.group()
    prediction = BINARY_LABEL_MAP.get(prediction, prediction)

    # If keyword extraction was applied,
    # add latency and token usage details from that call to overall details
    if use_keywords:
        details["latency"] += key_phrases_call_details["latency"]
        for key in {"prompt_tokens", "completion_tokens", "total_tokens"}:
            details["usage"][key] += key_phrases_call_details["usage"][key]
    else:
        key_phrases = None

    return prediction, details, key_phrases, rating


def test_prompt(test_data: pd.DataFrame,
                prompt_template: Union[Callable, str],
                prompt_label: str,
                example_pool: pd.DataFrame,
                model: Llama,
                model_params: dict = None):
    """Test a film review classification prompt on IMDB data subset."""
    logger.info(f"Testing prompt '{prompt_label}' with model '{model.name}'...")

    # Iterate through test dataframe rows and get the prediction for each review text
    test_data = test_data.copy()
    result_values = []
    call_details = []
    predictions = []
    key_phrases_list = []
    ratings = []
    for _, row in test_data.iterrows():
        prediction, details, key_phrases, rating = classify_imdb_review(
            review_text=row[IMDB_REVIEW_TEXT_FIELD],
            model=model,
            prompt_template=prompt_template,
            prompt_label=prompt_label,
            model_params=model_params,
            example_pool=example_pool
        )
        predictions.append(prediction)
        call_details.append(details)
        if key_phrases:
            key_phrases_list.append(key_phrases)
        ratings.append(rating)
        # Evaluate binary classification as true/false positive/negative
        if prediction is not None:
            binary_eval_result = binary_eval(
                row[IMDB_REVIEW_LABEL_FIELD],
                prediction
            )
        else:
            binary_eval_result = None
        result_values.append(binary_eval_result)

    # Add predictions and T/F P/N results to test dataframe
    pred_label = "prediction_" + prompt_label
    test_data[pred_label] = predictions
    result_label = "_".join([model.name, prompt_label])
    test_data[result_label] = result_values

    # Check if keywords are available, if so add to dataframe in separate column
    if len(key_phrases_list) > 0 and key_phrases_list[0] is not None:
        test_data[f"key_phrases_{model.name}"] = key_phrases_list
    
    # Check if ratings are available, if so add to dataframe in separate column
    if len(ratings) > 0 and ratings[0] is not None:
        test_data[f"rating_{model.name}"] = ratings

    # Drop any test data rows with empty/invalid predictions
    start_size = len(test_data)
    test_data = test_data.dropna()
    end_size = len(test_data)
    if end_size < start_size:
        logger.warning(f"Dropped {start_size - end_size} rows with invalid predictions")

    # Get dictionary of counts of TP, FP, TN, FN
    results = test_data[result_label].value_counts().to_dict()

    # Calculate F1 score
    f1_score = calculate_f1(test_data[IMDB_REVIEW_LABEL_FIELD], test_data[pred_label])

    return results, f1_score, call_details, test_data


def test_prompts_on_models(prompts: dict,
                           models: list[Llama],
                           test_data: pd.DataFrame,
                           example_pool: pd.DataFrame,
                           model_params: dict = None,
                           ):
    """Test one or more prompts with one or more models,
    aggregate results summary into Dataframe."""
    prompt_test_results = []
    for model in models:
        for prompt_label, prompt_template in prompts.items():
            results_entry = {
                "model": model.name,
                "prompt": prompt_label,
            }
            results, f1_score, call_details, test_data = test_prompt(
                test_data=test_data,
                prompt_template=prompt_template,
                prompt_label=prompt_label,
                model=model,
                model_params=model_params,
                example_pool=example_pool,
            )
            results_entry["F1"] = f1_score
            results_entry.update(results)
            for result in {"TP", "FP", "TN", "FN"}:
                if result not in results_entry:
                    results_entry[result] = 0
            # Add model parameters, minimally: temperature, top_p, top_k
            for param in {"temperature", "top_p", "top_k"}:
                results_entry[param] = call_details[0][param]
            if model_params:
                for key, value in model_params.items():
                    if key not in results_entry:
                        results_entry[key] = value
            # Take average of latency and token usage
            results_entry["latency"] = mean(
                [call_detail["latency"] for call_detail in call_details]
            )
            results_entry["prompt_tokens"] = mean(
                [call_detail["usage"]["prompt_tokens"] for call_detail in call_details]
            )
            results_entry["completion_tokens"] = mean(
                [call_detail["usage"]["completion_tokens"] for call_detail in call_details]
            )
            results_entry["total_tokens"] = mean(
                [call_detail["usage"]["total_tokens"] for call_detail in call_details]
            )
            prompt_test_results.append(results_entry)
    prompt_test_results = pd.DataFrame(prompt_test_results)
    return prompt_test_results, test_data


def save_test_results(summary_df: pd.DataFrame, sample_df: pd.DataFrame, run_outdir: str):
    """Save test summary and raw results on sample data to TSV files."""
    summary_outfile = os.path.join(run_outdir, "results-summary.tsv")
    summary_df.to_csv(summary_outfile, sep="\t", index=False)
    logger.info(f"Wrote test summary to {summary_outfile}")
    results_outfile = os.path.join(run_outdir, "imdb-sample-results.tsv")
    logger.info(f"Wrote test results to {results_outfile}")
    sample_df.to_csv(results_outfile, sep="\t", index=False)


if __name__ == "__main__":
    # Parse input
    parser = argparse.ArgumentParser(
        description='Test several SLM prompts on a subset of the IMDB dataset.'
    )
    parser.add_argument('--test_size', type=int, default=500, help='Number of test examples')
    parser.add_argument('--test_label', type=str, default=None, help='Optional test label')
    args = parser.parse_args()
    test_size = args.test_size
    min_test_examples_per_class = int(test_size / 2)

    # Load and sample IMDB data
    logger.info("Loading IMDB dataset...")
    imdb_train_data = load_imdb("train")
    imdb_test_data = load_imdb("test")
    logger.info("Sampling from IMDB dataset...")
    imdb_test_sample = sample_from_imdb(
        imdb_test_data,
        min_examples_per_class=min_test_examples_per_class
    )
    logger.info(f"Drew test sample of {len(imdb_test_sample)} IMDB reviews")

    # Test various prompt methods with both Qwen models
    prompt_test_results, imdb_test_sample = test_prompts_on_models(
        prompts=PROMPT_METHODS,
        models=DEFAULT_MODELS,
        test_data=imdb_test_sample,
        example_pool=imdb_train_data,
        model_params={
            "temperature": 0,
            "top_p": 0.10,
            "top_k": 5,
            "max_tokens": 200,
        }
    )

    # Create run out directory
    run_outdir = create_run_outdir(args.test_label)

    # Write test results with today's date/time and commit hash
    save_test_results(prompt_test_results, imdb_test_sample, run_outdir)

    # Visualize results
    plot_outdir = os.path.join(run_outdir, "plots")
    os.makedirs(plot_outdir, exist_ok=True)
    plot_f1_bar_graph(
        prompt_test_results,
        outfile=os.path.join(plot_outdir, "f1-comparison-by-model-and-prompt.png"),
    )
    plot_f1_latency_scatterplot(
        prompt_test_results,
        outfile=os.path.join(plot_outdir, "f1-vs-latency-scatterplot.png"),
    )
    for model in DEFAULT_MODELS:
        for prompt_label in PROMPT_METHODS:
            model_prompt_plot_outdir = os.path.join(plot_outdir, model.name, prompt_label)
            os.makedirs(model_prompt_plot_outdir, exist_ok=True)
            result_label = "_".join([model.name, prompt_label])
            create_tfpn_histogram_by_wordcount(
                imdb_test_sample,
                result_label,
                outfile=os.path.join(model_prompt_plot_outdir, "wordcount-histogram.png"),
            )
            plot_confusion_matrix(
                imdb_test_sample,
                result_label,
                outfile=os.path.join(model_prompt_plot_outdir, "confusion-matrix.png"),
            )
