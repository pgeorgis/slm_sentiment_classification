import argparse
import copy
import glob
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
from prompts.prompt_methods import (FEWSHOT_PROMPT_METHODS, PROMPT_METHODS,
                                    QUANTITATIVE_PROMPT_METHODS)
from prompts.prompt_templates import (
    chain_of_thought_with_likelihood_to_rewatch_prompt,
    extract_key_phrases_prompt, keyword_sentiment_analysis_prompt,
    rating_based_sentiment_analysis_prompt)
from prompts.system_messages import (FILM_REVIEW_CLASSIFIER,
                                     FILM_REVIEW_SUMMARIZER)
from query_slm import Prompt, query_slm
from utils import create_datestamp, create_timestamp

VALID_REVIEW_LABELS = {"positive", "negative"}
VALID_REVIEW_REGEX = re.compile("|".join(VALID_REVIEW_LABELS))
PREDICTION_PREFIX = "prediction_"


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
    """Extracts the rating value from a JSON string.

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


def postprocess_predicted_rating(predicted_rating: str,
                                 from_json=False,
                                 positive_threshold: int = 5) -> str | None:
    """Postprocess a predicted rating to classify as 'positive' or 'negative'.

    Args:
        predicted_rating (str): Predicted rating of the film review on 1-10 scale.
        from_json (bool, optional): Extract predicted rating from a json object (vs. from raw text).
        positive_threshold (int, optional): Threshold for positive rating. Defaults to 5.

    Returns:
        str|None:   'positive' or 'negative' classification of the review.
                    None is returned if no numeric rating is found.
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


def quantitative_sentiment_classification(prompt: Prompt,
                                          model: Llama,
                                          n_calls: int = 3,
                                          temperature: float = 0.4,
                                          ):
    """Handle quantitative prompting methods for sentiment analysis."""
    # Determine whether numeric estimate should be extracted from raw text or from json
    if prompt.template == rating_based_sentiment_analysis_prompt:
        from_json = False
    else:
        from_json = True

    # Make N calls and aggregate numeric estimates
    ratings = []
    details = defaultdict(lambda: 0)
    for i in range(n_calls):
        rating_model_params = {"temperature": temperature}
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
    if prompt.template == chain_of_thought_with_likelihood_to_rewatch_prompt:
        likelihood_to_rewatch = rating
        rating = None
    else:
        likelihood_to_rewatch = None
    return prediction, rating, likelihood_to_rewatch, details


def classify_imdb_review(review_text: str,
                         model: Llama,
                         prompt_template: Union[Callable, str],
                         prompt_label: str = None,
                         example_pool: pd.DataFrame = None,
                         system_message: str = FILM_REVIEW_CLASSIFIER,
                         model_params: dict = None):
    """Classify an IMDB review as a 'positive' or 'negative' review."""

    # Initialize Prompt object
    prompt = Prompt(
        prompt_template,
        system_message=system_message,
        prompt_id=prompt_label,
    )

    # Initialize additional details dictionary
    additional_details = defaultdict(lambda: None)

    # Check if using keyword-based sentiment analysis; if so, first extract keywords
    use_keywords = prompt_template == keyword_sentiment_analysis_prompt
    if use_keywords:
        key_phrases, key_phrases_call_details = extract_review_keywords(review_text)
        prompt.generate_prompt(key_phrases=key_phrases)
        additional_details["key_phrases"] = key_phrases

    # Fewshot methods: explicitly specify example pool and N examples
    elif prompt_template in FEWSHOT_PROMPT_METHODS:
        prompt.generate_prompt(
            review_text=review_text,
            example_pool=example_pool,
            n_examples=FEWSHOT_EXAMPLE_N
        )

    # Other methods: standard prompt generation
    else:
        prompt.generate_prompt(review_text=review_text)

    # Initialize model parameters
    if not model_params:
        model_params = {}

    # Handle sentiment analysis methods using quantitative/numeric estimates
    if prompt_template in QUANTITATIVE_PROMPT_METHODS:
        prediction, rating, likelihood_to_rewatch, details = quantitative_sentiment_classification(
            prompt=prompt, model=model,
        )
        additional_details["rating"] = rating
        additional_details["likelihood_to_rewatch"] = likelihood_to_rewatch

    # Non-quantitative sentiment analysis methods
    else:
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

    return prediction, details, additional_details


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
    additional_details_lists = defaultdict(lambda: [])
    for _, row in test_data.iterrows():
        prediction, details, additional_details = classify_imdb_review(
            review_text=row[IMDB_REVIEW_TEXT_FIELD],
            model=model,
            prompt_template=prompt_template,
            prompt_label=prompt_label,
            model_params=model_params,
            example_pool=example_pool
        )
        predictions.append(prediction)
        call_details.append(details)
        for detail_field, value in additional_details.items():
            additional_details_lists[detail_field].append(value)
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
    pred_label = PREDICTION_PREFIX + prompt_label
    test_data[pred_label] = predictions
    result_label = "_".join([model.name, prompt_label])
    test_data[result_label] = result_values

    # Add additional details to dataframe in separate columns
    for detail_field, values in additional_details_lists.items():
        if any(v is not None for v in values):
            test_data[f"{detail_field}-{model.name}"] = values

    # Drop any test data rows with empty/invalid predictions
    start_size = len(test_data)
    scorable_test_data = copy.deepcopy(test_data)
    scorable_test_data = scorable_test_data.dropna()
    end_size = len(scorable_test_data)
    if end_size < start_size:
        logger.warning(f"Dropped {start_size - end_size} rows with invalid predictions")

    # Get dictionary of counts of TP, FP, TN, FN
    results = scorable_test_data[result_label].value_counts().to_dict()

    # Calculate F1 score
    f1_score = calculate_f1(
        scorable_test_data[IMDB_REVIEW_LABEL_FIELD],
        scorable_test_data[pred_label]
    )

    return results, f1_score, call_details, test_data


def test_prompts_on_models(prompts: dict,
                           models: list[Llama],
                           test_data: pd.DataFrame,
                           example_pool: pd.DataFrame,
                           model_params: dict = None,
                           run_outdir: str = None,
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

            # Write intermediate/checkpoint results after each prompt is tested
            if (len(prompts) > 1 or len(models) > 1) and run_outdir is not None:
                intermediate_results = pd.DataFrame(prompt_test_results)
                save_test_results(
                    summary_df=intermediate_results,
                    sample_df=test_data,
                    run_outdir=run_outdir,
                    test_label="intermediate",
                )
    prompt_test_results = pd.DataFrame(prompt_test_results)
    return prompt_test_results, test_data


def save_test_results(summary_df: pd.DataFrame,
                      sample_df: pd.DataFrame,
                      run_outdir: str,
                      test_label: str = None):
    """Save test summary and raw results on sample data to TSV files."""
    if test_label is None:
        summary_outfile_path = os.path.join(run_outdir, "results-summary.tsv")
        results_outfile_path = os.path.join(run_outdir, "imdb-sample-results.tsv")
    else:
        summary_outfile_path = os.path.join(run_outdir, f"{test_label}_results-summary.tsv")
        results_outfile_path = os.path.join(run_outdir, f"{test_label}_imdb-sample-results.tsv")
    summary_df.to_csv(summary_outfile_path, sep="\t", index=False)
    logger.info(f"Wrote test summary to {summary_outfile_path}")
    logger.info(f"Wrote test results to {results_outfile_path}")
    sample_df.to_csv(results_outfile_path, sep="\t", index=False)


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

    # Create run out directory
    run_outdir = create_run_outdir(args.test_label)

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
        },
        run_outdir=run_outdir,
    )

    # Write test results with today's date/time and commit hash
    save_test_results(prompt_test_results, imdb_test_sample, run_outdir)
    # Clean up any intermediate results
    intermediate_outfiles = glob.glob(os.path.join(run_outdir, "intermediate*"))
    for intermediate_file in intermediate_outfiles:
        os.remove(intermediate_file)

    # Drop NaN prediction entries from test results and create plots
    prediction_columns = [
        col for col in imdb_test_sample.columns
        if col.startswith(PREDICTION_PREFIX)
    ]
    imdb_test_sample = imdb_test_sample.dropna(subset=prediction_columns)

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
