import logging
import os
import re
from functools import lru_cache
from statistics import mean
from typing import Callable, Union

import pandas as pd
from llama_cpp import Llama

from eval import (BINARY_LABEL_MAP, binary_eval, calculate_f1,
                  create_tfpn_histogram_by_wordcount, plot_confusion_matrix)
from load_imdb_data import load_imdb, sample_from_imdb
from prompts.prompt_templates import (chain_of_thought_prompt,
                                      extract_key_phrases_prompt,
                                      keyword_sentiment_analysis_prompt,
                                      oneshot_review_classification,
                                      zeroshot_review_classification)
from prompts.system_messages import (FILM_REVIEW_CLASSIFIER,
                                     FILM_REVIEW_SUMMARIZER)
from query_slm import Prompt, query_slm
from slm_models import qwen_05B, qwen_15B
from utils import create_timestamp, get_git_commit_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COMMIT_HASH = get_git_commit_hash()
START_TIME = create_timestamp()

VALID_REVIEW_LABELS = {"positive", "negative"}
VALID_REVIEW_REGEX = re.compile("|".join(VALID_REVIEW_LABELS))

PROMPT_METHODS = {
    "zeroshot": zeroshot_review_classification,
    "oneshot": oneshot_review_classification,
    "chain-of-thought": chain_of_thought_prompt,
    "keyword-based_sentiment_analysis": keyword_sentiment_analysis_prompt,
}

DEFAULT_MODELS = [qwen_05B, qwen_15B]


def create_run_outdir():
    """Create an outdir for a single test run."""
    results_outdir = os.path.abspath("results")
    run_outdir = os.path.join(results_outdir, f"{START_TIME}_{COMMIT_HASH}")
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


def classify_imdb_review(review_text: str,
                         model: Llama,
                         prompt_template: Prompt,
                         prompt_label: str = None,
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
    else:
        prompt.generate_prompt(review_text=review_text)
    if not model_params:
        model_params = {}
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
    
    return prediction, details, key_phrases


def test_prompt(test_data: pd.DataFrame,
                prompt_template: Union[Callable, str],
                prompt_label: str,
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
    for _, row in test_data.iterrows():
        prediction, details, key_phrases = classify_imdb_review(
            review_text=row["review"],
            model=model,
            prompt_template=prompt_template,
            prompt_label=prompt_label,
            model_params=model_params,
        )
        predictions.append(prediction)
        call_details.append(details)
        if key_phrases:
            key_phrases_list.append(key_phrases)
        # Evaluate binary classification as true/false positive/negative
        binary_eval_result = binary_eval(row["label"], prediction) if prediction is not None else None
        result_values.append(binary_eval_result)

    # Add predictions and T/F P/N results to test dataframe
    pred_label = "prediction_" + prompt_label
    test_data[pred_label] = predictions
    result_label = "_".join([model.name, prompt_label])
    test_data[result_label] = result_values
    
    # Check if keywords are available, if so add to dataframe in separate column
    if len(key_phrases_list) > 0 and key_phrases_list[0] is not None:
        test_data["key_phrases"] = key_phrases_list
    
    # Drop any test data rows with empty/invalid predictions
    start_size = len(test_data)
    test_data = test_data.dropna()
    end_size = len(test_data)
    if end_size < start_size:
        logger.warning(f"Dropped {start_size - end_size} rows with invalid predictions")
    
    # Get dictionary of counts of TP, FP, TN, FN
    results = test_data[result_label].value_counts().to_dict()

    # Calculate F1 score
    f1_score = calculate_f1(test_data["label"], test_data[pred_label])

    return results, f1_score, call_details, test_data


def test_prompts_on_models(prompts: dict, models: list, test_data: pd.DataFrame, model_params: dict = None):
    """Test one or more prompts with one or more models, aggregate results summary into Dataframe."""
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
            results_entry["latency"] = mean([call_detail["latency"] for call_detail in call_details])
            results_entry["prompt_tokens"] = mean([call_detail["usage"]["prompt_tokens"] for call_detail in call_details])
            results_entry["completion_tokens"] = mean([call_detail["usage"]["completion_tokens"] for call_detail in call_details])
            results_entry["total_tokens"] = mean([call_detail["usage"]["total_tokens"] for call_detail in call_details])
            prompt_test_results.append(results_entry)
    prompt_test_results = pd.DataFrame(prompt_test_results)
    return prompt_test_results, test_data


def save_test_results(summary_df: pd.DataFrame, sample_df: pd.DataFrame, run_outdir: str):
    """Save test summary and raw results on sample data to TSV files."""
    summary_outfile = os.path.join(run_outdir, f"results-summary.tsv")
    summary_df.to_csv(summary_outfile, sep="\t", index=False)
    logger.info(f"Wrote test summary to {summary_outfile}")
    results_outfile = os.path.join(run_outdir, f"imdb-sample-results.tsv")
    logger.info(f"Wrote test results to {results_outfile}")
    sample_df.to_csv(results_outfile, sep="\t", index=False)


if __name__ == "__main__":
    # Load and sample IMDB data
    logger.info("Loading IMDB dataset...")
    imdb_data = load_imdb("test")
    logger.info("Sampling from IMDB dataset...")
    imdb_sample = sample_from_imdb(imdb_data, examples_per_class=100)
    logger.info(f"Drew sample of {len(imdb_sample)} IMDB reviews")
    
    # Create run out directory
    run_outdir = create_run_outdir()

    # Test various prompt methods with both Qwen models
    prompt_test_results, imdb_sample = test_prompts_on_models(
        prompts=PROMPT_METHODS,
        models=DEFAULT_MODELS,
        test_data=imdb_sample,
        model_params={
            "temperature": 0,
            "top_p": 0.99,
            "top_k": 5,
        }
    )

    # Write test results with today's date/time and commit hash
    save_test_results(prompt_test_results, imdb_sample, run_outdir)

    # Visualize results
    for model in DEFAULT_MODELS:
        for prompt_label in PROMPT_METHODS:
            plot_outdir = os.path.join(run_outdir, model.name, prompt_label)
            os.makedirs(plot_outdir, exist_ok=True)
            result_label = "_".join([model.name, prompt_label])
            create_tfpn_histogram_by_wordcount(
                imdb_sample,
                result_label,
                outfile=os.path.join(plot_outdir, "wordcount-histogram.png"),
            )
            plot_confusion_matrix(
                imdb_sample,
                result_label,
                outfile=os.path.join(plot_outdir, "confusion-matrix.png"),
            )
