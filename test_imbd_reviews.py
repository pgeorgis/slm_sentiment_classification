import logging
import os
import re
from typing import Callable, Union

import pandas as pd
from llama_cpp import Llama

from eval import BINARY_LABEL_MAP, binary_eval, calculate_f1
from load_imdb_data import load_imdb, sample_from_imdb
from prompts.prompt_templates import (chain_of_thought_prompt,
                                      oneshot_review_classification,
                                      zeroshot_review_classification)
from prompts.system_messages import FILM_REVIEW_CLASSIFIER
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
}


def classify_imdb_review(review_text: str,
                         model: Llama,
                         prompt_template: Prompt,
                         system_message: str=FILM_REVIEW_CLASSIFIER):
    """Classify an IMDB review as a 'positive' or 'negative' review."""
    prompt = Prompt(
        prompt_template,
        system_message=system_message,
    )
    prompt.generate_prompt(review_text=review_text)
    prediction = query_slm(model, prompt)
    prediction = prediction.strip().lower()
    match_prediction = VALID_REVIEW_REGEX.search(prediction)
    if not match_prediction:
        logger.warning(f"Unexpected response: {prediction}")
        prediction = ""
    else:
        prediction = match_prediction.group()
    prediction = BINARY_LABEL_MAP.get(prediction, prediction)
    return prediction


def test_prompt(test_data: pd.DataFrame, prompt_template: Union[Callable, str], prompt_label: str, model: Llama):
    """Test a film review classification prompt on IMDB data subset."""
    logger.info(f"Testing prompt '{prompt_label}' with model '{model.name}'...")
    test_data = test_data.copy()
    pred_label = "prediction_" + prompt_label
    test_data[pred_label] = test_data["review"].apply(
        classify_imdb_review,
        model=model,
        prompt_template=prompt_template,
    )
    f1_score = calculate_f1(test_data["label"], test_data[pred_label])
    result_values = []
    for _, row in test_data.iterrows():
        result_values.append(binary_eval(row["label"], row[pred_label]))
    result_label = "result_" + prompt_label
    test_data[result_label] = result_values
    results = test_data[result_label].value_counts().to_dict()
    return results, f1_score, test_data


def test_prompts_on_models(prompts: dict, models: list, test_data: pd.DataFrame):
    """Test one or more prompts with one or more models, aggregate results summary into Dataframe."""
    prompt_test_results = []
    for model in models:
        for prompt_label, prompt_template in prompts.items():
            results_entry = {
                "model": model.name,
                "prompt": prompt_label,
            }
            results, f1_score, test_data = test_prompt(
                test_data=test_data,
                prompt_template=prompt_template,
                prompt_label=prompt_label,
                model=model,
            )
            results_entry.update(results)
            for result in {"TP", "FP", "TN", "FN"}:
                if result not in results_entry:
                    results_entry[result] = 0
            results_entry["F1"] = f1_score
            prompt_test_results.append(results_entry)
    prompt_test_results = pd.DataFrame(prompt_test_results)
    return prompt_test_results, test_data


logger.info("Loading IMDB dataset...")
imdb_data = load_imdb("test")
logger.info("Sampling from IMDB dataset...")
imdb_sample = sample_from_imdb(imdb_data, examples_per_class=10)
logger.info(f"Drew sample of {len(imdb_sample)} IMDB reviews")

# Test various prompt methods with both Qwen models
prompt_test_results, imdb_sample = test_prompts_on_models(
    prompts=PROMPT_METHODS,
    models=[qwen_05B, qwen_15B],
    test_data=imdb_sample,
)

# Write test results with today's date/time and commit hash
outdir = os.path.abspath("results")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, f"{START_TIME}_{COMMIT_HASH}_results.tsv")
prompt_test_results.to_csv(outfile, sep="\t", index=False)
logger.info(f"Wrote test results to {outfile}")
