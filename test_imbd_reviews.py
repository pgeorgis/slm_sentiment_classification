import logging
import re

from eval import BINARY_LABEL_MAP, binary_eval, calculate_f1
from load_imdb_data import load_imdb, sample_from_imdb
from prompts.prompt_templates import zeroshot_review_classification, oneshot_review_classification, chain_of_thought_prompt
from prompts.system_messages import FILM_REVIEW_CLASSIFIER
from query_slm import Prompt, query_slm
from slm_models import llm_qwen_05B, llm_qwen_15B

from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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

logger.info("Loading IMDB dataset...")
imdb_data = load_imdb("test")
logger.info("Sampling from IMDB dataset...")
imdb_sample = sample_from_imdb(imdb_data, examples_per_class=100)
logger.info(f"Drew sample of {len(imdb_sample)} IMDB reviews")

# Test various prompt methods
for model in [llm_qwen_05B, llm_qwen_15B]:
    for prompt_label, prompt_template in PROMPT_METHODS.items():
        logger.info(f"Testing prompt method '{prompt_label}' ...")
        imdb_sample = imdb_sample.copy()
        pred_label = "prediction_" + prompt_label
        imdb_sample[pred_label] = imdb_sample["review"].apply(
            classify_imdb_review,
            model=model,
            prompt_template=prompt_template,
        )
        f1_score = calculate_f1(imdb_sample["label"], imdb_sample[pred_label])
        result_values = []
        for _, row in imdb_sample.iterrows():
            result_values.append(binary_eval(row["label"], row[pred_label]))
        result_label = "result_" + prompt_label
        imdb_sample[result_label] = result_values
        results = imdb_sample[result_label].value_counts().to_dict()
        logger.info(f"F1: {f1_score}")
        logger.info(results)
