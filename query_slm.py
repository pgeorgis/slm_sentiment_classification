import logging
import time

from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def query_slm(model: Llama, messages: list, temperature: float=0, **kwargs):
    """Query an SLM model with input messages for chat completion."""

    # Query model and time latency
    start_time = time.time()
    response = model.create_chat_completion(
        temperature=temperature,
        messages=messages,
        **kwargs
    )
    end_time = time.time()

    # Log model latency in seconds
    latency = round(end_time - start_time, 2)
    logger.info(f"Model latency: {latency} seconds")

    # Log token usage
    logger.info(f"Token usage: {response['usage']}")

    # Extract response
    response_choice = response["choices"][0]
    response_content = response_choice["message"]["content"]

    # Check finish reason and log warning if not "stop"
    finish_reason = response_choice["finish_reason"]
    if finish_reason != "stop":
        logger.warning(f"Model response finish reason: {finish_reason}")

    return response_content

