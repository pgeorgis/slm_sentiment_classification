import logging
import json
import time
from typing import Callable, Union
import uuid

from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Prompt:
    """Initialize Prompt class."""

    def __init__(self, template: Union[Callable, str], system_message: str, prompt_id: str = ""):
        """Initialize Prompt instance."""
        self.prompt_id = prompt_id if prompt_id else str(uuid.uuid4())
        self.template = template
        self.system_message = system_message
        self.prompt = None
        self.responses = []

    def generate_prompt(self, prompt_outfile=None, **kwargs):
        """Generate prompt from system message and template."""
        if isinstance(self.template, str):
            user_msg_content = self.template
        elif isinstance(self.template, Callable):
            user_msg_content = self.template(**kwargs)
        else:
            raise TypeError(f"Prompt template must be a string or function, found {type(self.template)}")
        messages = [
            {
                "role": "system",
                "content": self.system_message
            },
            {
                "role": "user",
                "content": user_msg_content
            }
        ]
        self.prompt = messages
        if prompt_outfile:
            self.log_prompt(prompt_outfile)
        return self.prompt

    def save_response(self, response):
        """Save an SLM response to the prompt."""
        self.responses.append(response)

    def log_prompt(self, outfile):
        """Log assembled prompt to an output file."""
        with open(outfile, "w", encoding="utf-8") as outf:
            json.dump(self.prompt, outf, indent=4)


def query_slm(model: Llama, prompt: Prompt, temperature: float=0, **kwargs):
    """Query an SLM model with input messages for chat completion."""

    # Extract prompt content
    prompt_content = prompt.prompt

    # Query model and time latency
    start_time = time.time()
    response = model.create_chat_completion(
        temperature=temperature,
        messages=prompt_content,
        **kwargs
    )
    end_time = time.time()
    prompt.save_response(response)

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
