"""Initialize Qwen2.5 SLM models."""

from llama_cpp import Llama

# 0.5B (500M) parameter model
llm_qwen_05B = Llama.from_pretrained(
    repo_id="bartowski/Qwen2.5-0.5B-Instruct-GGUF",
    filename="*Q5_K_M.gguf",
    verbose=False
)

# 1.5B parameter model
llm_qwen_15B = Llama.from_pretrained(
    repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
    filename="*Q5_K_M.gguf",
    verbose=False
)
