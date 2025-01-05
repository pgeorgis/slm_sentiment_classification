"""Initialize Qwen2.5 SLM models."""

from llama_cpp import Llama

N_CTX = 32769

# 0.5B (500M) parameter model
qwen_05B = Llama.from_pretrained(
    repo_id="bartowski/Qwen2.5-0.5B-Instruct-GGUF",
    filename="*Q5_K_M.gguf",
    verbose=False,
    n_ctx=N_CTX,
)
qwen_05B.name = "Qwen-0.5B"

# 1.5B parameter model
qwen_15B = Llama.from_pretrained(
    repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
    filename="*Q5_K_M.gguf",
    verbose=False,
    n_ctx=N_CTX,
)
qwen_15B.name = "Qwen-1.5B"
