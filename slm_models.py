"""Initialize Qwen2.5 SLM models."""

from llama_cpp import Llama

N_CTX = 32769

def get_qwen05B(name="Qwen-0.5B"):
    """Create an instance of 0.5B (500M) parameter Qwen model."""
    qwen_05B = Llama.from_pretrained(
        repo_id="bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        filename="*Q5_K_M.gguf",
        verbose=False,
        n_ctx=N_CTX,
    )
    qwen_05B.name = name
    return qwen_05B


def get_qwen15B(name="Qwen-1.5B"):
    """Create an instance of 1.5B parameter Qwen model."""
    qwen_15B = Llama.from_pretrained(
        repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        filename="*Q5_K_M.gguf",
        verbose=False,
        n_ctx=N_CTX,
    )
    qwen_15B.name = name
    return qwen_15B
