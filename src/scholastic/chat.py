"""Chat helpers for inference against a base model and optional LoRA adapter."""

from pathlib import Path

from mlx_lm import generate, load


def load_model(model_path: str | Path, adapter_path: str | Path | None = None):
    """Load an MLX model, optionally with a LoRA adapter applied."""
    if adapter_path is None:
        return load(str(model_path))
    return load(str(model_path), adapter_path=str(adapter_path))


def chat(model, tokenizer, prompt: str, max_tokens: int = 300) -> str:
    """Run a single-turn chat against the loaded model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=text, max_tokens=max_tokens).strip()


def compare(prompts: list[str], named_models: list[tuple[str, object, object]], max_tokens: int = 200) -> None:
    """Print a side-by-side comparison of multiple model variants on the same prompts.

    named_models: list of (label, model, tokenizer) tuples.
    """
    for p in prompts:
        print(f"\nQ: {p}")
        for label, model, tok in named_models:
            out = chat(model, tok, p, max_tokens=max_tokens)
            preview = out[:240] + ("…" if len(out) > 240 else "")
            print(f"  {label}: {preview}")
        print("-" * 80)
