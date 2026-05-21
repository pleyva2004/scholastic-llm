"""scholastic-llm Gradio demo on Hugging Face Spaces (CPU).

Loads Qwen 2.5 7B-Instruct (int8) + PEFT adapter
`pleyva2004/scholastic-llm-sft-v2-iter400-peft` and exposes a streaming
chat interface. Running on free CPU hardware — expect ~30-60 s per response.
"""

from __future__ import annotations

from threading import Thread

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_REPO = "pleyva2004/scholastic-llm-sft-v2-iter400-peft"

# Load on CPU. 7B bf16 is ~15GB; CPU basic Space has 16GB RAM. Tight.
print(f"Loading base model {BASE_MODEL}…")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,  # 14GB weights — tight on 16GB CPU Space
    low_cpu_mem_usage=True,
)
print(f"Applying PEFT adapter {ADAPTER_REPO}…")
model = PeftModel.from_pretrained(base, ADAPTER_REPO).eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
print("Ready (CPU mode — responses will take 30-60 s).")


def respond(message: str, history):
    """Streaming chat callback. `history` is a list of {role, content} dicts."""
    messages = []
    if history:
        for h in history:
            if isinstance(h, dict) and "role" in h and "content" in h:
                messages.append(h)
            elif isinstance(h, (list, tuple)) and len(h) == 2:
                u, a = h
                if u:
                    messages.append({"role": "user", "content": u})
                if a:
                    messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=420,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for chunk in streamer:
        partial += chunk
        yield partial


EXAMPLES = [
    "Is the soul immortal?",
    "What is the relationship between grace and free will?",
    "Why does God permit suffering?",
    "Can a human truly know God?",
    "How do you reconcile divine foreknowledge with free will?",
    "What does it mean to love one's neighbor?",
]


NOTICE_MD = """
# scholastic-llm — interactive demo

Qwen 2.5 7B-Instruct + LoRA adapter trained on 377 teacher-distilled
scholastic Q&A pairs. The model responds in Aquinas's *Summa* form
("Whether…", "Objection 1…", "On the contrary…", "I answer that…"),
grounded in the Catechism of the Catholic Church (CCC, 1992).

> **⚠ Research experiment, not theological authority.** The model can
> hallucinate CCC paragraph numbers and confidently misstate doctrine.
> Outputs must not be cited as catechetical instruction. For doctrinal
> questions consult the actual Catechism, a qualified priest, or a
> trained theologian.

The first request takes ~10–15 s to warm up the ZeroGPU slot;
subsequent requests in the same session stream at ~1–3 s per response.

📄 [Paper](https://pleyva2004.github.io/scholastic-llm/main.pdf) ·
🗒️ [One-pager](https://pleyva2004.github.io/scholastic-llm/poster.pdf) ·
💻 [Repo](https://github.com/pleyva2004/scholastic-llm) ·
🤗 [Adapter (MLX)](https://huggingface.co/pleyva2004/scholastic-llm-sft-v2-iter400) ·
🤗 [Adapter (PEFT)](https://huggingface.co/pleyva2004/scholastic-llm-sft-v2-iter400-peft)
"""


with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", neutral_hue="stone")) as demo:
    gr.Markdown(NOTICE_MD)
    gr.ChatInterface(
        fn=respond,
        type="messages",
        examples=EXAMPLES,
        cache_examples=False,
        chatbot=gr.Chatbot(height=480, type="messages"),
    )


if __name__ == "__main__":
    demo.launch()
