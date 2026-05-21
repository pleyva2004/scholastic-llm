---
title: scholastic-llm demo
emoji: "📜"
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: "3.12"
pinned: false
hardware: cpu-basic
short_description: Qwen 2.5 7B + LoRA in Aquinas's Summa form (CPU, slow)
license: mit
models:
  - pleyva2004/scholastic-llm-sft-v2-iter400-peft
  - Qwen/Qwen2.5-7B-Instruct
tags:
  - lora
  - peft
  - mlx
  - philosophy
  - catechism
  - scholastic
  - qwen
---

# scholastic-llm — interactive demo

Try the fine-tuned model in your browser. The adapter is the recommended
Phase 2 checkpoint (`sft-v2-iter400`) from the
[`scholastic-llm` project](https://github.com/pleyva2004/scholastic-llm)
applied to `Qwen/Qwen2.5-7B-Instruct`.

The model has been fine-tuned to respond to philosophical and theological
questions in a scholastic / Aquinas-Summa register, grounded in the
Catechism of the Catholic Church (CCC, 1992), with Augustinian rhetorical
moves for existential prompts.

> **⚠ Research experiment, not theological authority.** Model can
> hallucinate CCC paragraph numbers and confidently misstate doctrine.
> Outputs must not be cited as catechetical instruction. For doctrinal
> questions consult the actual Catechism, a qualified priest, or a
> trained theologian.

## Resources

- 📄 [Paper](https://pleyva2004.github.io/scholastic-llm/main.pdf)
- 🗒️ [One-pager](https://pleyva2004.github.io/scholastic-llm/poster.pdf)
- 💻 [Repo](https://github.com/pleyva2004/scholastic-llm)
- 🤗 [Adapter (PEFT format)](https://huggingface.co/pleyva2004/scholastic-llm-sft-v2-iter400-peft)
- 🤗 [Adapter (MLX format)](https://huggingface.co/pleyva2004/scholastic-llm-sft-v2-iter400)

## Hardware

Runs on Hugging Face **free CPU** (cpu-basic, 16 GB RAM, 2 vCPU). The
7B model is large for this tier — initial load takes ~2-3 minutes, and
each response takes ~30-60 seconds. For faster inference, the same
adapter can be run locally on Apple Silicon via `mlx-lm` (see the
[repo](https://github.com/pleyva2004/scholastic-llm)) or on any GPU
machine via the [PEFT adapter](https://huggingface.co/pleyva2004/scholastic-llm-sft-v2-iter400-peft).
