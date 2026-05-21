"""Convert an MLX-LM LoRA adapter to PEFT (HuggingFace) format.

MLX-LM-LoRA stores LoRA weights as:
    model.layers.{i}.{module_path}.lora_a   shape (in_features, rank)
    model.layers.{i}.{module_path}.lora_b   shape (rank, out_features)
with `forward = x @ A @ B * scale` (no transpose).

PEFT stores LoRA weights as:
    base_model.model.model.layers.{i}.{module_path}.lora_A.weight   shape (rank, in_features)
    base_model.model.model.layers.{i}.{module_path}.lora_B.weight   shape (out_features, rank)
with `forward = x @ A^T @ B^T * (lora_alpha / r)`.

So conversion = (a) prefix keys with `base_model.model.`, (b) rename
`lora_a/b` → `lora_A.weight/lora_B.weight`, (c) transpose both, (d) write
a PEFT-format adapter_config.json with `lora_alpha = scale × rank` and
`layers_to_transform = <indices found in the safetensors>` so PEFT applies
adapters only to the same top-N layers MLX trained.

Usage:
    python scripts/mlx_to_peft.py adapters/scholastic-v2-iter400 ./out/peft-v2-iter400
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MLX_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)\.lora_([ab])$")
BASE_MODEL_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"


def convert(src_dir: Path, dst_dir: Path, base_model: str) -> None:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # --- Read MLX config -------------------------------------------------
    mlx_cfg_path = src_dir / "adapter_config.json"
    with mlx_cfg_path.open() as fh:
        mlx_cfg = json.load(fh)
    lora_params = mlx_cfg.get("lora_parameters", mlx_cfg)
    rank = int(lora_params.get("rank", 8))
    scale = float(lora_params.get("scale", 10.0))
    lora_alpha = int(round(scale * rank))  # PEFT folds scale into alpha
    print(f"  source: {mlx_cfg_path}")
    print(f"  rank={rank}, scale={scale} → lora_alpha={lora_alpha}")

    # --- Load + remap + transpose tensors --------------------------------
    src_tensors = load_file(str(src_dir / "adapters.safetensors"))
    print(f"  loaded {len(src_tensors)} MLX tensors")

    dst_tensors: dict[str, torch.Tensor] = {}
    layer_ids: set[int] = set()
    target_modules: set[str] = set()
    skipped = 0

    for k, v in src_tensors.items():
        m = MLX_KEY_RE.match(k)
        if not m:
            print(f"  skip (unexpected key): {k}")
            skipped += 1
            continue
        layer_idx, module_path, ab = m.group(1), m.group(2), m.group(3)
        layer_ids.add(int(layer_idx))
        target_modules.add(module_path.split(".")[-1])  # e.g. q_proj

        new_letter = "A" if ab == "a" else "B"
        new_key = (
            f"base_model.model.model.layers.{layer_idx}.{module_path}"
            f".lora_{new_letter}.weight"
        )
        # MLX: lora_a is (in, rank), lora_b is (rank, out).
        # PEFT: lora_A.weight is (rank, in), lora_B.weight is (out, rank).
        dst_tensors[new_key] = v.T.contiguous()

    print(f"  wrote {len(dst_tensors)} PEFT tensors ({skipped} skipped)")
    print(f"  target modules: {sorted(target_modules)}")
    print(f"  layers: {min(layer_ids)}–{max(layer_ids)} (count={len(layer_ids)})")

    save_file(dst_tensors, str(dst_dir / "adapter_model.safetensors"))

    # --- Build PEFT config -----------------------------------------------
    peft_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted(target_modules),
        "layers_to_transform": sorted(layer_ids),
        "layers_pattern": "layers",
        "inference_mode": True,
        "fan_in_fan_out": False,
        "init_lora_weights": True,
        "modules_to_save": None,
        "auto_mapping": None,
        "revision": None,
        "rank_pattern": {},
        "alpha_pattern": {},
    }
    cfg_path = dst_dir / "adapter_config.json"
    with cfg_path.open("w") as fh:
        json.dump(peft_cfg, fh, indent=2)

    # --- Size sanity check -----------------------------------------------
    src_size = (src_dir / "adapters.safetensors").stat().st_size
    dst_size = (dst_dir / "adapter_model.safetensors").stat().st_size
    print(
        f"  size: {src_size/1e6:.1f} MB (MLX) → {dst_size/1e6:.1f} MB (PEFT) "
        f"= {dst_size/src_size:.2%}"
    )

    print(f"\nWrote PEFT adapter to: {dst_dir}")
    print(f"  adapter_model.safetensors")
    print(f"  adapter_config.json")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", help="MLX adapter directory (contains adapters.safetensors)")
    p.add_argument("dst", help="Output PEFT adapter directory")
    p.add_argument("--base", default=BASE_MODEL_DEFAULT, help=f"Base model HF id (default: {BASE_MODEL_DEFAULT})")
    args = p.parse_args()
    convert(Path(args.src), Path(args.dst), args.base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
