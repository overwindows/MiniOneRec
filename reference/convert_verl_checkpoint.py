#!/usr/bin/env python3
"""
Convert VERL FSDP checkpoint to HuggingFace format for evaluation.

VERL saves checkpoints in FSDP sharded format:
  global_step_X/actor/model_world_size_N_rank_*.pt

This script merges the shards and saves as HuggingFace model.

Usage:
    python convert_verl_checkpoint.py \
        --checkpoint_dir output_dir/rl_mind_xxx/global_step_160 \
        --output_dir output_dir/rl_mind_xxx/global_step_160_hf \
        --base_model output_dir/sft_mind_ranking_xxx/final_checkpoint
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
# Import distributed tensor for loading FSDP checkpoints with DTensors
try:
    import torch.distributed.tensor
except ImportError:
    pass  # Not available in all PyTorch versions

from safetensors.torch import save_file
from tqdm import tqdm


def merge_fsdp_shards(checkpoint_dir: str) -> dict:
    """
    Merge FSDP sharded model weights into a single state dict.

    VERL uses FSDP with sharded state dict saving, where each rank holds
    a shard of the model weights. We need to load all shards and concatenate them.
    """
    actor_dir = Path(checkpoint_dir) / "actor"

    if not actor_dir.exists():
        raise ValueError(f"Actor directory not found: {actor_dir}")

    # Find all model shards
    model_files = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))

    if not model_files:
        raise ValueError(f"No model shards found in {actor_dir}")

    print(f"Found {len(model_files)} model shards")

    # Load all shards
    shards = []
    for i, model_file in enumerate(model_files):
        print(f"Loading shard {i}: {model_file.name}")
        shard = torch.load(model_file, map_location="cpu", weights_only=False)
        shards.append(shard)

    print(f"Loaded {len(shards)} shards")

    # Get parameter names from first shard
    param_names = list(shards[0].keys())
    print(f"Found {len(param_names)} parameters")

    # Import DTensor for type checking
    from torch.distributed.tensor import DTensor

    # Merge shards - concatenate along the sharded dimension
    merged_state_dict = {}
    for param_name in tqdm(param_names, desc="Merging parameters"):
        # Collect local tensors from all shards
        local_tensors = []
        for shard in shards:
            tensor = shard[param_name]
            if isinstance(tensor, DTensor):
                local_tensors.append(tensor.to_local())
            else:
                local_tensors.append(tensor)

        # For DTensors, we need to concatenate along the sharded dimension
        # Typically FSDP shards along dimension 0
        first_tensor = shards[0][param_name]
        if isinstance(first_tensor, DTensor):
            # Get the placements to understand sharding
            placements = first_tensor.placements
            # Check if it's sharded (Shard placement) or replicated (Replicate placement)
            from torch.distributed.tensor.placement_types import Shard, Replicate

            is_sharded = any(isinstance(p, Shard) for p in placements)

            if is_sharded:
                # Find the sharded dimension
                shard_dim = 0
                for p in placements:
                    if isinstance(p, Shard):
                        shard_dim = p.dim
                        break

                # Concatenate along the sharded dimension
                merged_state_dict[param_name] = torch.cat(local_tensors, dim=shard_dim)
            else:
                # Replicated - all shards have the same data, just take the first
                merged_state_dict[param_name] = local_tensors[0]
        else:
            # Regular tensor - just take the first
            merged_state_dict[param_name] = local_tensors[0]

    print(f"Merged {len(merged_state_dict)} parameters")

    # Verify shapes match expected model dimensions
    sample_key = "model.embed_tokens.weight"
    if sample_key in merged_state_dict:
        print(f"Sample parameter '{sample_key}' shape: {merged_state_dict[sample_key].shape}")

    return merged_state_dict


def convert_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    base_model: str,
):
    """
    Convert VERL checkpoint to HuggingFace format.

    Args:
        checkpoint_dir: Path to VERL checkpoint (e.g., global_step_160)
        output_dir: Output directory for HuggingFace model
        base_model: Path to base model for config/tokenizer
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    base_model = Path(base_model)

    print("=" * 60)
    print("Converting VERL Checkpoint to HuggingFace Format")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Base model: {base_model}")
    print("=" * 60)
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config and tokenizer from base model
    print("Copying config and tokenizer from base model...")

    config_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ]

    for fname in config_files:
        src = base_model / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)
            print(f"  Copied: {fname}")

    # Also check huggingface subdirectory in checkpoint
    hf_dir = checkpoint_dir / "actor" / "huggingface"
    if hf_dir.exists():
        for fname in config_files:
            src = hf_dir / fname
            if src.exists() and not (output_dir / fname).exists():
                shutil.copy(src, output_dir / fname)
                print(f"  Copied from checkpoint: {fname}")

    print()

    # Load and merge model weights
    print("Loading model weights...")
    state_dict = merge_fsdp_shards(str(checkpoint_dir))

    # Save as safetensors (preferred) or pytorch
    print(f"\nSaving model to {output_dir}...")

    try:
        # Try safetensors first
        safetensors_path = output_dir / "model.safetensors"
        save_file(state_dict, str(safetensors_path))
        print(f"  Saved as safetensors: {safetensors_path}")
    except Exception as e:
        print(f"  Safetensors failed ({e}), saving as PyTorch...")
        pytorch_path = output_dir / "pytorch_model.bin"
        torch.save(state_dict, pytorch_path)
        print(f"  Saved as PyTorch: {pytorch_path}")

    print()
    print("=" * 60)
    print("Conversion completed!")
    print("=" * 60)
    print(f"\nTo evaluate the model:")
    print(f"  bash scripts/eval_mind_ranking.sh {output_dir} dev")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert VERL checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Path to VERL checkpoint directory (e.g., global_step_160)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for HuggingFace model"
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Path to base model for config/tokenizer (e.g., SFT checkpoint)"
    )

    args = parser.parse_args()

    convert_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
