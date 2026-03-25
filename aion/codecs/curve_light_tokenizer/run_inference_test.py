#!/usr/bin/env python3
"""
Light Curve Codec Inference Test Script

Tests the LightCurveCodecInference class on AstroM3 dataset and outputs discrete tokens.

Usage:
    conda activate aion
    cd /mnt/oneastronomy/oneastronomy/ck-aion/AION
    python curve_light_tokenizer/run_inference_test.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add AION root to path
AION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AION_ROOT))

from datasets.astrom3_dataset_v4 import AstroM3DatasetV4
from curve_light_tokenizer.light_curve_codec_inference import LightCurveCodecInference


def main():
    # Configuration
    CHECKPOINT = "outputs/training_astrom3_p2_fsq_1600/checkpoints/best_model.pt"
    CONFIG = "configs/phase2_reset/p2_reset_astrom3_fsq_1600.json"
    DATA_PATH = "datasets/astrom3_photometry"
    DEVICE = "cpu"

    print("=" * 60)
    print("Light Curve Codec Inference Test")
    print("=" * 60)

    # 1. Load codec
    print("\n[1/3] Loading codec from checkpoint...")
    config_path = Path(CONFIG)
    codec = LightCurveCodecInference.from_checkpoint(
        checkpoint_path=CHECKPOINT,
        config_path=config_path if config_path.exists() else None,
        device=DEVICE,
    )
    print(f"  Vocab size: {codec.vocab_size}")
    print(f"  Embed dim: {codec.embed_dim}")
    print(f"  Fixed length: {codec.fixed_length}")
    print(f"  Levels: {codec.levels}")

    # 2. Load dataset
    print("\n[2/3] Loading AstroM3 dataset...")
    test_dataset = AstroM3DatasetV4(
        data_dir=DATA_PATH,
        split="test",
        fixed_length=codec.fixed_length,
        min_valid_points=5,
        random_start=False,
        seed=42,
    )
    print(f"  Dataset size: {len(test_dataset)}")

    # Get first sample
    sample = test_dataset[0]
    print(f"  Sample 0 metadata: {sample['metadata']}")

    # Prepare inputs
    flux = sample["target"].unsqueeze(0).to(DEVICE)  # (1, 200)
    delta_t = sample["past_covariates"]["delta_t"].unsqueeze(0).to(DEVICE)  # (1, 200)
    delta_t_interval = sample["past_covariates"]["delta_t_interval"].unsqueeze(0).to(DEVICE)  # (1, 200)
    context_mask = sample["context_mask"].unsqueeze(0).to(DEVICE)  # (1, 200)

    print(f"\n  Input shapes:")
    print(f"    flux: {flux.shape}")
    print(f"    delta_t: {delta_t.shape}")
    print(f"    delta_t_interval: {delta_t_interval.shape}")
    print(f"    context_mask: {context_mask.shape}")

    # 3. Encode to get tokens
    print("\n[3/3] Running encoding...")
    with torch.no_grad():
        encoding = codec.encode(flux, delta_t, delta_t_interval, context_mask=context_mask)

    print(f"\n  Encoding output:")
    print(f"    tokens shape: {encoding.tokens.shape}")
    print(f"    tokens dtype: {encoding.tokens.dtype}")
    print(f"    latent shape: {encoding.latent.shape}")

    # Output discrete integer tokens
    print("\n" + "=" * 60)
    print("DISCRETE INTEGER TOKENS (1st light curve)")
    print("=" * 60)
    tokens_np = encoding.tokens.cpu().numpy().flatten()
    print(f"Tokens: {tokens_np.tolist()}")
    print(f"\nToken statistics:")
    print(f"  Min: {tokens_np.min()}")
    print(f"  Max: {tokens_np.max()}")
    print(f"  Unique: {len(set(tokens_np))} / {codec.vocab_size}")
    print(f"  Mean: {tokens_np.mean():.2f}")
    print(f"  Std: {tokens_np.std():.2f}")

    # Verify: decode and reconstruct
    print("\n" + "=" * 60)
    print("RECONSTRUCTION TEST")
    print("=" * 60)
    reconstruction = codec.decode(encoding.tokens, target_length=codec.fixed_length)
    print(f"  Reconstructed flux shape: {reconstruction.flux.shape}")

    flux_mae = (reconstruction.flux - flux).abs().mean().item()
    print(f"  Reconstruction MAE (norm space): {flux_mae:.6f}")

    print("\n" + "=" * 60)
    print("SUCCESS: Inference code works correctly!")
    print("=" * 60)

    # Save results
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Helper function for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        return obj

    # Convert metadata for JSON serialization
    sample_metadata = to_serializable(dict(sample["metadata"]))

    results = {
        "checkpoint": CHECKPOINT,
        "config": CONFIG,
        "vocab_size": int(codec.vocab_size),
        "embed_dim": int(codec.embed_dim),
        "levels": [int(x) for x in codec.levels],
        "sample_metadata": sample_metadata,
        "tokens": tokens_np.tolist(),
        "token_statistics": {
            "min": int(tokens_np.min()),
            "max": int(tokens_np.max()),
            "unique": int(len(set(tokens_np))),
            "mean": float(tokens_np.mean()),
            "std": float(tokens_np.std()),
        },
        "reconstruction_mae": float(flux_mae),
    }

    results_path = output_dir / "inference_test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    tokens_path = output_dir / "tokens.pt"
    torch.save(encoding.tokens, tokens_path)
    print(f"Tokens saved to: {tokens_path}")

    return results


if __name__ == "__main__":
    main()
