"""
Light Curve Codec Inference Module (AstroM3 Phase 2 - FSQ)

Provides a clean inference interface for the light curve codec with FSQ quantization.
Supports encoding light curves to discrete tokens and decoding tokens back to light curves.

Architecture:
- Encoder: Chronos-2-small backbone + grad_mlp projection (512D -> 5D) + FSQ quantizer (levels=[4,4,5,5,5], vocab=1600)
- Decoder: Transformer decoder (3 layers, 8 heads, 512D hidden) -> patch expansion (16 tokens per patch)

Usage:
    from light_curve_codec_inference import LightCurveCodecInference

    codec = LightCurveCodecInference.from_checkpoint(
        checkpoint_path="outputs/training_astrom3_p2_fsq_1600/checkpoints/best_model.pt",
        config_path="configs/phase2_reset/p2_reset_astrom3_fsq_1600.json",
        device="cuda"
    )

    # Encode: light curve -> tokens
    tokens = codec.encode(flux, delta_t, delta_t_interval)

    # Decode: tokens -> reconstructed light curve
    reconstruction = codec.decode(tokens)

    # Or use end-to-end
    reconstruction = codec.reconstruct(flux, delta_t, delta_t_interval)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from aion.codecs.light_curve_p2_reset_encoder import LightCurveEncoderP2Reset
from aion.codecs.light_curve_v5_1_decoder import LightCurveDecoderV5_1
from aion.codecs.quantizers import FiniteScalarQuantizer


@dataclass
class LightCurveOutput:
    """Container for reconstructed light curve outputs."""
    flux: torch.Tensor  # (B, T) - reconstructed flux values
    delta_t: torch.Tensor  # (B, T) - cumulative time from first observation
    delta_t_interval: torch.Tensor  # (B, T) - time intervals between observations

    # Optional: normalized space outputs
    flux_norm: Optional[torch.Tensor] = None
    delta_t_norm: Optional[torch.Tensor] = None
    delta_t_interval_norm: Optional[torch.Tensor] = None


@dataclass
class EncodingOutput:
    """Container for encoding outputs."""
    tokens: torch.Tensor  # (B, T) - discrete token indices
    latent: torch.Tensor  # (B, T, D) - continuous latent before quantization
    codes: Optional[torch.Tensor] = None  # (B,) - packed code indices (for FSQ)


class LightCurveCodecInference:
    """
    Inference interface for the AstroM3 Phase 2 Light Curve Codec with FSQ quantization.

    This class provides a clean API for:
    1. Encoding light curves to discrete tokens (for compression/storage)
    2. Decoding tokens back to reconstructed light curves
    3. End-to-end reconstruction

    Attributes:
        encoder: LightCurveEncoderP2Reset instance
        decoder: LightCurveDecoderV5_1 instance
        quantizer: FiniteScalarQuantizer (FSQ) instance
        config: Model configuration dictionary
        device: Device the models are loaded on
        fixed_length: Expected sequence length (default: 200)

    Example:
        >>> codec = LightCurveCodecInference.from_checkpoint(
        ...     "outputs/training_astrom3_p2_fsq_1600/checkpoints/best_model.pt"
        ... )
        >>> tokens = codec.encode(flux, delta_t, delta_t_interval)
        >>> reconstructed = codec.decode(tokens, target_length=200)
    """

    def __init__(
        self,
        encoder: LightCurveEncoderP2Reset,
        decoder: LightCurveDecoderV5_1,
        quantizer: FiniteScalarQuantizer,
        config: dict,
        device: str = "cpu",
    ):
        """
        Initialize the codec inference module.

        Args:
            encoder: Pre-configured encoder module
            decoder: Pre-configured decoder module
            quantizer: FSQ quantizer module
            config: Configuration dictionary
            device: Device to run inference on
        """
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.config = config
        self.device = device

        # Extract configuration
        self.fixed_length = config.get("data", {}).get("fixed_length", 200)
        self._embed_dim = config.get("fsq", {}).get("embed_dim", 5)
        self.levels = config.get("fsq", {}).get("quantizer", {}).get("levels", [4, 4, 5, 5, 5])

        # Move models to device
        self.encoder.to(device)
        self.decoder.to(device)
        self.quantizer.to(device)

        # Set to eval mode
        self.encoder.eval()
        self.decoder.eval()
        self.quantizer.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config_path: Optional[str | Path] = None,
        device: str = "cpu",
        phase1_checkpoint: Optional[str | Path] = None,
    ) -> "LightCurveCodecInference":
        """
        Load codec from a Phase 2 checkpoint.

        Args:
            checkpoint_path: Path to the Phase 2 checkpoint (.pt file)
            config_path: Optional path to config JSON (if not in checkpoint)
            device: Device to load models on ("cpu", "cuda", etc.)
            phase1_checkpoint: Optional path to Phase 1 checkpoint for backbone

        Returns:
            LightCurveCodecInference instance

        Example:
            >>> codec = LightCurveCodecInference.from_checkpoint(
            ...     "outputs/training_astrom3_p2_fsq_1600/checkpoints/best_model.pt",
            ...     device="cuda"
            ... )
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Load config from checkpoint or file
        if config_path and Path(config_path).exists():
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        elif ckpt.get("config"):
            config = ckpt["config"]
        else:
            # Use default config structure
            config = {
                "fsq": {
                    "embed_dim": 5,
                    "proj_type": "grad_mlp",
                    "proj_hidden_dim": 512,
                    "quantizer": {
                        "type": "fsq",
                        "levels": [4, 4, 5, 5, 5],
                        "eps": 0.001,
                    }
                },
                "decoder": {
                    "input_dim": 5,
                    "hidden_dim": 512,
                    "num_layers": 3,
                    "num_heads": 8,
                    "ff_dim": 2048,
                    "output_dim": 2,
                    "patch_size": 16,
                    "dropout": 0.1,
                },
                "data": {
                    "fixed_length": 200,
                }
            }

        # Build encoder
        encoder = LightCurveEncoderP2Reset(
            config=config,
            phase1_encoder=None,
        )

        # Build decoder
        decoder = LightCurveDecoderV5_1(
            input_dim=config.get("decoder", {}).get("input_dim", 5),
            hidden_dim=config.get("decoder", {}).get("hidden_dim", 512),
            num_layers=config.get("decoder", {}).get("num_layers", 3),
            num_heads=config.get("decoder", {}).get("num_heads", 8),
            ff_dim=config.get("decoder", {}).get("ff_dim", 2048),
            output_dim=config.get("decoder", {}).get("output_dim", 2),
            patch_size=config.get("decoder", {}).get("patch_size", 16),
            dropout=config.get("decoder", {}).get("dropout", 0.1),
        )

        # Build quantizer from config
        fsq_cfg = config.get("fsq", config.get("bottleneck", {}))
        q_cfg = fsq_cfg.get("quantizer", {})
        levels = q_cfg.get("levels", [4, 4, 5, 5, 5])
        eps = q_cfg.get("eps", 0.001)
        quantizer = FiniteScalarQuantizer(levels=levels, eps=eps)

        # Load state dicts
        encoder_state = ckpt.get("encoder_state_dict", {})
        decoder_state = ckpt.get("decoder_state_dict", {})

        if encoder_state:
            encoder.load_state_dict(encoder_state, strict=True)
        if decoder_state:
            decoder.load_state_dict(decoder_state, strict=True)

        # Move to device
        encoder.to(device)
        decoder.to(device)
        quantizer.to(device)

        return cls(encoder, decoder, quantizer, config, device)

    def encode(
        self,
        flux: torch.Tensor,
        delta_t: torch.Tensor,
        delta_t_interval: torch.Tensor,
        error: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> EncodingOutput:
        """
        Encode light curve(s) to discrete tokens.

        Args:
            flux: Flux values, shape (B, T) or (T,) for single sample
            delta_t: Cumulative time from first observation, shape (B, T) or (T,)
            delta_t_interval: Time intervals between observations, shape (B, T) or (T,)
            error: Optional flux errors, shape (B, T) or (T,)
            context_mask: Optional mask for valid observations, shape (B, T) or (T,)

        Returns:
            EncodingOutput containing:
                - tokens: Discrete token indices (B, N) - integer codes in [0, vocab_size)
                - latent: Continuous latent before quantization (B, N, D)
                - codes: Same as tokens, kept for backward compatibility

        Example:
            >>> flux = torch.randn(4, 200)  # 4 samples, 200 time points
            >>> delta_t = torch.arange(200).float().unsqueeze(0).expand(4, -1)
            >>> delta_t_interval = torch.ones(4, 200)
            >>> tokens = codec.encode(flux, delta_t, delta_t_interval)
            >>> print(tokens.tokens.shape)  # (4, N) where N = num_patches
            >>> print(tokens.tokens.dtype)  # torch.int32
        """
        # Handle single sample (add batch dimension)
        if flux.dim() == 1:
            flux = flux.unsqueeze(0)
            delta_t = delta_t.unsqueeze(0)
            delta_t_interval = delta_t_interval.unsqueeze(0)
            if error is not None:
                error = error.unsqueeze(0)
            if context_mask is not None:
                context_mask = context_mask.unsqueeze(0)

        B, T = flux.shape

        # Build input dictionary
        past_covariates = {
            "delta_t": delta_t.to(self.device),
            "delta_t_interval": delta_t_interval.to(self.device),
        }
        if error is not None:
            past_covariates["error"] = error.to(self.device)

        inputs = {
            "target": flux.to(self.device),
            "past_covariates": past_covariates,
        }
        if context_mask is not None:
            inputs["context_mask"] = context_mask.to(self.device)

        # Encode
        with torch.no_grad():
            out = self.encoder.encode(inputs)
            latent = out["latent"]  # (B, N, D) where N = num_patches * patch_size

            # Get discrete token indices using FSQ encode
            # FSQ quantizer expects (B, D, T) format, returns integer codes
            latent_ct = latent.transpose(1, 2)  # (B, D, N)
            codes = self.quantizer.encode(latent_ct)  # (B, N) - integer indices in [0, vocab_size)

        return EncodingOutput(
            tokens=codes,  # Integer token indices
            latent=latent,
            codes=codes,
        )

    def decode(
        self,
        tokens: torch.Tensor,
        target_length: Optional[int] = None,
    ) -> LightCurveOutput:
        """
        Decode discrete tokens back to reconstructed light curve.

        Args:
            tokens: Discrete token indices from encoder, shape (B, N) where N = num_patches
                   or (N,) for single sample. Values are integers in [0, vocab_size).
            target_length: Desired output sequence length (default: from config)

        Returns:
            LightCurveOutput containing reconstructed flux and timing information

        Example:
            >>> tokens = torch.randint(0, 1600, (4, 13))  # 4 samples, 13 patches
            >>> output = codec.decode(tokens, target_length=200)
            >>> print(output.flux.shape)  # (4, 200)
        """
        # Handle single sample
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        B = tokens.shape[0]
        target_length = target_length or self.fixed_length

        with torch.no_grad():
            # Decode integer tokens to quantized latent values
            # tokens shape: (B, N) -> quantized_ct: (B, D, N)
            quantized_ct = self.quantizer.decode(tokens)  # (B, D, N)
            quantized = quantized_ct.transpose(1, 2)  # (B, N, D)

            # Pass through decoder
            dec_out = self.decoder(quantized, target_length=target_length)

        flux_norm = dec_out["flux_norm"]
        delta_t_norm = dec_out["delta_t_norm"]
        delta_t_interval_norm = dec_out["delta_t_interval_norm"]

        # Note: To get raw flux values, we need loc_scale from encoder
        # For token-only decoding, we return normalized space values
        return LightCurveOutput(
            flux=flux_norm,  # In norm space (need inverse transform for raw)
            delta_t=delta_t_norm,
            delta_t_interval=delta_t_interval_norm,
            flux_norm=flux_norm,
            delta_t_norm=delta_t_norm,
            delta_t_interval_norm=delta_t_interval_norm,
        )

    def reconstruct(
        self,
        flux: torch.Tensor,
        delta_t: torch.Tensor,
        delta_t_interval: torch.Tensor,
        error: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_tokens: bool = False,
    ) -> LightCurveOutput | tuple[LightCurveOutput, EncodingOutput]:
        """
        End-to-end reconstruction: encode then decode.

        Args:
            flux: Flux values, shape (B, T) or (T,)
            delta_t: Cumulative time, shape (B, T) or (T,)
            delta_t_interval: Time intervals, shape (B, T) or (T,)
            error: Optional flux errors
            context_mask: Optional observation mask
            return_tokens: If True, also return encoding output

        Returns:
            LightCurveOutput with reconstructed light curve, optionally with EncodingOutput

        Example:
            >>> output = codec.reconstruct(flux, delta_t, delta_t_interval)
            >>> print(output.flux.shape)  # Same as input
        """
        # Encode to get discrete tokens
        encoding_out = self.encode(flux, delta_t, delta_t_interval, error, context_mask)

        # Decode from discrete tokens
        target_length = flux.shape[-1] if flux.dim() > 1 else flux.shape[0]
        output = self.decode(encoding_out.tokens, target_length=target_length)

        if return_tokens:
            return output, encoding_out
        return output

    def get_codebook_usage(self, tokens: torch.Tensor) -> float:
        """
        Calculate codebook usage statistics for a batch of tokens.

        Args:
            tokens: Token indices, shape (B, N) or (N,)

        Returns:
            Fraction of unique codes used (0.0 to 1.0)
        """
        if tokens.dim() > 1:
            tokens = tokens.flatten()
        unique_codes = torch.unique(tokens).numel()
        total_codes = self.quantizer.codebook_size
        return unique_codes / total_codes

    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size (total number of possible codes)."""
        return self.quantizer.codebook_size

    @property
    def embed_dim(self) -> int:
        """Returns the embedding dimension."""
        return self._embed_dim

    def to_dict(self) -> dict:
        """Export codec configuration to a dictionary."""
        return {
            "embed_dim": self.embed_dim,
            "levels": self.levels,
            "vocab_size": self.vocab_size,
            "fixed_length": self.fixed_length,
            "config": self.config,
        }

    def save_config(self, path: str | Path) -> None:
        """Save codec configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_codec_from_phase1(
    phase1_checkpoint: str | Path,
    output_path: Optional[str | Path] = None,
) -> LightCurveCodecInference:
    """
    Create a Phase 2 codec from a Phase 1 checkpoint (for migration/testing).

    This is useful for:
    1. Testing the inference pipeline before Phase 2 training
    2. Comparing Phase 1 (continuous) vs Phase 2 (quantized) outputs

    Args:
        phase1_checkpoint: Path to Phase 1 checkpoint
        output_path: Optional path to save the new Phase 2 style checkpoint

    Returns:
        LightCurveCodecInference instance (with continuous latent, no quantization)
    """
    raise NotImplementedError(
        "Phase 1 to Phase 2 conversion requires architectural changes. "
        "Use from_checkpoint() with a Phase 2 checkpoint instead."
    )
