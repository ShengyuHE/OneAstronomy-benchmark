from pathlib import Path
from typing import Optional, Type

import torch
from jaxtyping import Float, Integer

from aion.codecs.base import Codec
from aion.codecs.curve_light_tokenizer.light_curve_codec_inference import (
    EncodingOutput,
    LightCurveCodecInference,
    LightCurveOutput,
)
from aion.codecs.quantizers import Quantizer
from aion.modalities import LightCurveM3


class LightCurveM3Codec(Codec):
    """Codec for LightCurveM3 modality using AstroM3 Phase 2 FSQ model.

    This codec wraps LightCurveCodecInference to provide a standard Codec interface
    for encoding and decoding light curve data.
    """

    DEFAULT_CHECKPOINT_PATH = "/mnt/nanhu_200TBnas/default/zwg/weight/astrom3_p2_fsq_1600/best_model.pt"

    def __init__(self, inference_codec: LightCurveCodecInference):
        super().__init__()
        self._inference_codec = inference_codec

    @property
    def modality(self) -> Type[LightCurveM3]:
        return LightCurveM3

    @property
    def quantizer(self) -> Quantizer:
        return self._inference_codec.quantizer

    def _encode(self, x: LightCurveM3) -> Float[torch.Tensor, "b c n_tokens"]:
        encoding_output = self._inference_codec.encode(
            flux=x.flux,
            delta_t=x.delta_t,
            delta_t_interval=x.delta_t_interval,
            error=x.error,
            context_mask=x.context_mask,
        )
        return encoding_output.latent

    def _decode(
        self,
        z: Float[torch.Tensor, "b c n_tokens"],
        **metadata,
    ) -> LightCurveM3:
        target_length = metadata.get("target_length", self._inference_codec.fixed_length)
        codes = self.quantizer.encode(z)
        light_curve_output = self._inference_codec.decode(codes, target_length=target_length)
        return LightCurveM3(
            flux=light_curve_output.flux,
            delta_t=light_curve_output.delta_t,
            delta_t_interval=light_curve_output.delta_t_interval,
        )

    def encode(self, x: LightCurveM3) -> Integer[torch.Tensor, "b n_tokens"]:
        if not isinstance(x, self.modality):
            raise ValueError(
                f"Input type {type(x).__name__} does not match the modality of the codec {self.modality.__name__}"
            )
        encoding_output = self._inference_codec.encode(
            flux=x.flux,
            delta_t=x.delta_t,
            delta_t_interval=x.delta_t_interval,
            error=x.error,
            context_mask=x.context_mask,
        )
        return encoding_output.tokens

    def decode(
        self,
        z: Integer[torch.Tensor, "b n_tokens"],
        **metadata,
    ) -> LightCurveM3:
        target_length = metadata.get("target_length", self._inference_codec.fixed_length)
        light_curve_output = self._inference_codec.decode(z, target_length=target_length)
        return LightCurveM3(
            flux=light_curve_output.flux,
            delta_t=light_curve_output.delta_t,
            delta_t_interval=light_curve_output.delta_t_interval,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        device: str = "cpu",
        **kwargs,
    ) -> "LightCurveM3Codec":
        if checkpoint_path is None:
            checkpoint_path = cls.DEFAULT_CHECKPOINT_PATH
        inference_codec = LightCurveCodecInference.from_checkpoint(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
        )
        return cls(inference_codec=inference_codec)
