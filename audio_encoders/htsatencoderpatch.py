"""
HTSATEncoderPatched - A patched HTSAT encoder for flexible training workflows.

This class extends HTSATEncoder to support multiple training modes with explicit
control over gradient flow and feature extraction:

Attributes:
    config (HTSATPatchedConfig): Configuration object with training parameters.
    freeze_encoder (bool): Whether to freeze the encoder backbone.
    train_projection_when_frozen (bool): Whether to train projection head when 
        encoder is frozen.

Methods:
    _extract_features(waveform: torch.Tensor) -> torch.Tensor
        Runs the feature extractor and HTSAT forward pass to extract encoder features.
        Handles multiple output formats from the underlying model.
        
        Args:
            waveform: Input audio waveform tensor.
            
        Returns:
            Extracted feature tensor from the model output.
    
    encode(audio_input, sample_rate, return_tokens, inference_mode) 
        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Encodes audio input into HTSAT features with explicit inference/training control.
        Supports multiple input formats (file paths, tensors, or lists of paths).
        Respects gradient requirements based on inference_mode and freeze_encoder settings.
        
        Args:
            audio_input: Audio input (file path string, tensor, or list of paths).
            sample_rate: Sample rate for audio processing.
            return_tokens: If True, returns both projected features and raw features.
            inference_mode: If True, disables gradients; if False, respects freeze settings.
            
        Returns:
            Projected features, optionally with raw features if return_tokens=True.
    
    forward(audio_input, sample_rate) -> torch.Tensor
        Forward pass supporting three training modes:
        - Mode A: Frozen encoder + trainable projection head
        - Mode B: Fully frozen (inference)
        - Mode C: Full/partial encoder fine-tuning (unfrozen)
        
        Args:
            audio_input: Audio input (file path or tensor).
            sample_rate: Sample rate for audio processing.
            
        Returns:
            Projected feature embeddings.
Patched HTSAT encoder module for accent-aware training workflows.

This file keeps the original encoder intact and provides patched classes that:
- support projection-head-only training when the encoder is frozen
- optionally run full/partial encoder fine-tuning
- expose explicit inference_mode control in encode()
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from audio_encoders.htsat_encoder import HTSATConfig, HTSATEncoder


@dataclass
class HTSATPatchedConfig(HTSATConfig):
    """HTSAT config with extra control for frozen-encoder training."""

    train_projection_when_frozen: bool = True


class HTSATEncoderPatched(HTSATEncoder):
    """
    Patched HTSAT encoder with cleaner training/inference modes.

    Modes:
    - frozen encoder + trainable projection (default)
    - fully frozen inference
    - full/partial fine-tuning when unfreezing encoder
    """

    def __init__(
        self,
        config: Optional[HTSATPatchedConfig] = None,
        model_path: Optional[str] = None,
        freeze_encoder: bool = True,
    ):
        cfg = config or HTSATPatchedConfig()
        super().__init__(config=cfg, model_path=model_path, freeze_encoder=freeze_encoder)
        self.train_projection_when_frozen = getattr(cfg, "train_projection_when_frozen", True)

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Run feature extractor + HTSAT forward and return encoder features.
        """
        if hasattr(self.feature_extractor, "__call__"):
            inputs = self.feature_extractor(
                waveform.cpu().numpy(),
                sampling_rate=self.config.sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            if hasattr(outputs, "pooler_output"):
                return outputs.pooler_output.unsqueeze(1)
            return outputs[0]

        return self.model(waveform)

    def encode(
        self,
        audio_input: Union[str, torch.Tensor, List[str]],
        sample_rate: int = 16000,
        return_tokens: bool = False,
        inference_mode: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode audio into HTSAT features with explicit grad/no-grad behavior.
        """
        if isinstance(audio_input, str):
            waveform = self.load_audio(audio_input)
            sample_rate = self.config.sample_rate
        elif isinstance(audio_input, list):
            waveforms = [self.load_audio(p) for p in audio_input]
            waveform = torch.cat(waveforms, dim=0)
            sample_rate = self.config.sample_rate
        else:
            waveform = audio_input

        waveform = self.preprocess(waveform, sample_rate)
        waveform = waveform.to(self.config.device, dtype=self.config.dtype)

        if inference_mode:
            with torch.no_grad():
                features = self._extract_features(waveform)
                projected = self.projection(features)
        elif self.freeze_encoder and self.train_projection_when_frozen:
            # Keep encoder frozen but let projection receive gradients.
            with torch.no_grad():
                features = self._extract_features(waveform)
            projected = self.projection(features)
        elif self.freeze_encoder and not self.train_projection_when_frozen:
            with torch.no_grad():
                features = self._extract_features(waveform)
                projected = self.projection(features)
        else:
            features = self._extract_features(waveform)
            projected = self.projection(features)

        if return_tokens:
            return projected, features
        return projected

    def forward(
        self,
        audio_input: Union[str, torch.Tensor],
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Forward pass supporting frozen-head training and full fine-tuning.
        """
        if isinstance(audio_input, str):
            waveform = self.load_audio(audio_input)
            sample_rate = self.config.sample_rate
        else:
            waveform = audio_input

        waveform = self.preprocess(waveform, sample_rate)
        waveform = waveform.to(self.config.device, dtype=self.config.dtype)

        # Mode A: frozen encoder, train projection head only.
        if self.freeze_encoder and self.train_projection_when_frozen:
            with torch.no_grad():
                features = self._extract_features(waveform)
            return self.projection(features)

        # Mode B: fully frozen inference behavior.
        if self.freeze_encoder and not self.train_projection_when_frozen:
            with torch.no_grad():
                features = self._extract_features(waveform)
                return self.projection(features)

        # Mode C: full/partial HTSAT fine-tuning.
        features = self._extract_features(waveform)
        return self.projection(features)
