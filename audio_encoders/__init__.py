"""
Audio Encoders for Chibu-Chat

This module provides audio encoding capabilities using:
- HTSAT (Hierarchical Token Semantic Audio Transformer)
- Whisper Large v3 (OpenAI's speech encoder)

Both encoders extract rich audio representations that can be used
for multimodal language model training and inference.
"""

from .htsat_encoder import HTSATEncoder
from .whisper_encoder import WhisperLargeV3Encoder

__all__ = ["HTSATEncoder", "WhisperLargeV3Encoder"]
