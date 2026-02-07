"""
Whisper Large v3 Audio Encoder

This module implements the Whisper Large v3 encoder for extracting rich
audio representations. Whisper's encoder is particularly effective for
speech understanding tasks due to its training on 680,000 hours of
multilingual audio data.

Reference:
    "Robust Speech Recognition via Large-Scale Weak Supervision" (OpenAI 2022)
    https://arxiv.org/abs/2212.04356

Usage:
    encoder = WhisperLargeV3Encoder()
    audio_features = encoder.encode("path/to/audio.wav")
    # or
    audio_features = encoder.encode(waveform_tensor, sample_rate=16000)
"""

import os
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

try:
    from transformers import WhisperProcessor, WhisperModel, WhisperFeatureExtractor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


@dataclass
class WhisperEncoderConfig:
    """Configuration for Whisper Large v3 Encoder."""
    model_name: str = "openai/whisper-large-v3"
    sample_rate: int = 16000  # Whisper uses 16kHz
    max_duration: int = 30  # seconds (Whisper's max context)
    n_mels: int = 128  # Whisper large v3 uses 128 mel bins
    output_dim: int = 1280  # Whisper large encoder dimension
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    use_projection: bool = True  # Project to custom dimension
    projection_dim: int = 1280  # Match nanochat model_dim for d20
    use_flash_attention: bool = True  # Use flash attention if available
    chunk_length: int = 30  # Chunk length for long audio processing


class WhisperLargeV3Encoder(nn.Module):
    """
    Whisper Large v3 Audio Encoder.
    
    Extracts rich audio representations using OpenAI's Whisper encoder.
    The encoder produces 1500 tokens for 30 seconds of audio (50 tokens/sec).
    
    Attributes:
        config: WhisperEncoderConfig with model parameters
        model: The Whisper encoder model
        processor: Audio preprocessing pipeline
        projection: Optional projection layer to match LLM dimensions
    """
    
    def __init__(
        self,
        config: Optional[WhisperEncoderConfig] = None,
        model_path: Optional[str] = None,
        freeze_encoder: bool = True,
    ):
        """
        Initialize Whisper Large v3 Encoder.
        
        Args:
            config: WhisperEncoderConfig object. If None, uses defaults.
            model_path: Path to custom checkpoint. If None, loads from HuggingFace.
            freeze_encoder: Whether to freeze encoder weights (recommended for LLM training).
        """
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
        
        self.config = config or WhisperEncoderConfig()
        self.freeze_encoder = freeze_encoder
        
        # Load model
        self._load_model(model_path)
        
        # Optional projection layer to match LLM embedding dimension
        if self.config.use_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.config.output_dim, self.config.projection_dim),
                nn.LayerNorm(self.config.projection_dim),
                nn.GELU(),
            )
        else:
            self.projection = nn.Identity()
        
        # Move to device
        self.to(self.config.device)
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            self._freeze_encoder()
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load Whisper model from HuggingFace or local checkpoint."""
        if model_path and os.path.exists(model_path):
            # Load from local checkpoint
            print(f"Loading Whisper from local checkpoint: {model_path}")
            self.processor = WhisperProcessor.from_pretrained(model_path)
            full_model = WhisperModel.from_pretrained(
                model_path,
                torch_dtype=self.config.dtype,
            )
            self.model = full_model.encoder
        else:
            # Load from HuggingFace
            print(f"Loading Whisper from HuggingFace: {self.config.model_name}")
            
            # Configure for flash attention if available
            model_kwargs = {
                "torch_dtype": self.config.dtype,
            }
            
            if self.config.use_flash_attention:
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    print("Flash attention not available, using default attention")
            
            self.processor = WhisperProcessor.from_pretrained(self.config.model_name)
            full_model = WhisperModel.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            # Extract only the encoder
            self.model = full_model.encoder
            
            # Update output_dim based on actual model
            self.config.output_dim = self.model.config.d_model
            print(f"Whisper encoder output dimension: {self.config.output_dim}")
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        print("Whisper encoder weights frozen")
    
    def unfreeze_encoder(self, unfreeze_layers: Optional[int] = None):
        """
        Unfreeze encoder parameters for fine-tuning.
        
        Args:
            unfreeze_layers: Number of final layers to unfreeze.
                           If None, unfreezes all layers.
        """
        if unfreeze_layers is None:
            for param in self.model.parameters():
                param.requires_grad = True
            print("All Whisper encoder weights unfrozen")
        else:
            # Unfreeze last N transformer layers
            num_layers = len(self.model.layers)
            start_layer = max(0, num_layers - unfreeze_layers)
            
            for i, layer in enumerate(self.model.layers):
                if i >= start_layer:
                    for param in layer.parameters():
                        param.requires_grad = True
            
            # Also unfreeze layer norm
            for param in self.model.layer_norm.parameters():
                param.requires_grad = True
            
            print(f"Last {unfreeze_layers} Whisper encoder layers unfrozen")
    
    def load_audio(
        self,
        audio_path: str,
        target_sr: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load audio file and resample if necessary.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate. If None, uses config.sample_rate.
            
        Returns:
            Audio waveform tensor of shape (num_samples,)
        """
        target_sr = target_sr or self.config.sample_rate
        
        if HAS_LIBROSA:
            waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            waveform = torch.from_numpy(waveform)
        else:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
        
        return waveform
    
    def preprocess(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        padding: str = "longest",
        return_attention_mask: bool = True,
    ) -> dict:
        """
        Preprocess audio waveform for Whisper.
        
        Args:
            waveform: Audio tensor of shape (batch, samples) or (samples,)
            sample_rate: Sample rate of the input audio
            padding: Padding strategy ("longest", "max_length", or "do_not_pad")
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Dictionary with input_features and optionally attention_mask
        """
        # Ensure numpy array for processor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        # Handle batch dimension
        if waveform.ndim == 1:
            waveform = [waveform]
        elif waveform.ndim == 2:
            waveform = list(waveform)
        
        # Use Whisper processor
        inputs = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=padding,
            return_attention_mask=return_attention_mask,
        )
        
        return inputs
    
    def _chunk_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> List[torch.Tensor]:
        """
        Split long audio into chunks for processing.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            List of audio chunks
        """
        chunk_samples = self.config.chunk_length * sample_rate
        total_samples = waveform.shape[-1]
        
        if total_samples <= chunk_samples:
            return [waveform]
        
        chunks = []
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[..., start:end]
            
            # Pad last chunk if necessary
            if chunk.shape[-1] < chunk_samples:
                padding = chunk_samples - chunk.shape[-1]
                chunk = F.pad(chunk, (0, padding))
            
            chunks.append(chunk)
        
        return chunks
    
    @torch.no_grad()
    def encode(
        self,
        audio_input: Union[str, torch.Tensor, List[str]],
        sample_rate: int = 16000,
        return_attention_mask: bool = False,
        pool_output: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode audio to feature representations.
        
        Args:
            audio_input: Audio file path, waveform tensor, or list of paths
            sample_rate: Sample rate if waveform tensor is provided
            return_attention_mask: If True, also returns attention mask
            pool_output: If True, returns pooled (mean) representation
            
        Returns:
            Audio features of shape (batch, seq_len, projection_dim)
            If pool_output=True: (batch, projection_dim)
            If return_attention_mask=True, also returns attention mask
        """
        # Handle different input types
        if isinstance(audio_input, str):
            waveform = self.load_audio(audio_input)
            sample_rate = self.config.sample_rate
        elif isinstance(audio_input, list):
            waveforms = [self.load_audio(p) for p in audio_input]
            # Pad to same length
            max_len = max(w.shape[-1] for w in waveforms)
            waveforms = [F.pad(w, (0, max_len - w.shape[-1])) for w in waveforms]
            waveform = torch.stack(waveforms)
            sample_rate = self.config.sample_rate
        else:
            waveform = audio_input
        
        # Check if we need to chunk long audio
        max_samples = self.config.max_duration * self.config.sample_rate
        if waveform.shape[-1] > max_samples:
            return self._encode_long_audio(waveform, sample_rate, pool_output)
        
        # Preprocess
        inputs = self.preprocess(waveform, sample_rate)
        input_features = inputs["input_features"].to(
            self.config.device, dtype=self.config.dtype
        )
        
        # Forward through encoder
        encoder_outputs = self.model(input_features)
        hidden_states = encoder_outputs.last_hidden_state
        
        # Apply projection
        projected = self.projection(hidden_states)
        
        # Optional pooling
        if pool_output:
            projected = projected.mean(dim=1)
        
        if return_attention_mask and "attention_mask" in inputs:
            return projected, inputs["attention_mask"].to(self.config.device)
        return projected
    
    def _encode_long_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        pool_output: bool = False,
    ) -> torch.Tensor:
        """
        Encode audio longer than max_duration by chunking.
        
        Args:
            waveform: Long audio waveform
            sample_rate: Sample rate
            pool_output: Whether to pool the output
            
        Returns:
            Concatenated or pooled features from all chunks
        """
        chunks = self._chunk_audio(waveform, sample_rate)
        
        all_features = []
        for chunk in chunks:
            inputs = self.preprocess(chunk, sample_rate)
            input_features = inputs["input_features"].to(
                self.config.device, dtype=self.config.dtype
            )
            
            encoder_outputs = self.model(input_features)
            hidden_states = encoder_outputs.last_hidden_state
            projected = self.projection(hidden_states)
            all_features.append(projected)
        
        # Concatenate along sequence dimension
        combined = torch.cat(all_features, dim=1)
        
        if pool_output:
            return combined.mean(dim=1)
        return combined
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        if self.config.use_projection:
            return self.config.projection_dim
        return self.config.output_dim
    
    def get_num_tokens(self, duration: float) -> int:
        """
        Get the number of output tokens for a given audio duration.
        
        Whisper produces 1500 tokens for 30 seconds = 50 tokens/second.
        
        Args:
            duration: Audio duration in seconds
            
        Returns:
            Number of output tokens
        """
        tokens_per_second = 50  # Whisper's fixed rate
        return int(duration * tokens_per_second)
    
    def forward(
        self,
        audio_input: Union[str, torch.Tensor],
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Forward pass for training (with gradients if encoder is unfrozen).
        
        Args:
            audio_input: Audio file path or waveform tensor
            sample_rate: Sample rate if waveform tensor is provided
            
        Returns:
            Audio features of shape (batch, seq_len, projection_dim)
        """
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encode(audio_input, sample_rate)
        else:
            # Re-implement without @torch.no_grad for training
            if isinstance(audio_input, str):
                waveform = self.load_audio(audio_input)
                sample_rate = self.config.sample_rate
            else:
                waveform = audio_input
            
            inputs = self.preprocess(waveform, sample_rate)
            input_features = inputs["input_features"].to(
                self.config.device, dtype=self.config.dtype
            )
            
            encoder_outputs = self.model(input_features)
            hidden_states = encoder_outputs.last_hidden_state
            
            return self.projection(hidden_states)
    
    def get_audio_token_embeddings(
        self,
        audio_input: Union[str, torch.Tensor],
        sample_rate: int = 16000,
        num_query_tokens: int = 32,
    ) -> torch.Tensor:
        """
        Get fixed-length audio token embeddings using learned queries.
        
        This is useful for architectures that need a fixed number of
        audio tokens regardless of audio duration (like Q-Former).
        
        Args:
            audio_input: Audio file path or waveform tensor
            sample_rate: Sample rate if waveform tensor is provided
            num_query_tokens: Number of output tokens to produce
            
        Returns:
            Audio token embeddings of shape (batch, num_query_tokens, projection_dim)
        """
        # Get variable-length features
        features = self.encode(audio_input, sample_rate)
        
        # Simple approach: use adaptive pooling to get fixed tokens
        # For production, you'd want to use learned query tokens (Q-Former style)
        batch_size = features.shape[0]
        features_transposed = features.transpose(1, 2)  # (batch, dim, seq_len)
        pooled = F.adaptive_avg_pool1d(features_transposed, num_query_tokens)
        return pooled.transpose(1, 2)  # (batch, num_query_tokens, dim)


class WhisperAudioTokenizer:
    """
    Utility class to convert audio to discrete tokens using Whisper.
    
    This can be used for audio tokenization in language model training.
    """
    
    def __init__(
        self,
        encoder: Optional[WhisperLargeV3Encoder] = None,
        codebook_size: int = 8192,
        num_codebooks: int = 4,
    ):
        """
        Initialize the audio tokenizer.
        
        Args:
            encoder: Pre-initialized Whisper encoder. If None, creates new one.
            codebook_size: Size of the quantization codebook
            num_codebooks: Number of residual vector quantization codebooks
        """
        self.encoder = encoder or WhisperLargeV3Encoder()
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        
        # Simple random codebook initialization
        # In production, train these with VQ-VAE or RVQ
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, self.encoder.get_output_dim()))
            for _ in range(num_codebooks)
        ])
    
    def tokenize(
        self,
        audio_input: Union[str, torch.Tensor],
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Convert audio to discrete tokens.
        
        Args:
            audio_input: Audio file path or waveform tensor
            sample_rate: Sample rate
            
        Returns:
            Token indices of shape (batch, seq_len, num_codebooks)
        """
        features = self.encoder.encode(audio_input, sample_rate)
        
        all_indices = []
        residual = features
        
        for codebook in self.codebooks:
            # Find nearest codebook entry
            distances = torch.cdist(residual, codebook.unsqueeze(0))
            indices = distances.argmin(dim=-1)
            all_indices.append(indices)
            
            # Update residual
            quantized = F.embedding(indices, codebook)
            residual = residual - quantized
        
        return torch.stack(all_indices, dim=-1)


def test_whisper_encoder():
    """Test Whisper encoder functionality."""
    print("Testing Whisper Large v3 Encoder...")
    
    # Create encoder
    config = WhisperEncoderConfig(
        use_projection=True,
        projection_dim=1280,
        use_flash_attention=False,  # Disable for compatibility
    )
    
    try:
        encoder = WhisperLargeV3Encoder(config=config)
        print(f"✓ Encoder initialized")
        print(f"  - Model: {config.model_name}")
        print(f"  - Output dimension: {encoder.get_output_dim()}")
        print(f"  - Device: {encoder.config.device}")
        
        # Test with dummy audio (16kHz, 5 seconds)
        dummy_waveform = torch.randn(16000 * 5)
        features = encoder.encode(dummy_waveform, sample_rate=16000)
        print(f"✓ Encoding successful")
        print(f"  - Input: 5 seconds of audio at 16kHz")
        print(f"  - Output shape: {features.shape}")
        
        # Test token estimation
        tokens_30s = encoder.get_num_tokens(30.0)
        print(f"  - Tokens for 30s audio: {tokens_30s}")
        
        # Test pooled output
        pooled = encoder.encode(dummy_waveform, sample_rate=16000, pool_output=True)
        print(f"  - Pooled output shape: {pooled.shape}")
        
        # Test fixed-length tokens
        fixed_tokens = encoder.get_audio_token_embeddings(
            dummy_waveform, sample_rate=16000, num_query_tokens=32
        )
        print(f"  - Fixed tokens shape: {fixed_tokens.shape}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_whisper_encoder()
