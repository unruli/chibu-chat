"""
HTSAT: Hierarchical Token Semantic Audio Transformer Encoder

This module implements the HTSAT audio encoder for extracting hierarchical
audio representations. HTSAT uses a Swin Transformer backbone with specialized
audio processing for robust feature extraction.

Reference:
    "HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound
    Classification and Detection" (ICASSP 2022)
    https://arxiv.org/abs/2202.00874

Usage:
    encoder = HTSATEncoder()
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
    from transformers import AutoFeatureExtractor, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


@dataclass
class HTSATConfig:
    """Configuration for HTSAT Encoder."""
    model_name: str = "lukewys/laion_clap_htsat_unfused"
    sample_rate: int = 32000  # HTSAT typically uses 32kHz
    duration: int = 10  # seconds
    mel_bins: int = 64
    fmin: int = 50
    fmax: int = 14000
    window_size: int = 1024
    hop_size: int = 320
    output_dim: int = 768  # HTSAT-base output dimension
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    use_projection: bool = True  # Project to custom dimension
    projection_dim: int = 1280  # Match nanochat model_dim for d20


class HTSATEncoder(nn.Module):
    """
    Hierarchical Token Semantic Audio Transformer Encoder.
    
    Extracts hierarchical audio representations using HTSAT architecture.
    Supports loading from HuggingFace or custom checkpoints.
    
    Attributes:
        config: HTSATConfig with model parameters
        model: The underlying HTSAT model
        feature_extractor: Audio preprocessing pipeline
        projection: Optional projection layer to match LLM dimensions
    """
    
    def __init__(
        self,
        config: Optional[HTSATConfig] = None,
        model_path: Optional[str] = None,
        freeze_encoder: bool = True,
    ):
        """
        Initialize HTSAT Encoder.
        
        Args:
            config: HTSATConfig object. If None, uses defaults.
            model_path: Path to custom checkpoint. If None, loads from HuggingFace.
            freeze_encoder: Whether to freeze encoder weights (recommended for LLM training).
        """
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
        
        self.config = config or HTSATConfig()
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
        """Load HTSAT model from HuggingFace or local checkpoint."""
        if model_path and os.path.exists(model_path):
            # Load from local checkpoint
            print(f"Loading HTSAT from local checkpoint: {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            self.model = self._build_htsat_model()
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.feature_extractor = self._build_feature_extractor()
        else:
            # Load from HuggingFace
            print(f"Loading HTSAT from HuggingFace: {self.config.model_name}")
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Failed to load from HuggingFace: {e}")
                print("Falling back to CLAP HTSAT model...")
                self._load_clap_htsat()
    
    def _load_clap_htsat(self):
        """Load HTSAT via LAION-CLAP."""
        try:
            from transformers import ClapModel, ClapProcessor
            
            clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            self.model = clap_model.audio_model
            self.feature_extractor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            self.config.output_dim = 768
        except Exception as e:
            raise RuntimeError(f"Failed to load HTSAT model: {e}")
    
    def _build_feature_extractor(self):
        """Build mel spectrogram feature extractor."""
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.window_size,
            hop_length=self.config.hop_size,
            n_mels=self.config.mel_bins,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
        )
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        print("HTSAT encoder weights frozen")
    
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
            print("All HTSAT encoder weights unfrozen")
        else:
            # Unfreeze last N layers (implementation depends on model structure)
            layers = list(self.model.children())
            for layer in layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Last {unfreeze_layers} HTSAT layers unfrozen")
    
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
            Audio waveform tensor of shape (1, num_samples)
        """
        target_sr = target_sr or self.config.sample_rate
        
        if HAS_LIBROSA:
            waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        else:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
        
        return waveform
    
    def preprocess(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Preprocess audio waveform for HTSAT.
        
        Args:
            waveform: Audio tensor of shape (batch, samples) or (samples,)
            sample_rate: Sample rate of the input audio
            
        Returns:
            Preprocessed features ready for the model
        """
        # Ensure correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Resample if necessary
        if sample_rate != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.config.sample_rate
            ).to(waveform.device)
            waveform = resampler(waveform)
        
        # Pad or truncate to target duration
        target_length = self.config.sample_rate * self.config.duration
        current_length = waveform.shape[-1]
        
        if current_length > target_length:
            waveform = waveform[..., :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            waveform = F.pad(waveform, (0, padding))
        
        return waveform
    
    @torch.no_grad()
    def encode(
        self,
        audio_input: Union[str, torch.Tensor, List[str]],
        sample_rate: int = 16000,
        return_tokens: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode audio to feature representations.
        
        Args:
            audio_input: Audio file path, waveform tensor, or list of paths
            sample_rate: Sample rate if waveform tensor is provided
            return_tokens: If True, also returns hierarchical token features
            
        Returns:
            Audio features of shape (batch, seq_len, projection_dim)
            If return_tokens=True, also returns token features
        """
        # Handle different input types
        if isinstance(audio_input, str):
            waveform = self.load_audio(audio_input)
            sample_rate = self.config.sample_rate
        elif isinstance(audio_input, list):
            waveforms = [self.load_audio(p) for p in audio_input]
            waveform = torch.cat(waveforms, dim=0)
            sample_rate = self.config.sample_rate
        else:
            waveform = audio_input
        
        # Preprocess
        waveform = self.preprocess(waveform, sample_rate)
        waveform = waveform.to(self.config.device, dtype=self.config.dtype)
        
        # Extract features using the model
        if hasattr(self.feature_extractor, '__call__'):
            # HuggingFace processor
            inputs = self.feature_extractor(
                waveform.cpu().numpy(),
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
            # Get the appropriate output
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state
            elif hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output.unsqueeze(1)
            else:
                features = outputs[0]
        else:
            # Direct model forward
            features = self.model(waveform)
        
        # Apply projection
        projected = self.projection(features)
        
        if return_tokens:
            return projected, features
        return projected
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        if self.config.use_projection:
            return self.config.projection_dim
        return self.config.output_dim
    
    def get_num_tokens(self, duration: float) -> int:
        """
        Estimate number of output tokens for a given audio duration.
        
        Args:
            duration: Audio duration in seconds
            
        Returns:
            Estimated number of output tokens
        """
        # HTSAT produces tokens based on the spectrogram temporal resolution
        num_frames = int(duration * self.config.sample_rate / self.config.hop_size)
        # Account for Swin Transformer pooling (typically 4x reduction per stage)
        num_tokens = num_frames // 16  # Approximate based on HTSAT architecture
        return max(1, num_tokens)
    
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
            # Need to re-implement without @torch.no_grad for training
            if isinstance(audio_input, str):
                waveform = self.load_audio(audio_input)
                sample_rate = self.config.sample_rate
            else:
                waveform = audio_input
            
            waveform = self.preprocess(waveform, sample_rate)
            waveform = waveform.to(self.config.device, dtype=self.config.dtype)
            
            if hasattr(self.feature_extractor, '__call__'):
                inputs = self.feature_extractor(
                    waveform.cpu().numpy(),
                    sampling_rate=self.config.sample_rate,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                
                if hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state
                else:
                    features = outputs[0]
            else:
                features = self.model(waveform)
            
            return self.projection(features)


def test_htsat_encoder():
    """Test HTSAT encoder functionality."""
    print("Testing HTSAT Encoder...")
    
    # Create encoder
    config = HTSATConfig(
        use_projection=True,
        projection_dim=1280,
    )
    
    try:
        encoder = HTSATEncoder(config=config)
        print(f"✓ Encoder initialized")
        print(f"  - Output dimension: {encoder.get_output_dim()}")
        print(f"  - Device: {encoder.config.device}")
        
        # Test with dummy audio
        dummy_waveform = torch.randn(1, 32000 * 10)  # 10 seconds at 32kHz
        features = encoder.encode(dummy_waveform, sample_rate=32000)
        print(f"✓ Encoding successful")
        print(f"  - Input shape: {dummy_waveform.shape}")
        print(f"  - Output shape: {features.shape}")
        
        # Test token estimation
        tokens_10s = encoder.get_num_tokens(10.0)
        print(f"  - Estimated tokens for 10s audio: {tokens_10s}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_htsat_encoder()
