from cleanunet import CleanUNet
import torchaudio
import torch
import numpy as np
from typing import Union, Optional
import os
from denoiser.utils import chunk_audio, unchunk_audio
from tqdm import tqdm

class DenoiserAudio():
    """
    Denoiser module.
    
    Args:
        device (str, Optional): Device to use. Defaults to GPU if available.
        chunk_length_s (int, Optional): Length of a single chunk in second.
        max_batch_size (int, Optional): Maximum size of a batch to infer in one instance.
    """
    
    def __init__(self,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 chunk_length_s: int = 3,
                 max_batch_size: int = 20,
                 verbose: bool = True) -> None:
        
        self.device = device
        self.chunk_length_s = chunk_length_s
        self.max_batch_size = max_batch_size
        self.verbose = verbose
        
        if self.verbose:
            print(f"Initializing DenoiserAudio with device={device}, chunk_length_s={chunk_length_s}, max_batch_size={max_batch_size}")
        
        self.model = CleanUNet.from_pretrained(device=device)
        
    @staticmethod
    def load_audio_and_resample(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """
        Loads audio and resamples to target sampling rate. Returns single channel.

        Args:
            audio_path (str): Path to audio file.
            target_sr (int, optional): Target sampling rate. Defaults to 16000.

        Returns:
            torch.Tensor: Tensor of audio file, returns single channel only.
        """
        try:
            audio_wav, s_rate = torchaudio.load(audio_path)
            
            # If multi-channel, convert to mono
            if audio_wav.shape[0] > 1:
                audio_wav = torch.mean(audio_wav, dim=0, keepdim=True)
                
            # Resample if needed
            if s_rate != target_sr:
                resampler = torchaudio.transforms.Resample(s_rate, target_sr)
                audio_wav = resampler(audio_wav)
                
            return audio_wav[0]  # Return single channel
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {str(e)}")
    
    @staticmethod
    def audio_processing(audio_wav: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Process audio. Trims all zeros.

        Args:
            audio_wav (torch.Tensor | np.ndarray): Audio wave loaded.

        Raises:
            TypeError: If file is not a Tensor or an Array.

        Returns:
            np.ndarray: Zero trimmed audio array. In future, vad will be used.
        """
        if isinstance(audio_wav, torch.Tensor):
            # Ensure it's on CPU and convert to numpy
            audio_wav = audio_wav.cpu().detach().numpy()
            return np.trim_zeros(audio_wav)
        elif isinstance(audio_wav, np.ndarray):
            return np.trim_zeros(audio_wav)
        else:
            raise TypeError(f'Only supports numpy.ndarray or torch.Tensor file types. Got {type(audio_wav)}')
    
    def denoise(self, audio_chunks: torch.Tensor, max_batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Denoises noisy audio chunks.

        Args:
            audio_chunks (torch.Tensor): Tensor of all noisy audio chunks.
            max_batch_size (int, optional): Same as for model initialization. Defaults to None.

        Returns:
            torch.Tensor: Denoised audio tensors.
        """
        if max_batch_size is None:
            max_batch_size = self.max_batch_size
        
        # Ensure audio_chunks has the right shape for the model
        if audio_chunks.dim() == 2:
            # Convert (num_chunks, chunk_length) to (num_chunks, 1, chunk_length)
            audio_chunks = audio_chunks.unsqueeze(1)
            
        num_chunks = audio_chunks.shape[0]
        batches = torch.split(audio_chunks, max_batch_size)
        
        if self.verbose:
            print("*" * 20)
            print(f"Total number of chunks: {num_chunks}")
            print(f"Processing {len(batches)} batches with up to {max_batch_size} chunks each")
            print("Audio chunks shape:", audio_chunks.shape)
            print("*" * 20)
        
        denoised_audio = []
        for batch in tqdm(batches) if self.verbose else batches:
            # Prepare batch for model - should be (batch_size, chunk_length)
            # CleanUNet expects (batch_size, waveform) format
            if batch.dim() == 3:  # (batch_size, 1, chunk_length)
                batch_input = batch.squeeze(1)
            else:
                batch_input = batch
                
            # Move to device
            batch_input = batch_input.to(self.device)
            
            # Process with model
            with torch.no_grad():
                batch_output = self.model(batch_input)
            
            # Prepare output for unchunk_audio
            batch_output = batch_output.unsqueeze(1)  # Add channel dimension
            
            # Move back to CPU and free GPU memory
            batch_output = batch_output.to('cpu')
            torch.cuda.empty_cache()
            
            # Unchunk this batch
            denoised_batch = unchunk_audio(batch_output)
            denoised_audio.append(denoised_batch)
            
        # Combine all batches
        denoised_audio = torch.cat(denoised_audio, dim=1)
        return denoised_audio
    
    def __call__(
        self, 
        noisy_audio_path: str,
        output_path: Optional[str] = None,
        target_sr: int = 16000
    ) -> np.ndarray:
        """
        Denoises a given audio file.

        Args:
            noisy_audio_path (str): Path to the audio file.
            output_path (str, optional): Path to save the output. If None, don't save.
            target_sr (int, optional): Target sampling rate. Defaults to 16000.

        Returns:
            np.ndarray: Denoised audio waveform. Single channel.
        """
        if self.verbose:
            print(f"Processing audio file: {noisy_audio_path}")
        
        # Load and normalize audio
        noisy_audio = self.load_audio_and_resample(audio_path=noisy_audio_path, target_sr=target_sr)
        noisy_audio_np = self.audio_processing(audio_wav=noisy_audio)
        
        if self.verbose:
            print(f"Loaded audio with shape: {noisy_audio_np.shape}")
        
        # Create chunks
        audio_chunks = chunk_audio(
            audio_signal=noisy_audio_np, 
            sampling_rate=target_sr, 
            chunk_length_sec=self.chunk_length_s
        )
        
        if self.verbose:
            print(f"Created {audio_chunks.shape[0]} chunks with shape: {audio_chunks.shape}")
        
        # Denoise chunks
        denoised_audio = self.denoise(audio_chunks=audio_chunks)
        
        if self.verbose:
            print(f"Denoised audio shape: {denoised_audio.shape}")
        
        # Process output
        denoised_audio_np = self.audio_processing(denoised_audio[0])
        
        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            denoised_tensor = torch.from_numpy(denoised_audio_np).unsqueeze(0)
            torchaudio.save(output_path, denoised_tensor, target_sr)
            if self.verbose:
                print(f"Saved denoised audio to: {output_path}")
        
        return denoised_audio_np