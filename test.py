import os
import numpy as np
import tensorflow as tf
import soundfile as sf
from final_denoiser import Config, AudioProcessor, AudioDenoiser

def test_denoise_model(model_path, noisy_file, output_file="denoised_output.wav"):

    # Initialize config and processor
    config = Config()
    processor = AudioProcessor(config)
    
    # Load denoiser
    denoiser = AudioDenoiser(model_path, processor)
    
    # Load noisy audio
    noisy_audio = processor.load_audio(noisy_file)
    if noisy_audio is None:
        print(f"Failed to load noisy audio file: {noisy_file}")
        return False
    
    # Convert to mel spectrogram
    noisy_spec = processor.audio_to_melspectrogram(noisy_audio)
    
    # Process in chunks of 32 frames with 50% overlap
    chunk_size = 32
    hop_size = chunk_size // 2
    num_chunks = (noisy_spec.shape[1] - chunk_size) // hop_size + 1
    
    # Initialize output spectrogram
    denoised_spec = np.zeros_like(noisy_spec)
    overlap_count = np.zeros_like(noisy_spec)
    
    for i in range(num_chunks):
        start_idx = i * hop_size
        end_idx = start_idx + chunk_size
        
        # Extract chunk and add batch/channel dimensions
        chunk = noisy_spec[:, start_idx:end_idx]
        chunk = chunk[np.newaxis, :, :, np.newaxis]
        
        # Denoise chunk
        denoised_chunk = denoiser.model.predict(chunk, verbose=0)[0, :, :, 0]
        
        # Add to output with overlap
        denoised_spec[:, start_idx:end_idx] += denoised_chunk
        overlap_count[:, start_idx:end_idx] += 1
    
    # Average overlapping sections
    denoised_spec = denoised_spec / np.maximum(overlap_count, 1)

    # Convert back to audio and normalize
    denoised_audio = processor.melspectrogram_to_audio(denoised_spec)
    denoised_audio = denoised_audio / np.max(np.abs(denoised_audio))  # Prevent clipping

    # Save results
    sf.write(output_file, denoised_audio, config.sr)
    
    print(f"Denoised audio saved to {output_file}")
    return True


if __name__ == "__main__":
    # Path to the latest model
    model_path = "models/audio_denoiser_model.h5"
    
    # Path to the noisy audio file for testing
    noisy_file = "data/noisy/p232_023.wav"  # Change this to your noisy audio file path
    
    # Run the denoising test
    test_denoise_model(model_path, noisy_file, output_file="gooo_output.wav")