import torch
import numpy as np

def chunk_audio(audio_signal: np.ndarray, 
                sampling_rate: int, 
                chunk_length_sec: int = 3) -> torch.Tensor:
    '''
    Creates chunks of audio from a long audio file.
    
    Args:
        audio_signal (np.ndarray): The audio signal to chunk
        sampling_rate (int): The sampling rate of the audio
        chunk_length_sec (int): Length of each chunk in seconds
        
    Returns:
        torch.Tensor: Chunked audio tensor with shape (num_chunks, chunk_length)
    '''
    # Calculate samples per chunk
    samples_per_chunk = int(sampling_rate * chunk_length_sec)
    
    # Handle edge case for short audio files
    if len(audio_signal) < samples_per_chunk:
        # Pad with zeros if audio is shorter than chunk length
        padded_signal = np.zeros(samples_per_chunk)
        padded_signal[:len(audio_signal)] = audio_signal
        return torch.from_numpy(padded_signal).float().unsqueeze(0)
    
    # Calculate number of complete chunks
    num_chunks = len(audio_signal) // samples_per_chunk
    
    # Handle remainder
    remainder = len(audio_signal) % samples_per_chunk
    
    if remainder > 0:
        # Create padded array with one extra chunk
        padded_length = (num_chunks + 1) * samples_per_chunk
        padded_signal = np.zeros(padded_length)
        padded_signal[:len(audio_signal)] = audio_signal
        chunks = padded_signal.reshape(num_chunks + 1, samples_per_chunk)
    else:
        # No padding needed
        truncated_signal = audio_signal[:num_chunks * samples_per_chunk]
        chunks = truncated_signal.reshape(num_chunks, samples_per_chunk)
    
    chunks_tensor = torch.from_numpy(chunks).float()
    return chunks_tensor

def unchunk_audio(chunked_audio: torch.Tensor, 
                  overlap: int = 0,
                  verbose: bool = True) -> torch.Tensor:
    '''
    Combines a batch of audio chunks into one single audio file.
    
    Args:
        chunked_audio (torch.Tensor): Tensor of chunked audio (flexible dimensions)
        overlap (int): Number of samples to overlap between chunks
        verbose (bool): Whether to print debug information
        
    Returns:
        torch.Tensor: Combined audio tensor with shape (1, total_length)
    '''
    
    if verbose:
        print(f"Original chunked_audio shape: {chunked_audio.shape}")
    
    # Handle different tensor dimensions
    if chunked_audio.dim() == 2:
        # If input is (num_chunks, chunk_length), add channel dimension
        chunked_audio = chunked_audio.unsqueeze(1)
        if verbose:
            print(f"After adding channel dim: {chunked_audio.shape}")
    
    elif chunked_audio.dim() == 4:
        # If input is (num_chunks, 1, 1, chunk_length) or any 4D tensor
        # Reshape to (num_chunks, 1, chunk_length) to ensure compatibility
        num_chunks = chunked_audio.shape[0]
        chunk_length = chunked_audio.shape[-1]
        # Fix for 4D tensor input - properly squeeze the extra dimension
        chunked_audio = chunked_audio.squeeze(2)  # Remove the extra dimension
        if verbose:
            print(f"After reshaping 4D tensor: {chunked_audio.shape}")
    
    # Now ensure the shape is (num_chunks, 1, chunk_length)
    if chunked_audio.dim() != 3:
        raise ValueError(f"Unable to process tensor with shape {chunked_audio.shape} after dimension adjustment")
    
    # Find the channel and chunk length dimensions
    if chunked_audio.shape[1] == 1:
        # Standard shape (num_chunks, 1, chunk_length)
        num_chunks, _, chunk_length = chunked_audio.shape
    elif chunked_audio.shape[2] == 1:
        # Swapped dimensions (num_chunks, chunk_length, 1)
        chunked_audio = chunked_audio.permute(0, 2, 1)
        num_chunks, _, chunk_length = chunked_audio.shape
    else:
        # Assume the second dimension is channels and the third is the chunk length
        num_chunks, channels, chunk_length = chunked_audio.shape
        if channels != 1:
            if verbose:
                print(f"Warning: Expected channel dimension to be 1, got {channels}. Using first channel.")
            chunked_audio = chunked_audio[:, 0:1, :]
    
    if verbose:
        print(f"Processing tensor with shape: {chunked_audio.shape}")
    
    if overlap >= chunk_length:
        raise ValueError("Overlap must be less than chunk length")
    
    # Calculate output length accounting for overlap
    output_length = (num_chunks - 1) * (chunk_length - overlap) + chunk_length
    output = torch.zeros(1, output_length, device=chunked_audio.device)
    
    # Add chunks with overlap
    for i in range(num_chunks):
        start = i * (chunk_length - overlap)
        end = start + chunk_length
        try:
            output[0, start:end] += chunked_audio[i, 0, :]
        except Exception as e:
            if verbose:
                print(f"Error adding chunk {i}: {e}")
                print(f"Chunk shape: {chunked_audio[i].shape}")
                print(f"Output segment shape: {output[0, start:end].shape}")
            raise
    
    # Apply tapering for smooth transitions if overlap exists
    if overlap > 0:
        taper = torch.linspace(0, 1, overlap, device=chunked_audio.device)
        for i in range(1, num_chunks):
            start = i * (chunk_length - overlap)
            output[0, start:start+overlap] *= taper
            output[0, start-overlap:start] *= (1 - taper)
    
    return output