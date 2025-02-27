import os
import tempfile
import streamlit as st
import numpy as np
import soundfile as sf
from final_denoiser import Config, AudioProcessor, AudioDenoiser

# Load the model once when the app starts
@st.cache_resource
def load_model():
    config = Config()  # Ensure Config is defined
    processor = AudioProcessor(config)
    return AudioDenoiser('models/audio_denoiser_model.h5', processor)

def denoise_audio(denoiser, processor, input_path, output_path):
    # Load noisy audio
    noisy_audio = processor.load_audio(input_path)
    if noisy_audio is None:
        st.error(f"Failed to load noisy audio file: {input_path}")
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
    sf.write(output_path, denoised_audio, processor.config.sr)
    
    return True

def main():
    st.set_page_config(page_title="Audio Denoiser", layout="wide")
    st.title("🎵 Audio Denoiser")
    st.write("Upload your audio file to remove background noise while preserving voice content")

    try:
        denoiser = load_model()
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Save to temp file
        temp_dir = tempfile.gettempdir()
        input_path = os.path.join(temp_dir, "input_audio.wav")
        output_path = os.path.join(temp_dir, "denoised_audio.wav")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        status_text.text("Analyzing and denoising audio...")
        progress_bar.progress(30)

        # Denoise audio
        try:
            if denoise_audio(denoiser, denoiser.processor, input_path, output_path):
                progress_bar.progress(70)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original Audio")
                    st.audio(input_path)

                with col2:
                    st.subheader("Denoised Audio")
                    st.audio(output_path)

                progress_bar.progress(100)
                status_text.text("✅ Complete!")

        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    main()