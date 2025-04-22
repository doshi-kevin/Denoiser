import gradio as gr
import os
import torch
import torchaudio
import tempfile
import numpy as np
import soundfile as sf
from denoiser import DenoiserAudio

# Initialize denoiser with proper settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
denoise = DenoiserAudio(
    device=device,
    chunk_length_s=3,
    max_batch_size=20,
    verbose=True
)

def process_audio(audio, progress=gr.Progress()):
    """
    Process uploaded audio to denoise it.
    
    Args:
        audio: Input audio file path or tuple of (sample_rate, audio_data)
        progress: Gradio progress tracker
        
    Returns:
        tuple: Path to denoised audio file and status message
    """
    if audio is None:
        return None, "⚠️ No audio file uploaded. Please upload an audio file."
    
    # Create temp directory if it doesn't exist
    os.makedirs("Temp", exist_ok=True)
    
    progress(0, "Preparing audio file...")
    
    temp_filepath = None
    
    try:
        # Handle different input types
        if isinstance(audio, tuple):  # Recorded audio
            sample_rate, audio_data = audio
            
            # Print debug info
            print(f"Received audio: sample_rate={sample_rate}, audio_data shape={np.array(audio_data).shape}, type={type(audio_data)}")
            
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir="Temp")
            temp_filepath = temp_file.name
            temp_file.close()
            
            # Ensure audio_data is in the right format
            audio_np = np.array(audio_data)
            
            # Use soundfile for more robust saving
            try:
                # Normalize audio data if needed
                if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                    max_val = max(abs(audio_np.max()), abs(audio_np.min()))
                    audio_np = audio_np / max_val
                
                # Write audio data to file
                sf.write(temp_filepath, audio_np, sample_rate)
                original_filepath = temp_filepath
                print(f"Successfully saved audio to {temp_filepath}")
                
            except Exception as save_error:
                print(f"Error with soundfile: {save_error}")
                # Fall back to torchaudio if soundfile fails
                audio_tensor = torch.tensor(audio_np)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                torchaudio.save(temp_filepath, audio_tensor, sample_rate)
                original_filepath = temp_filepath
                print(f"Successfully saved audio with torchaudio to {temp_filepath}")
                
        else:  # Uploaded file
            original_filepath = audio
            print(f"Using uploaded file: {original_filepath}")
        
        # Generate output filename
        filename = os.path.basename(original_filepath)
        denoised_filepath = os.path.join("Temp", f"denoised_{filename}")
        
        progress(0.2, "Processing audio...")
        
        # Denoise the audio (this will handle resampling internally)
        denoised_audio = denoise(
            noisy_audio_path=original_filepath,
            output_path=denoised_filepath
        )
        
        progress(0.9, "Finalizing...")
        
        # Clean up temporary file if it was created
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            
        return denoised_filepath, "✅ Audio denoised successfully!"
    
    except Exception as e:
        # Clean up on error
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except:
                pass
                
        print(f"Error in process_audio: {str(e)}")
        return None, f"❌ Error processing audio: {str(e)}"

# Define UI colors and theme
primary_color = "#3B82F6"  # Blue
secondary_color = "#E6F3FF"  # Light Blue
text_color = "#1F2937"  # Dark Gray

custom_css = f"""
    .gradio-container {{
        background-color: {secondary_color};
    }}
    
    .title-text {{
        color: {text_color};
        text-align: center;
        margin-bottom: 1rem;
    }}
    
    .footer-text {{
        text-align: center;
        margin-top: 1rem;
        color: {text_color};
    }}
    
    .tab-nav * {{
        color: {text_color};
    }}
"""

# App title
title = f"""
<h1 class="title-text">CleanUNet Audio Denoiser</h1>
<p style="text-align: center;">Remove noise from your audio files using NVIDIA's CleanUNet</p>
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.HTML(title)
    
    with gr.Tabs():
        with gr.TabItem("Upload Audio"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        sources=["upload"],
                        label="Upload Noisy Audio File",
                        elem_id="audio-input",
                        type="filepath"  # Explicitly set type to filepath for uploaded files
                    )
                    
                    upload_button = gr.Button(
                        "Denoise Audio", 
                        variant="primary",
                        elem_id="denoise-btn"
                    )
                    
                    status_output = gr.Textbox(
                        label="Status", 
                        placeholder="Upload an audio file and click 'Denoise Audio'",
                        interactive=False
                    )
                    
                with gr.Column():
                    audio_output = gr.Audio(
                        label="Denoised Audio", 
                        elem_id="audio-output",
                        interactive=False
                    )
                    
        with gr.TabItem("Record Audio"):
            with gr.Row():
                with gr.Column():
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        label="Record Audio",
                        elem_id="mic-input"
                    )
                    
                    record_button = gr.Button(
                        "Denoise Recorded Audio", 
                        variant="primary",
                        elem_id="record-denoise-btn"
                    )
                    
                    record_status = gr.Textbox(
                        label="Status", 
                        placeholder="Record audio and click 'Denoise Recorded Audio'",
                        interactive=False
                    )
                    
                with gr.Column():
                    record_output = gr.Audio(
                        label="Denoised Recording", 
                        elem_id="record-output",
                        interactive=False
                    )
    
    # Connect components with processing function
    upload_button.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=[audio_output, status_output]
    )
    
    record_button.click(
        fn=process_audio,
        inputs=mic_input,
        outputs=[record_output, record_status]
    )
    
    # Footer information
    footer = """
    <div class="footer-text">
        <p><b>Developed by:</b> A F M Mahfuzul Kabir</p>
        <p>Built with NVIDIA's CleanUNet | <a href="https://github.com/Kabir5296/Speech-Denoiser-System">GitHub Repository</a></p>
    </div>
    """
    gr.HTML(footer)

# Launch the app
if __name__ == "__main__":
    print("Starting Gradio app...")
    print("After launching, access the app at: http://127.0.0.1:7862 or http://localhost:7862")
    
    demo.launch(
        server_name="127.0.0.1",  # Changed from 0.0.0.0 to 127.0.0.1 for local access
        server_port=7862,
        share=True,  # Enable sharing for remote access
        inbrowser=True  # Open in browser automatically
    )