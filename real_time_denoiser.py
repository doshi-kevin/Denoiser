import numpy as np
import tensorflow as tf
import sounddevice as sd
from final_denoiser import AudioProcessor, Config

class RealTimeDenoiser:
    def __init__(self, model_path, chunk_size=1024, overlap=0.5):
        self.model = tf.keras.models.load_model(model_path)
        self.processor = AudioProcessor(Config())
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.buffer = np.zeros((self.chunk_size,), dtype=np.float32)

    def process_chunk(self, chunk):
        # Convert audio to mel spectrogram
        mel_spec = self.processor.audio_to_melspectrogram(chunk)
        
        # Ensure the spectrogram has the correct width (32 frames)
        if mel_spec.shape[1] < self.processor.config.target_length:
            # Pad with zeros if too short
            pad_width = ((0, 0), (0, self.processor.config.target_length - mel_spec.shape[1]))
            mel_spec = np.pad(mel_spec, pad_width, mode='constant')
        else:
            # Truncate if too long
            mel_spec = mel_spec[:, :self.processor.config.target_length]
            
        # Add batch and channel dimensions
        mel_spec = mel_spec[np.newaxis, :, :, np.newaxis]

        # Denoise the chunk
        denoised_spec = self.model.predict(mel_spec, verbose=0)[0, :, :, 0]

        # Convert back to audio
        denoised_audio = self.processor.melspectrogram_to_audio(denoised_spec)
        return denoised_audio


    def callback(self, indata, outdata, frames, time, status):
        if status:
            print("Status:", status)
        print(f"Processing chunk of size {len(indata)} samples")
        try:
            # Process the incoming audio chunk
            denoised_chunk = self.process_chunk(indata[:, 0])
            outdata[:] = denoised_chunk[:, np.newaxis]
            print("Chunk processed successfully")
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")


    def start(self):
        print("Initializing real-time denoising...")
        try:
            with sd.Stream(
                channels=1,
                callback=self.callback,
                blocksize=self.chunk_size,
                samplerate=Config().sr
            ):
                print(f"Real-time denoising started at {Config().sr}Hz. Press Ctrl+C to stop.")
                try:
                    while True:
                        sd.sleep(1000)
                except KeyboardInterrupt:
                    print("Real-time denoising stopped.")
        except Exception as e:
            print(f"Failed to start audio stream: {str(e)}")


if __name__ == "__main__":
    model_path = "models/audio_denoiser_model.h5"
    denoiser = RealTimeDenoiser(model_path)
    denoiser.start()
