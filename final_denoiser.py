import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration parameters
class Config:
    # Audio parameters
    sr = 48000  # Sample rate for clean/noisy paired dataset
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    max_db = 100
    ref_db = 20
    
    # Training parameters
    batch_size = 16
    epochs = 100
    validation_split = 0.2
    patience = 10
    
    # Model parameters
    target_length = 32  # Fixed length for spectrogram frames
    
    # Paths
    clean_dir = "path/to/clean_audio_files"
    noisy_dir = "path/to/noisy_audio_files"
    mozilla_dir = "path/to/mozilla_common_voice"
    urbansound_dir = "path/to/urbansound8k"
    model_save_path = "audio_denoiser_model.h5"
    tensorboard_log_dir = "logs/audio_denoiser"
    
    # Data augmentation
    augment = True
    
# Create the U-Net model
def create_unet_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder (Downsampling)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.2)(pool1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(drop1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.2)(pool2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(drop2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.3)(pool3)
    
    # Bottom (lowest resolution)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(drop3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.4)(conv4)
    
    # Decoder (Upsampling)
    up5 = UpSampling2D(size=(2, 2))(drop4)
    merge5 = Concatenate(axis=3)([conv3, up5])
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = Concatenate(axis=3)([conv2, up6])
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = Concatenate(axis=3)([conv1, up7])
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    # Output layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv7)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Audio processing utilities
class AudioProcessor:
    def __init__(self, config):
        self.config = config
    
    def load_audio(self, file_path, sr=None):
        """Load audio file and normalize"""
        try:
            if sr is None:
                sr = self.config.sr
            
            if file_path.endswith('.mp3'):
                # Use librosa for MP3 files
                audio, _ = librosa.load(file_path, sr=sr, mono=True)
            else:
                # Use soundfile for WAV files
                audio, file_sr = sf.read(file_path)
                if len(audio.shape) > 1:  # Check if stereo
                    audio = np.mean(audio, axis=1)  # Convert to mono
                if file_sr != sr:
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def audio_to_melspectrogram(self, audio):
        """Convert audio to log mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.config.sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(
            mel_spec, 
            ref=np.max, 
            top_db=self.config.max_db
        )
        
        # Normalize
        log_mel_spec = (log_mel_spec + self.config.max_db) / self.config.max_db
        
        return log_mel_spec
    
    def melspectrogram_to_audio(self, mel_spectrogram, denormalize=True):
        """Convert mel spectrogram back to audio"""
        if denormalize:
            # Denormalize
            mel_spectrogram = mel_spectrogram * self.config.max_db - self.config.max_db
        
        # Convert to power mel spectrogram
        mel_spec_power = librosa.db_to_power(mel_spectrogram)
        
        # Approximate inversion
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec_power,
            sr=self.config.sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        return audio
    
    def prepare_spectrograms(self, clean_files, noisy_files):
        """
        Prepare paired clean and noisy spectrograms for training.
        Returns normalized spectrograms with fixed length segments.
        """
        clean_specs = []
        noisy_specs = []
        
        for clean_file, noisy_file in tqdm(zip(clean_files, noisy_files), 
                                          total=len(clean_files),
                                          desc="Processing audio files"):
            # Load audio files
            clean_audio = self.load_audio(clean_file)
            noisy_audio = self.load_audio(noisy_file)
            
            if clean_audio is None or noisy_audio is None:
                continue
                
            # Make sure they have the same length
            min_len = min(len(clean_audio), len(noisy_audio))
            clean_audio = clean_audio[:min_len]
            noisy_audio = noisy_audio[:min_len]
            
            # Convert to mel spectrograms
            clean_spec = self.audio_to_melspectrogram(clean_audio)
            noisy_spec = self.audio_to_melspectrogram(noisy_audio)
            
            # Segment into fixed-length chunks
            for i in range(0, clean_spec.shape[1] - self.config.target_length, self.config.target_length // 2):
                clean_segment = clean_spec[:, i:i + self.config.target_length]
                noisy_segment = noisy_spec[:, i:i + self.config.target_length]
                
                # Ensure segments have the exact target length
                if clean_segment.shape[1] == self.config.target_length:
                    clean_specs.append(clean_segment)
                    noisy_specs.append(noisy_segment)
        
        # Convert to numpy arrays and add channel dimension
        clean_specs = np.array(clean_specs)[:, :, :, np.newaxis]
        noisy_specs = np.array(noisy_specs)[:, :, :, np.newaxis]
        
        print(f"Prepared {len(clean_specs)} spectrogram segments")
        print(f"Spectrogram shape: {clean_specs.shape}")
        
        return noisy_specs, clean_specs
    
    def prepare_spectrograms_unlabeled(self, audio_files, is_noise=False):
        """
        Prepare spectrograms from unlabeled audio files (Mozilla voice or UrbanSound8K)
        """
        specs = []
        
        for file in tqdm(audio_files, desc="Processing unlabeled audio"):
            # Load audio file
            audio = self.load_audio(file)
            
            if audio is None:
                continue
                
            # Convert to mel spectrogram
            spec = self.audio_to_melspectrogram(audio)
            
            # Segment into fixed-length chunks
            for i in range(0, spec.shape[1] - self.config.target_length, self.config.target_length // 2):
                segment = spec[:, i:i + self.config.target_length]
                
                # Ensure segments have the exact target length
                if segment.shape[1] == self.config.target_length:
                    specs.append(segment)
        
        # Convert to numpy array and add channel dimension
        specs = np.array(specs)[:, :, :, np.newaxis]
        
        print(f"Prepared {len(specs)} {'noise' if is_noise else 'voice'} spectrogram segments")
        print(f"Spectrogram shape: {specs.shape}")
        
        return specs
    
    def data_augmentation(self, clean_specs, noisy_specs):
        """Apply data augmentation to increase training data variety"""
        # Time stretching simulation (shift frames)
        aug_clean = []
        aug_noisy = []
        
        for i in range(len(clean_specs)):
            clean = clean_specs[i]
            noisy = noisy_specs[i]
            
            # Slight time shift (1-2 frames)
            if np.random.rand() > 0.5:
                shift = np.random.randint(1, 3)
                clean_shifted = np.pad(clean[:, :-shift], ((0, 0), (shift, 0), (0, 0)), mode='constant')
                noisy_shifted = np.pad(noisy[:, :-shift], ((0, 0), (shift, 0), (0, 0)), mode='constant')
                aug_clean.append(clean_shifted)
                aug_noisy.append(noisy_shifted)
                
            # Add small random noise to clean spectrograms (not too much)
            if np.random.rand() > 0.7:
                noise_level = np.random.uniform(0.02, 0.05)
                clean_noisy = clean + noise_level * np.random.randn(*clean.shape)
                clean_noisy = np.clip(clean_noisy, 0, 1)  # Keep within 0-1 range
                aug_clean.append(clean_noisy)
                aug_noisy.append(noisy)
                
        # Convert to numpy arrays
        aug_clean = np.array(aug_clean)
        aug_noisy = np.array(aug_noisy)
        
        # Combine original and augmented data
        combined_clean = np.vstack([clean_specs, aug_clean])
        combined_noisy = np.vstack([noisy_specs, aug_noisy])
        
        print(f"After augmentation: {len(combined_clean)} spectrogram segments")
        
        return combined_noisy, combined_clean

# Data loader class
class DataLoader:
    def __init__(self, config, processor):
        self.config = config
        self.processor = processor
    
    def get_paired_filenames(self):
        """Get paired clean and noisy audio files"""
        clean_files = sorted([os.path.join(self.config.clean_dir, f) 
                             for f in os.listdir(self.config.clean_dir) if f.endswith('.wav')])
        noisy_files = sorted([os.path.join(self.config.noisy_dir, f) 
                             for f in os.listdir(self.config.noisy_dir) if f.endswith('.wav')])
        
        # Ensure we have matching pairs
        if len(clean_files) != len(noisy_files):
            print("Warning: Number of clean and noisy files doesn't match")
            min_count = min(len(clean_files), len(noisy_files))
            clean_files = clean_files[:min_count]
            noisy_files = noisy_files[:min_count]
        
        print(f"Found {len(clean_files)} paired audio files")
        return clean_files, noisy_files
    
    def get_mozilla_files(self, max_files=None):
        """Get Mozilla Common Voice files"""
        mozilla_files = [os.path.join(self.config.mozilla_dir, f) 
                        for f in os.listdir(self.config.mozilla_dir) if f.endswith('.mp3')]
        
        if max_files and len(mozilla_files) > max_files:
            mozilla_files = mozilla_files[:max_files]
        
        print(f"Found {len(mozilla_files)} Mozilla voice files")
        return mozilla_files
    
    def get_urbansound_files(self):
        """Get UrbanSound8K noise files"""
        urbansound_files = []
        
        # Loop through all 10 folders
        for i in range(1, 11):
            folder_path = os.path.join(self.config.urbansound_dir, f"fold{i}")
            if os.path.exists(folder_path):
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
                urbansound_files.extend(files)
        
        print(f"Found {len(urbansound_files)} UrbanSound8K files")
        return urbansound_files
    
    def load_and_prepare_data(self):
        """Load and prepare all datasets for training"""
        # Load paired clean-noisy dataset
        clean_files, noisy_files = self.get_paired_filenames()
        X_noisy, y_clean = self.processor.prepare_spectrograms(clean_files, noisy_files)
        
        # Apply data augmentation if enabled
        if self.config.augment:
            X_noisy, y_clean = self.processor.data_augmentation(y_clean, X_noisy)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_noisy, y_clean, 
            test_size=self.config.validation_split,
            random_state=42
        )
        
        return X_train, X_val, y_train, y_val
    
    def load_additional_data(self, max_mozilla=200, max_urbansound=500):
        """Load additional data for fine-tuning or model improvement"""
        # Load Mozilla voice samples (limit to control memory usage)
        mozilla_files = self.get_mozilla_files(max_files=max_mozilla)
        if mozilla_files:
            voice_specs = self.processor.prepare_spectrograms_unlabeled(mozilla_files, is_noise=False)
        else:
            voice_specs = None
            
        # Load UrbanSound8K noise samples
        urbansound_files = self.get_urbansound_files()
        if len(urbansound_files) > max_urbansound:
            urbansound_files = urbansound_files[:max_urbansound]
            
        if urbansound_files:
            noise_specs = self.processor.prepare_spectrograms_unlabeled(urbansound_files, is_noise=True)
        else:
            noise_specs = None
            
        return voice_specs, noise_specs

# Training utilities
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=self.config.model_save_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # TensorBoard logging
            TensorBoard(log_dir=self.config.tensorboard_log_dir)
        ]
        return callbacks
    
    def train_model(self, model, X_train, y_train, X_val, y_val):
        """Train the denoising model"""
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
        # Display model summary
        model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def fine_tune_with_additional_data(self, model, voice_specs, noise_specs, X_train, y_train):
        """Fine-tune model with voice and noise data to improve performance"""
        if voice_specs is None or noise_specs is None:
            print("Skipping fine-tuning due to missing additional data")
            return None
            
        print("Fine-tuning model with voice recognition and noise classification...")
        
        # Create known targets for voice data (should be preserved)
        voice_targets = voice_specs.copy()
        
        # Create empty targets for noise data (should be removed)
        noise_targets = np.zeros_like(noise_specs)
        
        # Combine data for fine-tuning
        X_fine_tune = np.vstack([X_train[:1000], voice_specs[:500], noise_specs[:500]])
        y_fine_tune = np.vstack([y_train[:1000], voice_targets[:500], noise_targets[:500]])
        
        # Split for validation
        X_ft_train, X_ft_val, y_ft_train, y_ft_val = train_test_split(
            X_fine_tune, y_fine_tune, 
            test_size=0.2,
            random_state=42
        )
        
        # Fine-tune with a lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=5e-5),
            loss=MeanSquaredError(),
            metrics=['mae']
        )
        
        # Create callbacks with different path for fine-tuned model
        ft_callbacks = [
            ModelCheckpoint(
                filepath="audio_denoiser_finetuned_model.h5",
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Fine-tune for fewer epochs
        ft_history = model.fit(
            X_ft_train, y_ft_train,
            validation_data=(X_ft_val, y_ft_val),
            batch_size=self.config.batch_size,
            epochs=30,  # Fewer epochs for fine-tuning
            callbacks=ft_callbacks,
            verbose=1
        )
        
        return ft_history

# Inference utilities
class AudioDenoiser:
    def __init__(self, model_path, processor, chunk_size=32, overlap=0.5):
        self.model = tf.keras.models.load_model(model_path)
        self.processor = processor
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def denoise_file(self, input_file, output_file):
        """Denoise an audio file and save the result"""
        print(f"Denoising file: {input_file}")
        
        # Load audio
        audio = self.processor.load_audio(input_file)
        if audio is None:
            print(f"Failed to load audio file: {input_file}")
            return False
        
        # Convert to mel spectrogram
        mel_spec = self.processor.audio_to_melspectrogram(audio)
        
        # Process in chunks to handle files of any length
        chunk_size = self.chunk_size
        hop_size = int(chunk_size * (1 - self.overlap))
        
        # Pad if needed to ensure we can process the entire spectrogram
        pad_width = (0, (chunk_size - mel_spec.shape[1] % chunk_size) % chunk_size)
        mel_spec_padded = np.pad(mel_spec, ((0, 0), pad_width), mode='constant')
        
        # Initialize output spectrogram
        clean_spec = np.zeros_like(mel_spec_padded)
        
        # Keep track of overlaps for averaging
        overlap_count = np.zeros_like(mel_spec_padded)
        
        # Process each chunk
        num_chunks = (mel_spec_padded.shape[1] - chunk_size) // hop_size + 1
        
        for i in tqdm(range(num_chunks), desc="Processing chunks"):
            start_idx = i * hop_size
            end_idx = start_idx + chunk_size
            
            # Extract chunk
            chunk = mel_spec_padded[:, start_idx:end_idx]
            
            # Add batch and channel dimensions
            chunk = chunk[np.newaxis, :, :, np.newaxis]
            
            # Denoise chunk
            clean_chunk = self.model.predict(chunk, verbose=0)[0, :, :, 0]
            
            # Add to output with overlap
            clean_spec[:, start_idx:end_idx] += clean_chunk
            overlap_count[:, start_idx:end_idx] += 1
        
        # Average overlapping sections
        clean_spec = clean_spec / np.maximum(overlap_count, 1)
        
        # Remove padding
        if pad_width[1] > 0:
            clean_spec = clean_spec[:, :-pad_width[1]]
        
        # Convert back to audio
        clean_audio = self.processor.melspectrogram_to_audio(clean_spec)
        
        # Save denoised audio
        sf.write(output_file, clean_audio, self.processor.config.sr)
        
        print(f"Denoised audio saved to: {output_file}")
        return True
    
    def visualize_denoising(self, input_file, output_file=None):
        """Visualize the denoising process with spectrograms"""
        # Load audio
        audio = self.processor.load_audio(input_file)
        if audio is None:
            print(f"Failed to load audio file: {input_file}")
            return
        
        # Get a sample segment for visualization
        audio_segment = audio[:int(self.processor.config.sr * 5)]  # 5 seconds
        
        # Convert to mel spectrogram
        mel_spec = self.processor.audio_to_melspectrogram(audio_segment)
        
        # Ensure spectrogram has proper shape
        if mel_spec.shape[1] < self.chunk_size:
            pad_width = (0, self.chunk_size - mel_spec.shape[1])
            mel_spec = np.pad(mel_spec, ((0, 0), pad_width), mode='constant')
        
        # Process in chunks
        denoised_spec = np.zeros_like(mel_spec)
        
        for i in range(0, mel_spec.shape[1] - self.chunk_size + 1, self.chunk_size // 2):
            chunk = mel_spec[:, i:i + self.chunk_size]
            chunk = chunk[np.newaxis, :, :, np.newaxis]
            
            denoised_chunk = self.model.predict(chunk, verbose=0)[0, :, :, 0]
            
            # Add to output (simple overlap-add)
            if i == 0:
                denoised_spec[:, i:i + self.chunk_size] = denoised_chunk
            else:
                # Crossfade for smoother transitions
                fade_len = self.chunk_size // 2
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                
                # Apply crossfade
                denoised_spec[:, i:i + fade_len] = (
                    denoised_spec[:, i:i + fade_len] * fade_out +
                    denoised_chunk[:, :fade_len] * fade_in
                )
                denoised_spec[:, i + fade_len:i + self.chunk_size] = denoised_chunk[:, fade_len:]
        
        # Convert back to audio
        noisy_audio = self.processor.melspectrogram_to_audio(mel_spec)
        denoised_audio = self.processor.melspectrogram_to_audio(denoised_spec)
        
        # Plot spectrograms
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        librosa.display.specshow(
            librosa.power_to_db(librosa.feature.melspectrogram(y=audio_segment, sr=self.processor.config.sr)),
            sr=self.processor.config.sr,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Noisy Mel Spectrogram')
        
        plt.subplot(2, 2, 2)
        librosa.display.specshow(
            librosa.power_to_db(librosa.feature.melspectrogram(y=denoised_audio, sr=self.processor.config.sr)),
            sr=self.processor.config.sr,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Denoised Mel Spectrogram')
        
        plt.subplot(2, 2, 3)
        plt.plot(audio_segment)
        plt.title('Noisy Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        plt.subplot(2, 2, 4)
        plt.plot(denoised_audio)
        plt.title('Denoised Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            print(f"Visualization saved to: {output_file}")
        else:
            plt.show()
        
        # Save audio clips
        if output_file:
            noisy_out = output_file.replace('.png', '_noisy.wav')
            denoised_out = output_file.replace('.png', '_denoised.wav')
            
            sf.write(noisy_out, noisy_audio, self.processor.config.sr)
            sf.write(denoised_out, denoised_audio, self.processor.config.sr)
            
            print(f"Audio samples saved to: {noisy_out} and {denoised_out}")

# Main execution function
def train_audio_denoiser(config_params=None):
    """Main function to train audio denoiser model"""
    # Initialize configuration
    config = Config()
    if config_params:
        for key, value in config_params.items():
            setattr(config, key, value)
    
    # Initialize processor and data loader
    processor = AudioProcessor(config)
    data_loader = DataLoader(config, processor)
    
    # Load and prepare data
    print("Loading and preparing paired data...")
    X_train, X_val, y_train, y_val = data_loader.load_and_prepare_data()
    
    # Create model
    print("Creating U-Net model...")
    input_shape = (config.n_mels, config.target_length, 1)
    model = create_unet_model(input_shape)
    
    # Train model
    # Train model
    print("Training model...")
    trainer = ModelTrainer(config)
    history = trainer.train_model(model, X_train, y_train, X_val, y_val)
    
    # Load additional data for fine-tuning
    print("Loading additional data for fine-tuning...")
    voice_specs, noise_specs = data_loader.load_additional_data()
    
    # Fine-tune with additional data
    print("Fine-tuning model...")
    ft_history = trainer.fine_tune_with_additional_data(model, voice_specs, noise_specs, X_train, y_train)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("Training complete. Model saved to:", config.model_save_path)
    return model

# Usage example
if __name__ == "__main__":
    # Set your paths here
    config_params = {
        'clean_dir': './data/clean',
        'noisy_dir': './data/noisy',
        'mozilla_dir': './data/mozilla_common_voice',
        'urbansound_dir': './data/urbansound8k',
        'model_save_path': './models/audio_denoiser_model.h5',
        'tensorboard_log_dir': './logs/audio_denoiser'
    }
    
    # Create necessary directories
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # Train model
    model = train_audio_denoiser(config_params)
    
    # Initialize processor for inference
    config = Config()
    processor = AudioProcessor(config)
    
    # Initialize denoiser
    denoiser = AudioDenoiser('models/audio_denoiser_model.h5', processor)
    
    # Test on a sample file
    sample_file = './data/noisy/test_sample.wav'
    if os.path.exists(sample_file):
        denoiser.denoise_file(sample_file, './results/denoised_sample.wav')
        denoiser.visualize_denoising(sample_file, './results/visualization.png')

# Frontend Integration 
# Create a simple Flask API for the denoising service

def create_flask_app():
    from flask import Flask, request, jsonify, send_file
    import tempfile
    import uuid
    
    app = Flask(__name__)
    
    # Set up the model
    config = Config()
    processor = AudioProcessor(config)
    denoiser = AudioDenoiser('models/audio_denoiser_model.h5', processor)
    
    @app.route('/denoise', methods=['POST'])
    def denoise_audio():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if file:
            # Save uploaded file to temporary location
            temp_dir = tempfile.gettempdir()
            input_path = os.path.join(temp_dir, f"input_{uuid.uuid4()}.wav")
            output_path = os.path.join(temp_dir, f"denoised_{uuid.uuid4()}.wav")
            
            try:
                file.save(input_path)
                
                # Denoise the file
                success = denoiser.denoise_file(input_path, output_path)
                
                if success:
                    # Return the denoised file
                    return send_file(output_path, as_attachment=True, 
                                     download_name="denoised_audio.wav")
                else:
                    return jsonify({'error': 'Denoising failed'}), 500
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
                
            finally:
                # Clean up temporary files
                try:
                    if os.path.exists(input_path):
                        os.remove(input_path)
                except:
                    pass
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'ok', 'model': 'loaded'}), 200
        
    return app

# Create a simple Streamlit frontend for the application
def create_streamlit_app():
    """
    Save this code to app.py and run with: streamlit run app.py
    
    Requirements:
    pip install streamlit librosa tensorflow soundfile matplotlib
    """
    import streamlit as st
    import os
    import tempfile
    
    st.set_page_config(page_title="Audio Denoiser", layout="wide")
    
    st.title("🎵 Audio Denoiser")
    st.write("Upload your audio file to remove background noise while preserving voice content")
    
    @st.cache_resource
    def load_model():
        config = Config()
        processor = AudioProcessor(config)
        return AudioDenoiser('models/audio_denoiser_model.h5', processor)
    
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
        vis_path = os.path.join(temp_dir, "visualization.png")
        
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        status_text.text("Analyzing and denoising audio...")
        progress_bar.progress(30)
        
        # Denoise audio
        try:
            denoiser.denoise_file(input_path, output_path)
            progress_bar.progress(70)
            
            # Generate visualization
            status_text.text("Generating visualization...")
            denoiser.visualize_denoising(input_path, vis_path)
            progress_bar.progress(90)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Audio")
                st.audio(input_path)
                
            with col2:
                st.subheader("Denoised Audio")
                st.audio(output_path)
                
            st.subheader("Spectrogram Visualization")
            st.image(vis_path)
            
            # Download button
            with open(output_path, "rb") as file:
                btn = st.download_button(
                    label="Download Denoised Audio",
                    data=file,
                    file_name="denoised_audio.wav",
                    mime="audio/wav"
                )
                
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            
    # Show model information
    with st.expander("About the Model"):
        st.write("""
        This audio denoiser uses a U-Net convolutional neural network architecture to remove background noise while preserving speech content.
        
        The model was trained on three datasets:
        1. Paired clean/noisy audio recordings
        2. Mozilla Common Voice dataset for speech preservation
        3. UrbanSound8K dataset for noise identification
        
        The model processes audio by:
        1. Converting to mel spectrograms
        2. Processing through the neural network
        3. Converting back to audio waveforms
        """)
        
        st.image("https://miro.medium.com/max/1400/1*f7YOaE4TWubwaFF7Z1fzNw.png", 
                caption="U-Net Architecture Visualization")

# Error Handling Utilities
class ErrorHandler:
    @staticmethod
    def check_data_compatibility(X, y=None):
        """Check if data shapes are compatible with the model"""
        # Check input dimensions
        if len(X.shape) != 4:
            raise ValueError(f"Input must be 4D (batch, height, width, channels), found shape {X.shape}")
            
        # Check if spectrograms have consistent dimensions
        if X.shape[2] != 32 and X.shape[2] != 31:  # Allow for small differences
            raise ValueError(f"Expected spectrogram width around 32, found {X.shape[2]}")
            
        # If targets provided, check compatibility
        if y is not None:
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Input and target batch sizes don't match: {X.shape[0]} vs {y.shape[0]}")
                
            if X.shape[1] != y.shape[1] or abs(X.shape[2] - y.shape[2]) > 1:
                raise ValueError(f"Input shape {X.shape[1:3]} and target shape {y.shape[1:3]} are incompatible")
    
    @staticmethod
    def fix_dimension_mismatch(X, y=None, target_width=32):
        """Fix dimension mismatches by padding or cropping"""
        # Fix input width if needed
        if X.shape[2] != target_width:
            if X.shape[2] < target_width:
                # Pad
                pad_width = ((0, 0), (0, 0), (0, target_width - X.shape[2]), (0, 0))
                X = np.pad(X, pad_width, mode='constant')
            else:
                # Crop
                X = X[:, :, :target_width, :]
                
        # Fix target dimensions if provided
        if y is not None:
            if y.shape[2] != target_width:
                if y.shape[2] < target_width:
                    # Pad
                    pad_width = ((0, 0), (0, 0), (0, target_width - y.shape[2]), (0, 0))
                    y = np.pad(y, pad_width, mode='constant')
                else:
                    # Crop
                    y = y[:, :, :target_width, :]
            return X, y
        
        return X
    
    @staticmethod
    def handle_inhomogeneous_data(data_list):
        """Handle inhomogeneous data shapes when loading from different sources"""
        # Find the most common shape
        shapes = {}
        for item in data_list:
            if item is not None:
                shape_key = str(item.shape[1:])  # Use shape except batch dimension as key
                if shape_key in shapes:
                    shapes[shape_key] += 1
                else:
                    shapes[shape_key] = 1
        
        # Find most common shape
        most_common_shape = max(shapes.items(), key=lambda x: x[1])[0]
        most_common_shape = eval(most_common_shape)  # Convert string back to tuple
        
        # Standardize shapes
        result = []
        for item in data_list:
            if item is None:
                result.append(None)
                continue
                
            current_shape = item.shape[1:]
            if current_shape != most_common_shape:
                # Reshape to match the most common shape
                reshaped_item = np.zeros((item.shape[0],) + most_common_shape)
                
                # Copy what fits
                min_h = min(current_shape[0], most_common_shape[0])
                min_w = min(current_shape[1], most_common_shape[1])
                min_c = min(current_shape[2], most_common_shape[2]) if len(current_shape) > 2 else 1
                
                # Handle dimensions
                if len(current_shape) == 3 and len(most_common_shape) == 3:
                    reshaped_item[:, :min_h, :min_w, :min_c] = item[:, :min_h, :min_w, :min_c]
                elif len(current_shape) == 2 and len(most_common_shape) == 3:
                    reshaped_item[:, :min_h, :min_w, 0] = item[:, :min_h, :min_w]
                    
                result.append(reshaped_item)
            else:
                result.append(item)
        
        return result

# Main execution
if __name__ == "__main__":
    print("Audio Denoiser Application")
    print("1. Train model")
    print("2. Run Flask API")
    print("3. Run Streamlit UI")
    
    choice = input("Select an option (1-3): ")
    
    if choice == "1":
        print("Training model...")
        train_audio_denoiser()
    elif choice == "2":
        print("Starting Flask API...")
        app = create_flask_app()
        app.run(debug=True, host='0.0.0.0', port=5000)
    elif choice == "3":
        print("To run Streamlit UI, execute: streamlit run app.py")
        # This script should be saved as app.py
        # Streamlit is run from command line, not within Python
    else:
        print("Invalid choice")