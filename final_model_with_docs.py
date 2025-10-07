#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified Denoising U-Net and ResNet Comparison (Complete, Corrected)
Includes: U-Net and ResNet Architectures, EDA, and Model Diagrams.
"""

# Standard Python and utility libraries
import os, sys, math, random, glob, shutil, time, functools, itertools
from pathlib import Path

# Numerical and Plotting libraries
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow and Keras for deep learning
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow.signal as tfs # TensorFlow Signal for STFT/ISTFT

# Audio I/O and Playback
from scipy.io import wavfile
from IPython.display import Audio, display

# Check for visualization dependencies
try:
    import pydot
    import graphviz
    HAS_PLOT_UTILITIES = True
except ImportError:
    print("WARNING: pydot or graphviz not installed. Model diagrams cannot be generated.")
    HAS_PLOT_UTILITIES = False

# ----------------------------------------------------------------------
## Environment Constants
# ----------------------------------------------------------------------
# This chunk defines all global hyperparameters and environment settings
# for reproducibility, audio processing, and model training.

SEED = 1337
# Set seeds for full reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SR = 16000 # Sample Rate (Hz)
SEGMENT_SEC = 2.0 # Duration of each audio segment
SEGMENT = int(SR * SEGMENT_SEC) # Total number of samples per segment (32000)

# Short-Time Fourier Transform (STFT) parameters
N_FFT = 1024 # FFT size, determining the number of frequency bins
HOP = 256 # Number of samples between successive STFT frames
WIN_LENGTH = 1024 # Length of the analysis window (should be equal to N_FFT for Hann window)
N_MELS = 128 # Number of Mel bands (unused in this STFT-based model)
PAD_MODE = 'REFLECT' # Not used, but kept for context

# Training hyperparameters
BATCH_SIZE = 8
EPOCHS = 8
STEPS_PER_EPOCH = 600
VAL_STEPS = 80
LEARNING_RATE = 3e-4
WARMUP_STEPS = 500
EMA_DECAY = 0.999 # Decay rate for Exponential Moving Average
CHECKPOINT_DIR = '/kaggle/working/denoiser_ckpt'
EXPORT_DIR = '/kaggle/working/denoiser_export'
EPSILON = 1e-8 # Global epsilon for numerical stability

# Spectrogram dimensions derived from STFT parameters (Crucial for model I/O shape)
TIME_STEPS = (SEGMENT - WIN_LENGTH) // HOP + 1 # 122 STFT frames
FREQ_BINS = N_FFT // 2 + 1 # 513 unique frequency bins (up to Nyquist)

# Setup directories for saving model artifacts
CHECKPOINT_DIR_UNET = os.path.join(CHECKPOINT_DIR, 'unet')
CHECKPOINT_DIR_RESNET = os.path.join(CHECKPOINT_DIR, 'resnet')
EXPORT_DIR_UNET = os.path.join(EXPORT_DIR, 'unet')
EXPORT_DIR_RESNET = os.path.join(EXPORT_DIR, 'resnet')
MODEL_DIAGRAM_DIR = '/kaggle/working/model_diagrams'

os.makedirs(CHECKPOINT_DIR_UNET, exist_ok = True)
os.makedirs(CHECKPOINT_DIR_RESNET, exist_ok = True)
os.makedirs(EXPORT_DIR_UNET, exist_ok = True)
os.makedirs(EXPORT_DIR_RESNET, exist_ok = True)
os.makedirs(MODEL_DIAGRAM_DIR, exist_ok = True)


# ----------------------------------------------------------------------
## Audio Utility IO functions
# ----------------------------------------------------------------------
# These functions handle basic audio manipulation: normalization, reading, and writing WAV files.

def norm_audio(x):
    """
    Normalizes a NumPy array audio signal to have a maximum absolute value of 1.0.

    Args:
        x (np.ndarray): Input audio signal.

    Returns:
        np.ndarray: Normalized audio signal.
    """
    x = np.asarray(x, dtype=np.float32)
    mx = np.max(np.abs(x)) + 1e-9 # Add epsilon for safety
    return x / mx

def read_wav_mono(path, target_sr=SR):
    """
    Reads a WAV file, converts it to mono (if stereo), and normalizes it.

    Args:
        path (str): Path to the WAV file.
        target_sr (int): The expected sample rate (used for context, no actual resampling here).

    Returns:
        tuple: (normalized_audio_array, sample_rate)
    """
    sr,y = wavfile.read(path)
    y = y.astype(np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1) # Convert stereo to mono by averaging channels
    # Resampling step omitted for simplicity in a self-contained notebook
    return norm_audio(y), target_sr

def write_wav(path, y, sr=SR):
    """
    Writes a normalized float audio array to a 16-bit WAV file.

    Args:
        path (str): Output path for the WAV file.
        y (np.ndarray): Audio signal (normalized float).
        sr (int): Sample rate.
    """
    y = np.asarray(y, dtype=np.float32)
    # Re-normalize and scale to fit 16-bit integer range (approx +/- 32767)
    y = (y/(np.max(np.abs(y)) + 1e-9)*0.99)
    wavfile.write(path, sr, (y*32767.0).astype(np.int16))

# ----------------------------------------------------------------------
## Signal Transforms (Simplified)
# ----------------------------------------------------------------------
# TensorFlow Signal (tfs) functions for STFT and ISTFT, which are crucial for
# converting the time-domain audio to the time-frequency domain (spectrogram)
# and back.

def stft(sig):
    """
    Computes the Short-Time Fourier Transform (STFT) of an audio signal.

    Args:
        sig (tf.Tensor): Input audio waveform.

    Returns:
        tf.Tensor: Complex-valued STFT output (Time x Frequency bins).
    """
    return tfs.stft(
        sig,
        frame_length=WIN_LENGTH,
        frame_step=HOP,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window # Use Hann window for synthesis/analysis
    )

def istft(stft_c, length=SEGMENT):
    """
    Computes the Inverse Short-Time Fourier Transform (ISTFT) to reconstruct the waveform.

    Args:
        stft_c (tf.Tensor): Complex-valued STFT.
        length (int): Target length of the output waveform.

    Returns:
        tf.Tensor: Reconstructed audio waveform.
    """
    return tfs.inverse_stft(
        stft_c,
        frame_length=WIN_LENGTH,
        frame_step=HOP,
        window_fn=tf.signal.hann_window,
    )

def eps():
    """Returns the global epsilon value for numerical stability."""
    return EPSILON

# ----------------------------------------------------------------------
## Visualization functions
# ----------------------------------------------------------------------
# Functions for plotting audio waveforms, spectrograms, and the predicted mask
# for visual analysis and comparison of model performance.

def plot_waveforms(noisy, clean=None, enhanced=None, sr=SR, title='Waveforms'):
    """Plots the time-domain waveforms for noisy, clean, and enhanced audio."""
    plt.figure(figsize=(12, 3))
    t = np.arange(len(noisy))/sr
    plt.plot(t, noisy, label='Noisy', linewidth=0.8)
    if clean is not None:
        plt.plot(t[:len(clean)], clean, label='Clean', alpha=0.7, linewidth=0.8)
    if enhanced is not None:
        plt.plot(t[:len(enhanced)], enhanced, label='Enhanced', alpha=0.9, linewidth=0.8)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.legend()
    plt.tight_layout()
    plt.show()

def spec_db(mag):
    """Converts a magnitude spectrogram to the decibel (dB) scale."""
    return 20.0 * np.log10(np.maximum(mag, 1e-8))

def plot_spectrograms(noisy_mag, clean_mag=None, enhanced_mag=None, sr=SR, title="Spectrograms"):
    """Plots the magnitude spectrograms (in dB) for comparison."""
    num_plots = 1 + (1 if clean_mag is not None else 0) + (1 if enhanced_mag is not None else 0)
    fig, axs = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4))
    if num_plots == 1:
        axs = np.array([axs])

    specs = [noisy_mag]
    titles = ["Noisy | dB"]
    if clean_mag is not None:
        specs.append(clean_mag)
        titles.append("Clean | dB")
    if enhanced_mag is not None:
        specs.append(enhanced_mag)
        titles.append("Enhanced | dB")

    for i, (spec, title) in enumerate(zip(specs, titles)):
        # Display the spectrogram, using an extent for correct axis labels (Time on X, Freq on Y)
        im = axs[i].imshow(spec_db(spec).T, origin="lower", aspect="auto",
                           extent=[0, spec.shape[0]*HOP/sr, 0, sr/2])
        axs[i].set_title(title); axs[i].set_xlabel("Time [s]")
        if i == 0: axs[i].set_ylabel("Freq [Hz]")
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_mask(mask, title="Predicted Mask"):
    """Visualizes the predicted time-frequency mask."""
    plt.figure(figsize=(6,4))
    plt.imshow(mask.T, origin="lower", aspect="auto",
               extent=[0, mask.shape[0]*HOP/SR, 0, SR/2])
    plt.title(title); plt.xlabel("Time [s]"); plt.ylabel("Freq [Hz]")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------
## 📊 EDA Functions (NEW)
# ----------------------------------------------------------------------
# Exploratory Data Analysis (EDA) functions to characterize the synthetic
# noisy/clean audio data, focusing on amplitude (RMS) and noise level (SNR).

def plot_rms_distributions(clean_rms, noisy_rms, noise_rms, title="RMS Distributions"):
    """Plots the histograms of Root Mean Square (RMS) values for various audio segments."""
    plt.figure(figsize=(10, 4))
    
    plt.hist(clean_rms, bins=50, alpha=0.6, label='Clean Audio RMS', density=True)
    plt.hist(noisy_rms, bins=50, alpha=0.6, label='Noisy Audio RMS', density=True)
    plt.hist(noise_rms, bins=50, alpha=0.6, label='Noise Audio RMS', density=True)
    
    plt.title(title)
    plt.xlabel('Root Mean Square (RMS) Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_snr_distribution(snr_values_db, title="Distribution of Generated SNRs"):
    """Plots the histogram of Signal-to-Noise Ratios (in dB)."""
    plt.figure(figsize=(8, 4))
    
    plt.hist(snr_values_db, bins=50, alpha=0.7, color='purple')
    
    plt.title(title)
    plt.xlabel('Signal-to-Noise Ratio (SNR) in dB')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()

def calculate_batch_metrics(dataset, num_batches_to_sample=100):
    """
    Samples batches from the dataset to calculate segment-wise RMS and SNR statistics.
    
    Args:
        dataset (tf.data.Dataset): The training dataset.
        num_batches_to_sample (int): Number of batches to sample for analysis.

    Returns:
        tuple: (all_clean_rms, all_noisy_rms, all_noise_rms, all_snr_db)
    """
    all_clean_rms, all_noisy_rms, all_noise_rms, all_snr_db = [], [], [], []
    
    for i, (noisy_batch, clean_batch) in enumerate(dataset.take(num_batches_to_sample)):
        # Convert to numpy for easy calculation
        noisy = noisy_batch.numpy()
        clean = clean_batch.numpy()
        
        # Noise is approximated as the difference between noisy and clean signal
        noise = noisy - clean
        
        # Calculate RMS (Root Mean Square): sqrt(mean(x^2))
        clean_rms = np.sqrt(np.mean(clean**2, axis=1))
        noisy_rms = np.sqrt(np.mean(noisy**2, axis=1))
        noise_rms = np.sqrt(np.mean(noise**2, axis=1))
        
        # Calculate SNR in dB: 10 * log10(P_signal / P_noise)
        P_clean = np.mean(clean**2, axis=1) + EPSILON # Signal Power
        P_noise = np.mean(noise**2, axis=1) + EPSILON # Noise Power
        snr_db = 10 * np.log10(P_clean / P_noise)
        
        all_clean_rms.extend(clean_rms)
        all_noisy_rms.extend(noisy_rms)
        all_noise_rms.extend(noise_rms)
        all_snr_db.extend(snr_db)
        
        if i >= num_batches_to_sample:
            break
            
    return np.array(all_clean_rms), np.array(all_noisy_rms), np.array(all_noise_rms), np.array(all_snr_db)

# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
## Synthetic Noise Generator
# ----------------------------------------------------------------------
# Functions to create synthetic clean and noise audio segments and mix them
# at a target Signal-to-Noise Ratio (SNR).

def gen_tone(duration, sr=SR):
    """Generates a random pure tone or chirp with a simple envelope."""
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    f0 = np.random.uniform(100, 1000)
    y = np.sin(2*np.pi*f0*t)
    if np.random.rand() < 0.5:
        f1 = np.random.uniform(200, 2000)
        chirp = np.sin(2*np.pi*(f0 + (f1-f0)*t/duration)*t)
        y = 0.6*y + 0.4*chirp
    # Apply a smooth fade-in/fade-out envelope
    env = 0.5*(1-np.cos(2*np.pi*np.minimum(1.0, t/duration)))
    return norm_audio(y * env)

def gen_noise(duration, sr=SR):
    """Generates a mixture of white, pink, and babble (synthetic speech-like) noise."""
    n = int(sr*duration)
    # White Noise (Equal energy per frequency)
    white = np.random.randn(n).astype(np.float32)
    # Pink Noise (Energy decreases by 3dB per octave)
    freqs = np.fft.rfftfreq(n, 1/sr)
    pink_spec = (np.random.randn(len(freqs))+1j*np.random.randn(len(freqs)))/np.maximum(freqs, 1.0)
    pink = np.fft.irfft(pink_spec, n=n).astype(np.float32)
    # Babble Noise (Multiple tones summed together)
    babble = np.zeros(n, dtype=np.float32)
    for _ in range(np.random.randint(3, 7)):
        babble += gen_tone(duration, sr)
    babble = babble / (np.max(np.abs(babble)) + 1e-9)
    # Mix the three noise types
    mix = 0.5*white/np.max(np.abs(white)+1e-9) + 0.3*pink/np.max(np.abs(pink)+1e-9) + 0.2*babble
    return norm_audio(mix)


def random_segment(y, length):
    """Extracts a random segment of a specific length, padding if necessary."""
    if len(y) <= length:
        pad = length - len(y)
        # Reflect padding is used if the signal is shorter than the segment length
        y = np.pad(y, (0, pad), mode='reflect')
        return y
    start = np.random.randint(0, len(y)-length)
    return y[start:start+length]

def mix_clean_noise(clean, noise, snr_db=None):
    """
    Adjusts the noise level and mixes clean audio and noise at a specified SNR.

    Args:
        clean (np.ndarray): Clean audio segment.
        noise (np.ndarray): Noise audio segment.
        snr_db (float, optional): Target SNR in dB. If None, a random SNR is chosen.

    Returns:
        tuple: (noisy_audio, normalized_clean_audio, normalized_noise_audio)
    """
    if snr_db is None:
        # Generate a random SNR between -5 dB (very noisy) and 15 dB (mostly clean)
        snr_db = np.random.uniform(-5, 15)
    
    # Normalize clean and noise to have unit standard deviation (RMS) before mixing
    c = clean / (np.std(clean)+1e-9)
    n = noise / (np.std(noise)+1e-9)
    
    rms_c = np.sqrt(np.mean(c**2)+1e-9)
    rms_n = np.sqrt(np.mean(n**2)+1e-9)

    # Calculate target noise RMS based on SNR formula: SNR_dB = 20 * log10(RMS_c / RMS_n)
    target_rms_n = rms_c / (10**(snr_db/20.0))
    
    # Scale the noise to the target RMS
    n = n * (target_rms_n / (rms_n + 1e-9))
    
    # Mix
    noisy = c + n
    
    return norm_audio(noisy), norm_audio(c), norm_audio(n)

# ----------------------------------------------------------------------
## Pipeline
# ----------------------------------------------------------------------
# Functions to build the data loading and preprocessing pipeline using TensorFlow
# datasets for efficient training.

def wav_loader_factory(clean_paths, noise_paths):
    """
    Factory function to create the data loading and mixing function.
    In a real scenario, clean_paths/noise_paths would be lists of file paths.
    Here, they are empty, forcing the use of synthetic data generation.
    """
    def load_and_mix(_):
        """Loads or generates clean and noise segments, then mixes them."""
        # Clean Signal
        if clean_paths:
            cp = random.choice(clean_paths)
            c, _sr = read_wav_mono(cp, SR)
        else:
            # Synthetic "clean" signal (pure tone/chirp)
            c = gen_tone(SEGMENT_SEC)
            
        # Noise Signal (90% chance to use real noise if paths are provided)
        if noise_paths and np.random.rand() < 0.9:
            npth = random.choice(noise_paths)
            n, _sr = read_wav_mono(npth, SR)
        else:
            # Synthetic noise (mixed white, pink, babble)
            n = gen_noise(SEGMENT_SEC + 1.0)
            
        # Segment the signals
        c_seg = random_segment(c, SEGMENT)
        n_seg = random_segment(n, SEGMENT)
        
        # Mix with a random SNR
        noisy, clean, noise = mix_clean_noise(c_seg, n_seg)
        
        # Return the (noisy_input, clean_target) pair
        return noisy.astype(np.float32), clean.astype(np.float32)
    return load_and_mix

def tf_dataset(clean_paths, noise_paths, batch_size, steps):
    """
    Creates a TensorFlow Dataset from the audio generation function.

    Args:
        clean_paths, noise_paths (list): File paths (empty in this notebook).
        batch_size (int): Number of samples per batch.
        steps (int): Number of training steps per epoch.

    Returns:
        tf.data.Dataset: Configured dataset for training/validation.
    """
    def gen():
        """The actual Python generator that yields (noisy, clean) pairs."""
        loader = wav_loader_factory(clean_paths, noise_paths)
        # Generate enough data for multiple epochs (steps * batch_size * 2)
        for _ in range(steps * batch_size * 2):
            yield loader(None)
            
    # Define the shape and type of the generator output
    output_sig = (tf.TensorSpec(shape=(SEGMENT,), dtype=tf.float32),
                  tf.TensorSpec(shape=(SEGMENT,), dtype=tf.float32))
                  
    # Create the dataset from the generator
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)
    ds = ds.shuffle(8192, reshuffle_each_iteration=True) # Shuffle the data
    ds = ds.batch(batch_size, drop_remainder=True) # Batch the data
    ds = ds.prefetch(tf.data.AUTOTUNE) # Prefetch for faster I/O operations
    return ds

# Instantiate the training and validation datasets (using synthetic data)
train_ds = tf_dataset([], [], BATCH_SIZE, STEPS_PER_EPOCH)
val_ds = tf_dataset([], [], BATCH_SIZE, VAL_STEPS)

# ----------------------------------------------------------------------
## Model I/O & Preprocessing (Shared Logic)
# ----------------------------------------------------------------------
# These functions define the common input processing (STFT) and output
# reconstruction (ISTFT) steps, shared by both U-Net and ResNet.

def create_model_input_layers(name):
    """
    Creates the input layer and the initial STFT layers to get magnitude and phase.
    
    Args:
        name (str): Prefix for layer names.

    Returns:
        tuple: (inp, M, mag, phase) where M is the magnitude spectrogram with 
               an added channel dimension (Time x Freq x 1).
    """
    inp = keras.Input(shape=(SEGMENT,), name='audio_in') # Raw audio input (Time,)

    # STFT & Magnitude/Phase Extraction
    Xc = layers.Lambda(stft, name=name+'_stft_complex')(inp) # Complex STFT
    mag = layers.Lambda(tf.abs, name=name+'_magnitude')(Xc) # Magnitude spectrogram
    phase = layers.Lambda(tf.math.angle, name=name+'_phase')(Xc) # Phase spectrogram (angle)

    # Add channel dimension: (B, T, F) -> (B, T, F, 1) to match 2D CNN input convention
    M = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name=name+'_expand_dims_channel')(mag)
    return inp, M, mag, phase

def create_model_output_layers(mask_output, mag_input, phase_input, name):
    """
    Applies the network's predicted mask, reconstructs the complex STFT, and performs ISTFT.

    Args:
        mask_output (tf.Tensor): Output from the main network body (e.g., U-Net decoder).
        mag_input (tf.Tensor): The noisy magnitude spectrogram from `create_model_input_layers`.
        phase_input (tf.Tensor): The noisy phase spectrogram.
        name (str): Prefix for layer names.

    Returns:
        tuple: (enhanced_audio, output_mask, input_magnitude)
    """
    time_steps = TIME_STEPS
    freq_bins = FREQ_BINS

    # 1. Final Mask Activation: Constrain mask to [0, 1] using Sigmoid
    out_mask = layers.Conv2D(1, 1, activation='sigmoid', name=name+'_mask_output')(mask_output)
    # Remove channel dimension: (B, T, F, 1) -> (B, T, F)
    out_mask = layers.Lambda(tf.squeeze, arguments={'axis': -1}, name=name+'_squeeze_mask')(out_mask)

    # 2. Enhanced Magnitude: Apply the predicted mask to the noisy magnitude
    enh_mag = layers.Multiply(name=name+'_apply_mask')([out_mask, mag_input])

    # 3. Reconstruct Complex STFT: Use enhanced magnitude and NOISY phase (a common simplification)
    enh_complex = layers.Lambda(
        # Complex(real, imag) = Complex(Mag * cos(Phase), Mag * sin(Phase))
        lambda x: tf.complex(x[0] * tf.cos(x[1]), x[0] * tf.sin(x[1])),
        output_shape=(time_steps, freq_bins),
        name=name+'_complex-reconstruct'
    )([enh_mag, phase_input])

    # 4. Inverse STFT: Convert back to time-domain audio
    enh_audio = layers.Lambda(
        istft,
        output_shape=(SEGMENT,),
        name=name+'_istft_final'
    )(enh_complex)

    return enh_audio, out_mask, mag_input


# ----------------------------------------------------------------------
## ResNet Architecture
# ----------------------------------------------------------------------
# Definition of the Residual Network (ResNet) architecture for denoising.

def resnet_block(x, filters, kernel_size=(3, 3), name='res_block'):
    """
    Simplified ResNet block (Pre-Activation style) with a residual connection.

    Args:
        x (tf.Tensor): Input tensor (Time x Freq x Channels).
        filters (int): Number of output filters for the convolutions.
        kernel_size (tuple): Size of the convolutional kernel.
        name (str): Name prefix for the block's layers.

    Returns:
        tf.Tensor: Output of the residual block.
    """
    shortcut = x

    # First Conv -> BN -> ReLU
    x = layers.Conv2D(filters, kernel_size, padding='same', name=name+'_conv1')(x)
    x = layers.BatchNormalization(name=name+'_bn1')(x)
    x = layers.Activation('relu', name=name+'_relu1')(x)

    # Second Conv -> BN
    x = layers.Conv2D(filters, kernel_size, padding='same', name=name+'_conv2')(x)
    x = layers.BatchNormalization(name=name+'_bn2')(x)

    # Handle dimension mismatch for the shortcut connection (projection shortcut)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same', name=name+'_shortcut_conv')(shortcut)
        shortcut = layers.BatchNormalization(name=name+'_shortcut_bn')(shortcut)

    # Add the shortcut (residual connection) to the main path
    x = layers.Add(name=name+'_add')([x, shortcut])
    
    # Final ReLU activation
    x = layers.Activation('relu', name=name+'_relu_out')(x)
    return x

def build_resnet():
    """Builds the full ResNet model for Spectrogram Masking."""
    name = 'ResNet'
    # Initial I/O setup (audio -> magnitude spectrogram)
    inp, M, mag, phase = create_model_input_layers(name)

    # Initial Convolution
    x = layers.Conv2D(32, (7, 7), padding='same', name=name+'_initial_conv')(M)
    x = layers.BatchNormalization(name=name+'_initial_bn')(x)
    x = layers.Activation('relu', name=name+'_initial_relu')(x)

    # ResNet Blocks (Maintain spatial resolution T x F, only change channel depth)
    x = resnet_block(x, 64, name=name+'_res1')
    x = resnet_block(x, 64, name=name+'_res2')
    x = resnet_block(x, 128, name=name+'_res3')
    x = resnet_block(x, 128, name=name+'_res4')
    x = resnet_block(x, 128, name=name+'_res5')

    # Output reconstruction (masking, complex-reconstruction, ISTFT)
    enh_audio, out_mask, _mag = create_model_output_layers(x, mag, phase, name)
    return keras.Model(inp, outputs=[enh_audio, out_mask, mag], name='ResNet_Denoiser')


# ----------------------------------------------------------------------
## U-Net Architecture
# ----------------------------------------------------------------------
# Definition of the U-Net architecture, known for image/spectrogram segmentation/generation.

def unet_block(x, filters, name, down=True):
    """
    Basic downsampling (encoder) or upsampling (decoder) block for U-Net.
    
    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters.
        name (str): Name prefix.
        down (bool): If True, it's a downsampling block (Conv2D with stride 2).
                     If False, it's an upsampling block (Conv2DTranspose with stride 2).

    Returns:
        tf.Tensor: Output of the block.
    """
    if down:
        # Encoder: Conv2D with stride 2 reduces Time and Freq dimensions by half
        x = layers.Conv2D(filters, 3, strides=2, padding='same', name=name+'_conv')(x)
        x = layers.BatchNormalization(name=name+'_bn')(x)
        x = layers.Activation('relu', name=name+'_relu')(x)
        return x
    else:
        # Decoder: Conv2DTranspose with stride 2 increases Time and Freq dimensions by two
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same', name=name+'_deconv')(x)
        x = layers.BatchNormalization(name=name+'_bn')(x)
        x = layers.Activation('relu', name=name+'_relu')(x)
        return x

def build_unet():
    """Builds the full U-Net model for Spectrogram Masking."""
    name = 'UNet'
    # Initial I/O setup (audio -> magnitude spectrogram)
    inp, M, mag, phase = create_model_input_layers(name)

    # Encoder (Downsampling Path)
    # E0: Initial 3x3 Conv without pooling/striding (T x F x 32)
    e1 = layers.Conv2D(32, 3, padding='same', activation='relu', name=name+'_e0_conv')(M) 
    # D1: (T/2 x F/2 x 64)
    d1 = unet_block(e1, 64, name+'_down1') 
    # D2: (T/4 x F/4 x 128)
    d2 = unet_block(d1, 128, name+'_down2')
    # D3: (T/8 x F/8 x 256)
    d3 = unet_block(d2, 256, name+'_down3')
    # Bottleneck: Deepest layer
    bott = layers.Conv2D(512, 3, padding='same', activation='relu', name=name+'_bottleneck')(d3)

    # Decoder (Upsampling Path)
    # U3: Up-sample back to D2 size (T/4 x F/4 x 256)
    u3 = unet_block(bott, 256, name+'_up3', down=False)
    # Cropping/Padding is often needed in U-Net due to odd dimensions from stride 2
    u3 = layers.Cropping2D(cropping=((0, 1), (0, 1)), name=name+'_crop_u3_to_d2')(u3)
    # Skip connection from encoder (d2) to decoder (u3)
    u3 = layers.Concatenate()([u3, d2])

    # U2: Up-sample back to D1 size (T/2 x F/2 x 128)
    u2 = unet_block(u3, 128, name+'_up2', down=False)
    u2 = layers.Cropping2D(cropping=((0, 1), (0, 1)), name=name+'_crop_u2_to_d1')(u2)
    u2 = layers.Concatenate()([u2, d1])

    # U1: Up-sample back to E1 size (T x F x 64)
    u1 = unet_block(u2, 64, name+'_up1', down=False)
    u1 = layers.Cropping2D(cropping=((0, 0), (0, 1)), name=name+'_crop_u1_to_e1')(u1)
    u1 = layers.Concatenate()([u1, e1])

    # Output reconstruction (masking, complex-reconstruction, ISTFT)
    enh_audio, out_mask, _mag = create_model_output_layers(u1, mag, phase, name)
    return keras.Model(inp, outputs=[enh_audio, out_mask, mag], name='UNet_Denoiser')

# Instantiate both models
model_unet = build_unet()
model_resnet = build_resnet()

# ----------------------------------------------------------------------
# (Model Visualization Chunk)
# This chunk generates and displays the architectural diagrams for both models 
# if the necessary plotting utilities (pydot, graphviz) are installed.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
## Simplified and Stable Loss Functions
# ----------------------------------------------------------------------
# Loss functions are defined to measure the difference between the enhanced 
# audio and the clean target, operating in both the time and magnitude domains.

def l1_mag_loss(true_audio, pred_audio):
    """
    L1 loss on the magnitude spectrograms (Mean Absolute Error).
    Penalizes differences in frequency content.

    Args:
        true_audio, pred_audio (tf.Tensor): Clean and enhanced time-domain audio.

    Returns:
        tf.Tensor: Mean L1 magnitude loss.
    """
    Y = stft(true_audio) # Clean STFT
    P = stft(pred_audio) # Enhanced STFT
    return tf.reduce_mean(tf.abs(tf.abs(Y) - tf.abs(P)))

def l1_time_loss(true_audio, pred_audio):
    """
    L1 loss on the time-domain waveform (Mean Absolute Error).
    Encourages waveform matching.

    Args:
        true_audio, pred_audio (tf.Tensor): Clean and enhanced time-domain audio.

    Returns:
        tf.Tensor: Mean L1 time loss.
    """
    return tf.reduce_mean(tf.abs(true_audio - pred_audio))

def total_loss(true_audio, pred_audio):
    """
    A weighted combination of time-domain and magnitude-domain L1 losses.

    Args:
        true_audio, pred_audio (tf.Tensor): Clean and enhanced time-domain audio.

    Returns:
        tf.Tensor: Combined loss value.
    """
    return (0.6 * l1_time_loss(true_audio, pred_audio) +
            0.4 * l1_mag_loss(true_audio, pred_audio))

# ----------------------------------------------------------------------
## SI-SDR Metric
# ----------------------------------------------------------------------
# Scale-Invariant Source-to-Distortion Ratio (SI-SDR) is a standard, robust metric
# for evaluating source separation and enhancement quality. Higher is better.

def si_sdr(true_audio, pred_audio):
    """
    Calculates the SI-SDR metric between true and predicted audio.
    
    Args:
        true_audio (tf.Tensor): The clean target signal (x).
        pred_audio (tf.Tensor): The enhanced signal (s).

    Returns:
        tf.Tensor: The SI-SDR value in dB.
    """
    x = true_audio
    s = pred_audio

    # Center the signals (zero-mean)
    x_zm = x - tf.reduce_mean(x, axis=-1, keepdims=True)
    s_zm = s - tf.reduce_mean(x, axis=-1, keepdims=True)

    # Calculate alpha for optimal scaling (projection of s onto x)
    alpha_num = tf.reduce_sum(s_zm * x_zm, axis=-1, keepdims=True)
    alpha_den = tf.reduce_sum(x_zm**2, axis=-1, keepdims=True) + eps()
    alpha = alpha_num / alpha_den
    
    # Signal component (projection)
    proj = alpha * x_zm

    # Error component (interference + noise + artifacts)
    e = s_zm - proj

    # Power calculation
    signal_power = tf.reduce_sum(proj**2, axis=-1)
    error_power = tf.reduce_sum(e**2, axis=-1)
    
    # SI-SDR in linear scale
    ratio = (signal_power + eps()) / (error_power + eps())

    # SI-SDR in dB
    si_sdr_val = 10.0 * tf.math.log(ratio) / tf.math.log(10.0)
    return si_sdr_val


# ----------------------------------------------------------------------
## Optimizer (FIXED: Separate optimizers)
# ----------------------------------------------------------------------
# Custom learning rate schedule (Warmup Cosine Decay) and optimizers setup.

class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning Rate Scheduler that applies a linear warmup followed by a 
    cosine decay to the base learning rate.
    """
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base_lr = base_lr
        self.warmup = warmup_steps
        self.total = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = tf.cast(self.warmup, tf.float32)
        total = tf.cast(self.total, tf.float32)
        
        # If step < warmup_steps, LR increases linearly.
        # Otherwise, LR follows a cosine curve decaying to zero.
        lr = tf.where(
            step < warm,
            self.base_lr * (step / tf.maximum(1.0, warm)),
            0.5 * self.base_lr * (1 + tf.cos(np.pi * (step - warm) / tf.maximum(1.0, (total-warm))))
        )
        return lr


TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
lr_schedule = WarmupCosine(LEARNING_RATE, WARMUP_STEPS, TOTAL_STEPS)

# Separate Adam optimizers are created for each model
opt_unet = keras.optimizers.Adam(learning_rate=lr_schedule, name='Adam_UNet')
opt_resnet = keras.optimizers.Adam(learning_rate=lr_schedule, name='Adam_ResNet')


# ----------------------------------------------------------------------
## Exponential Moving Average (EMA)
# ----------------------------------------------------------------------
# Implementation of Exponential Moving Average, which often leads to more stable
# and better performing final models by averaging weights over training steps.

class EMA:
    """
    Manages Exponential Moving Average of model weights.
    The 'shadow' weights are used for inference/evaluation, while the regular 
    weights are used for training.
    """
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        # Only track trainable variables that support assignment (i.e., not constant)
        self.trainable_vars = [w for w in model.weights if hasattr(w, 'assign')]
        # Initialize shadow variables with the current model weights
        self.shadow = [tf.Variable(tf.identity(w)) for w in self.trainable_vars]

    def update(self, model):
        """
        Updates the shadow weights: s = decay * s + (1 - decay) * w.
        """
        # Ensure we're updating the correct weights (in case model has changed)
        self.trainable_vars = [w for w in model.weights if hasattr(w, 'assign')]
        for w, s in zip(self.trainable_vars, self.shadow):
            s.assign(self.decay * s + (1.0 - self.decay) * w)

    def apply_to(self, model):
        """
        Replaces the current model weights with the EMA (shadow) weights for inference.
        Saves a backup of the original weights.
        """
        self.trainable_vars = [w for w in model.weights if hasattr(w, 'assign')]
        # Store a backup of the current training weights
        self.backup = [tf.Variable(tf.identity(w)) for w in self.trainable_vars]
        # Apply shadow weights to the model
        for w, s in zip(self.trainable_vars, self.shadow):
            w.assign(s)

    def restore(self, model):
        """
        Restores the original training weights from the backup after inference.
        """
        self.trainable_vars = [w for w in model.weights if hasattr(w, 'assign')]
        if self.backup:
            for w, b in zip(self.trainable_vars, self.backup):
                w.assign(b)
            self.backup = None

# Initialize EMA for both models
ema_unet = EMA(model_unet, EMA_DECAY)
ema_resnet = EMA(model_resnet, EMA_DECAY)

# ----------------------------------------------------------------------
## Custom Training Loop with Gradient Safety
# ----------------------------------------------------------------------
# Training logic using TensorFlow's graph mode (`@tf.function`) for performance,
# including mechanisms for gradient clipping and NaN/Inf gradient filtering.

# Metrics for UNet
train_loss_unet = keras.metrics.Mean(name='train_loss_unet')
val_loss_unet = keras.metrics.Mean(name='val_loss_unet')
train_si_sdr_unet = keras.metrics.Mean(name='train_si_sdr_unet')
val_si_sdr_unet = keras.metrics.Mean(name='val_si_sdr_unet')

# Metrics for ResNet
train_loss_resnet = keras.metrics.Mean(name='train_loss_resnet')
val_loss_resnet = keras.metrics.Mean(name='val_loss_resnet')
train_si_sdr_resnet = keras.metrics.Mean(name='train_si_sdr_resnet')
val_si_sdr_resnet = keras.metrics.Mean(name='val_si_sdr_resnet')


def apply_gradients_safely(model, loss_metric, optimizer, ema, noisy, clean):
    """
    Performs a single forward/backward pass and weight update for one model.
    Includes gradient clipping and checks for unstable (NaN/Inf) gradients.
    
    Args:
        model (keras.Model): The model to train (UNet or ResNet).
        loss_metric (keras.metrics.Mean): The training loss metric for this model.
        optimizer (keras.optimizers.Optimizer): The optimizer for this model.
        ema (EMA): The EMA object for this model.
        noisy (tf.Tensor): Noisy audio batch.
        clean (tf.Tensor): Clean audio batch.

    Returns:
        tf.Tensor: Enhanced audio output.
    """
    with tf.GradientTape() as tape:
        # Forward pass: model predicts enhanced audio, mask, and noisy magnitude
        enhanced_audio, mask, mag = model(noisy, training=True)
        # Calculate loss
        loss_value = total_loss(clean, enhanced_audio)
        
        # Cast loss to float32 if mixed precision is used
        if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
            loss_value = tf.cast(loss_value, tf.float32)

    # Calculate gradients
    grads = tape.gradient(loss_value, model.trainable_variables)

    # Gradient Clipping: prevents large gradients from causing instability
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=5.0)

    # Gradient Safety: Filter out NaN or Inf gradients
    filtered_grads_and_vars = []
    for grad, var in zip(grads, model.trainable_variables):
        if grad is not None:
            grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
            grad = tf.where(tf.math.is_inf(grad), tf.zeros_like(grad), grad)
            filtered_grads_and_vars.append((grad, var))

    # Apply gradients and update weights
    optimizer.apply_gradients(filtered_grads_and_vars)

    # Update Exponential Moving Average of weights
    ema.update(model)
    
    # Update loss metric state
    loss_metric.update_state(loss_value)
    return enhanced_audio

@tf.function
def train_step_models(noisy, clean):
    """Performs a single training step for both UNet and ResNet models simultaneously."""
    # Train UNet
    enh_unet = apply_gradients_safely(model_unet, train_loss_unet, opt_unet, ema_unet, noisy, clean)
    # Update SI-SDR metric for UNet
    train_si_sdr_unet.update_state(si_sdr(clean, enh_unet))

    # Train ResNet
    enh_resnet = apply_gradients_safely(model_resnet, train_loss_resnet, opt_resnet, ema_resnet, noisy, clean)
    # Update SI-SDR metric for ResNet
    train_si_sdr_resnet.update_state(si_sdr(clean, enh_resnet))

@tf.function
def val_step_models(noisy, clean):
    """Performs a single validation step for both models (no gradient calculation)."""
    # Validate UNet
    # Use training=False for evaluation/inference mode (e.g., disables Batch Norm updates/Dropout)
    enh_unet, _, _ = model_unet(noisy, training=False)
    val_loss_unet.update_state(total_loss(clean, enh_unet))
    val_si_sdr_unet.update_state(si_sdr(clean, enh_unet))

    # Validate ResNet
    enh_resnet, _, _ = model_resnet(noisy, training=False)
    val_loss_resnet.update_state(total_loss(clean, enh_resnet))
    val_si_sdr_resnet.update_state(si_sdr(clean, enh_resnet))


# ----------------------------------------------------------------------
# (Training Loop Chunk)
# This section executes the main training loop, performing EDA first, then iterating 
# through epochs, logging metrics, and saving checkpoints for both models.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# (Visualization and Comparison Chunk)
# This section plots the training history (loss and SI-SDR curves) and 
# performs a qualitative inference check on a single validation sample.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
## Inference Export for the Best Model
# ----------------------------------------------------------------------
# This final chunk determines the best performing model based on the validation
# SI-SDR and exports it as a ready-to-use SavedModel, wrapped in a simple class.

final_val_sdr_u = history['unet_val_si_sdr'][-1] if history['unet_val_si_sdr'] else -np.inf
final_val_sdr_r = history['resnet_val_si_sdr'][-1] if history['resnet_val_si_sdr'] else -np.inf

# Select the model with the higher final validation SI-SDR
if final_val_sdr_u >= final_val_sdr_r:
    best_model = model_unet
    best_ema = ema_unet
    export_path = EXPORT_DIR_UNET
    best_model_name = "UNet"
else:
    best_model = model_resnet
    best_ema = ema_resnet
    export_path = EXPORT_DIR_RESNET
    best_model_name = "ResNet"

print(f"\nOverall Best Model (based on final val SI-SDR): **{best_model_name}** ({max(final_val_sdr_u, final_val_sdr_r):.2f}dB)")


best_ema.apply_to(best_model) # Apply EMA weights for export

class InferenceWrapper(keras.Model):
    """
    Simple wrapper class to create a callable inference function with a fixed 
    input signature for TensorFlow SavedModel export.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base
        
    # Defines the signature for the exported function: 1D float32 tensor
    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
    def denoise(self, audio):
        """Denoises an input audio signal."""
        audio = tf.expand_dims(audio, 0) # Add batch dimension
        enh, _mask, _mag = self.base(audio, training=False)
        return tf.squeeze(enh, 0) # Remove batch dimension for output

# Export the best model
infer_model = InferenceWrapper(best_model)
try:
    tf.saved_model.save(infer_model, export_path)
    print(f"Best model ({best_model_name}) Exported to: {export_path}")
except Exception as e:
    print(f"WARNING: Could not export model (check training success). Error: {e}")

best_ema.restore(best_model) # Restore original training weights

print("\nReady! Use: denoise_file('/kaggle/input/your.wav', out_path='/kaggle/working/clean.wav', show=True)")

def denoise_file(wav_path, out_path=None, show=True):
    """
    A utility function to load an audio file, denoise it using the best trained 
    model, and optionally save and visualize the result.
    
    NOTE: This simple implementation is primarily designed for segments of length 
    `SEGMENT` (32000 samples) and may not handle arbitrary lengths optimally 
    without block processing.
    """
    y, _ = read_wav_mono(wav_path, SR)
    T = len(y)

    # Pad the input audio to ensure its length is a multiple of SEGMENT
    pad = ( (math.ceil(T / SEGMENT) * SEGMENT) - T )
    y_pad = np.pad(y, (0, pad), mode='reflect').astype(np.float32)

    best_ema.apply_to(best_model) # Use EMA weights for inference
    if T != SEGMENT:
        print("WARNING: Inference function is simplistic. Only inputs of length 32000 are guaranteed to work.")

    # Run inference
    enh, _, _ = best_model(tf.convert_to_tensor(y_pad[None, ...]), training=False)
    best_ema.restore(best_model) # Restore training weights

    # Trim padding and get the final enhanced audio
    enh = enh.numpy()[0][:T]

    if out_path:
        write_wav(out_path, enh, SR)

    if show:
        print("Playing (Noisy -> Enhanced)")
        display(Audio(y, rate=SR))
        display(Audio(enh, rate=SR))
        N_spec = stft(y)
        E_spec = stft(enh)
        plot_spectrograms(np.abs(N_spec).numpy(), None, np.abs(E_spec).numpy(), sr=SR, title="Noisy / Enhanced Spectrogram (dB)")
        plot_waveforms(y, None, enh, sr=SR, title="Inference Waveforms")

    return enh

# Demo playback:
demo_clean = gen_tone(2.0)
demo_noise = gen_noise(2.0)
demo_noisy, demo_clean, _ = mix_clean_noise(demo_clean, demo_noise, snr_db=0.0)
print("\nDemo playback:")
display(Audio(demo_noisy, rate=SR))
best_ema.apply_to(best_model)
demo_enh, _, _ = best_model(tf.convert_to_tensor(demo_noisy[None, ...]), training=False)
best_ema.restore(best_model)
demo_enh = demo_enh.numpy()[0]
display(Audio(demo_enh, rate=SR))
N_demo_spec = stft(demo_noisy)
E_demo_spec = stft(demo_enh)
plot_spectrograms(np.abs(N_demo_spec).numpy(), None, np.abs(E_demo_spec).numpy(), title="Demo Noisy / Enhanced Spec (dB)")
