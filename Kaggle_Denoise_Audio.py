#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-05T18:07:41.476Z
"""

import os, sys, math, random, glob, shutil, time, functools, itertools
from pathlib import Path 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow import keras
import tensorflow.signal as tfs 

from scipy.io import wavfile 
from IPython.display import Audio, display

# ## Environment Constants


SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SR = 16000
SEGMENT_SEC = 2.0
SEGMENT = int(SR * SEGMENT_SEC)
N_FFT = 1024 
HOP = 256
WIN_LENGTH = 1024 
N_MELS = 128
PAD_MODE = 'REFLECT'

BATCH_SIZE = 8
EPOCHS = 15
STEPS_PER_EPOCH = 600
VAL_STEPS = 80
LEARNING_RATE = 3e-4 
WARMUP_STEPS = 500
EMA_DECAY = 0.999
CHECKPOINT_DIR = '/kaggle/working/denoiser_ckpt'
EXPORT_DIR = '/kaggle/working/denoiser_export'

os.makedirs(CHECKPOINT_DIR, exist_ok = True)
os.makedirs(EXPORT_DIR, exist_ok = True)

# ## Audio Utility IO functions


def norm_audio(x):
    x = np.asarray(x, dtype=np.float32)
    mx = np.max(np.abs(x)) + 1e-9
    return x / mx

def read_wav_mono(path, target_sr=SR):
    sr,y = wavfile.read(path)
    y = y.astype(np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr!= target_sr:
        y = tf.audio.resample(y, sr, target_sr).numpy()
    return norm_audio(y), target_sr

def write_wav(path, y, sr=SR):
    y = np.asarray(y, dtype=np.float32)
    y = (y/(np.max(np.abs(y)) + 1e-9)*0.99)
    # scaling up from [-1, 1] to 32767
    wavfile.write(path, sr, (y*32767.0).astype(np.int16))

# ## Signal Transforms


# performs short time fourier transform 
# outputs 2d array of complex numbers
def stft(sig):
    return tfs.stft(
        sig, 
        frame_length=WIN_LENGTH, # how many samples to look at once
        frame_step=HOP, # how much to hop forward, in our case 1024 - 256 samples will be overlapped
        fft_length=N_FFT, # how many frequency bins result from each analysis
        window_fn=tf.signal.hann_window # smooths edges to avoid sharp transitions
    )

# converts time frequency complex representation to time domain audio signal
def istft(stft_c, length):
    return tfs.inverse_stft(
        stft_c,
        frame_length=WIN_LENGTH,
        frame_step=HOP,
        window_fn=tf.signal.hann_window,
        output_length=length
    )

def complex_mag(stft_c):
    return tf.abs(stft_c)

def eps():
    return 1e-8

# goes from linear resolution to mel_resolution
# lower frequency bins are spaced close together (small pitch differences noticable to humans)
MEL_FILTER = tfs.linear_to_mel_weight_matrix(
    num_mel_bins=N_MELS,
    num_spectrogram_bins=N_FFT//2 + 1, # linear frequency bins from STFT input
    sample_rate=SR,
    lower_edge_hertz=0.0,
    upper_edge_hertz=SR/2 # Nyquist frequency (half the sample rate)
)

# ## Visualization functions


def plot_waveforms(noisy, clean=None, enhanced=None, sr=SR, title='Waveforms'):
    plt.figure(figsize=(12, 3))
    t = np.arange(len(noisy))/sr
    plt.plot(t, noisy, label='Noisy', linewidth=0.8)
    if clean is not None:
        plt.plot(t[:len(clean)], clean, label='Clean', alpha=0.7, linewidth=0.8)
    if enchanced is not None:
        plt.plot(t[:len(enhanced)], enhanced, label='Enchanced', alpha=0.9, linewidht=0.8)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.legend()
    plt.tight_layout()
    plt.show()

def spec_db(mag):
    return 20.0 * np.log10(np.maximum(mag, 1e-8))

def plot_spectrograms(noisy_mag, clean_mag=None, enhanced_mag=None, sr=SR, title="Spectrograms"):
    fig, axs = plt.subplots(1, 3 if (clean_mag is not None and enhanced_mag is not None) else 1, figsize=(15, 4))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    im0 = axs[0].imshow(spec_db(noisy_mag).T, origin="lower", aspect="auto", 
                        extent=[0, noisy_mag.shape[0]*HOP/sr, 0, sr/2])
    axs[0].set_title("Noisy | dB")
    axs[0].set_xlabel("Time [s]"); axs[0].set_ylabel("Freq [Hz]")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    if clean_mag is not None and enhanced_mag is not None:
        im1 = axs[1].imshow(spec_db(clean_mag).T, origin="lower", aspect="auto",
                            extent=[0, clean_mag.shape[0]*HOP/sr, 0, sr/2])
        axs[1].set_title("Clean | dB"); axs[1].set_xlabel("Time [s]")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        im2 = axs[2].imshow(spec_db(enhanced_mag).T, origin="lower", aspect="auto",
                            extent=[0, enhanced_mag.shape[0]*HOP/sr, 0, sr/2])
        axs[2].set_title("Enhanced | dB"); axs[2].set_xlabel("Time [s]")
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_mask(mask, title="Predicted Mask"):
    plt.figure(figsize=(6,4))
    plt.imshow(mask.T, origin="lower", aspect="auto",
               extent=[0, mask.shape[0]*HOP/SR, 0, SR/2])
    plt.title(title); plt.xlabel("Time [s]"); plt.ylabel("Freq [Hz]")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()

# ## Synthetic Noise Generator (Clean and Noisy dataset)


def gen_tone(duration, sr=SR):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    # generating a time array 16k points per sec
    f0 = np.random.uniform(100, 1000) # base frequency
    y = np.sin(2*np.pi*f0*t) # pure sine wave at frequency f0
    # blend pure freqency with chirp with 50% probab
    if np.random.rand() < 0.5:
        f1 = np.random.uniform(200, 2000)
        # tone whose frequency changes over time, keeps changing from f0 to f1 linearly
        chirp = np.sin(2*np.pi*(f0 + (f1-f0)*t/duration)*t)
        y = 0.6*y + 0.4*chirp
    env = 0.5*(1-np.cos(2*np.pi*np.minimum(1.0, t/duration)))
    # smooth cosine curve controlling volume over time, prevents sudden starts or stops
    return norm_audio(y * env)

def gen_noise(duration, sr=SR):
    n = int(sr*duration)
    # white noise: energy concentration equal
    white = np.random.randn(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1/sr) # 1d array
    # pink noise: energy concentrate more at lower frequency
    pink_spec = (np.random.randn(len(freqs))+1j*np.random.randn(len(freqs)))/np.maximum(freqs, 1.0)
    # random complex numbers to generate noise, frequency below zero stays same
    pink = np.fft.irfft(pink_spec, n=n).astype(np.float32)
    # convert back to time domain
    babble = np.zeros(n, dtype=np.float32)
    # summing up several tones (3 to 6) to simulate overlapping sounds
    for _ in range(np.random.randint(3, 7)):
        babble += gen_tone(duration, sr)
    babble = babble / (np.max(np.abs(babble)) + 1e-9)
    mix = 0.5*white/np.max(np.abs(white)+1e-9) + 0.3*pink/np.max(np.abs(pink)+1e-9) + 0.2*babble
    return norm_audio(mix)
    

def random_segment(y, length):
    if len(y) <= length:
        pad = length - len(y)
        y = np.pad(y, (0, pad), mode='reflect')
        return y
    start = np.random.randint(0, len(y)-length)
    return y[start:start+length]

def mix_clean_noise(clean, noise, snr_db=None):
    if snr_db is None:
        snr_db = np.random.uniform(-5, 15)
    # normalizing both
    c = clean / (np.std(clean)+1e-9)
    n = noise / (np.std(noise)+1e-9)
    # getting rms of both signals
    rms_c = np.sqrt(np.mean(c**2)+1e-9)
    rms_n = np.sqrt(np.mean(n**2)+1e-9)
    target_rms_n = rms_c / (10**(snr_db/20.0))
    # scaling the noise to get the desired ratio
    n = n * (target_rms_n / (rms_n + 1e-9))
    noisy = c + n
    return norm_audio(noisy), norm_audio(c), norm_audio(n)

# ## Pipeline


def wav_loader_factory(clean_paths, noise_paths):
    # loader function yields one noisy, clean pair (is iterable)
    def load_and_mix(_):
        if clean_paths:
            cp = random.choice(clean_paths)
            c, _sr = read_wav_mono(cp, SR)
        else:
            c = gen_tone(SEGMENT_SEC)
        if noise_paths and np.random.rand() < 0.9:
            npth = random.choice(noise_paths)
            n, _sr = read_wav_mono(npth, SR)
        else:
            n = gen_noise(SEGMENT_SEC + 1.0)
        c_seg = random_segment(c, SEGMENT)
        n_seg = random_segment(n, SEGMENT)
        # noisy will be the model input and clean will be the target
        noisy, clean, noise = mix_clean_noise(c_seg, n_seg)
        return noisy.astype(np.float32), clean.astype(np.float32)
    return load_and_mix
        

# Example workflow:
# For an example step size of 600
# Each call to loader function returns a (noisy, clean) pair, ((32000,), (32000,))
# Gen functions inside tf_dataset calls the loader function 600 * 8 * 2 = 9600 times
# For each epoch a fresh pool is generate 9600 new samples
# The samples are shuffled
# From this pool, batches of 8 are created, so in total 600 batches of 8 samples are created
# Train_ds is an iterable object
# Calling next() on it yields one batch -> ((8, 32000), (8, 32000))
# You can call the next function 600 times

def tf_dataset(clean_paths, noise_paths, batch_size, steps):
    # generator function to call the loader function 2 * required amount times (helps in shuffling)
    def gen(): # stream
        loader = wav_loader_factory(clean_paths, noise_paths)
        for _ in range(steps * batch_size * 2):
            yield loader(None)
    # output dimensions
    output_sig = (tf.TensorSpec(shape=(SEGMENT,), dtype=tf.float32),
                  tf.TensorSpec(shape=(SEGMENT,), dtype=tf.float32))
    # reiterable (generates fresh pool for every epoch)
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)
    ds = ds.shuffle(8192, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    # asynchronously preparing the next batch while the current one is being processed
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = tf_dataset([], [], BATCH_SIZE, STEPS_PER_EPOCH)
val_ds = tf_dataset([], [], BATCH_SIZE, VAL_STEPS)

noisy_b, clean_b = next(iter(train_ds))
print("Batch shapes:", noisy_b.shape, clean_b.shape)

# ## Model


def unet_block(x, filters, name, down=True):
    # downsampling (encoder)
    # convolution operation, decreases the dimensions and captures only the import features
    # batch norm in every block to stabalize training
    if down:
        x = layers.Conv2D(filters, 3, strides=2, padding='same', name=name+'_conv')(x)
        x = layers.BatchNormalization(name=name+'_bn')(x)
        x = layers.Activation('relu', name=name+'_relu')(x)
        return x
    # upsampling (decoder)
    # transpose convolution operation, increases the dimensions, learns to fill in the gaps
    else:
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same', name=name+'_deconv')(x)
        x = layers.BatchNormalization(name=name+'_bn')(x)
        x = layers.Activation('relu', name=name+'_relu')(x)
        return x

class STFTLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.WIN_LENGTH = WIN_LENGTH
        self.HOP = HOP
        self.N_FFT = N_FFT
    def call(self, inputs):
        Xc = tfs.stft(
            inputs, 
            frame_length=self.WIN_LENGTH,
            frame_step=self.HOP,
            fft_length=self.N_FFT,
            window_fn=tfs.hann_window
        )
        return Xc
    def compute_output_spec(self, input_spec):
        T = input_spec.shape[1]
        num_frames = (T - self.WIN_LENGTH) // self.HOP + 1
        num_bins = self.N_FFT // 2 + 1
        output_shape = (input_spec.shape[0], num_frames, num_bins)
        return keras.KerasTensor(output_shape, dtype=tf.complex64)
    

class MagnitudeLayer(layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs)
    def compute_output_spec(self, input_spec):
        return keras.KerasTensor(input_spec.shape, dtype=tf.float32)

class PhaseLayer(layers.Layer):
    def call(self, inputs):
        return tf.math.angle(inputs)
    def compute_output_spec(self, input_spec):
        return keras.KerasTensor(input_spec.shape, dtype=tf.float32)

class ExpandDimsLayer(layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, -1) 

    def compute_output_spec(self, inputs_spec):
        output_shape = inputs_spec.shape + (1,)
        return keras.KerasTensor(output_shape, dtype=inputs_spec.dtype)

class SqueezeLayer(layers.Layer):
    def call(self, inputs):
        return tf.squeeze(inputs, axis=-1)
    def compute_output_spec(self, input_spec):
        output_shape = input_spec.shape[:-1]
        return keras.KerasTensor(output_shape, dtype=input_spec.dtype)

class PhaseToRealLayer(layers.Layer):
    def call(self, inputs):
        return tf.cos(inputs)

    def compute_output_spec(self, inputs_spec):
        return keras.KerasTensor(inputs_spec.shape, dtype=tf.float32)

class PhaseToImagLayer(layers.Layer):
    def call(self, inputs):
        return tf.sin(inputs)

    def compute_output_spec(self, inputs_spec):
        return keras.KerasTensor(inputs_spec.shape, dtype=tf.float32)

@tf.autograph.experimental.do_not_convert
def inverse_stft_wrapper(Xc):
    return tfs.inverse_stft(
        Xc, 
        frame_length=WIN_LENGTH,
        frame_step=HOP,
        window_fn=tfs.hann_window,
        output_length=SEGMENT
    )

class ISTFTLayer(layers.Layer):
    """Wraps tfs.inverse_stft to reconstruct the time-domain audio."""
    def __init__(self, output_length, **kwargs):
        super().__init__(**kwargs)
        self.WIN_LENGTH = WIN_LENGTH
        self.HOP = HOP
        self.output_length = output_length
        self.window = tfs.hann_window(WIN_LENGTH)

    def call(self, inputs):
        enh_audio = tfs.inverse_stft(
            inputs,
            frame_length=self.WIN_LENGTH,
            frame_step=self.HOP,
            window_fn=self.window,  
            output_length=self.output_length
        )
        return enh_audio

    def compute_output_spec(self, inputs_spec):
        batch_size = inputs_spec.shape[0]
        output_shape = (batch_size, self.output_length)
        return keras.KerasTensor(output_shape, dtype=tf.float32)

# Convolutions capture local structures like harmocis and noise texture
# learn how energy is distributed across different frequencies

def build_unet(n_mels=None):
    inp = keras.Input(shape=(SEGMENT,), name='audio_in')
    # size of an individual sample from the batch, batch handled automatically during training

    # fourier transform
    Xc = STFTLayer(name='stft_complex')(inp)
    
    # taking magnitude and phase angle from complex number output
    mag = MagnitudeLayer(name='magnitude')(Xc) # spectral quality
    phase = PhaseLayer(name='phase')(Xc) # timing, position within wave

    # adding additional dimension to make it compatible with conv layer
    M = ExpandDimsLayer(name='expand_dims_channel')(mag)

    # Encoder block (passing stft output through 5 conv layers)
    # Reduces resolution through downsampling (lets the neurons see the bigger picture)
    e1 = layers.Conv2D(32, 3, padding='same', activation='relu')(M)
    d1 = unet_block(e1, 64, 'down1')
    d2 = unet_block(d1, 128, 'down2')
    d3 = unet_block(d2, 256, 'down3')
    bott = layers.Conv2D(512, 3, padding='same', activation='relu')(d3)

    # Decoder block
    # each output of upsampling is compared with original version through skip connection
    u3 = unet_block(bott, 256, 'up3', down=False)
    u3 = layers.Cropping2D(cropping=((0, 1), (0, 1)), name='crop_u3_to_d2')(u3)
    u3 = layers.Concatenate()([u3, d2])
    u2 = unet_block(u3, 128, 'up2', down=False)
    u2 = layers.Cropping2D(cropping=((0, 1), (0, 1)), name='crop_u2_to_d1')(u2)
    u2 = layers.Concatenate()([u2, d1])
    u1 = unet_block(u2, 64, 'up1', down=False)
    u1 = layers.Cropping2D(cropping=((0, 0), (0, 1)), name='crop_u1_to_e1')(u1)
    u1 = layers.Concatenate()([u1, e1])

    # soft mask prediction for each time frequency bin of input
    # soft mask values (0-1), closer to 1 means the part is clear speech and should be preserved
    out_mask = layers.Conv2D(1, 1, activation='sigmoid', name='mask')(u1)
    out_mask = SqueezeLayer(name='squeeze_mask')(out_mask)

    # enchanced magnitude after applying the soft mask
    # after applying the mask we get the predicted clean output which is compared with target output
    enh_mag = layers.Multiply(name='apply_mask')([out_mask, mag]) 

    # Reconstruct Real and Imaginary components (using the new Phase layers)
    real = layers.Multiply(name='real_part')([enh_mag, PhaseToRealLayer()(phase)])
    imag = layers.Multiply(name='imag_part')([enh_mag, PhaseToImagLayer()(phase)])

    # Combine to complex STFT (Requires ComplexReconstructionLayer)
    enh_complex = layers.Lambda(
        lambda inputs: tf.complex(inputs[0], inputs[1]),
        name='complex_reconstruct'
    )([real, imag])
    
    # Inverse STFT (Requires ISTFTLayer)
    enh_audio = layers.Lambda(
        lambda Xc: tfs.inverse_stft(
                Xc, 
                frame_length=WIN_LENGTH, 
                frame_step=HOP, 
                window_fn=tf.signal.hann_window,
            ),
            name='istft_final'
    )(enh_complex)

    return keras.Model(inp, outputs=[enh_audio, out_mask, mag], name='UNet_Denoiser')

model = build_unet()
model.summary()

# ## Loss Functions


# Average absolute difference between actual and predicted
def l1_mag_loss(true_audio, pred_audio):
    Y = tfs.stft(true_audio, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
    P = tfs.stft(pred_audio, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
    return tf.reduce_mean(tf.abs(tf.abs(Y)-tf.abs(P)))

# difference relative to clean spectrogram, focus on matching overall energy dist
# penalizing relative error rather than absolute
def spectral_convergance(true_audio, pred_audio):
    Y = tfs.stft(true_audio, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
    P = tfs.stft(pred_audio, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
    num = tf.norm(tf.abs(Y)-tf.abs(P), ord='fro')
    den = tf.norm(tf.abs(Y), ord='fro') + eps()
    return num / den

# Multi Resolution STFT loss
# compares absolute difference for different window and hop sizes

def mrstft_loss(true_audio, pred_audio):
    cfgs = [
        (1024, 526), (512, 128), (2048, 512)
    ]
    loss = 0.0
    for nfft, hop in cfgs:
        Y = tfs.stft(true_audio, hop, nfft, window_fn=tf.signal.hann_window)
        P = tfs.stft(pred_audio, hop, nfft, window_fn=tf.signal.hann_window)
        loss += tf.reducde_mean(tf.abs(tf.abs(Y)-tf.abs(P)))
    return loss / len(cfgs)

# point -> as long as the waveform shape matches, amplitude mismatches are ignored
# penalize unwanted signal content that cannot be explained by rescaling clean signal
def si_sdr(true_audio, pred_audio):
    x = true_audio
    s = pred_audio
    x_zm = x - tf.reduce_mean(x, axis=-1, keepdims=True)
    s_zm = s - tf.reduce_mean(x, axis=-1, keepdims=True)
    proj = tf.reduce_sum(s_zm * x_zm, axis=-1) / (tf.reduce_sum(x_zm**2, axis=-1, keepdims=True) + eps()) * x_zm
    e = s_zm - proj
    si_sdr_val = 10 * tf.math.log((tf.reduce_sum(proj**2, axis=-1)+eps()) / (tf.reduce_sum(e**2, axis=-1)+eps()))
    return si_sdr_val

# Scale invariant signal to distortion ratio loss -> Quality of reconstructed waveform in time domain
def si_sdr_loss(true_audio, pred_audio):
    return -tf.reduce_mean(si_sdr(true_audio, pred_audio))

# combined loss
def total_loss(true_audio, pred_audio):
    return (0.5 * l1_mag_loss(true_audio, pred_audio) +
            0.2 * spectral_convergence(true_audio, pred_audio) +
            0.2 * mrstft_loss(true_audio, pred_audio) +
            0.1 * si_sdr_loss(true_audio, pred_audio))

# ## Optimizer


# Two phase learning rate scheduling 
# First gradually increase the learning rate (warmup) and then smoothly decrease it throughout training
# The decrease follows a smooth cosine curve

class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base_lr = base_lr # starting learning rate
        self.warmup = warmup_steps # increase upto warmup steps
        self.total = total_steps

    def __call__(self, step): # gives learning rate at given step
        step = tf.cast(step, tf.float32)
        warm = tf.cast(self.warmup, tf.float32)
        total = tf.cast(self.total, tf.float32)
        lr = tf.where(
            step < warm,
            self.base_lr * (step / tf.maximum(1.0, warm)),
            0.5 * self.base_lr * (1 + tf.cos(np.pi * (step - warm) / tf.maximum(1.0, (total-warm))))
        )
        return lr


TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
lr_schedule = WarmupCosine(LEARNING_RATE, WARMUP_STEPS, TOTAL_STEPS)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)

# ## Exponential Moving Average


# Exponential Moving Average for better generalization and stable performance
# Weighted average of past weights is stored with recent weights having higher weight
# Hence, to get the new weight value not just the latest weights are used

class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay # how weight to be given to the older stored average weights
        self.shadow = [tf.identity(w) for w in model.weights]

    def update(self, model):
        # new_weight = old_weight * decay + current_model_weight * (1 - decay)
        for i, w in enumerate(model.weights):
            self.shadow[i].assign(self.decay * self.shadow[i] + (1.0 - self.decay) * w)
        # change is more smooth without jumping around

    def apply_to(self, model):
        self.backup = [tf.identity(w) for w in model.weights] # temporary backup
        for w, s in zip(model.weights, self.shadow):
            w.assign(t)

    def restore(self, model):
        # reverts model back original weights before apply_to was called
        # allowing toggling weight type during training and evaluation
        for w, b in zip(model.weights, self.backup):
            w.assign(b)
        self.backup = None

ema = EMA(model, EMA_DECAY)

# # Custom Training Loop with Custom loss and EMA


from tensorflow import keras

train_loss_metric = keras.metrics.Mean()
val_loss_metric = keras.metrics.Mean()
train_si_sdr_metric = keras.metrics.Mean()
val_si_sdr_metric = keras.metrics.Mean()

@tf.function 
def train_step(noisy, clean):
    with tf.GradientTape() as tape:
        enhanced_audio, mask, mag = model(noisy, training=True)
        loss = total_loss(clean, enhanced_audio)
        if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
            loss = tf.cast(loss, tf.float32)
    # calculating the gradient of the loss wrt to all the weights 
    grads = tape.gradient(loss, model.trainable_variables)
    # applying the gradients 
    opt.apply_gradients(zip(grads, model.trainable_variables))
    # updating the ema weights 
    ema.update(model)
    # updating the metric states
    train_loss_metric.update_state(loss)
    train_si_sdr_metric.update_state(si_sdr(clean, enhanced_audio))

@tf.function 
def val_step(noisy, clean):
    enhanced_audio, mask, mag = model(noisy, training=False)
    loss = total_loss(clean, enhanced_audio)
    val_loss_metric.update_state(loss)
    val_si_sdr_metric.update_state(si_sdr(clean, enhanced_audio))

# # Training Loop


history = {'loss': [], 'val_loss': [], 'si_sdr': [], 'val_si_sdr': []}
global_step = 0

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"- Name: {gpu.name}, Type: {gpu.device_type}")
else:
    print("WARNING: No GPUs detected by TensorFlow. Check your notebook settings.")

for epoch in range(1, EPOCHS+1):
    print(f'\nEpoch {epoch}/{EPOCHS}')
    # resetting all the metric states to zero at the start of each epoch
    train_loss_metric = keras.metrics.Mean()
    val_loss_metric = keras.metrics.Mean()
    train_si_sdr_metric = keras.metrics.Mean()
    val_si_sdr_metric = keras.metrics.Mean()

    for step, (noisy, clean) in enumerate(train_ds.take(STEPS_PER_EPOCH), start=1):
        # train step
        train_step(noisy, clean)
        global_step += 1
        if step % 100 == 0:
            print(f"  step {step}/{STEPS_PER_EPOCH}  lr={opt.lr(global_step).numpy():.6f}  "
                  f"loss={train_loss_metric.result().numpy():.4f}  "
                  f"SI-SDR={train_si_sdr_metric.result().numpy():.2f}dB")
    # temporarily apply ema weights during the validation phase
    ema.apply_to(model)
    for step, (noisy, clean) in enumerate(val_ds.take(VAL_STEPS), start=1):
        val_step(noisy, clean)
    # restore the old weights to resume training 
    ema.restore(model)

    tr_loss = float(train_loss_metric.result().numpy())
    va_loss = float(val_loss_metric.result().numpy())
    tr_sdr = float(train_si_sdr_metric.result().numpy())
    va_sdr = float(val_si_sdr_metric.result().numpy())

    history['loss'].append(tr_loss)
    history['val_loss'].append(va_loss)
    history['si_sdr'].append(tr_sdr)
    history['val_si_sdr'].append(va_sdr)

    # save checkpoint 
    ema.apply_to(model)
    model.save_weights(os.path.join(CHECKPOINT_DIR, f'epoch{epoch:0.2d}.weights.h5'))
    ema.restore(model)

    print(f"Epoch {epoch} done. train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
          f"train_SI-SDR={tr_sdr:.2f}dB val_SI-SDR={va_sdr:.2f}dB")

plt.figure(figsize=(10,3))
plt.plot(history["loss"], label="train loss")
plt.plot(history["val_loss"], label="val loss")
plt.title("Loss Curves")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,3))
plt.plot(history["si_sdr"], label="train SI-SDR (dB)")
plt.plot(history["val_si_sdr"], label="val SI-SDR (dB)")
plt.title("SI-SDR Curves")
plt.xlabel("Epoch"); plt.ylabel("dB"); plt.legend(); plt.tight_layout(); plt.show()

# # Qualitative Check


ema.apply_to(model)

noisy_b, clean_b = next(iter(val_ds))
enh_b, mask_b, mag_b = model(noisy_b, training=False)

idx = 0
noisy = noisy_b[idx].numpy()
clean = clean_b[idx].numpy()
enh = enh_b[idx].numpy()

print("Playing audio (Noisy -> Enhanced -> Clean)")
display(Audio(noisy, rate=SR))
display(Audio(enh, rate=SR))
display(Audio(clean, rate=SR))

N = tfs.stft(noisy, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
C = tfs.stft(clean, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
E = tfs.stft(enh,   WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
plot_spectrograms(np.abs(N).numpy(), np.abs(C).numpy(), np.abs(E).numpy(),
                  sr=SR, title="Noisy / Clean / Enhanced (Mag dB)")
plot_mask(mask_b[idx].numpy(), title="Predicted Soft Mask")
plot_waveforms(noisy, clean, enh, sr=SR, title="Waveforms")

ema.restore(model)

ema.apply_to(model)

class InferenceWrapper(keras.Model):
    def __init__(self, base):
        super().__init__()
        self.base = base
    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
    def denoise(self, audio):
        audio = tf.expand_dims(audio, 0)
        enh, _mask, _mag = self.base(audio, training=False)
        return tf.squeeze(enh, 0)

infer_model = InferenceWrapper(model)
tf.save_model.save(infer_model, EXPORT_DIR)
print("Exported to: ", EXPORT_DIR)

def denoise_file(wav_path, out_path=None, show=True):
    y, _ = read_wav_mono(wav_path, SR)
    # pad/trim to multiple of hop via center-padding for a nicer output (optional)
    T = len(y)
    pad = ( (math.ceil(T / HOP) * HOP) - T )
    y_pad = np.pad(y, (0, pad), mode='reflect').astype(np.float32)

    ema.apply_to(model)
    enh, mask, _ = model(tf.convert_to_tensor(y_pad[None, ...]), training=False)
    ema.restore(model)

    enh = enh.numpy()[0][:T]

    if out_path:
        write_wav(out_path, enh, SR)

    if show:
        print("Playing (Noisy -> Enhanced)")
        display(Audio(y, rate=SR))
        display(Audio(enh, rate=SR))
        N = tfs.stft(y, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
        E = tfs.stft(enh, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
        plot_spectrograms(np.abs(N).numpy(), None, None, sr=SR, title="Noisy Spectrogram (dB)")
        plot_spectrograms(np.abs(E).numpy(), None, None, sr=SR, title="Enhanced Spectrogram (dB)")
        plot_waveforms(y, None, enh, sr=SR, title="Inference Waveforms")
    return enh


print("\nReady! Use: denoise_file('/kaggle/input/your.wav', out_path='/kaggle/working/clean.wav')")

demo_clean = gen_tone(2.0)
demo_noise = gen_noise(2.0)
demo_noisy, demo_clean, _ = mix_clean_noise(demo_clean, demo_noise, snr_db=0.0)
_ = denoise_file(wav_path=None if True else "", out_path=None)  # no-op to show usage
print("Demo playback:")
display(Audio(demo_noisy, rate=SR))
ema.apply_to(model)
demo_enh, _, _ = model(tf.convert_to_tensor(demo_noisy[None, ...]), training=False)
ema.restore(model)
demo_enh = demo_enh.numpy()[0]
display(Audio(demo_enh, rate=SR))
plot_spectrograms(np.abs(tfs.stft(demo_noisy, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)).numpy(),
                  None, None, title="Demo Noisy Spec (dB)")
plot_spectrograms(np.abs(tfs.stft(demo_enh, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)).numpy(),
                  None, None, title="Demo Enhanced Spec (dB)")