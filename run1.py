#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified Denoising U-Net: Stable Loss and Layers
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
SEGMENT = int(SR * SEGMENT_SEC) # 32000
N_FFT = 1024 
HOP = 256
WIN_LENGTH = 1024 
N_MELS = 128
PAD_MODE = 'REFLECT' # Not used, but kept for context

BATCH_SIZE = 8
EPOCHS = 15
STEPS_PER_EPOCH = 600
VAL_STEPS = 80
LEARNING_RATE = 3e-4 
WARMUP_STEPS = 500
EMA_DECAY = 0.999
CHECKPOINT_DIR = '/kaggle/working/denoiser_ckpt'
EXPORT_DIR = '/kaggle/working/denoiser_export'
EPSILON = 1e-8 # Global epsilon for stability

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
    wavfile.write(path, sr, (y*32767.0).astype(np.int16))

# ## Signal Transforms (Simplified)

# Reusable STFT/ISTFT functions
def stft(sig):
    return tfs.stft(
        sig, 
        frame_length=WIN_LENGTH,
        frame_step=HOP, 
        fft_length=N_FFT, 
        window_fn=tf.signal.hann_window
    )

def istft(stft_c, length=SEGMENT):
    return tfs.inverse_stft(
        stft_c,
        frame_length=WIN_LENGTH,
        frame_step=HOP,
        window_fn=tf.signal.hann_window,
    )

def eps():
    return EPSILON

# ## Visualization functions (Retained)

def plot_waveforms(noisy, clean=None, enhanced=None, sr=SR, title='Waveforms'):
    plt.figure(figsize=(12, 3))
    t = np.arange(len(noisy))/sr
    plt.plot(t, noisy, label='Noisy', linewidth=0.8)
    if clean is not None:
        plt.plot(t[:len(clean)], clean, label='Clean', alpha=0.7, linewidth=0.8)
    if enhanced is not None:
        # Note: Original code had typo 'enchanced' and 'linewidht'
        plt.plot(t[:len(enhanced)], enhanced, label='Enhanced', alpha=0.9, linewidth=0.8)
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

# ## Synthetic Noise Generator (Retained)

def gen_tone(duration, sr=SR):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    f0 = np.random.uniform(100, 1000)
    y = np.sin(2*np.pi*f0*t)
    if np.random.rand() < 0.5:
        f1 = np.random.uniform(200, 2000)
        chirp = np.sin(2*np.pi*(f0 + (f1-f0)*t/duration)*t)
        y = 0.6*y + 0.4*chirp
    env = 0.5*(1-np.cos(2*np.pi*np.minimum(1.0, t/duration)))
    return norm_audio(y * env)

def gen_noise(duration, sr=SR):
    n = int(sr*duration)
    white = np.random.randn(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, 1/sr)
    pink_spec = (np.random.randn(len(freqs))+1j*np.random.randn(len(freqs)))/np.maximum(freqs, 1.0)
    pink = np.fft.irfft(pink_spec, n=n).astype(np.float32)
    babble = np.zeros(n, dtype=np.float32)
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
    c = clean / (np.std(clean)+1e-9)
    n = noise / (np.std(noise)+1e-9)
    rms_c = np.sqrt(np.mean(c**2)+1e-9)
    rms_n = np.sqrt(np.mean(n**2)+1e-9)
    target_rms_n = rms_c / (10**(snr_db/20.0))
    n = n * (target_rms_n / (rms_n + 1e-9))
    noisy = c + n
    return norm_audio(noisy), norm_audio(c), norm_audio(n)

# ## Pipeline (Retained)

def wav_loader_factory(clean_paths, noise_paths):
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
        noisy, clean, noise = mix_clean_noise(c_seg, n_seg)
        return noisy.astype(np.float32), clean.astype(np.float32)
    return load_and_mix
        
def tf_dataset(clean_paths, noise_paths, batch_size, steps):
    def gen():
        loader = wav_loader_factory(clean_paths, noise_paths)
        for _ in range(steps * batch_size * 2):
            yield loader(None)
    output_sig = (tf.TensorSpec(shape=(SEGMENT,), dtype=tf.float32),
                  tf.TensorSpec(shape=(SEGMENT,), dtype=tf.float32))
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)
    ds = ds.shuffle(8192, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = tf_dataset([], [], BATCH_SIZE, STEPS_PER_EPOCH)
val_ds = tf_dataset([], [], BATCH_SIZE, VAL_STEPS)

# ## Model (Simplified U-Net)

def unet_block(x, filters, name, down=True):
    if down:
        x = layers.Conv2D(filters, 3, strides=2, padding='same', name=name+'_conv')(x)
        x = layers.BatchNormalization(name=name+'_bn')(x)
        x = layers.Activation('relu', name=name+'_relu')(x)
        return x
    else:
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same', name=name+'_deconv')(x)
        x = layers.BatchNormalization(name=name+'_bn')(x)
        x = layers.Activation('relu', name=name+'_relu')(x)
        return x

def build_unet():
    inp = keras.Input(shape=(SEGMENT,), name='audio_in')

    # STFT & Magnitude/Phase Extraction (using Lambda for simplicity)
    Xc = layers.Lambda(stft, name='stft_complex')(inp)
    mag = layers.Lambda(tf.abs, name='magnitude')(Xc)
    phase = layers.Lambda(tf.math.angle, name='phase')(Xc)

    # Add channel dimension: (B, T, F) -> (B, T, F, 1)
    M = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), name='expand_dims_channel')(mag)

    # Encoder block
    e1 = layers.Conv2D(32, 3, padding='same', activation='relu')(M)
    d1 = unet_block(e1, 64, 'down1')
    d2 = unet_block(d1, 128, 'down2')
    d3 = unet_block(d2, 256, 'down3')
    bott = layers.Conv2D(512, 3, padding='same', activation='relu')(d3)

    # Decoder block 
    u3 = unet_block(bott, 256, 'up3', down=False)
    u3 = layers.Cropping2D(cropping=((0, 1), (0, 1)), name='crop_u3_to_d2')(u3) 
    u3 = layers.Concatenate()([u3, d2])
    
    u2 = unet_block(u3, 128, 'up2', down=False)
    u2 = layers.Cropping2D(cropping=((0, 1), (0, 1)), name='crop_u2_to_d1')(u2)
    u2 = layers.Concatenate()([u2, d1])
    
    u1 = unet_block(u2, 64, 'up1', down=False)
    u1 = layers.Cropping2D(cropping=((0, 0), (0, 1)), name='crop_u1_to_e1')(u1)
    u1 = layers.Concatenate()([u1, e1])

    # Mask Prediction: (B, T, F, 1) -> (B, T, F)
    out_mask = layers.Conv2D(1, 1, activation='sigmoid', name='mask')(u1)
    out_mask = layers.Lambda(tf.squeeze, arguments={'axis': -1}, name='squeeze_mask')(out_mask)

    # Enhanced Magnitude
    enh_mag = layers.Multiply(name='apply_mask')([out_mask, mag]) 

    # Reconstruct Complex STFT (Using noisy phase)
    enh_complex = layers.Lambda(
        lambda x: tf.complex(x[0] * tf.cos(x[1]), x[0] * tf.sin(x[1])),
        output_shape=(122, 513),
        name='complex-reconstruct'
    )([enh_mag, phase])
    
    # Inverse STFT
    enh_audio = layers.Lambda(
        istft, 
        output_shape=(SEGMENT,),
        name='istft_final'
    )(enh_complex)

    return keras.Model(inp, outputs=[enh_audio, out_mask, mag], name='UNet_Denoiser')

model = build_unet()
# model.summary()

# ## Simplified and Stable Loss Functions

# 1. L1 Magnitude Loss (Frequency Domain MAE)
def l1_mag_loss(true_audio, pred_audio):
    Y = stft(true_audio)
    P = stft(pred_audio)
    return tf.reduce_mean(tf.abs(tf.abs(Y) - tf.abs(P)))

# 2. L1 Time Loss (Time Domain MAE)
def l1_time_loss(true_audio, pred_audio):
    return tf.reduce_mean(tf.abs(true_audio - pred_audio))

# 3. Combined Loss (SIMPLE and STABLE)
def total_loss(true_audio, pred_audio):
    # Balanced weights for stability. Can be tuned later.
    return (0.6 * l1_time_loss(true_audio, pred_audio) +
            0.4 * l1_mag_loss(true_audio, pred_audio))


# ## SI-SDR Metric (Used for tracking, not training loss)

def si_sdr(true_audio, pred_audio):
    x = true_audio
    s = pred_audio
    
    x_zm = x - tf.reduce_mean(x, axis=-1, keepdims=True)
    s_zm = s - tf.reduce_mean(x, axis=-1, keepdims=True) 

    # Projection (alpha * x_zm)
    alpha_num = tf.reduce_sum(s_zm * x_zm, axis=-1, keepdims=True)
    alpha_den = tf.reduce_sum(x_zm**2, axis=-1, keepdims=True) + eps()
    alpha = alpha_num / alpha_den
    proj = alpha * x_zm
    
    # Error component
    e = s_zm - proj
    
    # Power and Ratio (with stability epsilon)
    signal_power = tf.reduce_sum(proj**2, axis=-1)
    error_power = tf.reduce_sum(e**2, axis=-1)
    ratio = (signal_power + eps()) / (error_power + eps())
    
    # SI-SDR in dB: 10 * log10(ratio)
    si_sdr_val = 10.0 * tf.math.log(ratio) / tf.math.log(10.0)
    
    return si_sdr_val


# ## Optimizer

class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base_lr = base_lr
        self.warmup = warmup_steps
        self.total = total_steps

    def __call__(self, step):
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

# ## Exponential Moving Average (EMA) - Robust Implementation

class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay 
        # Only track weights that are Variables (i.e., have the .assign method)
        self.trainable_vars = [w for w in model.weights if hasattr(w, 'assign')] 
        self.shadow = [tf.Variable(tf.identity(w)) for w in self.trainable_vars]

    def update(self, model):
        # The update is performed on the shadow (tf.Variable)
        for w, s in zip(self.trainable_vars, self.shadow):
            s.assign(self.decay * s + (1.0 - self.decay) * w)

    def apply_to(self, model):
        # Ensure backup weights are tf.Variable
        self.backup = [tf.Variable(tf.identity(w)) for w in self.trainable_vars]
        # Assign shadow weights to the model
        for w, s in zip(self.trainable_vars, self.shadow):
            w.assign(s)

    def restore(self, model):
        # Assign backup weights back to the model
        for w, b in zip(self.trainable_vars, self.backup):
            w.assign(b)
        self.backup = None

ema = EMA(model, EMA_DECAY)

# ## Custom Training Loop with Gradient Safety

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
            
    grads = tape.gradient(loss, model.trainable_variables)
    
    # 🌟 Gradient Safety Measures 🌟
    # 1. Global Norm Clipping: Prevents exploding gradients.
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=5.0) 

    # 2. NaN/Inf Filtering: Prevents optimizer crash from bad gradients.
    grads_and_vars = zip(grads, model.trainable_variables)
    filtered_grads_and_vars = []
    for grad, var in grads_and_vars:
        if grad is not None:
            # Replace any NaN/Inf gradients with zeros
            grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
            grad = tf.where(tf.math.is_inf(grad), tf.zeros_like(grad), grad)
            filtered_grads_and_vars.append((grad, var))

    # Apply gradients 
    opt.apply_gradients(filtered_grads_and_vars)
    
    ema.update(model)
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
    
    train_loss_metric.reset_state()
    val_loss_metric.reset_state()
    train_si_sdr_metric.reset_state()
    val_si_sdr_metric.reset_state()

    for step, (noisy, clean) in enumerate(train_ds.take(STEPS_PER_EPOCH), start=1):
        # train step
        train_step(noisy, clean)
        global_step += 1
        
        # Access learning rate via the schedule object (safest method)
        current_lr = lr_schedule(tf.constant(global_step, dtype=tf.int64)).numpy()
        
        if step % 100 == 0:
            print(f"  step {step}/{STEPS_PER_EPOCH}  lr={current_lr:.6f}  "
                  f"loss={train_loss_metric.result().numpy():.4f}  "
                  f"SI-SDR={train_si_sdr_metric.result().numpy():.2f}dB")
                  
    # validation phase
    ema.apply_to(model)
    for step, (noisy, clean) in enumerate(val_ds.take(VAL_STEPS), start=1):
        val_step(noisy, clean)
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
    model.save_weights(os.path.join(CHECKPOINT_DIR, f'epoch{epoch:02d}.weights.h5'))
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

N = stft(noisy)
C = stft(clean)
E = stft(enh)
plot_spectrograms(np.abs(N).numpy(), np.abs(C).numpy(), np.abs(E).numpy(),
                  sr=SR, title="Noisy / Clean / Enhanced (Mag dB)")
plot_mask(mask_b[idx].numpy(), title="Predicted Soft Mask")
plot_waveforms(noisy, clean, enh, sr=SR, title="Waveforms")

ema.restore(model)

# # Inference Export

ema.apply_to(model)

class InferenceWrapper(keras.Model):
    def __init__(self, base):
        super().__init__()
        self.base = base
    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
    def denoise(self, audio):
        audio = tf.expand_dims(audio, 0)
        # Note: Model returns 3 outputs, but inference only needs the first
        enh, _mask, _mag = self.base(audio, training=False) 
        return tf.squeeze(enh, 0)

infer_model = InferenceWrapper(model)
# Note: Saving the model with tf.save_model.save is the recommended way to save tf.function wrappers
tf.saved_model.save(infer_model, EXPORT_DIR) 
print("Exported to: ", EXPORT_DIR)

def denoise_file(wav_path, out_path=None, show=True):
    # This function expects the audio length to be a multiple of SEGMENT (32000) for a clean run.
    y, _ = read_wav_mono(wav_path, SR)
    T = len(y)
    
    # Pad/trim to ensure segment length is handled
    pad = ( (math.ceil(T / SEGMENT) * SEGMENT) - T )
    # Pad to the nearest SEGMENT boundary, not HOP, for the inference wrapper style 
    y_pad = np.pad(y, (0, pad), mode='reflect').astype(np.float32)

    ema.apply_to(model)
    # The current model is built for SEGMENT length. This will only work correctly 
    # if the input is close to SEGMENT length or if the model handles variable input shape (it doesn't).
    # Since the infer_model is saved, we use the saved model's method.
    if T != SEGMENT:
        print("WARNING: Inference function is simplistic. Only inputs of length 32000 are guaranteed to work.")

    # Using the denoise wrapper method (requires loading the saved model or running the saved wrapper function)
    # For in-notebook testing, we'll simplify by running the base model with the segment length check disabled (risky):
    enh, _, _ = model(tf.convert_to_tensor(y_pad[None, ...]), training=False)
    ema.restore(model)

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


ema.restore(model) # Restore to original state after export

print("\nReady! Use: denoise_file('/kaggle/input/your.wav', out_path='/kaggle/working/clean.wav', show=True)")

# Demo playback:
demo_clean = gen_tone(2.0)
demo_noise = gen_noise(2.0)
demo_noisy, demo_clean, _ = mix_clean_noise(demo_clean, demo_noise, snr_db=0.0)
print("Demo playback:")
display(Audio(demo_noisy, rate=SR))
ema.apply_to(model)
demo_enh, _, _ = model(tf.convert_to_tensor(demo_noisy[None, ...]), training=False)
ema.restore(model)
demo_enh = demo_enh.numpy()[0]
display(Audio(demo_enh, rate=SR))
N_demo_spec = stft(demo_noisy)
E_demo_spec = stft(demo_enh)
plot_spectrograms(np.abs(N_demo_spec).numpy(), None, np.abs(E_demo_spec).numpy(), title="Demo Noisy / Enhanced Spec (dB)")
