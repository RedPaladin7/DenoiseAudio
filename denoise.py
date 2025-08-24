# ================================================================
# Kaggle Notebook: Neural Audio Denoiser (TensorFlow / Keras)
# - U-Net on log-magnitude STFT predicting soft mask
# - Rich losses: L1 + Spectral Convergence + MR-STFT + SI-SDR
# - Mixed precision, EMA, cosine LR schedule with warmup
# - Visualizations & one-call inference utility
# ================================================================

import os, sys, math, random, glob, shutil, time, functools, itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.signal as tfs

from scipy.io import wavfile
from IPython.display import Audio, display

# --------------------------
# Environment & Reproducibility
# --------------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Detect/enable mixed precision (safe on most Kaggle GPUs)
try:
    if tf.config.list_physical_devices('GPU'):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled.")
    else:
        print("No GPU detected; training will use CPU.")
except Exception as e:
    print("Mixed precision not enabled:", e)

# --------------------------
# Config
# --------------------------
SR             = 16000       # Sample rate
SEGMENT_SEC    = 2.0         # Training segment length (seconds)
SEGMENT       = int(SR * SEGMENT_SEC)
N_FFT          = 1024
HOP            = 256
WIN_LENGTH     = 1024
N_MELS         = 128
PAD_MODE       = "REFLECT"

# Training
BATCH_SIZE     = 8
EPOCHS         = 15
STEPS_PER_EPOCH= 600         # increase if you have more data
VAL_STEPS      = 80
LEARNING_RATE  = 3e-4
WARMUP_STEPS   = 500
EMA_DECAY      = 0.999
CHECKPOINT_DIR = "/kaggle/working/denoiser_ckpt"
EXPORT_DIR     = "/kaggle/working/denoiser_export"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# --------------------------
# Utility: audio I/O
# --------------------------
def norm_audio(x):
    x = np.asarray(x, dtype=np.float32)
    mx = np.max(np.abs(x)) + 1e-9
    return x / mx

def read_wav_mono(path, target_sr=SR):
    sr, y = wavfile.read(path)
    y = y.astype(np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    # Resample if needed (simple tf-based)
    if sr != target_sr:
        y = tf.audio.resample(y, sr, target_sr).numpy()
    return norm_audio(y), target_sr

def write_wav(path, y, sr=SR):
    y = np.asarray(y, dtype=np.float32)
    y = (y / (np.max(np.abs(y)) + 1e-9) * 0.99)
    wavfile.write(path, sr, (y * 32767.0).astype(np.int16))

# --------------------------
# Signal Transforms
# --------------------------
def stft(sig):
    return tfs.stft(
        sig,
        frame_length=WIN_LENGTH,
        frame_step=HOP,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window
    )

def istft(stft_c, length):
    return tfs.inverse_stft(
        stft_c,
        frame_length=WIN_LENGTH,
        frame_step=HOP,
        window_fn=tf.signal.hann_window,
        # NOTE: length helps trim/pad
        output_length=length
    )

def complex_mag(stft_c):
    return tf.abs(stft_c)

def eps():
    return 1e-8

# Mel filter for visualizations & optional mel-loss (we'll keep it for plots)
MEL_FILTER = tfs.linear_to_mel_weight_matrix(
    num_mel_bins=N_MELS,
    num_spectrogram_bins=N_FFT//2 + 1,
    sample_rate=SR,
    lower_edge_hertz=0.0,
    upper_edge_hertz=SR/2
)

# --------------------------
# Visualizations
# --------------------------
def plot_waveforms(noisy, clean=None, enhanced=None, sr=SR, title="Waveforms"):
    plt.figure(figsize=(12, 3))
    t = np.arange(len(noisy))/sr
    plt.plot(t, noisy, label='Noisy', linewidth=0.8)
    if clean is not None:
        plt.plot(t[:len(clean)], clean, label='Clean', alpha=0.7, linewidth=0.8)
    if enhanced is not None:
        plt.plot(t[:len(enhanced)], enhanced, label='Enhanced', alpha=0.9, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Time [s]")
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

# --------------------------
# Data: where to read clean/noise WAVs (optional)
# --------------------------
CLEAN_DIRS = glob.glob("/kaggle/input/*/clean")  # e.g., /kaggle/input/your-ds/clean/*.wav
NOISE_DIRS = glob.glob("/kaggle/input/*/noise")  # e.g., /kaggle/input/your-ds/noise/*.wav

CLEAN_WAVS = sorted(list(itertools.chain.from_iterable(
    [glob.glob(os.path.join(d, "**/*.wav"), recursive=True) for d in CLEAN_DIRS]
)))
NOISE_WAVS = sorted(list(itertools.chain.from_iterable(
    [glob.glob(os.path.join(d, "**/*.wav"), recursive=True) for d in NOISE_DIRS]
)))

print(f"Found {len(CLEAN_WAVS)} clean wavs and {len(NOISE_WAVS)} noise wavs.")

# --------------------------
# Synthetic Data Generators (if no dataset available)
# --------------------------
def gen_tone(duration, sr=SR):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    f0 = np.random.uniform(100, 1000)
    y = np.sin(2*np.pi*f0*t)
    # random amplitude modulation & chirp blend
    if np.random.rand() < 0.5:
        f1 = np.random.uniform(200, 2000)
        chirp = np.sin(2*np.pi*(f0 + (f1-f0)*t/duration)*t)
        y = 0.6*y + 0.4*chirp
    # random envelope
    env = 0.5*(1 - np.cos(2*np.pi*np.minimum(1.0, t/duration)))
    return norm_audio(y * env)

def gen_noise(duration, sr=SR):
    n = int(sr*duration)
    # White
    white = np.random.randn(n).astype(np.float32)
    # Pink (1/f)
    freqs = np.fft.rfftfreq(n, 1/sr)
    pink_spec = (np.random.randn(len(freqs)) + 1j*np.random.randn(len(freqs))) / np.maximum(freqs, 1.0)
    pink = np.fft.irfft(pink_spec, n=n).astype(np.float32)
    # Babble-ish (sum of random voices simulated by multiple tones)
    babble = np.zeros(n, dtype=np.float32)
    for _ in range(np.random.randint(3,7)):
        babble += gen_tone(duration, sr)
    babble = babble / (np.max(np.abs(babble)) + 1e-9)
    # Random mix
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
    c = clean / (np.std(clean) + 1e-9)
    n = noise / (np.std(noise) + 1e-9)
    # scale noise for SNR
    rms_c = np.sqrt(np.mean(c**2) + 1e-9)
    rms_n = np.sqrt(np.mean(n**2) + 1e-9)
    target_rms_n = rms_c / (10**(snr_db/20.0))
    n = n * (target_rms_n / (rms_n + 1e-9))
    noisy = c + n
    return norm_audio(noisy), norm_audio(c), norm_audio(n)

# --------------------------
# tf.data pipeline
# --------------------------
def wav_loader_factory(clean_paths, noise_paths):
    # Preload (small metadata only); read wav on-the-fly
    def load_and_mix(_):
        # choose clean
        if clean_paths:
            cp = random.choice(clean_paths)
            c, _sr = read_wav_mono(cp, SR)
        else:
            c = gen_tone(SEGMENT_SEC)
        # choose noise
        if noise_paths and np.random.rand() < 0.9:
            npth = random.choice(noise_paths)
            n, _sr = read_wav_mono(npth, SR)
        else:
            n = gen_noise(SEGMENT_SEC + 1.0)  # slight extra
        # random segments
        c_seg = random_segment(c, SEGMENT)
        n_seg = random_segment(n, SEGMENT)
        noisy, clean, noise = mix_clean_noise(c_seg, n_seg)
        return noisy.astype(np.float32), clean.astype(np.float32)
    return load_and_mix

def tf_dataset(clean_paths, noise_paths, batch_size, steps):
    def gen():
        loader = wav_loader_factory(clean_paths, noise_paths)
        for _ in range(steps * batch_size * 2):  # over-generate to help shuffling
            yield loader(None)
    output_sig = (tf.TensorSpec(shape=(SEGMENT,), dtype=tf.float32),
                  tf.TensorSpec(shape=(SEGMENT,), dtype=tf.float32))
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)
    ds = ds.shuffle(8192, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = tf_dataset(CLEAN_WAVS, NOISE_WAVS, BATCH_SIZE, STEPS_PER_EPOCH)
val_ds   = tf_dataset(CLEAN_WAVS, NOISE_WAVS, BATCH_SIZE, VAL_STEPS)

# Peek one batch for visualization sanity
noisy_b, clean_b = next(iter(train_ds))
print("Batch shapes:", noisy_b.shape, clean_b.shape)

# --------------------------
# Model: U-Net on log-mag STFT predicting soft mask
# --------------------------
def stft_layer(x):
    # x: (B, T)
    X = tf.numpy_function(lambda a: tfs.stft(a, WIN_LENGTH, HOP, N_FFT,
                                             window_fn=tf.signal.hann_window).numpy(),
                          [x], Tout=tf.complex64)
    # tf.numpy_function loses shape info:
    X.set_shape([None, None, N_FFT//2 + 1])
    return X

class STFTMagLayer(layers.Layer):
    def call(self, x):
        # x: (B, T)
        X = tfs.stft(x, frame_length=WIN_LENGTH, frame_step=HOP, fft_length=N_FFT,
                     window_fn=tf.signal.hann_window)
        mag = tf.abs(X)  # (B, F, T) actually (B, frames, bins)
        phase = tf.math.angle(X)
        # we prefer (B, T, F) layout for CNNs -> transpose
        return tf.transpose(mag, [0,1,2]), tf.transpose(phase, [0,1,2]), X

def db_log(x):
    return tf.math.log(x + 1e-6)

def inv_db_log(x):
    return tf.math.expm1(x)

def unet_block(x, filters, name, down=True):
    if down:
        x = layers.Conv2D(filters, 3, strides=2, padding="same", name=name+"_conv")(x)
        x = layers.BatchNormalization(name=name+"_bn")(x)
        x = layers.Activation("relu", name=name+"_relu")(x)
        return x
    else:
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", name=name+"_deconv")(x)
        x = layers.BatchNormalization(name=name+"_bn")(x)
        x = layers.Activation("relu", name=name+"_relu")(x)
        return x

def build_unet(n_mels=None):
    # input: raw audio -> STFT mag (log) -> CNN -> mask -> apply to complex STFT -> iSTFT
    inp = keras.Input(shape=(SEGMENT,), name="audio_in")

    # STFT magnitude/phase
    Xc = tfs.stft(inp, frame_length=WIN_LENGTH, frame_step=HOP, fft_length=N_FFT,
                  window_fn=tf.signal.hann_window)                               # (B, T', F)
    mag = tf.abs(Xc)
    phase = tf.math.angle(Xc)                                                    # (B, T', F)

    # To image-like tensor: (B, T', F, 1)
    M = tf.expand_dims(mag, -1)

    # Encoder
    e1 = layers.Conv2D(32, 3, padding="same", activation="relu")(M)
    d1 = unet_block(e1, 64, "down1", down=True)
    d2 = unet_block(d1, 128, "down2", down=True)
    d3 = unet_block(d2, 256, "down3", down=True)
    bott = layers.Conv2D(512, 3, padding="same", activation="relu")(d3)

    # Decoder with skip connections
    u3 = unet_block(bott, 256, "up3", down=False)
    u3 = layers.Concatenate()([u3, d2])
    u2 = unet_block(u3, 128, "up2", down=False)
    u2 = layers.Concatenate()([u2, d1])
    u1 = unet_block(u2, 64, "up1", down=False)
    u1 = layers.Concatenate()([u1, e1])

    out_mask = layers.Conv2D(1, 1, activation="sigmoid", name="mask")(u1)  # (B, T', F, 1)
    out_mask = tf.squeeze(out_mask, -1)                                    # (B, T', F)

    # Apply mask on magnitude, reconstruct complex STFT and iSTFT
    enh_mag = out_mask * mag                                               # (B, T', F)
    real = enh_mag * tf.cos(phase)
    imag = enh_mag * tf.sin(phase)
    enh_complex = tf.complex(real, imag)
    enh_audio = tfs.inverse_stft(enh_complex, frame_length=WIN_LENGTH, frame_step=HOP,
                                 window_fn=tf.signal.hann_window, 
                                 output_length=SEGMENT)
    # For training, output both enhanced audio and intermediates
    return keras.Model(inp, outputs=[enh_audio, out_mask, mag], name="UNet_Denoiser")

model = build_unet()
model.summary()

# --------------------------
# Losses
# --------------------------
def l1_mag_loss(true_audio, pred_audio):
    # Compare magnitude spectrograms (L1)
    Y = tfs.stft(true_audio, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
    P = tfs.stft(pred_audio, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
    return tf.reduce_mean(tf.abs(tf.abs(Y) - tf.abs(P)))

def spectral_convergence(true_audio, pred_audio):
    Y = tfs.stft(true_audio, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
    P = tfs.stft(pred_audio, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
    num = tf.norm(tf.abs(Y) - tf.abs(P), ord='fro')
    den = tf.norm(tf.abs(Y), ord='fro') + eps()
    return num / den

def mrstft_loss(true_audio, pred_audio):
    # Multi-resolution STFT: different hops & fft sizes
    cfgs = [
        (1024, 256), (512, 128), (2048, 512)
    ]
    loss = 0.0
    for nfft, hop in cfgs:
        Y = tfs.stft(true_audio, nfft, hop, nfft, window_fn=tf.signal.hann_window)
        P = tfs.stft(pred_audio, nfft, hop, nfft, window_fn=tf.signal.hann_window)
        loss += tf.reduce_mean(tf.abs(tf.abs(Y) - tf.abs(P)))
    return loss / len(cfgs)

def si_sdr(true_audio, pred_audio):
    # Scale-Invariant SDR (higher is better) -> we use negative as loss
    x = true_audio
    s = pred_audio
    x_zm = x - tf.reduce_mean(x, axis=-1, keepdims=True)
    s_zm = s - tf.reduce_mean(s, axis=-1, keepdims=True)
    proj = tf.reduce_sum(s_zm * x_zm, axis=-1, keepdims=True) / (tf.reduce_sum(x_zm**2, axis=-1, keepdims=True) + eps()) * x_zm
    e = s_zm - proj
    si_sdr_val = 10 * tf.math.log((tf.reduce_sum(proj**2, axis=-1) + eps()) / (tf.reduce_sum(e**2, axis=-1) + eps())) / tf.math.log(10.0)
    return si_sdr_val

def si_sdr_loss(true_audio, pred_audio):
    return -tf.reduce_mean(si_sdr(true_audio, pred_audio))

# Combined loss
def total_loss(true_audio, pred_audio):
    # weights can be tuned
    return (0.5 * l1_mag_loss(true_audio, pred_audio) +
            0.2 * spectral_convergence(true_audio, pred_audio) +
            0.2 * mrstft_loss(true_audio, pred_audio) +
            0.1 * si_sdr_loss(true_audio, pred_audio))

# --------------------------
# Optimizer with Warmup + Cosine Decay
# --------------------------
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
            0.5 * self.base_lr * (1 + tf.cos(np.pi * (step - warm) / tf.maximum(1.0, (total - warm))))
        )
        return lr

TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
lr_schedule = WarmupCosine(LEARNING_RATE, WARMUP_STEPS, TOTAL_STEPS)
opt = keras.optimizers.Adam(learning_rate=lr_schedule)

# --------------------------
# Exponential Moving Average (EMA) weights
# --------------------------
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = [tf.identity(w) for w in model.weights]

    def update(self, model):
        for i, w in enumerate(model.weights):
            self.shadow[i].assign(self.decay * self.shadow[i] + (1.0 - self.decay) * w)

    def apply_to(self, model):
        self.backup = [tf.identity(w) for w in model.weights]
        for w, s in zip(model.weights, self.shadow):
            w.assign(s)

    def restore(self, model):
        for w, b in zip(model.weights, self.backup):
            w.assign(b)
        self.backup = None

ema = EMA(model, EMA_DECAY)

# --------------------------
# Training Step (custom train loop to include custom loss and EMA)
# --------------------------
train_loss_metric = keras.metrics.Mean()
val_loss_metric   = keras.metrics.Mean()
train_si_sdr_metric = keras.metrics.Mean()
val_si_sdr_metric   = keras.metrics.Mean()

@tf.function
def train_step(noisy, clean):
    with tf.GradientTape() as tape:
        enhanced_audio, mask, mag = model(noisy, training=True)
        loss = total_loss(clean, enhanced_audio)
        # Make sure numeric stability in mixed precision
        if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
            loss = tf.cast(loss, tf.float32)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    ema.update(model)
    # metrics
    train_loss_metric.update_state(loss)
    train_si_sdr_metric.update_state(si_sdr(clean, enhanced_audio))

@tf.function
def val_step(noisy, clean):
    enhanced_audio, mask, mag = model(noisy, training=False)
    loss = total_loss(clean, enhanced_audio)
    val_loss_metric.update_state(loss)
    val_si_sdr_metric.update_state(si_sdr(clean, enhanced_audio))

# --------------------------
# Training Loop
# --------------------------
history = {"loss": [], "val_loss": [], "si_sdr": [], "val_si_sdr": []}
global_step = 0

for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    train_loss_metric.reset_states()
    val_loss_metric.reset_states()
    train_si_sdr_metric.reset_states()
    val_si_sdr_metric.reset_states()

    for step, (noisy, clean) in enumerate(train_ds.take(STEPS_PER_EPOCH), start=1):
        train_step(noisy, clean)
        global_step += 1
        if step % 100 == 0:
            print(f"  step {step}/{STEPS_PER_EPOCH}  lr={opt.lr(global_step).numpy():.6f}  "
                  f"loss={train_loss_metric.result().numpy():.4f}  "
                  f"SI-SDR={train_si_sdr_metric.result().numpy():.2f}dB")

    # Validation (with EMA-applied weights)
    ema.apply_to(model)
    for step, (noisy, clean) in enumerate(val_ds.take(VAL_STEPS), start=1):
        val_step(noisy, clean)
    ema.restore(model)

    tr_loss = float(train_loss_metric.result().numpy())
    va_loss = float(val_loss_metric.result().numpy())
    tr_sdr  = float(train_si_sdr_metric.result().numpy())
    va_sdr  = float(val_si_sdr_metric.result().numpy())

    history["loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["si_sdr"].append(tr_sdr)
    history["val_si_sdr"].append(va_sdr)

    # Save checkpoint (EMA weights for inference)
    ema.apply_to(model)
    model.save_weights(os.path.join(CHECKPOINT_DIR, f"epoch{epoch:02d}.weights.h5"))
    ema.restore(model)

    print(f"Epoch {epoch} done. train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
          f"train_SI-SDR={tr_sdr:.2f}dB val_SI-SDR={va_sdr:.2f}dB")

# --------------------------
# Plot training curves
# --------------------------
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

# --------------------------
# Quick qualitative check (visual + audio) on a fresh batch
# --------------------------
ema.apply_to(model)  # use EMA weights for eval

noisy_b, clean_b = next(iter(val_ds))
enh_b, mask_b, mag_b = model(noisy_b, training=False)

idx = 0
noisy = noisy_b[idx].numpy()
clean = clean_b[idx].numpy()
enh   = enh_b[idx].numpy()

print("Playing audio (Noisy -> Enhanced -> Clean)")
display(Audio(noisy, rate=SR))
display(Audio(enh, rate=SR))
display(Audio(clean, rate=SR))

# Spectrograms & mask
N = tfs.stft(noisy, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
C = tfs.stft(clean, WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
E = tfs.stft(enh,   WIN_LENGTH, HOP, N_FFT, window_fn=tf.signal.hann_window)
plot_spectrograms(np.abs(N).numpy(), np.abs(C).numpy(), np.abs(E).numpy(),
                  sr=SR, title="Noisy / Clean / Enhanced (Mag dB)")
plot_mask(mask_b[idx].numpy(), title="Predicted Soft Mask")
plot_waveforms(noisy, clean, enh, sr=SR, title="Waveforms")

ema.restore(model)

# --------------------------
# Export SavedModel (EMA weights) for easy re-use
# --------------------------
ema.apply_to(model)

# Wrap only the audio output for SavedModel signature
class InferenceWrapper(keras.Model):
    def __init__(self, base):
        super().__init__()
        self.base = base
    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
    def denoise(self, audio):
        audio = tf.expand_dims(audio, 0)  # (1, T)
        enh, _mask, _mag = self.base(audio, training=False)
        return tf.squeeze(enh, 0)

infer_model = InferenceWrapper(model)
tf.saved_model.save(infer_model, EXPORT_DIR)
print("Exported to:", EXPORT_DIR)

# --------------------------
# Inference helper: denoise a file and visualize
# --------------------------
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

# Optional demo: generate a synthetic noisy utterance and denoise it
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
