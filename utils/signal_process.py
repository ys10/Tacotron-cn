# coding=utf-8
from __future__ import print_function, division

from configs.hyper_params import SignalProcessConfig
from scipy import signal
import os
import numpy as np
import librosa
import copy
import matplotlib.pyplot as plt
plt.switch_backend('pdf')

config = SignalProcessConfig


def _wav2spectrogram(wav_path):
    """
    Returns normalized log(mel_spectrogram) and log(magnitude_spectrogram) from raw wave.
    :param wav_path: A string. The full path of a wave file.
    :return:
        mel: A 2d array of shape (T, n_mels) <- Transposed
        mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    """
    # Loading sound file
    y, sr = librosa.load(wav_path, sr=config.sample_rate)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Pre-emphasis
    y = np.append(y[0], y[1:] - config.pre_emphasis * y[:-1])

    # short-time Fourier transform
    linear = librosa.stft(y=y,
                          n_fft=config.n_fft,
                          hop_length=config.hop_length,
                          win_length=config.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, config.n_fft, config.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - config.ref_db + config.max_db) / config.max_db, 1e-8, 1)
    mag = np.clip((mag - config.ref_db + config.max_db) / config.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def spectrogram2wav(mag):
    """
    Generate wave file from spectrogram.
    :param mag:
    :return:
    """
    # transpose
    mag = mag.T
    # de-noramlize
    mag = (np.clip(mag, 0, 1) * config.max_db) - config.max_db + config.ref_db
    # to amplitude
    mag = np.power(10.0, mag * 0.05)
    # wav reconstruction
    wav = _griffin_lim(mag)
    # de-preemphasis
    wav = signal.lfilter([1], [1, -config.pre_emphasis], wav)
    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def _invert_spectrogram(spectrogram):
    """
    Inverse short-time Fourier transform.
    :param spectrogram:
    :return:
    """
    return librosa.istft(spectrogram, config.hop_length, win_length=config.win_length, window="hann")


def _griffin_lim(spectrogram):
    """
    Applies Griffin-Lim's raw.
    :param spectrogram: magnitude spectrogram.
    :return: waveform
    """
    x_best = copy.deepcopy(spectrogram)
    for i in range(config.n_iter):
        x_t = _invert_spectrogram(x_best)
        est = librosa.stft(x_t, config.n_fft, config.hop_length, win_length=config.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        x_best = spectrogram * phase
    x_t = _invert_spectrogram(x_best)
    y = np.real(x_t)

    return y


def plot_alignment(alignment, global_step, align_dir):
    """
    Plots the alignments.
    :param alignment: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    :param global_step: (int) global step
    :param align_dir: (string) log directory.
    :return:
    """
    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im)
    plt.title('{} Steps'.format(global_step))
    plt.savefig('{}/alignment_{}k.png'.format(align_dir, global_step//1000), format='png')


def load_spectrogram(wav_path, reduction):
    """
    :param wav_path:
    :param reduction:
    :return:
    """
    filename = os.path.basename(wav_path)
    mel, mag = _wav2spectrogram(wav_path)
    t = mel.shape[0]
    num_padding = reduction - (t % reduction) if t % reduction != 0 else 0  # for reduction
    mel = np.pad(mel, [[0, num_padding], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_padding], [0, 0]], mode="constant")
    return filename, mel.reshape((-1, config.n_mels * reduction)), mag
