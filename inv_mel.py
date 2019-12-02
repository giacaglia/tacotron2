import librosa
import librosa.filters
import numpy as np
from hparams import create_hparams
from scipy.io import wavfile
import os

# Conversions
_mel_basis = None
_inv_mel_basis = None

v_hparams = create_hparams()

def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, v_hparams.sampling_rate, wav.astype(np.int16))

def _build_mel_basis():
    assert v_hparams.mel_fmax <= v_hparams.sampling_rate // 2
    return librosa.filters.mel(v_hparams.sampling_rate, v_hparams.filter_length, n_mels=v_hparams.n_mel_channels,
            fmin=v_hparams.mel_fmin, fmax=v_hparams.mel_fmax)

def get_hop_size():
    hop_size = v_hparams.hop_length
    return hop_size

def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)

def _stft(y):
    return librosa.stft(y=y, n_fft=v_hparams.filter_length, hop_length=get_hop_size())

def _istft(y):
    return librosa.istft(y, hop_length=get_hop_size())

def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(v_hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y

def _mel_to_linear(mel_spectrogram):
    global _inv_mel_basis
    if _inv_mel_basis is None:
         _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def inv_mel_spectrogram(mel_spectrogram):
    D = mel_spectrogram
    print('Max for numpy array: {}'.format(np.amax(mel_spectrogram)))
    S = _mel_to_linear(_db_to_amp(D + v_hparams.ref_level_db))  # Convert back to linear
    return _griffin_lim(S ** v_hparams.power)

mel_filename = './filelists/ljspeech/ljspeech-mel-04360.npy'
mel = np.load(mel_filename)
wav = inv_mel_spectrogram(mel.T)
save_wav(wav, os.path.join('./outdir/audio/', '{:03d}.wav'.format(4360)))
