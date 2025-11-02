import numpy as np
import librosa
from scipy.signal import correlate

def extract_features(y, sr):
    # MFCC (13) -> take mean over frames
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    # Spectral rolloff (85%)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean()

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    # RMS energy
    rms = librosa.feature.rms(y=y).mean()

    # Simple echo strength proxy via autocorrelation
    ac = correlate(y, y, mode='full')
    ac = ac[ac.size // 2:]  # positive lags
    # safe check
    if ac.size > 1:
        echo_strength = float(ac[1:].max() / (ac.max() + 1e-9))
    else:
        echo_strength = 0.0

    features = np.hstack([mfcc_mean, centroid, rolloff, zcr, rms, echo_strength])
    return features
