import librosa
import numpy as np

def load_audio(path, sr=22050, duration=None):
    """Load audio, convert to mono, normalize amplitude."""
    y, sr = librosa.load(path, sr=sr, duration=duration, mono=True)
    y = librosa.util.normalize(y)
    return y, sr
