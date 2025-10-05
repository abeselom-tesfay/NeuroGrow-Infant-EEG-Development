import numpy as np
import torch
from mne.time_frequency import psd_welch

def extract_features(raw):
    """
    Extract PSD features for all channels
    Returns numpy array: channels x freq_bins
    """
    psd, freqs = psd_welch(raw, fmin=1, fmax=40, n_fft=256)
    psd = np.log(psd + 1e-6)  # log transform
    return psd

def create_dataset(all_raws):
    """
    Build dataset and labels
    """
    X, y = [], []
    for raw in all_raws:
        features = extract_features(raw)
        X.append(features)
        # Age label from session number (as example)
        ses = int(raw.info['meas_date'].month % 4 + 1)  # placeholder
        y.append(ses)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

if __name__ == "__main__":
    from data_preprocessing import load_all_subjects
    raws = load_all_subjects()
    X, y = create_dataset(raws)
    print(f"Feature dataset shape: {X.shape}, Labels shape: {y.shape}")
