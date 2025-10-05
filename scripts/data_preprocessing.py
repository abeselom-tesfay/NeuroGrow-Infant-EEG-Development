# scripts/data_preprocessing.py

import os
import mne
import numpy as np
import pandas as pd

def load_sub_session(sub_path, ses):
    """
    Load EEG raw data for a single subject/session
    """
    eeg_path = os.path.join(sub_path, f"ses-{ses}", "eeg")
    vhdr_files = [f for f in os.listdir(eeg_path) if f.endswith(".vhdr")]
    
    raws = []
    for vhdr in vhdr_files:
        raw = mne.io.read_raw_brainvision(os.path.join(eeg_path, vhdr), preload=True)
        raw.filter(1., 40.)  # bandpass filter 1-40Hz
        raw.set_eeg_reference('average')
        raws.append(raw)
    return raws

def load_all_subjects(dataset_path="APPLESEED_Dataset"):
    """
    Load all subjects and sessions
    """
    subjects = [d for d in os.listdir(dataset_path) if d.startswith("sub")]
    all_data = []
    for sub in subjects:
        sub_path = os.path.join(dataset_path, sub)
        for ses in ["1","2","3","4"]:
            try:
                raws = load_sub_session(sub_path, ses)
                all_data.extend(raws)
            except:
                continue
    return all_data

if __name__ == "__main__":
    dataset_path = "APPLESEED_Dataset"
    all_raws = load_all_subjects(dataset_path)
    print(f"Total EEG recordings loaded: {len(all_raws)}")
