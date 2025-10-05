# NeuroGrow: Infant EEG Development Analysis

**NeuroGrow** is an advanced deep learning project designed to analyze infant EEG signals for neurodevelopmental insights. Using the APPLESEED EEG dataset, this project implements cutting-edge techniques to extract, model, and interpret brain activity across early developmental stages.

## Techniques and Methodology

### 1. Signal Preprocessing
- Bandpass filtering (1–40 Hz) to isolate relevant EEG frequencies.
- Average referencing to normalize signals across channels.
- Artifact handling and optional epoching around events.

### 2. Feature Extraction
- Power Spectral Density (PSD) computation for all EEG channels.
- Log-transformed PSD features to enhance signal interpretability.
- Time-frequency decomposition using wavelet transforms (optional) for dynamic pattern analysis.

### 3. Deep Learning Models
- **CNN+LSTM Hybrid**:  
  - Convolutional layers extract spatial patterns across EEG channels.  
  - LSTM layers capture temporal dependencies in neural signals.  
  - Fully connected layers classify developmental stages based on EEG features.
- **Supervised Learning**: Predict infant developmental stages using labeled EEG recordings.
- **Unsupervised Learning (Optional)**: Autoencoders for latent feature extraction and clustering of hidden EEG patterns.

### 4. Evaluation and Interpretability
- Accuracy and loss tracking during training.
- Visualization of EEG power distribution and channel-specific activity.
- Explainable AI techniques (e.g., feature importance, latent representations) to highlight neural patterns driving predictions.

### 5. Key Libraries
- **Data Handling:** NumPy, Pandas  
- **Signal Processing:** MNE, SciPy  
- **Visualization:** Matplotlib, Seaborn  
- **Deep Learning:** PyTorch  
- **ML Utilities:** Scikit-learn  

---

NeuroGrow demonstrates **state-of-the-art AI applied to pediatric EEG analysis**, providing insights into infant brain development while serving as a foundation for advanced neurodata research.