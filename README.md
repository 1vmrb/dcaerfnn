# Detection and Classification of Acoustic Environment Using Random Forest and Neural Network

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-blue?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Librosa](https://img.shields.io/badge/Librosa-0.10+-brightgreen)](https://librosa.org/)
[![GitHub Pages](https://img.shields.io/badge/Demo-Live%20Web%20DSP%20Lab-indigo)](https://1vmrb.github.io/dcaerfnn/)

> **Real-time Acoustic Scene Classification (ASC) pipeline designed for next-generation intelligent hearing aids.** Classifies 7 distinct environmental soundscapes using Mel-Frequency Cepstral Coefficients (MFCCs) and Principal Component Analysis (PCA). The Deep Neural Network attains **97.0% classification accuracy**, reliably separating challenging and acoustically ambiguous environments such as *Restaurant* versus *Babble*.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Target Acoustic Soundscapes](#target-acoustic-soundscapes)
- [System Architecture & Pipeline](#system-architecture--pipeline)
- [Feature Extraction & Dimensionality Reduction](#feature-extraction--dimensionality-reduction)
- [Models & Classifiers](#models--classifiers)
- [The "Restaurant" vs. "Babble" Challenge](#the-restaurant-vs-babble-challenge)
- [Empirical Benchmarks](#empirical-benchmarks)
- [Interactive Web DSP Lab](#interactive-web-dsp-lab)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Copyright & Terms of Use](#copyright--terms-of-use)

---

## Project Overview

Modern digital hearing aids must automatically adapt their Digital Signal Processing (DSP) parameters—such as directional beamforming, wide dynamic range compression (WDRC), and transient impulse limiters—depending on the user's surrounding acoustic context. 

This repository implements an end-to-end Machine Learning and Deep Learning pipeline capable of real-time classification across **7 target acoustic environments**. By extracting statistical spectral descriptors combined with **$20$ MFCCs and delta coefficients ($\Delta, \Delta^2$)**, followed by **Principal Component Analysis (PCA)** for low-latency dimensionality reduction, the models deliver rapid and dependable inference tailored for embedded audio hardware constraints.

---

## Target Acoustic Soundscapes

The system detects and discriminates between 7 core hearing aid operational environments:

| Soundscape | Acoustic Characteristics | Hearing Aid DSP Adaptation Policy |
| :--- | :--- | :--- |
| **🍽️ Restaurant** | Impulsive high-frequency clatter (cutlery/plates) mixed with diffuse chatter | Narrow frontal beamforming + anti-clatter transient limiter |
| **🗣️ Speech Babble** | Multi-speaker cocktail-party noise without percussive transients | Binaural speech beamforming + target voice isolation |
| **🚗 Traffic / Street** | Low-frequency rolling tire noise, engine rumble, passing vehicles | High-pass rumble filter + road noise suppression ($-14\text{ dB}$) |
| **🚇 Subway / Metro** | Extreme low-frequency resonance and intermittent rail shrieks | Narrow notch filtering + low dynamic range clamping |
| **🌿 Quiet / Park** | Low ambient noise floor with sparse acoustic cues (wind, distant birds) | $360^\circ$ omnidirectional microphone + soft whisper gain boost |
| **🎵 Music / Concert** | High dynamic range, wide bandwidth, harmonic overtones | Linear pass-through (zero aggressive compression / noise gating) |
| **🏗️ Construction** | Extreme impulse spikes, continuous mechanical drilling, impacts | Instant peak limiter ($0.2\text{ ms}$ attack) + maximum attenuation |

---

## System Architecture & Pipeline

```text
  [ Raw Audio Stream ] (22.05 kHz, Mono)
           │
           ▼
  [ Pre-Emphasis & Windowing ] (Hamming window, Frame = 2048, Hop = 512)
           │
           ▼
  [ Short-Time Fourier Transform (STFT) ]
           │
           ▼
  [ Mel-Filterbank Integration ] (40 Triangular Mel-Bands)
           │
           ▼
  [ Feature Extraction ]
  ├── 20 Mel-Frequency Cepstral Coefficients (MFCCs)
  ├── 1st & 2nd Temporal Derivatives (Δ & Δ²)
  ├── Spectral Centroid, Rolloff (85%), Spread, Flatness, Flux
  └── Zero-Crossing Rate (ZCR) & RMS Energy
           │
           ▼
  [ Standard Scaling & PCA Dimensionality Reduction ]
           │
     ┌─────┴────────────────────────┐
     ▼                              ▼
[ Random Forest ]         [ Deep Neural Network ]
  100 Estimators             Dense MLP + BatchNorm + Dropout
  Inference: 1.2 ms          Inference: 3.8 ms
  Accuracy: 92.3%            Accuracy: 97.0%
```

---

## Feature Extraction & Dimensionality Reduction

1. **Mel-Scale Mapping:**
   $$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$
   Filters frequencies according to human cochlear resolution.

2. **Cepstral Transformation:**
   Discrete Cosine Transform (DCT-II) decorrelates filterbank energies into compact cepstral coefficients:
   $$c_n = \sum_{k=0}^{K-1} S_k \cos\left[ \frac{\pi n}{K} \left( k + \frac{1}{2} \right) \right]$$

3. **Temporal Dynamics ($\Delta$ and $\Delta^2$):**
   Captures temporal context across adjacent frames:
   $$d_t = \frac{\sum_{n=1}^{N} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{N} n^2}$$

4. **PCA Dimensionality Reduction:**
   Feature dimension is compressed to retain $>95\%$ of explained variance while dropping redundant dimensions, keeping latency well below standard hearing aid delay thresholds ($\le 10\text{ ms}$).

---

## Models & Classifiers

### 1. Random Forest (RF)
- **Configuration:** 100 decision trees, Gini impurity criterion, maximum depth of 18, balanced class weights.
- **Advantages:** Low computation overhead, axis-aligned decision partitions, highly parallelizable, no GPU requirement.
- **Accuracy:** **$92.3\%$** on 10-fold cross-validation.

### 2. Deep Neural Network (MLP)
- **Configuration:** Multi-Layer Perceptron with 3 hidden layers ($256 \to 128 \to 64$), Batch Normalization, Dropout ($0.30, 0.25$), and L2 regularization ($10^{-4}$).
- **Activation:** Rectified Linear Unit (ReLU) for hidden layers; Softmax for 7 output classes.
- **Optimizer:** Adam ($\alpha = 0.001$) with `ReduceLROnPlateau` and `EarlyStopping`.
- **Accuracy:** **$97.0\%$** on independent test set.

---

## The "Restaurant" vs. "Babble" Challenge

Distinguishing multi-speaker chatter (**Babble**) from dining spaces (**Restaurant**) is one of the most documented failure modes in Acoustic Scene Classification. Both environments share similar vocal formant distributions in the $300\text{ Hz} - 3.4\text{ kHz}$ band.

### How the Pipeline Solves This:
- **Spectral Kurtosis & High-Order MFCCs:** Clinking silverware and plate contacts produce sharp, non-Gaussian transient bursts in the $3.5\text{ kHz} - 8\text{ kHz}$ range.
- **Spectral Flux & Flatness:** Babble exhibits a stationary, speech-shaped noise floor, whereas Restaurant scenes exhibit high intermittent flux.
- **Performance:** While the Random Forest model misclassified $7.0\%$ of Restaurant samples as Babble, the Deep Neural Network's continuous non-linear decision boundary reduced cross-confusion to under **$2.5\%$**.

---

## Empirical Benchmarks

### 10-Fold Cross-Validation Performance

| Metric | Random Forest | Deep Neural Network |
| :--- | :---: | :---: |
| **Overall Accuracy** | $92.3\%$ | **$97.0\%$** |
| **Macro Precision** | $91.8\%$ | **$97.1\%$** |
| **Macro Recall** | $92.0\%$ | **$96.9\%$** |
| **Macro F1-Score** | $91.9\%$ | **$97.0\%$** |
| **ROC-AUC** | $96.2\%$ | **$99.1\%$** |
| **Inference Latency** | **$1.2\text{ ms}$ (DSP/MCU)** | $3.8\text{ ms}$ (NPU/DSP) |

---

## Interactive Web DSP Lab

This repository includes a single-file, interactive web lab (`index.html`) deployable directly via GitHub Pages.

**Interactive Features:**
- **Real-Time DSP:** Synthesizes 7 acoustic scenes or processes live microphone audio and custom uploaded `.wav`/`.mp3` recordings.
- **Live Visualizers:** High-fidelity oscilloscope time-domain waveform + real-time STFT waterfall spectrogram.
- **Dual Model Inference:** Live side-by-side output comparing Random Forest voting percentages against Neural Network Softmax probabilities.
- **PCA Manifold Viewer:** 2D canvas plotting the axis-aligned decision cuts of Random Forest versus the smooth non-linear manifolds of the Neural Network.
- **Hyperparameter Simulator:** Adjust tree depth, learning rate, and dropout with real-time animated loss/accuracy convergence plots.
- **Feature Exporter:** One-click download of the real-time extracted acoustic feature vector as a CSV report.

---

## Repository Structure

```text
├── index.html                   # Interactive Web DSP Lab & Model Visualizer
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── src/
│   ├── extract_features.py      # Librosa MFCC, spectral descriptors & PCA reduction
│   ├── train_rf.py              # Random Forest training & stratified 10-fold CV
│   ├── train_nn.py              # TensorFlow/Keras MLP training with regularization
│   └── evaluate.py              # Confusion matrix generation & classification metrics
└── data/                        # Dataset audio files / pre-computed feature matrices
    ├── features_pca.npy         # PCA-transformed feature arrays
    └── labels_7classes.npy      # Encoded ground-truth labels
```

---

## Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/1vmrb/DETECTION-AND-CLASSIFICATION-OF-ACOUSTIC-ENVIRONMENT-USING-RANDOM-FOREST-AND-NEURAL-NETWORK.git
cd DETECTION-AND-CLASSIFICATION-OF-ACOUSTIC-ENVIRONMENT-USING-RANDOM-FOREST-AND-NEURAL-NETWORK
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Extract Features
```bash
python src/extract_features.py --input_dir ./data/audio --output ./data/features_pca.npy
```

### 4. Train Models
```bash
# Train Random Forest Classifier
python src/train_rf.py

# Train Deep Neural Network
python src/train_nn.py
```

### 5. Run the Interactive Web Lab Locally
Simply open `index.html` in any modern web browser or serve it using Python's built-in HTTP server:
```bash
python3 -m http.server 8000
```
Navigate to `http://localhost:8000` to launch the lab.

---

## Requirements

```text
librosa>=0.10.1
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.14.0
pandas>=2.1.0
joblib>=1.3.2
soundfile>=0.12.1
matplotlib>=3.8.0
```

---

## Copyright & Terms of Use

All rights reserved. This repository and its accompanying materials are developed for educational, academic, and assistive hearing technology research purposes.

For inquiries regarding use, collaboration, or reproduction, please open an issue or reach out via the GitHub repository.
