# Noise Robustness in Keyword Spotting (KWS)
This repository contains the implementation and experiments for the final project
“Noise Robustness in Keyword Spotting”, conducted as part of the Electrical Engineering program
(Signal Processing track).

The project investigates how background noise affects the performance of Keyword Spotting (KWS) systems, and whether integrating a Denoising module can improve robustness under noisy conditions.

### Project Overview
Keyword Spotting (KWS) systems are widely used in voice-controlled devices and human–machine interfaces.
However, their performance often degrades significantly in real-world noisy environments.

The goal of this project is to evaluate and improve the robustness of a KWS system by integrating a Denoiser module before the classification stage.

We evaluate three main scenarios:
1. Clean Audio → KWS
2. Noisy Audio → KWS
3. Noisy Audio → Denoiser → KWS

Performance is compared across different noise levels (SNR values) to assess the effectiveness of noise reduction techniques.

### Objectives

1. Build a baseline KWS system using MFCC features and a CNN-based model (DS-CNN).
2. Simulate realistic noisy environments by adding background noise at different SNR levels.
3. Design and train a Denoiser model (Autoencoder / UNet-based).
4. Evaluate the impact of denoising on KWS performance.
5. Compare different training strategies: Fine-Tuning / Joint (End-to-End) Training
6. Analyze performance using quantitative metrics and visualizations.

### Data Preparation
**Speech Dataset**
The project uses the Google Speech Commands v2 dataset:
https://www.kaggle.com/datasets/sylkaladin/speech-commands-v2
Download the dataset and place it in the appropriate data directory as defined in the project scripts.

**Noise Data**
Background noise samples are added to clean speech signals to simulate noisy environments at different SNR levels.

### Installation and Environment Setup
Prerequisites
1. Python 3.9 - 3.11
2. Git
3. (Optional) GPU with CUDA support for faster training
**Step 1 - Clone the repository**
https://github.com/MayaVB/kws_project.git
**Step 2 - Create and activate a virtual environment**
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
**Step 3 - Install dependencies**
pip install -r requirements.txt

### References
KWS on Microcontrollers, Google Research
https://arxiv.org/abs/1711.07128

### Authors
Ido Ben David
Avital Skop
Supervised by Maya Veisman 
Academic Supervisor: Prof. Sharon Gannot