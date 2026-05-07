# build_reverb_dataset.py
import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import convolve
import rir_generator as rir
import random
from tqdm import tqdm

# PATHS
clean_root = "/home/dsi/skopavi/Project/kws_project/data/raw/data_new"
output_root = "/home/dsi/skopavi/Project/kws_project/data/reverb/raw_reverb"

os.makedirs(output_root, exist_ok=True)


# PARAMS
RT60_LIST = [0.5]
room_dim = [6, 5, 3.5]
source_pos = [2, 2, 1.5]
mic_pos = [5, 4, 2]

# LOOP
folders = [f for f in os.listdir(clean_root)
           if os.path.isdir(os.path.join(clean_root, f))
           and f != "_background_noise_"]

folders = sorted(folders)  # ensure consistent order

for label in folders:
    in_dir = os.path.join(clean_root, label)
    out_dir = os.path.join(output_root, label)
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.endswith(".wav")]

    for fname in tqdm(files, desc=f"Reverb {label}"):

        path = os.path.join(in_dir, fname)
        clean, sr = librosa.load(path, sr=16000)

        # choose random RT60 for this sample
        # rt60 = random.choice(RT60_LIST)
        rt60 = RT60_LIST[0]  # use fixed RT60 for now

        # make RIR
        h = rir.generate(
            c=340,
            fs=sr,
            r=mic_pos,
            s=source_pos,
            L=room_dim,
            reverberation_time=rt60
        ).flatten()

        # make reverbed signal (convolution)
        reverbed = convolve(clean, h, mode="full")

        # cut to original length
        # reverbed = reverbed[:len(clean)]

        out_path = os.path.join(out_dir, fname)
        sf.write(out_path, reverbed, sr)

print("✅ Reverb dataset saved!")