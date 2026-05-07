# build_noise_dataset.py
import os
import pandas as pd
import librosa
import soundfile as sf
import random
from tqdm import tqdm

from kws.src.noise_dataset import mix_with_noise_at_snr

clean_root = "/home/dsi/skopavi/Project/kws_project/data/raw/data_new"
noise_root = clean_root + "/home/dsi/skopavi/Project/kws_project/data/raw/data_new/_background_noise_"

output_root = "/home/dsi/skopavi/Project/kws_project/data/noisy/generated_noisy"
meta_path = "/home/dsi/skopavi/Project/kws_project/data/generated_noisy_metadata.csv"

os.makedirs(output_root, exist_ok=True)

# load noise bank
noise_files = [f for f in os.listdir(noise_root) if f.endswith(".wav")]
noise_bank = {}

for f in noise_files:
    name = os.path.splitext(f)[0]
    y, _ = librosa.load(os.path.join(noise_root, f), sr=16000)
    noise_bank[name] = y

rows = []

folders = [f for f in os.listdir(clean_root)
           if os.path.isdir(os.path.join(clean_root, f))
           and f != "_background_noise_"]

folders = sorted(folders)  # ensure consistent order

for label in folders:
    in_dir = os.path.join(clean_root, label)
    out_dir = os.path.join(output_root, label)
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.endswith(".wav")]

    for fname in tqdm(files, desc=f"Processing {label}"):

        path = os.path.join(in_dir, fname)
        clean, _ = librosa.load(path, sr=16000)

        noise_name = random.choice(list(noise_bank.keys()))
        noise = noise_bank[noise_name]

        snr = random.choice([-10, 0, 10])

        noisy = mix_with_noise_at_snr(clean, noise, snr)

        out_path = os.path.join(out_dir, fname)
        sf.write(out_path, noisy, 16000)

        rows.append({
            "filename": fname,
            "label": label,
            "snr": snr,
            "noise": noise_name
        })

# save metadata
df_meta = pd.DataFrame(rows)
df_meta.to_csv(meta_path, index=False)

print("✅ Noisy dataset + metadata saved!")