# kaggle_download_google_speech_commands.py
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("neehakurelli/google-speech-commands")

# print("Path to dataset files:", path)

import os
os.environ["KAGGLEHUB_CACHE"] = "/home/dsi/skopavi/Project/kws_project/data/raw/data_new"

import kagglehub

path = kagglehub.dataset_download("neehakurelli/google-speech-commands")
print("Path to dataset files:", path)
