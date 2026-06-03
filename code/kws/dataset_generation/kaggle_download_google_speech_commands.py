# kaggle_download_google_speech_commands.py
import kagglehub
import os

# # Download latest version
# path = kagglehub.dataset_download("neehakurelli/google-speech-commands")

# print("Path to dataset files:", path)

os.environ["KAGGLEHUB_CACHE"] = "/home/dsi/skopavi/Project/kws_project/data/raw/data_new"

path = kagglehub.dataset_download("neehakurelli/google-speech-commands")
print("Path to dataset files:", path)
