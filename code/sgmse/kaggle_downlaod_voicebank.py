import os
import kagglehub

os.environ["KAGGLEHUB_CACHE"] = "/home/dsi/skopavi/Project/kws_project/sgmse_repo/data"

path = kagglehub.dataset_download("jweiqi/voicebank-demand-16k")

print("Downloaded to:", path)