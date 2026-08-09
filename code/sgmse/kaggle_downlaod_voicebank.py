import os
import kagglehub

os.environ["KAGGLEHUB_CACHE"] = "/home/dsi/skopavi/Project/kws_project/code/sgmse/data"

# path = kagglehub.dataset_download("jweiqi/voicebank-demand-16k")

# print("Downloaded to:", path)

path = kagglehub.dataset_download("aanhari/demand-dataset")

print("Path to dataset files:", path)