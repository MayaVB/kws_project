# calc_metrics.py 
from os.path import join 
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import librosa

from pystoi import stoi
import numpy as np

from sgmse.util.other import energy_ratios, mean_std

def valid_stats(arr):
    arr = [x for x in arr if x is not None]
    if len(arr) == 0:
        return 0, 0, 0
    return np.mean(arr), np.std(arr), len(arr)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True, help='Directory containing the clean data')
    parser.add_argument("--noisy_dir", type=str, required=True, help='Directory containing the noisy data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    args = parser.parse_args()

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": []}

    # Evaluate standard metrics
    noisy_files = []
    # noisy_files += sorted(glob(join(args.noisy_dir, '*.wav')))
    # noisy_files += sorted(glob(join(args.noisy_dir, '**', '*.wav')))
    noisy_files = sorted(glob(join(args.noisy_dir, '**', '*.wav'), recursive=True))
    print(f"DEBUG: Found {len(noisy_files)} noisy files to evaluate in {args.noisy_dir}")
    # for noisy_file in tqdm(noisy_files):
    for noisy_file in noisy_files:
        filename = noisy_file.replace(args.noisy_dir, "")[1:]
        if 'dB' in filename:
            clean_filename = filename.split("_")[0] + ".wav"
        else:
            clean_filename = filename
        x, sr_x = read(join(args.clean_dir, clean_filename))
        y, sr_y = read(join(args.noisy_dir, filename))
        x_hat, sr_x_hat = read(join(args.enhanced_dir, filename))
        assert sr_x == sr_y == sr_x_hat
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        x_hat = x_hat[:min_len]
        n = y - x 
        x_hat_16k = librosa.resample(x_hat, orig_sr=sr_x_hat, target_sr=16000) if sr_x_hat != 16000 else x_hat
        x_16k = librosa.resample(x, orig_sr=sr_x, target_sr=16000) if sr_x != 16000 else x
        data["filename"].append(filename)
        """
        data["pesq"].append(pesq(16000, x_16k, x_hat_16k, 'wb'))
        data["estoi"].append(stoi(x, x_hat, sr_x, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])
        """
        # PESQ
        try:
            pesq_val = pesq(16000, x_16k, x_hat_16k, 'wb')
        except:
            pesq_val = None
        # ESTOI
        try:
            estoi_val = stoi(x, x_hat, sr_x, extended=True)
            if estoi_val < 1e-4:  # מסנן garbage
                estoi_val = None
        except:
            estoi_val = None
        # SI metrics 
        try:
            si_sdr_val = energy_ratios(x_hat, x, n)[0]
            si_sir_val = energy_ratios(x_hat, x, n)[1]
            si_sar_val = energy_ratios(x_hat, x, n)[2]
        except:
            si_sdr_val, si_sir_val, si_sar_val = None, None, None

        data["pesq"].append(pesq_val)
        data["estoi"].append(estoi_val)
        data["si_sdr"].append(si_sdr_val)
        data["si_sir"].append(si_sir_val)
        data["si_sar"].append(si_sar_val)

    # Save results as DataFrame    
    df = pd.DataFrame(data)

    total = len(data["filename"])

    pesq_mean, pesq_std, pesq_count = valid_stats(data["pesq"])
    estoi_mean, estoi_std, estoi_count = valid_stats(data["estoi"])
    si_sdr_mean, si_sdr_std, si_sdr_count = valid_stats(data["si_sdr"])
    si_sir_mean, si_sir_std, si_sir_count = valid_stats(data["si_sir"])
    si_sar_mean, si_sar_std, si_sar_count = valid_stats(data["si_sar"])

    # Print results
    print(f"PESQ: {pesq_mean:.2f} ± {pesq_std:.2f} (N={pesq_count}/{total})")
    print(f"ESTOI: {estoi_mean:.2f} ± {estoi_std:.2f} (N={estoi_count}/{total})")
    print(f"SI-SDR: {si_sdr_mean:.1f} ± {si_sdr_std:.1f} (N={si_sdr_count}/{total})")
    print(f"SI-SIR: {si_sir_mean:.1f} ± {si_sir_std:.1f} (N={si_sir_count}/{total})")
    print(f"SI-SAR: {si_sar_mean:.1f} ± {si_sar_std:.1f} (N={si_sar_count}/{total})")

    print(f"PESQ success rate: {pesq_count/total:.2%}")
    print(f"ESTOI success rate: {estoi_count/total:.2%}")

    # Save average results to file
    # with open(join(args.enhanced_dir, "_avg_results.txt"), "w") as log:
        # log.write(...)
    log = open(join(args.enhanced_dir, "_avg_results.txt"), "w")
    log.write(f"PESQ: {pesq_mean:.2f} ± {pesq_std:.2f} (N={pesq_count}/{total})\n")
    log.write(f"ESTOI: {estoi_mean:.2f} ± {estoi_std:.2f} (N={estoi_count}/{total})\n")
    log.write(f"SI-SDR: {si_sdr_mean:.1f} ± {si_sdr_std:.1f} (N={si_sdr_count}/{total})\n")
    log.write(f"SI-SIR: {si_sir_mean:.1f} ± {si_sir_std:.1f} (N={si_sir_count}/{total})\n")
    log.write(f"SI-SAR: {si_sar_mean:.1f} ± {si_sar_std:.1f} (N={si_sar_count}/{total})\n")

    log.write(f"PESQ success rate: {pesq_count/total:.2%}\n")
    log.write(f"ESTOI success rate: {estoi_count/total:.2%}\n")

    # Save DataFrame as csv file
    df.to_csv(join(args.enhanced_dir, "_results.csv"), index=False)
