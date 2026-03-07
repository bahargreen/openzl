# # make_train_sample_serial.py
# import os, glob
# import numpy as np
# import pandas as pd
#
# IN_DIR = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"
# OUT_DIR = "/home/jamalids/Documents/6/train_samples_serial"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# for p in sorted(glob.glob(os.path.join(IN_DIR, "*.tsv"))):
#     name = os.path.splitext(os.path.basename(p))[0]
#     df = pd.read_csv(p, sep="\t")
#     x = df.values[:, 1].astype(np.float32, copy=False)
#     out = os.path.join(OUT_DIR, f"{name}.f32.raw")  # raw float32 bytes
#     x.tofile(out)
#
# print("Wrote:", OUT_DIR)
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import glob

FULL_DIR = "/home/jamalids/Documents/6/train_samples_serial"          # full .f32.raw files
SAMPLE_DIR = "/home/jamalids/Documents/6/train_samples_serial_small"  # per-dataset samples
MAX_BYTES = 2 * 1024 * 1024  # 2 MiB

os.makedirs(SAMPLE_DIR, exist_ok=True)

n = 0
for p in sorted(glob.glob(os.path.join(FULL_DIR, "*.f32.raw"))):
    base = os.path.basename(p)

    # Keep extension as .f32.raw so zli train doesn't skip files
    out = os.path.join(SAMPLE_DIR, base)

    with open(p, "rb") as fin:
        chunk = fin.read(MAX_BYTES)

    if not chunk:
        print(f"[SKIP] empty: {base}")
        continue

    with open(out, "wb") as fout:
        fout.write(chunk)

    n += 1
    print(f"[OK] {base} -> {os.path.basename(out)} ({len(chunk)} bytes)")

print(f"Wrote {n} sample files to: {SAMPLE_DIR}")