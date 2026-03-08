import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ====== change this ======
INPUT_DIR = Path("/home/jamalids/Documents/8")
PATTERN = "*_decomposition_stats.csv"

OUT_CSV = INPUT_DIR / "merged_standard_vs_tdt.csv"
OUT_GMEAN_CSV = INPUT_DIR / "gmean_standard_vs_tdt.csv"

OUT_RATIO_PNG = INPUT_DIR / "plot_ratio_by_dataset.png"
OUT_COMP_PNG = INPUT_DIR / "plot_compression_throughput_by_dataset.png"
OUT_DECOMP_PNG = INPUT_DIR / "plot_decompression_throughput_by_dataset.png"
# =========================


def pick_col(cols, must_have, must_not_have=None):
    must_not_have = must_not_have or []
    for c in cols:
        lc = c.lower()
        if all(x in lc for x in must_have) and not any(x in lc for x in must_not_have):
            return c
    return None


def geometric_mean(values):
    vals = pd.Series(values).dropna().astype(float)
    vals = vals[vals > 0]
    if len(vals) == 0:
        return np.nan
    return float(np.exp(np.log(vals).mean()))


records = []

for csv_path in sorted(INPUT_DIR.glob(PATTERN)):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    dataset_col = pick_col(cols, ["dataset", "name"]) or "dataset name"

    ratio_std_col = pick_col(cols, ["standard", "ratio"])
    comp_std_col = pick_col(cols, ["standard", "comp", "mb/s"])
    decomp_std_col = pick_col(cols, ["standard", "core", "decomp", "mb/s"])

    ratio_tdt_col = pick_col(cols, ["grouped", "ratio"])
    comp_tdt_col = pick_col(cols, ["grouped", "comp", "mb/s"])
    decomp_tdt_col = pick_col(cols, ["grouped", "core", "decomp", "mb/s"])

    if not all([ratio_std_col, comp_std_col, decomp_std_col]):
        print(f"Skip {csv_path.name}: missing standard columns")
        continue

    if not all([ratio_tdt_col, comp_tdt_col, decomp_tdt_col]):
        print(f"Skip {csv_path.name}: missing grouped/TDT columns")
        continue

    for _, row in df.iterrows():
        dataset = row.get(dataset_col, csv_path.stem)

        records.append({
            "dataset": dataset,
            "variant": "Standard",
            "ratio": row[ratio_std_col],
            "compression_throughput_MBps": row[comp_std_col],
            "decompression_throughput_MBps": row[decomp_std_col],
            "source_file": csv_path.name
        })

        records.append({
            "dataset": dataset,
            "variant": "TDT",
            "ratio": row[ratio_tdt_col],
            "compression_throughput_MBps": row[comp_tdt_col],
            "decompression_throughput_MBps": row[decomp_tdt_col],
            "source_file": csv_path.name
        })

merged = pd.DataFrame(records)

if merged.empty:
    raise ValueError("No valid rows found. Check input files and column names.")

merged.to_csv(OUT_CSV, index=False)
print(f"Saved merged CSV: {OUT_CSV}")


# -------- gmean table --------
gmean_df = (
    merged.groupby("variant", as_index=False)
    .agg(
        gmean_ratio=("ratio", geometric_mean),
        gmean_compression_throughput_MBps=("compression_throughput_MBps", geometric_mean),
        gmean_decompression_throughput_MBps=("decompression_throughput_MBps", geometric_mean),
    )
)

gmean_df.to_csv(OUT_GMEAN_CSV, index=False)
print("\nGeometric mean summary:")
print(gmean_df.to_string(index=False))
print(f"\nSaved gmean CSV: {OUT_GMEAN_CSV}")


# -------- dataset-level tables for plotting --------
ratio_pivot = merged.pivot_table(
    index="dataset", columns="variant", values="ratio", aggfunc="first"
).reset_index()

comp_pivot = merged.pivot_table(
    index="dataset", columns="variant", values="compression_throughput_MBps", aggfunc="first"
).reset_index()

decomp_pivot = merged.pivot_table(
    index="dataset", columns="variant", values="decompression_throughput_MBps", aggfunc="first"
).reset_index()

# keep stable order
dataset_order = sorted(merged["dataset"].dropna().unique())
ratio_pivot = ratio_pivot.set_index("dataset").reindex(dataset_order).reset_index()
comp_pivot = comp_pivot.set_index("dataset").reindex(dataset_order).reset_index()
decomp_pivot = decomp_pivot.set_index("dataset").reindex(dataset_order).reset_index()


def plot_two_bar(df, ycols, title, ylabel, out_png):
    x = np.arange(len(df))
    width = 0.36

    plt.figure(figsize=(max(10, len(df) * 0.7), 6))
    plt.bar(x - width/2, df[ycols[0]], width=width, alpha=0.45, label="Standard")
    plt.bar(x + width/2, df[ycols[1]], width=width, alpha=0.90, label="TDT")

    plt.xticks(x, df["dataset"], rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel("Dataset")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_png}")


# -------- plots --------
plot_two_bar(
    ratio_pivot,
    ycols=["Standard", "TDT"],
    title="Compression Ratio by Dataset",
    ylabel="Compression Ratio",
    out_png=OUT_RATIO_PNG
)

plot_two_bar(
    comp_pivot,
    ycols=["Standard", "TDT"],
    title="Compression Throughput by Dataset",
    ylabel="Compression Throughput (MB/s)",
    out_png=OUT_COMP_PNG
)

plot_two_bar(
    decomp_pivot,
    ycols=["Standard", "TDT"],
    title="Decompression Throughput by Dataset",
    ylabel="Decompression Throughput (MB/s)",
    out_png=OUT_DECOMP_PNG
)