import re
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/home/jamalids/Documents/6/6/rsim_f32_decomposition_stats.csv"

DATASET_FILTER = None          # e.g., "hdr_night_f32"
ORDER_PREF = "col"             # "col", "row", or None

EXCLUDE_PREFIXES = {
    "standard.struct_packed_4",
    "standard.struct_packed",
    "decomp.struct_packed_4.col",
    "decomp.struct_packed_4.row",
    "decomp.struct_packed.col",
    "decomp.struct_packed.row",
}

# Custom legend renaming: keys are prefixes (exact match), values are legend labels you want.
# Add as many as you like.
LEGEND_RENAME = {
    "standard.sddl_4": "SSDL",
    "standard.serial": "Serial",
    "standard.numeric": "Numeric",
    "standard.float_deconstruct_m4": "FloatDeconstruct",
    "standard.struct_4": "Struct ",
    "decomp.sddl.col": "TDT + SSDL",
    "decomp.sddl.row": "TDT + SSDL",
    "decomp.struct_groups.col": "TDT + struct",
    "decomp.serial.col": "TDT + serial"
}

OUT_PNG = "/home/jamalids/Documents/6/rsim_f32_ratio-through3.png"


def _legend_name(prefix: str) -> str:
    prefix = prefix.strip()

    # exact-match override first
    if prefix in LEGEND_RENAME:
        return LEGEND_RENAME[prefix]

    # default naming
    if prefix.startswith("decomp."):
        base = prefix[len("decomp."):]            # e.g., "sddl.col"
        base = base.replace(".", " ")             # e.g., "sddl col"
        return "TDT + " + base.upper() if base.startswith("sddl") else "TDT + " + base

    return prefix


def find_ratio_throughput_pairs(columns):
    cols = list(columns)
    colset = set(cols)
    pairs = []
    for c in cols:
        if not c.endswith(".ratio"):
            continue
        prefix = c[:-len(".ratio")]
        thr = f"{prefix}.MBps"
        if thr in colset:
            pairs.append((c, thr, prefix))
    return pairs


def main():
    df = pd.read_csv(CSV_PATH)

    if "dataset" not in df.columns:
        raise KeyError("Expected a 'dataset' column in the CSV.")

    dataset = df["dataset"].iloc[0] if DATASET_FILTER is None else DATASET_FILTER
    sub = df[df["dataset"] == dataset].copy()
    if sub.empty:
        raise ValueError(f"No rows found for dataset={dataset!r}")

    pairs = find_ratio_throughput_pairs(sub.columns)

    # keep only one order for decomposed if requested
    if ORDER_PREF in ("col", "row"):
        kept = []
        for rcol, tcol, prefix in pairs:
            if prefix.startswith("decomp."):
                parts = prefix.split(".")
                # expects: decomp.<method>.<col|row>
                if len(parts) >= 3 and parts[-1] == ORDER_PREF:
                    kept.append((rcol, tcol, prefix))
            else:
                kept.append((rcol, tcol, prefix))
        pairs = kept

    # exclude specific prefixes
    pairs = [(r, t, p) for (r, t, p) in pairs if p not in EXCLUDE_PREFIXES]

    plt.figure()
    ax = plt.gca()

    for ratio_col, thr_col, prefix in pairs:
        x = pd.to_numeric(sub[ratio_col], errors="coerce")
        y = pd.to_numeric(sub[thr_col], errors="coerce")
        mask = x.notna() & y.notna()
        if not mask.any():
            continue
        ax.scatter(x[mask].values, y[mask].values, label=_legend_name(prefix))

    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Throughput (MBps)")
    ax.set_title(f"Compression ratio vs throughput ({dataset})")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )

    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"[OK] Wrote: {OUT_PNG}")


if __name__ == "__main__":
    main()