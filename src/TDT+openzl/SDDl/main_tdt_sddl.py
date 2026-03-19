# -*- coding: utf-8 -*-
"""
Main benchmark script for:
- standard SDDL
- grouped / decomposed SDDL

This script reports:
- compression ratio
- core compression time / MB/s
- core decompression time / MB/s
- full restore time / MB/s
- full restore correctness

Fast-path logic:
- first checks whether decomposition preserves original byte order
- if no reorder is needed, uses lightweight path
- otherwise falls back to grouped/repacked path

Designed to work with:
    openzl_sddl_helpers.py
"""

import os
import time
import numpy as np
import pandas as pd

from utils import (
    tuple_to_string,
    list_to_string,
)

try:
    import openzl.ext as zl
    _HAS_OPENZL = True
except Exception:
    zl = None
    _HAS_OPENZL = False

import openzl_sddl_helpers as sddlh


# ============================================================
# CONFIG
# ============================================================

DATASET_FOLDER = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/34"
OUT_LOG_DIR = "/home/jamalids/Documents/7"

# type width: 2=float16, 4=float32, 8=float64
M = 4

# optional single-dataset filter, e.g. "citytemp_f32"
ONLY_DATASET = None

# helper configs
sddlh.N_WARMUP = 1
sddlh.N_RUNS = 10
sddlh.SDDL_MAX_RECORDS_PER_CHUNK = 2_000_000
sddlh.SDDL_MBPS_DENOM = "orig"   # "aligned" or "orig"


# ============================================================
# Byte Grouped (Static Clustering)
# ============================================================

compConfigMap = {
    "acs_wht_f32": [[[1, 2], [3], [4]], [[3], [1, 2], [4]],[[4], [3], [1,2]],[[4], [3], [2,1]]],
    "g24_78_usb2_f32": [[[1, 2, 3], [4]]],
    "jw_mirimage_f32": [[[1, 2, 3], [4]]],
    "spitzer_irac_f32": [[[1, 2], [3], [4]]],
    "turbulence_f32": [[[1, 2], [3], [4]]],
    "wave_f32": [[[1, 2], [3], [4]]],
    "hdr_night_f32": [[[1, 4], [2], [3]]],
    "ts_gas_f32": [[[1], [2, 3], [4]]],
    "solar_wind_f32": [[[1], [2, 3], [4]], [[1,2,3], [4]],[[1], [2], [3], [4]]],
    "tpch_lineitem_f32": [[[1, 2, 3], [4]]],
    "tpcds_web_f32": [[[4], [1, 2, 3]]],
    "tpcds_store_f32": [[[1, 2, 3], [4]]],
    "tpcds_catalog_f32": [[[1, 2], [3], [4]]],
    "citytemp_f32": [[[1, 2], [3], [4]], [[3], [1, 2], [4]],[[4], [3], [1,2]],[[4], [3], [2,1]]],
    "hst_wfc3_ir_f32": [[[1, 2], [3], [4]],[[1, 2, 3], [4]]],
    "hst_wfc3_uvis_f32": [[[1, 2], [3], [4]],[[1, 2, 3], [4]]],
    "rsim_f32": [[[1, 2], [3], [4]]],
    "astro_mhd_f64": [[[1, 2, 3, 4, 5, 6], [7], [8]]],
    "astro_pt_f64": [[[1, 2, 3, 4, 5, 6], [7], [8]]],
    "jane_street_f64": [[[1, 2, 3, 4, 5, 6], [7], [8]]],
    "msg_bt_f64": [[[1, 2, 3, 4, 5], [6], [7], [8]]],
    "num_brain_f64": [[[3, 2, 4, 5, 1, 6], [7], [8]]],
    "num_control_f64": [[[1, 2, 3, 4, 5, 6], [7], [8]]],
    "nyc_taxi2015_f64": [[[7, 4, 6], [5], [3, 2, 1, 8]]],
    "phone_gyro_f64": [[[4, 6], [8], [3, 2, 1, 7], [5]]],
    "tpch_order_f64": [[[1, 2, 3, 4], [7], [6, 5], [8]]],
    "tpcxbb_store_f64": [[[4, 2, 3], [1], [5], [7], [6], [8]]],
    "tpcxbb_web_f64": [[[4, 2, 3], [1], [5], [7], [6], [8]]],
    "wesad_chest_f64": [[[7, 5, 6], [8, 4, 1, 3, 2]]],
    "default": [[[1], [2], [3], [4]]],
}


# ============================================================
# Local timing helpers for full restore
# ============================================================

def _median_time_local(callable_fn) -> float:
    for _ in range(sddlh.N_WARMUP):
        callable_fn()

    times = []
    for _ in range(sddlh.N_RUNS):
        t0 = time.perf_counter()
        callable_fn()
        times.append(time.perf_counter() - t0)

    return float(np.median(np.asarray(times, dtype=np.float64)))


def _full_restore_standard_sddl_time(
    tool,
    serial_bytes: np.ndarray,
    record_size: int,
    max_records: int,
) -> tuple[int, float, int, bool]:
    """
    Full restore time for standard SDDL:
        decompress -> verify restored serial bytes
    """
    record_size = int(record_size)
    max_records = int(max_records)

    nbytes = int(serial_bytes.size)
    nbytes_aligned = nbytes - (nbytes % record_size)
    if nbytes_aligned <= 0:
        return 0, 0.0, 0, False

    serial_bytes = serial_bytes[:nbytes_aligned]
    chunk_bytes = record_size * max_records

    compressed_chunks = []
    expected_chunks = []

    off = 0
    while off < nbytes_aligned:
        end = min(off + chunk_bytes, nbytes_aligned)
        end = end - ((end - off) % record_size)
        if end <= off:
            break

        chunk = serial_bytes[off:end]
        comp = tool(chunk)

        regen = sddlh._decompress_serial_bytes(comp)
        if regen != chunk.tobytes(order="C"):
            return 0, 0.0, nbytes_aligned, False

        compressed_chunks.append(comp)
        expected_chunks.append(chunk)
        off = end

    compressed_total = sum(len(x) for x in compressed_chunks)

    def timed_pass():
        for comp, exp in zip(compressed_chunks, expected_chunks):
            regen = sddlh._decompress_serial_bytes(comp)
            if regen != exp.tobytes(order="C"):
                raise RuntimeError("standard SDDL full restore mismatch")

    dt = _median_time_local(timed_pass)
    return int(compressed_total), float(dt), int(nbytes_aligned), True


def _full_restore_grouped_sddl_time(
    tool,
    packed_bytes_F: np.ndarray,
    group_lens,
    decomp,
    expected_data: np.ndarray,
    record_size: int,
    max_records: int,
    no_reorder_layout: bool,
) -> tuple[int, float, int, bool]:
    """
    Full restore time for grouped/decomposed SDDL:
        decompress -> restore grouped bytes -> reconstruct float -> verify
    """
    record_size = int(record_size)
    max_records = int(max_records)

    nbytes = int(packed_bytes_F.size)
    nbytes_aligned = nbytes - (nbytes % record_size)
    if nbytes_aligned <= 0:
        return 0, 0.0, 0, False

    packed_bytes_F = packed_bytes_F[:nbytes_aligned]
    chunk_bytes = record_size * max_records

    compressed_chunks = []
    chunk_record_counts = []

    off = 0
    while off < nbytes_aligned:
        end = min(off + chunk_bytes, nbytes_aligned)
        end = end - ((end - off) % record_size)
        if end <= off:
            break

        chunk = packed_bytes_F[off:end]
        comp = tool(chunk)

        regen = sddlh._decompress_serial_bytes(comp)
        if regen != chunk.tobytes(order="C"):
            return 0, 0.0, nbytes_aligned, False

        compressed_chunks.append(comp)
        chunk_record_counts.append((end - off) // record_size)
        off = end

    compressed_total = sum(len(x) for x in compressed_chunks)

    def timed_pass():
        restored_parts = []
        for comp, nrec in zip(compressed_chunks, chunk_record_counts):
            regen = sddlh._decompress_serial_bytes(comp)
            x_part = sddlh._restore_grouped_bytes_to_array(
                packed_serial=regen,
                group_lens=group_lens,
                decomp=decomp,
                n=nrec,
                dtype=expected_data.dtype,
                no_reorder_layout=no_reorder_layout,
            )
            restored_parts.append(x_part)

        x_all = np.concatenate(restored_parts, axis=0)
        if not np.array_equal(x_all, expected_data):
            raise RuntimeError("grouped SDDL full restore mismatch")

    restored_parts = []
    for comp, nrec in zip(compressed_chunks, chunk_record_counts):
        regen = sddlh._decompress_serial_bytes(comp)
        x_part = sddlh._restore_grouped_bytes_to_array(
            packed_serial=regen,
            group_lens=group_lens,
            decomp=decomp,
            n=nrec,
            dtype=expected_data.dtype,
            no_reorder_layout=no_reorder_layout,
        )
        restored_parts.append(x_part)

    x_all = np.concatenate(restored_parts, axis=0)
    ok = np.array_equal(x_all, expected_data)

    if not ok:
        return int(compressed_total), 0.0, int(nbytes_aligned), False

    dt = _median_time_local(timed_pass)
    return int(compressed_total), float(dt), int(nbytes_aligned), True


# ============================================================
# Main evaluation
# ============================================================

def test_decomposition(
    data_set: np.ndarray,
    dataset_name: str,
    given_decomp,
    m=4,
    chunk_no=-1,
    out_log_dir="/home/jamalids/Documents/1",
):
    m = int(m)

    if not given_decomp:
        raise ValueError(f"No decomposition config found for dataset '{dataset_name}'")

    all_decomps = list(given_decomp)
    total_enum = len(all_decomps)

    len_bytes = int(data_set.nbytes)
    n = int(len(data_set))

    byte_planes = sddlh._byte_planes_view(data_set, m)

    sddl_groups_cache = {}
    sddl_standard_cache = {}

    def get_sddl_groups_tool(group_lens):
        key = tuple(int(x) for x in group_lens)
        if key not in sddl_groups_cache:
            sddl_groups_cache[key] = sddlh._OpenZLSDDLGroups(key)
        return sddl_groups_cache[key]

    def get_standard_sddl_tool(m_):
        m_ = int(m_)
        if m_ not in sddl_standard_cache:
            sddl_standard_cache[m_] = sddlh._OpenZLSDDLStandard(m_)
        return sddl_standard_cache[m_]

    def ratio(orig: int, comp: int) -> float:
        return float("inf") if comp == 0 else float(orig) / float(comp)

    std_serial_F = np.frombuffer(data_set.tobytes(), dtype=np.byte)

    # standard SDDL
    standard_sddl_tool = get_standard_sddl_tool(m)

    std_sddl_size, std_sddl_t, std_sddl_nproc = sddlh._core_time_and_size_sddl_chunked(
        standard_sddl_tool,
        std_serial_F,
        record_size=m,
        max_records=sddlh.SDDL_MAX_RECORDS_PER_CHUNK,
    )

    _, std_sddl_dt, std_sddl_dproc = sddlh._core_time_and_check_decompress_sddl_chunked(
        standard_sddl_tool,
        std_serial_F,
        record_size=m,
        max_records=sddlh.SDDL_MAX_RECORDS_PER_CHUNK,
    )

    _, std_sddl_full_t, std_sddl_full_proc, std_ok = _full_restore_standard_sddl_time(
        standard_sddl_tool,
        std_serial_F,
        record_size=m,
        max_records=sddlh.SDDL_MAX_RECORDS_PER_CHUNK,
    )

    stat_array = []

    for idx, decomp in enumerate(all_decomps):
        decomp = tuple(tuple(int(x) for x in g) for g in decomp)

        stats = {
            "dataset name": dataset_name,
            "original size": len_bytes,
            "type width": m,
            "Dimension": n,
            "decomposition": tuple_to_string(decomp),
            "chunk no": int(chunk_no),
        }

        no_reorder_layout = sddlh._is_no_reorder_decomp(decomp, m)

        group_lens = [len(g) for g in decomp]
        K = int(sum(group_lens))

        if no_reorder_layout:
            packed_bytes_F = std_serial_F
        else:
            packed_bytes_F, group_lens, K = sddlh._build_grouped_packed_bytes_col_fast(
                byte_planes=byte_planes,
                decomp=decomp,
            )

        # standard results
        stats["standard openzl_sddl ratio"] = ratio(len_bytes, std_sddl_size)
        stats["standard openzl_sddl comp time s"] = float(std_sddl_t)
        stats["standard openzl_sddl core decomp time s"] = float(std_sddl_dt)
        stats["standard openzl_sddl full restore time s"] = float(std_sddl_full_t)

        if sddlh.SDDL_MBPS_DENOM == "orig":
            stats["standard openzl_sddl comp MB/s"] = sddlh._mbps(len_bytes, std_sddl_t)
            stats["standard openzl_sddl core decomp MB/s"] = sddlh._mbps(len_bytes, std_sddl_dt)
            stats["standard openzl_sddl full restore MB/s"] = sddlh._mbps(len_bytes, std_sddl_full_t)
        else:
            stats["standard openzl_sddl comp MB/s"] = sddlh._mbps(std_sddl_nproc, std_sddl_t)
            stats["standard openzl_sddl core decomp MB/s"] = sddlh._mbps(std_sddl_dproc, std_sddl_dt)
            stats["standard openzl_sddl full restore MB/s"] = sddlh._mbps(std_sddl_full_proc, std_sddl_full_t)

        stats["standard openzl_sddl full restore correctness"] = bool(std_ok)

        # grouped / decomposed SDDL
        sddl_tool = get_sddl_groups_tool(group_lens)

        sddl_size_F, t_sddl_F, nproc_F = sddlh._core_time_and_size_sddl_chunked(
            sddl_tool,
            packed_bytes_F,
            record_size=K,
            max_records=sddlh.SDDL_MAX_RECORDS_PER_CHUNK,
        )

        _, t_sddl_d_F, dproc_F = sddlh._core_time_and_check_decompress_sddl_chunked(
            sddl_tool,
            packed_bytes_F,
            record_size=K,
            max_records=sddlh.SDDL_MAX_RECORDS_PER_CHUNK,
        )

        _, t_sddl_full_F, fullproc_F, ok_F = _full_restore_grouped_sddl_time(
            sddl_tool,
            packed_bytes_F=packed_bytes_F,
            group_lens=group_lens,
            decomp=decomp,
            expected_data=data_set,
            record_size=K,
            max_records=sddlh.SDDL_MAX_RECORDS_PER_CHUNK,
            no_reorder_layout=no_reorder_layout,
        )

        stats["grouped openzl_sddl col-order ratio"] = ratio(len_bytes, sddl_size_F)
        stats["grouped openzl_sddl col-order comp time s"] = float(t_sddl_F)
        stats["grouped openzl_sddl col-order core decomp time s"] = float(t_sddl_d_F)
        stats["grouped openzl_sddl col-order full restore time s"] = float(t_sddl_full_F)

        if sddlh.SDDL_MBPS_DENOM == "orig":
            stats["grouped openzl_sddl col-order comp MB/s"] = sddlh._mbps(len_bytes, t_sddl_F)
            stats["grouped openzl_sddl col-order core decomp MB/s"] = sddlh._mbps(len_bytes, t_sddl_d_F)
            stats["grouped openzl_sddl col-order full restore MB/s"] = sddlh._mbps(len_bytes, t_sddl_full_F)
        else:
            stats["grouped openzl_sddl col-order comp MB/s"] = sddlh._mbps(nproc_F, t_sddl_F)
            stats["grouped openzl_sddl col-order core decomp MB/s"] = sddlh._mbps(dproc_F, t_sddl_d_F)
            stats["grouped openzl_sddl col-order full restore MB/s"] = sddlh._mbps(fullproc_F, t_sddl_full_F)

        stats["grouped openzl_sddl col-order full restore correctness"] = bool(ok_F)
        stats["grouped openzl_sddl uses no-reorder fast path"] = bool(no_reorder_layout)
        stats["grouped openzl_sddl group count"] = int(len(group_lens))
        stats["grouped openzl_sddl total packed width"] = int(K)
        stats["grouped openzl_sddl group lens"] = list_to_string(group_lens)

        stat_array.append(stats)

        if out_log_dir and (((idx + 1) % 20 == 0) or ((idx + 1) == total_enum)):
            os.makedirs(out_log_dir, exist_ok=True)
            out_csv = os.path.join(out_log_dir, f"{dataset_name}_decomposition_stats.csv")
            pd.DataFrame(stat_array).to_csv(out_csv, index=False)

    return stat_array


# ============================================================
# Main
# ============================================================

def main():
    dataset_folder = DATASET_FOLDER
    m = M
    out_log_dir = OUT_LOG_DIR

    if not os.path.isdir(dataset_folder):
        print(f"Error: {dataset_folder} is not a valid directory.")
        return

    if not _HAS_OPENZL:
        print("[WARN] OpenZL not available; cannot run.")
        return

    for file_name in sorted(os.listdir(dataset_folder)):
        dataset_path = os.path.join(dataset_folder, file_name)
        if (not os.path.isfile(dataset_path)) or (not dataset_path.endswith(".tsv")):
            continue

        dataset_name = os.path.splitext(file_name)[0]

        if ONLY_DATASET is not None and dataset_name != ONLY_DATASET:
            continue

        print(f"Processing Dataset: {dataset_name}")

        data_df = pd.read_csv(dataset_path, sep="\t")

        if m == 2:
            sliced_data = data_df.values[:, 1].astype(np.float16)
        elif m == 4:
            sliced_data = data_df.values[:, 1].astype(np.float32)
        elif m == 8:
            sliced_data = data_df.values[:, 1].astype(np.float64)
        else:
            raise ValueError(f"Unsupported type width m={m}. Use 2, 4, or 8.")

        dataset_configs = compConfigMap.get(dataset_name, compConfigMap["default"])

        converted_decomps = []
        for grouping in dataset_configs:
            zero_based_groups = []
            for group in grouping:
                zero_based_groups.append(tuple(int(g) - 1 for g in group))
            converted_decomps.append(tuple(zero_based_groups))

        _ = test_decomposition(
            data_set=sliced_data,
            dataset_name=dataset_name,
            m=m,
            given_decomp=converted_decomps,
            out_log_dir=out_log_dir,
        )

        print(f"Done: wrote stats in {os.path.join(out_log_dir, f'{dataset_name}_decomposition_stats.csv')}")


if __name__ == "__main__":
    main()