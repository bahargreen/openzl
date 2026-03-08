# -*- coding: utf-8 -*-
"""
Main benchmark script for:
- standard Struct(m-byte records)
- grouped / decomposed Struct-separated

Grouped / decomposed Struct-separated means:
    byte-plane regrouping
    -> each group becomes its own stream
    -> each stream is converted to Struct(group_len)
    -> each stream is compressed separately

This script reports:
- compression ratio
- core compression time / MB/s
- core decompression time / MB/s
- full restore time / MB/s
- full restore correctness

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

DATASET_FOLDER = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"
OUT_LOG_DIR = "/home/jamalids/Documents/8"

# type width: 2=float16, 4=float32, 8=float64
M = 4

# optional single-dataset filter
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
    "acs_wht_f32": [[[1, 2], [3], [4]], [[3], [1, 2], [4]], [[4], [3], [1, 2]], [[4], [3], [2, 1]]],
    "g24_78_usb2_f32": [[[1, 2, 3], [4]]],
    "jw_mirimage_f32": [[[1, 2, 3], [4]]],
    "spitzer_irac_f32": [[[1, 2], [3], [4]]],
    "turbulence_f32": [[[1, 2], [3], [4]]],
    "wave_f32": [[[1, 2], [3], [4]]],
    "hdr_night_f32": [[[1, 4], [2], [3]]],
    "ts_gas_f32": [[[1], [2, 3], [4]]],
    "solar_wind_f32": [[[1], [2, 3], [4]], [[1, 2, 3], [4]], [[1], [2], [3], [4]]],
    "tpch_lineitem_f32": [[[1, 2, 3], [4]]],
    "tpcds_web_f32": [[[4], [1, 2, 3]]],
    "tpcds_store_f32": [[[1, 2, 3], [4]]],
    "tpcds_catalog_f32": [[[1, 2], [3], [4]]],
    "citytemp_f32": [[[1, 2], [3], [4]], [[3], [1, 2], [4]], [[4], [3], [1, 2]], [[4], [3], [2, 1]]],
    "hst_wfc3_ir_f32": [[[1, 2], [3], [4]], [[1, 2, 3], [4]], [[1, 4], [3], [2]]],
    "hst_wfc3_uvis_f32": [[[1, 2], [3], [4]], [[1, 2, 3], [4]]],
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
# Local timing helpers
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


def _full_restore_standard_struct_time(
    tool,
    serial_bytes: np.ndarray,
    record_size: int,
    max_records: int,
) -> tuple[int, float, int, bool]:
    """
    Full restore time for standard Struct:
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
                raise RuntimeError("standard Struct full restore mismatch")

    dt = _median_time_local(timed_pass)
    return int(compressed_total), float(dt), int(nbytes_aligned), True


def _compress_one_group_chunked(
    tool,
    serial_bytes: np.ndarray,
    record_size: int,
    max_records: int,
):
    """
    Compress one separated group stream chunk-by-chunk.

    Returns:
        compressed_total, comp_time, nproc, decomp_time, dproc,
        compressed_chunks, chunk_record_counts, ok_core
    """
    size_c, t_c, nproc = sddlh._core_time_and_size_sddl_chunked(
        tool,
        serial_bytes,
        record_size=record_size,
        max_records=max_records,
    )

    _, t_d, dproc = sddlh._core_time_and_check_decompress_sddl_chunked(
        tool,
        serial_bytes,
        record_size=record_size,
        max_records=max_records,
    )

    nbytes = int(serial_bytes.size)
    nbytes_aligned = nbytes - (nbytes % record_size)
    if nbytes_aligned <= 0:
        return 0, 0.0, 0, 0.0, 0, [], [], False

    serial_bytes = serial_bytes[:nbytes_aligned]
    chunk_bytes = record_size * max_records

    compressed_chunks = []
    chunk_record_counts = []

    off = 0
    ok_core = True
    while off < nbytes_aligned:
        end = min(off + chunk_bytes, nbytes_aligned)
        end = end - ((end - off) % record_size)
        if end <= off:
            break

        chunk = serial_bytes[off:end]
        comp = tool(chunk)

        regen = sddlh._decompress_serial_bytes(comp)
        if regen != chunk.tobytes(order="C"):
            ok_core = False
            break

        compressed_chunks.append(comp)
        chunk_record_counts.append((end - off) // record_size)
        off = end

    return (
        int(size_c),
        float(t_c),
        int(nproc),
        float(t_d),
        int(dproc),
        compressed_chunks,
        chunk_record_counts,
        bool(ok_core),
    )


def _full_restore_grouped_struct_separate_time(
    group_tools,
    separated_group_streams,
    group_lens,
    decomp,
    expected_data: np.ndarray,
    max_records: int,
) -> tuple[int, float, int, bool]:
    """
    Full restore for separated Struct groups:
        for each group:
            decompress its own stream
        then:
            merge separated streams back to original float array
    """
    if len(group_tools) != len(separated_group_streams):
        raise ValueError("group_tools and separated_group_streams length mismatch")

    if len(group_tools) == 0:
        return 0, 0.0, 0, False

    all_group_chunks = []
    all_group_record_counts = []
    compressed_total = 0
    total_processed = 0

    for tool, group_stream, glen in zip(group_tools, separated_group_streams, group_lens):
        nbytes = int(group_stream.size)
        nbytes_aligned = nbytes - (nbytes % glen)
        if nbytes_aligned <= 0:
            return 0, 0.0, 0, False

        group_stream = group_stream[:nbytes_aligned]
        chunk_bytes = int(glen) * int(max_records)

        compressed_chunks = []
        chunk_record_counts = []

        off = 0
        while off < nbytes_aligned:
            end = min(off + chunk_bytes, nbytes_aligned)
            end = end - ((end - off) % glen)
            if end <= off:
                break

            chunk = group_stream[off:end]
            comp = tool(chunk)

            regen = sddlh._decompress_serial_bytes(comp)
            if regen != chunk.tobytes(order="C"):
                return 0, 0.0, 0, False

            compressed_chunks.append(comp)
            chunk_record_counts.append((end - off) // glen)
            off = end

        all_group_chunks.append(compressed_chunks)
        all_group_record_counts.append(chunk_record_counts)
        compressed_total += sum(len(x) for x in compressed_chunks)
        total_processed += nbytes_aligned

    n_chunk_parts = len(all_group_chunks[0])

    for x in all_group_chunks:
        if len(x) != n_chunk_parts:
            raise RuntimeError("Different number of chunks across groups")

    for x in all_group_record_counts:
        if len(x) != n_chunk_parts:
            raise RuntimeError("Different record-count list lengths across groups")

    for part_idx in range(n_chunk_parts):
        base = all_group_record_counts[0][part_idx]
        for g in range(1, len(group_lens)):
            if all_group_record_counts[g][part_idx] != base:
                raise RuntimeError("Chunk record counts mismatch across groups")

    def timed_pass():
        restored_parts = []

        for part_idx in range(n_chunk_parts):
            nrec = all_group_record_counts[0][part_idx]
            restored_group_serials = []

            for g in range(len(group_lens)):
                regen = sddlh._decompress_serial_bytes(all_group_chunks[g][part_idx])
                restored_group_serials.append(np.frombuffer(regen, dtype=np.byte).copy())

            x_part = sddlh._restore_separated_grouped_bytes_to_array(
                separated_group_serials=restored_group_serials,
                group_lens=group_lens,
                decomp=decomp,
                n=nrec,
                dtype=expected_data.dtype,
            )
            restored_parts.append(x_part)

        x_all = np.concatenate(restored_parts, axis=0)
        if not np.array_equal(x_all, expected_data):
            raise RuntimeError("grouped Struct-separated full restore mismatch")

    restored_parts = []
    for part_idx in range(n_chunk_parts):
        nrec = all_group_record_counts[0][part_idx]
        restored_group_serials = []

        for g in range(len(group_lens)):
            regen = sddlh._decompress_serial_bytes(all_group_chunks[g][part_idx])
            restored_group_serials.append(np.frombuffer(regen, dtype=np.byte).copy())

        x_part = sddlh._restore_separated_grouped_bytes_to_array(
            separated_group_serials=restored_group_serials,
            group_lens=group_lens,
            decomp=decomp,
            n=nrec,
            dtype=expected_data.dtype,
        )
        restored_parts.append(x_part)

    x_all = np.concatenate(restored_parts, axis=0)
    ok = np.array_equal(x_all, expected_data)

    if not ok:
        return int(compressed_total), 0.0, int(total_processed), False

    dt = _median_time_local(timed_pass)
    return int(compressed_total), float(dt), int(total_processed), True


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

    struct_cache = {}

    def get_struct_tool(width):
        width = int(width)
        if width not in struct_cache:
            struct_cache[width] = sddlh._OpenZLStructStandard(width)
        return struct_cache[width]

    def ratio(orig: int, comp: int) -> float:
        return float("inf") if comp == 0 else float(orig) / float(comp)

    std_serial_F = np.frombuffer(data_set.tobytes(), dtype=np.byte)

    # ------------------------------------------------------------
    # standard Struct(m-byte records), once per dataset
    # ------------------------------------------------------------
    standard_struct_tool = get_struct_tool(m)

    std_struct_size, std_struct_t, std_struct_nproc = sddlh._core_time_and_size_sddl_chunked(
        standard_struct_tool,
        std_serial_F,
        record_size=m,
        max_records=sddlh.SDDL_MAX_RECORDS_PER_CHUNK,
    )

    _, std_struct_dt, std_struct_dproc = sddlh._core_time_and_check_decompress_sddl_chunked(
        standard_struct_tool,
        std_serial_F,
        record_size=m,
        max_records=sddlh.SDDL_MAX_RECORDS_PER_CHUNK,
    )

    _, std_struct_full_t, std_struct_full_proc, std_struct_ok = _full_restore_standard_struct_time(
        standard_struct_tool,
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

        group_lens_expected = [len(g) for g in decomp]
        group_count = len(group_lens_expected)

        separated_group_streams, group_lens = sddlh._build_grouped_separate_bytes_col_fast(
            byte_planes=byte_planes,
            decomp=decomp,
        )

        if list(group_lens) != list(group_lens_expected):
            raise RuntimeError(
                f"group_lens mismatch for {dataset_name}: "
                f"helper={group_lens}, expected={group_lens_expected}"
            )

        # ------------------------------------------------------------
        # standard Struct
        # ------------------------------------------------------------
        stats["standard openzl_struct ratio"] = ratio(len_bytes, std_struct_size)
        stats["standard openzl_struct comp time s"] = float(std_struct_t)
        stats["standard openzl_struct core decomp time s"] = float(std_struct_dt)
        stats["standard openzl_struct full restore time s"] = float(std_struct_full_t)

        if sddlh.SDDL_MBPS_DENOM == "orig":
            stats["standard openzl_struct comp MB/s"] = sddlh._mbps(len_bytes, std_struct_t)
            stats["standard openzl_struct core decomp MB/s"] = sddlh._mbps(len_bytes, std_struct_dt)
            stats["standard openzl_struct full restore MB/s"] = sddlh._mbps(len_bytes, std_struct_full_t)
        else:
            stats["standard openzl_struct comp MB/s"] = sddlh._mbps(std_struct_nproc, std_struct_t)
            stats["standard openzl_struct core decomp MB/s"] = sddlh._mbps(std_struct_dproc, std_struct_dt)
            stats["standard openzl_struct full restore MB/s"] = sddlh._mbps(std_struct_full_proc, std_struct_full_t)

        stats["standard openzl_struct full restore correctness"] = bool(std_struct_ok)

        # ------------------------------------------------------------
        # grouped / decomposed Struct-separated
        # ------------------------------------------------------------
        group_tools = [get_struct_tool(glen) for glen in group_lens]

        total_comp_size = 0
        total_comp_time = 0.0
        total_decomp_time = 0.0
        total_nproc = 0
        total_dproc = 0
        group_core_ok = True

        per_group_sizes = []
        per_group_comp_times = []
        per_group_decomp_times = []

        for tool, group_stream, glen in zip(group_tools, separated_group_streams, group_lens):
            (
                g_size,
                g_tcomp,
                g_nproc,
                g_tdecomp,
                g_dproc,
                _chunks,
                _chunk_rec_counts,
                g_ok_core,
            ) = _compress_one_group_chunked(
                tool=tool,
                serial_bytes=group_stream,
                record_size=glen,
                max_records=sddlh.SDDL_MAX_RECORDS_PER_CHUNK,
            )

            total_comp_size += g_size
            total_comp_time += g_tcomp
            total_decomp_time += g_tdecomp
            total_nproc += g_nproc
            total_dproc += g_dproc
            group_core_ok = group_core_ok and g_ok_core

            per_group_sizes.append(int(g_size))
            per_group_comp_times.append(float(g_tcomp))
            per_group_decomp_times.append(float(g_tdecomp))

        _, full_restore_t, full_restore_proc, full_restore_ok = _full_restore_grouped_struct_separate_time(
            group_tools=group_tools,
            separated_group_streams=separated_group_streams,
            group_lens=group_lens,
            decomp=decomp,
            expected_data=data_set,
            max_records=sddlh.SDDL_MAX_RECORDS_PER_CHUNK,
        )

        stats["grouped openzl_struct separate ratio"] = ratio(len_bytes, total_comp_size)
        stats["grouped openzl_struct separate comp time s"] = float(total_comp_time)
        stats["grouped openzl_struct separate core decomp time s"] = float(total_decomp_time)
        stats["grouped openzl_struct separate full restore time s"] = float(full_restore_t)

        if sddlh.SDDL_MBPS_DENOM == "orig":
            stats["grouped openzl_struct separate comp MB/s"] = sddlh._mbps(len_bytes, total_comp_time)
            stats["grouped openzl_struct separate core decomp MB/s"] = sddlh._mbps(len_bytes, total_decomp_time)
            stats["grouped openzl_struct separate full restore MB/s"] = sddlh._mbps(len_bytes, full_restore_t)
        else:
            stats["grouped openzl_struct separate comp MB/s"] = sddlh._mbps(total_nproc, total_comp_time)
            stats["grouped openzl_struct separate core decomp MB/s"] = sddlh._mbps(total_dproc, total_decomp_time)
            stats["grouped openzl_struct separate full restore MB/s"] = sddlh._mbps(full_restore_proc, full_restore_t)

        stats["grouped openzl_struct separate full restore correctness"] = bool(group_core_ok and full_restore_ok)
        stats["grouped openzl_struct separate group count"] = int(group_count)
        stats["grouped openzl_struct separate group lens"] = list_to_string(group_lens)
        stats["grouped openzl_struct separate per-group compressed sizes"] = list_to_string(per_group_sizes)
        stats["grouped openzl_struct separate per-group comp times s"] = list_to_string(per_group_comp_times)
        stats["grouped openzl_struct separate per-group decomp times s"] = list_to_string(per_group_decomp_times)

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