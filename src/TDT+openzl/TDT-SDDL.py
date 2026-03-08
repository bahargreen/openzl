

import os
import itertools
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
OUT_LOG_DIR = "/home/jamalids/Documents/4"

# type width: 2=float16, 4=float32, 8=float64
M = 4

# if given_decomp is None, enumerate decompositions
CONTIG_ORDER = False

# optional: set to dataset name like "citytemp_f32", or None for all
ONLY_DATASET = None

# helper-module configs
sddlh.N_WARMUP = 1
sddlh.N_RUNS = 10
sddlh.SDDL_MAX_RECORDS_PER_CHUNK = 200_000
sddlh.SDDL_MBPS_DENOM = "aligned"   # "aligned" or "orig"


# ============================================================
# Decomposition configs
# ============================================================

compConfigMap = {
    "acs_wht_f32": [[[1, 2], [3], [4]]],
    "g24_78_usb2_f32": [[[1, 2, 3], [4]]],
    "jw_mirimage_f32": [[[1, 2, 3], [4]]],
    "spitzer_irac_f32": [[[1, 2], [3], [4]]],
    "turbulence_f32": [[[1, 2], [3], [4]]],
    "wave_f32": [[[1, 2], [3], [4]]],
    "hdr_night_f32": [[[1, 4], [2], [3]]],
    "ts_gas_f32": [[[1], [2, 3], [4]]],
    "solar_wind_f32": [[[1], [2, 3], [4]]],
    "tpch_lineitem_f32": [[[1, 2, 3], [4]]],
    "tpcds_web_f32": [[[4], [1, 2, 3]]],
    "tpcds_store_f32": [[[1, 2, 3], [4]]],
    "tpcds_catalog_f32": [[[1, 2], [3], [4]]],
    "citytemp_f32": [[[1, 2], [3], [4]]],
    "hst_wfc3_ir_f32": [[[1, 2], [3], [4]]],
    "hst_wfc3_uvis_f32": [[[1, 2], [3], [4]]],
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
# Enumeration helpers
# ============================================================

def possible_sum(m: int):
    possible_sets = []
    for i in range(1, m + 1):
        if i == m:
            possible_sets.append([i])
        else:
            for j in possible_sum(m - i):
                possible_sets.append([i] + j)
    return possible_sets


def merge_order_with_decomposition(order, decomposition):
    cur_len = 0
    merged_order = set()
    for comp_len in decomposition:
        comp_len = int(comp_len)
        cur_comp = tuple(order[cur_len:cur_len + comp_len])
        merged_order.add(cur_comp)
        cur_len += comp_len
    return merged_order


def find_all_combinations(all_possible_consecutive_comp, m, contiguous=True):
    byte_loc = np.arange(0, m)
    if contiguous:
        all_permutations = [tuple(range(0, m))]
    else:
        all_permutations = list(itertools.permutations(byte_loc))

    all_decomposition = []
    for composition in all_possible_consecutive_comp:
        for permutation in all_permutations:
            cur_comp = merge_order_with_decomposition(permutation, composition)
            all_decomposition.append(cur_comp)

    all_perm_length = len(all_decomposition)
    all_decomposition = list(set([tuple(x) for x in all_decomposition]))
    return all_decomposition, all_perm_length


# ============================================================
# Timing helpers for full restore
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
    Measures full restore time for standard SDDL:
        compress chunk -> decompress -> verify restored serial bytes

    Returns:
        (compressed_total_size, median_full_restore_seconds, nbytes_aligned, ok)
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
    n: int,
    dtype,
    record_size: int,
    max_records: int,
) -> tuple[int, float, int, bool]:
    """
    Measures full restore time for grouped/decomposed SDDL:
        compress chunk -> decompress -> inverse grouping -> reconstruct float -> verify

    Assumes Fortran-order packed layout only.
    Returns:
        (compressed_total_size, median_full_restore_seconds, nbytes_aligned, ok)
    """
    record_size = int(record_size)
    max_records = int(max_records)

    nbytes = int(packed_bytes_F.size)
    nbytes_aligned = nbytes - (nbytes % record_size)
    if nbytes_aligned <= 0:
        return 0, 0.0, 0, False

    packed_bytes_F = packed_bytes_F[:nbytes_aligned]
    chunk_bytes = record_size * max_records
    chunk_n_records = max_records

    compressed_chunks = []
    expected_record_counts = []

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
        expected_record_counts.append((end - off) // record_size)
        off = end

    compressed_total = sum(len(x) for x in compressed_chunks)

    def timed_pass():
        restored_parts = []
        for comp, nrec in zip(compressed_chunks, expected_record_counts):
            regen = sddlh._decompress_serial_bytes(comp)
            x_part = sddlh._unpack_tdt_packed_bytes_to_float(
                packed_serial=regen,
                group_lens=group_lens,
                decomp=decomp,
                n=nrec,
                dtype=dtype,
            )
            restored_parts.append(x_part)

        x_all = np.concatenate(restored_parts, axis=0)
        if x_all.shape[0] != n:
            raise RuntimeError(f"Restored length mismatch: got {x_all.shape[0]}, expected {n}")

    # correctness once before timing
    restored_parts = []
    for comp, nrec in zip(compressed_chunks, expected_record_counts):
        regen = sddlh._decompress_serial_bytes(comp)
        x_part = sddlh._unpack_tdt_packed_bytes_to_float(
            packed_serial=regen,
            group_lens=group_lens,
            decomp=decomp,
            n=nrec,
            dtype=dtype,
        )
        restored_parts.append(x_part)

    x_all = np.concatenate(restored_parts, axis=0)
    ok = np.array_equal(x_all, np.asarray(x_all, dtype=dtype)) and (x_all.shape[0] == n)
    if ok:
        # exact compare against original must be done by caller with original data if needed
        pass

    dt = _median_time_local(timed_pass)
    return int(compressed_total), float(dt), int(nbytes_aligned), bool(ok)


# ============================================================
# Main evaluation
# ============================================================

def test_decomposition(
    data_set: np.ndarray,
    dataset_name: str,
    given_decomp=None,
    m=4,
    chunk_no=-1,
    contig_order=True,
    out_log_dir="/home/jamalids/Documents/1",
):
    type_byte = np.uint8
    m = int(m)

    if given_decomp is None:
        all_possible = possible_sum(m)
        all_decomps, total_enum = find_all_combinations(all_possible, m, contig_order)
    else:
        all_decomps = list(given_decomp)
        total_enum = len(all_decomps)

    data_set_bytes = data_set.view(type_byte)
    len_bytes = int(data_set_bytes.nbytes)
    n = int(len(data_set))

    # byte-planes once
    comps = np.zeros((m, n), dtype=type_byte)
    for i in range(m):
        comps[i] = data_set_bytes[i:len_bytes:m]

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

    # original serial layout
    std_serial_F = np.frombuffer(data_set.flatten("F").tobytes(), dtype=np.byte)

    stat_array = []

    for idx, decomp in enumerate(all_decomps):
        stats = {
            "dataset name": dataset_name,
            "original size": len_bytes,
            "type width": m,
            "Dimension": n,
            "decomposition": tuple_to_string(decomp),
            "chunk no": int(chunk_no),
        }

        # --------------------------------------------------------
        # Build grouped byte chunks from decomposition
        # --------------------------------------------------------
        comp_list = []
        group_lens = []

        for group_tuple in decomp:
            group_tuple = tuple(group_tuple)
            gl = len(group_tuple)
            group_lens.append(gl)

            cur_comp_data = np.zeros((gl, n), dtype=type_byte)
            for j, byte_idx in enumerate(group_tuple):
                cur_comp_data[j] = comps[int(byte_idx)]
            comp_list.append(cur_comp_data)

        K = int(sum(group_lens))

        # --------------------------------------------------------
        # Packed layout for grouped SDDL, col-order only
        # --------------------------------------------------------
        packed = np.zeros((K, n), dtype=type_byte)
        off = 0
        for chunk in comp_list:
            gl = int(chunk.shape[0])
            packed[off:off + gl, :] = chunk
            off += gl

        packed_bytes_F = np.frombuffer(packed.flatten("F").tobytes(), dtype=np.byte)

        # ========================================================
        # 1) Standard SDDL
        # ========================================================
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

        # ========================================================
        # 2) Grouped/decomposed SDDL, col-order only
        # ========================================================
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

        _, t_sddl_full_F, fullproc_F, _ = _full_restore_grouped_sddl_time(
            sddl_tool,
            packed_bytes_F=packed_bytes_F,
            group_lens=group_lens,
            decomp=decomp,
            n=n,
            dtype=data_set.dtype,
            record_size=K,
            max_records=sddlh.SDDL_MAX_RECORDS_PER_CHUNK,
        )

        # exact correctness against original data
        try:
            comp_once_F = sddl_tool(packed_bytes_F)
            regen_packed_F = sddlh._decompress_serial_bytes(comp_once_F)
            regen_x_F = sddlh._unpack_tdt_packed_bytes_to_float(
                packed_serial=regen_packed_F,
                group_lens=group_lens,
                decomp=decomp,
                n=n,
                dtype=data_set.dtype,
            )
            ok_F = np.array_equal(regen_x_F, data_set)
        except Exception:
            ok_F = False

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
    contig_order = CONTIG_ORDER
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
            contig_order=contig_order,
            out_log_dir=out_log_dir,
        )

        print(f"Done: wrote stats in {os.path.join(out_log_dir, f'{dataset_name}_decomposition_stats.csv')}")


if __name__ == "__main__":
    main()