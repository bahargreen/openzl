# -*- coding: utf-8 -*-
"""
SDDL-based compression for decomposed byte layouts (multi-stream, matches your decomposition)
+ throughput instrumentation that measures OpenZL core time correctly.

What “core time” means here:
- The timer covers ONLY the OpenZL call(s): tool(np_bytes) / tool(x) / tool(chunk)
- Any NumPy packing / flattening / tobytes / frombuffer work is done OUTSIDE the timer.

Reported per method:
- ratio (orig_bytes / compressed_bytes)
- core time (seconds)
- core throughput MB/s = processed_input_bytes / core_seconds / 1e6

For SDDL chunked:
- processed_input_bytes = nbytes_aligned actually fed (record-aligned)
  (Set SDDL_MBPS_DENOM = "orig" if you want to use len_bytes instead.)
"""

import os
import itertools
import time
import numpy as np
import pandas as pd

from utils import (
    tuple_to_string, compute_entropy, list_to_string,
    find_max_consecutive_similar_values
)

try:
    import openzl.ext as zl
    _HAS_OPENZL = True
except Exception:
    zl = None
    _HAS_OPENZL = False


# ============================================================
# Timing helpers
# ============================================================

N_WARMUP = 1          # warm-up calls (not recorded)
N_RUNS = 10           # timed calls; we report median over these runs
SDDL_MAX_RECORDS_PER_CHUNK = 200_000
SDDL_MBPS_DENOM = "aligned"   # "aligned" or "orig"


def _mbps(nbytes: int, seconds: float) -> float:
    return float("inf") if seconds <= 0.0 else (float(nbytes) / float(seconds) / 1e6)


def _median_time(callable_fn) -> float:
    # warm-up
    for _ in range(N_WARMUP):
        callable_fn()

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        callable_fn()
        times.append(time.perf_counter() - t0)

    return float(np.median(np.asarray(times, dtype=np.float64)))


# ============================================================
# OpenZL compressors (unchanged core)
# ============================================================

class _OpenZLSerialCompress:
    """OpenZL: Serial -> graphs.Compress()"""
    def __init__(self):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        self._comp = zl.Compressor()
        gid = zl.graphs.Compress()(self._comp)
        self._comp.select_starting_graph(gid)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")
        b = np_bytes.tobytes(order="C")
        cctx = zl.CCtx()
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return cctx.compress([zl.Input(zl.Type.Serial, b)])


class _OpenZLSerialZstd:
    """OpenZL: Serial -> graphs.Zstd()"""
    def __init__(self):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        self._comp = zl.Compressor()
        gid = zl.graphs.Zstd()(self._comp)
        self._comp.select_starting_graph(gid)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")
        b = np_bytes.tobytes(order="C")
        cctx = zl.CCtx()
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return cctx.compress([zl.Input(zl.Type.Serial, b)])


class _OpenZLFloatDeconstruct:
    """
    OpenZL: Numeric(float32/float64) -> Float{32,64}Deconstruct(sign_frac=Compress, exponent=Compress)
    Fallback: if input byte-length is not divisible by m, compress as Serial->Compress.
    """
    def __init__(self, m: int):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        if m not in (4, 8):
            raise ValueError("FloatDeconstruct only valid for m=4 or m=8")
        self.m = int(m)

        self._comp = zl.Compressor()
        if self.m == 4:
            gid = zl.nodes.Float32Deconstruct()(
                self._comp,
                sign_frac=zl.graphs.Compress(),
                exponent=zl.graphs.Compress(),
            )
        else:
            gid = zl.nodes.Float64Deconstruct()(
                self._comp,
                sign_frac=zl.graphs.Compress(),
                exponent=zl.graphs.Compress(),
            )
        self._comp.select_starting_graph(gid)

        self._fallback = zl.Compressor()
        fb_gid = zl.graphs.Compress()(self._fallback)
        self._fallback.select_starting_graph(fb_gid)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")

        u8 = np_bytes.view(np.uint8)
        if u8.nbytes % self.m != 0:
            b = np_bytes.tobytes(order="C")
            cctx = zl.CCtx()
            cctx.ref_compressor(self._fallback)
            cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
            return cctx.compress([zl.Input(zl.Type.Serial, b)])

        x = u8.view(np.float32) if self.m == 4 else u8.view(np.float64)
        cctx = zl.CCtx()
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return cctx.compress([zl.Input(zl.Type.Numeric, x)])


class _OpenZLNumericOnly:
    """Numeric-only: directly compress float16/float32/float64 as zl.Type.Numeric."""
    def __init__(self):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        self._comp = zl.Compressor()
        gid = zl.graphs.Compress()(self._comp)
        self._comp.select_starting_graph(gid)

    def __call__(self, x: np.ndarray) -> bytes:
        if not isinstance(x, np.ndarray):
            raise TypeError("OpenZL numeric-only expects a NumPy array")
        if x.dtype not in (np.float16, np.float32, np.float64):
            raise TypeError(f"Numeric-only expects float16/float32/float64, got {x.dtype}")
        cctx = zl.CCtx()
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return cctx.compress([zl.Input(zl.Type.Numeric, x)])


class _OpenZLSerialToStruct:
    """Serial -> struct(k) -> Compress"""
    def __init__(self, k: int):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        k = int(k)
        if k < 1:
            raise ValueError("struct size must be >= 1")
        self.k = k

        self._comp = zl.Compressor()
        gid = zl.nodes.ConvertSerialToStruct(struct_size_bytes=self.k)(
            self._comp,
            zl.graphs.Compress(),
        )
        self._comp.select_starting_graph(gid)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")
        b = np_bytes.tobytes(order="C")
        cctx = zl.CCtx()
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return cctx.compress([zl.Input(zl.Type.Serial, b)])


class _OpenZLPackedStruct:
    """Pack groups into ONE record per element (total K bytes), then interpret stream as struct(K)."""
    def __init__(self, K: int):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        K = int(K)
        if K < 1:
            raise ValueError("K must be >= 1")
        self.K = K

        self._comp = zl.Compressor()
        gid = zl.nodes.ConvertSerialToStruct(struct_size_bytes=self.K)(
            self._comp,
            zl.graphs.Compress(),
        )
        self._comp.select_starting_graph(gid)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")
        b = np_bytes.tobytes(order="C")
        cctx = zl.CCtx()
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return cctx.compress([zl.Input(zl.Type.Serial, b)])


# ============================================================
# SDDL multi-stream (matches decomposition groups)
# ============================================================

def _make_sddl_record_from_group_lens(group_lens) -> str:
    group_lens = [int(x) for x in group_lens]
    if any(g <= 0 for g in group_lens):
        raise ValueError("group_lens must be positive")

    lines = ["Rec = {"]
    for i, g in enumerate(group_lens, start=1):
        if g == 1:
            lines.append(f"  g{i} : Byte;")
        else:
            lines.append(f"  g{i} : Byte[{g}];")
    lines.append("};")
    lines.append("")
    lines.append(": Rec[_rem / sizeof Rec];")
    return "\n".join(lines)


class _SDDLSuccessorN(zl.FunctionGraph):
    def __init__(self, num_groups: int):
        super().__init__()
        self._num_groups = int(num_groups)
        if self._num_groups < 1:
            raise ValueError("num_groups must be >= 1")

    def function_graph_description(self) -> zl.FunctionGraphDescription:
        masks = [zl.TypeMask.Numeric, zl.TypeMask.Numeric] + [zl.TypeMask.Serial] * self._num_groups
        return zl.FunctionGraphDescription(
            name=f"sddl_successor_{2 + self._num_groups}",
            input_type_masks=masks,
        )

    def graph(self, state: zl.GraphState) -> None:
        for e in state.edges:
            zl.graphs.Compress().set_destination(e)


class _OpenZLSDDLGroups:
    def __init__(self, group_lens):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        group_lens = tuple(int(x) for x in group_lens)
        if len(group_lens) < 1:
            raise ValueError("group_lens must be non-empty")
        if any(g <= 0 for g in group_lens):
            raise ValueError("group_lens must be positive")

        self.group_lens = group_lens
        self.K = int(sum(group_lens))
        self._desc = _make_sddl_record_from_group_lens(group_lens)

        self._comp = zl.Compressor()
        succ_gid = self._comp.register_function_graph(_SDDLSuccessorN(num_groups=len(group_lens)))
        sddl_gid = zl.graphs.SDDL(description=self._desc, successor=succ_gid)(self._comp)
        self._comp.select_starting_graph(sddl_gid)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")
        b = np_bytes.tobytes(order="C")
        cctx = zl.CCtx()
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return cctx.compress([zl.Input(zl.Type.Serial, b)])


# ============================================================
# Decomposition configs (unchanged)
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
# Decomposition enumeration helpers (unchanged)
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
# Core-timed compression wrappers (OpenZL time only)
# ============================================================

def _core_time_and_size_serial(tool, np_byte_arrays: list[np.ndarray]) -> tuple[int, float]:
    """
    np_byte_arrays: list of 1-D np.byte arrays already prepared.
    Measures only tool(...) time, not preparation.
    Returns: (sum_compressed_bytes, median_seconds)
    """
    def one_pass():
        total = 0
        for arr in np_byte_arrays:
            total += len(tool(arr))
        return total

    out_size = one_pass()  # warm-up size (deterministic)
    def timed_pass():
        one_pass()

    dt = _median_time(timed_pass)
    return int(out_size), float(dt)


def _core_time_and_size_numeric(tool, x: np.ndarray) -> tuple[int, float]:
    def warm():
        return len(tool(x))
    out_size = warm()

    def timed():
        tool(x)

    dt = _median_time(timed)
    return int(out_size), float(dt)


def _core_time_and_size_sddl_chunked(tool, serial_bytes: np.ndarray, record_size: int, max_records: int):
    """
    serial_bytes is already prepared np.byte 1-D array.
    Measures only tool(chunk) time, not serial_bytes creation.
    Returns: (compressed_bytes, median_seconds, nbytes_aligned)
    """
    record_size = int(record_size)
    max_records = int(max_records)

    nbytes = int(serial_bytes.size)
    nbytes_aligned = nbytes - (nbytes % record_size)
    if nbytes_aligned <= 0:
        return 0, 0.0, 0

    serial_bytes = serial_bytes[:nbytes_aligned]
    chunk_bytes = record_size * max_records

    def one_pass():
        total = 0
        off = 0
        while off < nbytes_aligned:
            end = min(off + chunk_bytes, nbytes_aligned)
            end = end - ((end - off) % record_size)
            if end <= off:
                break
            total += len(tool(serial_bytes[off:end]))
            off = end
        return total

    out_size = one_pass()  # warm-up size
    def timed_pass():
        one_pass()

    dt = _median_time(timed_pass)
    return int(out_size), float(dt), int(nbytes_aligned)


# ============================================================
# Main evaluation
# ============================================================

def test_decomposition(
    data_set: np.ndarray,
    dataset_name: str,
    comp_tool_dict=None,
    given_decomp=None,
    m=4,
    chunk_no=-1,
    contig_order=True,
    out_log_dir="/home/jamalids/Documents/1",
):
    if comp_tool_dict is None:
        comp_tool_dict = {}

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

    # byte planes, precomputed once
    comps = np.zeros((m, len(data_set)), dtype=type_byte)
    for i in range(m):
        comps[i] = data_set_bytes[i:len_bytes:m]

    numeric_fallback = comp_tool_dict.get("openzl_serial", None)

    struct_cache = {}
    packed_struct_cache = {}
    sddl_groups_cache = {}

    def get_struct_tool(group_len: int):
        group_len = int(group_len)
        if group_len not in struct_cache:
            struct_cache[group_len] = _OpenZLSerialToStruct(group_len)
        return struct_cache[group_len]

    def get_packed_struct_tool(K: int):
        K = int(K)
        if K not in packed_struct_cache:
            packed_struct_cache[K] = _OpenZLPackedStruct(K)
        return packed_struct_cache[K]

    def get_sddl_groups_tool(group_lens):
        key = tuple(int(x) for x in group_lens)
        if key not in sddl_groups_cache:
            sddl_groups_cache[key] = _OpenZLSDDLGroups(key)
        return sddl_groups_cache[key]

    def ratio(orig: int, comp: int) -> float:
        return float("inf") if comp == 0 else float(orig) / float(comp)

    # Precompute standard serial bytes once (for standard serial/zstd/struct(K) style)
    std_serial_F = np.frombuffer(data_set.flatten("F").tobytes(), dtype=np.byte)

    stat_array = []
    for idx, decomp in enumerate(all_decomps):
        stats = {
            "dataset name": dataset_name,
            "original size": len_bytes,
            "type width": m,
            "Dimension": int(len(data_set)),
            "decomposition": tuple_to_string(decomp),
            "chunk no": int(chunk_no),
        }

        # Build comp_list as byte-group matrices
        comp_list = []
        group_lens = []
        for group_tuple in decomp:
            group_tuple = tuple(group_tuple)
            gl = len(group_tuple)
            group_lens.append(gl)
            cur_comp_data = np.zeros((gl, len(data_set)), dtype=type_byte)
            for j, byte_idx in enumerate(group_tuple):
                cur_comp_data[j] = comps[int(byte_idx)]
            comp_list.append(cur_comp_data)

        K = int(sum(group_lens))

        # Precompute decomposed serial byte arrays for col/row orders (one per group)
        decomp_serial_F = []
        decomp_serial_C = []
        for chunk in comp_list:
            decomp_serial_F.append(np.frombuffer(chunk.flatten("F").tobytes(), dtype=np.byte))
            decomp_serial_C.append(np.frombuffer(chunk.flatten("C").tobytes(), dtype=np.byte))

        # Precompute packed matrix + packed bytes (for packed struct, SDDL)
        packed = np.zeros((K, len(data_set)), dtype=type_byte)
        off = 0
        for chunk in comp_list:
            gl = int(chunk.shape[0])
            packed[off:off + gl, :] = chunk
            off += gl
        packed_bytes_F = np.frombuffer(packed.flatten("F").tobytes(), dtype=np.byte)
        packed_bytes_C = np.frombuffer(packed.flatten("C").tobytes(), dtype=np.byte)

        # ------------------------------------------------------------
        # Standard + decomposed for comp_tool_dict items (core time only)
        # ------------------------------------------------------------
        for comp_name, comp_tool in comp_tool_dict.items():
            if comp_name == "openzl_numeric_only":
                full_comp_size, full_t = _core_time_and_size_numeric(comp_tool, data_set)
            else:
                full_comp_size, full_t = _core_time_and_size_serial(comp_tool, [std_serial_F])

            if comp_name == "openzl_numeric_only":
                if numeric_fallback is None:
                    raise RuntimeError("openzl_numeric_only needs openzl_serial as fallback for decomposed chunks")
                decomp_col, t_col = _core_time_and_size_serial(numeric_fallback, decomp_serial_F)
                decomp_row, t_row = _core_time_and_size_serial(numeric_fallback, decomp_serial_C)
            else:
                decomp_col, t_col = _core_time_and_size_serial(comp_tool, decomp_serial_F)
                decomp_row, t_row = _core_time_and_size_serial(comp_tool, decomp_serial_C)

            stats[f"standard {comp_name} ratio"] = ratio(len_bytes, full_comp_size)
            stats[f"decomposed {comp_name} col-order ratio"] = ratio(len_bytes, decomp_col)
            stats[f"decomposed {comp_name} row-order ratio"] = ratio(len_bytes, decomp_row)

            stats[f"standard {comp_name} time s"] = float(full_t)
            stats[f"standard {comp_name} MB/s"] = _mbps(len_bytes, full_t)

            stats[f"decomposed {comp_name} col-order time s"] = float(t_col)
            stats[f"decomposed {comp_name} col-order MB/s"] = _mbps(len_bytes, t_col)

            stats[f"decomposed {comp_name} row-order time s"] = float(t_row)
            stats[f"decomposed {comp_name} row-order MB/s"] = _mbps(len_bytes, t_row)

        # ------------------------------------------------------------
        # Struct-groups (sum over groups), core time only
        # ------------------------------------------------------------
        struct_sum_col = 0
        struct_sum_row = 0
        t_struct_col = 0.0
        t_struct_row = 0.0

        for chunk_bytes, chunk in zip(decomp_serial_F, comp_list):
            gl = int(chunk.shape[0])
            tool = get_struct_tool(gl)
            sz, dt = _core_time_and_size_serial(tool, [chunk_bytes])
            struct_sum_col += sz
            t_struct_col += dt

        for chunk_bytes, chunk in zip(decomp_serial_C, comp_list):
            gl = int(chunk.shape[0])
            tool = get_struct_tool(gl)
            sz, dt = _core_time_and_size_serial(tool, [chunk_bytes])
            struct_sum_row += sz
            t_struct_row += dt

        stats["decomposed openzl_struct_groups col-order ratio"] = ratio(len_bytes, struct_sum_col)
        stats["decomposed openzl_struct_groups row-order ratio"] = ratio(len_bytes, struct_sum_row)

        stats["decomposed openzl_struct_groups col-order time s"] = float(t_struct_col)
        stats["decomposed openzl_struct_groups col-order MB/s"] = _mbps(len_bytes, t_struct_col)

        stats["decomposed openzl_struct_groups row-order time s"] = float(t_struct_row)
        stats["decomposed openzl_struct_groups row-order MB/s"] = _mbps(len_bytes, t_struct_row)

        # ------------------------------------------------------------
        # Packed-struct (single stream struct(K)), core time only
        # ------------------------------------------------------------
        packed_tool = get_packed_struct_tool(K)
        packed_col, t_packed_col = _core_time_and_size_serial(packed_tool, [packed_bytes_F])
        packed_row, t_packed_row = _core_time_and_size_serial(packed_tool, [packed_bytes_C])

        stats["decomposed openzl_struct_packed col-order ratio"] = ratio(len_bytes, packed_col)
        stats["decomposed openzl_struct_packed row-order ratio"] = ratio(len_bytes, packed_row)

        stats["decomposed openzl_struct_packed col-order time s"] = float(t_packed_col)
        stats["decomposed openzl_struct_packed col-order MB/s"] = _mbps(len_bytes, t_packed_col)

        stats["decomposed openzl_struct_packed row-order time s"] = float(t_packed_row)
        stats["decomposed openzl_struct_packed row-order MB/s"] = _mbps(len_bytes, t_packed_row)

        # ------------------------------------------------------------
        # SDDL multi-stream over packed layout, chunked, core time only
        # ------------------------------------------------------------
        sddl_tool = get_sddl_groups_tool(group_lens)

        sddl_size_F, t_sddl_F, nproc_F = _core_time_and_size_sddl_chunked(
            sddl_tool, packed_bytes_F, record_size=K, max_records=SDDL_MAX_RECORDS_PER_CHUNK
        )
        sddl_size_C, t_sddl_C, nproc_C = _core_time_and_size_sddl_chunked(
            sddl_tool, packed_bytes_C, record_size=K, max_records=SDDL_MAX_RECORDS_PER_CHUNK
        )

        stats["decomposed openzl_sddl_groups col-order ratio"] = ratio(len_bytes, sddl_size_F)
        stats["decomposed openzl_sddl_groups row-order ratio"] = ratio(len_bytes, sddl_size_C)

        stats["decomposed openzl_sddl_groups col-order time s"] = float(t_sddl_F)
        stats["decomposed openzl_sddl_groups row-order time s"] = float(t_sddl_C)

        if SDDL_MBPS_DENOM == "orig":
            stats["decomposed openzl_sddl_groups col-order MB/s"] = _mbps(len_bytes, t_sddl_F)
            stats["decomposed openzl_sddl_groups row-order MB/s"] = _mbps(len_bytes, t_sddl_C)
        else:
            stats["decomposed openzl_sddl_groups col-order MB/s"] = _mbps(nproc_F, t_sddl_F)
            stats["decomposed openzl_sddl_groups row-order MB/s"] = _mbps(nproc_C, t_sddl_C)

        stat_array.append(stats)

        if out_log_dir and (((idx + 1) % 20 == 0) or ((idx + 1) == total_enum)):
            os.makedirs(out_log_dir, exist_ok=True)
            out_csv = os.path.join(out_log_dir, f"{dataset_name}_decomposition_stats.csv")
            pd.DataFrame(stat_array).to_csv(out_csv, index=False)

    return stat_array


def main():
    dataset_folder = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32"
    m = 4
    contig_order = False

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
        print("Processing Dataset:", dataset_name)

        data_df = pd.read_csv(dataset_path, sep="\t")

        if m == 2:
            sliced_data = data_df.values[:, 1].astype(np.float16)
        elif m == 4:
            sliced_data = data_df.values[:, 1].astype(np.float32)
        else:
            sliced_data = data_df.values[:, 1].astype(np.float64)

        dataset_configs = compConfigMap.get(dataset_name, compConfigMap["default"])

        converted_decomps = []
        for grouping in dataset_configs:
            zero_based_groups = []
            for group in grouping:
                zero_based_groups.append(tuple(int(g) - 1 for g in group))
            converted_decomps.append(tuple(zero_based_groups))

        comp_tool_dict = {
            "openzl_serial": _OpenZLSerialCompress(),
            "openzl_zstd": _OpenZLSerialZstd(),
            "openzl_numeric_only": _OpenZLNumericOnly(),
        }

        if m in (4, 8):
            try:
                comp_tool_dict[f"openzl_float_deconstruct_m{m}"] = _OpenZLFloatDeconstruct(m=m)
            except Exception as e:
                print(f"[WARN] Could not init OpenZL float deconstruct for m={m}: {e}")

        _ = test_decomposition(
            data_set=sliced_data,
            dataset_name=dataset_name,
            m=m,
            comp_tool_dict=comp_tool_dict,
            given_decomp=converted_decomps,
            contig_order=contig_order,
            out_log_dir="/home/jamalids/Documents/4",
        )
        print(f"Done: wrote stats in /home/jamalids/Documents/4/{dataset_name}_decomposition_stats.csv")


if __name__ == "__main__":
    main()