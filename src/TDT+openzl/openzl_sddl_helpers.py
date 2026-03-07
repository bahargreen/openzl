# -*- coding: utf-8 -*-
"""
Helpers for:
- standard SDDL
- TDT + SDDL
- compression/decompression timing
- packed-byte reconstruction to float arrays

This file is intended to be imported from your main benchmarking script.
"""

import time
import numpy as np

try:
    import openzl.ext as zl
    _HAS_OPENZL = True
except Exception:
    zl = None
    _HAS_OPENZL = False


# ============================================================
# Global timing/config defaults
# These can be overridden from main script if needed.
# ============================================================

N_WARMUP = 1
N_RUNS = 10
SDDL_MAX_RECORDS_PER_CHUNK = 200_000
SDDL_MBPS_DENOM = "aligned"   # "aligned" or "orig"


# ============================================================
# Basic timing helpers
# ============================================================

def _mbps(nbytes: int, seconds: float) -> float:
    return float("inf") if seconds <= 0.0 else (float(nbytes) / float(seconds) / 1e6)


def _median_time(callable_fn) -> float:
    for _ in range(N_WARMUP):
        callable_fn()

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        callable_fn()
        times.append(time.perf_counter() - t0)

    return float(np.median(np.asarray(times, dtype=np.float64)))


def _median_decomp_time(callable_fn) -> float:
    for _ in range(N_WARMUP):
        callable_fn()

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        callable_fn()
        times.append(time.perf_counter() - t0)

    return float(np.median(np.asarray(times, dtype=np.float64)))


# ============================================================
# Existing timing wrappers you wanted in helper file
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

    out_size = one_pass()

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

    out_size = one_pass()

    def timed_pass():
        one_pass()

    dt = _median_time(timed_pass)
    return int(out_size), float(dt), int(nbytes_aligned)


# ============================================================
# Decompression helpers
# ============================================================

def _decompress_serial_bytes(compressed: bytes) -> bytes:
    """
    Decompress one OpenZL frame and expect exactly one Serial output.
    """
    if not _HAS_OPENZL:
        raise RuntimeError("OpenZL not available")

    dctx = zl.DCtx()
    outputs = dctx.decompress(compressed)

    if len(outputs) != 1:
        raise RuntimeError(f"Expected exactly one output, got {len(outputs)}")

    out = outputs[0]
    if out.type != zl.Type.Serial:
        raise RuntimeError(f"Expected Serial output, got {out.type}")

    return out.content.as_bytes()


def _core_time_and_check_decompress_serial(
    compressed: bytes,
    expected_serial: np.ndarray
) -> tuple[float, int]:
    """
    Measures only OpenZL decompression time.
    expected_serial: np.byte 1-D array with exact expected bytes.
    Returns:
        (median_seconds, decompressed_nbytes)
    """
    expected_b = expected_serial.tobytes(order="C")

    regen = _decompress_serial_bytes(compressed)
    if regen != expected_b:
        raise RuntimeError("Decompression mismatch / corruption detected")

    def timed():
        _decompress_serial_bytes(compressed)

    dt = _median_decomp_time(timed)
    return float(dt), int(len(expected_b))


def _core_time_and_check_decompress_sddl_chunked(
    tool,
    serial_bytes: np.ndarray,
    record_size: int,
    max_records: int,
) -> tuple[int, float, int]:
    """
    Compress chunk-by-chunk using SDDL tool, then measure chunk-by-chunk
    decompression time. Measures only OpenZL decompression time.

    Returns:
        (compressed_total_size, decomp_time_s, nbytes_aligned)
    """
    record_size = int(record_size)
    max_records = int(max_records)

    nbytes = int(serial_bytes.size)
    nbytes_aligned = nbytes - (nbytes % record_size)
    if nbytes_aligned <= 0:
        return 0, 0.0, 0

    serial_bytes = serial_bytes[:nbytes_aligned]
    chunk_bytes = record_size * max_records

    compressed_chunks = []

    off = 0
    while off < nbytes_aligned:
        end = min(off + chunk_bytes, nbytes_aligned)
        end = end - ((end - off) % record_size)
        if end <= off:
            break

        chunk = serial_bytes[off:end]
        comp = tool(chunk)

        regen = _decompress_serial_bytes(comp)
        if regen != chunk.tobytes(order="C"):
            raise RuntimeError("SDDL decompression mismatch / corruption detected")

        compressed_chunks.append(comp)
        off = end

    compressed_total = sum(len(x) for x in compressed_chunks)

    def timed_pass():
        for comp in compressed_chunks:
            _decompress_serial_bytes(comp)

    dt = _median_decomp_time(timed_pass)
    return int(compressed_total), float(dt), int(nbytes_aligned)


# ============================================================
# Struct wrappers
# ============================================================

class _OpenZLSerialToStruct:
    """Serial -> Struct(k) -> Compress"""
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
            raise TypeError("Expected NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("Expected byte/int8/uint8 array")

        b = np_bytes.tobytes(order="C")
        cctx = zl.CCtx()
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return cctx.compress([zl.Input(zl.Type.Serial, b)])


class _OpenZLPackedStruct:
    """Pack groups into one record per element, then Serial -> Struct(K) -> Compress"""
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
            raise TypeError("Expected NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("Expected byte/int8/uint8 array")

        b = np_bytes.tobytes(order="C")
        cctx = zl.CCtx()
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return cctx.compress([zl.Input(zl.Type.Serial, b)])


# ============================================================
# SDDL graph builders
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
    """
    TDT + SDDL:
    one packed serial stream, record fields follow decomposition groups.
    """
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
        succ_gid = self._comp.register_function_graph(
            _SDDLSuccessorN(num_groups=len(group_lens))
        )
        sddl_gid = zl.graphs.SDDL(
            description=self._desc,
            successor=succ_gid
        )(self._comp)
        self._comp.select_starting_graph(sddl_gid)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("Expected NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("Expected byte/int8/uint8 array")

        b = np_bytes.tobytes(order="C")
        cctx = zl.CCtx()
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return cctx.compress([zl.Input(zl.Type.Serial, b)])


class _OpenZLSDDLStandard:
    """
    Standard SDDL: one record per original element, as Byte[m].
    Example:
      float32 -> Byte[4]
      float64 -> Byte[8]
    """
    def __init__(self, m: int):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")

        self.m = int(m)
        if self.m < 1:
            raise ValueError("m must be >= 1")

        desc = f"""
Rec = {{
  x : Byte[{self.m}];
}};

: Rec[_rem / sizeof Rec];
""".strip()

        self._comp = zl.Compressor()
        succ_gid = self._comp.register_function_graph(_SDDLSuccessorN(num_groups=1))
        sddl_gid = zl.graphs.SDDL(description=desc, successor=succ_gid)(self._comp)
        self._comp.select_starting_graph(sddl_gid)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("Expected NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("Expected byte/int8/uint8 array")

        b = np_bytes.tobytes(order="C")
        cctx = zl.CCtx()
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return cctx.compress([zl.Input(zl.Type.Serial, b)])


# ============================================================
# Reconstruction helpers
# ============================================================

def _unpack_tdt_packed_bytes_to_float(
    packed_serial: bytes,
    group_lens,
    decomp,
    n: int,
    dtype
) -> np.ndarray:
    """
    Reverse TDT packing when packed bytes were produced with packed.flatten("F").
    """
    K = int(sum(group_lens))
    buf = np.frombuffer(packed_serial, dtype=np.uint8)
    if buf.size != K * n:
        raise ValueError(f"Unexpected packed size: got {buf.size}, expected {K * n}")

    packed = buf.reshape((K, n), order="F")

    m = K
    comps_recovered = np.zeros((m, n), dtype=np.uint8)

    off = 0
    for group_tuple, gl in zip(decomp, group_lens):
        gl = int(gl)
        chunk = packed[off:off + gl, :]
        for local_j, byte_idx in enumerate(group_tuple):
            comps_recovered[int(byte_idx), :] = chunk[local_j]
        off += gl

    raw = np.empty(m * n, dtype=np.uint8)
    for i in range(m):
        raw[i:m * n:m] = comps_recovered[i]

    return raw.view(dtype)


def _unpack_tdt_packed_bytes_to_float_rowmajor(
    packed_serial: bytes,
    group_lens,
    decomp,
    n: int,
    dtype
) -> np.ndarray:
    """
    Reverse TDT packing when packed bytes were produced with packed.flatten("C").
    """
    K = int(sum(group_lens))
    buf = np.frombuffer(packed_serial, dtype=np.uint8)
    if buf.size != K * n:
        raise ValueError(f"Unexpected packed size: got {buf.size}, expected {K * n}")

    packed = buf.reshape((K, n), order="C")

    m = K
    comps_recovered = np.zeros((m, n), dtype=np.uint8)

    off = 0
    for group_tuple, gl in zip(decomp, group_lens):
        gl = int(gl)
        chunk = packed[off:off + gl, :]
        for local_j, byte_idx in enumerate(group_tuple):
            comps_recovered[int(byte_idx), :] = chunk[local_j]
        off += gl

    raw = np.empty(m * n, dtype=np.uint8)
    for i in range(m):
        raw[i:m * n:m] = comps_recovered[i]

    return raw.view(dtype)