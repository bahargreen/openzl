# -*- coding: utf-8 -*-
"""
Helpers for:
- standard SDDL
- grouped/decomposed SDDL
- grouped/decomposed Struct with separate streams
- compression/decompression core timing
- packed-byte reconstruction to float arrays

Notes:
- Struct path uses zl.nodes.ConvertSerialToStruct(...)
- This helper is cleaned for GROUP struct mode
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


# ============================================================
# Layout helpers
# ============================================================

def _is_no_reorder_decomp(decomp, m: int) -> bool:
    """
    True only if decomposition preserves original byte order exactly,
    only changing contiguous group boundaries.
    """
    flat = []
    for group in decomp:
        g = tuple(int(x) for x in group)
        if len(g) == 0:
            return False
        if g != tuple(range(g[0], g[0] + len(g))):
            return False
        flat.extend(g)

    return flat == list(range(m))


def _byte_planes_view(data_set: np.ndarray, m: int) -> np.ndarray:
    """
    Zero-copy byte-plane view with shape (m, n).
    """
    n = int(len(data_set))
    u8 = data_set.view(np.uint8).reshape(n, m)
    return u8.T


def _build_grouped_packed_bytes_col_fast(
    byte_planes: np.ndarray,
    decomp,
) -> tuple[np.ndarray, list[int], int]:
    """
    Build grouped packed bytes in col-order.

    Returns:
        packed_bytes_F : 1-D np.byte
        group_lens     : list[int]
        K              : total packed width
    """
    group_lens = [len(tuple(g)) for g in decomp]
    K = int(sum(group_lens))

    grouped_rows = [byte_planes[list(tuple(g)), :] for g in decomp]
    packed = np.concatenate(grouped_rows, axis=0)
    packed_bytes_F = np.frombuffer(packed.tobytes(order="F"), dtype=np.byte)
    return packed_bytes_F, group_lens, K


def _build_grouped_separate_bytes_col_fast(
    byte_planes: np.ndarray,
    decomp,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Build one separate serial byte stream per group.

    For each group g with length gl:
        part shape = (gl, n)
        stream = part.T.reshape(-1)   -> records of width gl

    Returns:
        separated_streams : list[np.ndarray]
        group_lens        : list[int]
    """
    separated_streams = []
    group_lens = []

    for group in decomp:
        group = tuple(int(x) for x in group)
        gl = len(group)
        if gl <= 0:
            raise ValueError("Empty group in decomposition")

        part = byte_planes[list(group), :]   # shape (gl, n)
        serial = np.ascontiguousarray(part.T.reshape(-1), dtype=np.byte)
        separated_streams.append(serial)
        group_lens.append(gl)

    return separated_streams, group_lens


# ============================================================
# Core compression timing
# ============================================================

def _core_time_and_size_sddl_chunked(
    tool,
    serial_bytes: np.ndarray,
    record_size: int,
    max_records: int,
):
    """
    Measures only tool(chunk) time, not serial_bytes creation.

    Returns:
        (compressed_bytes, median_seconds, nbytes_aligned)
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
# Core decompression helpers
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


def _core_time_and_check_decompress_sddl_chunked(
    tool,
    serial_bytes: np.ndarray,
    record_size: int,
    max_records: int,
) -> tuple[int, float, int]:
    """
    Compress chunk-by-chunk, then measure only OpenZL decompression time.

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
            raise RuntimeError("Decompression mismatch / corruption detected")

        compressed_chunks.append(comp)
        off = end

    compressed_total = sum(len(x) for x in compressed_chunks)

    def timed_pass():
        for comp in compressed_chunks:
            _decompress_serial_bytes(comp)

    dt = _median_time(timed_pass)
    return int(compressed_total), float(dt), int(nbytes_aligned)


# ============================================================
# SDDL graph builders
# ============================================================

def _make_sddl_record_from_group_lens(group_lens) -> str:
    group_lens = [int(x) for x in group_lens]
    if len(group_lens) < 1:
        raise ValueError("group_lens must be non-empty")
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
    Grouped / decomposed SDDL:
    one packed serial stream, record fields follow byte-group decomposition.
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
# Struct graph builder for group mode
# ============================================================

class _OpenZLStructStandard:
    """
    Serial -> struct(record_size) -> Compress

    Uses zl.nodes.ConvertSerialToStruct(...), which is the working API
    in your OpenZL binding.
    """
    def __init__(self, record_size: int):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")

        self.record_size = int(record_size)
        if self.record_size < 1:
            raise ValueError("record_size must be >= 1")

        self._comp = zl.Compressor()
        gid = zl.nodes.ConvertSerialToStruct(struct_size_bytes=self.record_size)(
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
# Reconstruction helpers
# ============================================================

def _restore_no_reorder_bytes_to_array(
    packed_serial: bytes,
    n: int,
    dtype,
) -> np.ndarray:
    """
    Fast path when grouped layout is identical to original byte order.
    No inverse grouping needed.
    """
    arr = np.frombuffer(packed_serial, dtype=dtype, count=n)
    return arr.copy()


def _unpack_tdt_packed_bytes_to_float(
    packed_serial: bytes,
    group_lens,
    decomp,
    n: int,
    dtype
) -> np.ndarray:
    """
    Reverse grouped/decomposed packing when packed bytes were produced with:
        packed.flatten("F")
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

    return raw.view(dtype).copy()


def _restore_grouped_bytes_to_array(
    packed_serial: bytes,
    group_lens,
    decomp,
    n: int,
    dtype,
    no_reorder_layout: bool,
) -> np.ndarray:
    """
    Unified restore entry point for packed grouped layout.
    """
    if no_reorder_layout:
        return _restore_no_reorder_bytes_to_array(
            packed_serial=packed_serial,
            n=n,
            dtype=dtype,
        )

    return _unpack_tdt_packed_bytes_to_float(
        packed_serial=packed_serial,
        group_lens=group_lens,
        decomp=decomp,
        n=n,
        dtype=dtype,
    )


def _restore_separated_grouped_bytes_to_array(
    separated_group_serials,
    group_lens,
    decomp,
    n: int,
    dtype,
) -> np.ndarray:
    """
    Restore original array from separate group streams.

    Each separated_group_serials[g] contains n records,
    each record width = group_lens[g].

    We rebuild byte-planes in original order, then cast back to dtype.
    """
    group_lens = [int(x) for x in group_lens]
    m = int(sum(group_lens))

    out_planes = np.empty((m, n), dtype=np.uint8)

    if len(separated_group_serials) != len(group_lens):
        raise ValueError("Mismatch between separated_group_serials and group_lens")

    for serial, group, gl in zip(separated_group_serials, decomp, group_lens):
        buf = np.frombuffer(np.asarray(serial, dtype=np.byte).tobytes(), dtype=np.uint8)
        if buf.size != n * gl:
            raise ValueError(f"Unexpected separated group size: got {buf.size}, expected {n * gl}")

        part = buf.reshape((n, gl)).T

        for local_idx, global_plane in enumerate(group):
            out_planes[int(global_plane), :] = part[local_idx, :]

    raw = np.empty(m * n, dtype=np.uint8)
    for i in range(m):
        raw[i:m * n:m] = out_planes[i]

    return raw.view(dtype).copy()