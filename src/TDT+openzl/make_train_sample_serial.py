# -*- coding: utf-8 -*-
"""
SDDL-based compression for decomposed byte layouts (multi-stream, matches your decomposition)
+ throughput instrumentation that measures OpenZL core time correctly
+ OPTIONAL: load and use a trained OpenZL compressor (.zlc) instead of graphs.Compress() defaults

This version fixes the trained-.zlc path for YOUR nanobind bindings:
- CCtx.select_starting_graph exists, and requires graph: openzl.ext.GraphID (not int).
- GraphID CANNOT be constructed safely from Python (nanobind "uninitialized GraphID" warning).
  Therefore, we MUST obtain a real GraphID instance from the Compressor itself (graph_ids()).
- The compressor "serialize_to_json/to_json" output is NOT strict JSON in your build,
  so we do not json.loads() it; we only regex for graph keys "#<id>:" if needed.
"""

import os
import itertools
import time
import re
from typing import List, Tuple, Dict, Any, Optional

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

N_WARMUP = 1
N_RUNS = 10
SDDL_MAX_RECORDS_PER_CHUNK = 200_000
SDDL_MBPS_DENOM = "aligned"   # "aligned" or "orig"


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
# Custom graphs used by this file (dependency registration)
# ============================================================

class _SDDLSuccessorN(zl.FunctionGraph):
    """
    Used in _OpenZLSDDLGroups.
    If a trained .zlc references this FunctionGraph, it MUST be registered before deserialize().
    """
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


def _register_dependencies(comp: "zl.Compressor") -> None:
    for ng in range(1, 9):
        try:
            comp.register_function_graph(_SDDLSuccessorN(num_groups=ng))
        except Exception:
            pass


# ============================================================
# Trained compressor loader (.zlc) for nanobind bindings
# ============================================================

def _dump_text(comp: "zl.Compressor") -> str:
    for meth in ("serialize_to_json", "to_json"):
        if hasattr(comp, meth):
            try:
                s = getattr(comp, meth)()
                if isinstance(s, str) and s:
                    return s
            except Exception:
                pass
    return ""


def _extract_graph_ids_from_dump_text(s: str) -> List[int]:
    """
    Extract ONLY graph ids from keys like: "#10": { ... }
    This avoids matching unrelated integers.
    """
    if not s:
        return []
    hits = re.findall(r'["\']?#(\d+)["\']?\s*:', s)
    gids = sorted({int(x) for x in hits if x.isdigit()})
    return gids


def _digits_in_obj_str(x: Any) -> Optional[int]:
    try:
        t = str(x)
    except Exception:
        return None
    # accepts "#10", "GraphID(#10)", etc.
    m = re.search(r"#(\d+)", t)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"\s*(\d+)\s*", t)
    if m:
        return int(m.group(1))
    return None


def _get_graph_id_objects(comp: "zl.Compressor") -> List[Any]:
    """
    Return a list of REAL GraphID objects created by the binding.
    This is essential: constructing GraphID in Python can yield an uninitialized instance.
    """
    # Most likely API
    for meth in ("graph_ids", "get_graph_ids", "list_graph_ids"):
        if hasattr(comp, meth):
            try:
                ids = getattr(comp, meth)()
                if isinstance(ids, (list, tuple)):
                    return list(ids)
            except Exception:
                pass

    # Sometimes graphs() returns mapping with GraphID keys
    for meth in ("graphs", "get_graphs", "list_graphs"):
        if hasattr(comp, meth):
            try:
                g = getattr(comp, meth)()
                if isinstance(g, dict):
                    # keys might be GraphID objects
                    keys = list(g.keys())
                    if keys:
                        return keys
            except Exception:
                pass

    return []


def _best_effort_start_gid_int(comp: "zl.Compressor") -> int:
    """
    Prefer official API. If it yields 0 or None, fallback to graph keys in dump.
    """
    # official API first
    for attr in ("get_starting_graph_id", "starting_graph_id", "get_starting_graph", "starting_graph"):
        if hasattr(comp, attr):
            try:
                v = getattr(comp, attr)
                gid = v() if callable(v) else v
                if gid is not None:
                    return int(gid)
            except Exception:
                pass

    # dump fallback
    gids = _extract_graph_ids_from_dump_text(_dump_text(comp))
    if len(gids) == 1:
        return gids[0]
    if gids:
        return gids[0]
    return 0


def _pick_real_graph_id(comp: "zl.Compressor", preferred_gid_int: int) -> Any:
    """
    Choose an initialized GraphID object from the compressor that matches preferred_gid_int.
    If no exact match exists, fallback to the only/first available GraphID object.
    """
    objs = _get_graph_id_objects(comp)
    if not objs:
        raise RuntimeError(
            "Cannot find any GraphID objects on this Compressor. "
            "Your binding must expose Compressor.graph_ids() (or similar) for trained .zlc use."
        )

    # exact match by string digits
    for g in objs:
        di = _digits_in_obj_str(g)
        if di is not None and di == int(preferred_gid_int):
            return g

    # If preferred_gid_int is 0 but compressor only has "#10", use that
    if int(preferred_gid_int) == 0 and len(objs) == 1:
        return objs[0]

    # fallback: first
    return objs[0]


class _OpenZLFromZLC:
    """
    Loads a trained OpenZL compressor from a .zlc file and prepares it for compression.

    Critical behavior for your environment:
      - Must set CCtx starting graph via CCtx.select_starting_graph(comp, GraphID_obj)
      - GraphID_obj must be an initialized instance obtained from the Compressor (NOT constructed in Python).
    """
    def __init__(self, zlc_path: str, register_deps: bool = True, debug: bool = False):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        if not zlc_path or not os.path.isfile(zlc_path):
            raise ValueError(f"zlc_path not found: {zlc_path}")

        with open(zlc_path, "rb") as f:
            blob = f.read()
        if not blob:
            raise ValueError(f"Empty .zlc file: {zlc_path}")

        self._comp = zl.Compressor()
        if register_deps:
            _register_dependencies(self._comp)

        self._comp.deserialize(blob)

        gid_int = _best_effort_start_gid_int(self._comp)

        # If API says 0 but dump shows graphs, prefer the dump gid
        dump_gids = _extract_graph_ids_from_dump_text(_dump_text(self._comp))
        if int(gid_int) == 0 and dump_gids and 0 not in dump_gids:
            gid_int = dump_gids[0]

        # Pick a REAL GraphID object from compressor that matches gid_int (or fallback)
        self._start_gid_int = int(gid_int)
        self._start_graph_id_obj = _pick_real_graph_id(self._comp, self._start_gid_int)

        # Select on compressor too (safe)
        if hasattr(self._comp, "select_starting_graph"):
            try:
                # some builds accept int here
                self._comp.select_starting_graph(int(self._start_gid_int))
            except Exception:
                pass

        if debug:
            try:
                js = _dump_text(self._comp)
                if js:
                    print("[ZLC] Loaded OK. start_gid_int =", self._start_gid_int,
                          " start_graph_obj =", str(self._start_graph_id_obj),
                          " JSON prefix:", js[:300])
            except Exception:
                pass

    @property
    def compressor(self) -> "zl.Compressor":
        return self._comp

    def bind_cctx(self, cctx: "zl.CCtx") -> None:
        """
        Bind compressor + format + set starting graph on CCtx (REQUIRED in your build).
        """
        cctx.ref_compressor(self._comp)
        cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

        if not hasattr(cctx, "select_starting_graph"):
            raise RuntimeError("CCtx.select_starting_graph missing, but your build requires it")

        # MUST pass the initialized GraphID object, not an int
        cctx.select_starting_graph(self._comp, self._start_graph_id_obj)


class _TrainedSerialTool:
    """
    Serial tool using a trained .zlc, with a persistent CCtx (do not time CCtx creation).
    """
    def __init__(self, zlc_path: str, debug: bool = True):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        self._z = _OpenZLFromZLC(zlc_path, register_deps=True, debug=debug)
        self._cctx = zl.CCtx()
        self._z.bind_cctx(self._cctx)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("expects NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("expects byte/int8/uint8")
        b = np_bytes.tobytes(order="C")
        return self._cctx.compress([zl.Input(zl.Type.Serial, b)])


class _TrainedNumericTool:
    """
    Numeric tool using a trained .zlc, with a persistent CCtx.
    """
    def __init__(self, zlc_path: str, debug: bool = True):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        self._z = _OpenZLFromZLC(zlc_path, register_deps=True, debug=debug)
        self._cctx = zl.CCtx()
        self._z.bind_cctx(self._cctx)

    def __call__(self, x: np.ndarray) -> bytes:
        if not isinstance(x, np.ndarray):
            raise TypeError("expects NumPy array")
        if x.dtype not in (np.float16, np.float32, np.float64):
            raise TypeError(f"expects float16/float32/float64, got {x.dtype}")
        return self._cctx.compress([zl.Input(zl.Type.Numeric, x)])


# ============================================================
# OpenZL compressors (baseline graph-built tools) with persistent CCtx
# ============================================================

class _OpenZLSerialCompress:
    """OpenZL: Serial -> graphs.Compress()"""
    def __init__(self):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        self._comp = zl.Compressor()
        gid = zl.graphs.Compress()(self._comp)
        self._comp.select_starting_graph(gid)

        self._cctx = zl.CCtx()
        self._cctx.ref_compressor(self._comp)
        self._cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")
        b = np_bytes.tobytes(order="C")
        return self._cctx.compress([zl.Input(zl.Type.Serial, b)])


class _OpenZLSerialZstd:
    """OpenZL: Serial -> graphs.Zstd()"""
    def __init__(self):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        self._comp = zl.Compressor()
        gid = zl.graphs.Zstd()(self._comp)
        self._comp.select_starting_graph(gid)

        self._cctx = zl.CCtx()
        self._cctx.ref_compressor(self._comp)
        self._cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")
        b = np_bytes.tobytes(order="C")
        return self._cctx.compress([zl.Input(zl.Type.Serial, b)])


class _OpenZLFloatDeconstruct:
    """
    OpenZL: Numeric(float32/float64) -> Float{32,64}Deconstruct(sign_frac=Compress, exponent=Compress)
    Fallback: if input byte-length not divisible by m, compress as Serial->Compress.
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

        self._cctx = zl.CCtx()
        self._cctx.ref_compressor(self._comp)
        self._cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

        self._fb_cctx = zl.CCtx()
        self._fb_cctx.ref_compressor(self._fallback)
        self._fb_cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")

        u8 = np_bytes.view(np.uint8)
        if u8.nbytes % self.m != 0:
            b = np_bytes.tobytes(order="C")
            return self._fb_cctx.compress([zl.Input(zl.Type.Serial, b)])

        x = u8.view(np.float32) if self.m == 4 else u8.view(np.float64)
        return self._cctx.compress([zl.Input(zl.Type.Numeric, x)])


class _OpenZLNumericOnly:
    """Numeric-only: directly compress float16/float32/float64 as zl.Type.Numeric."""
    def __init__(self):
        if not _HAS_OPENZL:
            raise RuntimeError("OpenZL not available")
        self._comp = zl.Compressor()
        gid = zl.graphs.Compress()(self._comp)
        self._comp.select_starting_graph(gid)

        self._cctx = zl.CCtx()
        self._cctx.ref_compressor(self._comp)
        self._cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

    def __call__(self, x: np.ndarray) -> bytes:
        if not isinstance(x, np.ndarray):
            raise TypeError("OpenZL numeric-only expects a NumPy array")
        if x.dtype not in (np.float16, np.float32, np.float64):
            raise TypeError(f"Numeric-only expects float16/float32/float64, got {x.dtype}")
        return self._cctx.compress([zl.Input(zl.Type.Numeric, x)])


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

        self._cctx = zl.CCtx()
        self._cctx.ref_compressor(self._comp)
        self._cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")
        b = np_bytes.tobytes(order="C")
        return self._cctx.compress([zl.Input(zl.Type.Serial, b)])


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

        self._cctx = zl.CCtx()
        self._cctx.ref_compressor(self._comp)
        self._cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")
        b = np_bytes.tobytes(order="C")
        return self._cctx.compress([zl.Input(zl.Type.Serial, b)])


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

        self._cctx = zl.CCtx()
        self._cctx.ref_compressor(self._comp)
        self._cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

    def __call__(self, np_bytes: np.ndarray) -> bytes:
        if not isinstance(np_bytes, np.ndarray):
            raise TypeError("OpenZL expects a NumPy array")
        if np_bytes.dtype not in (np.byte, np.int8, np.uint8):
            raise TypeError("OpenZL expects dtype byte/int8/uint8")
        b = np_bytes.tobytes(order="C")
        return self._cctx.compress([zl.Input(zl.Type.Serial, b)])


# ============================================================
# Decomposition configs (clustering)
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
# Decomposition enumeration helpers
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

def _core_time_and_size_serial(tool, np_byte_arrays: List[np.ndarray]) -> Tuple[int, float]:
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


def _core_time_and_size_numeric(tool, x: np.ndarray) -> Tuple[int, float]:
    out_size = len(tool(x))

    def timed():
        tool(x)

    dt = _median_time(timed)
    return int(out_size), float(dt)


def _core_time_and_size_sddl_chunked(tool, serial_bytes: np.ndarray, record_size: int, max_records: int):
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

    m = int(m)
    type_byte = np.uint8

    if given_decomp is None:
        all_possible = possible_sum(m)
        all_decomps, total_enum = find_all_combinations(all_possible, m, contig_order)
    else:
        all_decomps = list(given_decomp)
        total_enum = len(all_decomps)

    x_contig = np.ascontiguousarray(data_set)
    raw_u8 = x_contig.view(np.uint8)
    raw_i8 = x_contig.view(np.int8)
    len_bytes = int(raw_u8.nbytes)

    comps = np.zeros((m, x_contig.size), dtype=type_byte)
    for i in range(m):
        comps[i] = raw_u8[i:len_bytes:m]

    def ratio(orig: int, comp: int) -> float:
        return float("inf") if comp == 0 else float(orig) / float(comp)

    def _put(stats: dict, prefix: str, orig_bytes: int, comp_bytes: int, seconds: float, mbps_bytes: Optional[int] = None):
        stats[f"{prefix}.ratio"] = ratio(orig_bytes, comp_bytes)
        stats[f"{prefix}.time_s"] = float(seconds)
        denom = orig_bytes if mbps_bytes is None else int(mbps_bytes)
        stats[f"{prefix}.MBps"] = _mbps(denom, seconds)

    tool_serial = comp_tool_dict.get("openzl_serial", None)
    tool_numeric = comp_tool_dict.get("openzl_numeric_only", None)
    tool_float = comp_tool_dict.get(f"openzl_float_deconstruct_m{m}", None)

    struct_cache: Dict[int, Any] = {}

    def get_struct_tool(group_len: int):
        group_len = int(group_len)
        if group_len not in struct_cache:
            struct_cache[group_len] = _OpenZLSerialToStruct(group_len)
        return struct_cache[group_len]

    tool_struct_std = get_struct_tool(m)

    sddl_cache: Dict[Tuple[int, ...], Any] = {}

    def get_sddl_groups_tool(group_lens):
        key = tuple(int(x) for x in group_lens)
        if key not in sddl_cache:
            sddl_cache[key] = _OpenZLSDDLGroups(key)
        return sddl_cache[key]

    tool_sddl_std = get_sddl_groups_tool([m])

    standard_metrics: Dict[str, Any] = {}

    if tool_serial is not None:
        sz, dt = _core_time_and_size_serial(tool_serial, [raw_i8])
        _put(standard_metrics, "standard.serial", len_bytes, sz, dt)

    if tool_numeric is not None:
        sz, dt = _core_time_and_size_numeric(tool_numeric, x_contig)
        _put(standard_metrics, "standard.numeric", len_bytes, sz, dt)

    if tool_float is not None:
        sz, dt = _core_time_and_size_serial(tool_float, [raw_i8])
        _put(standard_metrics, f"standard.float_deconstruct_m{m}", len_bytes, sz, dt)

    if tool_struct_std is not None:
        sz, dt = _core_time_and_size_serial(tool_struct_std, [raw_i8])
        _put(standard_metrics, f"standard.struct_{m}", len_bytes, sz, dt)

    sz, dt, nproc = _core_time_and_size_sddl_chunked(
        tool_sddl_std, raw_i8, record_size=m, max_records=SDDL_MAX_RECORDS_PER_CHUNK
    )
    if SDDL_MBPS_DENOM == "orig":
        _put(standard_metrics, f"standard.sddl_{m}", len_bytes, sz, dt, mbps_bytes=len_bytes)
    else:
        _put(standard_metrics, f"standard.sddl_{m}", len_bytes, sz, dt, mbps_bytes=nproc)

    stat_array = []
    packed_struct_cache: Dict[int, Any] = {}

    def get_packed_struct_tool(K: int):
        K = int(K)
        if K not in packed_struct_cache:
            packed_struct_cache[K] = _OpenZLPackedStruct(K)
        return packed_struct_cache[K]

    for idx, decomp in enumerate(all_decomps):
        stats = {
            "dataset": dataset_name,
            "original_bytes": len_bytes,
            "m": m,
            "N": int(x_contig.size),
            "decomposition": tuple_to_string(decomp),
            "chunk_no": int(chunk_no),
        }
        stats.update(standard_metrics)

        comp_list = []
        group_lens = []
        for group_tuple in decomp:
            group_tuple = tuple(group_tuple)
            gl = len(group_tuple)
            group_lens.append(gl)

            cur = np.zeros((gl, x_contig.size), dtype=type_byte)
            for j, byte_idx in enumerate(group_tuple):
                cur[j] = comps[int(byte_idx)]
            comp_list.append(cur)

        K = int(sum(group_lens))

        decomp_serial_F = [np.frombuffer(chunk.flatten("F").tobytes(), dtype=np.int8) for chunk in comp_list]
        decomp_serial_C = [np.frombuffer(chunk.flatten("C").tobytes(), dtype=np.int8) for chunk in comp_list]

        packed = np.zeros((K, x_contig.size), dtype=type_byte)
        off = 0
        for chunk in comp_list:
            gl = int(chunk.shape[0])
            packed[off:off + gl, :] = chunk
            off += gl

        packed_bytes_F = np.frombuffer(packed.flatten("F").tobytes(), dtype=np.int8)
        packed_bytes_C = np.frombuffer(packed.flatten("C").tobytes(), dtype=np.int8)

        if tool_serial is not None:
            szF, dtF = _core_time_and_size_serial(tool_serial, decomp_serial_F)
            szC, dtC = _core_time_and_size_serial(tool_serial, decomp_serial_C)
            _put(stats, "decomp.serial.col", len_bytes, szF, dtF)
            _put(stats, "decomp.serial.row", len_bytes, szC, dtC)

        struct_sum_F = 0
        struct_sum_C = 0
        t_struct_F = 0.0
        t_struct_C = 0.0

        for chunk_bytes, chunk in zip(decomp_serial_F, comp_list):
            gl = int(chunk.shape[0])
            tool = get_struct_tool(gl)
            sz, dt = _core_time_and_size_serial(tool, [chunk_bytes])
            struct_sum_F += sz
            t_struct_F += dt

        for chunk_bytes, chunk in zip(decomp_serial_C, comp_list):
            gl = int(chunk.shape[0])
            tool = get_struct_tool(gl)
            sz, dt = _core_time_and_size_serial(tool, [chunk_bytes])
            struct_sum_C += sz
            t_struct_C += dt

        _put(stats, "decomp.struct_groups.col", len_bytes, struct_sum_F, t_struct_F)
        _put(stats, "decomp.struct_groups.row", len_bytes, struct_sum_C, t_struct_C)

        packed_tool = get_packed_struct_tool(K)
        szF, dtF = _core_time_and_size_serial(packed_tool, [packed_bytes_F])
        szC, dtC = _core_time_and_size_serial(packed_tool, [packed_bytes_C])
        _put(stats, "decomp.struct_packed.col", len_bytes, szF, dtF)
        _put(stats, "decomp.struct_packed.row", len_bytes, szC, dtC)

        sddl_tool = get_sddl_groups_tool(group_lens)

        szF, dtF, nprocF = _core_time_and_size_sddl_chunked(
            sddl_tool, packed_bytes_F, record_size=K, max_records=SDDL_MAX_RECORDS_PER_CHUNK
        )
        szC, dtC, nprocC = _core_time_and_size_sddl_chunked(
            sddl_tool, packed_bytes_C, record_size=K, max_records=SDDL_MAX_RECORDS_PER_CHUNK
        )

        if SDDL_MBPS_DENOM == "orig":
            _put(stats, "decomp.sddl.col", len_bytes, szF, dtF, mbps_bytes=len_bytes)
            _put(stats, "decomp.sddl.row", len_bytes, szC, dtC, mbps_bytes=len_bytes)
        else:
            _put(stats, "decomp.sddl.col", len_bytes, szF, dtF, mbps_bytes=nprocF)
            _put(stats, "decomp.sddl.row", len_bytes, szC, dtC, mbps_bytes=nprocC)

        stat_array.append(stats)

        if out_log_dir and (((idx + 1) % 20 == 0) or ((idx + 1) == total_enum)):
            os.makedirs(out_log_dir, exist_ok=True)
            out_csv = os.path.join(out_log_dir, f"{dataset_name}_decomposition_stats.csv")
            pd.DataFrame(stat_array).to_csv(out_csv, index=False)

    return stat_array


# ============================================================
# main()
# ============================================================

def main():
    dataset_folder = "/home/jamalids/Documents/2D/data1/Fcbench/Fcbench-dataset/32/34"
    m = 4
    contig_order = False

    TRAINED_ZLC_SERIAL = os.environ.get("OPENZL_TRAINED_ZLC_SERIAL", "").strip()
    TRAINED_ZLC_NUMERIC = os.environ.get("OPENZL_TRAINED_ZLC_NUMERIC", "").strip()

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

        if TRAINED_ZLC_SERIAL:
            comp_tool_dict["openzl_serial"] = _TrainedSerialTool(TRAINED_ZLC_SERIAL, debug=True)
            print(f"[INFO] Using trained SERIAL compressor: {TRAINED_ZLC_SERIAL}")

        if TRAINED_ZLC_NUMERIC:
            comp_tool_dict["openzl_numeric_only"] = _TrainedNumericTool(TRAINED_ZLC_NUMERIC, debug=True)
            print(f"[INFO] Using trained NUMERIC compressor: {TRAINED_ZLC_NUMERIC}")

        if m in (4, 8):
            comp_tool_dict[f"openzl_float_deconstruct_m{m}"] = _OpenZLFloatDeconstruct(m=m)

        _ = test_decomposition(
            data_set=sliced_data,
            dataset_name=dataset_name,
            m=m,
            comp_tool_dict=comp_tool_dict,
            given_decomp=converted_decomps,
            contig_order=contig_order,
            out_log_dir="/home/jamalids/Documents/6",
        )

        print(f"Done: wrote stats in /home/jamalids/Documents/6/{dataset_name}_decomposition_stats.csv")


if __name__ == "__main__":
    main()