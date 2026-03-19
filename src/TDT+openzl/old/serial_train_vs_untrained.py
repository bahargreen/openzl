# -*- coding: utf-8 -*-
import os
import re
import time
import argparse
from typing import Any, Optional, List, Tuple

import numpy as np
import pandas as pd

try:
    import openzl.ext as zl
    _HAS_OPENZL = True
except Exception:
    zl = None
    _HAS_OPENZL = False


def _median_time(fn, warmup: int, runs: int) -> float:
    for _ in range(max(0, int(warmup))):
        fn()
    ts = []
    for _ in range(max(1, int(runs))):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(ts, dtype=np.float64)))


def _mbps(nbytes: int, seconds: float) -> float:
    return float("inf") if seconds <= 0.0 else (float(nbytes) / float(seconds) / 1e6)


def _dump_json(comp: "zl.Compressor") -> str:
    for meth in ("serialize_to_json", "to_json"):
        if hasattr(comp, meth):
            try:
                s = getattr(comp, meth)()
                if isinstance(s, str) and s:
                    return s
            except Exception:
                pass
    return ""


def _extract_graph_ints(s: str) -> List[int]:
    # Matches "#10": { ... }
    hits = re.findall(r'["\']?#(\d+)["\']?\s*:', s or "")
    return sorted({int(x) for x in hits if x.isdigit()})


def _looks_like_graphid_obj(x: Any) -> bool:
    # nanobind wrappers usually stringify like "#10" or "GraphID(#10)"
    try:
        t = str(x)
    except Exception:
        return False
    return ("#" in t) and any(ch.isdigit() for ch in t)


def _probe_trained_graphid_obj(comp: "zl.Compressor", debug: bool) -> Optional[Any]:
    """
    Best-effort: try to obtain a real GraphID object.
    Works only if comp.select_starting_graph(i) returns a GraphID instance.
    """
    if not hasattr(comp, "select_starting_graph"):
        return None

    js = _dump_json(comp)
    cand = _extract_graph_ints(js)
    # add a small fallback range
    for i in range(0, 33):
        if i not in cand:
            cand.append(i)
    cand = list(dict.fromkeys(cand))

    last_err = None
    for gid_int in cand:
        try:
            ret = comp.select_starting_graph(int(gid_int))
            if ret is not None and _looks_like_graphid_obj(ret):
                if debug:
                    print(f"[trained] Compressor.select_starting_graph({gid_int}) -> {ret}")
                return ret
        except Exception as e:
            last_err = e

    if debug and last_err is not None:
        print("[trained] probing failed; last error:", last_err)
    return None


def _load_tsv_column_as_serial_bytes(tsv_path: str, col: int, dtype: str, sep: str) -> Tuple[bytes, int]:
    df = pd.read_csv(tsv_path, sep=sep)
    if df.shape[1] <= col:
        raise RuntimeError(f"--col {col} out of range. ncols={df.shape[1]}")

    v = df.values[:, col]
    if dtype == "f16":
        x = v.astype(np.float16, copy=False)
    elif dtype == "f32":
        x = v.astype(np.float32, copy=False)
    elif dtype == "f64":
        x = v.astype(np.float64, copy=False)
    else:
        raise RuntimeError("dtype must be f16/f32/f64")

    x = np.ascontiguousarray(x)
    b = x.view(np.uint8).tobytes(order="C")  # one-time conversion; keep out of timing
    return b, len(b)


class SerialUntrained:
    """
    graphs.Compress() on Serial.
    CRITICAL: set graph on CCtx via CCtx.select_starting_graph(comp, GraphID).
    """
    def __init__(self, debug: bool = False):
        self._comp = zl.Compressor()
        gid_obj = zl.graphs.Compress()(self._comp)   # often GraphID object in py binding

        # also set on compressor (fine; may be int/GraphID depending on binding)
        if hasattr(self._comp, "select_starting_graph"):
            try:
                self._comp.select_starting_graph(gid_obj)
            except Exception:
                pass

        self._cctx = zl.CCtx()
        self._cctx.ref_compressor(self._comp)
        self._cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

        # MUST set graph on CCtx in your build
        if not hasattr(self._cctx, "select_starting_graph"):
            raise RuntimeError("Your CCtx has no select_starting_graph; cannot run in this binding.")
        try:
            self._cctx.select_starting_graph(self._comp, gid_obj)
        except Exception as e:
            raise RuntimeError(
                f"Failed to set CCtx starting graph for UNTRAINED. "
                f"gid_obj={gid_obj} type={type(gid_obj)} error={e}"
            )

        if debug:
            print("[untrained] gid_obj:", gid_obj, "type:", type(gid_obj))

    def compress_bytes(self, b: bytes) -> bytes:
        return self._cctx.compress([zl.Input(zl.Type.Serial, b)])


class SerialTrained:
    """
    Uses trained .zlc.
    CRITICAL: need a real GraphID object to call CCtx.select_starting_graph.
    """
    def __init__(self, zlc_path: str, debug: bool = False):
        if not zlc_path or not os.path.isfile(zlc_path):
            raise ValueError(f"zlc_path not found: {zlc_path}")

        blob = open(zlc_path, "rb").read()
        if not blob:
            raise ValueError("Empty .zlc")

        self._comp = zl.Compressor()
        self._comp.deserialize(blob)

        self._cctx = zl.CCtx()
        self._cctx.ref_compressor(self._comp)
        self._cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

        if not hasattr(self._cctx, "select_starting_graph"):
            raise RuntimeError("CCtx.select_starting_graph not exposed; trained unusable in this binding.")

        graph_id_obj = _probe_trained_graphid_obj(self._comp, debug=debug)
        if graph_id_obj is None:
            js = _dump_json(self._comp)
            prefix = js[:250] if js else "<no json available>"
            raise RuntimeError(
                "Cannot obtain a GraphID object from this trained .zlc with current Python bindings.\n"
                "Your binding likely does NOT expose Compressor.graph_ids() (or equivalent), and "
                "Compressor.select_starting_graph(i) did not return a GraphID.\n"
                f"JSON prefix:\n{prefix}"
            )

        # must set graph on CCtx
        self._cctx.select_starting_graph(self._comp, graph_id_obj)

        if debug:
            print("[trained] graph_id_obj:", graph_id_obj, "type:", type(graph_id_obj))

    def compress_bytes(self, b: bytes) -> bytes:
        return self._cctx.compress([zl.Input(zl.Type.Serial, b)])


def _bench(label: str, tool, b: bytes, nbytes: int, warmup: int, runs: int) -> None:
    # one untimed call to get compressed size
    comp_size = len(tool.compress_bytes(b))

    def timed():
        tool.compress_bytes(b)

    dt = _median_time(timed, warmup=warmup, runs=runs)
    ratio = float("inf") if comp_size == 0 else float(nbytes) / float(comp_size)

    print(label)
    print(f"  orig={nbytes}  comp={comp_size}  ratio={ratio:.4f}")
    print(f"  time_s={dt:.6f}  MB/s={_mbps(nbytes, dt):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-tsv", required=True)
    ap.add_argument("--col", type=int, default=1)
    ap.add_argument("--sep", type=str, default="\t")
    ap.add_argument("--dtype", choices=["f16", "f32", "f64"], default="f32")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not _HAS_OPENZL:
        raise SystemExit("OpenZL not available in this environment.")

    b, nbytes = _load_tsv_column_as_serial_bytes(args.input_tsv, args.col, args.dtype, args.sep)
    print(f"[input] {args.input_tsv}")
    print(f"[bytes] {nbytes}")

    untrained = SerialUntrained(debug=args.debug)
    _bench("UNTRAINED Serial (graphs.Compress)", untrained, b, nbytes, args.warmup, args.runs)

    zlc = os.environ.get("OPENZL_TRAINED_ZLC_SERIAL", "").strip()
    if zlc:
        try:
            trained = SerialTrained(zlc, debug=args.debug)
            _bench(f"TRAINED Serial (.zlc): {zlc}", trained, b, nbytes, args.warmup, args.runs)
        except Exception as e:
            print("[TRAINED] FAILED:", str(e))


if __name__ == "__main__":
    main()