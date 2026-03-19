#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shlex
import subprocess
import time
from pathlib import Path


def run_capture(cmd: list[str]) -> subprocess.CompletedProcess:
    # captures stdout/stderr for logging/debug if needed
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def human_bytes(n: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} PiB"


def make_train_sample(
    src_file: Path,
    dst_sample: Path,
    max_bytes: int,
    mode: str,
) -> int:
    """
    Create a limited-size training sample file from src_file.
    mode:
      - head: first max_bytes
      - middle: centered window of max_bytes
      - tail: last max_bytes
    Returns: bytes written.
    """
    size = src_file.stat().st_size
    dst_sample.parent.mkdir(parents=True, exist_ok=True)

    take = min(max_bytes, size)
    if take <= 0:
        raise ValueError("max_bytes must be > 0")

    if mode == "head":
        offset = 0
    elif mode == "tail":
        offset = max(0, size - take)
    elif mode == "middle":
        offset = max(0, (size // 2) - (take // 2))
    else:
        raise ValueError(f"unknown mode: {mode}")

    with src_file.open("rb") as fsrc, dst_sample.open("wb") as fdst:
        fsrc.seek(offset)
        remaining = take
        buf = 4 * 1024 * 1024
        while remaining > 0:
            chunk = fsrc.read(min(buf, remaining))
            if not chunk:
                break
            fdst.write(chunk)
            remaining -= len(chunk)

    return dst_sample.stat().st_size


def safe_name(p: Path) -> str:
    # stable name for outputs
    return p.stem


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-dataset: build a limited training sample, train .zlc, compress full file with trained model, write one CSV row per dataset."
    )
    ap.add_argument("--zli", required=True, help="Path to zli binary, e.g. ./cmakebuild/cli/zli")
    ap.add_argument("--profile", default="serial", help="Profile for training (default: serial)")

    ap.add_argument("--in-dir", required=True, help="Folder containing datasets (files)")
    ap.add_argument("--glob", default="*", help="File glob inside in-dir (default: *)")

    ap.add_argument("--out-dir", required=True, help="Output folder (zlc/, samples/, compressed/, results.csv)")
    ap.add_argument("--max-train-mib", type=int, default=64, help="Max train sample size in MiB (default: 64)")
    ap.add_argument("--sample-mode", choices=["head", "middle", "tail"], default="head", help="Which part of file to sample (default: head)")

    ap.add_argument("--skip-existing", action="store_true", help="Skip dataset if outputs already exist")
    ap.add_argument("--csv", default="results.csv", help="CSV filename inside out-dir (default: results.csv)")

    args = ap.parse_args()

    zli = Path(args.zli)
    if not zli.exists():
        raise SystemExit(f"zli not found: {zli}")

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"in-dir not a directory: {in_dir}")

    out_dir = Path(args.out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "zlc").mkdir(parents=True, exist_ok=True)
    (out_dir / "compressed").mkdir(parents=True, exist_ok=True)

    max_bytes = int(args.max_train_mib) * 1024 * 1024
    csv_path = out_dir / args.csv

    files = sorted(in_dir.glob(args.glob))
    files = [p for p in files if p.is_file()]
    if not files:
        raise SystemExit("No input files matched. Check --glob.")

    # CSV header
    if not csv_path.exists():
        csv_path.write_text(
            "dataset,file,orig_bytes,train_sample_bytes,train_seconds,compress_seconds,compressed_bytes,ratio\n",
            encoding="utf-8",
        )

    for f in files:
        ds = safe_name(f)
        sample_path = out_dir / "samples" / f"{ds}.train_sample"
        zlc_path = out_dir / "zlc" / f"{ds}.{args.profile}.zlc"
        out_comp = out_dir / "compressed" / f"{ds}.{args.profile}.trained.zl"

        if args.skip_existing and zlc_path.exists() and out_comp.exists():
            print(f"[skip] {f.name} (exists)")
            continue

        orig_bytes = f.stat().st_size
        sample_bytes = make_train_sample(f, sample_path, max_bytes=max_bytes, mode=args.sample_mode)

        print(f"\n[dset] {f.name}")
        print(f"  orig:   {human_bytes(orig_bytes)}")
        print(f"  sample: {human_bytes(sample_bytes)} ({args.sample_mode})")

        # 1) train
        train_cmd = [
            str(zli),
            "train",
            "--profile", args.profile,
            str(sample_path),
            "--output", str(zlc_path),
        ]
        t0 = time.perf_counter()
        proc_train = run_capture(train_cmd)
        t1 = time.perf_counter()
        train_s = t1 - t0

        # 2) compress
        comp_cmd = [
            str(zli),
            "compress",
            "--zlc", str(zlc_path),
            str(f),
            "--output", str(out_comp),
        ]
        t2 = time.perf_counter()
        proc_comp = run_capture(comp_cmd)
        t3 = time.perf_counter()
        comp_s = t3 - t2

        comp_bytes = out_comp.stat().st_size
        ratio = (orig_bytes / comp_bytes) if comp_bytes > 0 else 0.0

        # write one row per dataset
        line = (
            f"{ds},{f.name},{orig_bytes},{sample_bytes},{train_s:.6f},{comp_s:.6f},{comp_bytes},{ratio:.6f}\n"
        )
        with csv_path.open("a", encoding="utf-8") as w:
            w.write(line)

        print(f"  zlc: {zlc_path}")
        print(f"  out: {out_comp}  ({human_bytes(comp_bytes)})  ratio={ratio:.3f}")
        print(f"  train_s={train_s:.3f}  compress_s={comp_s:.3f}")

        # If you want to keep logs, uncomment:
        # (out_dir / "logs").mkdir(exist_ok=True)
        # (out_dir / "logs" / f"{ds}.train.stdout").write_text(proc_train.stdout, encoding="utf-8")
        # (out_dir / "logs" / f"{ds}.train.stderr").write_text(proc_train.stderr, encoding="utf-8")
        # (out_dir / "logs" / f"{ds}.compress.stdout").write_text(proc_comp.stdout, encoding="utf-8")
        # (out_dir / "logs" / f"{ds}.compress.stderr").write_text(proc_comp.stderr, encoding="utf-8")

    print(f"\n[done] CSV: {csv_path}")


if __name__ == "__main__":
    main()