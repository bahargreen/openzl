from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from .layout import (
    pack_by_decomposition,
    unpack_by_decomposition,
    decomposition_to_name,
)
from .utils import median_runtime, mb_per_sec
from .codecs import (
    compress_standard_sddl,
    decompress_standard_sddl,
    compress_grouped_sddl,
    decompress_grouped_sddl,
)


@dataclass
class BenchmarkRow:
    dataset: str
    method: str
    decomposition_name: str
    dtype: str
    n_values: int
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    transform_time_s: float
    inverse_transform_time_s: float
    encode_time_codec_s: float
    decode_time_codec_s: float
    encode_time_end_to_end_s: float
    decode_time_end_to_end_s: float
    encode_throughput_mb_s: float
    decode_throughput_mb_s: float
    roundtrip_ok: bool


def benchmark_standard(
    dataset_name: str,
    values: np.ndarray,
    n_runs: int,
    n_warmup: int,
) -> BenchmarkRow:
    """
    Benchmark standard SDDL on the original byte layout.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    raw_bytes = values.tobytes(order="C")
    original_bytes = len(raw_bytes)
    bytes_per_value = values.dtype.itemsize

    encode_time, compressed = median_runtime(
        lambda: compress_standard_sddl(raw_bytes, bytes_per_value),
        n_runs=n_runs,
        n_warmup=n_warmup,
    )

    decode_time, restored_bytes = median_runtime(
        lambda: decompress_standard_sddl(compressed),
        n_runs=n_runs,
        n_warmup=n_warmup,
    )

    roundtrip_ok = restored_bytes == raw_bytes

    return BenchmarkRow(
        dataset=dataset_name,
        method="standard_sddl",
        decomposition_name="none",
        dtype=str(values.dtype),
        n_values=int(values.size),
        original_bytes=int(original_bytes),
        compressed_bytes=int(len(compressed)),
        compression_ratio=(float(original_bytes) / float(len(compressed))) if len(compressed) else 0.0,
        transform_time_s=0.0,
        inverse_transform_time_s=0.0,
        encode_time_codec_s=float(encode_time),
        decode_time_codec_s=float(decode_time),
        encode_time_end_to_end_s=float(encode_time),
        decode_time_end_to_end_s=float(decode_time),
        encode_throughput_mb_s=float(mb_per_sec(original_bytes, encode_time)),
        decode_throughput_mb_s=float(mb_per_sec(original_bytes, decode_time)),
        roundtrip_ok=bool(roundtrip_ok),
    )


def benchmark_grouped(
    dataset_name: str,
    values: np.ndarray,
    decomposition: Sequence[Sequence[int]],
    n_runs: int,
    n_warmup: int,
) -> BenchmarkRow:
    """
    Benchmark grouped / transformed SDDL using a byte-plane decomposition.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    original_bytes = int(values.nbytes)
    decomposition_name = decomposition_to_name(decomposition)
    group_lens = [len(group) for group in decomposition]

    transform_time, packed_result = median_runtime(
        lambda: pack_by_decomposition(values, decomposition),
        n_runs=n_runs,
        n_warmup=n_warmup,
    )
    grouped_bytes, _packed_record_size = packed_result

    encode_time, compressed = median_runtime(
        lambda: compress_grouped_sddl(grouped_bytes, group_lens),
        n_runs=n_runs,
        n_warmup=n_warmup,
    )

    decode_time, decoded_grouped_bytes = median_runtime(
        lambda: decompress_grouped_sddl(compressed),
        n_runs=n_runs,
        n_warmup=n_warmup,
    )

    inverse_time, restored_values = median_runtime(
        lambda: unpack_by_decomposition(
            decoded_grouped_bytes,
            dtype=values.dtype,
            decomposition=decomposition,
            n_values=int(values.size),
        ),
        n_runs=n_runs,
        n_warmup=n_warmup,
    )

    roundtrip_ok = np.array_equal(restored_values, values)

    return BenchmarkRow(
        dataset=dataset_name,
        method="grouped_sddl",
        decomposition_name=decomposition_name,
        dtype=str(values.dtype),
        n_values=int(values.size),
        original_bytes=int(original_bytes),
        compressed_bytes=int(len(compressed)),
        compression_ratio=(float(original_bytes) / float(len(compressed))) if len(compressed) else 0.0,
        transform_time_s=float(transform_time),
        inverse_transform_time_s=float(inverse_time),
        encode_time_codec_s=float(encode_time),
        decode_time_codec_s=float(decode_time),
        encode_time_end_to_end_s=float(transform_time + encode_time),
        decode_time_end_to_end_s=float(decode_time + inverse_time),
        encode_throughput_mb_s=float(mb_per_sec(original_bytes, transform_time + encode_time)),
        decode_throughput_mb_s=float(mb_per_sec(original_bytes, decode_time + inverse_time)),
        roundtrip_ok=bool(roundtrip_ok),
    )


def benchmark_all_decompositions(
    dataset_name: str,
    values: np.ndarray,
    decompositions: Sequence[Sequence[Sequence[int]]],
    n_runs: int,
    n_warmup: int,
) -> list[BenchmarkRow]:
    """
    Run baseline + all grouped decompositions for one dataset.
    """
    rows: list[BenchmarkRow] = []

    rows.append(
        benchmark_standard(
            dataset_name=dataset_name,
            values=values,
            n_runs=n_runs,
            n_warmup=n_warmup,
        )
    )

    for decomposition in decompositions:
        rows.append(
            benchmark_grouped(
                dataset_name=dataset_name,
                values=values,
                decomposition=decomposition,
                n_runs=n_runs,
                n_warmup=n_warmup,
            )
        )

    return rows


def row_to_dict(row: BenchmarkRow) -> dict:
    return asdict(row)


def rows_to_dicts(rows: Sequence[BenchmarkRow]) -> list[dict]:
    return [asdict(row) for row in rows]