from __future__ import annotations

from typing import Sequence
import numpy as np


def validate_decomposition(decomposition: Sequence[Sequence[int]], bytes_per_value: int) -> None:
    if not decomposition:
        raise ValueError("Decomposition cannot be empty.")

    flat = [int(i) for group in decomposition for i in group]
    expected = list(range(bytes_per_value))

    if sorted(flat) != expected:
        raise ValueError(
            f"Invalid decomposition {decomposition}. "
            f"Expected each byte index exactly once from 0 to {bytes_per_value - 1}."
        )


def values_to_byte_matrix(values: np.ndarray) -> np.ndarray:
    """
    Convert a 1D numeric array into shape (n_values, bytes_per_value) as uint8.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")

    byte_view = values.view(np.uint8)
    bytes_per_value = values.dtype.itemsize
    return byte_view.reshape(-1, bytes_per_value)


def pack_by_decomposition(values: np.ndarray, decomposition: Sequence[Sequence[int]]) -> tuple[bytes, int]:
    """
    Reorder/group bytes according to decomposition and return packed bytes
    plus packed record size.
    """
    byte_matrix = values_to_byte_matrix(values)
    bytes_per_value = values.dtype.itemsize
    validate_decomposition(decomposition, bytes_per_value)

    grouped_parts = []
    for group in decomposition:
        grouped_parts.append(byte_matrix[:, list(group)])

    packed = np.concatenate(grouped_parts, axis=1)
    return packed.tobytes(order="C"), packed.shape[1]


def unpack_by_decomposition(
    packed_bytes: bytes,
    dtype: np.dtype,
    decomposition: Sequence[Sequence[int]],
    n_values: int,
) -> np.ndarray:
    """
    Inverse of pack_by_decomposition.
    """
    dtype = np.dtype(dtype)
    bytes_per_value = dtype.itemsize
    validate_decomposition(decomposition, bytes_per_value)

    packed_record_size = sum(len(group) for group in decomposition)
    packed = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(n_values, packed_record_size)

    restored = np.empty((n_values, bytes_per_value), dtype=np.uint8)

    col_start = 0
    for group in decomposition:
        width = len(group)
        restored[:, list(group)] = packed[:, col_start:col_start + width]
        col_start += width

    return restored.reshape(-1).view(dtype)


def decomposition_to_name(decomposition: Sequence[Sequence[int]]) -> str:
    parts = []
    for group in decomposition:
        parts.append("[" + ",".join(str(i) for i in group) + "]")
    return "_".join(parts)