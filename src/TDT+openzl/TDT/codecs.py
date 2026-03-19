from __future__ import annotations

from typing import Sequence
import numpy as np

from . import openzl_sddl_helpers as sddlh

print("HELPER FILE =", sddlh.__file__)
print("HAS FUNC =", hasattr(sddlh, "get_standard_sddl_tool"))


def compress_standard_sddl(raw_bytes: bytes, bytes_per_value: int) -> bytes:
    tool = sddlh.get_standard_sddl_tool(int(bytes_per_value))
    raw_u8 = np.frombuffer(raw_bytes, dtype=np.uint8)
    return tool(raw_u8)


def decompress_standard_sddl(compressed_bytes: bytes) -> bytes:
    return sddlh._decompress_serial_bytes(compressed_bytes)


def compress_grouped_sddl(grouped_bytes: bytes, group_lens: Sequence[int]) -> bytes:
    tool = sddlh.get_grouped_sddl_tool(tuple(int(x) for x in group_lens))
    grouped_u8 = np.frombuffer(grouped_bytes, dtype=np.uint8)
    return tool(grouped_u8)


def decompress_grouped_sddl(compressed_bytes: bytes) -> bytes:
    return sddlh._decompress_serial_bytes(compressed_bytes)