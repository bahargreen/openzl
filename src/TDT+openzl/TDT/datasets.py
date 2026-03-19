from __future__ import annotations

from pathlib import Path
import numpy as np


def load_tsv_column(
    file_path: str | Path,
    value_column: int = 1,
    dtype: str = "float32",
    skip_header: bool = False,
) -> np.ndarray:
    file_path = Path(file_path)

    data = np.loadtxt(
        file_path,
        delimiter="\t",
        skiprows=1 if skip_header else 0,
        usecols=[value_column],
        dtype=dtype,
    )

    if data.ndim == 0:
        data = np.array([data], dtype=dtype)

    return data.astype(dtype, copy=False)