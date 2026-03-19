from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from .datasets import load_tsv_column
from .benchmark import benchmark_all_decompositions, rows_to_dicts


def load_decompositions(config_path: str | Path, dataset_name: str) -> list[list[list[int]]]:
    """
    Load decompositions for a dataset from YAML.

    Expected YAML shape:
        dataset_name:
          - [[0], [1], [2], [3]]
          - [[0,1], [2], [3]]

        default:
          - [[0], [1], [2], [3]]
    """
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML in {config_path}: expected a mapping at top level.")

    if dataset_name in cfg:
        decomps = cfg[dataset_name]
    elif "default" in cfg:
        decomps = cfg["default"]
    else:
        raise KeyError(
            f"No decomposition config found for dataset '{dataset_name}', "
            f"and no 'default' entry exists in {config_path}."
        )

    if not isinstance(decomps, list):
        raise ValueError(
            f"Invalid decomposition entry for dataset '{dataset_name}': expected a list."
        )

    return decomps


def write_rows_to_csv(rows: list[dict], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark standard and grouped OpenZL SDDL layouts for one dataset."
    )

    parser.add_argument(
        "--dataset-file",
        required=True,
        help="Path to the input TSV file.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Logical dataset name used to select decompositions from YAML.",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--value-column",
        type=int,
        default=1,
        help="Zero-based TSV column index containing numeric values. Default: 1",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="NumPy dtype for the value column, e.g. float32 or float64. Default: float32",
    )
    parser.add_argument(
        "--skip-header",
        action="store_true",
        help="Skip the first TSV row.",
    )
    parser.add_argument(
        "--decomp-config",
        required=True,
        help="Path to YAML file containing decomposition candidates.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of timed runs per measurement. Default: 5",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=1,
        help="Number of warmup runs before timing. Default: 1",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    dataset_file = Path(args.dataset_file)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    decompositions = load_decompositions(
        config_path=args.decomp_config,
        dataset_name=args.dataset_name,
    )

    values = load_tsv_column(
        file_path=dataset_file,
        value_column=args.value_column,
        dtype=args.dtype,
        skip_header=args.skip_header,
    )

    rows = benchmark_all_decompositions(
        dataset_name=args.dataset_name,
        values=values,
        decompositions=decompositions,
        n_runs=args.n_runs,
        n_warmup=args.n_warmup,
    )

    row_dicts = rows_to_dicts(rows)
    write_rows_to_csv(row_dicts, args.out_csv)

    print(f"Loaded {values.size} values from {dataset_file}")
    print(f"Applied {len(decompositions)} grouped decomposition(s) plus 1 baseline")
    print(f"Wrote results to {args.out_csv}")


if __name__ == "__main__":
    main()