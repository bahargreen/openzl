# TDT for OpenZL SDDL

This repository studies whether a byte-plane transformation can improve OpenZL SDDL compression for floating-point data by changing the byte layout presented to the codec before encoding.

The current implementation focuses on one-dimensional floating-point columns. It compares the original byte layout against transformed layouts obtained by grouping or reordering byte planes, then measures compression ratio, codec runtime, and end-to-end reconstruction cost.

## Core idea

For each floating-point value, the bytes are first exposed explicitly as byte planes. A candidate decomposition then defines how these byte planes are grouped or reordered before being serialized and compressed with OpenZL SDDL.

The workflow is:

1. Read a numeric column from a dataset.
2. Interpret each value as a fixed-width byte sequence.
3. Build a transformed layout using a user-provided byte-plane decomposition.
4. Compress the transformed byte stream with OpenZL SDDL.
5. Decompress and reconstruct the original values exactly.
6. Report compression ratio, codec encode/decode time, and end-to-end timing including transformation overhead.

This makes it possible to test whether layout-aware preprocessing improves compression behavior relative to a standard SDDL baseline.

## Repository structure

```text
TDT+openzl/
  TDT/
    __init__.py
    cli1.py
    benchmark.py
    codecs.py
    datasets.py
    layout.py
    utils.py
    openzl_sddl_helpers.py
    configs/
      decompositions.yaml