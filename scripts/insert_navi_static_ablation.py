#!/usr/bin/env python3
"""Insert audited static w/o SliceOpt metrics into the final ablation CSV."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


CIRCUITS = (
    "BV",
    "SwapTest",
    "Permutation",
    "MCMT",
    "Comparator",
    "QFTAdder",
    "CuccaroAdder",
    "PQC",
    "IQP",
    "Random",
    "QV",
    "VQE",
)
METRICS = (
    "F_norm",
    "Cumulative Error",
    "EPR Pair Consumption",
)
NEW_COLUMN = "CommOpt+MapOpt"


def _read_static_values(path: Path) -> dict[tuple[str, str], str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_circuit: dict[str, dict[str, str]] = {}
    config_fingerprints: set[str] = set()
    for row in rows:
        circuit = str(row.get("Circuit", ""))
        if circuit in by_circuit:
            raise ValueError(f"duplicate static result for {circuit}: {path}")
        if str(row.get("Variant ID")) != "static_commopt_mapopt":
            raise ValueError(f"unexpected variant for {circuit}: {row.get('Variant ID')!r}")
        if str(row.get("Full Span Safeguard Enabled", "")).lower() != "false":
            raise ValueError(f"full-span safeguard must be disabled for {circuit}")
        fingerprint = str(row.get("Config Fingerprint", ""))
        if not fingerprint:
            raise ValueError(f"missing config fingerprint for {circuit}")
        config_fingerprints.add(fingerprint)
        by_circuit[circuit] = row

    if set(by_circuit) != set(CIRCUITS):
        missing = sorted(set(CIRCUITS) - set(by_circuit))
        extra = sorted(set(by_circuit) - set(CIRCUITS))
        raise ValueError(f"static results are incomplete: missing={missing}, extra={extra}")
    if len(config_fingerprints) != 1:
        raise ValueError(
            "static results must share one config fingerprint, got "
            f"{sorted(config_fingerprints)}"
        )

    return {
        (circuit, metric): str(by_circuit[circuit][metric])
        for circuit in CIRCUITS
        for metric in METRICS
    }


def insert_static_results(target_csv: Path, static_csv: Path) -> None:
    raw = target_csv.read_bytes()
    if not raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"target CSV must retain its UTF-8 BOM: {target_csv}")
    if raw.count(b"\r\n") != raw.count(b"\n"):
        raise ValueError(f"target CSV must use CRLF line endings: {target_csv}")

    with target_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        original_fields = list(reader.fieldnames or [])
        original_rows = list(reader)

    required_fields = {"Circuit", "Metrics", "SliceOpt"}
    if not required_fields.issubset(original_fields):
        raise ValueError(
            f"target CSV is missing fields {sorted(required_fields - set(original_fields))}"
        )
    keys = [(row["Circuit"], row["Metrics"]) for row in original_rows]
    expected_keys = {(circuit, metric) for circuit in CIRCUITS for metric in METRICS}
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError("target CSV must contain exactly the expected 12x3 rows")

    static_values = _read_static_values(static_csv)
    if NEW_COLUMN in original_fields:
        output_fields = original_fields
        preserved_fields = [field for field in original_fields if field != NEW_COLUMN]
    else:
        insert_at = original_fields.index("SliceOpt") + 1
        output_fields = original_fields[:insert_at] + [NEW_COLUMN] + original_fields[insert_at:]
        preserved_fields = original_fields

    output_rows = []
    for original in original_rows:
        updated = dict(original)
        updated[NEW_COLUMN] = static_values[(original["Circuit"], original["Metrics"])]
        output_rows.append(updated)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=target_csv.parent,
        prefix=f".{target_csv.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=output_fields,
                lineterminator="\r\n",
            )
            writer.writeheader()
            writer.writerows(output_rows)
            handle.flush()
            os.fsync(handle.fileno())

        with open(temporary_name, newline="", encoding="utf-8-sig") as handle:
            verification_reader = csv.DictReader(handle)
            verification_fields = list(verification_reader.fieldnames or [])
            verification_rows = list(verification_reader)
        if verification_fields != output_fields:
            raise RuntimeError("output field verification failed")
        for before, after in zip(original_rows, verification_rows):
            if any(before[field] != after[field] for field in preserved_fields):
                raise RuntimeError("an existing ablation value changed during insertion")
        os.replace(temporary_name, target_csv)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-csv",
        type=Path,
        default=Path("outputs/navi_ablation_new.csv"),
    )
    parser.add_argument(
        "--static-csv",
        type=Path,
        default=Path(
            "outputs/navi_static_ablation_legacy_cat_no_fullspan/variant_results.csv"
        ),
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    insert_static_results(arguments.target_csv, arguments.static_csv)
    print(
        f"[STATIC_INSERT_DONE] target={arguments.target_csv} "
        f"source={arguments.static_csv}",
        flush=True,
    )
