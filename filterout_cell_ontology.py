import argparse
import csv
import os
from glob import glob, escape as glob_escape
from typing import Iterable, List, Set


def normalize_text(value: str) -> str:
    """Normalize cell type text for matching.

    Current rule: strip surrounding whitespace. Keep case sensitivity as-is.
    """
    return value.strip()


def read_ontology_terms(ontology_csv_path: str, encoding: str = "utf-8") -> Set[str]:
    """Read ontology terms from `cl_term_map.csv` and return a set of strings.

    Expected columns: `key` and `cl_label`.
    Rows with empty values are ignored.
    """
    term_set: Set[str] = set()

    if not os.path.isfile(ontology_csv_path):
        raise FileNotFoundError(
            f"Ontology CSV not found: {ontology_csv_path}"
        )

    with open(ontology_csv_path, mode="r", newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(
                f"Ontology CSV missing header: {ontology_csv_path}"
            )
        # Remove BOM if present in the first header
        reader.fieldnames = [
            (name.lstrip("\ufeff") if name is not None else name)
            for name in reader.fieldnames
        ]

        for row in reader:
            key_val = row.get("key")
            label_val = row.get("cl_label")
            if key_val is not None and key_val.strip() != "":
                term_set.add(normalize_text(key_val))
            if label_val is not None and label_val.strip() != "":
                term_set.add(normalize_text(label_val))

    return term_set


def find_final_csvs(input_dir: str) -> List[str]:
    """Find `3-final-*.csv` directly under the given directory (non-recursive).

    Important: escape the directory portion to treat literal special chars like
    '[' and ']' in folder names.
    """
    safe_dir = glob_escape(input_dir)
    pattern = os.path.join(safe_dir, "3-final-*.csv")
    return sorted(glob(pattern))


def compute_output_path(final_csv_path: str) -> str:
    """Compute output path: same folder, name becomes `4-final-filtered-*.csv`.

    If input is `.../3-final-XYZ.csv`, output becomes `.../4-final-filtered-XYZ.csv`.
    If the input does not start with `3-final-`, the prefix is not assumed and the
    whole name is appended after `4-final-filtered-`.
    """
    parent = os.path.dirname(final_csv_path)
    base = os.path.basename(final_csv_path)
    prefix = "3-final-"
    rest = base[len(prefix):] if base.startswith(prefix) else base
    return os.path.join(parent, f"4-final-filtered-{rest}")


def filter_final_csv(
    final_csv_path: str,
    ontology_terms: Set[str],
    encoding: str = "utf-8",
) -> int:
    """Filter rows from a final CSV where `full_cell_type_name` is in ontology terms.

    Returns the number of rows removed.
    """
    output_path = compute_output_path(final_csv_path)

    with open(final_csv_path, mode="r", newline="", encoding=encoding) as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError(
                f"Final CSV missing header: {final_csv_path}"
            )
        # Clean potential BOM
        reader.fieldnames = [
            (name.lstrip("\ufeff") if name is not None else name)
            for name in reader.fieldnames
        ]

        fieldnames = reader.fieldnames
        if fieldnames is None or "full_cell_type_name" not in fieldnames:
            print(
                f"[WARN] Skip file without 'full_cell_type_name' column: {final_csv_path}"
            )
            return 0

        rows_out: List[dict] = []
        removed_count = 0

        for row in reader:
            cell_type_value = row.get("full_cell_type_name", "")
            normalized_value = normalize_text(cell_type_value) if cell_type_value is not None else ""

            if normalized_value and normalized_value in ontology_terms:
                removed_count += 1
                continue
            rows_out.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="w", newline="", encoding=encoding) as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(
        f"[INFO] Filtered {removed_count} rows -> {os.path.relpath(output_path, start=os.path.dirname(final_csv_path))}"
    )
    return removed_count


def run(
    input_dir: str,
    ontology_csv_path: str,
    encoding: str = "utf-8",
) -> None:
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    print(f"[INFO] Scanning directory: {input_dir}")
    final_csv_paths = find_final_csvs(input_dir)
    if not final_csv_paths:
        print("[WARN] No files matched pattern '3-final-*.csv' in the directory.")
        return

    print(f"[INFO] Loading ontology terms from: {ontology_csv_path}")
    ontology_terms = read_ontology_terms(ontology_csv_path, encoding=encoding)
    print(f"[INFO] Loaded {len(ontology_terms)} ontology terms")

    total_removed = 0
    for csv_path in final_csv_paths:
        removed = filter_final_csv(csv_path, ontology_terms, encoding=encoding)
        total_removed += removed

    print(f"[INFO] Done. Total removed rows across files: {total_removed}")


def parse_args(argv: Iterable[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter out rows from 3-final-*.csv where full_cell_type_name appears in "
            "the ontology CSV (either 'key' or 'cl_label'). Output files are saved "
            "as 4-final-filtered-*.csv in the same folders."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing 3-final-*.csv files (non-recursive)",
    )
    parser.add_argument(
        "--ontology-csv",
        default=os.path.join("./cell_ontology", "cl_term_map.csv"),
        help="Path to ontology CSV with columns 'key' and 'cl_label'",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for reading/writing CSVs (default: utf-8)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(
        input_dir=args.input_dir,
        ontology_csv_path=args.ontology_csv,
        encoding=args.encoding,
    )