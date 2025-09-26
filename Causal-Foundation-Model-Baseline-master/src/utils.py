import os
import sys
import csv
import pickle

import numpy as np

from datetime import datetime


def _ensure_dir(fp):
    directory = os.path.dirname(fp)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_pickle(fp, data):
    """Serialize inference and training outputs.

    The function name is kept for backward compatibility, but the
    implementation now dispatches on the requested file extension. This lets
    us save results as either ``.pkl`` (via :mod:`pickle`) or ``.npy``/``.npz``
    (via :mod:`numpy`).
    """

    _ensure_dir(fp)
    ext = os.path.splitext(fp)[1].lower()

    if ext == ".npy":
        # Allow saving arbitrary Python objects such as dicts via NumPy.
        np.save(fp, data, allow_pickle=True)
    elif ext == ".npz":
        if not isinstance(data, dict):
            raise ValueError("Saving to .npz expects a dict-like object")
        np.savez(fp, **data)
    else:
        with open(fp, "wb+") as f:
            pickle.dump(data, f)


def read_pickle(fp):
    ext = os.path.splitext(fp)[1].lower()

    if ext == ".npy":
        data = np.load(fp, allow_pickle=True)
        # Arrays saved with allow_pickle=True return an object ndarray. When we
        # stored a single Python object (e.g., a dict), ``np.load`` returns a
        # 0-d array that exposes the item via ``.item()``. Fallback to the
        # raw array otherwise.
        if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
            return data.item()
        return data
    elif ext == ".npz":
        with np.load(fp, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    else:
        with open(fp, "rb") as f:
            data = pickle.load(f)
        return data


def read_csv(fp, fieldnames=None, delimiter=',', str_keys=[]):
    data = []
    with open(fp) as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        # iterate and append
        for item in reader:
            data.append(item)
    return data


def csv_has_manifest_header(fp):
    """Return ``True`` if ``fp`` looks like a SEA manifest CSV.

    A manifest enumerates dataset assets via ``fp_data``/``fp_graph``/``split``
    columns. When users provide a raw tabular CSV we can detect that it lacks
    these columns and adapt accordingly.
    """

    try:
        with open(fp, newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
    except FileNotFoundError:
        return False

    if not header:
        return False

    normalized = {col.strip().lower() for col in header if col}
    required = {"fp_data", "fp_graph", "split"}
    return required.issubset(normalized)


def load_tabular_csv(fp):
    """Load a numeric matrix from a generic CSV file.

    The helper trims common pandas-style index columns, removes fully empty
    columns (for example those created by trailing commas), and returns a
    NumPy ``float32`` array ready for downstream processing.
    """

    with open(fp, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, [])

    data = np.genfromtxt(fp, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if header:
        first = header[0].strip().lower()
        if first in {"", "index", "unnamed: 0"}:
            data = data[:, 1:]

    if data.size == 0:
        raise ValueError(f"No numeric columns detected in {fp}")

    if np.isnan(data).any():
        valid_cols = ~np.all(np.isnan(data), axis=0)
        data = data[:, valid_cols]

    return data.astype(np.float32, copy=False)


def materialize_manifest_for_csv(csv_path, output_root, algorithm):
    """Create a SEA-compatible manifest for a raw tabular CSV.

    Parameters
    ----------
    csv_path: str
        Path to the user-provided CSV containing raw observational data.
    output_root: str
        Directory where the intermediate ``.npy`` assets and manifest should be
        written. The folder is created if it does not already exist.
    algorithm: str
        Name of the inference algorithm. Interventional methods ("gies")
        require a regimes file even when all rows are observational, so we
        synthesise one on the fly.

    Returns
    -------
    str
        Absolute path to the generated manifest CSV.
    """

    csv_path = os.path.abspath(csv_path)
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    dataset_dir = os.path.join(output_root, base)
    os.makedirs(dataset_dir, exist_ok=True)

    data = load_tabular_csv(csv_path)
    num_rows, num_cols = data.shape

    fp_data = os.path.join(dataset_dir, "data.npy")
    fp_graph = os.path.join(dataset_dir, "graph.npy")
    fp_regime = os.path.join(dataset_dir, "regimes.csv")
    manifest_fp = os.path.join(dataset_dir, "manifest.csv")

    np.save(fp_data, data)
    np.save(fp_graph, np.zeros((num_cols, num_cols), dtype=np.int64))

    needs_regime = algorithm == "gies"
    if needs_regime:
        with open(fp_regime, "w", newline="") as f:
            for _ in range(num_rows):
                f.write("\n")
    else:
        # Ensure the file exists for consistency even if unused.
        open(fp_regime, "w").close()

    with open(manifest_fp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fp_data", "fp_graph", "fp_regime", "split"])
        writer.writerow([fp_data, fp_graph, fp_regime if needs_regime else "", "test"])

    return manifest_fp

# -------- general

def get_timestamp():
    return datetime.now().strftime('%H:%M:%S')


def printt(*args, **kwargs):
    print(get_timestamp(), *args, **kwargs)


def get_suffix(metric):
    suffix = "model_best_"
    suffix = suffix + "{global_step}_{epoch}_{"
    suffix = suffix + metric + ":.3f}_{val_loss:.3f}"
    return suffix

