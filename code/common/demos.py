"""Loading the demonstration datasets defined in configs/datasets.yaml."""

import numpy as np
import yaml

from common.env_utils import ROOT

BASELINE_PATH = ROOT / "data" / "demos_baseline.npz"


def dataset_indices():
    return yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())


def load_demos(name="baseline_100", path=BASELINE_PATH):
    """Returns {key: array[N, ...]} for the episodes of dataset `name`."""
    idx = np.asarray(dataset_indices()[name])
    with np.load(path) as f:
        return {k: f[k][idx] for k in f.files}
