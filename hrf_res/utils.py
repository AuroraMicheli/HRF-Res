import inspect
import random

import numpy as np
import torch

from .models import HRFRes, LIFRes, RON, apply_sparse_input_projection

MODELS = {"hrf_res": HRFRes, "lif_res": LIFRes, "ron": RON}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg, n_inp, readout_mode, device):
    """Instantiate the model described by a config dict (``cfg["model"]``),
    including the sparse input projection when ``input_density < 1``."""
    cls = MODELS[cfg["model"]]
    params = inspect.signature(cls).parameters
    kwargs = {k: v for k, v in cfg.items() if k in params}
    if "readout_mode" in params:
        kwargs["readout_mode"] = readout_mode
    model = cls(n_inp=n_inp, device=device, **kwargs).to(device)
    if cfg.get("input_density", 1.0) < 1.0:
        apply_sparse_input_projection(model, cfg["input_density"], n_inp, cfg["n_hid"], device)
    return model
