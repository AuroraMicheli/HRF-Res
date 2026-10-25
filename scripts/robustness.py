"""
Robustness to test-time perturbations (Appendix C).

    python scripts/robustness.py configs/robustness/forda_hrf_res.yaml

The reservoir and readout are trained on clean data as in train.py and then
evaluated, without retraining, on perturbed test inputs:
  - additive Gaussian noise with standard deviation ``level * std(input)``,
    for every level in ``noise_levels``;
  - linear resampling of the input sequence to round(T * rate) steps, for
    every rate in ``rates``.
Noise draws depend only on (seed, level), so HRF-Res and LIF-Res see the
same perturbed inputs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hrf_res.data import load_dataset, to_sequence
from hrf_res.utils import build_model, set_seed


def resample(u, rate):
    """Linear interpolation of (B, T, C) inputs to round(T * rate) steps."""
    if rate == 1.0:
        return u
    new_T = max(2, int(round(u.shape[1] * rate)))
    return F.interpolate(u.permute(0, 2, 1), size=new_T, mode="linear", align_corners=False).permute(0, 2, 1)


@torch.no_grad()
def features(model, loader, dataset, device, sigma=0.0, noise_seed=None, rate=1.0):
    if noise_seed is not None:
        torch.manual_seed(noise_seed)
    feats, labels = [], []
    for x, y in loader:
        u = to_sequence(dataset, x.to(device))
        if sigma > 0:
            u = u + torch.randn_like(u) * sigma
        f, _ = model(resample(u, rate))
        feats.append(f.cpu())
        labels.append(y.reshape(-1))
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = cfg["dataset"]
    train_loader, test_loader, info = load_dataset(dataset, args.data_dir, cfg["batch_train"], cfg["batch_test"])
    input_std = torch.cat([to_sequence(dataset, x).flatten() for x, _ in train_loader]).std().item()

    trials = []
    for k in range(cfg.get("n_trials", 5)):
        seed = cfg.get("seed", 42) + k
        set_seed(seed)
        model = build_model(cfg, info["n_inp"], cfg.get("readout_mode", "final"), device)
        train_x, train_y = features(model, train_loader, dataset, device)
        scaler = preprocessing.StandardScaler().fit(train_x)
        clf = LogisticRegression(C=cfg.get("readout_C", 1.0), max_iter=2000, n_jobs=-1)
        clf.fit(scaler.transform(train_x), train_y)

        def score(**perturbation):
            x, y = features(model, test_loader, dataset, device, **perturbation)
            return 100 * clf.score(scaler.transform(x), y)

        res = {"seed": seed,
               "noise": {str(lv): score(sigma=lv * input_std, noise_seed=900000 + seed * 1000 + int(round(lv * 1000)))
                         for lv in cfg.get("noise_levels", [])},
               "rate": {str(r): score(rate=r) for r in cfg.get("rates", [])}}
        print(res)
        trials.append(res)

    summary = {"config": cfg, "input_std": input_std, "trials": trials}
    for kind in ("noise", "rate"):
        summary[kind] = {lv: (float(np.mean([t[kind][lv] for t in trials])), float(np.std([t[kind][lv] for t in trials])))
                         for lv in trials[0][kind]}
        for lv, (m, s) in summary[kind].items():
            print(f"{kind} {lv:>5}: {m:.2f} +/- {s:.2f}%")

    config_path = Path(args.config)
    out_file = Path(args.out_dir) / config_path.parent.name / f"{config_path.stem}.json"
    os.makedirs(out_file.parent, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
