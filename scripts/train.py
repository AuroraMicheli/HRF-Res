"""
Train and evaluate a reservoir from a config file (Tables 1-3, Fig. 4).

    python scripts/train.py configs/table1/smnist_hrf_res.yaml
    python scripts/train.py configs/fig4/forda.yaml --data_dir /path/to/data

For each trial k the reservoir is drawn with seed ``seed + k``, features are
extracted on the training and test sets, and a logistic-regression readout is
fitted on the training features. Configs with a list of ``readout_modes``
evaluate every mode on the same reservoir. Results are written as JSON to
``outputs/<config folder>/<config name>.json``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hrf_res import energy
from hrf_res.data import load_dataset, pixel_permutation, cifar_noise
from hrf_res.readout import extract_features, fit_readout
from hrf_res.utils import build_model, set_seed


def run_trial(cfg, model, train_loader, test_loader, info, device, perm, noise):
    name = cfg["dataset"]
    train_x, train_y, rates = extract_features(model, train_loader, name, device, perm, noise)
    test_x, test_y, _ = extract_features(model, test_loader, name, device, perm, noise)
    train_acc, test_acc = fit_readout(train_x, train_y, test_x, test_y,
                                      C=cfg.get("readout_C", 1.0),
                                      solver=cfg.get("readout_solver", "lbfgs"),
                                      max_iter=cfg.get("readout_max_iter", 2000))
    T = info["seq_length"]
    if cfg["model"] == "ron":
        e = energy.dense_energy(info["n_inp"], cfg["n_hid"], T)
    else:
        e = energy.spiking_energy(rates["r_hrf"], rates["r_lif"], cfg["n_hid"], T,
                                  model.n_lif2hrf_connections)
    return {"train_acc": train_acc, "test_acc": test_acc, "energy_J": e["Energy_J"], **rates}


def summarise(trials):
    out = {"trials": trials}
    for key in trials[0]:
        values = np.array([t[key] for t in trials])
        out[f"{key}_mean"], out[f"{key}_std"] = float(values.mean()), float(values.std())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--n_trials", type=int, help="override the number of trials in the config")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    n_trials = args.n_trials or cfg.get("n_trials", 5)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    modes = cfg.get("readout_modes", [cfg.get("readout_mode", "final")])

    train_loader, test_loader, info = load_dataset(
        cfg["dataset"], args.data_dir, cfg.get("batch_train", 128), cfg.get("batch_test", 128),
        num_steps=cfg.get("num_steps"), spatial_factor=cfg.get("spatial_factor"),
        max_time=cfg.get("max_time", 1.4))
    perm = pixel_permutation(cfg.get("perm_seed", 0), device) if cfg["dataset"] == "psmnist" else None
    noise = cifar_noise(info["seq_length"], device) if cfg["dataset"] == "npcifar10" else None

    results = {mode: [] for mode in modes}
    for k in range(n_trials):
        for mode in modes:
            # re-seeding per mode keeps the same reservoir for all readout modes
            set_seed(cfg.get("seed", 42) + k)
            model = build_model(cfg, info["n_inp"], mode, device)
            res = run_trial(cfg, model, train_loader, test_loader, info, device, perm, noise)
            results[mode].append(res)
            print(f"trial {k + 1}/{n_trials}  {mode:22s} test acc {res['test_acc']:.2f}%  "
                  f"energy {res['energy_J']:.3e} J")

    summary = {"config": cfg, "n_trials": n_trials,
               "results": {mode: summarise(r) for mode, r in results.items()}}
    for mode, s in summary["results"].items():
        print(f"{mode:22s} test acc {s['test_acc_mean']:.2f} +/- {s['test_acc_std']:.2f}%  "
              f"energy {s['energy_J_mean']:.3e} J")

    config_path = Path(args.config)
    out_file = Path(args.out_dir) / config_path.parent.name / f"{config_path.stem}.json"
    os.makedirs(out_file.parent, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
