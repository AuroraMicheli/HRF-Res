"""
Random hyperparameter search (Appendix A).

    python scripts/search.py configs/search/forda_hrf_res.yaml

Each sampled configuration is evaluated with a single seed on a stratified
10% validation split of the training set; the test set is never used.
Continuous ranges spanning at least an order of magnitude are sampled
log-uniformly, the others uniformly; ``{choice: [...]}`` samples from a set.
All trials are written to outputs/search/<name>.json and the best
configuration to outputs/search/<name>_best.yaml, ready for train.py.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hrf_res.data import load_dataset, pixel_permutation, cifar_noise
from hrf_res.readout import extract_features, fit_readout
from hrf_res.utils import build_model, set_seed


def sample(space, rng):
    params = {}
    for key, spec in space.items():
        if isinstance(spec, dict):
            params[key] = rng.choice(spec["choice"])
        else:
            lo, hi = spec
            if lo > 0 and hi / lo >= 10:
                params[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                params[key] = float(rng.uniform(lo, hi))
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--n_configs", type=int, help="override the number of sampled configurations")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        search = yaml.safe_load(f)
    with open(search["base"]) as f:
        base = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = base["dataset"]

    train_loader, _, info = load_dataset(dataset, args.data_dir, base.get("batch_train", 128), base.get("batch_test", 128),
                                         num_steps=base.get("num_steps"), spatial_factor=base.get("spatial_factor"),
                                         max_time=base.get("max_time", 1.4))
    # fixed sample order, so that the validation split is the same for every trial
    loader = data.DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False,
                             collate_fn=train_loader.collate_fn, num_workers=train_loader.num_workers)
    perm = pixel_permutation(base.get("perm_seed", 0), device) if dataset == "psmnist" else None
    noise = cifar_noise(info["seq_length"], device) if dataset == "npcifar10" else None

    rng = random.Random(args.seed)
    trials, split = [], None
    for i in range(args.n_configs or search.get("n_configs", 100)):
        cfg = {**base, **sample(search["space"], rng)}
        set_seed(args.seed)
        model = build_model(cfg, info["n_inp"], cfg.get("readout_mode", "final"), device)
        x, y, _ = extract_features(model, loader, dataset, device, perm, noise)
        if split is None:
            split = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=0)
        tr, va = split
        train_acc, valid_acc = fit_readout(x[tr], y[tr], x[va], y[va], C=cfg.get("readout_C", 1.0),
                                           solver=cfg.get("readout_solver", "lbfgs"),
                                           max_iter=cfg.get("readout_max_iter", 2000))
        trials.append({"params": {k: cfg[k] for k in search["space"]}, "train_acc": train_acc, "valid_acc": valid_acc})
        print(f"[{i + 1}] valid {valid_acc:.2f}%  {trials[-1]['params']}")

    trials.sort(key=lambda t: t["valid_acc"], reverse=True)
    name = Path(args.config).stem
    out_dir = Path(args.out_dir) / "search"
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(trials, f, indent=2)
    best = {**base, **trials[0]["params"]}
    with open(out_dir / f"{name}_best.yaml", "w") as f:
        yaml.safe_dump(best, f, sort_keys=False)
    print(f"best validation accuracy {trials[0]['valid_acc']:.2f}%, config saved to {out_dir / (name + '_best.yaml')}")


if __name__ == "__main__":
    main()
