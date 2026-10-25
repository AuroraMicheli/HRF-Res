"""
Reservoir characterisation (Sec. 4.1, Figs. 2-3, Appendix B).

    python scripts/characterise.py configs/characterisation/forda.yaml
    python scripts/characterise.py configs/characterisation/forda.yaml --hann
    python scripts/characterise.py configs/characterisation/forda.yaml --cka_sizes 100 200 500 1000 0
    python scripts/characterise.py configs/characterisation/forda.yaml --washout 0 0.1 0.25 0.5

For every seed, an HRF-Res and a LIF-Res reservoir are built and analysed:
  - CKA between temporal-mean reservoir states and class labels (Fig. 3A);
  - frequency sweep: preferred frequency f* and Q-factor of every neuron
    (Figs. 2 and 3B). ``--hann`` applies a Hann window before the DFT.
The two CKA controls of Appendix B use one fixed reservoir (the first seed):
``--cka_sizes`` recomputes CKA on random subsets of the test set (0 = full
test set) and ``--washout`` discards an initial fraction of each trajectory.
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
from hrf_res.data import load_dataset, to_sequence
from hrf_res.models import HRFRes
from hrf_res.utils import build_model, set_seed

MODELS = ("hrf_res", "lif_res")


def trajectory(model, u):
    """Yield the reservoir state used for readout (y for HRF-Res, the membrane
    potential for LIF-Res) at every timestep."""
    B, T, _ = u.shape
    zeros = lambda: torch.zeros(B, model.n_hid, device=u.device)
    if isinstance(model, HRFRes):
        x, y, v, s, ref = zeros(), zeros(), zeros(), zeros(), zeros()
        for t in range(T):
            x, y, v, s, ref, _ = model.step(u[:, t], x, y, v, s, ref)
            yield y
    else:
        r, s, v = zeros(), zeros(), zeros()
        for t in range(T):
            r, s, v, _ = model.step(u[:, t], r, s, v)
            yield r


@torch.no_grad()
def mean_states(model, loader, dataset, device, max_samples=None, washout=(0.0,)):
    """Temporal mean of the reservoir state for the first ``max_samples`` test
    samples, after discarding each fraction in ``washout`` of the trajectory.
    Returns ({fraction: features}, labels)."""
    feats = {w: [] for w in washout}
    labels, n = [], 0
    for x, y in loader:
        if max_samples is not None and n >= max_samples:
            break
        u = to_sequence(dataset, x.to(device))
        T = u.shape[1]
        start = {w: int(round(w * T)) for w in washout}
        sums = {w: 0.0 for w in washout}
        for t, state in enumerate(trajectory(model, u)):
            for w in washout:
                if t >= start[w]:
                    sums[w] = sums[w] + state
        for w in washout:
            feats[w].append((sums[w] / (T - start[w])).cpu().numpy())
        labels.append(y.reshape(-1).numpy())
        n += u.shape[0]
    feats = {w: np.concatenate(f)[:max_samples] for w, f in feats.items()}
    return feats, np.concatenate(labels)[:max_samples]


def cka(feats, labels):
    """Linear CKA between row-normalised features and the label kernel."""
    X = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    Y = (labels[:, None] == labels[None, :]).astype(float)
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kx, Ky = H @ (X @ X.T) @ H, H @ Y @ H
    hxy, hxx, hyy = np.sum(Kx * Ky), np.sum(Kx * Kx), np.sum(Ky * Ky)
    if hxx < 1e-10 or hyy < 1e-10:
        return 0.0
    return float(hxy / np.sqrt(hxx * hyy))


@torch.no_grad()
def frequency_sweep(model, dt, n_inp, freqs, T, device, hann=False):
    """Steady-state response power of every neuron to a sinusoid broadcast to
    all input channels, at each frequency. Returns an (n_freqs, N) array."""
    t_vec = torch.arange(T, dtype=torch.float32, device=device)
    responses = np.zeros((len(freqs), model.n_hid))
    for i, f in enumerate(freqs):
        u = torch.sin(2 * np.pi * f * t_vec * dt).reshape(1, T, 1).expand(1, T, n_inp)
        states = torch.stack(list(trajectory(model, u)), dim=1)[0].cpu().numpy()
        steady = states[T // 2:]
        if hann:
            window = np.hanning(steady.shape[0])
            power = np.abs(np.fft.rfft(steady * window[:, None], axis=0)) ** 2 / np.sum(window ** 2)
        else:
            power = np.abs(np.fft.rfft(steady, axis=0)) ** 2
        fft_freqs = np.fft.rfftfreq(steady.shape[0], d=dt)
        responses[i] = power[np.argmin(np.abs(fft_freqs - f))]
    return responses


def q_factor(freqs, responses):
    """Per-neuron Q = f* / half-power bandwidth, with the -3 dB crossings
    interpolated on a log-frequency axis. Q = 0 if either crossing is missing."""
    log_f = np.log10(freqs)
    Q = np.zeros(responses.shape[1])
    for j in range(responses.shape[1]):
        r = responses[:, j]
        if r.max() <= 0:
            continue
        pk = np.argmax(r)
        half = r[pk] / 2.0
        lower = upper = None
        for k in range(pk - 1, -1, -1):
            if r[k] <= half:
                frac = (half - r[k]) / (r[k + 1] - r[k]) if r[k + 1] - r[k] > 1e-12 else 0.0
                lower = log_f[k] + frac * (log_f[k + 1] - log_f[k])
                break
        for k in range(pk + 1, len(freqs)):
            if r[k] <= half:
                if r[k - 1] - r[k] > 1e-12:
                    upper = log_f[k - 1] + (r[k - 1] - half) / (r[k - 1] - r[k]) * (log_f[k] - log_f[k - 1])
                else:
                    upper = log_f[k]
                break
        if lower is not None and upper is not None:
            bandwidth = 10 ** upper - 10 ** lower
            Q[j] = freqs[pk] / bandwidth if bandwidth > 0 else 0.0
    return Q


def build_pair(cfg, seed, n_inp, device):
    models = {}
    for name in MODELS:
        set_seed(seed)
        models[name] = build_model(cfg[name], n_inp, "mean", device)
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--hann", action="store_true", help="Hann window before the DFT")
    parser.add_argument("--skip", nargs="*", default=[], choices=["cka", "sweep"])
    parser.add_argument("--cka_sizes", nargs="*", type=int, help="CKA sample-size control")
    parser.add_argument("--n_repeats", type=int, default=20)
    parser.add_argument("--washout", nargs="*", type=float, help="CKA washout control")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset, seeds = cfg["dataset"], cfg["seeds"]
    _, test_loader, info = load_dataset(dataset, args.data_dir, batch_test=cfg.get("batch_test", 128),
                                        num_steps=cfg.get("num_steps"),
                                        spatial_factor=cfg.get("spatial_factor"))
    sw = cfg["sweep"]
    freqs = np.logspace(np.log10(sw["f_min"]), np.log10(sw["f_max"]), sw.get("n_freqs", 30))
    out = {"config": cfg, "hann": args.hann, "seeds": []}

    for seed in seeds:
        models = build_pair(cfg, seed, info["n_inp"], device)
        res = {"seed": seed}
        for name, model in models.items():
            if "cka" not in args.skip:
                feats, labels = mean_states(model, test_loader, dataset, device, cfg["cka_samples"])
                res[f"cka_{name}"] = cka(feats[0.0], labels)
            if "sweep" not in args.skip:
                responses = frequency_sweep(model, cfg[name]["dt"], info["n_inp"], freqs,
                                            sw["T"], device, args.hann)
                Q = q_factor(freqs, responses)
                pref = freqs[np.argmax(responses, axis=0)]
                res[f"resonant_{name}"] = float((Q > 0).mean())
                res[f"q_mean_{name}"] = float(Q[Q > 0].mean()) if (Q > 0).any() else 0.0
                res[f"pref_freq_std_{name}"] = float(pref.std())
                res[f"pref_freq_{name}"] = pref.tolist()
                res[f"q_{name}"] = Q.tolist()
        print({k: round(v, 4) for k, v in res.items() if isinstance(v, float)})
        out["seeds"].append(res)

    if args.cka_sizes or args.washout:
        models = build_pair(cfg, seeds[0], info["n_inp"], device)
        rng = np.random.RandomState(seeds[0])
        for name, model in models.items():
            if args.cka_sizes:
                feats, labels = mean_states(model, test_loader, dataset, device)
                rows = {}
                for n in args.cka_sizes:
                    if n == 0 or n >= len(labels):
                        rows["full"] = [cka(feats[0.0], labels)]
                    else:
                        rows[n] = [cka(feats[0.0][idx], labels[idx]) for idx in
                                   (rng.choice(len(labels), n, replace=False) for _ in range(args.n_repeats))]
                out.setdefault("cka_sizes", {})[name] = {str(n): (float(np.mean(v)), float(np.std(v)))
                                                         for n, v in rows.items()}
            if args.washout:
                feats, labels = mean_states(model, test_loader, dataset, device, cfg["cka_samples"],
                                            washout=tuple(args.washout))
                out.setdefault("washout", {})[name] = {str(w): cka(f, labels) for w, f in feats.items()}
        print({k: out[k] for k in ("cka_sizes", "washout") if k in out})

    config_path = Path(args.config)
    suffix = "_hann" if args.hann else ""
    out_file = Path(args.out_dir) / config_path.parent.name / f"{config_path.stem}{suffix}.json"
    os.makedirs(out_file.parent, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out, f)
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
