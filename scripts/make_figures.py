"""
Figures 2-4 of the paper from the outputs of characterise.py and train.py.

    python scripts/make_figures.py                 # reads outputs/, writes figures/

Fig. 2 (bandpass_summary) and Fig. 3 (repr_analysis_combined) need
outputs/characterisation/<dataset>.json; Fig. 4 (readout_ablation) needs
outputs/fig4/<dataset>.json. Bars show mean +/- std over seeds; the
preferred-frequency histograms of Fig. 3B use the first seed. Datasets
without outputs are skipped.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATASETS = ["smnist", "forda", "shd", "dvs_gesture"]
LABELS = {"smnist": "sMNIST", "forda": "FordA", "shd": "SHD", "dvs_gesture": "DVS\nGesture"}
TITLES = {"smnist": "sMNIST", "forda": "FordA", "shd": "SHD", "dvs_gesture": "DVS Gesture"}
HRF_COLOR, LIF_COLOR = "#1A78C2", "#E8402A"
MODELS = (("hrf_res", "HRF-Res", HRF_COLOR), ("lif_res", "LIF-Res", LIF_COLOR))
ERROR_KW = dict(ecolor="#333333", elinewidth=0.8, capsize=2.5, capthick=0.8)

READOUTS = [("final", "Final", "#1f4e79"), ("mean", "Mean", "#2e75b6"), ("rms", "RMS", "#5ba3d9"),
            ("std", "Std", "#9dc3e6"), ("rms_std_final", "Combined (RMS+Std+Final)", "#70ad47"),
            ("spikes_mean", "Spikes (mean rate)", "#c55a11")]


def paper_style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02})


def load(folder, suffix=""):
    out = {}
    for ds in DATASETS:
        path = folder / f"{ds}{suffix}.json"
        if path.exists():
            with open(path) as f:
                out[ds] = json.load(f)
    return out


def stat(res, key, scale=1.0):
    values = scale * np.array([s[key] for s in res["seeds"]])
    return values.mean(), values.std()


def bar_pair(ax, char, key, scale=1.0):
    """Grouped HRF-Res/LIF-Res bars with error bars; returns the bar handles."""
    ds = list(char)
    x, width, handles = np.arange(len(ds)), 0.35, []
    for offset, (model, _, color) in zip((-width / 2, width / 2), MODELS):
        m, s = zip(*(stat(char[d], f"{key}_{model}", scale) for d in ds))
        handles.append(ax.bar(x + offset, m, width, yerr=s, color=color, edgecolor="white",
                              linewidth=0.5, error_kw=ERROR_KW))
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[d] for d in ds])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(handles, [m[1] for m in MODELS], frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    return handles


def fig2(char, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.4))
    plt.subplots_adjust(wspace=0.35, left=0.10, right=0.97, top=0.87, bottom=0.30)
    bar_pair(axes[0], char, "resonant", 100)
    axes[0].set_ylabel("Band-pass neurons (%)")
    axes[0].yaxis.set_label_coords(-0.18, 0.40)
    axes[0].set_title("Fraction of resonant neurons")
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    bar_pair(axes[1], char, "q_mean")
    axes[1].set_ylabel("Mean Q-factor")
    axes[1].set_title("Band-pass sharpness")
    top = max(sum(stat(c, f"q_mean_{m}")) for c in char.values() for m, _, _ in MODELS)
    axes[1].set_ylim(0, top * 1.30)
    for ax, letter in zip(axes, "AB"):
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")
    save(fig, out_dir, "bandpass_summary")


def fig3(char, out_dir):
    fig = plt.figure(figsize=(9.5, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.8], wspace=0.14, left=0.08, right=0.98, top=0.87, bottom=0.30)
    ax = fig.add_subplot(gs[0])
    bar_pair(ax, char, "cka")
    ax.set_ylabel("CKA score")
    ax.set_title("Class-label alignment")
    top = max(sum(stat(c, f"cka_{m}")) for c in char.values() for m, _, _ in MODELS)
    ax.set_ylim(0, top * 1.30)
    ax.text(-0.22, 1.06, "A", transform=ax.transAxes, fontsize=13, fontweight="bold")

    axes = [fig.add_subplot(gs[1].subgridspec(1, 4, wspace=0.10)[0, i]) for i in range(4)]
    f_min = min(c["config"]["sweep"]["f_min"] for c in char.values())
    f_max = max(c["config"]["sweep"]["f_max"] for c in char.values())
    bins = np.logspace(np.log10(f_min), np.log10(f_max), 30)
    for i, (ax, ds) in enumerate(zip(axes, DATASETS)):
        if ds not in char:
            ax.set_visible(False)
            continue
        first = char[ds]["seeds"][0]
        for model, _, color in MODELS:
            f = np.array(first[f"pref_freq_{model}"])
            w = np.ones_like(f) / len(f)
            ax.hist(f, bins=bins, weights=w, color=color, alpha=0.45)
            ax.hist(f, bins=bins, weights=w, histtype="step", color=color, linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xlim(f_min / 3, f_max)
        ax.set_title(TITLES[ds])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Fraction of neurons")
            ax.text(-0.42, 1.06, "B", transform=ax.transAxes, fontsize=13, fontweight="bold")
    axes[-1].legend(handles=[Patch(facecolor=c, alpha=0.7, label=l) for _, l, c in MODELS], frameon=False,
                    loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    mid_x = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    fig.text(mid_x, 0.08, "Preferred frequency (Hz)", ha="center", fontsize=10)
    save(fig, out_dir, "repr_analysis_combined")


def fig4(abl, out_dir):
    fig, axes = plt.subplots(1, len(abl), figsize=(5 * len(abl), 3.5), squeeze=False)
    fig.subplots_adjust(wspace=0.25, left=0.04, right=0.99, top=0.93, bottom=0.18)
    width = 0.13
    x = np.array(list(range(len(READOUTS) - 1)) + [len(READOUTS) - 1 + 0.08 / width]) * width
    x -= x.mean()
    for i, (ax, (ds, res)) in enumerate(zip(axes[0], abl.items())):
        means = [res["results"][m]["test_acc_mean"] for m, _, _ in READOUTS]
        stds = [res["results"][m]["test_acc_std"] for m, _, _ in READOUTS]
        bars = ax.bar(x, means, width=width * 0.88, color=[c for _, _, c in READOUTS],
                      edgecolor="white", linewidth=0.5, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5,
                    capthick=0.8, zorder=4)
        best = bars[int(np.argmax(means))]
        best.set_edgecolor("black")
        best.set_linewidth(1.5)
        ax.axvline((x[-2] + x[-1]) / 2, color="gray", linewidth=0.6, linestyle="--", zorder=2)
        ax.set_ylim(min(means) - max(stds) - 1.5, max(means) + max(stds) + 1.5)
        ax.set_xticks([])
        ax.set_title(TITLES[ds], fontweight="bold", pad=4)
        ax.yaxis.grid(True, linewidth=0.4, color="lightgray", zorder=0)
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_ylabel("Test Accuracy (%)", labelpad=4)
    fig.legend(handles=[Patch(facecolor=c, edgecolor="white", linewidth=0.5, label=l) for _, l, c in READOUTS],
               loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.2, columnspacing=1.0)
    save(fig, out_dir, "readout_ablation")


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {out_dir / name}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--fig_dir", default="figures")
    args = parser.parse_args()
    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)

    paper_style()
    char = load(out_dir / "characterisation")
    if char:
        fig2(char, fig_dir)
        fig3(char, fig_dir)
    with plt.rc_context({"font.family": "serif", "font.size": 19, "axes.titlesize": 19, "axes.labelsize": 17,
                         "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 20}):
        abl = load(out_dir / "fig4")
        if abl:
            fig4(abl, fig_dir)
    if not char and not abl:
        print(f"no outputs found in {out_dir}")


if __name__ == "__main__":
    main()
