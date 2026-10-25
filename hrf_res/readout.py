"""Feature extraction from a fixed reservoir and the trained linear readout."""

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from .data import to_sequence


@torch.no_grad()
def extract_features(model, loader, dataset, device, perm=None, noise=None):
    """Run the reservoir over a loader.

    Returns (features, labels, rates), where rates holds the mean HRF
    (reservoir) and LIF (encoder) firing rates, averaged over batches.
    """
    model.eval()
    feats, labels, r_hrf, r_lif = [], [], [], []
    for x, y in loader:
        x = to_sequence(dataset, x.to(device), perm=perm, noise=noise)
        f, r = model(x)
        feats.append(f.cpu())
        labels.append(y.reshape(-1))
        if r:
            r_hrf.append(r["r_hrf"])
            r_lif.append(r["r_lif"])
    rates = {}
    if r_hrf:
        rates = {"r_hrf": torch.stack(r_hrf).mean().item(),
                 "r_lif": torch.stack(r_lif).mean().item()}
    return torch.cat(feats).numpy(), torch.cat(labels).numpy(), rates


def fit_readout(train_feats, train_labels, test_feats, test_labels, C=1.0,
                solver="lbfgs", max_iter=2000):
    """Standardise features and fit an L2-regularised logistic regression.

    Returns (train_acc, test_acc) in percent.
    """
    scaler = preprocessing.StandardScaler().fit(train_feats)
    train_feats, test_feats = scaler.transform(train_feats), scaler.transform(test_feats)
    clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter, n_jobs=-1)
    clf.fit(train_feats, train_labels)
    return (100 * clf.score(train_feats, train_labels),
            100 * clf.score(test_feats, test_labels))
