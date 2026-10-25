"""
Dataset loaders.

``load_dataset(name, data_dir, ...)`` returns (train_loader, test_loader, info).
Batches are (x, y) with x already shaped as a sequence (B, T, n_inp), except
for the image datasets, which are turned into sequences by ``to_sequence``.

Expected layout under ``data_dir`` (see README for download instructions):
    MNIST/, cifar-10-batches-py/     downloaded by torchvision
    FordA/FordA_TRAIN.txt, ...       UCR archive (.txt or .tsv)
    FordB/, Wafer/, Earthquakes/     UCR archive
    SHD/shd_train.h5, shd_test.h5
    DVSGesture/, NMNIST/             downloaded by tonic
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


UCR_DATASETS = ("FordA", "FordB", "Wafer", "Earthquakes")
SPLIT_SEED = 0  # fixes the MNIST/CIFAR-10 validation split and the npCIFAR-10 noise


# Image datasets (sMNIST, psMNIST, npCIFAR-10)

def _image_loaders(dataset_cls, data_dir, n_valid, batch_train, batch_test, drop_last):
    train = dataset_cls(root=data_dir, train=True, transform=transforms.ToTensor(), download=True)
    test = dataset_cls(root=data_dir, train=False, transform=transforms.ToTensor(), download=True)
    # The last n_valid training images are held out for hyperparameter search
    # and are never used for training, also when evaluating on the test set.
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train, valid = data.random_split(train, [len(train) - n_valid, n_valid], generator=generator)
    return (data.DataLoader(train, batch_size=batch_train, shuffle=True, drop_last=drop_last),
            data.DataLoader(valid, batch_size=batch_test, shuffle=False, drop_last=drop_last),
            data.DataLoader(test, batch_size=batch_test, shuffle=False, drop_last=drop_last))


def pixel_permutation(seed=0, device="cpu"):
    """Fixed pixel order for psMNIST, independent of the model seed."""
    return torch.randperm(784, generator=torch.Generator().manual_seed(seed)).to(device)


def cifar_noise(seq_length=1000, device="cpu"):
    """Gaussian noise appended to every npCIFAR-10 sequence (shared by all samples)."""
    g = torch.Generator().manual_seed(SPLIT_SEED)
    return torch.randn(1, seq_length - 32, 96, generator=g).to(device)


def to_sequence(name, x, perm=None, noise=None):
    """Turn a batch of images into input sequences (B, T, n_inp)."""
    B = x.shape[0]
    if name == "smnist":
        return x.reshape(B, 1, 784).permute(0, 2, 1)
    if name == "psmnist":
        return x.reshape(B, 1, 784).permute(0, 2, 1)[:, perm, :]
    if name == "npcifar10":
        # row-wise: 32 steps of 32 pixels x 3 channels, then the noise
        return torch.cat((x.permute(0, 2, 1, 3).reshape(B, 32, 96), noise.expand(B, -1, -1)), dim=1)
    return x


# UCR time series (FordA, FordB, Wafer, Earthquakes)

class _SeriesDataset(data.Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, i):
        return torch.Tensor(self.x[i]).reshape(-1, 1), torch.Tensor([self.y[i]])

    def __len__(self):
        return len(self.y)


def _read_ucr(name, data_dir):
    base = Path(data_dir) / name
    for ext in (".tsv", ".txt"):
        train_file, test_file = base / f"{name}_TRAIN{ext}", base / f"{name}_TEST{ext}"
        if train_file.exists():
            break
    else:
        raise FileNotFoundError(f"{name}_TRAIN.tsv/.txt not found in {base}")
    delimiter = "\t" if train_file.suffix == ".tsv" else None
    train, test = np.loadtxt(train_file, delimiter=delimiter), np.loadtxt(test_file, delimiter=delimiter)
    return train[:, 1:], train[:, 0], test[:, 1:], test[:, 0]


def _ucr_loaders(name, data_dir, batch_train, batch_test):
    x_train, y_train, x_test, y_test = _read_ucr(name, data_dir)
    if name != "Earthquakes":           # labels are -1/1 in the archive
        y_train, y_test = (y_train + 1) / 2, (y_test + 1) / 2
    # FordA was run with drop_last=True; with 3601 training series and batches
    # of 120 this drops one (random) training series per run.
    drop_last = name == "FordA"
    return (data.DataLoader(_SeriesDataset(x_train, y_train), batch_size=batch_train,
                            shuffle=True, drop_last=drop_last),
            data.DataLoader(_SeriesDataset(x_test, y_test), batch_size=batch_test,
                            shuffle=False, drop_last=drop_last))


# SHD

class SHDDataset(data.Dataset):
    """Spiking Heidelberg Digits, binned into num_steps frames over max_time seconds.

    Events are kept in sparse form and densified per sample.
    """

    def __init__(self, h5_path, num_steps=250, max_time=1.4, n_channels=700):
        import h5py
        self.num_steps, self.n_channels = num_steps, n_channels
        self.events, self.labels = [], []
        with h5py.File(h5_path, "r") as f:
            times, units, labels = f["spikes"]["times"], f["spikes"]["units"], f["labels"][()]
            for i in range(len(labels)):
                bins = np.floor(times[i][()] / max_time * num_steps).astype(np.int32)
                self.events.append((np.clip(bins, 0, num_steps - 1), units[i][()].astype(np.int32)))
                self.labels.append(int(labels[i]))

    def __getitem__(self, i):
        frame = np.zeros((self.num_steps, self.n_channels), dtype=np.float32)
        np.add.at(frame, self.events[i], 1.0)
        return torch.FloatTensor(frame), torch.LongTensor([self.labels[i]])

    def __len__(self):
        return len(self.labels)


def _shd_loaders(data_dir, batch_train, batch_test, num_steps, max_time):
    root = Path(data_dir) / "SHD"
    train = SHDDataset(root / "shd_train.h5", num_steps, max_time)
    test = SHDDataset(root / "shd_test.h5", num_steps, max_time)
    return (data.DataLoader(train, batch_size=batch_train, shuffle=True),
            data.DataLoader(test, batch_size=batch_test, shuffle=False))


# Event-camera datasets (DVS Gesture, N-MNIST)

def _event_loaders(name, data_dir, batch_train, batch_test, num_steps, spatial_factor):
    """Events binned into num_steps frames, average-pooled by spatial_factor
    and binarised; each frame is flattened to (2 * H * W) input channels."""
    import tonic
    from tonic import DiskCachedDataset

    dataset_cls = {"dvs_gesture": tonic.datasets.DVSGesture, "nmnist": tonic.datasets.NMNIST}[name]
    W, H, C = dataset_cls.sensor_size
    to_frames = tonic.transforms.ToFrame(sensor_size=dataset_cls.sensor_size, n_time_bins=num_steps)
    root = str(Path(data_dir) / {"dvs_gesture": "DVSGesture", "nmnist": "NMNIST"}[name])
    os.makedirs(root, exist_ok=True)

    def collate(batch):
        xs, ys = [], []
        for frames, label in batch:
            t = torch.tensor(frames, dtype=torch.float32)          # (T, C, H, W)
            if spatial_factor > 1:
                t = F.avg_pool2d(t.view(-1, 1, H, W), spatial_factor, spatial_factor)
            xs.append((t.reshape(num_steps, -1) > 0).float())
            ys.append(label)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    loaders = []
    for train, batch in ((True, batch_train), (False, batch_test)):
        ds = dataset_cls(save_to=root, train=train, transform=to_frames)
        split = "train" if train else "test"
        ds = DiskCachedDataset(ds, cache_path=os.path.join(root, f"cache_{split}_T{num_steps}_sf{spatial_factor}"))
        loaders.append(data.DataLoader(ds, batch_size=batch, shuffle=train, collate_fn=collate,
                                       num_workers=4, pin_memory=True))
    return loaders



def load_dataset(name, data_dir="data", batch_train=128, batch_test=128,
                 num_steps=None, spatial_factor=None, max_time=1.4, validation=False):
    """Returns (train_loader, test_loader, info).

    With ``validation=True`` the test loader is replaced by the held-out
    validation split (image datasets) used for hyperparameter search.
    ``info`` holds n_inp, n_out and seq_length.
    """
    if name in ("smnist", "psmnist"):
        train, valid, test = _image_loaders(torchvision.datasets.MNIST, data_dir, 3000,
                                            batch_train, batch_test, drop_last=False)
        info = dict(n_inp=1, n_out=10, seq_length=784)
    elif name == "npcifar10":
        train, valid, test = _image_loaders(torchvision.datasets.CIFAR10, data_dir, 3000,
                                            batch_train, batch_test, drop_last=True)
        info = dict(n_inp=96, n_out=10, seq_length=1000)
    elif name in [d.lower() for d in UCR_DATASETS]:
        ucr_name = UCR_DATASETS[[d.lower() for d in UCR_DATASETS].index(name)]
        train, test = _ucr_loaders(ucr_name, data_dir, batch_train, batch_test)
        info = dict(n_inp=1, n_out=2, seq_length=train.dataset[0][0].shape[0])
    elif name == "shd":
        train, test = _shd_loaders(data_dir, batch_train, batch_test, num_steps or 250, max_time)
        info = dict(n_inp=700, n_out=20, seq_length=num_steps or 250)
    elif name in ("dvs_gesture", "nmnist"):
        train, test = _event_loaders(name, data_dir, batch_train, batch_test, num_steps, spatial_factor)
        side = {"dvs_gesture": 128, "nmnist": 34}[name] // spatial_factor
        info = dict(n_inp=2 * side * side, n_out={"dvs_gesture": 11, "nmnist": 10}[name],
                    seq_length=num_steps)
    else:
        raise ValueError(f"unknown dataset '{name}'")

    if validation:
        if name not in ("smnist", "psmnist", "npcifar10"):
            raise ValueError(f"no predefined validation split for '{name}'")
        test = valid
    return train, test, info
