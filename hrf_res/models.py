"""
Reservoir models used in the paper.

    HRFRes  LIF encoder + Harmonic Resonate-and-Fire (HRF) reservoir (proposed)
    LIFRes  LIF encoder + heterogeneous LIF reservoir (controlled baseline)
    RON     non-spiking Randomized Oscillators Network (Ceni et al., 2024)

All weights are random and fixed; only a linear readout is trained on the
features returned by ``forward``. Inputs are batch-first, shape (B, T, n_inp).

Weight matrices follow the notation of the paper (stored transposed, so that
states are row vectors):
    U  input -> LIF encoder        (n_inp, n_hid)
    W  reservoir -> LIF encoder    (n_hid, n_hid), rescaled to spectral radius rho
    V  LIF encoder -> reservoir    (n_hid, n_hid)
    b  encoder bias                (n_hid,)

The order and device of every random draw is part of the model definition:
changing it changes the sampled reservoir for a given seed.
"""

import numpy as np
import torch
import torch.nn as nn


LIF_ENCODER_TAU = 20.0      # encoder membrane time constant (fixed)
HRF_REFRACTORY_TAU = 0.25   # decay constant of the HRF refractory threshold

MEMBRANE_READOUTS = ("final", "mean", "rms", "std", "rms_std_final")
SPIKE_READOUTS = ("spikes_mean", "spikes_rms_std_final")


def spectral_norm_scaling(W, rho_desired):
    """Rescale W to spectral radius ``rho_desired``."""
    e, _ = np.linalg.eig(W.cpu())
    rho_curr = max(abs(e))
    return W * (rho_desired / rho_curr)


def uniform_range(centre, width):
    """(low, high) of a uniform distribution of given centre and width."""
    return (centre - width / 2., centre + width / 2.)


def apply_sparse_input_projection(model, input_density, n_inp, n_hid, device):
    """Sparsify U in place for high-dimensional inputs.

    Each entry is kept with probability ``input_density``; every reservoir
    neuron and every input channel keeps at least one connection. Kept
    weights are rescaled by 1/sqrt(input_density). Returns the number of
    kept connections.
    """
    mask = (torch.rand(n_inp, n_hid, device=device) < input_density).float()
    for j in range(n_hid):
        if mask[:, j].sum() == 0:
            mask[torch.randint(0, n_inp, (1,)), j] = 1.0
    for i in range(n_inp):
        if mask[i, :].sum() == 0:
            mask[i, torch.randint(0, n_hid, (1,))] = 1.0
    scale = 1.0 / np.sqrt(input_density)
    model.U.data = model.U.data * mask * scale
    return int(mask.sum().item())


def _sparse_mask(n, connectivity, device):
    return (torch.rand(n, n, device=device) < connectivity).float()


def _summary_features(mode, final, total, total_sq, T):
    """Readout features from per-neuron sums over time."""
    if mode == "final":
        return final
    if mode == "mean":
        return total / T
    rms = torch.sqrt(total_sq / T + 1e-8)
    if mode == "rms":
        return rms
    std = torch.sqrt(torch.clamp(total_sq / T - (total / T) ** 2, min=1e-8))
    if mode == "std":
        return std
    return torch.cat([rms, std, final], dim=1)      # "rms_std_final"


class HRFRes(nn.Module):
    """LIF encoder + HRF reservoir.

    Per-neuron stiffness and damping are drawn uniformly with the given
    centre and width: gamma_i ~ U(gamma -/+ gamma_range/2), and likewise
    for epsilon. ``readout_mode`` selects the features returned by forward:

        "final"                 y at the last timestep             (B, N)
        "mean" / "rms" / "std"  temporal statistic of y            (B, N)
        "rms_std_final"         [RMS | Std | final] of y           (B, 3N)
        "spikes_mean"           mean HRF spike rate                (B, N)
        "spikes_rms_std_final"  [RMS | Std | final] of HRF spikes  (B, 3N)
    """

    def __init__(self, n_inp, n_hid, dt, gamma, gamma_range, epsilon,
                 epsilon_range, rho, input_scaling, theta_lif, theta_hrf,
                 connectivity_lif2hrf=1.0, connectivity_hrf2lif=1.0,
                 readout_mode="final", device="cpu"):
        super().__init__()
        assert readout_mode in MEMBRANE_READOUTS + SPIKE_READOUTS, readout_mode
        self.n_hid = n_hid
        self.dt = dt
        self.theta_lif = theta_lif
        self.theta_hrf = theta_hrf
        self.readout_mode = readout_mode
        self.device = device

        gamma_min, gamma_max = uniform_range(gamma, gamma_range)
        eps_min, eps_max = uniform_range(epsilon, epsilon_range)
        self.gamma = torch.rand(n_hid, device=device) * (gamma_max - gamma_min) + gamma_min
        self.epsilon = torch.rand(n_hid, device=device) * (eps_max - eps_min) + eps_min

        W = spectral_norm_scaling(2 * (2 * torch.rand(n_hid, n_hid) - 1), rho).to(device)
        if connectivity_hrf2lif < 1.0:
            mask = _sparse_mask(n_hid, connectivity_hrf2lif, device)
            W = W * mask
            self.n_hrf2lif_connections = int(mask.sum().item())
        else:
            self.n_hrf2lif_connections = n_hid ** 2
        self.W = nn.Parameter(W, requires_grad=False)

        self.U = nn.Parameter(torch.rand(n_inp, n_hid) * input_scaling, requires_grad=False)
        self.b = nn.Parameter((torch.rand(n_hid) * 2 - 1) * input_scaling, requires_grad=False)

        V = (torch.rand(n_hid, n_hid, device=device) * 2 - 1) * 2.0
        if connectivity_lif2hrf < 1.0:
            mask = _sparse_mask(n_hid, connectivity_lif2hrf, device)
            V = V * mask
            self.n_lif2hrf_connections = int(mask.sum().item())
        else:
            self.n_lif2hrf_connections = n_hid ** 2
        self.V = nn.Parameter(V, requires_grad=False)

        self.ref_decay = torch.exp(torch.as_tensor(-dt / HRF_REFRACTORY_TAU, device=device))

    def step(self, u, x, y, v, s, ref):
        """One timestep. x, y: HRF velocity/position; v: encoder potential;
        s: HRF spikes; ref: refractory threshold offset of the HRF neurons."""
        dt = self.dt

        # LIF encoder (soft reset)
        current = torch.matmul(u, self.U) + torch.matmul(s, self.W) + self.b
        v = v + dt * (-v / LIF_ENCODER_TAU + current)
        s_lif = (v > self.theta_lif).float()
        v = v - s_lif * self.theta_lif

        # HRF reservoir (semi-implicit Euler, no reset, adaptive threshold)
        drive = torch.matmul(s_lif, self.V)
        x = x + dt * (drive - self.gamma * y - self.epsilon * x)
        y = y + dt * x
        s = (y - self.theta_hrf - ref > 0).float()
        ref = ref * self.ref_decay + s

        return x, y, v, s, ref, s_lif

    def forward(self, u):
        """Returns (features, {"r_hrf", "r_lif"}) with mean firing rates."""
        B, T, _ = u.shape
        zeros = lambda: torch.zeros(B, self.n_hid, device=self.device)
        x, y, v, s, ref = zeros(), zeros(), zeros(), zeros(), zeros()

        spike_readout = self.readout_mode in SPIKE_READOUTS
        total, total_sq = zeros(), zeros()
        n_hrf_spikes = n_lif_spikes = 0.0

        for t in range(T):
            x, y, v, s, ref, s_lif = self.step(u[:, t], x, y, v, s, ref)
            signal = s if spike_readout else y
            total += signal
            total_sq += signal ** 2
            n_hrf_spikes += s.sum()
            n_lif_spikes += s_lif.sum()

        mode = self.readout_mode.replace("spikes_", "") if spike_readout else self.readout_mode
        features = _summary_features(mode, s if spike_readout else y, total, total_sq, T)
        rates = {"r_hrf": (n_hrf_spikes / (B * T * self.n_hid)).detach(),
                 "r_lif": (n_lif_spikes / (B * T * self.n_hid)).detach()}
        return features, rates


class LIFRes(nn.Module):
    """LIF encoder + heterogeneous LIF reservoir (soft reset).

    Identical to HRFRes except for the reservoir neurons: per-neuron membrane
    time constants and thresholds are drawn log-uniformly over
    tau_m -/+ tau_m_range/2 and theta_res -/+ theta_res_range/2. Membrane
    readouts use the reservoir potential in place of y.
    """

    def __init__(self, n_inp, n_hid, dt, tau_m, tau_m_range, theta_res,
                 theta_res_range, rho, input_scaling, theta_lif,
                 connectivity_lif2res=1.0, connectivity_res2lif=1.0,
                 readout_mode="final", device="cpu"):
        super().__init__()
        assert readout_mode in ("final", "mean", "rms_std_final"), readout_mode
        self.n_hid = n_hid
        self.dt = dt
        self.theta_lif = theta_lif
        self.readout_mode = readout_mode
        self.device = device

        tau_lo, tau_hi = uniform_range(tau_m, tau_m_range)
        th_lo, th_hi = uniform_range(theta_res, theta_res_range)
        tau_lo, th_lo = max(tau_lo, 1e-3), max(th_lo, 1e-6)
        self.tau_m = torch.exp(
            torch.FloatTensor(n_hid).uniform_(np.log(tau_lo), np.log(tau_hi))).to(device)
        self.theta_res = torch.exp(
            torch.FloatTensor(n_hid).uniform_(np.log(th_lo), np.log(th_hi))).to(device)

        W = spectral_norm_scaling(2 * (2 * torch.rand(n_hid, n_hid) - 1), rho).to(device)
        if connectivity_res2lif < 1.0:
            mask = _sparse_mask(n_hid, connectivity_res2lif, device)
            W = W * mask
            self.n_hrf2lif_connections = int(mask.sum().item())
        else:
            self.n_hrf2lif_connections = n_hid ** 2
        self.W = nn.Parameter(W, requires_grad=False)

        self.U = nn.Parameter(torch.rand(n_inp, n_hid) * input_scaling, requires_grad=False)
        self.b = nn.Parameter((torch.rand(n_hid) * 2 - 1) * input_scaling, requires_grad=False)

        V = (torch.rand(n_hid, n_hid, device=device) * 2 - 1) * 2.0
        if connectivity_lif2res < 1.0:
            mask = _sparse_mask(n_hid, connectivity_lif2res, device)
            V = V * mask
            self.n_lif2hrf_connections = int(mask.sum().item())
        else:
            self.n_lif2hrf_connections = n_hid ** 2
        self.V = nn.Parameter(V, requires_grad=False)

    def step(self, u, r, s, v):
        """One timestep. r: reservoir potential; s: reservoir spikes;
        v: encoder potential."""
        dt = self.dt

        current = torch.matmul(u, self.U) + torch.matmul(s, self.W) + self.b
        v = v + dt * (-v / LIF_ENCODER_TAU + current)
        s_lif = (v > self.theta_lif).float()
        v = v - s_lif * self.theta_lif

        drive = torch.matmul(s_lif, self.V)
        r = r + dt * (-r / self.tau_m + drive)
        s = (r > self.theta_res).float()
        r = r - s * self.theta_res

        return r, s, v, s_lif

    def forward(self, u):
        """Returns (features, {"r_hrf", "r_lif"}); r_hrf is the reservoir rate."""
        B, T, _ = u.shape
        zeros = lambda: torch.zeros(B, self.n_hid, device=self.device)
        r, s, v = zeros(), zeros(), zeros()
        total, total_sq = zeros(), zeros()
        n_res_spikes = n_lif_spikes = 0.0

        for t in range(T):
            r, s, v, s_lif = self.step(u[:, t], r, s, v)
            total += r
            total_sq += r ** 2
            n_res_spikes += s.sum()
            n_lif_spikes += s_lif.sum()

        features = _summary_features(self.readout_mode, r, total, total_sq, T)
        rates = {"r_hrf": (n_res_spikes / (B * T * self.n_hid)).detach(),
                 "r_lif": (n_lif_spikes / (B * T * self.n_hid)).detach()}
        return features, rates


class RON(nn.Module):
    """Randomized Oscillators Network (non-spiking baseline).

    Features are the oscillator positions at the last timestep.
    """

    def __init__(self, n_inp, n_hid, dt, gamma, gamma_range, epsilon,
                 epsilon_range, rho, input_scaling, device="cpu"):
        super().__init__()
        self.n_hid = n_hid
        self.dt = dt
        self.device = device

        gamma_min, gamma_max = uniform_range(gamma, gamma_range)
        eps_min, eps_max = uniform_range(epsilon, epsilon_range)
        self.gamma = torch.rand(n_hid, device=device) * (gamma_max - gamma_min) + gamma_min
        self.epsilon = torch.rand(n_hid, device=device) * (eps_max - eps_min) + eps_min

        W = spectral_norm_scaling(2 * (2 * torch.rand(n_hid, n_hid) - 1), rho)
        self.W = nn.Parameter(W, requires_grad=False)
        self.U = nn.Parameter(torch.rand(n_inp, n_hid) * input_scaling, requires_grad=False)
        self.b = nn.Parameter((torch.rand(n_hid) * 2 - 1) * input_scaling, requires_grad=False)

    def step(self, u, x, y):
        x = x + self.dt * (torch.tanh(
            torch.matmul(u, self.U) + torch.matmul(y, self.W) + self.b)
            - self.gamma * y - self.epsilon * x)
        y = y + self.dt * x
        return x, y

    def forward(self, u):
        y = torch.zeros(u.size(0), self.n_hid).to(self.device)
        x = torch.zeros(u.size(0), self.n_hid).to(self.device)
        for t in range(u.size(1)):
            x, y = self.step(u[:, t], x, y)
        return y, {}
