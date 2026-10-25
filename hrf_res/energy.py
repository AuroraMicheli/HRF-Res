"""Energy model (Sec. 3.3 and Appendix A.2 of the paper).

Spiking models are charged one accumulate (AC) per synaptic operation (SOP),
non-spiking models one multiply-accumulate (MAC) per synapse per timestep,
using 45 nm CMOS estimates (Horowitz, 2014).
"""

E_AC = 0.9e-12   # J per accumulate / SOP
E_MAC = 4.6e-12  # J per multiply-accumulate

# MACs per neuron per timestep for the activity-independent state updates
STATE_MACS_PER_NEURON = {"hrf_res": 4, "lif_res": 2, "ron": 3}


def spiking_energy(r_hrf, r_lif, n_hid, T, n_lif2hrf_connections):
    """Synaptic energy of HRF-Res / LIF-Res for one input sequence.

    Every reservoir spike reaches all n_hid encoder neurons (W is dense);
    every encoder spike reaches its average fan-out in V.
    """
    hrf_sops = r_hrf * n_hid * T * n_hid
    lif_sops = r_lif * n_hid * T * (n_lif2hrf_connections / n_hid)
    sops = hrf_sops + lif_sops
    return {"SOPs": sops, "Energy_J": sops * E_AC}


def dense_energy(n_inp, n_hid, T):
    """Synaptic energy of a non-spiking reservoir (RON): U and W at every step."""
    macs = T * (n_inp * n_hid + n_hid * n_hid)
    return {"MACs": macs, "Energy_J": macs * E_MAC}


def state_update_energy(model, n_hid, T):
    """Energy of the O(N) neuron state updates (energy breakdown table in the appendix)."""
    macs = STATE_MACS_PER_NEURON[model] * n_hid * T
    return {"MACs": macs, "Energy_J": macs * E_MAC}
