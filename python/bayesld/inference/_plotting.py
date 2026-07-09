"""Matplotlib helpers for visualising posterior demographic draws."""

from __future__ import annotations

import numpy as np


def _size_trajectory(demography, times: np.ndarray) -> np.ndarray:
    dbg = demography.debug()
    return dbg.population_size_trajectory(times)[:, 0]


def _time_grid(demographies, n_time: int = 256) -> np.ndarray:
    t_max = 0.0
    event_times = {0.0}
    for demography in demographies:
        for t in demography.debug().epoch_start_time:
            if np.isfinite(t):
                t = float(t)
                event_times.add(t)
                t_max = max(t_max, t)
    if t_max <= 0:
        t_max = 1.0
    dense = np.linspace(0.0, t_max, n_time)
    return np.unique(np.concatenate([np.array(sorted(event_times)), dense]))


def plot_demography(demographies, ax=None, *, color="#4c72b0"):
    """Plot one line per ``msprime.Demography`` draw; returns the axes."""
    import matplotlib.pyplot as plt

    demographies = list(demographies)
    if not demographies:
        raise ValueError("demographies must contain at least one msprime.Demography")

    if ax is None:
        _, ax = plt.subplots()

    times = _time_grid(demographies)
    trajectories = np.column_stack([_size_trajectory(d, times) for d in demographies])
    alpha = 0.9
    tiny = np.finfo(float).tiny

    for sizes in trajectories.T:
        ax.plot(times, np.maximum(sizes, tiny), color=color, alpha=alpha, linewidth=0.9)

    ax.set_yscale("log")
    ax.set_xlabel("generations ago")
    ax.set_ylabel("Population size")
    return ax
