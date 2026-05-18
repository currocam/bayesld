#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ipython",
#     "numpy",
#     "pandas",
#     "phlash[gpu]",
# ]
#
# [tool.uv.sources]
# phlash = { git = "https://github.com/jthlab/phlash" , rev = "96a6e3f8e01053271e88d21c97862198422d9ae0"}
# ///
import pickle
import sys

import numpy as np
import phlash

contigs_path = sys.argv[1]
pkl_path = sys.argv[2]
npz_path = sys.argv[3]

with open(contigs_path, "rb") as f:
    contigs = pickle.load(f)

# From https://academic.oup.com/mbe/article/39/1/msab311/6413643
generation_time = 2
mutation_rate_per_year = 1.98e-9
mutation_rate = mutation_rate_per_year * generation_time

test_data = contigs[0]
train_data = contigs[1:]
results = phlash.fit(data=train_data, test_data=test_data, mutation_rate=mutation_rate)

# Save fitted results
with open(pkl_path, "wb") as f:
    pickle.dump(results, f)

# Posterior Ne trajectories
times = np.array([dm.eta.t[1:] for dm in results])
T = np.geomspace(times.min(), times.max(), 1000)
Nes = np.array([dm.eta(T, Ne=True) for dm in results])
np.savez(npz_path, T=T, Nes=Nes)
