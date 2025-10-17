import pymc as pm
import numpy as np
import arviz as az
import bayesld.models

# Define bins
x = np.linspace(0.20, 10, 50)
left = x[:-1]
right = x[1:]


def prediction(alpha):
    Nec = 5_000
    time = 50
    Nea = Nec * np.exp(-alpha * time)
    with pm.Model() as model:
        LD = pm.Deterministic(
            "LD",
            bayesld.models.expected_ld_exponential(
                Nec,
                Nea,
                time,
                alpha,
                left / 100,
                right / 100,
                sample_size=1000,
            ),
        )
        # We don't have theoretical predictions for `sigma`
        pm.Normal("x", LD, sigma=0.01)
        trace = az.extract(pm.sample(draws=1, chain=1, tune=0, progressbar=False))
        return trace.LD.to_numpy().mean(axis=1)


# Get predictions
alphas = np.asarray([0.005, 0.01, 0.02, 0.05, 0.1])
preds = np.asarray([prediction(alpha) for alpha in alphas])
midpoints = (left + right) / 2
np.savez_compressed("data.npz", predictions=preds, alphas=alphas, midpoints=midpoints)
