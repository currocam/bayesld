---
title: Population structure
marimo-version: 0.24.0
width: medium
pyproject: |-
  requires-python = ">=3.12"
  dependencies = [
      "bayesld==0.1.0",
      "marimo",
      "matplotlib==3.10.9",
      "msprime==1.4.1",
      "numpy==2.5.2",
      "scienceplots==2.2.2",
      "sympy==1.14.0",
  ]
---

```python {.marimo}
import marimo as mo
import bayesld
```

```python {.marimo}
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import sympy as sp
```

```python {.marimo}
plt.style.use("science")
plt.style.use("bin/theme.mplstyle")
plt.rc("figure", autolayout=True)
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams["pgf.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"

ONE_MM = 1 / 25.4
SINGLE_COL = 85 * ONE_MM
DOUBLE_COL = SINGLE_COL * 2
ONE_HALF_COL = SINGLE_COL * 1.5
```

```python {.marimo}
left_bins, right_bins = bayesld.linear_bins()
bin_midpoints = (left_bins + right_bins) / 2
SAMPLE_SIZE = 30
```

First, we consider a simple constant population that recieves migrants (forward-in-time) form a mainland population.

```python {.marimo}
# Montecarlo simulations
def sim(ne1, ne2, m, seed = 1234):
    import msprime
    demo = msprime.Demography.isolated_model(initial_size=[ne1, ne2])
    demo.add_migration_rate_change(source="pop_0", dest="pop_1", time=0, rate=m)
    return  bayesld.sim_sufficient_stats(
        samples=SAMPLE_SIZE,
        demography=demo,
        left_bins=left_bins, right_bins=right_bins,
        mutation_rate=1e-8, recombination_rate=1e-8,
        sequence_length=right_bins[-1]*2/1e-8,
        random_seed=seed,
        rtol=0.1,  # adaptive stopping
        ploidy=2, num_workers=10
)
```

Consider we take two samples from the focal population. We denote the coalescence probability density as $f(t)$ and we want to compute

$$
S(u) = \int_0^ \infty f(t)\exp(-2tu)\,dt
$$

Using phase-type theory:

```python {.marimo}
sym_lam1, sym_lam2, sym_m, sym_u = sp.symbols("lambda1 lambda2 m u", positive=True)
sub_intensity = sp.Matrix(
    [
        [-(sym_lam1 + 2 * sym_m), 2 * sym_m, 0],  # coalesce or migrate
        [0, -sym_m, sym_m],                    # migrate
        [0, 0, -sym_lam2],                        # coalesce at c
    ]
)
alpha_init = sp.Matrix([[1, 0, 0]])  # both lineages sampled in pop_0
exit_rates = -sub_intensity * sp.ones(3, 1)
exit_rates.T  # two absorbing states

# Laplace transform of a phase-type M(x)
# https://en.wikipedia.org/wiki/Phase-type_distribution
sym_x = sp.symbols("x")
laplace_transform = sp.simplify(
    (alpha_init * (sym_x * sp.eye(3) - sub_intensity).inv() * exit_rates)[0]
)
expected_S_symbolic = sp.factor(
    sp.together(sp.simplify(laplace_transform.subs(sym_x, 2 * sym_u)))
)
expected_S_symbolic
```

```python {.marimo}
expected_S_analytical = sp.lambdify(
    (sym_lam1, sym_lam2, sym_m, sym_u), expected_S_symbolic, modules="numpy"
)
expected_S_analytical(1 / (2 * 2000), 1 / (2 * 2000), 1e-4, 0.05)
def correct_ld_finite_sample(mu, sample_size):
    # Number of haploid samples
    S = 2 * sample_size

    # Compute correction parameters
    beta = 1 / (S - 1) ** 2
    alpha = ((S**2 - S + 2) ** 2) / ((S**2 - 3 * S + 2) ** 2)

    # Apply correction formula
    return (alpha - beta) * mu + 4 * beta

def expected_ld(ne1, ne2, m, left_bins, right_bins, n_quad=10):
    import numpy as np
    l = np.asarray(left_bins)[:, None]
    r = np.asarray(right_bins)[:, None]
    x, w = np.polynomial.legendre.leggauss(n_quad)
    c = (r - l) / 2 * x + (r + l) / 2  # nodes, shape (n_bins, n_quad)
    lam1, lam2 = 1 / (2 * ne1), 1 / (2 * ne2)  # focal deme, source deme
    return correct_ld_finite_sample((expected_S_analytical(lam1, lam2, m, c) * w / 2).sum(axis=1), SAMPLE_SIZE)
```

```python {.marimo}
FOCAL_SIZES = [2000, 5000]        # ne1, the sampled deme
SOURCE_SIZES = [2000, 20_000]      # ne2, the deme migrants come from
MIGRATION_RATES = [1e-3, 1e-1]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(DOUBLE_COL, ONE_HALF_COL),
    dpi=300,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
x = bin_midpoints * 100  # centimorgan, matching convention elsewhere
for _i, _ne2 in enumerate(SOURCE_SIZES):
    for _j, _ne1 in enumerate(FOCAL_SIZES):
        _ax = axes[_i, _j]
        for _k, _m in enumerate(MIGRATION_RATES):
            _color = f"C{_k}"
            _ax.plot(
                x,
                expected_ld(_ne1, _ne2, _m, left_bins, right_bins),
                color=_color,
                lw=2,
                label=f"$m=10^{{{np.log10(_m):.0f}}}$",
            )
            _ld = sim(_ne1, _ne2, _m)[1]
            _mean = _ld.mean(axis=0)
            _stderr = _ld.std(axis=0, ddof=1) / np.sqrt(_ld.shape[0])
            _ax.plot(x, _mean, color=_color, lw=2, ls="--")
            _ax.fill_between(
                x, _mean - _stderr, _mean + _stderr, color=_color, alpha=0.15, linewidth=0
            )
        _ax.set_title(f"$N_1={_ne1:,}$, $N_2={_ne2:,}$", fontsize="medium")
        _ax.set_yscale("log")
        if _i == 1:
            _ax.set_xlabel("Genetic distance (centimorgan)")
        if _j == 0:
            _ax.set_ylabel(r"$X_iX_jY_iY_j$")

_handles = [
    plt.Line2D([], [], color="k", lw=2, ls="-", label="Closed-form"),
    plt.Line2D([], [], color="k", lw=2, ls="--", label="Monte Carlo"),
] + axes[0, 0].get_legend_handles_labels()[0]
fig.legend(
    handles=_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    frameon=False,
)
fig
```

```python {.marimo}
fig.savefig("results/popstructure/plot_unidirectional.pdf")
fig.savefig("results/popstructure/plot_unidirectional.pgf")
```