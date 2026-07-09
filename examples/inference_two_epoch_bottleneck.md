# bayesld

`bayesld` is a Python package for Bayesian inference of very recent demography.

## Motivation

This notebook presents a comprehensive analysis of a very recent bottleneck using simulated data. Working with simulated data here serves a double purpose. First, we use it to conduct a full analysis and illustrate the main usage of bayesld. Second, this notebook can be adapted to other scenarios to perform (1) a verification step and (2) a power analysis. `bayesld` falls within the scope of model-based inference, which means that for a given demography we can simulate a dataset, and for a given dataset we can obtain parameters. It is unreasonable to trust a demographic model that does not work even on simulated data, so we advise verifying your model first (the verification step). It is also extremely useful to study a priori the effect of adding more samples to an analysis or, for example, how large a population in decline must be, or for how long it must decline, before it can be detected (power analysis).

## Ecosystem

This package relies on msprime, a well-established population genetics simulator, and ArviZ, a language-agnostic ecosystem for statistics, visualisation, and diagnostics in Bayesian workflows. Both libraries have extensive documentation and capabilities far beyond what we use here.

```{code-cell} ipython3
import bayesld
import msprime
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
```

## Case study: a very recent bottleneck

Consider a relatively small population that undergoes an instantaneous change in size 30 generations ago:

```{code-cell} ipython3
true_params = {
    # ancestral population size
    "Ne_a": 4000,
    # bottleneck strength
    "decline_fraction": 0.76,
    # generations ago
    "time_bottleneck": 30,
}
```

We define an [msprime](https://tskit.dev/msprime/docs/stable/demography.html) demographic model with the true parameters:

```{code-cell} ipython3
true_demography = msprime.Demography()
true_demography.add_population(initial_size=true_params["decline_fraction"] * true_params["Ne_a"])
true_demography.add_population_parameters_change(
    time=true_params["time_bottleneck"],
    initial_size=true_params["Ne_a"],
)
true_demography
```

Optionally, we can visualize it using `demesdraw` (especially useful for more complex scenarios):

```{code-cell} ipython3
import demesdraw

demesdraw.size_history(true_demography.to_demes())
```

## Dataset

`bayesld` performs inference over a set of informative summary statistics computed across windows (which are assumed to be independent and identically distributed). These statistics are the mean observed genetic diversity (also known as sample heterozygosity) and a measure of linkage disequilibrium (LD). To measure LD, we compute $\overline{X_i X_j Y_i Y_j}$, the mean product of the (centred and standardised) genotypes across all pairs of individuals, for many pairs of loci $X$ and $Y$ separated by increasing distances. Pairs of genotypes are then aggregated in different bins. 

Both summary statistics can be computed from an unphased VCF, provided that estimates of the recombination rate and mutation rate are available. The package provides utilities to extract this information from a real VCF, either from the command line or through the Python API (see bayesld.data_from_vcf).  Both use cases are covered in a separate tutorial (XXX). 

Here we work with simulated data, so we can skip creating an intermediate VCF and use the function `bayesld.sim_sufficient_stats` instead. Internally, this function wraps `msprime.sim_ancestry`, `msprime.sim_mutations`, and `bayesld.data_from_tree_sequence` (the last of which you can use directly if you simulated your data with a different engine, such as SLiM).

+++

### Binning scheme

Any binning scheme can be used; the right choice depends on the dataset. We cover this in more detail in a separate tutorial, along with how to extract the windowed summary statistics from a VCF. Here, we use the default settings:

```{code-cell} ipython3
left_bins_morgan, right_bins_morgan = bayesld.linear_bins()
midpoints = (left_bins_morgan+right_bins_morgan) / 2
midpoints
```

### Sample size and genomic windows

Resolving the very recent past often requires large sample sizes. Here we simulate a moderately large sample of 30 diploid individuals across 100 genomic windows of 20 cM each (corresponding to 20 chromosomes of 1 Morgan). I further assume a mutation rate and recombination rate of 1e-8.

```{code-cell} ipython3
mutation_rate = 1e-8
recombination_rate = 1e-8
window_length_in_morgan = 0.20 
window_length_in_bp = window_length_in_morgan/ recombination_rate
num_samples = 30
num_windows = 100

# Sanity check: the genomic window must be larger than the largest bin
assert np.all(window_length_in_morgan > right_bins_morgan), "Decrease bin distances!"
```

We are ready to simulate the dataset:

```{code-cell} ipython3
obs_pi, obs_ld = bayesld.sim_sufficient_stats(
    demography=true_demography,
    samples = num_samples,
    left_bins=left_bins_morgan,
    right_bins=right_bins_morgan,
    mutation_rate=mutation_rate,
    recombination_rate=recombination_rate,
    sequence_length=window_length_in_bp,
    random_seed=216789,
    num_replicates=num_windows,
    model = "hudson",
    num_workers=8
)
```

Next, we visualize the simulate dataset (our "empirical" dataset). As expected, we observe, on average, a decay of LD as a function of the genetic distance between pairs of SNPs. Error bars show the standard error of the observed mean.

```{code-cell} ipython3
fig, (ax_pi, ax_ld) = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={"width_ratios": [3, 6]})
ax_pi.hist(obs_pi, color = "C0", edgecolor="black")
ax_pi.set_xlabel("Mean genetic diversity)")
_mean =  obs_ld.mean(axis=0)
_stderr = obs_ld.std(axis=0) / np.sqrt(num_windows)
ax_ld.vlines(
    midpoints,
    _mean - _stderr, 
   _mean + _stderr,
    capstyle="round", linewidth=8, alpha=0.5, color="C1"
)
ax_ld.plot(midpoints, obs_ld.mean(axis=0), "o", color="C1")
ax_ld.set_xlabel("Distance (Morgan)")
ax_ld.set_ylabel(r"$\mathbb{E}[X_i X_j Y_i Y_j]$")
plt.tight_layout()
plt.show()
```

## Fitting the first model: constant size

It is often recommended to start with a simpler model and iteratively fit more complex ones. A reasonable first choice is a constant-size demography. If it fits well, that may suggest the focal population is at equilibrium and has not been affected by any recent demographic disturbance.

A constant-size demography can be modelled as a `bayesld.PiecewiseConstant` with a single epoch. Let's instantiate a model and pass it our empirical dataset via the `.with_data` method.

```{code-cell} ipython3
from bayesld.inference import PiecewiseConstant
constant_model = PiecewiseConstant(num_epochs=1).with_data(
    mean_diversity=obs_pi,
    mean_ld=obs_ld,
    left_bins=left_bins_morgan,
    right_bins=right_bins_morgan,
    recombination_rate=recombination_rate,
    mutation_rate=mutation_rate,
    num_samples=num_samples,
    sequence_length=window_length_in_bp
)
```

Inside a Jupyter (or marimo) notebook, we can get an overview of the different parts of the model by simply returning it as the last expression in a cell. Next, we go through the main elements that make up the model.

```{code-cell} ipython3
constant_model
```

Any Bayesian model consists of a prior and a likelihood distribution. 

#### The prior

Here, the prior is a probability distribution assigned to the parameters of the chosen parametric form $N_e(t)$ (that is, a distribution probability over a "family" demographies). `PiecewiseConstant` places a [Log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution) prior on $N_e$ (which conveniently forces the parameter to stay positive). For a single epoch this means:

$$
\begin{align}
N \sim \text{LogNormal}(\mu_N, \sigma_N) \\
N_e(t) = N
\end{align}
$$

where $\mu_N$ and $\sigma_N$ are specified by the user through the `with_prior` method. 

By default, we estimate a prior from the data (often called an [empirical Bayes prior](https://en.wikipedia.org/wiki/Empirical_Bayes_method)). We can inspect it by first drawing samples from the prior (with the `.sample_prior` method). The output is an `xarray.DataTree` (the rich data structure used by ArviZ), so we can explore it using the extensive tooling that ArviZ provides.

```{code-cell} ipython3
prior_constant = constant_model.sample_prior(draws=4000, chains=2, seed=32167)
az.summary(prior_constant,kind = "stats")
```

In this case, the default prior is very uninformative, with an 89% credible interval covering a wide range of 11–45000 diploid individuals. Let's continue with it.

+++

#### The (surrogate) likelihood

Here, the likelihood is a probability distribution assigned to the observed data. That is, a probability distribution for the windowed mean summary statistics. If genomic windows are large enough and contain many [SNPs](https://en.wikipedia.org/wiki/Single-nucleotide_polymorphism), this distribution is well approximated by a [Multivariate Normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution). 

`bayesld` combines (biased) analytical predictions and (unbiased) Monte Carlo simulations to **jointly** learn the form of the likelihood (i.e. a surrogate likelihood) together with the posterior probability of the parameters[^1]. 

#### Active learning rounds

The process of fitting the model involves *augmenting* the empirical dataset with a of Monte Carlo estimates of the log-likelihood. In practice, a few dozens combinations or parameters in the area of high posterior probability are enough. Of course, we don't know where that region is before fitting the model. To avoid wasting time/resources, `bayesld` uses a sequence of active learning rounds where we fit the model with increasingly better approximations of the log-likelihood and augment the empirical dataset with combinations of high probabiliyty. 

Fitting the model involves *augmenting* the empirical dataset with Monte Carlo estimates of the log-likelihood. In practice, a few dozen parameter combinations in the region of high posterior probability are enough. Of course, we don't know where that region is before fitting the model. To avoid wasting time/resources, `bayesld` uses a sequence of active learning rounds: we fit the model with increasingly better approximations of the log-likelihood and augment the empirical dataset with high-probability parameter combinations.

The cost of each Monte Carlo estimate grows with the population's effective size. On the bright side, the initial analytical approximation is also better at high $N_e$. This means that, to keep the error acceptable, many simulations are needed when they are cheap, whereas when they are expensive only a small number (or none) are needed.

+++

### Posterior distribution for a constant model

As promised, we now fit the constant demography (remember, to a population that has experienced a bottleneck). 

I will perform 2 rounds of active learning with 20 points each. Increasing the number of points and decreasing `rtol` might increase the _confidence_ of the joint model on the surrogate likelihood and result on better posterior estimates (at the cost of a higher runtime). I recommend choosing a value smaller than 0.1 if you have many cores. 

```{code-cell} ipython3
%%time 
for i in range(2):
    constant_model = constant_model.active_learning_round(
        num_points=20, rtol=0.10, min_replicates=50,
        verbose=False, num_workers=8, seed = 2178
    )
constant_model
```

In addition to the quick iterations done during active learning, we finally sample from the (surrogate) posterior using MCMC. 

The package API is designed to be flexible enough so you can split computation across different runs or machines (for example, maximizing parallelism in an HPC, whereas running MCMC in a local laptop. If you want to save the model and continue later, you can just save into using `pickle`.

```{code-cell} ipython3
posterior_constant = constant_model.sample(
    draws=2000, num_workers = 4,
    chains=4, seed=32167, verbose=False
)
```

From the ArviZ package, we have a variety of opinionated tools to diagnose MCMC convergence. As a rule of thumb, we aim for an $\hat r < 1.01$ and a effective sample size `ess` greater than 400.

```{code-cell} ipython3
az.summary(posterior_constant, var_names = "Ne_values", kind = "diagnostics")
```

Alternatively, we may plot at the so-called traceplot and verify that the output looks life a fuzzy catterpilar.

```{code-cell} ipython3
az.plot_trace(posterior_constant, var_names = "Ne_values");
```

If we directly inspect the distribution of the estimated $N_e$, we observe an estimate close to 4000 diploid individuals.

```{code-cell} ipython3
az.plot_dist(posterior_constant, var_names="Ne_values",ci_prob=0.95)
plt.show()
```

It is essential to evaluate the absolute goodness of fit — that is, how well the estimated demography fits the data. Large deviations are a sign of severe model misspecification. In Bayesian jargon, this is known as posterior predictive checks. Here, I chose to only plot the predicted LD pattern.

Overall, the absolute goodess of fit is not terrible although we do observe a positive bias in the predicted LD pattern. This is expected: the real data is undergoing a (mild) bottleneck, and as such experiences higher drift (and lower $N_e$). A stronger bottleneck will accentuate bias.

```{code-cell} ipython3
def plot_posterior_predictive(
    idata, observed_pi, observed_ld, midpoints, colors=("C0", "C1")
):
    fig, (ax_pi, ax_ld) = plt.subplots(1, 2, figsize=(11, 4))
    ppc = idata["posterior_predictive"]
    num_windows = np.asarray(observed_ld).shape[0]
    # Diversity distribution
    _mean_dist = ppc["observed_pi"].mean(dim=["window"]).values.ravel()
    ax_pi.axvline(
        observed_pi.mean(),
        color=colors[0],
        linewidth=2,
        label="Observed",
    )
    ax_pi.hist(
        _mean_dist,
        bins="auto",
        color=colors[1],
        alpha=0.6,
        density=True,
        label="Predicted",
    )
    ax_pi.legend()
    ax_pi.set_xlabel(r"Genetic diversity ($\pi$)")
    # LD means and standard errors
    predicted_mean = ppc["observed_ld"].mean(dim=("chain", "draw", "window"))
    predicted_se = (
        ppc["observed_ld"].std(dim=("chain", "draw", "window"))
        / np.sqrt(num_windows)
    )
    observed_mean = np.asarray(observed_ld).mean(axis=0)
    observed_se = np.asarray(observed_ld).std(axis=0) / np.sqrt(num_windows)

    for mean, se, color, label in (
        (observed_mean, observed_se, colors[0], "Observed"),
        (predicted_mean, predicted_se, colors[1], "Predicted"),
    ):
        ax_ld.scatter(midpoints, mean, color=color, label=label)
        ax_ld.vlines(
            midpoints, mean - se, mean + se,
            color=color, alpha=0.3, linewidth=8, capstyle="round",
        )

    ax_ld.set_xlabel("Genetic distance (Morgan)")
    ax_ld.set_ylabel(r"Linkage disequilibrium ($\mathbb{E}[X_i X_j Y_i Y_j]$)")
    ax_ld.legend()

    fig.tight_layout()
    return fig
```

<details>
  <summary>More on posterior predictive checks</summary>
    I've found the mean informative of model misspecification in this model. However, it is often useful to look at the entire distribution across windows. We can use the `.plot_ppc_dist`. From the fitted model we can simulate new datasets and plot the distribution across windows. Ideally, we should not be able to tell them apart (other than because it's coloured in black). 

    Try running 

    ```python
    az.plot_ppc_dist(posterior_constant, cols=["bin"], var_names="observed_ld")
    ```
</details>

```{code-cell} ipython3
plot_posterior_predictive(
    posterior_constant, obs_pi, obs_ld, midpoints
);
```

## Fitting a two-epoch model

Next, I will fit a two-epoch piecewise constant model (which, in this case, corresponds to the actual demography).

```{code-cell} ipython3
two_epoch_model = PiecewiseConstant(num_epochs=2).with_data(
    mean_diversity=obs_pi,
    mean_ld=obs_ld,
    left_bins=left_bins_morgan,
    right_bins=right_bins_morgan,
    recombination_rate=recombination_rate,
    mutation_rate=mutation_rate,
    num_samples=num_samples,
    sequence_length=window_length_in_bp
)
```

Rather than going with the default prior, I will specify one using the `.with_prior` method. How to specify priors is a big topic in Bayesian inference. Priors can be chosen from the literature (for example, from a meta-analysis of demographic analysis) to reflect _a priori_ knowledge. That is often called an informative prior. 

Suppose in this case we are aiming to model a known bottleneck (because of ecological data). It might make sense to choose a prior that reflects that knowledge. However, it is often recommended to use, so-called, weakly informative priors: a prior that sets most probability mass only on biologically realistic scenarios while allowing the data to contradict our domain-knowledge (as that can indicate a bug in the code or model miss-specification).

Here, I set weakly informative priors that are consistent with either a recent growth or decrease in $N_e$. 

$$
\begin{align}
N_c \sim \text{LogNormal}(\log(10000), 1) \\
N_a \sim \text{LogNormal}(\log(10000), 1) \\
t_0 \sim \text{LogNormal}(\log(100), 1) \\
N_e(t) = N_c if t < t_0, else N_a
\end{align}
$$

```{code-cell} ipython3
two_epoch_model = two_epoch_model.with_prior(
    mu_log_ne = np.log([10_000, 10_000]),
    sigma_log_ne = np.array([1.0, 1.0]),
    mu_log_t = np.log([100]),
    sigma_log_t = np.array([1.0]),
)
two_epoch_model
```

It is always a good idea to examine the implications of the prior. As before, we can sample a set of draws from the prior.

```{code-cell} ipython3
prior_2epoch = two_epoch_model.sample_prior()
az.summary(prior_2epoch, kind="stats")
```

We can also plot the $N_e$ trajectories with `.plot_demography`

```{code-cell} ipython3
_samples = az.extract(prior_2epoch, num_samples=50, random_seed=1)
two_epoch_model.plot_demography(_samples)
plt.show()
```

Additionally, you can use the `.to_demography` to obtain a `msprime` Demographic object from which you can simulate genetic data.

+++

Let's repeat the same fitting procedure, using 4 rounds of active learning followed by MCMC sampling (often, you want to increase the number of rounds and data points with the complexity of the model).

```{code-cell} ipython3
%%time 
for i in range(4):
    two_epoch_model = two_epoch_model.active_learning_round(
        num_points=30, rtol=0.1, min_replicates=50,
        verbose=False, num_workers=8, seed = 13576
    )
```

```{code-cell} ipython3
posterior_2epoch = two_epoch_model.sample(
    draws=2000, num_workers = 4,
    chains=4, seed=32167, verbose=False
)
```

As before, we can check if the Markov chain has converged:

```{code-cell} ipython3
az.summary(posterior_2epoch, var_names =["Ne_values", "t_boundaries"], kind = "diagnostics")
```

In this case, we are interested in the start of the bottleneck and decline fraction, which we can compute from the trace. The computed CI match well the ground truth:

```{code-cell} ipython3
true_params
```

```{code-cell} ipython3
posterior_2epoch["posterior"]["fraction"] = posterior_2epoch["posterior"]["Ne_values"][:, :, 0] / posterior_2epoch["posterior"]["Ne_values"][:, :, 1]
az.summary(posterior_2epoch, var_names=["t_boundaries", "fraction", "Ne_values"])
```

When evaluated the models, it is important to not only look at the marginal distributions, but to account for the correlation among variables. Intuitively, there's a strong correlation between when the predicted bottleneck started and its strength (as more recent bottlenecks are only compatible with the data if the decline was stronger).

```{code-cell} ipython3
az.plot_pair(
    posterior_2epoch,
    var_names=["t_boundaries", "fraction"],
)
```

As before, we can evaluate the absolute goodness of fit, which in this case seems to be much better:

```{code-cell} ipython3
plot_posterior_predictive(
    posterior_2epoch, obs_pi, obs_ld, midpoints
);
```

Finally, we can visualize the trajectories implied by a few draws from the posterior (as we did with the prior)

```{code-cell} ipython3
_samples = az.extract(posterior_2epoch, num_samples=50, random_seed=1)
two_epoch_model.plot_demography(_samples, color = "C0");
```

## Fitting an exponential model

Because we are working with simulated data, we know that the true demography is a two-epoch piecewisse constant. It is of interest, however, see how other families would fit this case. 

Next, I fit a piecewise-exponential model, where the main difference is that I assume changes in $N_e$ happen in a continous fashion. 

$$
\begin{align}
N_a \sim \text{LogNormal}(\log(10000), 1) \\
t_0 \sim \text{LogNormal}(\log(100), 1) \\
\text{log-fold} \sim \text{Normal}(0, 0.5) \\
N_c = N_a \times \exp(\text{log-fold}) \\
N_e(t) = N_c \cdot \exp(\frac{-t \cdot \text{log-fold}}{t_0}) if t < t_0, else N_a
\end{align}
$$

```{code-cell} ipython3
from bayesld.inference import PiecewiseExponential
```

```{code-cell} ipython3
exp_model = PiecewiseExponential().with_data(
    mean_diversity=obs_pi,
    mean_ld=obs_ld,
    left_bins=left_bins_morgan,
    right_bins=right_bins_morgan,
    recombination_rate=recombination_rate,
    mutation_rate=mutation_rate,
    num_samples=num_samples,
    sequence_length=window_length_in_bp
).with_prior(
    mu_log_ne_a=np.log(10_000), sigma_log_ne_a=1,
    mu_log_t = np.log(100),sigma_log_t = 1,
    mu_log_alpha_fold = 0.0, sigma_log_alpha_fold=0.5
)
exp_model
```

We sample a few trajectories from the prior:

```{code-cell} ipython3
prior_exp = exp_model.sample_prior()
_samples = az.extract(prior_exp, num_samples=50, random_seed=1)
exp_model.plot_demography(_samples);
```

As before, the chosen prior seems a fine weakly informative prior. Next, I repeat the fitting procedure.

```{code-cell} ipython3
%%time 
for i in range(4):
    exp_model = exp_model.active_learning_round(
        num_points=30, rtol=0.1, min_replicates=50,
        verbose=False, num_workers=8, seed = 13576
    )
```

```{code-cell} ipython3
posterior_exp = exp_model.sample(
    draws=2000, num_workers = 4, chains=4,
    seed=32167, verbose=False
)
```

First, we check if the Markov chain has converged:

```{code-cell} ipython3
az.summary(posterior_exp, var_names =["Ne_c", "Ne_a", "t0", "log_alpha_fold"], kind = "diagnostics")
```

The estimated decline fraction has not change much, although confidence intervals have increase in width (reflecting higher uncertainty under this model). When the bottleneck started, however, is now overestimated compared to the actual demographic model where the bottleneck happened instantaneously (rather than "smootly"). 

```{code-cell} ipython3
posterior_exp["posterior"]["fraction"] = posterior_exp["posterior"]["Ne_c"] / posterior_exp["posterior"]["Ne_a"]
az.summary(posterior_exp, var_names=["t0", "fraction", "Ne_a"])
```

As before, we can evaluate the absolute goodness of fit, which in this case is not very different form the two-epoch model. This means that, without generalizing, both demographic models are weakly identifiable from this genetic dataset only. 

```{code-cell} ipython3
plot_posterior_predictive(
    posterior_exp, obs_pi, obs_ld, midpoints
);
```

Finally, we can visualize the trajectories implied by a few draws from the posterior (as we did with the prior)

```{code-cell} ipython3
_samples = az.extract(posterior_exp, num_samples=50, random_seed=1)
exp_model.plot_demography(_samples)
```

## Fitting a random-walk

So far, we have fitted three models: a constant population size, a two-epoch piecewise constant and piecewise-exponential. What about more complex dynamics? Sometimes, it might interesting to fit more flexible model (those that allows for more complex dynamics). 

`bayesld` offers a "Random-walk" model where we model $N_e$ as a step-like function across a predefined grid. Here, I chose a dense 20-epoch grid with allows for changes every 5 generation in the past 100 generations.

```{code-cell} ipython3
predefined_grid = np.arange(0, 101, 1)[1:]
predefined_grid
```

In this model, the prior serves the purpose of regularizing the estimates. With 20-epochs we might likely overfit. We use the prior to indicate that, unless the data says otherwise, we prefer smooth trajectories. This is controlled by the $\sigma_{\text{step}}$ hyper parameter. 

$$
\begin{align}
N_{100} \sim \text{LogNormal}(\log(10000), 1) \\
\log(N_{i+1})-\log(N_{i+1}) \sim \text{Normal}(0, 0.05) : 0 < i < 20 \\
% NOtation for some step like function
N_e(t) = ...
\end{align}
$$

```{code-cell} ipython3
from bayesld.inference import RandomWalk

walk_model = RandomWalk(predefined_grid).with_data(
    mean_diversity=obs_pi,
    mean_ld=obs_ld,
    left_bins=left_bins_morgan,
    right_bins=right_bins_morgan,
    recombination_rate=recombination_rate,
    mutation_rate=mutation_rate,
    num_samples=num_samples,
    sequence_length=window_length_in_bp
).with_prior(
    mu_log_ne=np.log(10_000),
    sigma_log_ne=1,
    sigma_step=0.05
)
walk_model
```

In this case, it is diffucult to have intuitions about the prior. Let's sample from the prior:

```{code-cell} ipython3
prior_walk = walk_model.sample_prior()
_samples = az.extract(prior_walk, num_samples=50, random_seed=1)
walk_model.plot_demography(_samples)
```

Let's repeat the fitting procedure (notice this model will take a few more minutes to sample from).

```{code-cell} ipython3
%%time 
for i in range(4):
    walk_model = walk_model.active_learning_round(
        num_points=30, rtol=0.1, min_replicates=50,
        verbose=True, num_workers=8, seed = 13576
    )
```

```{code-cell} ipython3
posterior_walk = walk_model.sample(
    draws=2000, num_workers = 4, chains=4,
    seed=32167, verbose=True
)
```

The estimated ratio between contemporary and ancient population size has not changed dramatically:

```{code-cell} ipython3
posterior_walk["posterior"]["fraction"] = posterior_walk["posterior"]["Ne_values"][:, :, 0] / posterior_walk["posterior"]["Ne_values"][:, :, -1]
az.summary(posterior_walk, var_names="fraction")
```

If we visualized the trajectories, we can realize the estimated posterior is quite different from the one implied by the prior (which means we have learned from the data!).

```{code-cell} ipython3
_samples = az.extract(posterior_walk, num_samples=50, random_seed=1)
walk_model.plot_demography(_samples)
```

# Advance topics

Up to here, the tutorial gives an overview of the model-based inference framework behind `bayesld`. Next, I disguress a bit into (1) model comparison and model stacking and (2) assessing model convergence.

+++

## Model compariosn and model stacking

So far, we have fitted 4 different models and assess their absolute goodness of fit to the data. 
In Bayesian inference, the prior can often be used to regularize estimates and fitting a model rich (such as the random-walk) might be a good idea. However, it depends on the _a priori_ knowledge. If we have external evidence that the reason of the bottleneck was, indeed, abrupt (cause by a founder event or a catasthropic event), it is not optimal to extract conclussion from a model that either dot not enable it (such as the epxonential model) or penalizes it (the random walk). 

Still, we may want to do choose one model. Cross-fold validation is a popular approach to compare models. The idea is too choose the model that best predicts unseen genomic data (which is a very reasoanble criteria to optimize). In the last few years, a technique called Pareto-Smooth Importance Sampling Leave-One-Out Cross Validation has been popularized as a very fast approximation which does not require to fit the same model many times. I will choose how to compare differnet models based on it here. A more detail explanation can be found in the corresponding chapset of the book [Exploratory Analysis of Bayesian Models](https://arviz-devs.github.io/EABM/Chapters/Model_comparison.html). 

First, we compute the ELPD (short for expected log-predictive density, a measure of goodness of fit in unseen data) using the `az.loo` function from the `Arviz` package.

```{code-cell} ipython3
loo = {
    "constant" : az.loo(pos),
    "two_epoch" : az.loo(posterior_2epoch),
    "exponential" : az.loo(posterior_exp),
    "walk" : az.loo(posterior_walk),
}
```

According to the expected 

```{code-cell} ipython3
az.plot_compare(az.compare(loo))
```

In practice, we don't need to choose models. If the
