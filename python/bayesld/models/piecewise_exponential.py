"""
Piecewise-exponential demographic inference model.

``PiecewiseExponentialDemography`` models::

    Ne(t) = Ne_c * exp(-alpha * t)   for t < t0
    Ne(t) = Ne_a                     for t >= t0

where ``alpha = log_fold_change / t0``.

Uses a single unified Stan program with GP bias-correction always present.
When no synthetic MC evaluation data has been accumulated (``n_synthetic = 0``),
the GP has nothing to learn from but the corrected LD is still used in the
likelihood, ensuring a single code path.
"""

import pathlib
import tempfile
from typing import Optional

import numpy as np

_STAN_DIR = pathlib.Path(__file__).resolve().parent.parent / "stan"
_THREADS_OPTS = {"cpp_options": {"STAN_THREADS": "true"}}

_DEFAULT_N_QUAD = 16

DEBUG = False


def _default_prior(diversity: np.ndarray, mutation_rate: float) -> str:
    ne_hat = float(np.mean(diversity)) / (4.0 * mutation_rate)
    log_ne_mu = float(np.log(ne_hat))
    return (
        f"    log_Ne_a ~ normal({log_ne_mu:.4f}, 1.0);\n"
        f"    log_t0   ~ normal({np.log(100.0):.4f}, 0.5);\n"
        f"    log_fold_change ~ normal(0, 1.0);"
    )


_DEFAULT_PARAMETERS = """\
    real<offset=log_ne_offset> log_Ne_a;
    real log_t0;
    real log_fold_change;"""

_DEFAULT_TRANSFORMED_PARAMETERS = """\
    real log_Ne_c = log_Ne_a + log_fold_change;
    real<lower=0> Ne_c = exp(log_Ne_c);
    real<lower=0> Ne_a = exp(log_Ne_a);
    real<lower=0> t0   = exp(log_t0);
    real alpha = log_fold_change / t0;"""


def _generate_stan(
    prior: str = "    log_Ne_a ~ normal(3, 1.0);\n    log_t0 ~ normal(4.6, 0.5);\n    log_fold_change ~ normal(0, 1.0);",
    parameters: str = _DEFAULT_PARAMETERS,
    transformed_parameters: str = _DEFAULT_TRANSFORMED_PARAMETERS,
) -> str:
    """Assemble the complete Stan program for the piecewise-exponential model.

    Three injection points: ``parameters``, ``transformed_parameters``
    (injected at top of transformed parameters block), and ``prior``.
    GP bias-correction blocks are always included.
    """
    gp_fn = (_STAN_DIR / "functions" / "gpbasisfun_functions.stan").read_text()
    shared_fn = (_STAN_DIR / "functions" / "shared.stan").read_text()
    model_fn = (_STAN_DIR / "functions" / "piecewise_exponential.stan").read_text()

    debug_tp = ""
    debug_model = ""
    if DEBUG:
        debug_tp = '    print("Ne_c = ", Ne_c, " Ne_a = ", Ne_a, " t0 = ", t0, " alpha = ", alpha);'
        debug_model = """\
    print("corrected_ld = ", corrected_ld);
    print("gp_bias_ld = ", gp_bias_ld);"""

    code = f"""\
functions {{
// ---- gpbasisfun_functions.stan ----
{gp_fn}
// ---- shared.stan ----
{shared_fn}
// ---- piecewise_exponential.stan ----
{model_fn}
}}

data {{
    int<lower=1> n_bins;
    int<lower=2> num_windows;
    vector[n_bins] left_bins;
    vector[n_bins] right_bins;
    real<lower=0> mutation_rate;
    int<lower=1> sample_size;
    vector[num_windows] pi_array;
    matrix[num_windows, n_bins] ld_mat;
    int<lower=1> n_quad;
    vector[n_quad] gl_nodes;
    vector[n_quad] gl_weights;

    // ── GP surrogate evaluation dataset (LD-bias) ──
    int<lower=0> n_synthetic;
    matrix[n_synthetic, n_bins] eval_rel_bias;
    matrix[n_synthetic, n_bins] eval_eps_rel;
    real<lower=0> hsgp_c;
    int<lower=1>  hsgp_m;
    real<lower=0> gp_alpha_std;
}}

transformed data {{
    real mean_div = mean(pi_array);
    real<lower=0> sigma_div = sd(pi_array);
    real<lower=0> sem_div   = sigma_div / sqrt(num_windows);

    vector[n_bins] mean_ld;
    vector<lower=0>[n_bins] sigma_ld;
    vector<lower=0>[n_bins] sem_ld;
    for (b in 1:n_bins) {{
        mean_ld[b]  = mean(col(ld_mat, b));
        sigma_ld[b] = sd(col(ld_mat, b));
        sem_ld[b]   = sigma_ld[b] / sqrt(num_windows);
    }}
    real log_ne_offset = log(mean_div / (4.0 * mutation_rate));

    // GP: bin midpoints and standardised r
    vector[n_bins] bin_midpoints = (left_bins + right_bins) / 2.0;
    real r_mu  = mean(bin_midpoints);
    real r_sig = sd(bin_midpoints);
    vector[n_bins] r_std = (bin_midpoints - r_mu) / r_sig;
    real L_r = hsgp_c * max(abs(r_std));
    matrix[n_bins, hsgp_m] PHI_r = PHI(n_bins, hsgp_m, L_r, r_std);

    // Pack per-bin data for map_rect parallelism
    array[n_bins, 2 + 2 * n_quad] real mr_bin_data;
    array[n_bins, 1] int mr_bin_int;
    array[n_bins] vector[0] mr_theta;
    for (b in 1:n_bins) {{
        mr_bin_data[b, 1] = left_bins[b];
        mr_bin_data[b, 2] = right_bins[b];
        for (k in 1:n_quad) {{
            mr_bin_data[b, 2 + k] = gl_nodes[k];
            mr_bin_data[b, 2 + n_quad + k] = gl_weights[k];
        }}
        mr_bin_int[b, 1] = n_quad;
    }}
}}

parameters {{
{parameters}
    real<lower=0> gp_rho_r;
    real<lower=0> gp_alpha;
    vector[hsgp_m] beta_r;
}}

transformed parameters {{
{transformed_parameters}
{debug_tp}
    vector[hsgp_m] spd_r = diagSPD_EQ(gp_alpha, gp_rho_r, L_r, hsgp_m);
    vector[n_bins] gp_bias_ld = PHI_r * (spd_r .* beta_r);
    real expected_pi = mu_div_piecewise_exponential(Ne_c, Ne_a, t0, alpha, mutation_rate,
                                                    n_quad, gl_nodes, gl_weights);
    vector[4] mr_phi = [Ne_c, Ne_a, t0, alpha]';
    vector[n_bins] approx_expected_ld = correct_ld_finite_sample(
        map_rect(mu_ld_shard_pe, mr_phi, mr_theta, mr_bin_data, mr_bin_int),
        sample_size
    );
    vector[n_bins] corrected_ld = approx_expected_ld .* (1.0 + gp_bias_ld);
}}

model {{
    // --- GP surrogate ---
    gp_rho_r ~ inv_gamma(5, 5);
    gp_alpha ~ normal(0, gp_alpha_std);
    beta_r ~ std_normal();
    if (n_synthetic > 0) {{
        to_vector(eval_rel_bias) ~ normal(
            to_vector(rep_matrix(to_row_vector(gp_bias_ld), n_synthetic)),
            to_vector(eval_eps_rel));
    }}

    // --- user prior ---
{prior}

{debug_model}
    mean_div ~ normal(expected_pi, sem_div);
    target += normal_lpdf(mean_ld | corrected_ld, sem_ld) / n_bins;
}}

generated quantities {{
    vector[num_windows] log_lik;
    for (w in 1:num_windows) {{
        log_lik[w] = normal_lpdf(pi_array[w] | expected_pi, sigma_div)
                   + normal_lpdf(to_vector(ld_mat[w]) | corrected_ld, sigma_ld) / n_bins;
    }}
}}
"""
    return code


class PiecewiseExponentialDemography:
    """
    Bayesian inference of Ne under a piecewise-exponential demographic model.

    Ne(t) = Ne_c * exp(-alpha * t) for t < t0, Ne_a for t >= t0.
    alpha = log_fold_change / t0.

    Uses a single unified Stan program with GP bias-correction always present.

    Parameters
    ----------
    diversity : array-like, shape (num_windows,)
    ld : array-like, shape (num_windows, num_bins)
    mutation_rate, recombination_rate : float
    num_samples : int
    left_bins, right_bins : array-like, shape (num_bins,)
    sequence_length : float or None
    ploidy : int
    num_workers : int
    hsgp_c : float
    hsgp_m : int
    n_quad : int
    gp_alpha_std : float
    prior : str or None
    parameters : str
    transformed_parameters : str
    """

    def __init__(
        self,
        diversity: np.ndarray,
        ld: np.ndarray,
        mutation_rate: float,
        recombination_rate: float,
        num_samples: int,
        left_bins: np.ndarray,
        right_bins: np.ndarray,
        sequence_length: Optional[float] = None,
        ploidy: int = 2,
        num_workers: int = 1,
        hsgp_c: float = 1.5,
        hsgp_m: int = 10,
        n_quad: int = _DEFAULT_N_QUAD,
        gp_alpha_std: float = 0.005,
        prior: Optional[str] = None,
        parameters: str = _DEFAULT_PARAMETERS,
        transformed_parameters: str = _DEFAULT_TRANSFORMED_PARAMETERS,
    ):
        self._diversity = np.asarray(diversity, dtype=float)
        self._ld = np.asarray(ld, dtype=float)
        self._mutation_rate = float(mutation_rate)
        self._recombination_rate = float(recombination_rate)
        self._sequence_length = (
            float(sequence_length) if sequence_length is not None else None
        )
        self._ploidy = int(ploidy)
        self._num_samples = int(num_samples)
        self._left_bins = np.asarray(left_bins, dtype=float)
        self._right_bins = np.asarray(right_bins, dtype=float)
        self._num_workers = int(num_workers)
        self._hsgp_c = float(hsgp_c)
        self._hsgp_m = int(hsgp_m)
        self._gp_alpha_std = float(gp_alpha_std)

        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
        self._gl_nodes = gl_nodes
        self._gl_weights = gl_weights

        self._synthetic_points: list[dict] = []

        self._prior = (
            prior
            if prior is not None
            else _default_prior(self._diversity, self._mutation_rate)
        )
        self._parameters = parameters
        self._transformed_parameters = transformed_parameters

        self._tmpdir = pathlib.Path(tempfile.mkdtemp())
        self._stan_code = _generate_stan(
            self._prior, self._parameters, self._transformed_parameters
        )
        self._model = self._compile(self._stan_code)

    # ──────────────────────────────────────────────────────────────────────
    # Compilation
    # ──────────────────────────────────────────────────────────────────────

    def _compile(self, code: str):
        import cmdstanpy

        stan_file = self._tmpdir / "piecewise_exponential_ne.stan"
        stan_file.write_text(code)
        return cmdstanpy.CmdStanModel(stan_file=str(stan_file), **_THREADS_OPTS)

    # ──────────────────────────────────────────────────────────────────────
    # Primitives
    # ──────────────────────────────────────────────────────────────────────

    def get_stan_code(self) -> str:
        return self._stan_code

    @property
    def model(self):
        return self._model

    def stan_data(self) -> dict:
        n_syn = len(self._synthetic_points)
        data = {
            "n_bins": int(len(self._left_bins)),
            "num_windows": int(len(self._diversity)),
            "left_bins": self._left_bins,
            "right_bins": self._right_bins,
            "mutation_rate": self._mutation_rate,
            "sample_size": self._num_samples,
            "pi_array": self._diversity,
            "ld_mat": self._ld,
            "n_quad": len(self._gl_nodes),
            "gl_nodes": self._gl_nodes,
            "gl_weights": self._gl_weights,
            "n_synthetic": n_syn,
            "hsgp_c": self._hsgp_c,
            "hsgp_m": self._hsgp_m,
            "gp_alpha_std": self._gp_alpha_std,
        }
        if n_syn > 0:
            data["eval_rel_bias"] = np.array(
                [p["rel_bias"] for p in self._synthetic_points]
            )
            data["eval_eps_rel"] = np.array(
                [p["eps_rel"] for p in self._synthetic_points]
            )
        else:
            n_bins = int(len(self._left_bins))
            data["eval_rel_bias"] = np.empty((0, n_bins))
            data["eval_eps_rel"] = np.empty((0, n_bins))
        return data

    # ──────────────────────────────────────────────────────────────────────
    # Data and prior mutation
    # ──────────────────────────────────────────────────────────────────────

    def update_data(
        self,
        diversity: Optional[np.ndarray] = None,
        ld: Optional[np.ndarray] = None,
        mutation_rate: Optional[float] = None,
        recombination_rate: Optional[float] = None,
        num_samples: Optional[int] = None,
        sequence_length: Optional[float] = None,
    ) -> None:
        if diversity is not None:
            self._diversity = np.asarray(diversity, dtype=float)
        if ld is not None:
            self._ld = np.asarray(ld, dtype=float)
        if mutation_rate is not None:
            self._mutation_rate = float(mutation_rate)
        if recombination_rate is not None:
            self._recombination_rate = float(recombination_rate)
        if num_samples is not None:
            self._num_samples = int(num_samples)
        if sequence_length is not None:
            self._sequence_length = float(sequence_length)

    def update_prior(
        self,
        prior: Optional[str] = None,
        parameters: Optional[str] = None,
        transformed_parameters: Optional[str] = None,
        gp_alpha_std: Optional[float] = None,
    ) -> None:
        needs_recompile = False
        if prior is not None and prior != self._prior:
            self._prior = prior
            needs_recompile = True
        if parameters is not None and parameters != self._parameters:
            self._parameters = parameters
            needs_recompile = True
        if (
            transformed_parameters is not None
            and transformed_parameters != self._transformed_parameters
        ):
            self._transformed_parameters = transformed_parameters
            needs_recompile = True
        if gp_alpha_std is not None:
            self._gp_alpha_std = float(gp_alpha_std)

        if needs_recompile:
            self._stan_code = _generate_stan(
                self._prior, self._parameters, self._transformed_parameters
            )
            self._model = self._compile(self._stan_code)

    # ──────────────────────────────────────────────────────────────────────
    # Synthetic bias data
    # ──────────────────────────────────────────────────────────────────────

    @property
    def synthetic_points(self) -> list[dict]:
        return list(self._synthetic_points)

    def add_synthetic_points(self, points: list[dict]) -> None:
        required = {"rel_bias", "eps_rel"}
        for p in points:
            if not required.issubset(p):
                raise ValueError(f"Each point must have keys {required}; got {set(p)}")
        self._synthetic_points.extend(points)

    # ──────────────────────────────────────────────────────────────────────
    # MC evaluation (internal)
    # ──────────────────────────────────────────────────────────────────────

    def _mc_eval(
        self,
        ne_c: float,
        ne_a: float,
        t0: float,
        alpha: float,
        mc_seed: int,
        rtol: float = 0.01,
        model=None,
    ) -> dict:
        import msprime

        from .. import deterministic as det
        from .. import montecarlo

        if model is None:
            model = msprime.SMCK(k=1)

        _, det_ld_raw = det.expected_piecewise_exponential(
            ne_c,
            ne_a,
            t0,
            alpha,
            self._left_bins,
            self._right_bins,
            self._mutation_rate,
            sample_size=self._num_samples,
            ploidy=self._ploidy,
        )
        det_ld = np.asarray(det_ld_raw)
        _, mc_ld_reps = montecarlo.expected_piecewise_exponential(
            ne_c,
            ne_a,
            t0,
            alpha,
            self._left_bins,
            self._right_bins,
            self._mutation_rate,
            self._recombination_rate,
            self._sequence_length,
            self._num_samples,
            random_seed=mc_seed,
            ploidy=self._ploidy,
            model=model,
            num_workers=self._num_workers,
            rtol=rtol,
        )
        mc_ld_reps = np.asarray(mc_ld_reps)
        assert len(mc_ld_reps) > 1, (
            f"MC evaluation returned only {len(mc_ld_reps)} replicate(s); "
            "need at least 2 for a meaningful SE estimate."
        )
        mc_ld_rel = mc_ld_reps / det_ld - 1.0
        return {
            "rel_bias": mc_ld_rel.mean(axis=0),
            "eps_rel": mc_ld_rel.std(axis=0, ddof=1) / np.sqrt(len(mc_ld_reps)),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Active learning
    # ──────────────────────────────────────────────────────────────────────

    def active_learn_bias(
        self,
        n_points_per_iter: int = 5,
        n_iter: int = 5,
        max_tolerance: float = 0.1,
        strategy: str = "pathfinder",
        model=None,
        seed: Optional[int] = None,
        progress_bar: bool = True,
    ) -> None:
        if self._sequence_length is None:
            raise ValueError(
                "sequence_length must be provided at initialisation "
                "to use active_learn_bias."
            )

        from tqdm.auto import tqdm

        rng = np.random.default_rng(seed)

        for iteration in range(n_iter):
            draws = self._get_draws(n_points_per_iter, strategy, rng)
            print(
                f"[{strategy.upper()} iter={iteration + 1}/{n_iter} "
                f"n_synthetic={len(self._synthetic_points)}]  "
                f"Ne_c={draws['Ne_c'].mean():,.0f}  Ne_a={draws['Ne_a'].mean():,.0f}  "
                f"t0={draws['t0'].mean():.1f}"
            )

            iterator = tqdm(
                range(len(draws["Ne_c"])),
                desc=f"Active learning (iter {iteration + 1})",
                disable=not progress_bar,
            )
            for i in iterator:
                mc_seed = int(rng.integers(2**31))
                self._synthetic_points.append(
                    self._mc_eval(
                        float(draws["Ne_c"][i]),
                        float(draws["Ne_a"][i]),
                        float(draws["t0"][i]),
                        float(draws["alpha"][i]),
                        mc_seed,
                        rtol=max_tolerance,
                        model=model,
                    )
                )

    def _get_draws(self, n_draws: int, strategy: str, rng: np.random.Generator) -> dict:
        data = self.stan_data()

        if strategy != "pathfinder":
            raise ValueError(
                f"Unknown strategy {strategy!r}; only 'pathfinder' is supported."
            )

        pf = self._model.pathfinder(
            data=data,
            draws=n_draws,
            seed=int(rng.integers(10_000)),
            num_threads=self._num_workers,
            show_console=False,
        )
        return {
            "Ne_c": np.asarray(pf.stan_variable("Ne_c"))[:n_draws],
            "Ne_a": np.asarray(pf.stan_variable("Ne_a"))[:n_draws],
            "t0": np.asarray(pf.stan_variable("t0"))[:n_draws],
            "alpha": np.asarray(pf.stan_variable("alpha"))[:n_draws],
        }

    # ──────────────────────────────────────────────────────────────────────
    # Happy path
    # ──────────────────────────────────────────────────────────────────────

    def sample(self, chains: int = 2, **kwargs):
        import arviz
        import xarray as xr

        kwargs.setdefault("threads_per_chain", self._num_workers)
        kwargs.setdefault("show_console", False)
        data = self.stan_data()

        fits = []
        for i in range(chains):
            chain_seed = kwargs.get("seed", 12345) + i
            fit = self._model.sample(
                data=data,
                chains=1,
                seed=chain_seed,
                **{k: v for k, v in kwargs.items() if k != "seed"},
            )
            fits.append(fit)

        trees = [arviz.from_cmdstanpy(f) for f in fits]
        if len(trees) == 1:
            return trees[0]

        children = {}
        for group in trees[0].children:
            children[group] = xr.concat([t[group].ds for t in trees], dim="chain")
        return xr.DataTree.from_dict(children)
