"""
Part 1 — Ground Truth Validation: Synthetic Experiments
=========================================================
ICU model with one pairwise interaction:

  F(x)(t) = x1*phi1(t) + x2*phi2(t) + x3*phi3(t)
           + alpha*(x1-mu)*(x2-mu)*phi12(t)

  phi1(t)  = exp(-0.2*t)                      baseline recovery
  phi2(t)  = exp(-(t-10)^2 / 2)               early shock (t~10h)
  phi3(t)  = exp(-(t-18)^2 / 2)               late deterioration (t~18h)
  phi12(t) = exp(-(t-5)^2  / 2)               X1*X2 interaction (t~5h)

  Xi ~ Uniform[0,1],  mu = 0.5,  Var(Xi) = 1/12
  alpha = 0.5  (interaction is non-negligible but doesn't dominate)

Three settings:
  (A) Analytical  — closed-form pure effects, exact Möbius coefficients
  (B) Oracle      — Möbius estimated on the TRUE model, n background samples
  (C) ML          — GBT fitted on n noisy training samples, then Möbius estimated

Three temporal granularities:
  - Time-resolved  : full effect curve over T
  - Time-specific  : evaluation at landmark hours {0, 5, 10, 18}
  - Time-aggregated: scalar Phi_S = int int K(t,s) f_S(s) ds dt  (identity kernel)

Outputs (saved to plots/part1_experiments/):
  fig1_n_recovery.pdf         — L2 error vs n for all settings and subsets
  fig2_effect_comparison.pdf  — direct comparison of analytical vs oracle vs ML
                                 for all three temporal resolutions (one figure,
                                 three rows: resolved / specific / aggregated)
  fig3_reconstruction_error.pdf — error broken down by subset S, main vs interaction

Usage:
  python part1_synthetic_experiments.py [--n_runs 30] [--seed 0]

  --n_runs   Number of Monte Carlo repetitions (default: 30; use 1 for quick test)
  --seed     Base random seed (default: 0)
"""

import os
import argparse
import itertools
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import eigh
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_runs', type=int, default=30,
                   help='Monte Carlo repetitions (default 30; use 1 for quick test)')
    p.add_argument('--seed',   type=int, default=0,
                   help='Base random seed')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Directories and global grid
# ---------------------------------------------------------------------------

PLOT_DIR = os.path.join('plots', 'part1_experiments')
os.makedirs(PLOT_DIR, exist_ok=True)

T_MAX    = 24.0
T_POINTS = 240          # reduced from 480 for speed; still fine-grained enough
t_grid   = np.linspace(0, T_MAX, T_POINTS)
dt       = t_grid[1] - t_grid[0]

MU    = 0.5
VAR_X = 1.0 / 12.0
ALPHA = 0.5             # interaction strength

# n values for the recovery curve
N_VALUES = [50, 100, 250, 500, 1000, 2000]

# Landmark time points for time-specific evaluation
T_LANDMARKS = [0.0, 5.0, 10.0, 18.0]

# Noise model parameters
NOISE_SIGMA2 = None     # set automatically to 20% of signal variance (see below)
NOISE_ELL    = 2.0      # OU length-scale for noise (shorter than explanation kernel)

# GBT hyperparameters (fixed across all n)
GBT_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42,
)

# Colour palette (consistent with icu_illustration.py)
C_ANALYTICAL = '#1b2631'   # near-black
C_ORACLE     = '#2a9d8f'   # teal
C_ML         = '#e9c46a'   # amber
C_X1         = '#c1121f'
C_X2         = '#2a9d8f'
C_X3         = '#e9c46a'
C_X12        = '#8338ec'   # purple — interaction
C_PROB       = '#e63946'

SUBSET_COLORS = {
    (1,):   C_X1,
    (2,):   C_X2,
    (3,):   C_X3,
    (1, 2): C_X12,
    (1, 3): '#f4a261',
    (2, 3): '#457b9d',
}
SUBSET_LABELS = {
    (1,):   r'$f_{\{X_1\}}$',
    (2,):   r'$f_{\{X_2\}}$',
    (3,):   r'$f_{\{X_3\}}$',
    (1, 2): r'$f_{\{X_1,X_2\}}$ (interaction)',
    (1, 3): r'$f_{\{X_1,X_3\}}$',
    (2, 3): r'$f_{\{X_2,X_3\}}$',
}

# All non-empty subsets of {1,2,3} (as sorted tuples)
ALL_SUBSETS = [(1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
# We focus reporting on the 4 most interesting ones
REPORT_SUBSETS = [(1,), (2,), (3,), (1,2)]

# ===========================================================================
# 1.  Model and analytical ground truth
# ===========================================================================

def phi1(t): return np.exp(-0.2 * t)
def phi2(t): return np.exp(-0.5 * (t - 10.0)**2)
def phi3(t): return np.exp(-0.5 * (t - 18.0)**2)
def phi12(t): return np.exp(-0.5 * (t -  5.0)**2)   # interaction basis


def model_true(x, t):
    """
    True model F(x)(t) for x shape (..., 3) and t shape (T,).
    Returns shape (..., T).
    """
    x = np.atleast_2d(x)          # (N, 3)
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    out = (x1[:, None] * phi1(t)[None, :]
         + x2[:, None] * phi2(t)[None, :]
         + x3[:, None] * phi3(t)[None, :]
         + ALPHA * (x1 - MU)[:, None] * (x2 - MU)[:, None] * phi12(t)[None, :])
    return out   # (N, T)


def analytical_pure_effects(t):
    """
    Closed-form H-FD pure effects for all subsets S, under Xi~U[0,1] independent,
    evaluated at the distribution mean (returns effect FUNCTIONS, not at a point).

    For the local prediction game at a specific x*, the pure effects are:
      f_{j}(x*)(t)    = (x*_j - mu) * phi_j(t)            for j in {1,2,3}
      f_{1,2}(x*)(t)  = alpha*(x1*-mu)*(x2*-mu)*phi12(t)
      f_{1,3}(x*)(t)  = 0   (no interaction term)
      f_{2,3}(x*)(t)  = 0
      f_{1,2,3}(x*)(t)= 0

    We evaluate at x* = (0.8, 0.9, 0.7) as in the paper.
    Returns dict: subset_tuple -> (T,) array.
    """
    x_star = np.array([0.8, 0.9, 0.7])
    effects = {}
    effects[(1,)]   = (x_star[0] - MU) * phi1(t)
    effects[(2,)]   = (x_star[1] - MU) * phi2(t)
    effects[(3,)]   = (x_star[2] - MU) * phi3(t)
    effects[(1, 2)] = ALPHA * (x_star[0] - MU) * (x_star[1] - MU) * phi12(t)
    effects[(1, 3)] = np.zeros_like(t)
    effects[(2, 3)] = np.zeros_like(t)
    effects[(1,2,3)]= np.zeros_like(t)
    return effects


def analytical_aggregated(effects_dict, t):
    """Integrate pure effects over T (identity kernel time-aggregated)."""
    return {S: np.trapz(eff, t) for S, eff in effects_dict.items()}


def compute_signal_variance(t, n_mc=20000, rng=None):
    """
    Estimate Var(F(X)(t)) averaged over t via Monte Carlo,
    used to set noise level to 20% of signal variance.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (n_mc, 3))
    F = model_true(X, t)           # (n_mc, T)
    var_t = F.var(axis=0)          # (T,)
    return float(var_t.mean())


# ===========================================================================
# 2.  Noise covariance and sample generation
# ===========================================================================

def make_ou_noise_cov(t, sigma2, ell):
    """OU covariance matrix for noise: Cov(eps(t), eps(s)) = sigma2*exp(-|t-s|/ell)."""
    K = sigma2 * np.exp(-np.abs(t[:, None] - t[None, :]) / ell)
    return K


def sample_noise(t, sigma2, ell, n, rng):
    """
    Draw n independent noise trajectories from GP(0, K_noise).
    Returns (n, T) array.
    """
    K = make_ou_noise_cov(t, sigma2, ell)
    # Cholesky with jitter
    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(t)))
    Z = rng.standard_normal((len(t), n))
    return (L @ Z).T   # (n, T)


def generate_training_data(n, t, sigma2, ell, rng):
    """
    Generate n noisy trajectory observations.
    Returns:
      X_train : (n, 3)
      Y_train : (n, T)   noisy trajectories
      F_train : (n, T)   clean trajectories (for oracle)
    """
    X = rng.uniform(0, 1, (n, 3))
    F = model_true(X, t)
    eps = sample_noise(t, sigma2, ell, n, rng)
    Y = F + eps
    return X, Y, F


# ===========================================================================
# 3.  Möbius / cooperative game estimation
# ===========================================================================

X_STAR = np.array([0.8, 0.9, 0.7])

def subset_to_mask(S, p=3):
    """Convert subset tuple (1-indexed) to boolean mask."""
    mask = np.zeros(p, dtype=bool)
    for j in S:
        mask[j - 1] = True
    return mask


def impute_x(x_star, X_bg, S_mask):
    """
    Marginal imputation: replace features outside S with draws from X_bg.
    Returns x_imputed shape (n_bg, p).
    """
    n_bg = len(X_bg)
    x_imp = X_bg.copy()
    for j in range(len(x_star)):
        if S_mask[j]:
            x_imp[:, j] = x_star[j]
    return x_imp


def compute_game_values(predict_fn, x_star, X_bg, t):
    """
    Compute local prediction game values v(S)(t) for all S,
    using marginal imputation.

    predict_fn: callable (N, p) -> (N, T)
    Returns dict: subset_tuple -> (T,) array
    """
    v = {}
    p = len(x_star)
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(1, p+1), r)
        for r in range(0, p+1)
    ))
    for S in all_S:
        if len(S) == 0:
            # v(empty) = E[F(X)(t)]
            preds = predict_fn(X_bg)   # (n_bg, T)
            v[()] = preds.mean(axis=0)
        else:
            mask = subset_to_mask(S, p)
            x_imp = impute_x(x_star, X_bg, mask)
            preds = predict_fn(x_imp)
            v[S] = preds.mean(axis=0)
    return v


def mobius_transform(v_dict, p=3):
    """
    Möbius inversion on the set function v.
    m(S) = sum_{L subset S} (-1)^{|S|-|L|} v(L)

    Returns dict: subset_tuple -> (T,) array  [or scalar if v values are scalar]
    """
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(1, p+1), r)
        for r in range(0, p+1)
    ))
    m = {}
    for S in all_S:
        val = None
        for L in itertools.chain.from_iterable(
            itertools.combinations(S, r) for r in range(len(S)+1)
        ):
            sign = (-1) ** (len(S) - len(L))
            term = v_dict.get(L, v_dict.get((), None))
            if L == ():
                term = v_dict[()]
            else:
                term = v_dict[L]
            if val is None:
                val = sign * term
            else:
                val = val + sign * term
        m[S] = val
    return m


def estimate_mobius(predict_fn, x_star, X_bg, t):
    """
    Full pipeline: game values -> Möbius inversion.
    Returns dict: subset -> (T,) array of pure effects.
    Note: m[()] is the baseline (empty set), m[S] for S non-empty are pure effects.
    """
    v = compute_game_values(predict_fn, x_star, X_bg, t)
    m = mobius_transform(v)
    return m


# ===========================================================================
# 4.  GBT model wrapper
# ===========================================================================

def fit_gbt(X_train, Y_train, t):
    """
    Fit a shared GBT on long-format data.
    Features: (x1, x2, x3, t_norm, sin(2pi*t/T), cos(2pi*t/T))
    Target: Y(t)

    Returns a predict function: (N, 3) -> (N, T)
    """
    n, T = Y_train.shape
    assert T == len(t)

    # Build long-format training data
    # Each row: (x1, x2, x3, t_j, sin_t, cos_t)
    t_norm = t / T_MAX
    sin_t  = np.sin(2 * np.pi * t / T_MAX)
    cos_t  = np.cos(2 * np.pi * t / T_MAX)

    # Tile features over time
    X_rep = np.repeat(X_train, T, axis=0)           # (n*T, 3)
    t_rep = np.tile(t_norm, n)[:, None]             # (n*T, 1)
    sin_rep = np.tile(sin_t, n)[:, None]
    cos_rep = np.tile(cos_t, n)[:, None]
    X_long = np.hstack([X_rep, t_rep, sin_rep, cos_rep])  # (n*T, 6)
    Y_long = Y_train.ravel()                        # (n*T,)

    gbt = GradientBoostingRegressor(**GBT_PARAMS)
    gbt.fit(X_long, Y_long)

    def predict_fn(X):
        """X: (N, 3) -> (N, T)"""
        N = len(X)
        X_rep_ = np.repeat(X, T, axis=0)
        t_rep_ = np.tile(t_norm, N)[:, None]
        sin_rep_ = np.tile(sin_t, N)[:, None]
        cos_rep_ = np.tile(cos_t, N)[:, None]
        X_long_ = np.hstack([X_rep_, t_rep_, sin_rep_, cos_rep_])
        preds = gbt.predict(X_long_)
        return preds.reshape(N, T)

    return predict_fn


def make_oracle_predict(t):
    """Predict function wrapping the true model (no noise)."""
    def predict_fn(X):
        return model_true(X, t)
    return predict_fn


# ===========================================================================
# 5.  Error metrics
# ===========================================================================

def l2_error_normalized(est, truth, t):
    """
    Normalized L2 error: ||est - truth||_2 / max(||truth||_2, eps)
    est, truth: (T,) arrays
    """
    norm_truth = np.sqrt(np.trapz(truth**2, t))
    norm_err   = np.sqrt(np.trapz((est - truth)**2, t))
    return norm_err / max(norm_truth, 1e-10)


def aggregated_error(est_scalar, truth_scalar):
    """Relative absolute error on scalar (time-aggregated) value."""
    return abs(est_scalar - truth_scalar) / max(abs(truth_scalar), 1e-10)


# ===========================================================================
# 6.  Single-run experiment
# ===========================================================================

def run_single(n, t, sigma2, rng):
    """
    One Monte Carlo run for a given n.
    Returns dict with errors and estimated effects for all settings.
    """
    # --- Analytical ground truth ---
    truth_effects = analytical_pure_effects(t)
    truth_agg     = analytical_aggregated(truth_effects, t)

    # --- Background samples (shared across oracle and ML) ---
    X_bg = rng.uniform(0, 1, (n, 3))

    # --- Oracle Möbius (true model, n background samples) ---
    oracle_pred = make_oracle_predict(t)
    m_oracle    = estimate_mobius(oracle_pred, X_STAR, X_bg, t)

    # --- ML Möbius (GBT fitted on noisy data) ---
    X_train, Y_train, _ = generate_training_data(n, t, sigma2, NOISE_ELL, rng)
    ml_pred   = fit_gbt(X_train, Y_train, t)
    m_ml      = estimate_mobius(ml_pred, X_STAR, X_bg, t)

    # --- Compute errors per subset ---
    results = {'oracle': {}, 'ml': {}}
    for S in REPORT_SUBSETS:
        truth = truth_effects[S]
        truth_sc = truth_agg[S]

        for tag, m in [('oracle', m_oracle), ('ml', m_ml)]:
            est = m.get(S, np.zeros_like(t))
            est_sc = float(np.trapz(est, t))
            results[tag][S] = {
                'effect':   est,
                'l2_err':   l2_error_normalized(est, truth, t),
                'agg_err':  aggregated_error(est_sc, truth_sc),
                'agg_val':  est_sc,
            }

    results['truth_effects'] = truth_effects
    results['truth_agg']     = truth_agg
    results['m_oracle']      = m_oracle
    results['m_ml']          = m_ml
    return results


# ===========================================================================
# 7.  Multi-run experiment loop
# ===========================================================================

def run_experiments(n_runs, base_seed, t, sigma2):
    """
    Run the full experiment grid: n_runs repetitions x N_VALUES x {oracle, ml}.
    Returns nested dict: n -> metric -> subset -> array of shape (n_runs,)
    """
    rng_master = np.random.default_rng(base_seed)
    seeds = rng_master.integers(0, 2**31, size=n_runs)

    # Storage: results[n][tag][S] = list of error values across runs
    from collections import defaultdict
    l2_errors  = {n: {'oracle': {S: [] for S in REPORT_SUBSETS},
                       'ml':     {S: [] for S in REPORT_SUBSETS}}
                  for n in N_VALUES}
    agg_errors = {n: {'oracle': {S: [] for S in REPORT_SUBSETS},
                       'ml':     {S: [] for S in REPORT_SUBSETS}}
                  for n in N_VALUES}

    # Also store one representative run's effects (last n, last seed) for comparison plots
    representative = None

    total_runs = len(N_VALUES) * n_runs
    run_count  = 0
    for n in N_VALUES:
        for run_idx in range(n_runs):
            rng = np.random.default_rng(seeds[run_idx])
            res = run_single(n, t, sigma2, rng)
            for tag in ('oracle', 'ml'):
                for S in REPORT_SUBSETS:
                    l2_errors[n][tag][S].append(res[tag][S]['l2_err'])
                    agg_errors[n][tag][S].append(res[tag][S]['agg_err'])
            run_count += 1
            if run_count % 10 == 0:
                print(f'  Progress: {run_count}/{total_runs}')

            # Save representative run (largest n, last run)
            if n == N_VALUES[-1] and run_idx == n_runs - 1:
                representative = res

    # Convert lists to arrays
    for n in N_VALUES:
        for tag in ('oracle', 'ml'):
            for S in REPORT_SUBSETS:
                l2_errors[n][tag][S]  = np.array(l2_errors[n][tag][S])
                agg_errors[n][tag][S] = np.array(agg_errors[n][tag][S])

    return l2_errors, agg_errors, representative


# ===========================================================================
# 8.  Plotting helpers
# ===========================================================================

def style_ax(ax, ylabel=None, xlabel='Time (h)', ylim=None):
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylim:   ax.set_ylim(*ylim)
    # shade phases
    ax.axvspan(8, 12, alpha=0.05, color=C_X2, zorder=0)
    ax.axvspan(16, 20, alpha=0.05, color=C_X3, zorder=0)


def savefig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# ===========================================================================
# 9.  Fig 1 — N-recovery plot
# ===========================================================================

def plot_n_recovery(l2_errors, agg_errors, n_runs):
    n_subsets = len(REPORT_SUBSETS)
    fig, axes = plt.subplots(2, n_subsets, figsize=(4 * n_subsets, 8))
    fig.suptitle(
        f'N-Recovery: Normalized L2 error vs training size\n'
        f'(mean ± 1 std over {n_runs} runs)',
        fontsize=12, fontweight='bold'
    )

    for col, S in enumerate(REPORT_SUBSETS):
        color = SUBSET_COLORS[S]
        label = SUBSET_LABELS[S]

        for row, (metric_dict, metric_name, ylabel) in enumerate([
            (l2_errors,  'L2',  'Normalised L2 error'),
            (agg_errors, 'Agg', 'Relative agg. error'),
        ]):
            ax = axes[row, col]
            for tag, ls, marker, lw in [
                ('oracle', '-',  'o', 2.0),
                ('ml',     '--', 's', 2.0),
            ]:
                means = np.array([metric_dict[n][tag][S].mean() for n in N_VALUES])
                stds  = np.array([metric_dict[n][tag][S].std()  for n in N_VALUES])
                label_tag = 'Oracle' if tag == 'oracle' else 'ML (GBT)'
                ax.plot(N_VALUES, means, ls=ls, marker=marker, color=color,
                        lw=lw, label=label_tag, ms=5)
                ax.fill_between(N_VALUES, means - stds, means + stds,
                                alpha=0.15, color=color)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xticks(N_VALUES)
            ax.set_xticklabels([str(n) for n in N_VALUES], fontsize=7, rotation=30)
            ax.tick_params(labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=9, fontweight='bold', color=color)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 1:
                ax.set_xlabel('n', fontsize=8)
            if col == 0 and row == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, 'fig1_n_recovery.pdf')


# ===========================================================================
# 10. Fig 2 — Direct effect comparison (3 rows × 4 subset columns)
# ===========================================================================

def plot_effect_comparison(representative, t):
    """
    Three rows:
      Row 0: Time-resolved  — full effect curves
      Row 1: Time-specific  — bar chart at landmark hours
      Row 2: Time-aggregated— scalar bars
    Four columns: one per reported subset.
    """
    n_cols = len(REPORT_SUBSETS)
    fig = plt.figure(figsize=(4.5 * n_cols, 13))
    gs  = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.55, wspace=0.30,
                            left=0.08, right=0.97, top=0.93, bottom=0.06)
    fig.suptitle(
        'Effect comparison: Analytical vs Oracle vs ML\n'
        'Representative run (largest n), three temporal granularities',
        fontsize=11, fontweight='bold'
    )

    truth_effects = representative['truth_effects']
    truth_agg     = representative['truth_agg']
    m_oracle      = representative['m_oracle']
    m_ml          = representative['m_ml']

    # ---- Row 0: Time-resolved curves ----
    for col, S in enumerate(REPORT_SUBSETS):
        ax = fig.add_subplot(gs[0, col])
        color = SUBSET_COLORS[S]
        truth = truth_effects[S]
        est_or = m_oracle.get(S, np.zeros_like(t))
        est_ml = m_ml.get(S, np.zeros_like(t))

        ax.plot(t, truth,  color=C_ANALYTICAL, lw=2.5, label='Analytical', zorder=5)
        ax.plot(t, est_or, color=C_ORACLE,     lw=2.0, ls='--', label='Oracle',   zorder=4)
        ax.plot(t, est_ml, color=C_ML,         lw=2.0, ls=':',  label='ML (GBT)', zorder=3)
        style_ax(ax, ylabel='Effect (a.u.)' if col == 0 else None)
        ax.set_title(SUBSET_LABELS[S], fontsize=9, fontweight='bold', color=color)
        if col == 0:
            ax.legend(fontsize=7.5)
        if col == n_cols - 1:
            ax.text(1.02, 0.5, 'Time-resolved', transform=ax.transAxes,
                    fontsize=9, va='center', rotation=270, color='gray')

    # ---- Row 1: Time-specific (bar chart at landmarks) ----
    for col, S in enumerate(REPORT_SUBSETS):
        ax = fig.add_subplot(gs[1, col])
        color = SUBSET_COLORS[S]
        truth  = truth_effects[S]
        est_or = m_oracle.get(S, np.zeros_like(t))
        est_ml = m_ml.get(S, np.zeros_like(t))

        x_pos = np.arange(len(T_LANDMARKS))
        w = 0.25
        vals_truth = np.interp(T_LANDMARKS, t, truth)
        vals_or    = np.interp(T_LANDMARKS, t, est_or)
        vals_ml    = np.interp(T_LANDMARKS, t, est_ml)

        ax.bar(x_pos - w,   vals_truth, width=w, color=C_ANALYTICAL, alpha=0.85, label='Analytical')
        ax.bar(x_pos,       vals_or,    width=w, color=C_ORACLE,     alpha=0.85, label='Oracle')
        ax.bar(x_pos + w,   vals_ml,    width=w, color=C_ML,         alpha=0.85, label='ML (GBT)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f't={int(h)}h' for h in T_LANDMARKS], fontsize=7.5)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if col == 0:
            ax.set_ylabel('Effect at landmark (a.u.)', fontsize=9)
            ax.legend(fontsize=7)
        if col == n_cols - 1:
            ax.text(1.02, 0.5, 'Time-specific', transform=ax.transAxes,
                    fontsize=9, va='center', rotation=270, color='gray')

    # ---- Row 2: Time-aggregated (scalar bars) ----
    ax_agg = fig.add_subplot(gs[2, :])   # span all columns
    x_pos  = np.arange(len(REPORT_SUBSETS))
    w = 0.25

    vals_truth = [truth_agg[S] for S in REPORT_SUBSETS]
    vals_or    = [float(np.trapz(m_oracle.get(S, np.zeros_like(t)), t))
                  for S in REPORT_SUBSETS]
    vals_ml    = [float(np.trapz(m_ml.get(S, np.zeros_like(t)), t))
                  for S in REPORT_SUBSETS]

    ax_agg.bar(x_pos - w,   vals_truth, width=w, color=C_ANALYTICAL, alpha=0.85, label='Analytical')
    ax_agg.bar(x_pos,       vals_or,    width=w, color=C_ORACLE,     alpha=0.85, label='Oracle')
    ax_agg.bar(x_pos + w,   vals_ml,    width=w, color=C_ML,         alpha=0.85, label='ML (GBT)')
    ax_agg.set_xticks(x_pos)
    ax_agg.set_xticklabels([SUBSET_LABELS[S] for S in REPORT_SUBSETS], fontsize=9)
    ax_agg.axhline(0, color='gray', lw=0.6, ls=':')
    ax_agg.tick_params(labelsize=8)
    ax_agg.spines['top'].set_visible(False)
    ax_agg.spines['right'].set_visible(False)
    ax_agg.set_ylabel(r'$\Phi_S = \int f_S(t)\,dt$', fontsize=10)
    ax_agg.set_title('Time-aggregated (scalar) effects', fontsize=10)
    ax_agg.legend(fontsize=9)

    savefig(fig, 'fig2_effect_comparison.pdf')


# ===========================================================================
# 11. Fig 3 — Reconstruction error breakdown
# ===========================================================================

def plot_reconstruction_error(l2_errors, agg_errors, n_runs):
    """
    Two panels side by side:
      Left:  L2 error at each n, grouped by subset, for oracle vs ML
      Right: Aggregated error at each n, grouped by subset, for oracle vs ML
    Each panel shows all four REPORT_SUBSETS with distinct colors.
    Separate lines for Oracle (solid) and ML (dashed).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Reconstruction error by subset — mean over {n_runs} runs\n'
        'Main effects (X1, X2, X3) vs interaction (X1, X2)',
        fontsize=11, fontweight='bold'
    )

    for ax, metric_dict, metric_name, ylabel in [
        (axes[0], l2_errors,  'L2',  'Normalised L2 error'),
        (axes[1], agg_errors, 'Agg', 'Relative aggregated error'),
    ]:
        for S in REPORT_SUBSETS:
            color = SUBSET_COLORS[S]
            label = SUBSET_LABELS[S]
            for tag, ls, alpha in [('oracle', '-', 1.0), ('ml', '--', 0.8)]:
                means = np.array([metric_dict[n][tag][S].mean() for n in N_VALUES])
                stds  = np.array([metric_dict[n][tag][S].std()  for n in N_VALUES])
                suffix = ' (Oracle)' if tag == 'oracle' else ' (ML)'
                ax.plot(N_VALUES, means, ls=ls, color=color, lw=2.0,
                        marker='o' if tag=='oracle' else 's', ms=5,
                        label=label + suffix, alpha=alpha)
                ax.fill_between(N_VALUES, means - stds, means + stds,
                                alpha=0.08, color=color)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) for n in N_VALUES], fontsize=8, rotation=30)
        ax.tick_params(labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('n', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(metric_name + ' error', fontsize=10)

    # Single legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    savefig(fig, 'fig3_reconstruction_error.pdf')


# ===========================================================================
# 12. Main
# ===========================================================================

def main():
    args = parse_args()
    n_runs    = args.n_runs
    base_seed = args.seed

    print(f'\n{"="*60}')
    print(f'Part 1 Synthetic Experiments')
    print(f'  n_runs    = {n_runs}')
    print(f'  base_seed = {base_seed}')
    print(f'  N_VALUES  = {N_VALUES}')
    print(f'  T_POINTS  = {T_POINTS}')
    print(f'{"="*60}\n')

    # Set noise level to 20% of mean signal variance
    print('Calibrating noise level ...')
    rng_calib = np.random.default_rng(base_seed)
    sig_var   = compute_signal_variance(t_grid, n_mc=20000, rng=rng_calib)
    sigma2    = 0.20 * sig_var
    print(f'  Signal variance (mean over t): {sig_var:.4f}')
    print(f'  Noise sigma^2                : {sigma2:.4f}  (SNR ~ {sig_var/sigma2:.1f})')

    # Quick analytical check
    print('\nAnalytical pure effects at x* = (0.8, 0.9, 0.7):')
    truth_effects = analytical_pure_effects(t_grid)
    truth_agg     = analytical_aggregated(truth_effects, t_grid)
    for S in REPORT_SUBSETS:
        print(f'  Phi_{S} = {truth_agg[S]:.4f}  '
              f'(peak |f_S| = {np.abs(truth_effects[S]).max():.4f})')

    # Run experiments
    print(f'\nRunning {n_runs} x {len(N_VALUES)} = {n_runs*len(N_VALUES)} experiments ...')
    l2_errors, agg_errors, representative = run_experiments(
        n_runs, base_seed, t_grid, sigma2
    )

    # Plots
    print('\nGenerating plots ...')
    plot_n_recovery(l2_errors, agg_errors, n_runs)
    plot_effect_comparison(representative, t_grid)
    plot_reconstruction_error(l2_errors, agg_errors, n_runs)

    print(f'\nDone. All figures saved to {PLOT_DIR}/')


if __name__ == '__main__':
    main()