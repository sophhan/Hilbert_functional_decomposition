"""
Functional Explanation Framework — Combined Energy Demand Example
=================================================================

This script applies the functional feature attribution framework to two
energy demand prediction tasks, comparing a single-household dataset
(UCI IHEPC) with a national-grid dataset (NESO GB Demand).

A Random Forest is trained to map day-level calendar and lagged demand
features to daily demand trajectories. Cooperative game decompositions
(prediction, sensitivity, and risk games) are then computed under two
output kernels (identity and correlation), illustrating how the correlation
structure of the output distribution shapes the attribution profile.

Data availability
-----------------
The IHEPC dataset can be downloaded automatically via the `ucimlrepo`
package, or loaded from a local parquet cache at:
    data/household_power_consumption.parquet

The NESO demand data files must be obtained from the NESO data portal
and placed in the data/ directory as:
    data/demanddata_{year}.csv   for each year in NESO_YEARS

Game result caches are provided in game_results_energy/. If the raw data
is unavailable, all figures can be reproduced from these caches:

    game_results_energy/global_ihepc_sensitivity_v8.npz
    game_results_energy/global_ihepc_risk_v8.npz
    game_results_energy/global_neso_sensitivity_v8.npz
    game_results_energy/global_neso_risk_v8.npz
    game_results_energy/local_ihepc_*_v7.npz
    game_results_energy/local_neso_*_v7.npz
    game_results_energy/localpred_*_v7.npz
    game_results_energy/pdp_*_v7.npz

Cache version constants (do not change unless recomputing from scratch):
    CACHE_VERSION_GLOBAL = 'v8'   — global sensitivity / risk game caches
    CACHE_VERSION_LOCAL  = 'v7'   — local prediction game caches

Features
--------
Both datasets use day-level predictors:
    day_of_week     — integer 0–6 (Monday–Sunday)
    is_weekend      — binary: 1 on Saturday or Sunday
    month           — integer 1–12
    season          — integer 1–4 (Winter/Spring/Summer/Autumn)
    lag_daily_mean  — previous day's mean demand
    lag_morning     — previous day's morning-period mean demand
    lag_evening     — previous day's evening-period mean demand (NESO only)

Targets
-------
IHEPC: hourly mean global active power (kW), diurnal-mean-adjusted (24 time points)
NESO:  half-hourly national demand (MW), diurnal-mean-adjusted (48 time points)

Cooperative games
-----------------
Three game formulations are implemented:

  Prediction game (local):
      v(S)(t) = E_{X_{-S}}[ F(x*_S, X_{-S})(t) ]
      Measures how much each feature shifts the predicted trajectory
      for a specific explicand x*.

  Sensitivity game (global):
      v(S)(t) = Var_{X_S}[ E_{X_{-S}}[ F(X_S, X_{-S})(t) ] ]
      Measures how much of the trajectory variance is explained by S.
      Pure = closed Sobol; Partial = Shapley sensitivity; Full = total Sobol.

  Risk / MSE game (global):
      v(S)(t) = baseline_loss(t) - E[(Y(t) - E_{X_{-S}}[F(X_S,X_{-S})(t)])^2]
      Measures risk reduction relative to predicting with the global mean.
      Pure = pure risk reduction; Partial = SAGE; Full = PFI.

Global games use a double Monte Carlo estimator (n_outer, n_inner) with
shared background pools to reduce cross-coalition variance.

Output kernels
--------------
  Identity kernel:     K(t,s) = delta(t-s) — pointwise, no coupling
  Correlation kernel:  K derived from the empirical output covariance
      K(t,s) = Cov(Y(t), Y(s)) / sqrt(Var(Y(t)) * Var(Y(s)))
      This couples time points that co-vary across days in the observed
      demand data, making it data-adaptive and scale-free.

Key finding
-----------
The IHEPC correlation kernel shows a block-diagonal structure (local
temporal correlation), while the NESO kernel shows a broader daily pattern
(high correlation between morning and evening peaks). This difference in
correlation structure produces markedly different attribution profiles
for the same features, illustrating how kernel choice captures domain-
specific temporal dependence.

Output figures
--------------
All figures saved to plots/energy/:

    fig0_main_body.pdf                           — heatmap + network + sens panel
    fig1_global_risk_sensitivity_{tag}.pdf       — global sens + risk, 4 rows
    fig2_local_prediction_{tag}.pdf              — local prediction, 2 rows
    fig3_global_pdp_{tag}.pdf                    — PDP-style global prediction
    fig4_interactions_{tag}.pdf                  — top pairwise interactions
    fig5_networks_global.pdf                     — network plots, 2x3 grid

Usage
-----
    python energy_hfd.py

With precomputed caches (no raw data required):
    Ensure game_results_energy/*.npz are present, then run as above.
    Steps [5]–[8] load from cache automatically when available.
"""

import itertools
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.colors
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# 0.  Global settings
# ---------------------------------------------------------------------------

# Cache version strings — must match the .npz filenames in game_results_energy/.
# Change only if recomputing games from scratch with new settings.
CACHE_VERSION_GLOBAL = 'v8'   # global sensitivity / risk game caches
CACHE_VERSION_LOCAL  = 'v7'   # local prediction game caches (unchanged)

_HERE    = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 42
RF_N_EST = 300
RF_JOBS  = -1

BASE_PLOT_DIR  = os.path.join('plots', 'energy')
GAME_CACHE_DIR = os.path.join(_HERE, 'game_results_energy')

# Double-MC settings for global sensitivity / risk games
GLOBAL_N_OUTER = 100   # outer pool: draws of X_S (the "fixed" features)
GLOBAL_N_INNER = 100   # inner pool: draws of X_{-S} (marginalized features)

# Local-prediction-game settings
GLOBAL_N_INSTANCES = 30    # number of instances for averaged local-pred
GLOBAL_SAMPLE_SIZE = 100   # background samples per instance
N_FEAT_BINS        = 20    # bins for PDP marginal feature plots
PDP_N_INSTANCES    = 120   # instances for stratified PDP sample

# Features treated as continuous for PDP binning (vs. discrete)
CONTINUOUS_FEATURES = {'lag_daily_mean', 'lag_morning', 'lag_evening'}

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------

FS_SUPTITLE = 13
FS_TITLE    = 11
FS_AXIS     = 10
FS_TICK     = 9
FS_LEGEND   = 8.5
FS_ANNOT    = 8.5


def _fs(bump=0):
    """Return a SimpleNamespace of font sizes with an optional global bump."""
    return SimpleNamespace(
        suptitle=FS_SUPTITLE+bump, title=FS_TITLE+bump,
        axis=FS_AXIS+bump,   tick=FS_TICK+bump,
        legend=FS_LEGEND+bump, annot=FS_ANNOT+bump)

# ---------------------------------------------------------------------------
# Dataset-specific constants
# ---------------------------------------------------------------------------

# IHEPC: UCI Individual Household Electric Power Consumption
IHEPC_DATA_DIR  = os.path.join(_HERE, 'data')
IHEPC_DATA_FILE = os.path.join(IHEPC_DATA_DIR, 'household_power_consumption.parquet')
IHEPC_T        = 24                                         # hourly time points
IHEPC_LABELS   = ['{:02d}:00'.format(h) for h in range(IHEPC_T)]
IHEPC_TGRID    = np.arange(IHEPC_T, dtype=float)
IHEPC_FEATURES = ['day_of_week','is_weekend','month','season',
                  'lag_daily_mean','lag_morning']
IHEPC_MORNING  = (6, 10)    # morning activity window (hour indices)
IHEPC_EVENING  = (17, 22)   # evening activity window
IHEPC_SAMPLE   = {'prediction': 150, 'sensitivity': 200, 'risk': 200}
IHEPC_YLABEL   = {
    'prediction' : 'Effect on power (kW)',
    'sensitivity': r'Var$[F(t)]$ (kW$^2$)',
    'risk'       : r'Risk reduction (kW$^2$)',
}

# NESO: National Energy System Operator, GB half-hourly demand
NESO_DATA_DIR  = os.path.join(_HERE, 'data')
NESO_YEARS     = [2018, 2019, 2020, 2021, 2022]
NESO_T         = 48                                          # half-hourly time points
NESO_LABELS    = ['{:02d}:{:02d}'.format((i*30)//60,(i*30)%60) for i in range(NESO_T)]
NESO_TGRID     = np.arange(NESO_T, dtype=float)
NESO_FEATURES  = ['day_of_week','is_weekend','month','season',
                  'lag_daily_mean','lag_morning','lag_evening']
NESO_MORNING   = (12, 19)   # morning demand period (half-hour indices)
NESO_EVENING   = (34, 42)   # evening demand period
NESO_SAMPLE    = {'prediction': 150, 'sensitivity': 200, 'risk': 200}
NESO_YLABEL    = {
    'prediction' : 'Effect on demand (MW)',
    'sensitivity': r'Var$[F(t)]$ (MW$^2$)',
    'risk'       : r'Risk reduction (MW$^2$)',
}

GAME_TYPES        = ['prediction', 'sensitivity', 'risk']
GLOBAL_GAME_TYPES = ['sensitivity', 'risk']   # handled by GlobalFunctionalGame

# Feature colors (consistent across both datasets)
FEAT_COLORS = {
    'day_of_week'   : '#1f77b4',
    'is_weekend'    : '#ff7f0e',
    'month'         : '#2ca02c',
    'season'        : '#d62728',
    'lag_daily_mean': '#9467bd',
    'lag_morning'   : '#8c564b',
    'lag_evening'   : '#e377c2',
}

DS_LABEL = {
    'ihepc': 'UCI IHEPC\n(Single household, kW)',
    'neso' : 'NESO GB Demand\n(National grid, MW)',
}
DS_COLOR = {'ihepc': '#2a9d8f', 'neso': '#e76f51'}

# Short abbreviations for feature labels in network node diagrams
FEAT_ABBR = {
    'day_of_week'   : 'DoW', 'is_weekend'    : 'WeD',
    'month'         : 'Mon', 'season'        : 'Sea',
    'lag_daily_mean': 'LDM', 'lag_morning'   : 'LMo',
    'lag_evening'   : 'LEv',
}

SHADE_AM_COLOR = '#4a90e2'   # morning activity window shading
SHADE_PM_COLOR = '#e24a4a'   # evening activity window shading
SHADE_ALPHA    = 0.16

# Network node and edge colors
_NODE_POS = '#2a9d8f'   # positive attribution node
_NODE_NEG = '#e63946'   # negative attribution node
_EDGE_SYN = '#2a9d8f'   # synergistic (positive) interaction edge
_EDGE_RED = '#e63946'   # redundant (negative) interaction edge

# Display labels for pure / partial / full effects per game type
_SENS_LABELS = {
    'pure'   : r'Pure = closed Sobol',
    'partial': r'Partial = Shapley-sens.',
    'full'   : r'Full = total Sobol',
}
_RISK_LABELS = {
    'pure'   : r'Pure = pure risk',
    'partial': r'Partial = SAGE',
    'full'   : r'Full = PFI',
}
_LOCAL_PRED_LABELS = {
    'pure'   : 'Pure',
    'partial': 'Partial',
    'full'   : 'Full',
}
_GLOBAL_PRED_LABELS = {
    'pure'   : 'Pure (= PDP)',
    'partial': 'Partial (= global SHAP)',
    'full'   : 'Full',
}
_EFFECT_TYPES = ['pure', 'partial', 'full']


# ===========================================================================
# 0b.  Color shading helpers
# ===========================================================================

def _hex_to_rgb01(c):
    return matplotlib.colors.to_rgb(c)


def _shade_color(c, factor):
    """
    Lighten (factor > 0) or darken (factor < 0) a matplotlib color.
    factor=0 returns the original color.
    """
    r, g, b = _hex_to_rgb01(c)
    if factor >= 0:
        return (r + (1.0 - r) * factor,
                g + (1.0 - g) * factor,
                b + (1.0 - b) * factor)
    else:
        f = -factor
        return (r * (1.0 - f), g * (1.0 - f), b * (1.0 - f))


# ===========================================================================
# 1.  Infrastructure
# ===========================================================================

def _require_dir(path):
    os.makedirs(path, exist_ok=True)


def _month_to_season(m):
    """Map calendar month (1–12) to season index (1=Winter, ..., 4=Autumn)."""
    if m in (12, 1, 2):  return 1
    elif m in (3, 4, 5): return 2
    elif m in (6, 7, 8): return 3
    else:                return 4


# ===========================================================================
# 2.  Cache helpers
# ===========================================================================

def _cache_path_global(ds_tag, game_type):
    """
    Path for the single global-game cache file per (dataset, game_type).
    Global games do not depend on a specific explicand, so one file suffices.
    """
    return os.path.join(
        GAME_CACHE_DIR,
        'global_{}_{}_{}.npz'.format(ds_tag, game_type, CACHE_VERSION_GLOBAL))


def _cache_path_local(ds_tag, label):
    """Path for a named local-prediction game cache."""
    safe = label.replace(' ', '_').replace('-', '_')
    return os.path.join(
        GAME_CACHE_DIR,
        'local_{}_{}_{}.npz'.format(ds_tag, safe, CACHE_VERSION_LOCAL))


def _cache_path_local_pred_inst(ds_tag, instance_k):
    """Path for per-instance local prediction game cache (PDP / averaged SHAP)."""
    return os.path.join(
        GAME_CACHE_DIR,
        'localpred_{}_{:04d}_{}.npz'.format(ds_tag, instance_k, CACHE_VERSION_LOCAL))


def _save_cache(path, x_inst, pure, partial, full, moebius, n_players):
    """
    Save pure, partial (Shapley), full effects and the full Möbius dictionary
    to a compressed .npz file. Möbius keys are serialised as underscore-joined
    strings, with the empty set stored as 'empty'.
    """
    mob_keys, mob_arrays = [], []
    for S, arr in moebius.items():
        mob_keys.append('_'.join(str(x) for x in S) if S else 'empty')
        mob_arrays.append(arr)
    np.savez_compressed(
        path,
        x_inst=x_inst,
        **{'pure_{}'.format(i):    pure[i]    for i in range(n_players)},
        **{'partial_{}'.format(i): partial[i] for i in range(n_players)},
        **{'full_{}'.format(i):    full[i]    for i in range(n_players)},
        mob_keys=np.array(mob_keys, dtype=object),
        **{'mob_{}'.format(k): arr for k, arr in enumerate(mob_arrays)},
        n_players=np.array([n_players]),
        n_mob=np.array([len(mob_keys)]),
    )


def _load_cache(path, n_players):
    """
    Load a cache file saved by _save_cache.
    Reconstructs the Möbius dictionary from serialised key strings.
    """
    d = np.load(path, allow_pickle=True)
    x_inst  = d['x_inst']
    pure    = {i: d['pure_{}'.format(i)]    for i in range(n_players)}
    partial = {i: d['partial_{}'.format(i)] for i in range(n_players)}
    full    = {i: d['full_{}'.format(i)]    for i in range(n_players)}
    mob_keys = list(d['mob_keys'])
    n_mob    = int(d['n_mob'][0])
    moebius  = {}
    for k in range(n_mob):
        key_str = mob_keys[k]
        arr     = d['mob_{}'.format(k)]
        S = () if key_str == 'empty' else tuple(int(x) for x in key_str.split('_'))
        moebius[S] = arr
    return x_inst, pure, partial, full, moebius


# ===========================================================================
# 3.  Model
# ===========================================================================

class RFModel:
    """
    Random Forest regressor wrapper for multi-output trajectory prediction.
    Predicts the full T-dimensional demand trajectory from day-level features.
    """
    def __init__(self, random_state=RNG_SEED):
        self.model = RandomForestRegressor(
            n_estimators=RF_N_EST, n_jobs=RF_JOBS, random_state=random_state)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_te, Y_te):
        """Return R² on trajectory-level predictions."""
        Yp = self.predict(X_te)
        return 1.0 - np.sum((Y_te - Yp)**2) / np.sum((Y_te - Y_te.mean())**2)


# ===========================================================================
# 4.  Cooperative games
# ===========================================================================

class LocalPredictionGame:
    """
    Local (instance-specific) prediction game.

        v(S)(t) = E_{X_{-S}}[ F(x*_S, X_{-S})(t) ]

    Estimates the conditional expectation by marginalising over background
    samples with features in S fixed to the explicand x*.

    Parameters
    ----------
    predict_fn : callable (n, p) -> (n, T)
    X_bg       : ndarray (n_bg, p) — reference distribution
    x_exp      : ndarray (p,) — the specific input being explained
    T          : int — number of time points
    features   : list of feature names
    sample_size: number of background samples per coalition
    """
    def __init__(self, predict_fn, X_bg, x_exp, T, features,
                 sample_size=150, random_seed=RNG_SEED):
        self.predict_fn = predict_fn
        self.X_bg  = X_bg
        self.x_exp = x_exp
        self.T     = T
        self.n     = sample_size
        self.seed  = random_seed
        self.p     = len(features)
        self.player_names = list(features)
        self.coalitions   = np.array(
            list(itertools.product([False, True], repeat=self.p)), dtype=bool)
        self.nc  = len(self.coalitions)
        self._idx = {tuple(c): i for i, c in enumerate(self.coalitions)}
        self.values = None

    def _impute(self, coal):
        """Sample background rows and fix features in coal to x_exp."""
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(self.X_bg), size=self.n)
        X   = self.X_bg[idx].copy()
        for j in range(self.p):
            if coal[j]:
                X[:, j] = self.x_exp[j]
        return X

    def value_function(self, coal):
        X  = self._impute(coal)
        Yp = self.predict_fn(X)
        return Yp.mean(axis=0)

    def precompute(self):
        """Evaluate v(S) for all 2^p coalitions."""
        self.values = np.zeros((self.nc, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i+1) % 32 == 0 or i+1 == self.nc:
                print('      {}/{}'.format(i+1, self.nc))

    def __getitem__(self, c):
        return self.values[self._idx[c]]


class GlobalFunctionalGame:
    """
    Global sensitivity or risk game using a double Monte Carlo estimator.

    Sensitivity:
        v(S)(t) = Var_{X_S}[ E_{X_{-S}}[ F(X_S, X_{-S})(t) ] ]

        For each outer draw x_S^(k), compute the inner conditional mean
        g_S(x_S^(k))(t) = E_{X_{-S}}[F(x_S^(k), X_{-S})(t)].
        Then v(S)(t) = Var_k[ g_S(x_S^(k))(t) ].

    Risk (SAGE/PFI sign convention — positive means loss reduction):
        baseline_loss(t) = E[(Y(t) - mu(t))^2]  with mu = E[F(X)]
        full_loss(S)(t)  = E[(Y(t) - g_S(X_S)(t))^2]
        v(S)(t)          = baseline_loss(t) - full_loss(S)(t)

    Both games:
        v(empty)(t) = 0   — empty coalition explains nothing
        v(N)(t)     = Var[F(t)] or total risk reduction (maximum)

    A shared outer/inner pool is drawn once and reused across coalitions,
    reducing cross-coalition variance in the Möbius differences.

    Parameters
    ----------
    predict_fn : callable (n, p) -> (n, T)
    X_bg       : ndarray (n_bg, p) — background distribution
    Y_bg       : ndarray (n_bg, T) — observed targets (required for risk)
    game_type  : 'sensitivity' or 'risk'
    n_outer    : outer pool size (draws of X_S)
    n_inner    : inner pool size (draws of X_{-S})
    """
    def __init__(self, predict_fn, X_bg, Y_bg, T, features,
                 game_type, n_outer=GLOBAL_N_OUTER, n_inner=GLOBAL_N_INNER,
                 random_seed=RNG_SEED):
        if game_type not in GLOBAL_GAME_TYPES:
            raise ValueError('GlobalFunctionalGame only supports '
                             'sensitivity / risk; got {}'.format(game_type))
        if game_type == 'risk' and Y_bg is None:
            raise ValueError('Y_bg required for risk game.')
        self.predict_fn = predict_fn
        self.X_bg  = X_bg
        self.Y_bg  = Y_bg
        self.T     = T
        self.game_type = game_type
        self.n_outer   = n_outer
        self.n_inner   = n_inner
        self.seed  = random_seed
        self.p     = len(features)
        self.player_names = list(features)
        self.coalitions   = np.array(
            list(itertools.product([False, True], repeat=self.p)), dtype=bool)
        self.nc  = len(self.coalitions)
        self._idx = {tuple(c): i for i, c in enumerate(self.coalitions)}
        self.values = None

        # Draw shared outer/inner pools once — reused across all coalitions
        # so noise is correlated across S, reducing variance of Möbius differences.
        rng = np.random.default_rng(self.seed)
        self._outer_idx = rng.choice(len(X_bg), size=n_outer, replace=True)
        self._inner_idx = rng.choice(len(X_bg), size=n_inner, replace=True)

        # mu = E[F(X)] from the inner pool — used as the baseline predictor
        Xb            = self.X_bg[self._inner_idx]
        Yp_b          = self.predict_fn(Xb)
        self._mu      = Yp_b.mean(axis=0)

        if game_type == 'risk':
            Y_outer = self.Y_bg[self._outer_idx]
            self._baseline_loss = ((Y_outer - self._mu[None, :])**2).mean(axis=0)
        else:
            self._baseline_loss = None

    def _marginalized_predictions(self, coal):
        """
        Compute g_S(x_S^(k))(t) for all outer draws k via a batched prediction.

        For each outer draw x_S^(k), construct n_inner copies with X_S fixed to
        x_S^(k) and X_{-S} from the inner pool, then average predictions.
        Total batch size: n_outer * n_inner.

        Returns ndarray of shape (n_outer, T).
        """
        X_outer = self.X_bg[self._outer_idx]
        X_inner = self.X_bg[self._inner_idx]
        n_outer, n_inner, p = self.n_outer, self.n_inner, self.p
        X_batch = np.tile(X_inner, (n_outer, 1))
        for j in range(p):
            if coal[j]:
                X_batch[:, j] = np.repeat(X_outer[:, j], n_inner)
        Yp = self.predict_fn(X_batch)
        Yp = Yp.reshape(n_outer, n_inner, self.T)
        return Yp.mean(axis=1)

    def value_function(self, coal):
        """Evaluate v(coal)(t) for a single coalition."""
        if not any(coal):
            # Empty coalition: marginalized predictor = mu, so Var = 0 and
            # risk reduction = 0 by construction.
            return np.zeros(self.T)
        if all(coal):
            # Full coalition: no marginalization needed
            X_full = self.X_bg[self._outer_idx]
            Yp     = self.predict_fn(X_full)
            if self.game_type == 'sensitivity':
                return Yp.var(axis=0)
            else:
                Y         = self.Y_bg[self._outer_idx]
                full_loss = ((Y - Yp)**2).mean(axis=0)
                return self._baseline_loss - full_loss
        g_S = self._marginalized_predictions(coal)
        if self.game_type == 'sensitivity':
            return g_S.var(axis=0)
        else:
            Y         = self.Y_bg[self._outer_idx]
            full_loss = ((Y - g_S)**2).mean(axis=0)
            return self._baseline_loss - full_loss

    def precompute(self):
        """Evaluate v(S) for all 2^p coalitions."""
        self.values = np.zeros((self.nc, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i+1) % 16 == 0 or i+1 == self.nc:
                print('      {}/{}'.format(i+1, self.nc))

    def __getitem__(self, c):
        return self.values[self._idx[c]]


# ===========================================================================
# 5.  Möbius transform and Shapley values
# ===========================================================================

def moebius_transform(game):
    """
    Compute functional Möbius coefficients m_S(t) for all subsets S.

    Inverts v(S) = sum_{L ⊆ S} m(L) via inclusion-exclusion:
        m(S)(t) = sum_{L ⊆ S} (-1)^{|S|-|L|} v(L)(t)

    Returns dict mapping subset tuples -> ndarray of shape (T,).
    """
    p     = game.p
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(p), r) for r in range(p+1)))
    mob = {}
    for S in all_S:
        m = np.zeros(game.T)
        for L in itertools.chain.from_iterable(
                itertools.combinations(S, r) for r in range(len(S)+1)):
            c = tuple(i in L for i in range(p))
            m += (-1)**(len(S)-len(L)) * game[c]
        mob[S] = m
    return mob


def shapley_values(mob, p, T):
    """
    Compute Shapley values from Möbius coefficients:
        phi_i(t) = sum_{S: i in S} m(S)(t) / |S|

    Returns dict mapping feature index -> ndarray of shape (T,).
    """
    shap = {i: np.zeros(T) for i in range(p)}
    for S, m in mob.items():
        if len(S) == 0: continue
        for i in S: shap[i] += m / len(S)
    return shap


# ===========================================================================
# 6.  Kernels
# ===========================================================================

def kernel_identity(T):
    """Identity kernel: K(t,s) = delta(t-s). No temporal coupling."""
    return np.eye(T)


def kernel_correlation(Y_raw):
    """
    Empirical output correlation kernel derived from the observed demand data.

        K(t,s) = Cov(Y(t), Y(s)) / sqrt(Var(Y(t)) * Var(Y(s)))

    Ties together time points that co-vary across days in the observed data.
    Data-adaptive and bandwidth-free; different datasets yield different
    kernel shapes reflecting their specific temporal dependence structures.

    IHEPC: block-diagonal (local temporal correlation within morning/evening windows)
    NESO:  broader daily pattern (high correlation between demand peaks)

    Parameters
    ----------
    Y_raw : ndarray (n_days, T) — raw (unadjusted) demand trajectories

    Returns
    -------
    K : ndarray (T, T) — correlation matrix, symmetric, unit diagonal
    """
    C   = np.cov(Y_raw.T)
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    return np.clip(C / np.outer(std, std), -1.0, 1.0)


def apply_kernel(effect, K, dt=1.0):
    """
    Apply row-normalised kernel K to an effect curve:
        (Ke)(t) = [K(t,:) @ e] / [sum_s K(t,s) * dt]

    Row normalisation ensures attribution magnitudes are comparable across
    kernels with different total mass.
    """
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return (K/rs) @ effect * dt


# ===========================================================================
# 7.  Pure / partial / full effect helpers
# ===========================================================================

def _pure(mob, p, T):
    """
    Extract first-order (singleton) Möbius coefficients as pure effects.
    m_{i}(t) = effect attributable exclusively to feature i.
    """
    return {i: mob.get((i,), np.zeros(T)).copy() for i in range(p)}


def _full(mob, p, T):
    """
    Compute full effects by summing all Möbius terms containing feature i:
        full_i(t) = sum_{S: i in S} m_S(t)
    Includes all interaction terms involving i (PFI-style attribution).
    """
    f = {i: np.zeros(T) for i in range(p)}
    for S, m in mob.items():
        if len(S) == 0: continue
        for i in S: f[i] += m
    return f


# ===========================================================================
# 8.  Data loading
# ===========================================================================

def load_ihepc():
    """
    Load UCI IHEPC hourly electricity consumption data.

    Tries to load from a local parquet cache first. If not found, downloads
    from the UCI ML Repository via the `ucimlrepo` package.

    Computes day-level feature matrix and diurnal-mean-adjusted target matrix.

    Returns
    -------
    Dataset dict with keys: tag, X_np, Y_raw, Y_adj, diurnal, dates,
    features, T, t_grid, tlabels, sample, ylabel, morning, evening.
    """
    if os.path.isfile(IHEPC_DATA_FILE):
        print('  [IHEPC] Loading parquet cache ...')
        df = pd.read_parquet(IHEPC_DATA_FILE)
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['datetime']).dt.date.astype(str)
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    else:
        import importlib
        if importlib.util.find_spec('ucimlrepo') is None:
            raise RuntimeError('pip install ucimlrepo  (required to download IHEPC data)')
        from ucimlrepo import fetch_ucirepo
        print('  [IHEPC] Downloading from UCI ML Repository ...')
        ds = fetch_ucirepo(id=235)
        df = ds.data.features.copy()
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(
                df['Date']+' '+df['Time'], dayfirst=True, errors='coerce')
            df = df.drop(columns=['Date', 'Time'])
        else:
            df = df.reset_index()
            df.columns = ['datetime'] + list(df.columns[1:])
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        for col in [c for c in df.columns if c not in {'datetime','date','hour'}]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Global_active_power'])
        df['date'] = df['datetime'].dt.date.astype(str)
        df['hour'] = df['datetime'].dt.hour
        _require_dir(IHEPC_DATA_DIR)
        df.to_parquet(IHEPC_DATA_FILE, index=False)

    T = IHEPC_T
    hourly = (df.groupby(['date','hour'])['Global_active_power'].mean()
              .unstack('hour').reindex(columns=range(T)))
    hourly  = hourly[hourly.notna().sum(axis=1) == T]
    Y_raw   = hourly.values.astype(float)
    dates   = hourly.index.tolist()
    diurnal = Y_raw.mean(axis=0)
    Y_adj   = Y_raw - diurnal[None, :]

    records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str)
        m, dow = dt_obj.month, dt_obj.dayofweek
        lmean = (float(Y_raw.mean()) if i == 0
                 else float(Y_raw[i-1].mean()))
        lmorn = (float(Y_raw[:, IHEPC_MORNING[0]:IHEPC_MORNING[1]].mean()) if i == 0
                 else float(Y_raw[i-1, IHEPC_MORNING[0]:IHEPC_MORNING[1]].mean()))
        records.append({'day_of_week': float(dow), 'is_weekend': float(dow >= 5),
                        'month': float(m), 'season': float(_month_to_season(m)),
                        'lag_daily_mean': lmean, 'lag_morning': lmorn})
    X_day = pd.DataFrame(records, index=dates)
    print('  [IHEPC] {} days, mean={:.3f} kW'.format(len(dates), Y_raw.mean()))
    return {'tag': 'ihepc', 'X_np': X_day.to_numpy().astype(float),
            'Y_raw': Y_raw, 'Y_adj': Y_adj, 'diurnal': diurnal, 'dates': dates,
            'features': IHEPC_FEATURES, 'T': T, 't_grid': IHEPC_TGRID,
            'tlabels': IHEPC_LABELS, 'sample': IHEPC_SAMPLE, 'ylabel': IHEPC_YLABEL,
            'morning': IHEPC_MORNING, 'evening': IHEPC_EVENING}


def load_neso():
    """
    Load NESO GB half-hourly national electricity demand data.

    Reads demanddata_{year}.csv files from the data/ directory.
    Files must be downloaded from the NESO data portal separately.

    Computes day-level feature matrix and diurnal-mean-adjusted target matrix.

    Returns
    -------
    Dataset dict with keys: tag, X_np, Y_raw, Y_adj, diurnal, dates,
    features, T, t_grid, tlabels, sample, ylabel, morning, evening.
    """
    dfs = []
    for yr in NESO_YEARS:
        path = os.path.join(NESO_DATA_DIR, 'demanddata_{}.csv'.format(yr))
        if not os.path.isfile(path):
            raise RuntimeError(
                'Missing NESO data file: {}\n'
                'Please download from the NESO data portal.'.format(path))
        dfs.append(pd.read_csv(path, low_memory=False))
    raw = pd.concat(dfs, ignore_index=True)
    raw.columns = [c.strip().upper() for c in raw.columns]
    date_col   = next(c for c in raw.columns if 'DATE'   in c)
    period_col = next(c for c in raw.columns if 'PERIOD' in c)
    demand_col = 'ND' if 'ND' in raw.columns else 'TSD'   # national demand column
    raw[date_col]   = raw[date_col].astype(str).str.strip()
    raw[period_col] = pd.to_numeric(raw[period_col], errors='coerce')
    raw[demand_col] = pd.to_numeric(raw[demand_col], errors='coerce')
    raw = raw.dropna(subset=[date_col, period_col, demand_col])
    raw = raw[(raw[period_col] >= 1) & (raw[period_col] <= NESO_T)].copy()
    raw['period_idx'] = (raw[period_col] - 1).astype(int)
    pivot = raw.pivot_table(index=date_col, columns='period_idx',
                            values=demand_col, aggfunc='mean')
    pivot   = pivot.reindex(columns=range(NESO_T))
    pivot   = pivot[pivot.notna().sum(axis=1) == NESO_T]
    Y_raw   = pivot.values.astype(float)
    dates   = pivot.index.tolist()
    diurnal = Y_raw.mean(axis=0)
    Y_adj   = Y_raw - diurnal[None, :]

    T = NESO_T
    records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str)
        m, dow = dt_obj.month, dt_obj.dayofweek
        if i == 0:
            lmean = float(Y_raw.mean())
            lmorn = float(Y_raw[:, NESO_MORNING[0]:NESO_MORNING[1]].mean())
            leve  = float(Y_raw[:, NESO_EVENING[0]:NESO_EVENING[1]].mean())
        else:
            lmean = float(Y_raw[i-1].mean())
            lmorn = float(Y_raw[i-1, NESO_MORNING[0]:NESO_MORNING[1]].mean())
            leve  = float(Y_raw[i-1, NESO_EVENING[0]:NESO_EVENING[1]].mean())
        records.append({'day_of_week': float(dow), 'is_weekend': float(dow >= 5),
                        'month': float(m), 'season': float(_month_to_season(m)),
                        'lag_daily_mean': lmean, 'lag_morning': lmorn,
                        'lag_evening': leve})
    X_day = pd.DataFrame(records, index=dates)
    print('  [NESO] {} days, mean={:.0f} MW'.format(len(dates), Y_raw.mean()))
    return {'tag': 'neso', 'X_np': X_day.to_numpy().astype(float),
            'Y_raw': Y_raw, 'Y_adj': Y_adj, 'diurnal': diurnal, 'dates': dates,
            'features': NESO_FEATURES, 'T': T, 't_grid': NESO_TGRID,
            'tlabels': NESO_LABELS, 'sample': NESO_SAMPLE, 'ylabel': NESO_YLABEL,
            'morning': NESO_MORNING, 'evening': NESO_EVENING}


# ===========================================================================
# 9.  Game computation with caching
# ===========================================================================

def compute_global_game(ds, game_type, n_outer=GLOBAL_N_OUTER,
                        n_inner=GLOBAL_N_INNER, seed=RNG_SEED,
                        force_recompute=False):
    """
    Compute (or load from cache) a global sensitivity or risk game.

    Uses GlobalFunctionalGame with a double-MC estimator. Results are saved
    to a single cache file per (dataset, game_type).

    Returns
    -------
    (avg_shap, avg_pure, avg_full, avg_pairs, avg_mob)
    Note: 'avg_*' naming kept for plot-code compatibility; these are
    not averages over instances but the direct global-game effects.
    avg_pairs : dict (i,j) -> ndarray(T) — pairwise Möbius coefficients
    avg_mob   : full Möbius dictionary
    """
    _require_dir(GAME_CACHE_DIR)
    cache_path = _cache_path_global(ds['tag'], game_type)
    n_players  = len(ds['features'])

    if os.path.isfile(cache_path) and not force_recompute:
        print('    [{} {} global] loading from cache ...'.format(
            ds['tag'], game_type))
        _, pure, partial, full, mob = _load_cache(cache_path, n_players)
    else:
        print('    [{} {} global] computing double-MC '
              'n_outer={} n_inner={} ...'.format(
                  ds['tag'], game_type, n_outer, n_inner))
        T         = ds['T']
        features  = ds['features']
        Y_for_risk = ds['Y_adj'] if game_type == 'risk' else None
        game = GlobalFunctionalGame(
            predict_fn=ds['model'].predict, X_bg=ds['X_np'], Y_bg=Y_for_risk,
            T=T, features=features, game_type=game_type,
            n_outer=n_outer, n_inner=n_inner, random_seed=seed)
        game.precompute()
        mob     = moebius_transform(game)
        partial = shapley_values(mob, n_players, T)
        pure    = _pure(mob, n_players, T)
        full    = _full(mob, n_players, T)
        x_dummy = np.zeros(n_players)
        _save_cache(cache_path, x_dummy, pure, partial, full, mob, n_players)

    # Build pairwise dict for convenience in network and interaction plots
    pairs = {(i, j): mob.get((i, j), np.zeros(ds['T']))
             for i in range(n_players) for j in range(i+1, n_players)}
    return partial, pure, full, pairs, mob


def compute_local_game(ds, x_profile, label, force_recompute=False):
    """
    Compute or load the local prediction game for a named profile.

    Returns
    -------
    mob, shap (partial), pure, full — all dicts over features
    """
    _require_dir(GAME_CACHE_DIR)
    features  = ds['features']
    T         = ds['T']
    n_players = len(features)
    cache_path = _cache_path_local(ds['tag'], label + '_prediction')
    if os.path.isfile(cache_path) and not force_recompute:
        print('  Loading local prediction cache: {} {}'.format(ds['tag'], label))
        _, pure, partial, full, mob = _load_cache(cache_path, n_players)
        return mob, partial, pure, full
    print('  Computing local prediction game: {} {}'.format(ds['tag'], label))
    game = LocalPredictionGame(
        predict_fn=ds['model'].predict, X_bg=ds['X_np'],
        x_exp=x_profile, T=T, features=features,
        sample_size=ds['sample']['prediction'], random_seed=RNG_SEED)
    game.precompute()
    mob     = moebius_transform(game)
    partial = shapley_values(mob, n_players, T)
    pure    = _pure(mob, n_players, T)
    full    = _full(mob, n_players, T)
    _save_cache(cache_path, x_profile, pure, partial, full, mob, n_players)
    return mob, partial, pure, full


def load_per_instance_local_pred(ds, n_instances, seed):
    """
    Compute or load per-instance local prediction effects for n_instances
    randomly selected explicands.

    The average across instances yields the 'global SHAP = mean local SHAP'
    object, which is valid for prediction because the prediction game's
    characteristic function is linear in F.

    Returns
    -------
    results : list of dicts with keys 'x', 'pure', 'partial', 'full'
    X_bg[idxs] : ndarray of explicand feature vectors
    """
    _require_dir(GAME_CACHE_DIR)
    X_bg      = ds['X_np']
    features  = ds['features']
    T         = ds['T']
    n_players = len(features)
    rng  = np.random.default_rng(seed)
    idxs = rng.choice(len(X_bg), size=n_instances, replace=False)
    results = []
    for k, idx in enumerate(idxs):
        cache  = _cache_path_local_pred_inst(ds['tag'], k)
        x_inst = X_bg[idx]
        if os.path.isfile(cache):
            _, pure, partial, full, _ = _load_cache(cache, n_players)
        else:
            game = LocalPredictionGame(
                predict_fn=ds['model'].predict, X_bg=X_bg,
                x_exp=x_inst, T=T, features=features,
                sample_size=ds['sample']['prediction'],
                random_seed=seed+k)
            game.precompute()
            mob     = moebius_transform(game)
            partial = shapley_values(mob, n_players, T)
            pure    = _pure(mob, n_players, T)
            full    = _full(mob, n_players, T)
            _save_cache(cache, x_inst, pure, partial, full, mob, n_players)
        results.append({'x': x_inst, 'pure': pure,
                        'partial': partial, 'full': full})
        print('    [{} per-instance local pred] {}/{}'.format(
            ds['tag'], k+1, n_instances))
    return results, X_bg[idxs]


def average_local_pred(per_inst, n_players, T):
    """
    Average per-instance local prediction effects over all instances.
    Returns (avg_shap, avg_pure, avg_full).
    """
    sum_pure = {i: np.zeros(T) for i in range(n_players)}
    sum_part = {i: np.zeros(T) for i in range(n_players)}
    sum_full = {i: np.zeros(T) for i in range(n_players)}
    n = len(per_inst)
    for r in per_inst:
        for i in range(n_players):
            sum_pure[i] += r['pure'][i]
            sum_part[i] += r['partial'][i]
            sum_full[i] += r['full'][i]
    return ({i: sum_part[i]/n for i in range(n_players)},
            {i: sum_pure[i]/n for i in range(n_players)},
            {i: sum_full[i]/n for i in range(n_players)})


def _cache_path_pdp(ds_tag, instance_k):
    return os.path.join(
        GAME_CACHE_DIR,
        'pdp_{}_inst{:04d}_{}.npz'.format(ds_tag, instance_k, CACHE_VERSION_LOCAL))


def load_per_instance_effects_pdp(ds, seed):
    """
    Compute or load per-instance local prediction effects for a stratified
    sample of PDP_N_INSTANCES explicands.

    Stratification: one instance per calendar month (12 months), plus a
    random fill to reach PDP_N_INSTANCES total. Ensures seasonal coverage.

    Returns
    -------
    results      : list of dicts with keys 'x', 'pure', 'partial', 'full'
    X_bg[idxs]  : ndarray of explicand feature vectors
    """
    _require_dir(GAME_CACHE_DIR)
    X_bg      = ds['X_np']
    features  = ds['features']
    T         = ds['T']
    n_players = len(features)
    month_col = features.index('month')

    # One instance per month for seasonal stratification
    selected = []
    for m in range(1, 13):
        candidates = np.where(X_bg[:, month_col] == m)[0]
        if len(candidates) > 0:
            rng_m = np.random.default_rng(seed + m)
            selected.append(int(rng_m.choice(candidates)))

    rng  = np.random.default_rng(seed)
    pool = np.setdiff1d(np.arange(len(X_bg)), selected)
    rng.shuffle(pool)
    n_extra = PDP_N_INSTANCES - len(selected)
    if n_extra > 0:
        selected = selected + pool[:n_extra].tolist()
    idxs = np.array(selected[:PDP_N_INSTANCES])

    results = []
    for k, idx in enumerate(idxs):
        cache  = _cache_path_pdp(ds['tag'], k)
        x_inst = X_bg[idx]
        if os.path.isfile(cache):
            _, pure, partial, full, _ = _load_cache(cache, n_players)
        else:
            game = LocalPredictionGame(
                predict_fn=ds['model'].predict, X_bg=X_bg,
                x_exp=x_inst, T=T, features=features,
                sample_size=ds['sample']['prediction'],
                random_seed=seed+k)
            game.precompute()
            mob     = moebius_transform(game)
            partial = shapley_values(mob, n_players, T)
            pure    = _pure(mob, n_players, T)
            full    = _full(mob, n_players, T)
            _save_cache(cache, x_inst, pure, partial, full, mob, n_players)
        results.append({'x': x_inst, 'pure': pure,
                        'partial': partial, 'full': full})
        print('    [{} pdp pred] {}/{}'.format(ds['tag'], k+1, PDP_N_INSTANCES))
    return results, X_bg[idxs]


# ===========================================================================
# 10.  Plotting helpers
# ===========================================================================

def _xticks(ax, ds, sparse=False, fs=None):
    """Set x-axis tick labels to time-of-day strings."""
    tick_fs  = (fs.tick if fs else FS_TICK)
    T, tlabels = ds['T'], ds['tlabels']
    step = max(1, T//8) * (2 if sparse else 1)
    idxs = list(range(0, T, step))
    ax.set_xticks(idxs)
    ax.set_xticklabels([tlabels[i] for i in idxs],
                       rotation=45, ha='right', fontsize=tick_fs)
    ax.set_xlim(-0.5, T-0.5)


def _shade(ax, ds):
    """Add light background shading for morning and evening activity windows."""
    ax.axvspan(*ds['morning'], alpha=SHADE_ALPHA,
               color=SHADE_AM_COLOR, zorder=10, lw=0)
    ax.axvspan(*ds['evening'], alpha=SHADE_ALPHA,
               color=SHADE_PM_COLOR, zorder=10, lw=0)


def savefig(fig, name):
    """Save figure as PDF to BASE_PLOT_DIR and close it."""
    path = os.path.join(BASE_PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print('  Saved: {}'.format(path))
    plt.close(fig)


def _align_row(axes_list):
    """Set a common y-axis range across a list of axes."""
    ymin = min(ax.get_ylim()[0] for ax in axes_list)
    ymax = max(ax.get_ylim()[1] for ax in axes_list)
    for ax in axes_list:
        ax.set_ylim(ymin, ymax)


def _draw_bar(ax, effect_dicts, K, features, fs, legend_bbox=None, x_fmt=None):
    """
    Draw a horizontal bar chart of time-aggregated attribution magnitudes.

    Features ordered by descending partial (Shapley) importance.
    Pure / partial / full shown as three bars per feature using light /
    medium / dark shading of the feature color.
    """
    p    = len(features)
    imps = {et: {i: float(np.sum(np.abs(apply_kernel(effect_dicts[et][i], K))))
                 for i in range(p)}
            for et in _EFFECT_TYPES}
    order   = sorted(range(p), key=lambda i: imps['partial'][i], reverse=True)
    y_pos   = np.arange(len(order))
    bar_h   = 0.25
    offsets = {'pure': -bar_h, 'partial': 0.0, 'full': bar_h}
    shade_factors = {'pure': 0.55, 'partial': 0.0, 'full': -0.40}
    for et in _EFFECT_TYPES:
        sf         = shade_factors[et]
        bar_colors = [_shade_color(FEAT_COLORS[features[i]], sf) for i in order]
        ax.barh(y_pos+offsets[et],
                [imps[et][i] for i in order], height=bar_h,
                color=bar_colors, alpha=1.0, label=et, edgecolor='none')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([features[i] for i in order], fontsize=fs.tick)
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=fs.axis)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=fs.tick)
    ax.set_title('Time-aggregated', fontsize=fs.title, fontweight='bold')
    if x_fmt is not None:
        ax.xaxis.set_major_formatter(x_fmt)
    leg_handles = [
        Patch(facecolor=_shade_color('#888888',  0.55), edgecolor='none',
              label='pure (light)'),
        Patch(facecolor=_shade_color('#888888',  0.0),  edgecolor='none',
              label='partial (medium)'),
        Patch(facecolor=_shade_color('#888888', -0.40), edgecolor='none',
              label='full (dark)'),
    ]
    ax.legend(handles=leg_handles, fontsize=fs.legend,
              loc='upper left', bbox_to_anchor=(1.02, 1.0),
              bbox_transform=ax.transAxes,
              borderaxespad=0., framealpha=0.9)


def _add_bottom_legends_energy(fig, features, top_feats, fs,
                                feat_bbox=(0.04, 0.025),
                                kern_bbox=(0.04, 0.0),
                                kern_labels=('Identity kernel',
                                             'Correlation kernel')):
    """Add feature color legend and kernel linestyle legend at figure bottom."""
    feat_handles = [
        Line2D([0],[0], color=FEAT_COLORS[features[fi]], lw=1.8, ls='-',
               label=features[fi])
        for fi in top_feats]
    kern_handles = [
        Line2D([0],[0], color='#555', lw=1.8, ls='-',  label=kern_labels[0]),
        Line2D([0],[0], color='#555', lw=1.8, ls='--', label=kern_labels[1]),
    ]
    leg1 = fig.legend(handles=feat_handles, fontsize=fs.legend,
                      loc='lower left', bbox_to_anchor=feat_bbox,
                      framealpha=0.9, ncol=len(feat_handles))
    fig.add_artist(leg1)
    fig.legend(handles=kern_handles, fontsize=fs.legend,
               loc='lower left', bbox_to_anchor=kern_bbox,
               framealpha=0.9, ncol=2)


# ===========================================================================
# 11.  Network diagram helpers
# ===========================================================================

def _network_importances_global(avg_shap, avg_pure, avg_full, avg_pairs,
                                 p, T, K, effect_type='full', avg_mob=None):
    """
    Compute node importances and edge weights for the network diagram.

    Node importance: integral of |apply_kernel(effect_i, K)| over t.
    Node sign: sign of the integrated kernel-smoothed effect.
    Edge weight: integral of the pairwise Möbius / interaction coefficient.

    For partial (Shapley) edges, uses the Shapley interaction index
    (Grabisch & Roubens 1999):
        I^Sh_{ij} = sum_{S ⊇ {i,j}} m_S(t) / (|S| - 1)

    Returns
    -------
    node_imp  : ndarray (p,) — non-negative importances
    edge_imp  : dict (i,j) -> float — signed edge weights
    node_sign : ndarray (p,) — +1 or -1
    """
    if effect_type == 'pure':
        eff = avg_pure
    elif effect_type == 'partial':
        eff = avg_shap
    else:
        eff = avg_full
    t_grid    = np.arange(T, dtype=float)
    node_imp  = np.array([float(np.sum(np.abs(apply_kernel(eff[i], K))))
                          for i in range(p)])
    node_sign = np.array([np.sign(float(np.trapz(apply_kernel(eff[i], K), t_grid)))
                          for i in range(p)])
    edge_imp  = {}
    for i in range(p):
        for j in range(i+1, p):
            if effect_type == 'pure' or avg_mob is None:
                raw = avg_pairs.get((i, j), np.zeros(T))
            elif effect_type == 'partial':
                raw = np.zeros(T)
                for S, m in avg_mob.items():
                    if i in S and j in S:
                        raw = raw + m / (len(S) - 1)
            else:  # full
                raw = np.zeros(T)
                for S, m in avg_mob.items():
                    if i in S and j in S:
                        raw = raw + m
            val = float(np.trapz(apply_kernel(raw, K), t_grid))
            if abs(val) > 0:
                edge_imp[(i, j)] = val
    return node_imp, edge_imp, node_sign


def _draw_network(ax, features, node_imp, edge_imp, node_sign,
                  title, fs_title=None, fs_label=None):
    """
    Draw a circular network diagram where:
        - Node size encodes attribution magnitude
        - Node color encodes attribution sign (green=positive, red=negative)
        - Edge width encodes interaction magnitude
        - Edge color encodes interaction sign (synergistic=green, redundant=red)
    """
    import math
    fs_t = fs_title if fs_title is not None else FS_TITLE
    p     = len(features)
    angle = [math.pi/2 - 2*math.pi*i/p for i in range(p)]
    pos   = {i: (math.cos(a), math.sin(a)) for i, a in enumerate(angle)}
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, fontsize=fs_t, fontweight='bold', pad=4)
    max_imp  = float(node_imp.max()) if node_imp.max() > 0 else 1.0
    node_r   = {i: 0.13 + 0.20*(node_imp[i]/max_imp) for i in range(p)}
    max_edge = max((abs(v) for v in edge_imp.values()), default=1.0)
    max_edge = max(max_edge, 1e-12)
    for (i, j), val in edge_imp.items():
        xi, yi = pos[i]; xj, yj = pos[j]
        lw   = 0.4 + 6.5*abs(val)/max_edge
        col  = _EDGE_SYN if val > 0 else _EDGE_RED
        alph = 0.30 + 0.60*abs(val)/max_edge
        ax.plot([xi, xj], [yi, yj], color=col, lw=lw, alpha=alph,
                solid_capstyle='round', zorder=1)
    import matplotlib.patheffects as path_effects
    for i in range(p):
        x, y = pos[i]; r = node_r[i]
        fc = _NODE_POS if node_sign[i] >= 0 else _NODE_NEG
        ax.add_patch(plt.Circle((x, y), r, color=fc, ec='white',
                                linewidth=1.2, zorder=2, alpha=0.92))
        abbr    = FEAT_ABBR.get(features[i], features[i][:3])
        node_fs = fs_label if fs_label is not None else max(8.0, r*30)
        txt = ax.text(x, y, abbr, ha='center', va='center',
                      fontsize=node_fs, fontweight='bold',
                      color='#1a1a1a', zorder=4)
        txt.set_path_effects([
            path_effects.withStroke(linewidth=2.2, foreground='white')])
    pad = 0.36
    ax.set_xlim(-1.0-pad, 1.0+pad); ax.set_ylim(-1.0-pad, 1.0+pad)


# ===========================================================================
# 12.  Figure 0 — Main body: heatmap + network + sensitivity panel
# ===========================================================================

def fig0_main_body(ds_ih, ds_ne, K_ih, K_ne, global_sens_ih, global_sens_ne):
    """
    Generate fig0_main_body.pdf.

    Single-row summary with three panels per dataset:
        Left:   Correlation kernel heatmap K(t,s)
        Middle: Sensitivity network (full effects, correlation kernel)
        Right:  Total Sobol (full sensitivity) curves under identity vs
                correlation kernel, for the top-2 features

    The two datasets are separated by colored background boxes.
    The heatmap contrast between IHEPC (block-diagonal) and NESO (broader
    daily pattern) visually motivates the different attribution profiles.
    """
    FS_SUP=20; FS_T=16; FS_AX=15; FS_TK=13; FS_LEG=13; FS_NODE=12
    ID_ALPHA=0.40; ID_LW=1.6; MX_LW=2.2

    fig = plt.figure(figsize=(28, 5.8))
    gs  = GridSpec(1, 7, figure=fig,
                   width_ratios=[1.25, 1.2, 1.6, 0.18, 1.25, 1.2, 1.6],
                   wspace=0.28, left=0.04, right=0.98,
                   top=0.78, bottom=0.20)

    ax_ih_heat    = fig.add_subplot(gs[0])
    ax_ih_net     = fig.add_subplot(gs[1])
    ax_ih_gap     = fig.add_subplot(gs[2])
    ax_gap_spacer = fig.add_subplot(gs[3]); ax_gap_spacer.set_visible(False)
    ax_ne_heat    = fig.add_subplot(gs[4])
    ax_ne_net     = fig.add_subplot(gs[5])
    ax_ne_gap     = fig.add_subplot(gs[6])

    fig.suptitle(
        'Functional explanation framework: energy demand — '
        'correlation structure drives explanation shape\n'
        'UCI IHEPC (single household, kW)  vs  NESO GB Demand '
        '(national grid, MW)',
        fontsize=FS_SUP, fontweight='bold', y=0.98)

    def _heatmap(ax, ds, K, tag):
        T, tl = ds['T'], ds['tlabels']
        step  = max(1, T//6)
        ticks = list(range(0, T, step))
        im = ax.imshow(K, aspect='equal', origin='upper',
                       cmap='RdBu_r', vmin=-1.0, vmax=1.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels([tl[i] for i in ticks],
                           rotation=45, ha='right', fontsize=FS_TK)
        ax.set_yticks(ticks)
        ax.set_yticklabels([tl[i] for i in ticks], fontsize=FS_TK)
        ax.set_title(DS_LABEL[tag].replace('\n',' ') + '\ncorrelation kernel $K$',
                     fontsize=FS_AX, fontweight='bold', color=DS_COLOR[tag])
        am = (ds['morning'][0]+ds['morning'][1])//2
        ax.axhline(am, color='white', lw=0.8, ls='--', alpha=0.6)
        ax.axvline(am, color='white', lw=0.8, ls='--', alpha=0.6)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                          ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=FS_TK)

    _heatmap(ax_ih_heat, ds_ih, K_ih, 'ihepc')
    _heatmap(ax_ne_heat, ds_ne, K_ne, 'neso')

    net_handles = [Patch(facecolor=_NODE_POS, edgecolor='none', label='Positive'),
                   Patch(facecolor=_NODE_NEG, edgecolor='none', label='Negative')]
    for ax, ds, tag in [
        (ax_ih_net, ds_ih, 'ihepc'),
        (ax_ne_net, ds_ne, 'neso'),
    ]:
        features = ds['features']; p, T = len(features), ds['T']
        K        = K_ih if tag == 'ihepc' else K_ne
        g_sens   = global_sens_ih if tag == 'ihepc' else global_sens_ne
        avg_shap, avg_pure, avg_full, avg_pairs, avg_mob = g_sens
        ni, ei, ns = _network_importances_global(
            avg_shap, avg_pure, avg_full, avg_pairs, p, T, K, 'full',
            avg_mob=avg_mob)
        _draw_network(ax, features, ni, ei, ns,
                      '{} sensitivity\nfull (corr.)'.format(tag.upper()),
                      fs_title=FS_T, fs_label=FS_NODE)
        ax.legend(handles=net_handles, loc='lower center', ncol=2,
                  fontsize=FS_LEG, framealpha=0.88,
                  bbox_to_anchor=(0.5, -0.22), bbox_transform=ax.transAxes,
                  borderpad=0.45, handlelength=1.4)

    def _sens_full_panel(ax, ds, global_sens, K_corr, K_id, tag,
                          force_features=None):
        """
        Plot total Sobol (full sensitivity) curves under identity and
        correlation kernels for the top features. Faded lines show identity;
        solid lines show correlation kernel.

        force_features overrides the automatic top-2 selection (used to
        highlight specific features for NESO).
        """
        features = ds['features']; p, T = len(features), ds['T']
        t_grid   = ds['t_grid']
        avg_full = global_sens[2]
        if force_features is None:
            imps = {i: float(np.sum(np.abs(apply_kernel(avg_full[i], K_corr))))
                    for i in range(p)}
            fis  = sorted(imps, key=imps.get, reverse=True)[:2]
        else:
            fis = [features.index(f) for f in force_features]

        for fi in fis:
            col      = FEAT_COLORS[features[fi]]
            eff_id   = apply_kernel(avg_full[fi], K_id)
            eff_corr = apply_kernel(avg_full[fi], K_corr)
            ls = '-' if fi == fis[0] else '--'
            ax.plot(t_grid, eff_id, color=col, lw=ID_LW, ls=ls,
                    alpha=ID_ALPHA, zorder=2)
            ax.plot(t_grid, eff_corr, color=col, lw=MX_LW, ls=ls,
                    label=features[fi] + ' (corr.)', zorder=3)

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _xticks(ax, ds, sparse=True)
        ax.tick_params(labelsize=FS_TK)
        ax.set_xlabel('Time', fontsize=FS_AX)
        if tag == 'neso':
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(
                    lambda v, _: '{:.2f}'.format(v / 1e7)))
            ylabel = r'Var$[F(t)]$ (MW$^2$, $\times 10^7$)'
        else:
            ylabel = ds['ylabel']['sensitivity']
        ax.set_ylabel(ylabel, fontsize=FS_AX)
        feat_str = ', '.join(features[fi] for fi in fis)
        ax.set_title(
            'Total Sobol — corr. kernel\n'
            '{} — {}'.format(DS_LABEL[tag].split('\n')[0], feat_str),
            fontsize=FS_T-1, fontweight='bold', color=DS_COLOR[tag])
        corr_handles = [
            Line2D([0],[0], color=FEAT_COLORS[features[fi]],
                   lw=MX_LW, ls='-' if fi == fis[0] else '--',
                   label=features[fi] + ' (corr.)')
            for fi in fis]
        extra_id = Line2D([0],[0], color='gray', lw=ID_LW, ls='-',
                          alpha=ID_ALPHA, label='identity (faded)')
        # Shift IHEPC legend leftward to avoid the right panel edge
        leg_x = 0.40 if tag == 'ihepc' else 0.5
        ax.legend(handles=corr_handles + [extra_id],
                  fontsize=10, loc='upper center',
                  bbox_to_anchor=(leg_x, -0.25), ncol=len(fis)+1,
                  framealpha=0.85)
        _shade(ax, ds)

    K_id_ih = kernel_identity(ds_ih['T'])
    K_id_ne = kernel_identity(ds_ne['T'])
    _sens_full_panel(ax_ih_gap, ds_ih, global_sens_ih, K_ih, K_id_ih,
                     'ihepc', force_features=None)
    _sens_full_panel(ax_ne_gap, ds_ne, global_sens_ne, K_ne, K_id_ne,
                     'neso', force_features=['month', 'season'])

    # Shift each network panel slightly leftward to close gap with heatmap
    fig.canvas.draw()
    SHIFT_FRAC = 0.45
    for ax_heat, ax_net in [(ax_ih_heat, ax_ih_net),
                            (ax_ne_heat, ax_ne_net)]:
        bb_h = ax_heat.get_position()
        bb_n = ax_net.get_position()
        gap  = bb_n.x0 - bb_h.x1
        if gap <= 0: continue
        delta  = SHIFT_FRAC * gap
        new_x0 = bb_n.x0 - delta
        new_w  = bb_n.width + delta
        ax_net.set_position([new_x0, bb_n.y0, new_w, bb_n.height])

    # Colored background boxes separating the two dataset columns
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ne_xmin  = min(ax.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()).x0
        for ax in [ax_ne_heat, ax_ne_net, ax_ne_gap])
    x_split_ne = ne_xmin - 0.060

    def _bg_box_clipped(axes_list, color, xmin_clip, xmax_clip, pad=0.016,
                        extra_bottom=0.18, extra_top=0.07,
                        extra_left=0.0, extra_right=0.0):
        """Draw a rounded rectangle behind a group of axes."""
        xmins, ymins, xmaxs, ymaxs = [], [], [], []
        for ax in axes_list:
            bb     = ax.get_window_extent(renderer=renderer)
            bb_fig = bb.transformed(fig.transFigure.inverted())
            xmins.append(bb_fig.x0); ymins.append(bb_fig.y0)
            xmaxs.append(bb_fig.x1); ymaxs.append(bb_fig.y1)
        x0 = max(min(xmins) - pad - extra_left, xmin_clip)
        y0 = min(ymins) - pad - extra_bottom
        x1 = min(max(xmaxs) + pad + extra_right, xmax_clip)
        y1 = max(ymaxs) + pad + extra_top
        rect = FancyBboxPatch((x0, y0), x1-x0, y1-y0,
                              boxstyle='round,pad=0.006',
                              linewidth=1.4, edgecolor=color, facecolor=color,
                              alpha=0.10, transform=fig.transFigure,
                              zorder=0, clip_on=False)
        fig.add_artist(rect)

    _bg_box_clipped([ax_ih_heat, ax_ih_net, ax_ih_gap],
                    DS_COLOR['ihepc'], xmin_clip=0.0, xmax_clip=x_split_ne,
                    extra_right=0.02)
    _bg_box_clipped([ax_ne_heat, ax_ne_net, ax_ne_gap],
                    DS_COLOR['neso'], xmin_clip=0.0, xmax_clip=1.02,
                    extra_left=0.025, extra_right=0.015)
    return fig


# ===========================================================================
# 13.  Figure 1 — Global Risk + Sensitivity (4 rows x 4 cols)
# ===========================================================================

def fig1_global_risk_sensitivity(ds, global_effects, fs=None):
    """
    Generate fig1_global_risk_sensitivity_{tag}.pdf.

    Four-row layout:
        Row 0: Risk — Identity kernel
        Row 1: Risk — Correlation kernel
        Row 2: Sensitivity — Identity kernel
        Row 3: Sensitivity — Correlation kernel

    Each row: 3 time-resolved panels (pure / partial / full) + 1 bar panel.
    """
    if fs is None: fs = _fs(0)
    features = ds['features']; p = len(features); T = ds['T']
    tag    = ds['tag']
    K_id   = kernel_identity(T)
    K_corr = ds['K_corr']

    row_specs = [
        ('risk',        K_id,   'identity',    _RISK_LABELS,
         'Risk — Identity kernel'),
        ('risk',        K_corr, 'correlation', _RISK_LABELS,
         'Risk — Correlation kernel'),
        ('sensitivity', K_id,   'identity',    _SENS_LABELS,
         'Sensitivity — Identity kernel'),
        ('sensitivity', K_corr, 'correlation', _SENS_LABELS,
         'Sensitivity — Correlation kernel'),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(20, 4.2*4),
                             gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Global Sensitivity and Risk effects — pure / partial / full\n'
        '(double-MC, n_outer={}, n_inner={}) — {}'.format(
            GLOBAL_N_OUTER, GLOBAL_N_INNER,
            DS_LABEL[tag].replace('\n', '  ')),
        fontsize=fs.suptitle, fontweight='bold')

    top_k = 5; top_feats = None
    for r, (gtype, K, klabel, eff_labels, row_title) in enumerate(row_specs):
        avg_shap, avg_pure, avg_full, _, _ = global_effects[gtype]
        effect_dicts = {'pure': avg_pure, 'partial': avg_shap, 'full': avg_full}
        imps = {i: float(np.sum(np.abs(apply_kernel(avg_shap[i], K))))
                for i in range(p)}
        top = sorted(imps, key=imps.get, reverse=True)[:top_k]
        if top_feats is None: top_feats = top

        for c, etype in enumerate(_EFFECT_TYPES):
            ax  = axes[r, c]
            eff = effect_dicts[etype]
            for fi in top:
                ls = '--' if klabel == 'correlation' else '-'
                ax.plot(ds['t_grid'], apply_kernel(eff[fi], K),
                        color=FEAT_COLORS[features[fi]], lw=1.8, ls=ls)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _xticks(ax, ds, sparse=True, fs=fs)
            ax.tick_params(labelsize=fs.tick)
            ax.set_xlabel('Time', fontsize=fs.axis)
            ax.set_title(eff_labels[etype], fontsize=fs.title, fontweight='bold')
            if c == 0:
                ax.set_ylabel(ds['ylabel'][gtype], fontsize=fs.axis)
                ax.text(-0.30, 0.5, row_title, transform=ax.transAxes,
                        fontsize=fs.axis-1, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')
            _shade(ax, ds)

        _draw_bar(axes[r, 3], effect_dicts, K, features, fs)
        _align_row([axes[r, c] for c in range(3)])

    feat_handles = [
        Line2D([0],[0], color=FEAT_COLORS[features[fi]], lw=1.8, ls='-',
               label=features[fi])
        for fi in top_feats]
    kern_handles = [
        Line2D([0],[0], color='#555', lw=1.8, ls='-',  label='Identity kernel'),
        Line2D([0],[0], color='#555', lw=1.8, ls='--', label='Correlation kernel'),
    ]
    leg1 = fig.legend(handles=feat_handles, fontsize=fs.legend,
                      loc='lower left', bbox_to_anchor=(0.04, 0.025),
                      framealpha=0.9, ncol=len(feat_handles))
    fig.add_artist(leg1)
    fig.legend(handles=kern_handles, fontsize=fs.legend,
               loc='lower left', bbox_to_anchor=(0.04, 0.0),
               framealpha=0.9, ncol=2)
    plt.tight_layout(rect=[0.04, 0.07, 1, 0.94])
    fig.subplots_adjust(hspace=0.65, top=0.92)
    return fig


# ===========================================================================
# 14.  Figure 2 — Local Prediction (2 rows x 4 cols)
# ===========================================================================

def fig2_local_prediction(ds, local_pred, fs=None):
    """
    Generate fig2_local_prediction_{tag}.pdf.

    Two-row layout (identity kernel / correlation kernel) showing pure,
    partial, and full local prediction effects for the selected profile,
    plus time-aggregated bar charts.
    """
    if fs is None: fs = _fs(0)
    features = ds['features']; p = len(features); T = ds['T']
    tag    = ds['tag']
    K_id   = kernel_identity(T)
    K_corr = ds['K_corr']

    mob, shap, pure_eff, full_eff = local_pred

    row_specs = [
        (K_id,   'identity',    'Identity kernel'),
        (K_corr, 'correlation', 'Correlation kernel'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 4.2*2),
                             gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Local Prediction effects — pure / partial / full\n{}'.format(
            DS_LABEL[tag].replace('\n', '  ')),
        fontsize=fs.suptitle, fontweight='bold')

    top_k = 5; top_feats = None
    for r, (K, klabel, row_title) in enumerate(row_specs):
        effect_dicts = {'pure': pure_eff, 'partial': shap, 'full': full_eff}
        imps = {i: float(np.sum(np.abs(apply_kernel(shap[i], K))))
                for i in range(p)}
        top = sorted(imps, key=imps.get, reverse=True)[:top_k]
        if top_feats is None: top_feats = top

        for c, etype in enumerate(_EFFECT_TYPES):
            ax  = axes[r, c]
            eff = effect_dicts[etype]
            for fi in top:
                ls = '--' if klabel == 'correlation' else '-'
                ax.plot(ds['t_grid'], apply_kernel(eff[fi], K),
                        color=FEAT_COLORS[features[fi]], lw=1.8, ls=ls)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _xticks(ax, ds, sparse=True, fs=fs)
            ax.tick_params(labelsize=fs.tick)
            ax.set_xlabel('Time', fontsize=fs.axis)
            ax.set_title(_LOCAL_PRED_LABELS[etype],
                         fontsize=fs.title, fontweight='bold')
            if c == 0:
                ax.set_ylabel(ds['ylabel']['prediction'], fontsize=fs.axis)
                ax.text(-0.30, 0.5, 'Prediction\n{}'.format(row_title),
                        transform=ax.transAxes,
                        fontsize=fs.axis-1, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')
            _shade(ax, ds)

        neso_fmt = (matplotlib.ticker.FuncFormatter(
                       lambda x, _: '{:.0f}k'.format(x/1000))
                   if tag == 'neso' else None)
        _draw_bar(axes[r, 3], effect_dicts, K, features, fs, x_fmt=neso_fmt)
        _align_row([axes[r, c] for c in range(3)])

    _add_bottom_legends_energy(fig, features, top_feats, fs,
                                feat_bbox=(0.04, 0.030),
                                kern_bbox=(0.04, -0.010))
    plt.tight_layout(rect=[0.04, 0.10, 1, 0.91])
    fig.subplots_adjust(hspace=0.65, top=0.88)
    return fig


# ===========================================================================
# 15.  Figure 3 — Global Prediction PDP-style (4 rows x 3 cols)
# ===========================================================================

def _top2_features_avg_local(avg_local_pred, T, p):
    """Select top-2 features by integrated identity-kernel Shapley value."""
    K_id     = kernel_identity(T)
    avg_shap = avg_local_pred[0]
    imps     = {i: float(np.sum(np.abs(apply_kernel(avg_shap[i], K_id))))
                for i in range(p)}
    return sorted(imps, key=imps.get, reverse=True)[:2]


def _pdp_panel_energy(ax, fi, etype, K, X_background, per_instance,
                      features, selected_t_idxs, t_cmap, ds, fs):
    """
    Draw a single PDP panel: effects binned by feature value, with
    time-slice curves (dashed, colored by time) and a time-aggregated
    mean curve (solid black).

    Continuous features use equal-width bins; discrete features use
    their unique values directly as bin centers.
    """
    feat_name = features[fi]
    feat_vals = X_background[:, fi]

    if feat_name in CONTINUOUS_FEATURES:
        fmin, fmax  = feat_vals.min(), feat_vals.max()
        bins        = np.linspace(fmin, fmax, N_FEAT_BINS+1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_idx     = np.clip(np.digitize(feat_vals, bins)-1, 0, N_FEAT_BINS-1)
        n_bins      = N_FEAT_BINS
        is_discrete = False
    else:
        unique_vals = np.sort(np.unique(feat_vals))
        bin_centers = unique_vals
        n_bins      = len(unique_vals)
        bin_idx     = np.array([np.argmin(np.abs(unique_vals - v))
                                for v in feat_vals])
        is_discrete = True

    T = ds['T']
    bin_effects = np.full((n_bins, T), np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0: continue
        effects = np.array([apply_kernel(per_instance[k][etype][fi], K)
                            for k in np.where(mask)[0]])
        bin_effects[b] = effects.mean(axis=0)

    valid  = ~np.isnan(bin_effects[:, 0])
    n_t    = len(selected_t_idxs)
    colors = [t_cmap(i/(n_t-1)) for i in range(n_t)]
    for ti, col in zip(selected_t_idxs, colors):
        ax.plot(bin_centers[valid], bin_effects[valid, ti],
                color=col, lw=1.4, ls='--', alpha=0.85, label=ds['tlabels'][ti])
    # Time-aggregated mean (thick black line)
    pdp = np.nanmean(bin_effects, axis=1)
    ax.plot(bin_centers[valid], pdp[valid],
            color='black', lw=2.5, ls='-', zorder=5, label='time-agg.')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    if valid.any():
        xlo = bin_centers[valid].min()
        xhi = bin_centers[valid].max()
        if is_discrete:
            margin = 0.3
            ax.set_xlim(xlo - margin, xhi + margin)
            ax.set_xticks(bin_centers[valid])
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: '{:.0f}'.format(v)))
            ax.autoscale(False, axis='x')
        else:
            xrange = xhi - xlo
            margin = xrange * 0.03
            ax.set_xlim(xlo - margin, xhi + margin)
            ax.autoscale(False, axis='x')
            ax.xaxis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=6, prune='both'))
    ax.tick_params(labelsize=fs.tick)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha('right'); lbl.set_fontsize(fs.tick)
    ax.set_xlabel(feat_name, fontsize=fs.axis)


def fig3_global_pdp(ds, avg_local_pred, per_instance, X_background, fs=None):
    """
    Generate fig3_global_pdp_{tag}.pdf.

    PDP-style global prediction figure: per-instance local effects are
    binned by the top-2 features' marginal distributions.

    Four rows: top-2 features × 2 kernels (identity / correlation).
    Three columns: pure / partial / full effects.
    """
    if fs is None: fs = _fs(0)
    features = ds['features']; tag = ds['tag']
    p = len(features); T = ds['T']
    K_id   = kernel_identity(T)
    K_corr = ds['K_corr']

    top2 = _top2_features_avg_local(avg_local_pred, T, p)

    selected_t_idxs = [0, T//5, 2*T//5, 3*T//5, T-1]
    t_cmap = cm.get_cmap('plasma', len(selected_t_idxs))

    feat_row_specs = []
    for fi in top2:
        feat_row_specs.append((fi, K_id,   'Identity kernel'))
        feat_row_specs.append((fi, K_corr, 'Correlation kernel'))

    fig, axes = plt.subplots(4, 3, figsize=(18, 3.8*4))
    fig.suptitle(
        'Global Prediction effects — PDP-style — pure / partial / full\n'
        '(per-instance local effects, binned over feature range) — {}'.format(
            DS_LABEL[tag].replace('\n', '  ')),
        fontsize=fs.suptitle, fontweight='bold')

    for r, (fi, K, klabel) in enumerate(feat_row_specs):
        for c, etype in enumerate(_EFFECT_TYPES):
            ax = axes[r, c]
            _pdp_panel_energy(ax, fi, etype, K, X_background, per_instance,
                              features, selected_t_idxs, t_cmap, ds, fs)
            ax.set_title(_GLOBAL_PRED_LABELS[etype],
                         fontsize=fs.title, fontweight='bold')
            if c == 0:
                ax.set_ylabel(ds['ylabel']['prediction'], fontsize=fs.axis)
                ax.text(-0.22, 0.5, '{}\n{}'.format(features[fi], klabel),
                        transform=ax.transAxes,
                        fontsize=fs.axis-1, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')
        _align_row([axes[r, c] for c in range(3)])

    t_handles = [
        Line2D([0],[0], color=t_cmap(i/(len(selected_t_idxs)-1)),
               lw=1.4, ls='--', label=ds['tlabels'][ti])
        for i, ti in enumerate(selected_t_idxs)
    ] + [Line2D([0],[0], color='black', lw=2.5, ls='-', label='time-agg.')]
    fig.legend(handles=t_handles, fontsize=fs.legend,
               loc='lower center', ncol=len(t_handles),
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
    plt.tight_layout(rect=[0.04, 0.06, 1, 0.93])
    fig.subplots_adjust(hspace=0.80, top=0.91)
    return fig


# ===========================================================================
# 16.  Figure 4 — Local Pairwise Interactions (2 rows x 2 cols)
# ===========================================================================

def fig4_interactions(ds, local_pred, fs=None):
    """
    Generate fig4_interactions_{tag}.pdf.

    Shows the top-5 pairwise interaction effects m_{ij}(t) under
    identity and correlation kernels for the local prediction game.
    """
    if fs is None: fs = _fs(0)
    ax_tick_fs  = max(fs.tick-2, 6)
    ax_label_fs = max(fs.axis-1, 7)

    features = ds['features']; p = len(features); T = ds['T']
    tag    = ds['tag']
    K_id   = kernel_identity(T)
    K_corr = ds['K_corr']
    mob    = local_pred[0]

    PAIR_COLORS = ['#e63946', '#2a9d8f', '#8338ec', '#fb8500', '#457b9d']

    # Rank pairs by integrated correlation-kernel interaction magnitude
    pair_imp = {}
    for i in range(p):
        for j in range(i+1, p):
            raw = mob.get((i, j), np.zeros(T))
            pair_imp[(i, j)] = float(np.sum(np.abs(apply_kernel(raw, K_corr))))
    top5 = sorted(pair_imp, key=pair_imp.get, reverse=True)[:5]

    fig, axes = plt.subplots(2, 2, figsize=(12, 4.2*2),
                             gridspec_kw={'width_ratios': [3, 1.8]})
    fig.suptitle(
        'Local pairwise interaction effects — top-5 pairs'
        '\n{}'.format(DS_LABEL[tag].replace('\n', '  ')),
        fontsize=fs.suptitle, fontweight='bold')

    row_specs = [
        (K_id,   'identity',    'Interactions — Identity kernel'),
        (K_corr, 'correlation', 'Interactions — Correlation kernel'),
    ]

    for r, (K, klabel, row_label) in enumerate(row_specs):
        ax = axes[r, 0]
        for pair_idx, (i, j) in enumerate(top5):
            raw = mob.get((i, j), np.zeros(T))
            ls  = '--' if klabel == 'correlation' else '-'
            ax.plot(ds['t_grid'], apply_kernel(raw, K),
                    color=PAIR_COLORS[pair_idx], lw=1.8, ls=ls)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _shade(ax, ds)
        ax.set_xticks(list(range(0, T, max(1, T//8)*2)))
        ax.set_xticklabels([ds['tlabels'][i] for i in range(0, T, max(1, T//8)*2)],
                           rotation=45, ha='right', fontsize=ax_tick_fs)
        ax.set_xlim(-0.5, T-0.5)
        ax.tick_params(labelsize=ax_tick_fs)
        ax.set_xlabel('Time', fontsize=ax_label_fs)
        ax.set_ylabel(ds['ylabel']['prediction'], fontsize=ax_label_fs)
        ax.set_title('Pairwise interaction', fontsize=fs.title, fontweight='bold')
        ax.text(-0.18, 0.5, row_label, transform=ax.transAxes,
                fontsize=ax_label_fs-1, va='center', ha='right',
                rotation=90, color='#333', fontweight='bold')
        _align_row([axes[r, 0]])

        ax_bar = axes[r, 1]
        row_imps = []
        for i, j in top5:
            raw = mob.get((i, j), np.zeros(T))
            imp = float(np.sum(np.abs(apply_kernel(raw, K))))
            row_imps.append((imp, (i, j)))
        row_imps_sorted = sorted(row_imps, key=lambda x: x[0], reverse=True)
        y_pos = np.arange(len(row_imps_sorted))
        ax_bar.barh(y_pos,
                    [imp for imp, _ in row_imps_sorted],
                    color=[PAIR_COLORS[top5.index(ij)] for _, ij in row_imps_sorted],
                    alpha=0.85)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(
            ['{} x {}'.format(features[i], features[j])
             for _, (i, j) in row_imps_sorted],
            fontsize=ax_tick_fs-1)
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=ax_label_fs)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=ax_tick_fs)
        ax_bar.set_title('Time-aggregated', fontsize=fs.title, fontweight='bold')

    pair_handles = [
        Line2D([0],[0], color=PAIR_COLORS[k], lw=1.8,
               label='{} x {}'.format(features[i], features[j]))
        for k, (i, j) in enumerate(top5)]
    kern_handles = [
        Line2D([0],[0], color='#555', lw=1.8, ls='-',  label='Identity kernel'),
        Line2D([0],[0], color='#555', lw=1.8, ls='--', label='Correlation kernel'),
    ]
    leg1 = fig.legend(handles=pair_handles, fontsize=fs.legend,
                      loc='lower left', bbox_to_anchor=(0.04, 0.04),
                      framealpha=0.9, ncol=len(pair_handles))
    fig.add_artist(leg1)
    fig.legend(handles=kern_handles, fontsize=fs.legend,
               loc='lower left', bbox_to_anchor=(0.04, 0.0),
               framealpha=0.9, ncol=2)
    plt.tight_layout(rect=[0.04, 0.10, 1, 0.93])
    fig.subplots_adjust(hspace=0.55, top=0.88)
    return fig


# ===========================================================================
# 17.  Figure 5 — Network plots (2 rows x 3 cols)
# ===========================================================================

def fig5_networks_global(ds_ih, ds_ne, global_ih, global_ne, K_ih, K_ne):
    """
    Generate fig5_networks_global.pdf.

    Two-row, three-column grid:
        Rows:    IHEPC / NESO
        Columns: pure / partial / full sensitivity effects
    under the correlation kernel. Node size encodes total Sobol importance;
    edge width and color encode synergistic / redundant interactions.
    """
    row_specs  = [
        (ds_ih, global_ih, K_ih, 'IHEPC Sensitivity'),
        (ds_ne, global_ne, K_ne, 'NESO Sensitivity'),
    ]
    col_labels = ['Pure', 'Partial', 'Full']
    col_etypes = ['pure', 'partial', 'full']

    fig = plt.figure(figsize=(12.5, 9))
    fig.suptitle(
        'Global sensitivity network plots — correlation kernel\n'
        'IHEPC and NESO',
        fontsize=FS_SUPTITLE, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.12, wspace=0.10,
                           left=0.08, right=0.98,
                           top=0.90, bottom=0.10)

    for r, (ds, g_eff, K, row_label) in enumerate(row_specs):
        features = ds['features']; p = len(features); T = ds['T']
        avg_shap, avg_pure, avg_full, avg_pairs, avg_mob = g_eff['sensitivity']

        for c, etype in enumerate(col_etypes):
            ax = fig.add_subplot(gs[r, c])
            ni, ei, ns = _network_importances_global(
                avg_shap, avg_pure, avg_full, avg_pairs, p, T, K, etype,
                avg_mob=avg_mob)
            _draw_network(ax, features, ni, ei, ns,
                          col_labels[c] if r == 0 else '',
                          fs_title=FS_TITLE+1)
            if c == 0:
                ax.text(-0.03, 0.5, row_label,
                        transform=ax.transAxes,
                        fontsize=FS_AXIS+1, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')

    leg_handles = [
        Patch(facecolor=_NODE_POS, edgecolor='none', label='Positive effect'),
        Patch(facecolor=_NODE_NEG, edgecolor='none', label='Negative effect'),
    ]
    fig.legend(handles=leg_handles, loc='lower center', ncol=2,
               fontsize=FS_LEGEND+1, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.02))
    return fig


# ===========================================================================
# 18.  Sanity checks
# ===========================================================================

def sanity_check_global(global_eff, gtype, ds_tag):
    """
    Print sanity statistics for a global game result.

    Checks:
        v(empty) integral ≈ 0  (both games by construction)
        Shapley efficiency: sum_i phi_i = v(N) - v(empty)
        Count of features with negative integrated effect
        (should be 0 for sensitivity and risk, which are >= 0 by construction)
    """
    avg_shap, avg_pure, avg_full, _, avg_mob = global_eff
    p = len(avg_shap); T = avg_shap[0].shape[0]
    v_empty = avg_mob.get((), np.zeros(T))
    v_N     = sum(avg_mob.values())

    sum_partial = np.zeros(T)
    for i in range(p):
        sum_partial += avg_shap[i]
    eff_target = v_N - v_empty

    n_neg_pure = sum(1 for i in range(p)
                     if float(np.sum(apply_kernel(avg_pure[i], np.eye(T)))) < -1e-6)
    n_neg_part = sum(1 for i in range(p)
                     if float(np.sum(apply_kernel(avg_shap[i], np.eye(T)))) < -1e-6)
    n_neg_full = sum(1 for i in range(p)
                     if float(np.sum(apply_kernel(avg_full[i], np.eye(T)))) < -1e-6)

    print('  [sanity {} {}]'.format(ds_tag, gtype))
    print('    v(empty) integral = {:.4g}  (expect ~0)'.format(
        float(v_empty.sum())))
    print('    v(N) integral     = {:.4g}'.format(float(v_N.sum())))
    print('    sum partial integral  = {:.4g}'.format(float(sum_partial.sum())))
    print('    Shapley efficiency residual = {:.4g}'.format(
        float((sum_partial - eff_target).sum())))
    print('    # features with negative integrated effect  '
          '(pure/partial/full): {}/{}/{} of {}'.format(
              n_neg_pure, n_neg_part, n_neg_full, p))


# ===========================================================================
# 19.  Main entry point
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '='*60)
    print('  Energy Demand Example (IHEPC + NESO) — Combined')
    print('='*60)

    _require_dir(BASE_PLOT_DIR)
    _require_dir(GAME_CACHE_DIR)

    print('\n[1] Loading data ...')
    ds_ih = load_ihepc()
    ds_ne = load_neso()

    print('\n[2] Fitting Random Forest models ...')
    for ds, name in [(ds_ih, 'IHEPC'), (ds_ne, 'NESO')]:
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            ds['X_np'], ds['Y_adj'], test_size=0.2, random_state=RNG_SEED)
        m = RFModel()
        m.fit(X_tr, Y_tr)
        print('  [{}] Test R2: {:.4f}'.format(name, m.evaluate(X_te, Y_te)))
        ds['model'] = m

    print('\n[3] Building empirical correlation kernels ...')
    K_ih = kernel_correlation(ds_ih['Y_raw']); ds_ih['K_corr'] = K_ih
    K_ne = kernel_correlation(ds_ne['Y_raw']); ds_ne['K_corr'] = K_ne

    print('\n[4] Selecting representative day profiles ...')
    X_ih = ds_ih['X_np']; fn_ih = ds_ih['features']

    def find_ih(conds, lbl):
        """Select the median day matching given feature range conditions."""
        mask = np.ones(len(X_ih), dtype=bool)
        for f, (lo, hi) in conds.items():
            ci = fn_ih.index(f)
            mask &= (X_ih[:, ci] >= lo) & (X_ih[:, ci] <= hi)
        hits = X_ih[mask]
        if not len(hits): raise RuntimeError('No match: {}'.format(lbl))
        print('  IHEPC "{}": {} days'.format(lbl, len(hits)))
        return hits[len(hits)//2]

    x_ih1 = find_ih({'is_weekend': (-0.1, 0.1), 'day_of_week': (0.9, 4.1)},
                    'Typical weekday')

    X_ne = ds_ne['X_np']; fn_ne = ds_ne['features']

    def find_ne(conds, lbl):
        mask = np.ones(len(X_ne), dtype=bool)
        for f, (lo, hi) in conds.items():
            ci = fn_ne.index(f)
            mask &= (X_ne[:, ci] >= lo) & (X_ne[:, ci] <= hi)
        hits = X_ne[mask]
        if not len(hits): raise RuntimeError('No match: {}'.format(lbl))
        print('  NESO "{}": {} days'.format(lbl, len(hits)))
        return hits[len(hits)//2]

    x_ne1 = find_ne({'is_weekend': (-0.1, 0.1), 'season': (0.9, 1.1)},
                    'Winter weekday')
    print(dict(zip(fn_ih, x_ih1)))
    print(dict(zip(fn_ne, x_ne1)))

    print('\n[5] Loading / computing local prediction games ...')
    local_ih = compute_local_game(ds_ih, x_ih1, 'Typical_weekday')
    local_ne = compute_local_game(ds_ne, x_ne1, 'Winter_weekday')

    print('\n[6] Loading / computing global sensitivity + risk games ...')
    global_ih = {}; global_ne = {}
    for gtype in GLOBAL_GAME_TYPES:
        print('\n  IHEPC global {} ...'.format(gtype))
        global_ih[gtype] = compute_global_game(ds_ih, gtype)
        sanity_check_global(global_ih[gtype], gtype, 'ihepc')
        print('\n  NESO global {} ...'.format(gtype))
        global_ne[gtype] = compute_global_game(ds_ne, gtype)
        sanity_check_global(global_ne[gtype], gtype, 'neso')

    print('\n[7] Loading / computing per-instance local prediction (PDP) ...')
    per_inst_ih, X_pdp_ih = load_per_instance_effects_pdp(ds_ih, RNG_SEED)
    per_inst_ne, X_pdp_ne = load_per_instance_effects_pdp(ds_ne, RNG_SEED)

    print('\n[8] Building averaged local prediction (feature selection) ...')
    avg_pred_ih = average_local_pred(
        per_inst_ih, len(ds_ih['features']), ds_ih['T'])
    avg_pred_ne = average_local_pred(
        per_inst_ne, len(ds_ne['features']), ds_ne['T'])

    print('\n[9] Generating figures ...')

    savefig(
        fig0_main_body(ds_ih, ds_ne, K_ih, K_ne,
                       global_sens_ih=global_ih['sensitivity'],
                       global_sens_ne=global_ne['sensitivity']),
        'fig0_main_body.pdf')

    for ds, global_eff, tag in [
        (ds_ih, global_ih, 'ihepc'),
        (ds_ne, global_ne, 'neso'),
    ]:
        savefig(
            fig1_global_risk_sensitivity(ds, global_eff, fs=_fs(3)),
            'fig1_global_risk_sensitivity_{}.pdf'.format(tag))

    for ds, local_g, tag in [
        (ds_ih, local_ih, 'ihepc'),
        (ds_ne, local_ne, 'neso'),
    ]:
        savefig(
            fig2_local_prediction(ds, local_g, fs=_fs(3)),
            'fig2_local_prediction_{}.pdf'.format(tag))

    for ds, avg_pred, per_inst, X_pdp, tag in [
        (ds_ih, avg_pred_ih, per_inst_ih, X_pdp_ih, 'ihepc'),
        (ds_ne, avg_pred_ne, per_inst_ne, X_pdp_ne, 'neso'),
    ]:
        savefig(
            fig3_global_pdp(ds, avg_pred, per_inst, X_pdp, fs=_fs(3)),
            'fig3_global_pdp_{}.pdf'.format(tag))

    for ds, local_g, tag in [
        (ds_ih, local_ih, 'ihepc'),
        (ds_ne, local_ne, 'neso'),
    ]:
        savefig(
            fig4_interactions(ds, local_g, fs=_fs(0)),
            'fig4_interactions_{}.pdf'.format(tag))

    savefig(
        fig5_networks_global(ds_ih, ds_ne, global_ih, global_ne, K_ih, K_ne),
        'fig5_networks_global.pdf')

    print('\n' + '='*60)
    print('  Done.  Figures saved to {}/'.format(BASE_PLOT_DIR))
    print('  Game caches in {}/'.format(GAME_CACHE_DIR))
    print('='*60)