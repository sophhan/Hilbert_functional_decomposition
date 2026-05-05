"""
Functional Explanation Framework — Intraday SPY Volatility (Random Forest)
============================================================================

This script applies the functional feature attribution framework to an
intraday volatility prediction task using SPY 5-minute bars.

A Random Forest is trained to map day-level features to intraday volatility
trajectories (78 bars, 09:30–16:00 ET). Cooperative game decompositions
(prediction, sensitivity, and risk games) are then computed and visualised
under two output kernels (OU and causal), illustrating how kernel choice
shapes the attribution profile.

Data availability
-----------------
The raw bar data (spy_5min_cache.csv) and VIX daily data (vix_daily_cache.csv)
are proprietary and cannot be redistributed. To reproduce the paper figures,
use the precomputed game result caches provided in the repository:

    game_results/global_sensitivity_v7.npz
    game_results/global_risk_v7.npz
    game_results/local_prediction_High_VIX_Announcement_v5.npz
    game_results/global_prediction_inst*.npz   (per-instance prediction)
    game_results/pdp_spy_inst*.npz              (per-instance PDP)

Cache version constants (do not change unless recomputing from scratch):
    CACHE_VERSION_LOCAL  = 'v5'   — local prediction game caches
    CACHE_VERSION_GLOBAL = 'v7'   — global sensitivity / risk game caches

Running with cached results
---------------------------
If the raw data is unavailable, the script will fail at step [1] (data
loading). All downstream steps (figures) can be reproduced by ensuring
the .npz cache files are present in game_results/. The caches are loaded
automatically when found.

Features (day-level predictors)
--------------------------------
    vix_prev        — previous day's VIX closing level
    overnight_ret   — log return from previous close to today's open
    ann_indicator   — binary: 1 on FOMC / CPI / NFP announcement dates
    day_of_week     — integer 0–4 (Monday–Friday)
    trailing_rv     — average realized volatility over the past 5 days
    month           — integer 1–12

Target
------
    Y(t) = |log(close_t / open_t)|   for each 5-minute bar t
    Diurnal mean is subtracted before modelling (Y_adj = Y - E[Y(t)]).

Cooperative games
-----------------
Three game formulations are implemented:

  Prediction game (local):
      v(S)(t) = E_{X_{-S}}[ F(x*_S, X_{-S})(t) ]
      Measures how much each feature shifts the predicted trajectory
      relative to the background mean, for a specific explicand x*.

  Sensitivity game (global):
      v(S)(t) = Var_{X_S}[ E_{X_{-S}}[ F(X_S, X_{-S})(t) ] ]
      Measures how much of the trajectory variance is explained by S.
      Pure effect = closed Sobol index; Partial = Shapley sensitivity;
      Full = total Sobol index.

  Risk / MSE game (global):
      v(S)(t) = baseline_loss(t) - E[(Y(t) - E_{X_{-S}}[F(X_S,X_{-S})(t)])^2]
      Measures how much knowing S reduces prediction error relative to
      predicting with the global mean. Pure = pure risk reduction;
      Partial = SAGE; Full = PFI.

Both global games use a double Monte Carlo estimator with shared outer/inner
background pools (n_outer, n_inner), so the estimates are consistent across
coalitions.

Output kernels
--------------
  OU kernel (length_scale=8 bars):   symmetric local smoothing
  Causal kernel (length_scale=8):    one-sided, no future leakage
      ann_indicator uses the causal kernel by default, since announcement
      effects propagate forward in time only.

Output figures
--------------
All figures are saved to plots/spy/:

    fig0_main_body_summary.pdf        — two-row summary for the paper
    fig1_sensitivity_risk_global.pdf  — global sensitivity + risk, 4 rows
    fig2_local_prediction.pdf         — local prediction, 2 rows
    fig3_global_prediction_pdp.pdf    — PDP-style global prediction
    fig4_interactions.pdf             — top pairwise interaction effects

Usage
-----
    python spy_hfd.py

With precomputed caches (no raw data required):
    Ensure game_results/*.npz are present, then run as above.
    Steps [3]–[7] will load from cache automatically.
"""

import itertools
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0.  Settings
# ---------------------------------------------------------------------------

# Cache version strings — these must match the .npz filenames in game_results/.
# Change only if recomputing games from scratch with new settings.
CACHE_VERSION_LOCAL  = 'v5'   # local prediction game caches
CACHE_VERSION_GLOBAL = 'v7'   # global sensitivity / risk game caches

_HERE          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(_HERE, 'data')
CACHE_PATH     = os.path.join(DATA_DIR, 'spy_5min_cache.csv')
VIX_CACHE_PATH = os.path.join(DATA_DIR, 'vix_daily_cache.csv')
GAME_CACHE_DIR = os.path.join(_HERE, 'game_results')

VIX_LOOKBACK_DAYS = 5    # days to look back when resolving prev-day VIX
RNG_SEED  = 42
T_BARS    = 78            # number of 5-minute bars in a trading day
dt        = 1.0           # bar spacing (unit = 1 bar = 5 min)

RF_N_EST  = 300
RF_JOBS   = -1

# Monte Carlo sample sizes for game value estimation
SAMPLE_SIZE = {
    'prediction' : 200,   # background samples per coalition (local game)
    'sensitivity': 600,
    'risk'       : 600,
}

# Settings for the averaged-local-prediction object (legacy, for PDP plots)
GLOBAL_N_INSTANCES = 30    # number of explicands to average over
GLOBAL_SAMPLE_SIZE = 100   # background samples per explicand
N_FEAT_BINS        = 20    # bins for PDP marginal feature plots
PDP_N_INSTANCES    = 120   # instances used in the PDP stratified sample

# Double-MC settings for global sensitivity / risk games
GLOBAL_N_OUTER = 100   # outer loop: draws of X_S (the "fixed" features)
GLOBAL_N_INNER = 100   # inner loop: draws of X_{-S} (marginalized features)

# Bar time labels (09:30, 09:35, ..., 16:00)
_open_min  = 9 * 60 + 30
BAR_LABELS = [
    '{:02d}:{:02d}'.format(
        (_open_min + i * 5) // 60,
        (_open_min + i * 5) % 60,
    )
    for i in range(T_BARS)
]
t_grid = np.arange(T_BARS, dtype=float)

BASE_PLOT_DIR = os.path.join('plots', 'spy')

# Feature names in the order used by the model
DAY_FEATURE_NAMES = [
    'vix_prev',
    'overnight_ret',
    'ann_indicator',
    'day_of_week',
    'trailing_rv',
    'month',
]

# Dates with scheduled macroeconomic announcements (FOMC, CPI, NFP).
# ann_indicator = 1 on these dates; 0 otherwise.
ANNOUNCEMENT_DATES = {
    '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15',
    '2022-07-27', '2022-09-21', '2022-11-02', '2022-12-14',
    '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14',
    '2023-07-26', '2023-09-20', '2023-11-01', '2023-12-13',
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
    '2022-01-12', '2022-02-10', '2022-03-10', '2022-04-12',
    '2022-05-11', '2022-06-10', '2022-07-13', '2022-08-10',
    '2022-09-13', '2022-10-13', '2022-11-10', '2022-12-13',
    '2023-01-12', '2023-02-14', '2023-03-14', '2023-04-12',
    '2023-05-10', '2023-06-13', '2023-07-12', '2023-08-10',
    '2023-09-13', '2023-10-12', '2023-11-14', '2023-12-12',
    '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10',
    '2022-01-07', '2022-02-04', '2022-03-04', '2022-04-01',
    '2022-05-06', '2022-06-03', '2022-07-08', '2022-08-05',
    '2022-09-02', '2022-10-07', '2022-11-04', '2022-12-02',
    '2023-01-06', '2023-02-03', '2023-03-10', '2023-04-07',
    '2023-05-05', '2023-06-02', '2023-07-07', '2023-08-04',
    '2023-09-01', '2023-10-06', '2023-11-03', '2023-12-08',
    '2024-01-05', '2024-02-02', '2024-03-08', '2024-04-05',
}

# ===========================================================================
# 0b.  Color shading helper
# ===========================================================================

def _shade_color(c, factor):
    """
    Lighten (factor > 0) or darken (factor < 0) a matplotlib color.
    factor=0 returns the original color.
    """
    r, g, b = mcolors.to_rgb(c)
    if factor >= 0:
        return (r + (1.0 - r) * factor,
                g + (1.0 - g) * factor,
                b + (1.0 - b) * factor)
    else:
        f = -factor
        return (r * (1.0 - f), g * (1.0 - f), b * (1.0 - f))


# ===========================================================================
# 1.  Cache helpers
# ===========================================================================

def _require_dir(path):
    os.makedirs(path, exist_ok=True)


def _cache_path_global(game_type):
    """
    Path for the single global-game cache file (sensitivity or risk).
    Global games do not depend on a specific explicand, so one file
    suffices per game type.
    """
    return os.path.join(
        GAME_CACHE_DIR,
        'global_{}_{}.npz'.format(game_type, CACHE_VERSION_GLOBAL))


def _cache_path_local(label):
    """Path for a named local-prediction game cache (e.g. High-VIX profile)."""
    safe = label.replace(' ', '_').replace('-', '_')
    return os.path.join(
        GAME_CACHE_DIR,
        'local_prediction_{}_{}.npz'.format(safe, CACHE_VERSION_LOCAL))


def _cache_path_local_pred_inst(instance_k):
    """Path for per-instance local prediction game cache (PDP average)."""
    return os.path.join(
        GAME_CACHE_DIR,
        'global_prediction_inst{:04d}_{}.npz'.format(
            instance_k, CACHE_VERSION_LOCAL))


def _cache_path_pdp(instance_k):
    """Path for per-instance PDP cache."""
    return os.path.join(
        GAME_CACHE_DIR,
        'pdp_spy_inst{:04d}_{}.npz'.format(
            instance_k, CACHE_VERSION_LOCAL))


def _save_instance_cache(path, x_inst, pure, partial, full, moebius, n_players):
    """
    Save pure, partial (Shapley), full effects and the full Möbius dictionary
    to a compressed .npz file.

    Möbius keys (subset tuples) are serialised as underscore-joined strings,
    with the empty set stored as 'empty'.
    """
    mob_keys   = []
    mob_arrays = []
    for S, arr in moebius.items():
        mob_keys.append('_'.join(str(x) for x in S) if S else 'empty')
        mob_arrays.append(arr)
    np.savez_compressed(
        path,
        x_inst   = x_inst,
        **{'pure_{}'.format(i):    pure[i]    for i in range(n_players)},
        **{'partial_{}'.format(i): partial[i] for i in range(n_players)},
        **{'full_{}'.format(i):    full[i]    for i in range(n_players)},
        mob_keys   = np.array(mob_keys, dtype=object),
        **{'mob_{}'.format(idx): arr for idx, arr in enumerate(mob_arrays)},
        n_players  = np.array([n_players]),
        n_mob      = np.array([len(mob_keys)]),
    )


def _load_instance_cache(path, n_players):
    """
    Load a cache file saved by _save_instance_cache.
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
    for idx in range(n_mob):
        key_str = mob_keys[idx]
        arr     = d['mob_{}'.format(idx)]
        S = () if key_str == 'empty' else tuple(int(x) for x in key_str.split('_'))
        moebius[S] = arr
    return x_inst, pure, partial, full, moebius


# ===========================================================================
# 2.  Data loading
# ===========================================================================

def _validate_bars(bars):
    """Basic integrity checks on the bar cache DataFrame."""
    required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    missing  = required - set(bars.columns)
    if missing:
        raise RuntimeError('Bar cache missing columns: {}'.format(missing))
    if len(bars) == 0:
        raise RuntimeError('Bar cache is empty.')
    if bars['close'].isna().mean() > 0.01:
        raise RuntimeError('More than 1 pct of close prices are NaN.')


def _validate_pivot(pivot):
    """Require at least 100 complete trading days."""
    if len(pivot) < 100:
        raise RuntimeError('Only {} complete trading days.'.format(len(pivot)))


def _validate_feature_matrix(X_day):
    """Check that all required features are present and non-constant."""
    for col in DAY_FEATURE_NAMES:
        if col not in X_day.columns:
            raise RuntimeError('Missing feature: {}'.format(col))
        if X_day[col].nunique() <= 1 or X_day[col].std() < 1e-8:
            raise RuntimeError('Feature {} is constant.'.format(col))


def _resolve_close_column(df, label):
    """Extract a close price series from a potentially multi-indexed DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        for field in ('Close', 'Adj Close'):
            for ticker in ('^VIX', 'VIX', ''):
                if (field, ticker) in df.columns:
                    s = df[(field, ticker)].dropna()
                    if len(s) > 0:
                        return s
    else:
        for col in ('Close', 'Adj Close', 'close'):
            if col in df.columns:
                s = df[col].dropna()
                if len(s) > 0:
                    return s
    raise RuntimeError('{}: no Close column.'.format(label))


def load_vix(first_date, last_date):
    """
    Load daily VIX closing levels from cache or download via yfinance.
    Returns a dict mapping date string -> float.
    """
    if os.path.isfile(VIX_CACHE_PATH):
        print('    Reading VIX cache ...')
        df = pd.read_csv(VIX_CACHE_PATH, index_col=0)
        return {str(k): float(v) for k, v in df['vix'].dropna().items()}
    import yfinance as yf
    vix_start = (pd.Timestamp(first_date)
                 - pd.Timedelta(days=VIX_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    end_ex = (pd.Timestamp(last_date)
              + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    raw   = yf.download('^VIX', start=vix_start, end=end_ex,
                        progress=False, auto_adjust=True)
    close = _resolve_close_column(raw, 'yfinance ^VIX')
    result = {
        (str(i.date()) if hasattr(i, 'date') else str(i)): float(v)
        for i, v in close.items()
    }
    pd.DataFrame({'vix': result}).to_csv(VIX_CACHE_PATH)
    return result


def load_bars():
    """
    Load the 5-minute bar cache from disk.
    Raises RuntimeError if the cache file is not found — raw data is
    proprietary and not included in the repository.
    """
    if os.path.isfile(CACHE_PATH):
        print('  Loading bar cache ...')
        bars = pd.read_csv(CACHE_PATH)
        bars['timestamp'] = (
            pd.to_datetime(bars['timestamp'], utc=True)
            .dt.tz_convert('America/New_York'))
        _validate_bars(bars)
        print('  Loaded {:,} rows.'.format(len(bars)))
        return bars
    raise RuntimeError(
        'Bar cache not found at: {}\n'
        'Raw data is proprietary and not included in the repository.\n'
        'To reproduce figures, use the precomputed game caches in '
        'game_results/.'.format(CACHE_PATH))


def _resolve_vix_prev(vix_dict, date_str):
    """
    Find the most recent available VIX value strictly before date_str
    by scanning up to 14 calendar days back.
    """
    dt_obj = pd.Timestamp(date_str)
    for d in range(1, 15):
        key = str((dt_obj - pd.Timedelta(d, 'D')).date())
        if key in vix_dict:
            return vix_dict[key]
    if date_str in vix_dict:
        return vix_dict[date_str]
    raise RuntimeError('No VIX within 14 days of {}'.format(date_str))


def load_and_aggregate():
    """
    Load 5-minute bars, aggregate to daily feature matrix and target matrix.

    Pipeline:
        1. Load bars; compute |log(close/open)| per bar as realized volatility
        2. Pivot to (days x bars) matrix; drop days with <70 valid bars
        3. Fill missing bars with cross-day mean; subtract diurnal mean
        4. Build day-level feature matrix (VIX, overnight return, etc.)

    Returns
    -------
    X_day        : DataFrame (n_days, n_features)
    Y_day        : ndarray (n_days, T_BARS) — raw volatility trajectories
    Y_adj        : ndarray (n_days, T_BARS) — diurnal-mean-adjusted trajectories
    diurnal_mean : ndarray (T_BARS,) — average volatility profile
    """
    bars = load_bars()
    bars = bars.copy()
    bars['date']    = bars['timestamp'].dt.date.astype(str)
    bars['bar_idx'] = (
        (bars['timestamp'].dt.hour * 60
         + bars['timestamp'].dt.minute - 570) // 5
    ).astype(int)
    bars = bars[(bars['bar_idx'] >= 0) & (bars['bar_idx'] < T_BARS)].copy()
    bars['open']  = pd.to_numeric(bars['open'],  errors='coerce')
    bars['close'] = pd.to_numeric(bars['close'], errors='coerce')
    bars = bars[(bars['open'] > 0) & (bars['close'] > 0)].copy()
    bars['abs_log_ret'] = np.abs(np.log(bars['close'] / bars['open']))

    pivot = bars.pivot_table(
        index='date', columns='bar_idx',
        values='abs_log_ret', aggfunc='mean')
    pivot = pivot.reindex(columns=range(T_BARS))
    pivot = pivot[pivot.notna().sum(axis=1) >= 70].fillna(pivot.mean())
    _validate_pivot(pivot)

    Y_day        = pivot.values.astype(float)
    dates        = pivot.index.tolist()
    diurnal_mean = Y_day.mean(axis=0)
    Y_adj        = Y_day - diurnal_mean[None, :]

    print('  Complete trading days: {}'.format(len(dates)))
    vix_dict = load_vix(dates[0], dates[-1])

    records = []
    for i, date_str in enumerate(dates):
        dt_obj        = pd.Timestamp(date_str)
        vix_prev      = _resolve_vix_prev(vix_dict, date_str)
        day_bars      = bars[bars['date'] == date_str].sort_values('bar_idx')
        prev_bars     = bars[bars['date'] < date_str].sort_values(
                            ['date', 'bar_idx'])
        this_open     = float(day_bars.iloc[0]['open'])
        overnight_ret = (
            0.0 if len(prev_bars) == 0
            else float(np.log(this_open / float(prev_bars.iloc[-1]['close'])))
        )
        recent      = list(range(max(0, i - 5), i))
        trailing_rv = (float(Y_day[recent].mean()) if recent
                       else float(Y_day.mean()))
        records.append({
            'vix_prev'     : vix_prev,
            'overnight_ret': overnight_ret,
            'ann_indicator': float(date_str in ANNOUNCEMENT_DATES),
            'day_of_week'  : float(dt_obj.dayofweek),
            'trailing_rv'  : trailing_rv,
            'month'        : float(dt_obj.month),
        })

    X_day = pd.DataFrame(records, index=dates)
    _validate_feature_matrix(X_day)
    print('  Announcement days: {} ({:.1f}%)'.format(
        int(X_day['ann_indicator'].sum()),
        X_day['ann_indicator'].mean() * 100))
    return X_day, Y_day, Y_adj, diurnal_mean


# ===========================================================================
# 3.  Model
# ===========================================================================

class RFModel:
    """
    Random Forest regressor wrapper for multi-output trajectory prediction.

    Predicts the full T_BARS-dimensional volatility trajectory from a
    day-level feature vector. Evaluated by R² on the held-out test set.
    """
    def __init__(self, n_estimators=RF_N_EST,
                 n_jobs=RF_JOBS, random_state=RNG_SEED):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=n_jobs,
            random_state=random_state)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, Y_test):
        """Return R² on trajectory-level predictions."""
        Y_pred = self.predict(X_test)
        ss_res = np.sum((Y_test - Y_pred) ** 2)
        ss_tot = np.sum((Y_test - Y_test.mean()) ** 2)
        return 1.0 - ss_res / ss_tot


# ===========================================================================
# 4.  Cooperative games
# ===========================================================================

class LocalPredictionGame:
    """
    Local (instance-specific) prediction game.

        v(S)(t) = E_{X_{-S}}[ F(x*_S, X_{-S})(t) ]

    where x* is the explicand. Estimates the conditional expectation by
    marginalising over background samples with features in S fixed to x*.

    This is the standard marginal imputation estimator used in SHAP.

    Parameters
    ----------
    predict_fn   : callable (n, p) -> (n, T)
    X_background : ndarray (n_bg, p) — reference distribution
    x_explain    : ndarray (p,) — the specific input being explained
    sample_size  : number of background samples per coalition evaluation
    """
    def __init__(self, predict_fn, X_background, x_explain,
                 sample_size=200, random_seed=RNG_SEED):
        self.predict_fn   = predict_fn
        self.X_background = X_background
        self.x_explain    = x_explain
        self.sample_size  = sample_size
        self.random_seed  = random_seed
        self.T            = T_BARS
        self.n_players    = len(DAY_FEATURE_NAMES)
        self.player_names = list(DAY_FEATURE_NAMES)
        self.coalitions   = np.array(
            list(itertools.product([False, True], repeat=self.n_players)),
            dtype=bool)
        self.n_coalitions = len(self.coalitions)
        self._idx = {tuple(c): i for i, c in enumerate(self.coalitions)}
        self.values = None

    def _impute(self, coalition):
        """Sample background rows and fix features in coalition to x*."""
        rng = np.random.default_rng(self.random_seed)
        idx = rng.integers(0, len(self.X_background), size=self.sample_size)
        X   = self.X_background[idx].copy()
        for j in range(self.n_players):
            if coalition[j]:
                X[:, j] = self.x_explain[j]
        return X

    def value_function(self, coalition):
        X      = self._impute(coalition)
        Y_pred = self.predict_fn(X)
        return Y_pred.mean(axis=0)

    def precompute(self):
        """Evaluate v(S) for all 2^p coalitions."""
        self.values = np.zeros((self.n_coalitions, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i + 1) % 16 == 0 or i + 1 == self.n_coalitions:
                print('    {}/{} coalitions done.'.format(
                    i + 1, self.n_coalitions))

    def __getitem__(self, coalition):
        return self.values[self._idx[coalition]]


class GlobalFunctionalGame:
    """
    Global sensitivity or risk game using a double Monte Carlo estimator.

    Sensitivity:
        v(S)(t) = Var_{X_S}[ E_{X_{-S}}[ F(X_S, X_{-S})(t) ] ]

        For each outer draw x_S^(i), compute the inner conditional mean
        g_S(x_S^(i))(t) = E_{X_{-S}}[F(x_S^(i), X_{-S})(t)].
        Then v(S)(t) = Var_i[ g_S(x_S^(i))(t) ].

    Risk (SAGE/PFI sign convention — positive means loss reduction):
        baseline_loss(t) = E[(Y(t) - mu(t))^2]  with mu = E[F(X)]
        full_loss(S)(t)  = E[(Y(t) - g_S(X_S)(t))^2]
        v(S)(t)          = baseline_loss(t) - full_loss(S)(t)

    Both games:
        v(empty)(t) = 0   (empty coalition contributes nothing)
        v(N)(t)     = Var[F(t)] or total risk reduction (maximum)

    Implementation uses shared outer/inner index pools across coalitions
    to reduce Monte Carlo noise from index resampling.

    Parameters
    ----------
    predict_fn : callable (n, p) -> (n, T)
    X_bg       : ndarray (n_bg, p) — background distribution
    Y_bg       : ndarray (n_bg, T) — observed targets (required for risk)
    game_type  : 'sensitivity' or 'risk'
    n_outer    : size of the outer pool (X_S draws)
    n_inner    : size of the inner pool (X_{-S} draws)
    """
    GAME_TYPES_OK = ('sensitivity', 'risk')

    def __init__(self, predict_fn, X_bg, Y_bg, T, features,
                 game_type, n_outer=GLOBAL_N_OUTER, n_inner=GLOBAL_N_INNER,
                 random_seed=RNG_SEED):
        if game_type not in self.GAME_TYPES_OK:
            raise ValueError('GlobalFunctionalGame supports {} only; got {}'
                             .format(self.GAME_TYPES_OK, game_type))
        if game_type == 'risk' and Y_bg is None:
            raise ValueError('Y_bg required for risk game.')
        self.predict_fn = predict_fn
        self.X_bg = X_bg
        self.Y_bg = Y_bg
        self.T = T
        self.game_type = game_type
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.seed = random_seed
        self.n_players = len(features)
        self.player_names = list(features)
        self.coalitions = np.array(
            list(itertools.product([False, True], repeat=self.n_players)),
            dtype=bool)
        self.n_coalitions = len(self.coalitions)
        self._idx = {tuple(c): i for i, c in enumerate(self.coalitions)}
        self.values = None

        # Draw shared outer/inner pools once so noise is consistent across S
        rng = np.random.default_rng(self.seed)
        self._outer_idx = rng.choice(len(X_bg), size=n_outer, replace=True)
        self._inner_idx = rng.choice(len(X_bg), size=n_inner, replace=True)

        # mu = E[F(X)] estimated from the inner pool
        Xb         = self.X_bg[self._inner_idx]
        Yp_b       = self.predict_fn(Xb)
        self._mu   = Yp_b.mean(axis=0)

        # Precompute baseline loss for the risk game
        if game_type == 'risk':
            Y_outer = self.Y_bg[self._outer_idx]
            self._baseline_loss = ((Y_outer - self._mu[None, :])**2).mean(axis=0)
        else:
            self._baseline_loss = None

    def _marginalized_predictions(self, coal):
        """
        Compute g_S(x_S^(i))(t) for all outer draws i via a batched prediction.

        For each outer draw x_S^(i), construct n_inner copies with X_S fixed to
        x_S^(i) and X_{-S} drawn from the inner pool, then average predictions.
        Total batch size: n_outer * n_inner.
        """
        X_outer = self.X_bg[self._outer_idx]
        X_inner = self.X_bg[self._inner_idx]
        n_outer, n_inner, p = self.n_outer, self.n_inner, self.n_players
        X_batch = np.tile(X_inner, (n_outer, 1))
        for j in range(p):
            if coal[j]:
                X_batch[:, j] = np.repeat(X_outer[:, j], n_inner)
        Yp = self.predict_fn(X_batch)
        Yp = Yp.reshape(n_outer, n_inner, self.T)
        return Yp.mean(axis=1)   # shape (n_outer, T)

    def value_function(self, coal):
        """Evaluate v(coal)(t) for a single coalition."""
        if not any(coal):
            return np.zeros(self.T)
        if all(coal):
            # Full coalition: use raw predictions (no marginalization)
            X_full = self.X_bg[self._outer_idx]
            Yp = self.predict_fn(X_full)
            if self.game_type == 'sensitivity':
                return Yp.var(axis=0)
            else:
                Y = self.Y_bg[self._outer_idx]
                full_loss = ((Y - Yp)**2).mean(axis=0)
                return self._baseline_loss - full_loss
        g_S = self._marginalized_predictions(coal)
        if self.game_type == 'sensitivity':
            return g_S.var(axis=0)
        else:
            Y = self.Y_bg[self._outer_idx]
            full_loss = ((Y - g_S)**2).mean(axis=0)
            return self._baseline_loss - full_loss

    def precompute(self):
        """Evaluate v(S) for all 2^p coalitions."""
        self.values = np.zeros((self.n_coalitions, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i + 1) % 16 == 0 or i + 1 == self.n_coalitions:
                print('    {}/{} coalitions done.'.format(
                    i + 1, self.n_coalitions))

    def __getitem__(self, coalition):
        return self.values[self._idx[coalition]]


# ===========================================================================
# 5.  Möbius transform and Shapley values
# ===========================================================================

def functional_moebius_transform(game):
    """
    Compute the functional Möbius coefficients m_S(t) for all subsets S.

    Inverts the inclusion-exclusion relation v(S) = sum_{L ⊆ S} m(L):
        m(S)(t) = sum_{L ⊆ S} (-1)^{|S|-|L|} v(L)(t)

    Returns a dict mapping subset tuples -> ndarray of shape (T,).
    """
    p     = game.n_players
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(p), r) for r in range(p + 1)))
    moebius = {}
    for S in all_S:
        m = np.zeros(game.T)
        for L in itertools.chain.from_iterable(
            itertools.combinations(S, r) for r in range(len(S) + 1)
        ):
            coalition = tuple(i in L for i in range(p))
            m += (-1) ** (len(S) - len(L)) * game[coalition]
        moebius[S] = m
    return moebius


def shapley_from_moebius(moebius, n_players):
    """
    Compute Shapley values from the Möbius coefficients:
        phi_i(t) = sum_{S: i in S} m(S)(t) / |S|

    Returns a dict mapping feature index -> ndarray of shape (T,).
    """
    shapley = {i: np.zeros(T_BARS) for i in range(n_players)}
    for S, m in moebius.items():
        if len(S) == 0:
            continue
        for i in S:
            shapley[i] += m / len(S)
    return shapley


# ===========================================================================
# 6.  Kernels
# ===========================================================================

def kernel_identity(t):
    """Identity kernel: K(t,s) = delta(t-s). No temporal coupling."""
    return np.eye(len(t))


def kernel_ou(t, length_scale=8.0):
    """
    OU (exponential) kernel: K(t,s) = exp(-|t-s| / length_scale).
    Symmetric local smoothing. Default length_scale = 8 bars = 40 min.
    """
    return np.exp(-np.abs(t[:, None] - t[None, :]) / length_scale)


def kernel_causal(t, length_scale=8.0):
    """
    Causal (one-sided exponential) kernel: K(t,s) = exp(-(t-s)/ell) if t>=s.
    Zero for future times — no anticipation of future events.
    Used for ann_indicator, since announcement effects propagate forward only.
    """
    d = t[:, None] - t[None, :]
    return np.where(d >= 0, np.exp(-d / length_scale), 0.0)


def _normalize_kernel(K):
    """Row-normalise K so each row sums to 1 (under the dt measure)."""
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return K / rs


def apply_kernel(effect, K):
    """
    Apply row-normalised kernel K to an effect curve:
        (Ke)(t) = [K(t,:) @ e] / [sum_s K(t,s) * dt]
    """
    return _normalize_kernel(K) @ effect * dt


def get_feature_kernel(fi, pnames):
    """
    Return the appropriate kernel for feature fi.
    ann_indicator uses the causal kernel (forward-only propagation);
    all other features use the OU kernel.
    """
    if pnames[fi] == 'ann_indicator':
        return ('causal', kernel_causal(t_grid, length_scale=8.0))
    return ('ou', kernel_ou(t_grid, length_scale=8.0))


# ===========================================================================
# 7.  Pure / partial / full effect helpers
# ===========================================================================

def _pure_effects(moebius, n_players):
    """
    Extract first-order (singleton) Möbius coefficients as pure effects.
    m_{i}(t) = effect attributable exclusively to feature i.
    """
    return {i: moebius.get((i,), np.zeros(T_BARS)).copy()
            for i in range(n_players)}


def _full_effects(moebius, n_players):
    """
    Compute full effects by summing all Möbius terms that include feature i:
        full_i(t) = sum_{S: i in S} m_S(t)
    This is the "full" (PFI-style) attribution that includes all interaction terms.
    """
    full = {i: np.zeros(T_BARS) for i in range(n_players)}
    for S, m in moebius.items():
        if len(S) == 0:
            continue
        for i in S:
            full[i] += m
    return full


# ===========================================================================
# 8.  Game computation with caching
# ===========================================================================

def _run_or_load_local_pred(predict_fn, X_background,
                             x_inst, sample_size,
                             random_seed, cache_path, n_players):
    """
    Compute or load a local prediction game for a single explicand.

    If the cache file exists, loads it; otherwise runs the full game,
    computes Möbius coefficients and Shapley values, and saves to cache.

    Returns
    -------
    x_inst, pure, shap (partial), full, moebius
    """
    if os.path.isfile(cache_path):
        return _load_instance_cache(cache_path, n_players)
    game = LocalPredictionGame(
        predict_fn   = predict_fn,
        X_background = X_background,
        x_explain    = x_inst,
        sample_size  = sample_size,
        random_seed  = random_seed,
    )
    game.precompute()
    mob  = functional_moebius_transform(game)
    shap = shapley_from_moebius(mob, n_players)
    pure = _pure_effects(mob, n_players)
    full = _full_effects(mob, n_players)
    _save_instance_cache(cache_path, x_inst, pure, shap, full, mob, n_players)
    return x_inst, pure, shap, full, mob


def compute_global_game(predict_fn, X_background, Y_adj, game_type,
                         n_outer=GLOBAL_N_OUTER, n_inner=GLOBAL_N_INNER,
                         seed=RNG_SEED, force_recompute=False):
    """
    Compute (or load from cache) a global sensitivity or risk game.

    Uses the GlobalFunctionalGame double-MC estimator. Results are saved
    to a single cache file per game type (not per-instance, since global
    games do not depend on a specific explicand).

    Returns
    -------
    (avg_shap, avg_pure, avg_full) — each a dict {feature_idx: ndarray(T)}
    Note: names are kept as 'avg_*' for plot-code compatibility; these are
    not averages but the direct global-game effects from one double-MC run.
    """
    _require_dir(GAME_CACHE_DIR)
    cache_path = _cache_path_global(game_type)
    n_players  = len(DAY_FEATURE_NAMES)

    if os.path.isfile(cache_path) and not force_recompute:
        print('    [global {}] loading from cache ...'.format(game_type))
        _, pure, partial, full, _ = _load_instance_cache(cache_path, n_players)
    else:
        print('    [global {}] computing double-MC '
              'n_outer={} n_inner={} ...'.format(
                  game_type, n_outer, n_inner))
        Y_for_risk = Y_adj if game_type == 'risk' else None
        game = GlobalFunctionalGame(
            predict_fn=predict_fn, X_bg=X_background, Y_bg=Y_for_risk,
            T=T_BARS, features=DAY_FEATURE_NAMES, game_type=game_type,
            n_outer=n_outer, n_inner=n_inner, random_seed=seed)
        game.precompute()
        mob     = functional_moebius_transform(game)
        partial = shapley_from_moebius(mob, n_players)
        pure    = _pure_effects(mob, n_players)
        full    = _full_effects(mob, n_players)
        x_dummy = np.zeros(n_players)
        _save_instance_cache(cache_path, x_dummy, pure, partial, full,
                              mob, n_players)
    return partial, pure, full


def compute_local_prediction(predict_fn, X_background, x_hv, label):
    """
    Compute or load the local prediction game for a named explicand profile
    (e.g. the High-VIX Announcement day).

    Returns
    -------
    mob : Möbius dictionary
    shap : Shapley value dictionary
    """
    _require_dir(GAME_CACHE_DIR)
    n_players  = len(DAY_FEATURE_NAMES)
    cache_path = _cache_path_local(label)
    if os.path.isfile(cache_path):
        print('  Loading local prediction cache for "{}" ...'.format(label))
        _, pure, shap, full, mob = _load_instance_cache(cache_path, n_players)
        return mob, shap
    print('  Computing local prediction game for "{}" ...'.format(label))
    _, _, shap, _, mob = _run_or_load_local_pred(
        predict_fn, X_background, x_hv, SAMPLE_SIZE['prediction'],
        RNG_SEED, cache_path, n_players)
    return mob, shap


def load_per_instance_pred_for_global_avg(predict_fn, X_background, seed):
    """
    Compute or load per-instance local prediction effects for a random sample
    of GLOBAL_N_INSTANCES explicands, then return their average.

    This produces the 'mean local Shapley' object, which equals the global
    prediction Shapley value because the prediction game's characteristic
    function is linear in F. Used to select the top-2 most important features
    for the PDP plots.

    When X_background is None, all GLOBAL_N_INSTANCES cache files must already
    exist; the function loads them directly without drawing new indices.

    Returns
    -------
    avg_shap, avg_pure, avg_full — each averaged over instances
    """
    _require_dir(GAME_CACHE_DIR)
    n_players = len(DAY_FEATURE_NAMES)
    sum_part = {i: np.zeros(T_BARS) for i in range(n_players)}
    sum_pure = {i: np.zeros(T_BARS) for i in range(n_players)}
    sum_full = {i: np.zeros(T_BARS) for i in range(n_players)}

    if X_background is None:
        # Cache-only path: load each of the GLOBAL_N_INSTANCES files directly.
        for k in range(GLOBAL_N_INSTANCES):
            cache = _cache_path_local_pred_inst(k)
            if not os.path.isfile(cache):
                raise RuntimeError(
                    'Cache file missing: {}\n'
                    'Raw data is required to compute this from scratch.'.format(cache))
            _, pure, shap, full, _ = _load_instance_cache(cache, n_players)
            for i in range(n_players):
                sum_part[i] += shap[i]
                sum_pure[i] += pure[i]
                sum_full[i] += full[i]
            print('    [avg local pred] {}/{} (cached)'.format(
                k + 1, GLOBAL_N_INSTANCES))
    else:
        rng  = np.random.default_rng(seed)
        idxs = rng.choice(len(X_background), size=GLOBAL_N_INSTANCES, replace=False)
        for k, idx in enumerate(idxs):
            cache  = _cache_path_local_pred_inst(k)
            x_inst = X_background[idx]
            _, pure, shap, full, _ = _run_or_load_local_pred(
                predict_fn, X_background, x_inst,
                GLOBAL_SAMPLE_SIZE, seed + k, cache, n_players)
            for i in range(n_players):
                sum_part[i] += shap[i]
                sum_pure[i] += pure[i]
                sum_full[i] += full[i]
            cached = '(cached)' if os.path.isfile(cache) else ''
            print('    [avg local pred] {}/{} {}'.format(
                k + 1, GLOBAL_N_INSTANCES, cached))

    n = GLOBAL_N_INSTANCES
    return ({i: sum_part[i]/n for i in range(n_players)},
            {i: sum_pure[i]/n for i in range(n_players)},
            {i: sum_full[i]/n for i in range(n_players)})


def load_per_instance_effects_pdp(predict_fn, X_background, seed):
    """
    Compute or load per-instance local prediction effects for a stratified
    sample of PDP_N_INSTANCES explicands.

    Stratification: one instance per calendar month, plus a random fill
    to reach PDP_N_INSTANCES total. This ensures seasonal coverage.

    When X_background is None, all PDP_N_INSTANCES cache files must already
    exist; the function loads them directly without index selection, and
    returns a zero-filled X_pdp_subset placeholder (the feature values are
    stored inside each cache entry and are not needed by the plotting code).

    Returns
    -------
    results      : list of dicts with keys 'x', 'pure', 'partial', 'full'
    X_pdp_subset : ndarray (PDP_N_INSTANCES, p) — explicand feature matrix
    """
    _require_dir(GAME_CACHE_DIR)
    n_players = len(DAY_FEATURE_NAMES)

    if X_background is None:
        # Cache-only path: load PDP_N_INSTANCES files directly.
        results = []
        x_rows  = []
        for k in range(PDP_N_INSTANCES):
            cache = _cache_path_pdp(k)
            if not os.path.isfile(cache):
                raise RuntimeError(
                    'Cache file missing: {}\n'
                    'Raw data is required to compute this from scratch.'.format(cache))
            x_inst, pure, shap, full, _ = _load_instance_cache(cache, n_players)
            results.append({'x': x_inst, 'pure': pure,
                            'partial': shap, 'full': full})
            x_rows.append(x_inst)
            print('    [pdp pred] {}/{} (cached)'.format(k + 1, PDP_N_INSTANCES))
        return results, np.array(x_rows)

    month_col = DAY_FEATURE_NAMES.index('month')
    # Stratified selection: one instance per month
    selected = []
    for m in range(1, 13):
        candidates = np.where(X_background[:, month_col] == m)[0]
        if len(candidates) > 0:
            rng_m = np.random.default_rng(seed + m)
            selected.append(int(rng_m.choice(candidates)))
    rng  = np.random.default_rng(seed)
    pool = np.setdiff1d(np.arange(len(X_background)), selected)
    rng.shuffle(pool)
    n_extra = PDP_N_INSTANCES - len(selected)
    if n_extra > 0:
        selected = selected + pool[:n_extra].tolist()
    idxs = np.array(selected[:PDP_N_INSTANCES])
    results = []
    for k, idx in enumerate(idxs):
        cache  = _cache_path_pdp(k)
        x_inst = X_background[idx]
        _, pure, shap, full, _ = _run_or_load_local_pred(
            predict_fn, X_background, x_inst,
            SAMPLE_SIZE['prediction'], seed + k, cache, n_players)
        results.append({'x': x_inst, 'pure': pure,
                        'partial': shap, 'full': full})
        print('    [pdp pred] {}/{}'.format(k + 1, PDP_N_INSTANCES))
    return results, X_background[idxs]


# ===========================================================================
# 9.  Plotting helpers
# ===========================================================================

XTICK_IDXS   = list(range(0, T_BARS, 6))
XTICK_LABELS = [BAR_LABELS[i] for i in XTICK_IDXS]

FS_SUPTITLE = 14
FS_TITLE    = 11
FS_AXIS     = 10
FS_TICK     = 9
FS_LEGEND   = 9
FS_ANNOT    = 9

from types import SimpleNamespace

def _fs(bump=0):
    """Return a SimpleNamespace of font sizes with an optional global bump."""
    return SimpleNamespace(
        suptitle = FS_SUPTITLE + bump,
        title    = FS_TITLE    + bump,
        axis     = FS_AXIS     + bump,
        tick     = FS_TICK     + bump,
        legend   = FS_LEGEND   + bump,
        annot    = FS_ANNOT    + bump,
    )

# Feature colors (Wong 2011 CVD-safe palette)
FEAT_COLORS = {
    'vix_prev'     : '#1f77b4',
    'overnight_ret': '#2ca02c',
    'ann_indicator': '#ff7f0e',
    'day_of_week'  : '#9467bd',
    'trailing_rv'  : '#8c564b',
    'month'        : '#d62728',
}

# Y-axis labels per game type
GAME_YLABEL = {
    'prediction' : 'Effect on vol (%)',
    'sensitivity': r'Var$[F(t)]$ ($\%^2 \times 10^4$)',
    'risk'       : r'Risk reduction ($\%^2 \times 10^4$)',
}

_EFFECT_TYPES = ['pure', 'partial', 'full']

# Display labels for pure / partial / full effects per game type
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
_SENS_LABELS = {
    'pure'   : 'Pure = closed Sobol',
    'partial': 'Partial = Shapley-sens.',
    'full'   : 'Full = total Sobol',
}
_RISK_LABELS = {
    'pure'   : 'Pure = pure risk',
    'partial': 'Partial = SAGE',
    'full'   : 'Full = PFI',
}


def _scale(game_type):
    """Scaling factor for display: prediction in %, sensitivity/risk x1e4."""
    return 100.0 if game_type == 'prediction' else 1e4


def _set_time_axis(ax, sparse=False):
    """Set x-axis ticks to bar time labels."""
    step = 2 if sparse else 1
    ax.set_xticks(XTICK_IDXS[::step])
    ax.set_xticklabels(XTICK_LABELS[::step],
                       rotation=45, ha='right', fontsize=FS_TICK)
    ax.set_xlim(-0.5, T_BARS - 0.5)


def _period_shade(ax):
    """Shade the market open (first 6 bars) and close (last 6 bars) regions."""
    ax.axvspan(0,  6,  alpha=0.08, color='#ffd699', zorder=0)
    ax.axvspan(72, 78, alpha=0.08, color='#ffd699', zorder=0)


def _ann_vline(ax):
    """Draw a dashed vertical line at bar 54 (14:00 ET — FOMC decision time)."""
    ax.axvline(54, color='#555', lw=0.9, ls='--', alpha=0.6)


def savefig(fig, name):
    """Save figure as PDF to BASE_PLOT_DIR and close it."""
    path = os.path.join(BASE_PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print('  Saved: {}'.format(path))
    plt.close(fig)


def _align_row_ylims(axes_row):
    """Set a common y-axis range across a list of axes."""
    ymin = min(ax.get_ylim()[0] for ax in axes_row)
    ymax = max(ax.get_ylim()[1] for ax in axes_row)
    for ax in axes_row:
        ax.set_ylim(ymin, ymax)


def _draw_bar_panel(ax, effect_dicts, kern_fn, pnames, sc,
                    title='Time-aggregated', fs=None):
    """
    Draw a horizontal bar chart of time-aggregated attribution magnitudes.

    Features are ordered by descending partial (Shapley) importance.
    Pure / partial / full effects are shown as three bars per feature
    using light / medium / dark shading of the feature color.
    """
    if fs is None:
        fs = _fs(0)
    n_players = len(pnames)
    imps_all = {
        etype: {i: float(np.sum(np.abs(
            apply_kernel(effect_dicts[etype][i], kern_fn(i))))) * sc
            for i in range(n_players)}
        for etype in _EFFECT_TYPES
    }
    order   = sorted(range(n_players),
                     key=lambda i: imps_all['partial'][i], reverse=True)
    y_pos   = np.arange(len(order))
    bar_h   = 0.25
    offsets       = {'pure': -bar_h, 'partial': 0.0, 'full': bar_h}
    shade_factors = {'pure':  0.55,  'partial': 0.0, 'full': -0.40}
    for etype in _EFFECT_TYPES:
        sf = shade_factors[etype]
        bar_colors = [_shade_color(FEAT_COLORS[pnames[i]], sf) for i in order]
        ax.barh(y_pos + offsets[etype],
                [imps_all[etype][i] for i in order],
                height=bar_h, color=bar_colors, alpha=1.0,
                edgecolor='none', label=etype)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([pnames[i] for i in order], fontsize=fs.tick)
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=fs.axis)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=fs.tick)
    ax.set_title(title, fontsize=fs.title, fontweight='bold')
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


def _add_bottom_legends(fig, pnames, top_features, fs):
    """Add feature color legend and kernel linestyle legend at figure bottom."""
    feat_handles = [
        Line2D([0], [0], color=FEAT_COLORS[pnames[fi]], lw=1.8, ls='-',
               label=pnames[fi])
        for fi in top_features
    ]
    kern_handles = [
        Line2D([0], [0], color='#555', lw=1.8, ls='-',  label='OU kernel'),
        Line2D([0], [0], color='#555', lw=1.8, ls='--', label='Causal kernel'),
    ]
    leg1 = fig.legend(
        handles=feat_handles, fontsize=fs.legend, loc='lower left',
        bbox_to_anchor=(0.04, 0.04), framealpha=0.9, ncol=len(feat_handles))
    fig.add_artist(leg1)
    fig.legend(
        handles=kern_handles, fontsize=fs.legend, loc='lower left',
        bbox_to_anchor=(0.04, 0.0), framealpha=0.9, ncol=2)


# ===========================================================================
# 10.  Figure 1 — Global Sensitivity + Risk (4 rows x 4 cols)
# ===========================================================================

def plot_sensitivity_risk_global(global_sens, global_risk, pnames, fs=None):
    """
    Generate fig1_sensitivity_risk_global.pdf.

    Four-row layout:
        Row 0: Risk — Identity kernel
        Row 1: Risk — OU + Causal kernel
        Row 2: Sensitivity — Identity kernel
        Row 3: Sensitivity — OU + Causal kernel

    Each row: 3 time-resolved panels (pure / partial / full) + 1 bar panel.
    """
    if fs is None:
        fs = _fs(0)
    n_players = len(pnames)
    K_id      = kernel_identity(t_grid)

    def kern_id(fi):  return K_id
    def kern_mix(fi): return get_feature_kernel(fi, pnames)[1]

    row_specs = [
        ('risk',        global_risk, kern_id,  'identity',    _RISK_LABELS,
         'Risk — Identity kernel'),
        ('risk',        global_risk, kern_mix, 'OU + Causal', _RISK_LABELS,
         'Risk — OU + Causal kernel'),
        ('sensitivity', global_sens, kern_id,  'identity',    _SENS_LABELS,
         'Sensitivity — Identity kernel'),
        ('sensitivity', global_sens, kern_mix, 'OU + Causal', _SENS_LABELS,
         'Sensitivity — OU + Causal kernel'),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(20, 4.2 * 4),
                             gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Global Sensitivity and Risk effects — pure / partial / full\n'
        '(double-MC, n_outer={}, n_inner={}) — Random Forest'.format(
            GLOBAL_N_OUTER, GLOBAL_N_INNER),
        fontsize=fs.suptitle, fontweight='bold')

    top_k = 5; top_feats = None

    for r, (gtype, g_eff, kern_fn, klabel, eff_labels, row_title) in \
            enumerate(row_specs):
        sc = _scale(gtype)
        avg_shap, avg_pure, avg_full = g_eff
        effect_dicts = {'pure': avg_pure, 'partial': avg_shap, 'full': avg_full}

        # Select top-k features by integrated partial (Shapley) importance
        imps = {i: float(np.sum(np.abs(apply_kernel(avg_shap[i], kern_fn(i)))))
                for i in range(n_players)}
        top  = sorted(imps, key=imps.get, reverse=True)[:top_k]
        if top_feats is None:
            top_feats = top

        for c, etype in enumerate(_EFFECT_TYPES):
            ax  = axes[r, c]
            eff = effect_dicts[etype]
            for fi in top:
                K  = kern_fn(fi)
                # Use dashed line for ann_indicator under mixed kernel
                ls = '--' if pnames[fi] == 'ann_indicator' \
                             and klabel == 'OU + Causal' else '-'
                ax.plot(t_grid, apply_kernel(eff[fi], K) * sc,
                        color=FEAT_COLORS[pnames[fi]], lw=1.8, ls=ls)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax); _ann_vline(ax); _set_time_axis(ax, sparse=True)
            ax.tick_params(labelsize=fs.tick)
            ax.set_xlabel('Time', fontsize=fs.axis)
            ax.set_title(eff_labels[etype], fontsize=fs.title, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=fs.axis)
                ax.text(-0.44, 0.5, row_title, transform=ax.transAxes,
                        fontsize=fs.axis - 1, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')

        _draw_bar_panel(axes[r, 3], effect_dicts, kern_fn, pnames, sc, fs=fs)
        _align_row_ylims([axes[r, c] for c in range(3)])

    feat_handles_f1 = [
        Line2D([0], [0], color=FEAT_COLORS[pnames[fi]], lw=1.8, ls='-',
               label=pnames[fi])
        for fi in top_feats]
    kern_handles_f1 = [
        Line2D([0], [0], color='#555', lw=1.8, ls='-',  label='OU kernel'),
        Line2D([0], [0], color='#555', lw=1.8, ls='--', label='Causal kernel'),
    ]
    leg1 = fig.legend(handles=feat_handles_f1, fontsize=fs.legend,
                      loc='lower left', bbox_to_anchor=(0.08, 0.025),
                      framealpha=0.9, ncol=len(feat_handles_f1))
    fig.add_artist(leg1)
    fig.legend(handles=kern_handles_f1, fontsize=fs.legend,
               loc='lower left', bbox_to_anchor=(0.08, 0.0),
               framealpha=0.9, ncol=2)
    plt.tight_layout(rect=[0.08, 0.07, 1, 0.94])
    fig.subplots_adjust(hspace=0.65, top=0.92)
    return fig


# ===========================================================================
# 11.  Figure 2 — Local Prediction (2 rows x 4 cols)
# ===========================================================================

def plot_local_prediction(moebius_hv, shapley_hv, pnames, fs=None):
    """
    Generate fig2_local_prediction.pdf.

    Two-row layout (Identity kernel / OU + Causal kernel) showing pure,
    partial, and full local prediction effects for the High-VIX Announcement
    profile, plus time-aggregated bar charts.
    """
    if fs is None:
        fs = _fs(0)
    n_players = len(pnames)
    K_id = kernel_identity(t_grid)

    def kern_id(fi):  return K_id
    def kern_mix(fi): return get_feature_kernel(fi, pnames)[1]

    sc          = _scale('prediction')
    moebius     = moebius_hv['prediction']
    pure_eff    = _pure_effects(moebius, n_players)
    partial_eff = shapley_hv['prediction']
    full_eff    = _full_effects(moebius, n_players)

    top_k     = 5
    row_specs = [
        (kern_id,  'Identity kernel',    'identity'),
        (kern_mix, 'OU + Causal kernel', 'OU + Causal'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 4.2 * 2),
                             gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Local Prediction effects — pure / partial / full\n'
        'High-VIX Announcement profile — Random Forest',
        fontsize=fs.suptitle, fontweight='bold')

    top_feats = None
    for r, (kern_fn, klabel, kkey) in enumerate(row_specs):
        effect_dicts = {'pure': pure_eff, 'partial': partial_eff, 'full': full_eff}
        imps = {i: float(np.sum(np.abs(apply_kernel(partial_eff[i], kern_fn(i)))))
                for i in range(n_players)}
        top  = sorted(imps, key=imps.get, reverse=True)[:top_k]
        if top_feats is None:
            top_feats = top

        for c, etype in enumerate(_EFFECT_TYPES):
            ax  = axes[r, c]
            eff = effect_dicts[etype]
            for fi in top:
                K  = kern_fn(fi)
                ls = '--' if pnames[fi] == 'ann_indicator' \
                             and kkey == 'OU + Causal' else '-'
                ax.plot(t_grid, apply_kernel(eff[fi], K) * sc,
                        color=FEAT_COLORS[pnames[fi]], lw=1.8, ls=ls)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax); _ann_vline(ax); _set_time_axis(ax, sparse=True)
            ax.tick_params(labelsize=fs.tick)
            ax.set_xlabel('Time', fontsize=fs.axis)
            ax.set_title(_LOCAL_PRED_LABELS[etype],
                         fontsize=fs.title, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=fs.axis)
                ax.text(-0.30, 0.5, 'Prediction\n{}'.format(klabel),
                        transform=ax.transAxes, fontsize=fs.axis - 1,
                        va='center', ha='right', rotation=90,
                        color='#333', fontweight='bold')

        _draw_bar_panel(axes[r, 3], effect_dicts, kern_fn, pnames, sc, fs=fs)
        _align_row_ylims([axes[r, c] for c in range(3)])

    _add_bottom_legends(fig, pnames, top_feats, fs=fs)
    plt.tight_layout(rect=[0.04, 0.10, 1, 0.91])
    fig.subplots_adjust(hspace=0.65, top=0.88)
    return fig


# ===========================================================================
# 11b.  Figure 4 — Pairwise Interactions (2 rows x 2 cols)
# ===========================================================================

def plot_interactions(moebius_hv, pnames, fs=None):
    """
    Generate fig4_interactions.pdf.

    Shows the top-5 pairwise interaction effects m_{ij}(t) for the local
    prediction game under Identity and OU + Causal kernels.
    ann_indicator interactions use the causal kernel in the mixed row.
    """
    if fs is None:
        fs = _fs(0)
    ax_tick_fs  = max(fs.tick  - 2, 6)
    ax_label_fs = max(fs.axis  - 1, 7)

    n_players   = len(pnames)
    K_id        = kernel_identity(t_grid)
    K_ou        = kernel_ou(t_grid, 8.0)
    moebius     = moebius_hv['prediction']
    sc          = _scale('prediction')
    PAIR_COLORS = ['#e63946', '#2a9d8f', '#8338ec', '#fb8500', '#457b9d']

    # Rank all pairs by integrated OU-kernel interaction magnitude
    pair_imp = {}
    for i in range(n_players):
        for j in range(i + 1, n_players):
            raw = moebius.get((i, j), np.zeros(T_BARS))
            pair_imp[(i, j)] = float(np.sum(np.abs(apply_kernel(raw, K_ou))))
    top5 = sorted(pair_imp, key=pair_imp.get, reverse=True)[:5]

    fig, axes = plt.subplots(2, 2, figsize=(12, 4.2 * 2),
                             gridspec_kw={'width_ratios': [3, 1.8]})
    fig.suptitle(
        'Local pairwise interaction effects — top-5 pairs\n'
        'High-VIX Announcement profile — Random Forest',
        fontsize=fs.suptitle, fontweight='bold')

    inter_row_specs = [
        (K_id, 'identity',    'Interactions — Identity kernel'),
        (K_ou, 'OU + Causal', 'Interactions — OU + Causal kernel'),
    ]

    for r, (K_base, kkey, row_label) in enumerate(inter_row_specs):
        ax = axes[r, 0]
        fi_ann = pnames.index('ann_indicator')
        for pair_idx, (i, j) in enumerate(top5):
            raw = moebius.get((i, j), np.zeros(T_BARS))
            if kkey == 'OU + Causal':
                is_causal = (i == fi_ann or j == fi_ann)
                K_use = kernel_causal(t_grid, 8.0) if is_causal else K_ou
                ls    = '--' if is_causal else '-'
            else:
                K_use = K_id
                ls    = '-'
            ax.plot(t_grid, apply_kernel(raw, K_use) * sc,
                    color=PAIR_COLORS[pair_idx], lw=1.8, ls=ls)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax)
        ax.set_xticks(XTICK_IDXS[::2])
        ax.set_xticklabels(XTICK_LABELS[::2],
                           rotation=45, ha='right', fontsize=ax_tick_fs)
        ax.set_xlim(-0.5, T_BARS - 0.5)
        ax.tick_params(labelsize=ax_tick_fs)
        ax.set_xlabel('Time', fontsize=ax_label_fs)
        ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=ax_label_fs)
        ax.set_title('Pairwise interaction', fontsize=fs.title, fontweight='bold')
        ax.text(-0.18, 0.5, row_label, transform=ax.transAxes,
                fontsize=ax_label_fs - 1, va='center', ha='right',
                rotation=90, color='#333', fontweight='bold')
        _align_row_ylims([axes[r, 0]])

        ax_bar = axes[r, 1]
        row_imps = []
        for i, j in top5:
            raw = moebius.get((i, j), np.zeros(T_BARS))
            K_use = (kernel_causal(t_grid, 8.0)
                     if (kkey == 'OU + Causal' and (i == fi_ann or j == fi_ann))
                     else (K_id if kkey == 'identity' else K_ou))
            row_imps.append((float(np.sum(np.abs(apply_kernel(raw, K_use)))) * sc,
                             (i, j)))
        row_imps_sorted = sorted(row_imps, key=lambda x: x[0], reverse=True)
        y_pos = np.arange(len(row_imps_sorted))
        ax_bar.barh(y_pos,
                    [imp for imp, _ in row_imps_sorted],
                    color=[PAIR_COLORS[top5.index(ij)]
                           for _, ij in row_imps_sorted],
                    alpha=0.85)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(
            ['{} x {}'.format(pnames[i], pnames[j])
             for _, (i, j) in row_imps_sorted],
            fontsize=ax_tick_fs - 1)
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=ax_label_fs)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=ax_tick_fs)
        ax_bar.set_title('Time-aggregated', fontsize=fs.title, fontweight='bold')

    pair_handles = [
        Line2D([0], [0], color=PAIR_COLORS[k], lw=1.8,
               label='{} x {}'.format(pnames[i], pnames[j]))
        for k, (i, j) in enumerate(top5)]
    kern_handles = [
        Line2D([0], [0], color='#555', lw=1.8, ls='-',  label='OU kernel'),
        Line2D([0], [0], color='#555', lw=1.8, ls='--', label='Causal kernel'),
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
# 12.  Figure 3 — Global Prediction PDP-style (4 rows x 3 cols)
# ===========================================================================

def _top2_global_features_avg_local(avg_local_pred, pnames):
    """Select the two most important features by integrated OU Shapley value."""
    n_players = len(pnames)
    avg_shap, _, _ = avg_local_pred
    K_ou = kernel_ou(t_grid, 8.0)
    imps = {i: float(np.sum(np.abs(apply_kernel(avg_shap[i], K_ou))))
            for i in range(n_players)}
    return sorted(imps, key=imps.get, reverse=True)[:2]


def _pdp_panel(ax, fi, etype, kern_fn, X_background,
               per_instance_effects, pnames, sc,
               selected_t_idxs, t_cmap, fs, n_bins=N_FEAT_BINS):
    """
    Draw a single PDP panel: per-instance effects binned by feature value,
    with time-slice curves and a time-aggregated mean curve.

    For each bin of the marginal feature distribution, the average
    kernel-smoothed effect is computed across all instances in that bin.
    Time-slice curves show the effect at selected bar indices; the thick
    black line shows the time-aggregated (averaged over t) effect.
    """
    feat_vals   = X_background[:, fi]
    fmin, fmax  = feat_vals.min(), feat_vals.max()
    bins        = np.linspace(fmin, fmax, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx     = np.clip(np.digitize(feat_vals, bins) - 1, 0, n_bins - 1)
    K = kern_fn(fi)
    bin_effects = np.full((n_bins, T_BARS), np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        effects = np.array([
            apply_kernel(per_instance_effects[k][etype][fi], K)
            for k in np.where(mask)[0]
        ])
        bin_effects[b] = effects.mean(axis=0)
    valid  = ~np.isnan(bin_effects[:, 0])
    n_t    = len(selected_t_idxs)
    colors = [t_cmap(i / (n_t - 1)) for i in range(n_t)]
    for ti, col in zip(selected_t_idxs, colors):
        ax.plot(bin_centers[valid], bin_effects[valid, ti] * sc,
                color=col, lw=1.4, ls='--', alpha=0.85, label=BAR_LABELS[ti])
    # Time-aggregated mean curve
    pdp = np.nanmean(bin_effects, axis=1) * sc
    ax.plot(bin_centers[valid], pdp[valid],
            color='black', lw=2.5, ls='-', zorder=5, label='time-agg.')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.tick_params(labelsize=fs.tick)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, prune='both'))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha('right'); lbl.set_fontsize(fs.tick)
    ax.set_xlabel(pnames[fi], fontsize=fs.axis)


def plot_global_prediction_pdp(per_instance_effects, avg_local_pred,
                                X_background, pnames, fs=None):
    """
    Generate fig3_global_prediction_pdp.pdf.

    PDP-style global prediction figure: per-instance local effects are
    binned by the top-2 features' marginal distributions, showing how
    the attribution profile varies across the feature range.

    Four rows: top-2 features x 2 kernels (Identity / OU).
    Three columns: pure / partial / full effects.
    """
    if fs is None:
        fs = _fs(0)
    K_id = kernel_identity(t_grid)
    K_ou = kernel_ou(t_grid, 8.0)

    def kern_id(fi):    return K_id
    def kern_ou_fn(fi): return K_ou

    sc   = _scale('prediction')
    top2 = _top2_global_features_avg_local(avg_local_pred, pnames)

    selected_t_idxs = [0, 18, 42, 54, 77]
    t_cmap = cm.get_cmap('plasma', len(selected_t_idxs))

    feat_row_specs = []
    for fi in top2:
        feat_row_specs.append((fi, kern_id,    'Identity kernel'))
        feat_row_specs.append((fi, kern_ou_fn, 'OU kernel'))

    fig, axes = plt.subplots(4, 3, figsize=(18, 3.8 * 4))
    fig.suptitle(
        'Global Prediction effects — PDP-style — pure / partial / full\n'
        '(per-instance local effects, binned over feature range) — Random Forest',
        fontsize=fs.suptitle, fontweight='bold')

    for r, (fi, kern_fn, klabel) in enumerate(feat_row_specs):
        for c, etype in enumerate(_EFFECT_TYPES):
            ax = axes[r, c]
            _pdp_panel(ax, fi, etype, kern_fn, X_background,
                       per_instance_effects, pnames, sc,
                       selected_t_idxs, t_cmap, fs=fs)
            ax.set_title(_GLOBAL_PRED_LABELS[etype],
                         fontsize=fs.title, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=fs.axis)
                ax.text(-0.22, 0.5, '{}\n{}'.format(pnames[fi], klabel),
                        transform=ax.transAxes, fontsize=fs.axis - 1,
                        va='center', ha='right', rotation=90,
                        color='#333', fontweight='bold')
        _align_row_ylims([axes[r, c] for c in range(3)])

    t_handles = [
        Line2D([0], [0], color=t_cmap(i / (len(selected_t_idxs) - 1)),
               lw=1.4, ls='--', label=BAR_LABELS[ti])
        for i, ti in enumerate(selected_t_idxs)
    ] + [Line2D([0], [0], color='black', lw=2.5, ls='-', label='time-agg.')]
    fig.legend(handles=t_handles, fontsize=fs.legend, loc='lower center',
               ncol=len(t_handles), bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
    plt.tight_layout(rect=[0.04, 0.06, 1, 0.93])
    fig.subplots_adjust(hspace=0.80, top=0.91)
    return fig


# ===========================================================================
# 13.  Figure 0 — Main body summary (2 rows x 4 cols)
# ===========================================================================

def plot_main_body_summary(moebius_hv, shapley_hv, pnames):
    """
    Generate fig0_main_body_summary.pdf.

    Compact two-row summary figure for the paper's main body:
        Row 0: Local prediction — Identity kernel
        Row 1: Local prediction — OU + Causal kernel

    Each row shows pure, partial (Shapley), and the vix_prev x ann_indicator
    pairwise interaction effect, plus a time-aggregated bar chart.

    The annotation box on the partial panel reports the ratio of the
    ann_indicator's partial to pure integrated effect, quantifying how
    much the interaction with vix_prev amplifies its attribution.
    """
    n_players = len(pnames)
    fi_vix    = pnames.index('vix_prev')
    fi_ann    = pnames.index('ann_indicator')

    K_id     = kernel_identity(t_grid)
    K_ou     = kernel_ou(t_grid, length_scale=8.0)
    K_causal = kernel_causal(t_grid, length_scale=8.0)

    def kern_id(fi):  return K_id
    def kern_mix(fi): return K_causal if pnames[fi] == 'ann_indicator' else K_ou

    pure_pred    = _pure_effects(moebius_hv['prediction'], n_players)
    partial_pred = shapley_hv['prediction']
    full_pred    = _full_effects(moebius_hv['prediction'], n_players)

    row_specs = [
        ('prediction', GAME_YLABEL['prediction'],
         pure_pred, partial_pred, full_pred, kern_id,
         'Prediction\n(Identity)',
         'Pure', 'Partial', 'Full',
         K_id, _scale('prediction')),
        ('prediction', GAME_YLABEL['prediction'],
         pure_pred, partial_pred, full_pred, kern_mix,
         'Prediction\n(OU + Causal)',
         'Pure', 'Partial', 'Full',
         K_causal, _scale('prediction')),
    ]

    fig, axes = plt.subplots(
        2, 4, figsize=(19, 2.9 * 2),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Functional explanation framework: intraday volatility — '
        'kernel choice shapes the attribution profile\n'
        'High-VIX Announcement profile — Local Prediction',
        fontsize=FS_SUPTITLE, fontweight='bold')

    for r, (gtype, y_label, pure_eff, partial_eff, full_eff,
            kern_fn, row_label,
            lbl_pure, lbl_partial, lbl_full,
            K_inter, sc) in enumerate(row_specs):

        show = [fi_vix, fi_ann]

        def _plot_feat(ax, eff):
            for fi in show:
                K  = kern_fn(fi)
                ls = '--' if pnames[fi] == 'ann_indicator' else '-'
                ax.plot(t_grid, apply_kernel(eff[fi], K) * sc,
                        color=FEAT_COLORS[pnames[fi]], lw=2.0, ls=ls,
                        label=pnames[fi])

        # Pure effects panel
        ax = axes[r, 0]
        _plot_feat(ax, pure_eff)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax, sparse=True)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_ylabel(y_label, fontsize=FS_AXIS)
        ax.set_title(lbl_pure, fontsize=FS_TITLE - 1, fontweight='bold')
        ax.text(-0.32, 0.5, row_label, transform=ax.transAxes,
                fontsize=FS_AXIS, va='center', ha='right',
                rotation=90, color='#333', fontweight='bold')
        legend_loc = 'lower left' if r == 1 else 'upper left'
        ax.legend(fontsize=FS_LEGEND, loc=legend_loc, framealpha=0.9)

        # Partial (Shapley) effects panel with partial/pure ratio annotation
        ax = axes[r, 1]
        _plot_feat(ax, partial_eff)
        pure_int = float(np.sum(np.abs(
            apply_kernel(pure_eff[fi_ann], kern_fn(fi_ann))))) * sc
        part_int = float(np.sum(np.abs(
            apply_kernel(partial_eff[fi_ann], kern_fn(fi_ann))))) * sc
        ratio = part_int / pure_int if pure_int > 1e-12 else 1.0
        ax.text(0.03, 0.97,
                'ann: partial/pure = {:.2f}x'.format(ratio),
                transform=ax.transAxes, fontsize=FS_ANNOT - 1,
                va='top', ha='left', color=FEAT_COLORS['ann_indicator'],
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#ddd', alpha=0.9))
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax, sparse=True)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_title(lbl_partial, fontsize=FS_TITLE - 1, fontweight='bold')

        # Interaction panel: vix_prev x ann_indicator
        ax = axes[r, 2]
        raw    = moebius_hv['prediction'].get(
            (fi_vix, fi_ann), np.zeros(T_BARS))
        int_mx = apply_kernel(raw, K_inter) * sc
        pos    = np.where(int_mx >= 0, int_mx, 0.0)
        neg    = np.where(int_mx <  0, int_mx, 0.0)
        ax.fill_between(t_grid, 0, pos, color='#2a9d8f', alpha=0.30)
        ax.fill_between(t_grid, 0, neg, color='#e63946', alpha=0.30)
        ax.plot(t_grid, int_mx, color='#333', lw=1.8)
        integ = float(np.trapz(raw, t_grid)) * sc
        ax.text(0.03, 0.97,
                r'$\int m_{{ij}}\,dt$ = {:.3f}'.format(integ),
                transform=ax.transAxes, fontsize=FS_ANNOT,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#aaa', alpha=0.85))
        ax.set_title('Interaction — vix_prev x ann_indicator',
                     fontsize=FS_TITLE - 1, fontweight='bold')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax, sparse=True)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)

        # Time-aggregated bar chart
        effect_dicts_bar = {
            'pure': pure_eff, 'partial': partial_eff, 'full': full_eff}
        _draw_bar_panel(axes[r, 3], effect_dicts_bar, kern_fn, pnames, sc)
        _align_row_ylims([axes[r, c] for c in range(3)])

    plt.tight_layout(rect=[0.04, 0, 1, 0.91])
    fig.subplots_adjust(top=0.86, hspace=0.70)
    return fig


# ===========================================================================
# 14.  Sanity checks
# ===========================================================================

def sanity_check_global(global_eff, gtype):
    """
    Print sanity check statistics for a global game:
        - Sum of partial (Shapley) effects integrated over t
          (should equal total game value for sensitivity/risk)
        - Count of features with negative integrated effect
          (should be 0 for sensitivity and risk, which are non-negative by construction)
    """
    avg_shap, avg_pure, avg_full = global_eff
    p = len(avg_shap); T = avg_shap[0].shape[0]
    sum_partial = np.zeros(T)
    for i in range(p):
        sum_partial += avg_shap[i]
    K_id = np.eye(T)
    n_neg_pure = sum(1 for i in range(p)
                     if float(np.sum(apply_kernel(avg_pure[i], K_id))) < -1e-6)
    n_neg_part = sum(1 for i in range(p)
                     if float(np.sum(apply_kernel(avg_shap[i], K_id))) < -1e-6)
    n_neg_full = sum(1 for i in range(p)
                     if float(np.sum(apply_kernel(avg_full[i], K_id))) < -1e-6)
    print('  [sanity {}]'.format(gtype))
    print('    sum partial integral = {:.6g}'.format(float(sum_partial.sum())))
    print('    # features with negative integrated effect '
          '(pure/partial/full): {}/{}/{} of {}'.format(
              n_neg_pure, n_neg_part, n_neg_full, p))


# ===========================================================================
# 15.  Main entry point
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  SPY Intraday Volatility — Random Forest')
    print('=' * 60)

    _require_dir(BASE_PLOT_DIR)
    _require_dir(DATA_DIR)
    _require_dir(GAME_CACHE_DIR)

    pnames    = list(DAY_FEATURE_NAMES)
    n_players = len(pnames)

    # ------------------------------------------------------------------
    # Decide whether raw data is available.
    # If the bar cache is missing we skip steps [1]–[2] and load
    # everything from the precomputed game result caches instead.
    # All cache-loading functions accept X_background=None when operating
    # in cache-only mode.
    # ------------------------------------------------------------------
    _has_bar_data = os.path.isfile(CACHE_PATH)

    if _has_bar_data:
        print('\n[1] Loading data ...')
        X_day, Y_day, Y_adj, diurnal_mean = load_and_aggregate()
        X_day_np = X_day.to_numpy().astype(float)

        print('\n[2] Fitting Random Forest ...')
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            X_day_np, Y_adj, test_size=0.2, random_state=RNG_SEED)
        model = RFModel()
        model.fit(X_tr, Y_tr)
        r2 = model.evaluate(X_te, Y_te)
        print('  Test R2 (trajectory-level): {:.4f}'.format(r2))
        predict_fn   = model.predict
        X_background = X_day_np
        Y_background = Y_adj
    else:
        print('\n[1-2] Bar data not found — loading all results from game caches.')
        print('      (Set CACHE_PATH in settings if you have the raw data.)')
        predict_fn   = None   # never called; all functions detect caches first
        X_background = None
        Y_background = None

    print('\n[3] Global sensitivity / risk games (double-MC) ...')
    global_effects = {}
    for gtype in ('sensitivity', 'risk'):
        print('\n  global {} ...'.format(gtype))
        global_effects[gtype] = compute_global_game(
            predict_fn   = predict_fn,
            X_background = X_background,
            Y_adj        = Y_background,
            game_type    = gtype,
            n_outer      = GLOBAL_N_OUTER,
            n_inner      = GLOBAL_N_INNER,
            seed         = RNG_SEED,
        )
        sanity_check_global(global_effects[gtype], gtype)

    print('\n[4] Selecting High-VIX Announcement profile ...')
    if _has_bar_data:
        vix_col = DAY_FEATURE_NAMES.index('vix_prev')
        vix_p75 = float(np.percentile(X_background[:, vix_col], 75))

        def find_profile(conditions):
            """Select the median day matching given feature range conditions."""
            mask = np.ones(len(X_background), dtype=bool)
            for feat, (lo, hi) in conditions.items():
                ci = DAY_FEATURE_NAMES.index(feat)
                mask &= (X_background[:, ci] >= lo) & (X_background[:, ci] <= hi)
            hits = X_background[mask]
            if len(hits) == 0:
                raise RuntimeError('No day matches: {}'.format(conditions))
            print('  {} matching days.'.format(len(hits)))
            return hits[len(hits) // 2]

        x_hv = find_profile(
            {'ann_indicator': (0.9, 1.1), 'vix_prev': (vix_p75, 999)})
        print(dict(zip(DAY_FEATURE_NAMES, x_hv)))
    else:
        # The explicand x_hv is stored in the local prediction cache;
        # compute_local_prediction will load it from there.
        x_hv = None
        print('  (skipped — will be loaded from local prediction cache)')

    print('\n[5] Local prediction game (High-VIX Announcement profile) ...')
    mob_hv, shap_hv = compute_local_prediction(
        predict_fn, X_background, x_hv, 'High_VIX_Announcement')
    moebius_hv = {'prediction': mob_hv}
    shapley_hv = {'prediction': shap_hv}

    print('\n[6] Per-instance local prediction for feature selection (top-2) ...')
    avg_local_pred = load_per_instance_pred_for_global_avg(
        predict_fn, X_background, RNG_SEED)

    print('\n[7] Per-instance prediction effects for PDP plot ({} instances) ...'.format(
        PDP_N_INSTANCES))
    per_instance_effects, X_pdp_subset = load_per_instance_effects_pdp(
        predict_fn, X_background, RNG_SEED)

    print('\n[8] Generating figures ...')

    savefig(
        plot_main_body_summary(moebius_hv, shapley_hv, pnames),
        'fig0_main_body_summary.pdf')

    savefig(
        plot_sensitivity_risk_global(
            global_sens=global_effects['sensitivity'],
            global_risk=global_effects['risk'],
            pnames=pnames,
            fs=_fs(3)),
        'fig1_sensitivity_risk_global.pdf')

    savefig(
        plot_local_prediction(moebius_hv, shapley_hv, pnames, fs=_fs(3)),
        'fig2_local_prediction.pdf')

    savefig(
        plot_global_prediction_pdp(
            per_instance_effects=per_instance_effects,
            avg_local_pred=avg_local_pred,
            X_background=X_pdp_subset,
            pnames=pnames,
            fs=_fs(3)),
        'fig3_global_prediction_pdp.pdf')

    savefig(
        plot_interactions(moebius_hv, pnames, fs=_fs(0)),
        'fig4_interactions.pdf')

    print('\n' + '=' * 60)
    print('  Done.  Figures saved to {}/'.format(BASE_PLOT_DIR))
    print('  Game caches in {}/'.format(GAME_CACHE_DIR))
    print('=' * 60)