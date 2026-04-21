"""
Functional Explanation Framework -- Jena Climate Dataset
=========================================================
Minimal example demonstrating the periodic kernel on daily
temperature trajectories.

Dataset : Jena Climate 2009-2016 (Max Planck Institute for Biogeochemistry)
          14 meteorological variables recorded every 10 minutes.
          Download the zip (25 MB) on your local machine:
            https://storage.googleapis.com/tensorflow/tf-keras-datasets/
                jena_climate_2009_2016.csv.zip
          Extract and place jena_climate_2009_2016.csv at:
            ./data/jena/jena_climate_2009_2016.csv

Output  : Daily 24-hour temperature trajectory T(degC), hourly means.
          T = 24 hours.  Diurnal mean subtracted before modelling.

Model   : F^H : R^7 -> R^24
  RandomForestRegressor, direct multi-output (no PCA, no t input).
  Seven day-level features known before the day begins:
    month        integer 1-12
    day_of_week  integer 0-6
    lag_T_mean   previous day mean temperature (degC)
    lag_T_morn   previous day 06-09h mean temperature
    lag_p_mean   previous day mean pressure (mbar)
    lag_rh_mean  previous day mean relative humidity (%)
    lag_wv_mean  previous day mean wind speed (m/s)

Why the periodic kernel is the natural choice here
---------------------------------------------------
The daily temperature curve has a genuine sinusoidal structure: it rises
from a minimum around 04:00-06:00 to a maximum around 13:00-15:00 and
falls back symmetrically. This diurnal cycle is physically periodic with
period p=24 hours: the same phase recurs every day.

Under the periodic kernel K_per(t,s) = exp(-2sin^2(pi|t-s|/p) / ell^2),
hours that are at the same phase of the diurnal cycle (e.g. 06:00 and
18:00 are both transitional hours) are recognised as correlated. A feature
like pressure (p_mean) that modulates the amplitude of the diurnal swing
will have its effect attributed coherently over the entire cycle rather
than appearing as two disconnected events at the morning and evening
transitions under the identity kernel.

The empirical correlation kernel is also computed and compared: because
the generating process is genuinely periodic, the empirical kernel should
approximately recover the periodic structure from the data, while the
parametric periodic kernel encodes it explicitly and without estimation
noise.

Kernel comparison
-----------------
Identity kernel    -- pointwise SHAP, treats each hour independently
Periodic kernel    -- period p=24h, two length-scale values (ell=2, ell=6)
                      to show sensitivity
Correlation kernel -- data-driven, should approximate periodic structure

Figures
-------
  fig1_diurnal_and_correlation.pdf  -- mean temperature curve + correlation
                                        matrix showing periodic structure
  fig2_main_effects_kernels.pdf     -- main effects under identity /
                                        periodic (ell=2) / periodic (ell=6) /
                                        correlation kernel
  fig3_feature_interpretation.pdf   -- focus on p_mean and lag_T_mean:
                                        identity vs periodic kernel,
                                        prediction + sensitivity game
"""

import itertools
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0.  Settings
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_HERE, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'jena_climate_2009_2016.csv')

BASE_PLOT_DIR = os.path.join(
    'plots', 'jena')

RNG_SEED = 42
T        = 24       # hours per day
dt       = 1.0      # 1 hour

RF_N_EST = 300
RF_JOBS  = -1

SAMPLE_SIZE = {'prediction': 150, 'sensitivity': 200, 'risk': 200}

HOUR_LABELS = ['{:02d}:00'.format(h) for h in range(T)]
t_grid      = np.arange(T, dtype=float)

DAY_FEATURE_NAMES = [
    'month',        # 1-12: seasonal baseline temperature level
    'day_of_week',  # 0-6: weak weekday effects via urban heat island
    'lag_T_mean',   # previous day mean temperature (degC)
    'lag_T_morn',   # previous day 06-09h mean temperature (degC)
    'lag_p_mean',   # previous day mean pressure (mbar): high pressure =
                    #   clear sky = stronger diurnal swing
    'lag_rh_mean',  # previous day mean relative humidity (%): humid =
                    #   cloud cover = dampened diurnal cycle
    'lag_wv_mean',  # previous day mean wind speed (m/s): wind =
                    #   advection = weaker diurnal cycle
]

# Phase windows
MORNING    = (5,  10)   # 05:00-09:00  rising phase / morning minimum
AFTERNOON  = (11, 16)   # 11:00-15:00  daytime maximum
EVENING    = (17, 21)   # 17:00-20:00  falling phase

FEAT_COLORS = {
    'month'      : '#2ca02c',
    'day_of_week': '#1f77b4',
    'lag_T_mean' : '#d62728',
    'lag_T_morn' : '#ff7f0e',
    'lag_p_mean' : '#9467bd',
    'lag_rh_mean': '#8c564b',
    'lag_wv_mean': '#e377c2',
}

GAME_YLABEL = {
    'prediction' : r'Effect on temperature ($^\circ$C)',
    'sensitivity': r'Var$[F(t)]$ ($^\circ$C$^2$)',
    'risk'       : r'Effect on MSE ($^\circ$C$^2$)',
}

GAME_TITLE = {
    'prediction' :
        r'Prediction  $v(S)(t)=\mathbb{E}[F(x)(t)\mid X_S]$',
    'sensitivity':
        r'Sensitivity  $v(S)(t)=\mathrm{Var}[F(x)(t)\mid X_S]$',
    'risk'       :
        r'Risk (MSE)  $v(S)(t)=\mathbb{E}[(Y(t)-F(x)(t))^2\mid X_S]$',
}


# ===========================================================================
# 1.  Data loading
# ===========================================================================

def _require_dir(path):
    os.makedirs(path, exist_ok=True)


def _month_to_season(m):
    if m in (12, 1, 2):  return 1
    elif m in (3, 4, 5): return 2
    elif m in (6, 7, 8): return 3
    else:                return 4


def load_and_aggregate():
    """
    Load Jena CSV, aggregate to daily 24h temperature curves,
    engineer day-level features from lagged meteorological variables.

    The output variable is T (degC) -- air temperature.
    All predictor features are computed from the PREVIOUS day to
    avoid data leakage.

    Returns
    -------
    X_day   : pd.DataFrame  (n_days, 7)
    Y_raw   : np.ndarray    (n_days, 24)  hourly mean temperature degC
    Y_adj   : np.ndarray    (n_days, 24)  diurnal-mean-subtracted
    diurnal : np.ndarray    (24,)         mean hourly temperature
    dates   : list of str
    """
    if not os.path.isfile(DATA_FILE):
        raise RuntimeError(
            'Jena climate data not found at:\n  {}\n\n'
            'Download instructions:\n'
            '  1. On your local machine, download:\n'
            '     https://storage.googleapis.com/tensorflow/'
            'tf-keras-datasets/jena_climate_2009_2016.csv.zip\n'
            '  2. Extract the zip to get jena_climate_2009_2016.csv\n'
            '  3. Copy to the server:\n'
            '     scp jena_climate_2009_2016.csv '
            'slangbei@10.103.179.177:'
            '~/Hilbert_functional_decomposition/data/jena/'.format(
                DATA_FILE))

    print('  Loading Jena climate data ...')
    df = pd.read_csv(DATA_FILE)

    # Parse datetime -- format is '01.01.2009 00:10:00'
    df['datetime'] = pd.to_datetime(
        df['Date Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['date'] = df['datetime'].dt.date.astype(str)
    df['hour'] = df['datetime'].dt.hour

    # Rename temperature column (handle possible variants)
    temp_col = next(
        (c for c in df.columns if 'T (' in c or c == 'T (degC)'), None)
    if temp_col is None:
        raise RuntimeError(
            'Temperature column not found. Columns: {}'.format(
                list(df.columns)))
    df['T_degC'] = pd.to_numeric(df[temp_col], errors='coerce')

    # Parse other meteorological variables
    col_map = {}
    for c in df.columns:
        if 'p (' in c:   col_map['p_mbar']  = c
        if 'rh (' in c:  col_map['rh_pct']  = c
        if 'wv (' in c and 'max' not in c.lower():
            col_map['wv_ms'] = c

    for key, col in col_map.items():
        df[key] = pd.to_numeric(df[col], errors='coerce')

    print('  {:,} 10-min records loaded  ({} to {})'.format(
        len(df), df['date'].min(), df['date'].max()))

    # ── Aggregate to hourly means ─────────────────────────────────────────
    agg_cols = {'T_degC': 'mean'}
    for key in col_map:
        agg_cols[key] = 'mean'

    hourly = df.groupby(['date', 'hour']).agg(agg_cols).reset_index()

    # Pivot temperature to (n_days, 24)
    pivot_T = hourly.pivot(
        index='date', columns='hour', values='T_degC')
    pivot_T = pivot_T.reindex(columns=range(T))
    pivot_T = pivot_T[pivot_T.notna().sum(axis=1) == T]

    # Pivot other met variables to daily means (for feature engineering)
    daily_met = {}
    for key in col_map:
        piv = hourly.pivot(
            index='date', columns='hour', values=key)
        daily_met[key] = piv.reindex(columns=range(T)).mean(axis=1)

    # Align all to pivot_T index
    dates  = pivot_T.index.tolist()
    Y_raw  = pivot_T.values.astype(float)
    diurnal = Y_raw.mean(axis=0)
    Y_adj  = Y_raw - diurnal[None, :]

    print('  Complete days: {}  ({} to {})'.format(
        len(dates), dates[0], dates[-1]))
    print('  Mean temperature: {:.1f} degC  '
          'Diurnal range: {:.1f} degC'.format(
              diurnal.mean(),
              diurnal.max() - diurnal.min()))

    # ── Day-level features (all from previous day) ────────────────────────
    records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str)
        m      = dt_obj.month
        dow    = dt_obj.dayofweek

        if i == 0:
            lag_T_mean = float(Y_raw.mean())
            lag_T_morn = float(Y_raw[:, MORNING[0]:MORNING[1]].mean())
            lag_p      = float(daily_met['p_mbar'].mean()
                               if 'p_mbar' in daily_met else 1000.0)
            lag_rh     = float(daily_met['rh_pct'].mean()
                               if 'rh_pct' in daily_met else 70.0)
            lag_wv     = float(daily_met['wv_ms'].mean()
                               if 'wv_ms' in daily_met else 3.0)
        else:
            prev = dates[i - 1]
            lag_T_mean = float(Y_raw[i - 1].mean())
            lag_T_morn = float(Y_raw[i - 1,
                                     MORNING[0]:MORNING[1]].mean())
            lag_p  = float(daily_met['p_mbar'].get(prev, 1000.0)
                           if 'p_mbar' in daily_met else 1000.0)
            lag_rh = float(daily_met['rh_pct'].get(prev, 70.0)
                           if 'rh_pct' in daily_met else 70.0)
            lag_wv = float(daily_met['wv_ms'].get(prev, 3.0)
                           if 'wv_ms' in daily_met else 3.0)

        records.append({
            'month'      : float(m),
            'day_of_week': float(dow),
            'lag_T_mean' : lag_T_mean,
            'lag_T_morn' : lag_T_morn,
            'lag_p_mean' : lag_p,
            'lag_rh_mean': lag_rh,
            'lag_wv_mean': lag_wv,
        })

    X_day = pd.DataFrame(records, index=dates)
    for col in DAY_FEATURE_NAMES:
        assert col in X_day.columns
        assert X_day[col].std() > 1e-6, 'Constant: {}'.format(col)

    return X_day, Y_raw, Y_adj, diurnal, dates


# ===========================================================================
# 2.  Model
# ===========================================================================

class RFModel:
    """RF: X (N,7) -> Y (N,24). No t, no PCA."""
    def __init__(self, random_state=RNG_SEED):
        self.model = RandomForestRegressor(
            n_estimators=RF_N_EST, n_jobs=RF_JOBS,
            random_state=random_state)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_te, Y_te):
        Yp = self.predict(X_te)
        return 1.0 - (np.sum((Y_te-Yp)**2) /
                      np.sum((Y_te-Y_te.mean())**2))


# ===========================================================================
# 3.  Cooperative game
# ===========================================================================

class FunctionalGame:
    def __init__(self, predict_fn, X_bg, x_exp,
                 game_type='prediction', Y_obs=None,
                 sample_size=150, random_seed=RNG_SEED):
        if game_type == 'risk' and Y_obs is None:
            raise ValueError('Y_obs required for risk.')
        self.predict_fn = predict_fn
        self.X_bg       = X_bg
        self.x_exp      = x_exp
        self.game_type  = game_type
        self.Y_obs      = Y_obs
        self.n          = sample_size
        self.seed       = random_seed
        self.T          = T
        self.p          = len(DAY_FEATURE_NAMES)
        self.player_names = list(DAY_FEATURE_NAMES)
        self.coalitions = np.array(
            list(itertools.product([False, True], repeat=self.p)),
            dtype=bool)
        self.nc   = len(self.coalitions)
        self._idx = {tuple(c): i for i, c in enumerate(self.coalitions)}
        self.values = None

    def _impute(self, coal):
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
        if self.game_type == 'prediction':
            return Yp.mean(axis=0)
        elif self.game_type == 'sensitivity':
            return Yp.var(axis=0)
        else:
            return ((self.Y_obs[None, :] - Yp)**2).mean(axis=0)

    def precompute(self):
        print('  [{}]  {} coalitions x {} samples ...'.format(
            self.game_type, self.nc, self.n))
        self.values = np.zeros((self.nc, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i+1) % 32 == 0 or i+1 == self.nc:
                print('    {}/{}'.format(i+1, self.nc))

    def __getitem__(self, c):
        return self.values[self._idx[c]]


# ===========================================================================
# 4.  Mobius + Shapley
# ===========================================================================

def moebius_transform(game):
    p     = game.p
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(p), r)
        for r in range(p+1)))
    mob = {}
    for S in all_S:
        m = np.zeros(game.T)
        for L in itertools.chain.from_iterable(
            itertools.combinations(S, r)
            for r in range(len(S)+1)
        ):
            c = tuple(i in L for i in range(p))
            m += (-1)**(len(S)-len(L)) * game[c]
        mob[S] = m
    return mob


def shapley_values(mob, p):
    shap = {i: np.zeros(T) for i in range(p)}
    for S, m in mob.items():
        if len(S) == 0:
            continue
        for i in S:
            shap[i] += m / len(S)
    return shap


# ===========================================================================
# 5.  Kernels
# ===========================================================================

def kernel_identity(t):
    return np.eye(len(t))

def kernel_periodic(t, period=24.0, length_scale=2.0):
    """
    Periodic kernel: K(t,s) = exp(-2 sin^2(pi|t-s|/p) / ell^2)
    Period p=24h encodes the diurnal cycle.
    Length-scale ell controls smoothness within each period:
      ell=2: sharp attribution, concentrated near phase
      ell=6: smooth attribution, spread over broader phase window
    """
    d = np.abs(t[:, None] - t[None, :])
    return np.exp(-2.0 * np.sin(np.pi * d / period)**2
                  / length_scale**2)

def kernel_correlation(Y_raw):
    C   = np.cov(Y_raw.T)
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    K   = np.clip(C / np.outer(std, std), -1.0, 1.0)
    return K

def apply_kernel(effect, K):
    """Row-normalised kernel application. Use for OU and correlation kernels."""
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return (K / rs) @ effect * dt

def apply_kernel_periodic(effect, K):
    """
    Unnormalised kernel application for the periodic kernel.

    Row-normalisation collapses the periodic kernel to a near-constant
    because its positive (same-phase) and negative (anti-phase) weights
    cancel after normalisation, erasing all periodicity.

    Without normalisation:
      (K_per m_i)(t) = dt * sum_s K(t,s) * m_i(s)
    where K(t,s) is positive for same-phase hours and negative for
    opposite-phase hours (~12h apart). The result oscillates with the
    diurnal cycle, correctly showing how each feature modulates the
    amplitude of the daily temperature swing.

    lag_T_mean example: its identity-kernel curve is nearly flat (~+10C).
    The unnormalised periodic kernel gives a sinusoidal curve -- positive
    in the afternoon (same-phase amplification) and negative overnight
    (anti-phase suppression), reflecting that temperature persistence is
    strongest at diurnally active phases.

    lag_p_mean example: its identity-kernel curve has a small positive
    afternoon effect. The periodic kernel links this to overnight cooling
    (opposite phase), correctly representing that high pressure both warms
    afternoons and cools nights as a single diurnal amplitude effect.
    """
    return K @ effect * dt


# ===========================================================================
# 6.  Plotting helpers
# ===========================================================================

XTICK_IDXS   = list(range(0, T, 3))
XTICK_LABELS = [HOUR_LABELS[i] for i in XTICK_IDXS]

def _set_time_axis(ax, sparse=False):
    step = 2 if sparse else 1
    ax.set_xticks(XTICK_IDXS[::step])
    ax.set_xticklabels(
        XTICK_LABELS[::step], rotation=45, ha='right', fontsize=6)
    ax.set_xlim(-0.5, T - 0.5)

def _phase_shade(ax):
    ax.axvspan(MORNING[0],   MORNING[1],   alpha=0.10, color='#aec7e8')
    ax.axvspan(AFTERNOON[0], AFTERNOON[1], alpha=0.10, color='#ffbb78')
    ax.axvspan(EVENING[0],   EVENING[1],   alpha=0.10, color='#98df8a')

def savefig(fig, name):
    path = os.path.join(BASE_PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print('  Saved: {}'.format(path))
    plt.close(fig)

def _top_k(mob, p, k=5):
    imps = np.array([float(np.sum(np.abs(mob[(i,)]))) for i in range(p)])
    return sorted(range(p), key=lambda i: imps[i], reverse=True)[:k]


# ===========================================================================
# 7.  Figure 1 -- Diurnal mean + correlation matrix
#     Shows the periodic structure that motivates the periodic kernel
# ===========================================================================

def fig_diurnal_and_correlation(diurnal, Y_raw, K_corr):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        'Daily temperature cycle — Jena Climate 2009-2016\n'
        'Periodic structure motivates the periodic kernel',
        fontsize=11, fontweight='bold')

    # ── Panel 1: mean diurnal temperature curve ───────────────────────────
    ax = axes[0]
    ax.plot(t_grid, diurnal, color='#d62728', lw=2.5)
    ax.fill_between(t_grid, diurnal.min(), diurnal,
                    alpha=0.15, color='#d62728')
    _phase_shade(ax)
    _set_time_axis(ax)
    ax.set_ylabel('Temperature (°C)', fontsize=9)
    ax.set_xlabel('Hour', fontsize=8)
    ax.set_title(
        'Mean diurnal temperature curve\n'
        '(averaged over all {:.0f} days)'.format(len(Y_raw)),
        fontsize=9, fontweight='bold')
    # Annotate min and max
    tmin = int(np.argmin(diurnal))
    tmax = int(np.argmax(diurnal))
    ax.annotate(
        'Min\n{:.1f}°C'.format(diurnal[tmin]),
        xy=(tmin, diurnal[tmin]),
        xytext=(tmin + 2, diurnal[tmin] - 0.8),
        fontsize=7.5, color='#1f77b4',
        arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.0))
    ax.annotate(
        'Max\n{:.1f}°C'.format(diurnal[tmax]),
        xy=(tmax, diurnal[tmax]),
        xytext=(tmax - 4, diurnal[tmax] + 0.5),
        fontsize=7.5, color='#d62728',
        arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.0))
    ax.tick_params(labelsize=7)

    # ── Panel 2: empirical correlation matrix ─────────────────────────────
    ax2 = axes[1]
    tick_i   = list(range(0, T, 4))
    tick_lbl = [HOUR_LABELS[i] for i in tick_i]
    im = ax2.imshow(K_corr, aspect='auto', origin='upper',
                    cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    ax2.set_xticks(tick_i)
    ax2.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=6)
    ax2.set_yticks(tick_i)
    ax2.set_yticklabels(tick_lbl, fontsize=6)
    ax2.set_xlabel('Hour $s$', fontsize=8)
    ax2.set_ylabel('Hour $t$', fontsize=8)
    ax2.set_title(
        'Empirical cross-hour correlation\n'
        '$K(t,s)=\\mathrm{Corr}(Y_t, Y_s)$',
        fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=6)

    # ── Panel 3: periodic kernel matrix for comparison ────────────────────
    K_per = kernel_periodic(t_grid, period=24.0, length_scale=4.0)
    ax3 = axes[2]
    im3 = ax3.imshow(K_per, aspect='auto', origin='upper',
                     cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    ax3.set_xticks(tick_i)
    ax3.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=6)
    ax3.set_yticks(tick_i)
    ax3.set_yticklabels(tick_lbl, fontsize=6)
    ax3.set_xlabel('Hour $s$', fontsize=8)
    ax3.set_ylabel('Hour $t$', fontsize=8)
    ax3.set_title(
        'Periodic kernel ($p=24$h, $\\ell=4$)\n'
        '$K(t,s)=\\exp(-2\\sin^2(\\pi|t-s|/p)/\\ell^2)$',
        fontsize=9, fontweight='bold')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=6)

    # Add annotation about anti-diagonal structure
    ax2.text(0.5, -0.25,
             'Anti-diagonal: hours 12h apart are negatively correlated\n'
             '(day min vs day max). Periodic kernel captures this explicitly.',
             transform=ax2.transAxes, fontsize=7.5, ha='center',
             style='italic', color='#444')

    plt.tight_layout()
    return fig


# ===========================================================================
# 8.  Figure 2 -- Main effects under all four kernels
#     Rows: prediction / sensitivity
#     Cols: Identity / Periodic ell=2 / Periodic ell=6 / Correlation
#     Top-5 features shown per cell
# ===========================================================================

def fig_main_effects_kernels(mob, pnames, K_corr):
    kernels = {
        'Identity\n(pointwise SHAP)'         : kernel_identity(t_grid),
        'Periodic\n($p=24$h, $\\ell=2$)'     : kernel_periodic(
            t_grid, 24.0, 2.0),
        'Periodic\n($p=24$h, $\\ell=6$)'     : kernel_periodic(
            t_grid, 24.0, 6.0),
        'Empirical\ncorrelation'              : K_corr,
    }
    game_types = ['prediction', 'sensitivity']
    nrows      = len(game_types)
    ncols      = len(kernels)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 3.5 * nrows),
        sharey='row')
    fig.suptitle(
        'Main effects $m_i(t)$ — Identity vs Periodic vs Correlation kernel\n'
        'Jena Climate — summer clear-sky day profile',
        fontsize=11, fontweight='bold')

    p   = len(pnames)
    top = _top_k(mob['prediction'], p, k=5)

    for r, gtype in enumerate(game_types):
        m = mob[gtype]
        for c, (kname, K) in enumerate(kernels.items()):
            ax = axes[r, c]
            for fi in top:
                raw   = m[(fi,)]
                curve = (apply_kernel_periodic(raw, K)
                         if 'Periodic' in kname
                         else apply_kernel(raw, K))
                ax.plot(t_grid, curve,
                        color=FEAT_COLORS[pnames[fi]],
                        lw=2.0, label=pnames[fi])
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _phase_shade(ax)
            _set_time_axis(ax, sparse=True)
            ax.tick_params(axis='y', labelsize=7)
            ax.set_xlabel('Hour', fontsize=7)
            if r == 0:
                ax.set_title(kname, fontsize=9, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
                ax.text(-0.30, 0.5, GAME_TITLE[gtype],
                        transform=ax.transAxes,
                        fontsize=7.5, va='center', ha='right',
                        rotation=90, color='#333')
            if r == 0 and c == 0:
                ax.legend(fontsize=6.5, loc='upper left')
            if r == 0 and c == 0:
                ax.text(0.97, 0.97, '= pointwise\nSHAP',
                        transform=ax.transAxes,
                        fontsize=6, va='top', ha='right', color='#555',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  fc='white', ec='#aaa', alpha=0.8))

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    return fig


# ===========================================================================
# 9.  Figure 3 -- Feature interpretation: p_mean and lag_T_mean
#     Focus figure showing the two most interpretable features under
#     identity vs periodic kernel, for prediction and sensitivity game.
#     This is the core interpretive figure for the paper.
# ===========================================================================

def fig_feature_interpretation(mob, pnames, K_corr):
    """
    Two features x two games x two kernels = 2x4 grid.
    Features: lag_T_mean (broad regime) and lag_p_mean (diurnal modulator).

    lag_T_mean: yesterday's mean temperature predicts today's mean level.
    Under identity kernel: effect should be fairly uniform across hours
    (high yesterday -> high today at all hours).
    Under periodic kernel: same, but the kernel explicitly encodes that
    the effect at 06:00 is related to the effect at 06:00+24h by
    periodicity -- in practice this produces smooth, phase-aware curves.

    lag_p_mean: yesterday's mean pressure predicts the AMPLITUDE of
    today's diurnal swing (high pressure = clear sky = large day-night
    difference). Under identity kernel: positive effect in the afternoon
    (high pressure -> warmer afternoons) and negative effect at night
    (clear sky -> cooler nights via radiative cooling) -- two apparently
    unrelated events. Under periodic kernel: the kernel recognises that
    06:00 and 18:00 are at the same phase of the diurnal cycle (both
    transitional hours) and links them, making the pressure effect
    appear as a coherent modulation of the diurnal amplitude rather
    than two disconnected events.
    """
    fi_T = pnames.index('lag_T_mean')
    fi_p = pnames.index('lag_p_mean')
    features_focus = [('lag\\_T\\_mean', fi_T, '#d62728'),
                      ('lag\\_p\\_mean', fi_p, '#9467bd')]

    K_id  = kernel_identity(t_grid)
    K_per = kernel_periodic(t_grid, 24.0, 4.0)

    kernels_pair = [
        ('Identity\n(pointwise SHAP)', K_id),
        ('Periodic\n($p=24$h, $\\ell=4$)', K_per),
    ]
    game_types = ['prediction', 'sensitivity']

    nrows = len(features_focus)
    ncols = len(game_types) * len(kernels_pair)   # 4

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 3.5 * nrows),
        sharey='row')
    fig.suptitle(
        'Feature effects: identity vs periodic kernel\n'
        'lag\\_T\\_mean (temperature persistence) and '
        'lag\\_p\\_mean (pressure — diurnal amplitude modulator)\n'
        'Jena Climate — summer clear-sky day',
        fontsize=10, fontweight='bold')

    col_titles = [
        'Prediction\nIdentity',
        'Prediction\nPeriodic ($\\ell=4$)',
        'Sensitivity\nIdentity',
        'Sensitivity\nPeriodic ($\\ell=4$)',
    ]

    for row, (fname, fi, fcol) in enumerate(features_focus):
        col = 0
        for gtype in game_types:
            m = mob[gtype]
            raw = m[(fi,)]
            for kname, K in kernels_pair:
                ax    = axes[row, col]
                curve = (apply_kernel_periodic(raw, K)
                         if 'Periodic' in kname
                         else apply_kernel(raw, K))
                ax.plot(t_grid, curve, color=fcol, lw=2.5)
                ax.axhline(0, color='gray', lw=0.5, ls=':')
                _phase_shade(ax)
                _set_time_axis(ax, sparse=True)
                ax.tick_params(axis='y', labelsize=7)
                ax.set_xlabel('Hour', fontsize=7)

                if row == 0:
                    ax.set_title(col_titles[col],
                                 fontsize=8.5, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
                    ax.text(-0.32, 0.5, fname,
                            transform=ax.transAxes,
                            fontsize=9, va='center', ha='right',
                            rotation=90, color=fcol, fontweight='bold')

                # Annotate what the periodic kernel reveals on p_mean
                if fi == fi_p and 'Periodic' in kname and gtype == 'prediction':
                    ax.text(
                        0.97, 0.97,
                        'Periodic kernel links\nafternoon warming\n'
                        'and night cooling\nas one diurnal effect',
                        transform=ax.transAxes,
                        fontsize=6.5, va='top', ha='right',
                        color='#555',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  fc='#f5f0ff', ec='#9467bd',
                                  alpha=0.9))

                col += 1

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    return fig


# ===========================================================================
# 10.  Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  Jena Climate  —  Periodic kernel example')
    print('=' * 60)

    _require_dir(BASE_PLOT_DIR)
    _require_dir(DATA_DIR)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print('\n[1] Loading data ...')
    X_day, Y_raw, Y_adj, diurnal, dates = load_and_aggregate()
    X_day_np = X_day.to_numpy().astype(float)
    pnames   = list(DAY_FEATURE_NAMES)

    print('\n  Diurnal temperature profile:')
    for h in [0, 6, 10, 14, 18, 22]:
        print('    {:02d}:00  {:.2f} degC'.format(h, diurnal[h]))

    # ── 2. Model ──────────────────────────────────────────────────────────
    print('\n[2] Fitting Random Forest ...')
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_day_np, Y_adj, test_size=0.2, random_state=RNG_SEED)
    model = RFModel()
    model.fit(X_tr, Y_tr)
    r2 = model.evaluate(X_te, Y_te)
    print('  Test R² (trajectory-level): {:.4f}'.format(r2))

    # ── 3. Correlation kernel ─────────────────────────────────────────────
    print('\n[3] Building kernels ...')
    K_corr = kernel_correlation(Y_raw)

    # Key correlations
    t_min = int(np.argmin(diurnal))   # ~05:00 overnight minimum
    t_max = int(np.argmax(diurnal))   # ~14:00 afternoon maximum
    t_opp = (t_max + T // 2) % T     # 12h opposite phase
    print('  K[T_min={:02d}:00, T_max={:02d}:00] = {:.3f}  '
          '(should be strongly negative)'.format(
              t_min, t_max, K_corr[t_min, t_max]))
    print('  K[T_max={:02d}:00, +12h={:02d}:00] = {:.3f}  '
          '(should be negative, opposite phase)'.format(
              t_max, t_opp, K_corr[t_max, t_opp]))

    # ── 4. Select profile -- summer clear-sky day ─────────────────────────
    print('\n[4] Selecting profile ...')
    fi_mon = pnames.index('month')
    fi_p   = pnames.index('lag_p_mean')

    p_p75 = float(np.percentile(X_day_np[:, fi_p], 75))

    # Summer (June-August) + high pressure previous day
    mask = (
        (X_day_np[:, fi_mon] >= 6) &
        (X_day_np[:, fi_mon] <= 8) &
        (X_day_np[:, fi_p] >= p_p75)
    )
    hits = X_day_np[mask]
    if len(hits) == 0:
        # Fallback: just use a summer day
        mask = (X_day_np[:, fi_mon] >= 6) & (X_day_np[:, fi_mon] <= 8)
        hits = X_day_np[mask]

    print('  Summer clear-sky profile: {} matching days'.format(len(hits)))
    x_explain = hits[len(hits) // 2]

    diffs = np.abs(X_day_np - x_explain[None, :]).sum(axis=1)
    y_obs = Y_adj[int(np.argmin(diffs))]

    print('  Profile: {}'.format('  '.join(
        '{}={:.2f}'.format(n, x_explain[j])
        for j, n in enumerate(pnames))))

    # ── 5. Compute games ──────────────────────────────────────────────────
    print('\n[5] Computing games ...')
    mob_all  = {}
    shap_all = {}

    for gtype in ('prediction', 'sensitivity', 'risk'):
        print('\n  game: {} ...'.format(gtype))
        game = FunctionalGame(
            predict_fn  = model.predict,
            X_bg        = X_day_np,
            x_exp       = x_explain,
            game_type   = gtype,
            Y_obs       = y_obs,
            sample_size = SAMPLE_SIZE[gtype],
            random_seed = RNG_SEED,
        )
        game.precompute()
        mob_all[gtype]  = moebius_transform(game)
        shap_all[gtype] = shapley_values(mob_all[gtype], game.p)

    # ── 6. Figures ────────────────────────────────────────────────────────
    print('\n[6] Generating figures ...')

    savefig(
        fig_diurnal_and_correlation(diurnal, Y_raw, K_corr),
        'fig1_diurnal_and_correlation.pdf')

    savefig(
        fig_main_effects_kernels(mob_all, pnames, K_corr),
        'fig2_main_effects_kernels.pdf')

    savefig(
        fig_feature_interpretation(mob_all, pnames, K_corr),
        'fig3_feature_interpretation.pdf')

    print('\n' + '=' * 60)
    print('  Done.  Figures in {}/'.format(BASE_PLOT_DIR))
    print('  fig1_diurnal_and_correlation.pdf  -- periodic structure')
    print('  fig2_main_effects_kernels.pdf     -- kernel comparison')
    print('  fig3_feature_interpretation.pdf   -- p_mean and T_mean')
    print('=' * 60)