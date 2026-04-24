"""
Functional Explanation Framework -- Beijing PM2.5 Air Quality
=============================================================
Dataset : UCI Beijing PM2.5 (id=381), US Embassy station, 2010-2014.
          Fetched automatically via ucimlrepo (cached as parquet).

Output  : F^H : R^8 -> R^24
  Daily 24-hour log(PM2.5+1) trajectory, hourly means, T=24.
  Log-transform applied before modelling. Diurnal mean subtracted.
  Model: RandomForestRegressor, direct multi-output, no PCA, no t input.

Features (cooperative game players, all from previous day):
  month            integer 1-12
  is_heating       binary 1 if Nov-Mar (Beijing coal heating season)
  day_of_week      integer 0-6
  lag_pm25_mean    previous day mean log(PM2.5+1)
  lag_pm25_morning previous day 06-09h mean
  wind_sin         sin(prev day wind direction)
  wind_cos         cos(prev day wind direction)
  wspm_prev        previous day mean cumulated wind speed

Central storyline: the heating season ONSET day
------------------------------------------------
The heating onset day (first November heating day) is the most
theoretically interesting profile. On this day the features carry
conflicting temporal signals:

  lag_pm25_mean is LOW  (yesterday was pre-heating, clean air)
  is_heating    is 1    (today heating has switched on city-wide)
  wspm_prev     is moderate-high (cold front that triggered heating)

Under the identity kernel (pointwise SHAP), lag_pm25_mean shows
a SIGN REVERSAL across the day:
  - POSITIVE overnight: the fresh heating emissions accumulate
    under the nocturnal inversion regardless of clean yesterday
  - NEGATIVE in the afternoon: the clean-yesterday baseline
    correctly predicts lower afternoon PM2.5 after mixing layer rise

This sign reversal demonstrates that feature effects can be genuinely
time-heterogeneous -- the same feature has a positive effect overnight
and a negative effect in the afternoon on the same day.

Why the correlation kernel is inappropriate here
------------------------------------------------
The empirical correlation kernel is estimated from all ~1700 days,
dominated by persistent pollution regime days where the whole
trajectory moves together. Applying it to the onset day imposes a
uniform positive weighting, erasing the sign reversal entirely.

Kernel spectrum (Section 4.4)
-----------------------------
  Identity     -- maximum temporal resolution, reveals sign reversal
  OU / Gaussian -- local smoothing; preserves sign reversal, reduces noise
  Causal       -- restricts is_heating attribution to overnight/morning
  Correlation  -- appropriate for regime days (NESO); wrong here

Figures
-------
  fig0_main_body.pdf             -- MAIN BODY: 1x3 compact
  fig1_diurnal_and_correlation.pdf
  figA1_ppf_identity_onset.pdf   -- APPENDIX: pure/partial/full, id kernel
  figA2_kernel_all_features.pdf  -- APPENDIX: 5 kernels x top features
  figA4_network_all_games.pdf    -- APPENDIX: network plots, id kernel
"""

import itertools
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0.  Settings
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_HERE, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'beijing_pm25_embassy.parquet')

BASE_PLOT_DIR = os.path.join('plots', 'beijing_all_games')

RNG_SEED = 42
T        = 24
dt       = 1.0

RF_N_EST = 300
RF_JOBS  = -1

SAMPLE_SIZE = {'prediction': 150, 'sensitivity': 200, 'risk': 200}

HOUR_LABELS = ['{:02d}:00'.format(h) for h in range(T)]
t_grid      = np.arange(T, dtype=float)

HEATING_MONTHS = {11, 12, 1, 2, 3}

DAY_FEATURE_NAMES = [
    'month',
    'is_heating',
    'day_of_week',
    'lag_pm25_mean',
    'lag_pm25_morning',
    'wind_sin',
    'wind_cos',
    'wspm_prev',
]

OVERNIGHT  = (0,  6)
MORNING    = (6,  10)
AFTERNOON  = (11, 16)
EVENING    = (17, 21)

FEAT_COLORS = {
    'month'            : '#2ca02c',
    'is_heating'       : '#ff7f0e',
    'day_of_week'      : '#1f77b4',
    'lag_pm25_mean'    : '#d62728',
    'lag_pm25_morning' : '#9467bd',
    'wind_sin'         : '#8c564b',
    'wind_cos'         : '#e377c2',
    'wspm_prev'        : '#17becf',
}

# 3-letter abbreviations for network plots
FEAT_ABBR_BJ = {
    'month'            : 'Mon',
    'is_heating'       : 'Htg',
    'day_of_week'      : 'DoW',
    'lag_pm25_mean'    : 'LPM',
    'lag_pm25_morning' : 'LMo',
    'wind_sin'         : 'WSn',
    'wind_cos'         : 'WCs',
    'wspm_prev'        : 'Wsp',
}

GAME_YLABEL = {
    'prediction' : r'Effect on log(PM2.5+1)',
    'sensitivity': r'Var$[F(t)]$',
    'risk'       : r'Effect on MSE',
}

GAME_TITLE = {
    'prediction' :
        r'Prediction  $v(S)(t)=\mathbb{E}[F(x)(t)\mid X_S]$',
    'sensitivity':
        r'Sensitivity  $v(S)(t)=\mathrm{Var}[F(x)(t)\mid X_S]$',
    'risk'       :
        r'Risk (MSE)  $v(S)(t)=\mathbb{E}[(Y(t)-F(x)(t))^2\mid X_S]$',
}

# XAI labels for pure/partial/full
_XAI_LABELS_BJ = {
    ('prediction',  'pure')   : 'Pure  $m_i \\equiv$ PDP',
    ('prediction',  'partial'): 'Partial  $\\phi_i \\equiv$ SHAP',
    ('prediction',  'full')   : 'Full  $\\Phi_i \\equiv$ ICE-agg.',
    ('sensitivity', 'pure')   : 'Pure  $\\equiv$ Closed Sobol $\\tau^{\\mathrm{cl}}_i$',
    ('sensitivity', 'partial'): 'Partial  $\\equiv$ Shapley-sensitivity',
    ('sensitivity', 'full')   : 'Full  $\\equiv$ Total Sobol $\\bar{\\tau}_i$',
    ('risk',        'pure')   : 'Pure  $\\equiv$ Pure Risk',
    ('risk',        'partial'): 'Partial  $\\equiv$ SAGE',
    ('risk',        'full')   : 'Full  $\\equiv$ PFI',
}

_EFFECT_TYPES = ['pure', 'partial', 'full']

# Network node/edge colours (same convention as energy)
_NODE_POS = '#2a9d8f'
_NODE_NEG = '#e63946'
_EDGE_SYN = '#2a9d8f'
_EDGE_RED = '#e63946'

# Font sizes
FS_SUPTITLE = 12
FS_TITLE    = 10
FS_AXIS     = 9
FS_TICK     = 8
FS_LEGEND   = 8
FS_ANNOT    = 8


# ===========================================================================
# 1.  Data loading
# ===========================================================================

def _require_dir(path):
    os.makedirs(path, exist_ok=True)


def _wind_dir_to_radians(wd_series):
    dir_map = {
        'N': 0.0, 'NNE': np.pi/8, 'NE': np.pi/4,
        'ENE': 3*np.pi/8, 'E': np.pi/2, 'ESE': 5*np.pi/8,
        'SE': 3*np.pi/4, 'SSE': 7*np.pi/8, 'S': np.pi,
        'SSW': 9*np.pi/8, 'SW': 5*np.pi/4, 'WSW': 11*np.pi/8,
        'W': 3*np.pi/2, 'WNW': 13*np.pi/8, 'NW': 7*np.pi/4,
        'NNW': 15*np.pi/8,
    }
    rads = wd_series.map(dir_map).fillna(0.0)
    return np.sin(rads), np.cos(rads)


def _fetch_and_cache():
    import importlib
    if importlib.util.find_spec('ucimlrepo') is None:
        raise RuntimeError('pip install ucimlrepo')
    from ucimlrepo import fetch_ucirepo
    print('  Fetching Beijing PM2.5 US Embassy (id=381) ...')
    ds  = fetch_ucirepo(id=381)
    raw = ds.data.features.copy()
    rename = {'pm2.5': 'PM2.5', 'cbwd': 'wd', 'Iws': 'WSPM',
              'TEMP': 'TEMP', 'PRES': 'PRES', 'DEWP': 'DEWP'}
    raw = raw.rename(columns={k: v for k, v in rename.items()
                               if k in raw.columns})
    skip = {'wd'}
    for col in raw.columns:
        if col not in skip:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')
    if 'WSPM' in raw.columns:
        raw['WSPM'] = raw['WSPM'].clip(lower=0)
    _require_dir(DATA_DIR)
    raw.to_parquet(DATA_FILE, index=False)
    return raw


def load_and_aggregate():
    if os.path.isfile(DATA_FILE):
        print('  Loading parquet cache ...')
        df = pd.read_parquet(DATA_FILE)
    else:
        df = _fetch_and_cache()

    df['datetime'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour']], errors='coerce')
    df = df.dropna(subset=['datetime', 'PM2.5'])
    df['date'] = df['datetime'].dt.date.astype(str)
    df['hour'] = df['datetime'].dt.hour
    df['logpm25'] = np.log1p(df['PM2.5'].clip(lower=0))

    agg = df.groupby(['date', 'hour']).agg(
        logpm25=('logpm25', 'mean'),
        WSPM   =('WSPM',    'mean'),
        wd     =('wd',      lambda x: x.mode()[0] if len(x) > 0 else np.nan),
    ).reset_index()

    pivot = agg.pivot(index='date', columns='hour', values='logpm25')
    pivot = pivot.reindex(columns=range(T))
    pivot = pivot[pivot.notna().sum(axis=1) >= 20]
    pivot = pivot.fillna(pivot.mean())

    Y_raw   = pivot.values.astype(float)
    dates   = pivot.index.tolist()
    diurnal = Y_raw.mean(axis=0)
    Y_adj   = Y_raw - diurnal[None, :]

    print('  Complete days: {}  ({} to {})'.format(
        len(dates), dates[0], dates[-1]))

    daily_wspm = agg.groupby('date')['WSPM'].mean()
    daily_wd   = agg.groupby('date')['wd'].agg(
        lambda x: x.mode()[0] if len(x) > 0 else 'N')

    records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str)
        m, dow = dt_obj.month, dt_obj.dayofweek
        if i == 0:
            lag_mean = float(Y_raw.mean())
            lag_morn = float(Y_raw[:, MORNING[0]:MORNING[1]].mean())
            lag_wspm = float(daily_wspm.mean())
            lag_wd   = 'N'
        else:
            prev     = dates[i - 1]
            lag_mean = float(Y_raw[i - 1].mean())
            lag_morn = float(Y_raw[i - 1, MORNING[0]:MORNING[1]].mean())
            lag_wspm = float(daily_wspm.get(prev, daily_wspm.mean()))
            lag_wd   = daily_wd.get(prev, 'N')
        w_sin_val, w_cos_val = _wind_dir_to_radians(pd.Series([lag_wd]))
        records.append({
            'month'            : float(m),
            'is_heating'       : float(m in HEATING_MONTHS),
            'day_of_week'      : float(dow),
            'lag_pm25_mean'    : lag_mean,
            'lag_pm25_morning' : lag_morn,
            'wind_sin'         : float(w_sin_val.iloc[0]),
            'wind_cos'         : float(w_cos_val.iloc[0]),
            'wspm_prev'        : lag_wspm,
        })

    X_day = pd.DataFrame(records, index=dates)
    print('  Heating season days: {} ({:.1f}%)'.format(
        int(X_day['is_heating'].sum()),
        X_day['is_heating'].mean() * 100))
    return X_day, Y_raw, Y_adj, diurnal, dates


# ===========================================================================
# 2.  Model
# ===========================================================================

class RFModel:
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
        return 1.0 - (np.sum((Y_te - Yp)**2) /
                      np.sum((Y_te - Y_te.mean())**2))


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
            if (i + 1) % 64 == 0 or i + 1 == self.nc:
                print('    {}/{}'.format(i + 1, self.nc))

    def __getitem__(self, c):
        return self.values[self._idx[c]]


# ===========================================================================
# 4.  Möbius + Shapley
# ===========================================================================

def moebius_transform(game):
    p     = game.p
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(p), r)
        for r in range(p + 1)))
    mob = {}
    for S in all_S:
        m = np.zeros(game.T)
        for L in itertools.chain.from_iterable(
            itertools.combinations(S, r)
            for r in range(len(S) + 1)
        ):
            c = tuple(i in L for i in range(p))
            m += (-1)**(len(S) - len(L)) * game[c]
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

def kernel_ou(t, length_scale=4.0):
    return np.exp(-np.abs(t[:, None] - t[None, :]) / length_scale)

def kernel_gaussian(t, length_scale=4.0):
    d2 = (t[:, None] - t[None, :])**2
    return np.exp(-d2 / (2.0 * length_scale**2))

def kernel_causal(t, length_scale=4.0):
    d = t[:, None] - t[None, :]
    return np.where(d >= 0, np.exp(-d / length_scale), 0.0)

def kernel_correlation(Y_raw):
    C   = np.cov(Y_raw.T)
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    K   = np.clip(C / np.outer(std, std), -1.0, 1.0)
    return K

def apply_kernel(effect, K):
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return (K / rs) @ effect * dt


# ===========================================================================
# 6.  Pure / partial / full helpers
# ===========================================================================

def _pure_effects(mob, p):
    return {i: mob.get((i,), np.zeros(T)).copy() for i in range(p)}

def _full_effects(mob, p):
    full = {i: np.zeros(T) for i in range(p)}
    for S, m in mob.items():
        if len(S) == 0:
            continue
        for i in S:
            full[i] += m
    return full


# ===========================================================================
# 7.  Plotting helpers
# ===========================================================================

XTICK_IDXS   = list(range(0, T, 3))
XTICK_LABELS = [HOUR_LABELS[i] for i in XTICK_IDXS]

def _set_time_axis(ax, sparse=False):
    step = 2 if sparse else 1
    ax.set_xticks(XTICK_IDXS[::step])
    ax.set_xticklabels(XTICK_LABELS[::step],
                       rotation=45, ha='right', fontsize=FS_TICK - 1)
    ax.set_xlim(-0.5, T - 0.5)

def _phase_shade(ax):
    ax.axvspan(*OVERNIGHT,  alpha=0.10, color='#555555', zorder=0)
    ax.axvspan(*MORNING,    alpha=0.10, color='#4a90e2', zorder=0)
    ax.axvspan(*AFTERNOON,  alpha=0.10, color='#ffbb78', zorder=0)
    ax.axvspan(*EVENING,    alpha=0.10, color='#e24a4a', zorder=0)

def _top_k(mob, p, k=5):
    imps = np.array([float(np.sum(np.abs(mob[(i,)]))) for i in range(p)])
    return sorted(range(p), key=lambda i: imps[i], reverse=True)[:k]

def savefig(fig, name):
    path = os.path.join(BASE_PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print('  Saved: {}'.format(path))
    plt.close(fig)


# ===========================================================================
# 8.  Network draw helper
# ===========================================================================

def _draw_network_bj(ax, pnames, node_imp, edge_imp, node_sign, title):
    """
    Single network panel. Circular layout, clockwise from top.
    Node colour: teal=positive, red=negative effect.
    Inner white circle with 3-letter abbreviation.
    Edge colour: teal=synergy, red=redundancy.
    """
    import math
    p     = len(pnames)
    angle = [math.pi / 2 - 2 * math.pi * i / p for i in range(p)]
    pos   = {i: (math.cos(a), math.sin(a)) for i, a in enumerate(angle)}

    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight='bold')

    max_imp  = float(node_imp.max()) if node_imp.max() > 0 else 1.0
    node_r   = {i: 0.07 + 0.19 * (node_imp[i] / max_imp) for i in range(p)}
    max_edge = max((abs(v) for v in edge_imp.values()), default=1.0)
    max_edge = max_edge if max_edge > 0 else 1.0

    for (i, j), val in edge_imp.items():
        xi, yi = pos[i]; xj, yj = pos[j]
        lw   = 0.4 + 6.5 * abs(val) / max_edge
        col  = _EDGE_SYN if val > 0 else _EDGE_RED
        alph = 0.30 + 0.60 * abs(val) / max_edge
        ax.plot([xi, xj], [yi, yj], color=col, lw=lw, alpha=alph,
                solid_capstyle='round', zorder=1)

    for i in range(p):
        x, y = pos[i]
        r    = node_r[i]
        fc   = _NODE_POS if node_sign[i] >= 0 else _NODE_NEG
        circle = plt.Circle((x, y), r, color=fc, ec='white',
                             linewidth=1.2, zorder=2, alpha=0.88)
        ax.add_patch(circle)
        r_inner = r * 0.52
        inner   = plt.Circle((x, y), r_inner, color='white', ec='none',
                              zorder=3, alpha=0.95)
        ax.add_patch(inner)
        abbr = FEAT_ABBR_BJ.get(pnames[i], pnames[i][:3])
        ax.text(x, y, abbr,
                ha='center', va='center',
                fontsize=max(4.5, r * 22),
                fontweight='bold', color='#222', zorder=4)

    # tight limits matching doc38: nodes on unit circle, max node_r≈0.26
    pad = 0.32
    ax.set_xlim(-1.0 - pad, 1.0 + pad)
    ax.set_ylim(-1.0 - pad, 1.0 + pad)


def _network_importances_bj(mob, shap, p, K, effect_type):
    pure_eff = _pure_effects(mob, p)
    full_eff = _full_effects(mob, p)
    if effect_type == 'pure':
        eff = pure_eff
    elif effect_type == 'partial':
        eff = shap
    else:
        eff = full_eff

    node_imp  = np.array([
        float(np.sum(np.abs(apply_kernel(eff[i], K)))) for i in range(p)])
    node_sign = np.array([
        np.sign(float(np.trapz(apply_kernel(eff[i], K), t_grid)))
        for i in range(p)])
    edge_imp = {}
    for i in range(p):
        for j in range(i + 1, p):
            raw = mob.get((i, j), np.zeros(T))
            val = float(np.trapz(apply_kernel(raw, K), t_grid))
            if abs(val) > 0:
                edge_imp[(i, j)] = val
    return node_imp, edge_imp, node_sign


# ===========================================================================
# 9.  Figure 0 -- Main body (1 row x 3 cols)
#
#   Col 0: mean diurnal PM2.5 + correlation heatmap inset
#   Col 1: onset day SHAP prediction (identity kernel), top-4 features,
#           sign reversal annotated
#   Col 2: kernel comparison for lag_pm25_mean, prediction game,
#           identity / OU / correlation on one set of axes
# ===========================================================================

def fig0_main_body(diurnal, Y_raw, K_corr, mob_onset, shap_onset,
                   pnames, prof_results=None):
    p      = len(pnames)
    K_id   = kernel_identity(t_grid)
    K_ou   = kernel_ou(t_grid, 4.0)
    fi_lag = pnames.index('lag_pm25_mean')

    fig, axes = plt.subplots(
        1, 3,
        figsize=(26, 3.6),
        gridspec_kw={'wspace': 0.30, 'width_ratios': [1.1, 1.3, 1.3]})

    fig.suptitle(
        'Beijing PM2.5 — heating onset day: time-heterogeneous feature effects '
        'and kernel choice',
        fontsize=FS_SUPTITLE, fontweight='bold', y=1.04)

    # ── Col 0: diurnal curve + inset heatmap ─────────────────────────────
    ax = axes[0]
    pm25_diurnal = np.expm1(diurnal)
    ax.plot(t_grid, pm25_diurnal, color='#8B0000', lw=2.5)
    ax.fill_between(t_grid, 0, pm25_diurnal, alpha=0.15, color='#8B0000')
    _phase_shade(ax)
    _set_time_axis(ax)
    ax.set_ylabel(r'Mean PM2.5 ($\mu$g/m$^3$)', fontsize=FS_AXIS)
    ax.set_xlabel('Hour', fontsize=FS_AXIS)
    ax.set_title('Mean diurnal PM2.5\n({} days)'.format(len(Y_raw)),
                 fontsize=FS_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FS_TICK)

    # Inset: correlation heatmap — bottom-left, slightly smaller
    ax_ins = ax.inset_axes([0.01, 0.01, 0.38, 0.44])
    tick_i = list(range(0, T, 6))
    im = ax_ins.imshow(K_corr, aspect='auto', origin='upper',
                       cmap='RdBu_r', vmin=-0.2, vmax=1.0)
    ax_ins.set_xticks(tick_i)
    ax_ins.set_xticklabels([HOUR_LABELS[i] for i in tick_i],
                           rotation=45, ha='right', fontsize=3.5)
    ax_ins.set_yticks(tick_i)
    ax_ins.set_yticklabels([HOUR_LABELS[i] for i in tick_i], fontsize=3.5)
    ax_ins.set_title('Corr. kernel $K(t,s)$', fontsize=5.0,
                     fontweight='bold', pad=2)
    plt.colorbar(im, ax=ax_ins, fraction=0.08, pad=0.02).ax.tick_params(
        labelsize=3.5)

    # ── Col 1: Shapley values (partial effects), identity kernel ─────────
    ax = axes[1]
    fi_heat = pnames.index('is_heating')

    imps = {i: float(np.sum(np.abs(shap_onset['prediction'][i])))
            for i in range(p)}
    top4 = sorted(imps, key=imps.get, reverse=True)[:4]
    if fi_heat not in top4:
        top4 = sorted(imps, key=imps.get, reverse=True)[:3] + [fi_heat]

    for fi in top4:
        ax.plot(t_grid, shap_onset['prediction'][fi],
                color=FEAT_COLORS[pnames[fi]], lw=2.2,
                label=pnames[fi])

    ax.axhline(0, color='gray', lw=0.5, ls=':')
    _phase_shade(ax)
    _set_time_axis(ax)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=FS_AXIS)
    ax.set_xlabel('Hour', fontsize=FS_AXIS)
    ax.set_title('Onset day — Shapley $\\phi_i(t)$  (identity kernel)\n'
                 'Prediction game',
                 fontsize=FS_TITLE, fontweight='bold')
    ax.legend(fontsize=FS_LEGEND - 0.5, loc='upper right', ncol=2)

    # ── Col 2: Shapley values for lag_pm25_mean under 3 kernels ──────────
    ax = axes[2]
    shap_lag = shap_onset['prediction'][fi_lag]
    kernels_3 = [
        ('Identity (unweighted)', K_id,    '#333333', '-'),
        ('OU $\\ell=4$h',         K_ou,    '#1f77b4', '--'),
        ('Correlation (erases reversal)', K_corr, '#2a9d8f', '-.'),
    ]
    for kname, K, kcol, ls in kernels_3:
        ax.plot(t_grid, apply_kernel(shap_lag, K),
                color=kcol, lw=2.2, ls=ls, label=kname)

    ax.axhline(0, color='gray', lw=0.5, ls=':')
    _phase_shade(ax)
    _set_time_axis(ax)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=FS_AXIS)
    ax.set_xlabel('Hour', fontsize=FS_AXIS)
    ax.set_title('lag_pm25_mean — Shapley $\\phi_i(t)$ under 3 kernels\n'
                 'Onset day, prediction game',
                 fontsize=FS_TITLE, fontweight='bold')
    ax.legend(fontsize=FS_LEGEND - 0.5, loc='lower left')

    # Shared y-axis cols 1 and 2
    y1_min, y1_max = axes[1].get_ylim()
    y2_min, y2_max = axes[2].get_ylim()
    axes[1].set_ylim(min(y1_min, y2_min), max(y1_max, y2_max))
    axes[2].set_ylim(min(y1_min, y2_min), max(y1_max, y2_max))

    plt.tight_layout()
    return fig


# ===========================================================================
# 10. Figure 1 -- Diurnal + correlation (unchanged)
# ===========================================================================

def fig_diurnal_and_correlation(diurnal, Y_raw, K_corr):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        'Daily PM2.5 cycle — Beijing US Embassy station 2010-2014\n'
        'Overnight/morning stagnation cluster motivates correlation structure',
        fontsize=FS_SUPTITLE, fontweight='bold')

    ax = axes[0]
    pm25_diurnal = np.expm1(diurnal)
    ax.plot(t_grid, pm25_diurnal, color='#8B0000', lw=2.5)
    ax.fill_between(t_grid, 0, pm25_diurnal, alpha=0.15, color='#8B0000')
    _phase_shade(ax)
    _set_time_axis(ax)
    ax.set_ylabel(r'Mean PM2.5 ($\mu$g/m$^3$)', fontsize=FS_AXIS)
    ax.set_xlabel('Hour', fontsize=FS_AXIS)
    ax.set_title('Mean diurnal PM2.5\n({} days)'.format(len(Y_raw)),
                 fontsize=FS_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FS_TICK)

    ax2 = axes[1]
    tick_i   = list(range(0, T, 4))
    tick_lbl = [HOUR_LABELS[i] for i in tick_i]
    im = ax2.imshow(K_corr, aspect='auto', origin='upper',
                    cmap='RdBu_r', vmin=-0.2, vmax=1.0)
    ax2.set_xticks(tick_i); ax2.set_xticklabels(tick_lbl, rotation=45,
                                                 ha='right', fontsize=6)
    ax2.set_yticks(tick_i); ax2.set_yticklabels(tick_lbl, fontsize=6)
    ax2.set_xlabel('Hour $s$', fontsize=FS_AXIS)
    ax2.set_ylabel('Hour $t$', fontsize=FS_AXIS)
    ax2.set_title('Empirical cross-hour correlation\n'
                  '$K(t,s)=\\mathrm{Corr}(Y_t, Y_s)$',
                  fontsize=FS_TITLE, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=6)

    ax3 = axes[2]
    ov_mid = (OVERNIGHT[0] + OVERNIGHT[1]) // 2
    af_mid = (AFTERNOON[0] + AFTERNOON[1]) // 2
    mo_mid = (MORNING[0]   + MORNING[1])   // 2
    ax3.plot(t_grid, K_corr[ov_mid, :], color='#555555', lw=2.2,
             label='Row $t={}$ (overnight)'.format(HOUR_LABELS[ov_mid]))
    ax3.plot(t_grid, K_corr[af_mid, :], color='#ffbb78', lw=2.2, ls='--',
             label='Row $t={}$ (afternoon)'.format(HOUR_LABELS[af_mid]))
    ax3.axhline(0, color='gray', lw=0.5, ls=':')
    _phase_shade(ax3)
    _set_time_axis(ax3)
    ax3.set_xlabel('Hour $s$', fontsize=FS_AXIS)
    ax3.set_ylabel('$K(t,\\ s)$', fontsize=FS_AXIS)
    ax3.set_title('Kernel row slices\nOvernight vs afternoon',
                  fontsize=FS_TITLE, fontweight='bold')
    ax3.legend(fontsize=FS_LEGEND)
    ax3.tick_params(labelsize=FS_TICK)

    plt.tight_layout()
    return fig



# ===========================================================================
# 11. Figure A1 -- APPENDIX: pure / partial / full, identity kernel, onset day
#     3 rows (games) x 4 cols (pure / partial / full / importance bars)
# ===========================================================================

def fig_A1_ppf_identity(mob_onset, shap_onset, pnames):
    K_id = kernel_identity(t_grid)
    p    = len(pnames)

    fig, axes = plt.subplots(
        3, 4, figsize=(18, 12),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Pure / Partial / Full effects — Identity kernel — onset day\n'
        'Beijing US Embassy — heating season onset (November)',
        fontsize=FS_SUPTITLE, fontweight='bold')

    for r, gtype in enumerate(['prediction', 'sensitivity', 'risk']):
        mob  = mob_onset[gtype]
        shap = shap_onset[gtype]

        pure_eff = _pure_effects(mob, p)
        part_eff = shap
        full_eff = _full_effects(mob, p)
        effect_dicts = {'pure': pure_eff, 'partial': part_eff, 'full': full_eff}

        imps_partial = {
            i: float(np.sum(np.abs(apply_kernel(part_eff[i], K_id))))
            for i in range(p)}
        top5 = sorted(imps_partial, key=imps_partial.get, reverse=True)[:5]

        # ── Curve panels (cols 0-2) ───────────────────────────────────────
        col0_handles, col0_labels = [], []
        for c, etype in enumerate(_EFFECT_TYPES):
            ax  = axes[r, c]
            eff = effect_dicts[etype]
            lines = []
            for fi in top5:
                curve = apply_kernel(eff[fi], K_id)
                ln, = ax.plot(t_grid, curve,
                              color=FEAT_COLORS[pnames[fi]], lw=2.0,
                              label=pnames[fi])
                lines.append(ln)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _phase_shade(ax)
            _set_time_axis(ax)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Hour', fontsize=FS_AXIS)
            ax.set_title(_XAI_LABELS_BJ[(gtype, etype)],
                         fontsize=FS_TITLE, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=FS_AXIS)
                col0_handles = lines
                col0_labels  = [pnames[fi] for fi in top5]
                if r == 0:
                    ax.legend(handles=col0_handles, labels=col0_labels,
                              fontsize=FS_LEGEND - 0.5,
                              loc='upper right', framealpha=0.85)
                elif r == 2:
                    ax.legend(handles=col0_handles, labels=col0_labels,
                              fontsize=FS_LEGEND - 0.5,
                              loc='lower right', framealpha=0.85)
            if c == 1 and r == 1:
                ax.legend(handles=col0_handles, labels=col0_labels,
                          fontsize=FS_LEGEND - 0.5,
                          loc='lower right', framealpha=0.85)

        # ── Importance bars (col 3) ───────────────────────────────────────
        ax_bar = axes[r, 3]
        imps_all = {
            etype: {
                i: float(np.sum(np.abs(apply_kernel(effect_dicts[etype][i], K_id))))
                for i in range(p)}
            for etype in _EFFECT_TYPES}
        order = sorted(range(p),
                       key=lambda i: imps_all['partial'][i], reverse=True)
        y_pos   = np.arange(len(order))
        bar_h   = 0.25
        offsets = {'pure': -bar_h, 'partial': 0.0, 'full': bar_h}
        alphas  = {'pure': 0.55,   'partial': 0.90, 'full': 0.55}
        hatches = {'pure': '//',   'partial': '',   'full': '\\\\'}
        for etype in _EFFECT_TYPES:
            vals = [imps_all[etype][i] for i in order]
            ax_bar.barh(
                y_pos + offsets[etype], vals, height=bar_h,
                color=[FEAT_COLORS[pnames[i]] for i in order],
                alpha=alphas[etype], hatch=hatches[etype], label=etype)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([pnames[i] for i in order], fontsize=FS_TICK)
        ax_bar.axvline(0, color='gray', lw=0.8, ls=':')
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=FS_TICK)
        ax_bar.set_title('Integrated\nimportance\n(identity)',
                         fontsize=FS_TITLE, fontweight='bold')
        ax_bar.legend(fontsize=FS_LEGEND, loc='upper right')

    plt.tight_layout()
    return fig


# ===========================================================================
# 15. Figure A2 -- APPENDIX: all kernels x top features, prediction game
#     Rows: top-5 features  Cols: 5 kernels
# ===========================================================================

def fig_A2_kernel_all_features(mob_onset, pnames, K_corr):
    p = len(pnames)
    top5 = _top_k(mob_onset['prediction'], p, k=5)
    fi_lag = pnames.index('lag_pm25_mean')
    if fi_lag not in top5:
        top5 = _top_k(mob_onset['prediction'], p, k=4) + [fi_lag]

    kernels_ordered = [
        ('Identity',                    kernel_identity(t_grid),      '#333333'),
        ('OU  $\\ell=4$h',             kernel_ou(t_grid, 4.0),       '#1f77b4'),
        ('Gaussian  $\\ell=4$h',       kernel_gaussian(t_grid, 4.0), '#2ca02c'),
        ('Causal  $\\ell=4$h',         kernel_causal(t_grid, 4.0),   '#e76f51'),
        ('Correlation\n(negative ex.)', K_corr,                       '#2a9d8f'),
    ]

    n_rows = len(top5)
    n_cols = len(kernels_ordered)

    fig = plt.figure(figsize=(3.2 * n_cols, 3.0 * n_rows))
    gs = GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.50, wspace=0.25,
        top=0.93, bottom=0.07, left=0.10, right=0.98,
    )
    fig.suptitle(
        'Kernel comparison — top features — prediction game\n'
        'Beijing US Embassy — heating onset day',
        fontsize=FS_SUPTITLE, fontweight='bold', y=0.99)

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    for ri, fi in enumerate(top5):
        raw = mob_onset['prediction'].get((fi,), np.zeros(T))
        for ci, (kname, K, kcol) in enumerate(kernels_ordered):
            ax    = axes[ri, ci]
            curve = apply_kernel(raw, K)
            ax.plot(t_grid, curve, color=FEAT_COLORS[pnames[fi]], lw=2.2)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _phase_shade(ax)
            _set_time_axis(ax, sparse=True)
            ax.tick_params(labelsize=FS_TICK - 1)
            ax.set_xlabel('Hour', fontsize=FS_AXIS - 1)
            if ri == 0:
                ax.set_title(kname, fontsize=FS_TITLE,
                             fontweight='bold', color=kcol)
            if ci == 0:
                ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=FS_AXIS - 1)
                ax.text(-0.38, 0.5, pnames[fi],
                        transform=ax.transAxes,
                        fontsize=FS_AXIS - 1, va='center', ha='right',
                        rotation=90, color=FEAT_COLORS[pnames[fi]],
                        fontweight='bold')
            if fi == fi_lag:
                signs = np.sign(curve)
                crossings = np.where(np.diff(signs) != 0)[0]
                if len(crossings) > 0:
                    ax.axvline(crossings[0], color='#d62728',
                               lw=1.2, ls='--', alpha=0.7)

    return fig


# ===========================================================================
# 16. Figure A4 -- APPENDIX: network plots, all 3 games, identity kernel
#     3 rows (games) x 3 cols (pure / partial / full)
# ===========================================================================

def fig_A4_network_all_games(mob_onset, shap_onset, pnames):
    K_id = kernel_identity(t_grid)
    p    = len(pnames)

    game_specs = [
        ('prediction',  'Prediction',  'PDP',          'SHAP',               'ICE-agg.'),
        ('sensitivity', 'Sensitivity', 'Closed Sobol', 'Shapley-sensitivity', 'Total Sobol'),
        ('risk',        'Risk (MSE)',  'Pure Risk',    'SAGE',                'PFI'),
    ]

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(
        'Network plots — Identity kernel — all games\n'
        'Beijing US Embassy — heating onset day',
        fontsize=FS_SUPTITLE, fontweight='bold', y=1.01)

    gs = GridSpec(
        3, 3,
        figure=fig,
        hspace=0.08, wspace=0.08,
        left=0.09, right=0.98, top=0.91, bottom=0.07,
    )

    col_titles = ['Pure  $m_i$', 'Partial  $\\phi_i$  (SHAP)', 'Full  $\\Phi_i$']
    row_labels  = ['Prediction', 'Sensitivity', 'Risk (MSE)']

    for r, (gtype, glabel, lp, lpa, lf) in enumerate(game_specs):
        mob  = mob_onset[gtype]
        shap = shap_onset[gtype]

        for c, etype in enumerate(_EFFECT_TYPES):
            ax = fig.add_subplot(gs[r, c])
            node_imp, edge_imp, node_sign = _network_importances_bj(
                mob, shap, p, K_id, etype)
            title = col_titles[c] if r == 0 else ''
            _draw_network_bj(ax, pnames, node_imp, edge_imp, node_sign, title)
            if c == 0:
                ax.text(-0.03, 0.5, row_labels[r],
                        transform=ax.transAxes,
                        fontsize=FS_AXIS, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')

    # Node-only legend (mpatches) matching doc38 style
    leg_handles = [
        mpatches.Patch(color=_NODE_POS, label='Positive effect'),
        mpatches.Patch(color=_NODE_NEG, label='Negative effect'),
    ]
    fig.legend(handles=leg_handles, loc='lower center', ncol=2,
               fontsize=FS_LEGEND + 1, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    return fig


# ===========================================================================
# 18. Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  Beijing PM2.5 Air Quality  —  RF direct  (paper figures)')
    print('=' * 60)

    _require_dir(BASE_PLOT_DIR)
    _require_dir(DATA_DIR)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print('\n[1] Loading data ...')
    X_day, Y_raw, Y_adj, diurnal, dates = load_and_aggregate()
    X_day_np = X_day.to_numpy().astype(float)
    pnames   = list(DAY_FEATURE_NAMES)
    p        = len(pnames)

    # ── 2. Model ──────────────────────────────────────────────────────────
    print('\n[2] Fitting Random Forest ...')
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_day_np, Y_adj, test_size=0.2, random_state=RNG_SEED)
    model = RFModel()
    model.fit(X_tr, Y_tr)
    r2 = model.evaluate(X_te, Y_te)
    print('  Test R² (trajectory-level): {:.4f}'.format(r2))

    # ── 3. Kernels ────────────────────────────────────────────────────────
    print('\n[3] Building kernels ...')
    K_corr = kernel_correlation(Y_raw)

    # ── 4. Profiles ───────────────────────────────────────────────────────
    print('\n[4] Selecting profiles ...')
    fi_heat  = pnames.index('is_heating')
    fi_lag   = pnames.index('lag_pm25_mean')
    fi_wspm  = pnames.index('wspm_prev')

    lag_p75  = float(np.percentile(X_day_np[:, fi_lag],  75))
    lag_p25  = float(np.percentile(X_day_np[:, fi_lag],  25))
    wspm_p75 = float(np.percentile(X_day_np[:, fi_wspm], 75))

    def find_profile(conds, lbl):
        mask = np.ones(len(X_day_np), dtype=bool)
        for f, (lo, hi) in conds.items():
            ci = pnames.index(f)
            mask &= (X_day_np[:, ci] >= lo) & (X_day_np[:, ci] <= hi)
        hits = X_day_np[mask]
        if not len(hits):
            raise RuntimeError('No match: {}'.format(lbl))
        print('  "{}": {} days'.format(lbl, len(hits)))
        return hits[len(hits) // 2]

    x_heavy = find_profile(
        {'is_heating': (0.9, 1.1),
         'lag_pm25_mean': (lag_p75, 99),
         'wspm_prev': (0.0, 2.0)},
        'Heavy pollution winter')
    x_clean = find_profile(
        {'month': (5.9, 8.1),
         'lag_pm25_mean': (0.0, lag_p25),
         'wspm_prev': (wspm_p75, 99)},
        'Clean summer')
    x_onset = find_profile(
        {'month': (10.9, 11.1),
         'is_heating': (0.9, 1.1)},
        'Heating onset')

    def _y_obs(xp):
        diffs = np.abs(X_day_np - xp[None, :]).sum(axis=1)
        return Y_adj[int(np.argmin(diffs))]

    profile_defs = [
        ('Heavy pollution winter', x_heavy, _y_obs(x_heavy)),
        ('Clean summer',           x_clean, _y_obs(x_clean)),
        ('Heating onset',          x_onset, _y_obs(x_onset)),
    ]

    # ── 5. All games for ONSET day ────────────────────────────────────────
    print('\n[5] Computing all games (onset day) ...')
    x_prim, y_prim = x_onset, _y_obs(x_onset)
    mob_onset, shap_onset = {}, {}
    for gtype in ('prediction', 'sensitivity', 'risk'):
        print('  game: {} ...'.format(gtype))
        game = FunctionalGame(
            predict_fn  = model.predict,
            X_bg        = X_day_np,
            x_exp       = x_prim,
            game_type   = gtype,
            Y_obs       = y_prim,
            sample_size = SAMPLE_SIZE[gtype],
            random_seed = RNG_SEED,
        )
        game.precompute()
        mob_onset[gtype]  = moebius_transform(game)
        shap_onset[gtype] = shapley_values(mob_onset[gtype], game.p)

    # ── 6. Prediction game for all profiles ───────────────────────────────
    print('\n[6] Prediction game for all profiles ...')
    prof_results = {}
    for label, x_prof, y_prof in profile_defs:
        print('  profile: {} ...'.format(label))
        game = FunctionalGame(
            predict_fn  = model.predict,
            X_bg        = X_day_np,
            x_exp       = x_prof,
            game_type   = 'prediction',
            sample_size = SAMPLE_SIZE['prediction'],
            random_seed = RNG_SEED,
        )
        game.precompute()
        mob  = moebius_transform(game)
        shap = shapley_values(mob, game.p)
        prof_results[label] = (mob, shap)

    # ── 7. Figures ────────────────────────────────────────────────────────
    print('\n[7] Generating figures ...')

    savefig(
        fig0_main_body(diurnal, Y_raw, K_corr,
                       mob_onset, shap_onset, pnames,
                       prof_results=prof_results),
        'fig0_main_body.pdf')

    savefig(
        fig_diurnal_and_correlation(diurnal, Y_raw, K_corr),
        'fig1_diurnal_and_correlation.pdf')

    savefig(
        fig_A1_ppf_identity(mob_onset, shap_onset, pnames),
        'figA1_ppf_identity_onset.pdf')

    savefig(
        fig_A2_kernel_all_features(mob_onset, pnames, K_corr),
        'figA2_kernel_all_features.pdf')

    savefig(
        fig_A4_network_all_games(mob_onset, shap_onset, pnames),
        'figA4_network_all_games.pdf')

    print('\n' + '=' * 60)
    print('  Done.  Figures in {}/'.format(BASE_PLOT_DIR))
    print('  fig0_main_body.pdf            -- MAIN BODY')
    print('  fig1_diurnal_and_correlation.pdf')
    print('  figA1_ppf_identity_onset.pdf  -- APPENDIX: pure/partial/full')
    print('  figA2_kernel_all_features.pdf -- APPENDIX: 5 kernels x features')
    print('  figA4_network_all_games.pdf   -- APPENDIX: networks')
    print('=' * 60)