"""
Functional Explanation Framework -- Combined Energy Example
===========================================================
Two electricity demand datasets compared side-by-side to demonstrate
how the empirical correlation kernel adapts to the underlying data
structure:

  Dataset A -- UCI IHEPC (Individual Household Electric Power Consumption)
    Single French household, minute-level, 2006-2010.
    Aggregated to daily 24-hour trajectories (T=24).
    Fetched automatically via ucimlrepo (cached as parquet).

  Dataset B -- NESO GB Historic Demand
    GB national grid half-hourly demand, 2018-2022.
    Daily 48-period trajectories (T=48).
    Place demanddata_2018.csv ... demanddata_2022.csv in ./data/ngeso/

Figures
-------
  fig0_main_body.pdf                         -- main body summary
  fig1_correlation_matrices.pdf              -- side-by-side kernel comparison
  fig2_main_effects_ppf_identity_ihepc.pdf   -- pure/partial/full, identity kernel, IHEPC
  fig3_main_effects_ppf_identity_neso.pdf    -- pure/partial/full, identity kernel, NESO
  fig4_profiles_comparison.pdf               -- profile Shapley curves,
                                                identity vs correlation kernel
  fig5_main_effects_ppf_ihepc.pdf            -- pure/partial/full, corr. kernel, IHEPC
  fig6_main_effects_ppf_neso.pdf             -- pure/partial/full, corr. kernel, NESO
  fig7_sensitivity_gap_ihepc.pdf             -- functional Sobol gap, IHEPC
  fig8_sensitivity_gap_neso.pdf              -- functional Sobol gap, NESO
  fig_network_appendix_ihepc.pdf             -- 3x3 all games, corr. kernel, IHEPC
  fig_network_appendix_neso.pdf              -- 3x3 all games, corr. kernel, NESO
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0.  Global settings
# ---------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 42
RF_N_EST = 300
RF_JOBS  = -1

BASE_PLOT_DIR = os.path.join('plots', 'energy_comparison_all_games')

# ── Global font sizes ────────────────────────────────────────────────────
FS_SUPTITLE = 13
FS_TITLE    = 11
FS_AXIS     = 10
FS_TICK     = 9
FS_LEGEND   = 8.5
FS_ANNOT    = 8.5

# ── IHEPC settings ────────────────────────────────────────────────────────
IHEPC_DATA_DIR  = os.path.join(_HERE, 'data')
IHEPC_DATA_FILE = os.path.join(
    IHEPC_DATA_DIR, 'household_power_consumption.parquet')

IHEPC_T      = 24
IHEPC_LABELS = ['{:02d}:00'.format(h) for h in range(IHEPC_T)]
IHEPC_TGRID  = np.arange(IHEPC_T, dtype=float)

IHEPC_FEATURES = [
    'day_of_week',
    'is_weekend',
    'month',
    'season',
    'lag_daily_mean',
    'lag_morning',
]

IHEPC_MORNING = (6,  10)
IHEPC_EVENING = (17, 22)

IHEPC_SAMPLE = {'prediction': 150, 'sensitivity': 200, 'risk': 200}

IHEPC_YLABEL = {
    'prediction' : 'Effect on power (kW)',
    'sensitivity': r'Var$[F(t)]$ (kW$^2$)',
    'risk'       : r'Effect on MSE (kW$^2$)',
}

# ── NESO settings ─────────────────────────────────────────────────────────
NESO_DATA_DIR = os.path.join(_HERE, 'data')
NESO_YEARS    = [2018, 2019, 2020, 2021, 2022]

NESO_T      = 48
NESO_LABELS = [
    '{:02d}:{:02d}'.format((i * 30) // 60, (i * 30) % 60)
    for i in range(NESO_T)
]
NESO_TGRID = np.arange(NESO_T, dtype=float)

NESO_FEATURES = [
    'day_of_week',
    'is_weekend',
    'month',
    'season',
    'lag_daily_mean',
    'lag_morning',
    'lag_evening',
]

NESO_MORNING = (12, 19)
NESO_EVENING = (34, 42)

NESO_SAMPLE = {'prediction': 150, 'sensitivity': 200, 'risk': 200}

NESO_YLABEL = {
    'prediction' : 'Effect on demand (MW)',
    'sensitivity': r'Var$[F(t)]$ (MW$^2$)',
    'risk'       : r'Effect on MSE (MW$^2$)',
}

# ── Shared visual settings ────────────────────────────────────────────────
GAME_TYPES = ['prediction', 'sensitivity', 'risk']

GAME_TITLE = {
    'prediction' :
        r'Prediction  $v(S)(t)=\mathbb{E}[F(x)(t)\mid X_S]$',
    'sensitivity':
        r'Sensitivity  $v(S)(t)=\mathrm{Var}[F(x)(t)\mid X_S]$',
    'risk'       :
        r'Risk (MSE)  $v(S)(t)=\mathbb{E}[(Y(t)-F(x)(t))^2\mid X_S]$',
}

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
DS_COLOR = {
    'ihepc': '#2a9d8f',
    'neso' : '#e76f51',
}

# XAI label strings for pure/partial/full x game
_XAI_LABELS_E = {
    ('prediction',  'pure')   : 'Pure  $m_i$  $\\equiv$  PDP',
    ('prediction',  'partial'): 'Partial  $\\phi_i$  $\\equiv$  Shapley (SHAP)',
    ('prediction',  'full')   : 'Full  $\\Phi_i$  $\\equiv$  ICE-aggregate',
    ('sensitivity', 'pure')   : 'Pure  $\\equiv$  Closed Sobol  $\\tau^{\\mathrm{cl}}_i$',
    ('sensitivity', 'partial'): 'Partial  $\\equiv$  Shapley-sensitivity',
    ('sensitivity', 'full')   : 'Full  $\\equiv$  Total Sobol  $\\bar{\\tau}_i$',
    ('risk',        'pure')   : 'Pure  $\\equiv$  Pure Risk',
    ('risk',        'partial'): 'Partial  $\\equiv$  SAGE',
    ('risk',        'full')   : 'Full  $\\equiv$  PFI',
}

_EFFECT_TYPES_E = ['pure', 'partial', 'full']

# Legend locations per row for curve panels (col 0)
# row 0: upper left, row 1: lower left, row 2: lower left
_LEG_LOC_E = {0: 'upper left', 1: 'lower left', 2: 'lower left'}

# Network node / edge colours
_NODE_POS = '#2a9d8f'
_NODE_NEG = '#e63946'
_EDGE_SYN = '#2a9d8f'
_EDGE_RED = '#e63946'

# 3-letter abbreviations for network node labels
FEAT_ABBR = {
    'day_of_week'   : 'DoW',
    'is_weekend'    : 'WeD',
    'month'         : 'Mon',
    'season'        : 'Sea',
    'lag_daily_mean': 'LDM',
    'lag_morning'   : 'LMo',
    'lag_evening'   : 'LEv',
}


# ===========================================================================
# 1.  Shared infrastructure
# ===========================================================================

def _require_dir(path):
    os.makedirs(path, exist_ok=True)


def _month_to_season(m):
    if m in (12, 1, 2):  return 1
    elif m in (3, 4, 5): return 2
    elif m in (6, 7, 8): return 3
    else:                return 4


class RFModel:
    """Direct multi-output RF. No t input, no PCA."""
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
        return 1.0 - np.sum((Y_te - Yp)**2) / np.sum((Y_te - Y_te.mean())**2)


class FunctionalGame:
    def __init__(self, predict_fn, X_bg, x_exp, T,
                 features, game_type='prediction',
                 Y_obs=None, sample_size=150,
                 random_seed=RNG_SEED):
        if game_type == 'risk' and Y_obs is None:
            raise ValueError('Y_obs required for risk.')
        self.predict_fn = predict_fn
        self.X_bg       = X_bg
        self.x_exp      = x_exp
        self.T          = T
        self.game_type  = game_type
        self.Y_obs      = Y_obs
        self.n          = sample_size
        self.seed       = random_seed
        self.p          = len(features)
        self.player_names = list(features)
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
        print('    [{}] {} coalitions x {} samples x T={} ...'.format(
            self.game_type, self.nc, self.n, self.T))
        self.values = np.zeros((self.nc, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i+1) % 32 == 0 or i+1 == self.nc:
                print('      {}/{}'.format(i+1, self.nc))

    def __getitem__(self, c):
        return self.values[self._idx[c]]

    @property
    def empty_value(self):
        return self[tuple([False]*self.p)]


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


def shapley_values(mob, p, T):
    shap = {i: np.zeros(T) for i in range(p)}
    for S, m in mob.items():
        if len(S) == 0:
            continue
        for i in S:
            shap[i] += m / len(S)
    return shap


# ===========================================================================
# 2.  Kernels
# ===========================================================================

def kernel_identity(T):
    return np.eye(T)

def kernel_correlation(Y_raw):
    C   = np.cov(Y_raw.T)
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    K   = np.clip(C / np.outer(std, std), -1.0, 1.0)
    return K

def apply_kernel(effect, K, dt=1.0):
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return (K / rs) @ effect * dt


# ===========================================================================
# 3.  Pure / partial / full helpers
# ===========================================================================

def _pure_effects_e(mob, p, T):
    return {i: mob.get((i,), np.zeros(T)).copy() for i in range(p)}


def _full_effects_e(mob, p, T):
    full = {i: np.zeros(T) for i in range(p)}
    for S, m in mob.items():
        if len(S) == 0:
            continue
        for i in S:
            full[i] += m
    return full


# ===========================================================================
# 4.  IHEPC data loading
# ===========================================================================

def load_ihepc():
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
            raise RuntimeError('pip install ucimlrepo')
        from ucimlrepo import fetch_ucirepo
        print('  [IHEPC] Downloading from UCI ML Repo (id=235) ...')
        ds  = fetch_ucirepo(id=235)
        df  = ds.data.features.copy()
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'],
                dayfirst=True, errors='coerce')
            df = df.drop(columns=['Date', 'Time'])
        else:
            df = df.reset_index()
            df.columns = ['datetime'] + list(df.columns[1:])
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        for col in [c for c in df.columns if c not in
                    {'datetime', 'date', 'hour'}]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Global_active_power'])
        df['date'] = df['datetime'].dt.date.astype(str)
        df['hour'] = df['datetime'].dt.hour
        _require_dir(IHEPC_DATA_DIR)
        df.to_parquet(IHEPC_DATA_FILE, index=False)
        print('  [IHEPC] Cached to {}'.format(IHEPC_DATA_FILE))

    T = IHEPC_T
    hourly = (df.groupby(['date', 'hour'])['Global_active_power']
              .mean().unstack('hour').reindex(columns=range(T)))
    hourly = hourly[hourly.notna().sum(axis=1) == T]
    Y_raw  = hourly.values.astype(float)
    dates  = hourly.index.tolist()

    diurnal = Y_raw.mean(axis=0)
    Y_adj   = Y_raw - diurnal[None, :]

    records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str)
        m, dow = dt_obj.month, dt_obj.dayofweek
        if i == 0:
            lmean = float(Y_raw.mean())
            lmorn = float(Y_raw[:, IHEPC_MORNING[0]:IHEPC_MORNING[1]].mean())
        else:
            lmean = float(Y_raw[i-1].mean())
            lmorn = float(Y_raw[i-1, IHEPC_MORNING[0]:IHEPC_MORNING[1]].mean())
        records.append({
            'day_of_week'   : float(dow),
            'is_weekend'    : float(dow >= 5),
            'month'         : float(m),
            'season'        : float(_month_to_season(m)),
            'lag_daily_mean': lmean,
            'lag_morning'   : lmorn,
        })

    X_day = pd.DataFrame(records, index=dates)
    print('  [IHEPC] {} days, mean={:.3f} kW'.format(
        len(dates), Y_raw.mean()))

    return {
        'tag'     : 'ihepc',
        'X_np'    : X_day.to_numpy().astype(float),
        'Y_raw'   : Y_raw,
        'Y_adj'   : Y_adj,
        'diurnal' : diurnal,
        'dates'   : dates,
        'features': IHEPC_FEATURES,
        'T'       : T,
        't_grid'  : IHEPC_TGRID,
        'tlabels' : IHEPC_LABELS,
        'sample'  : IHEPC_SAMPLE,
        'ylabel'  : IHEPC_YLABEL,
        'morning' : IHEPC_MORNING,
        'evening' : IHEPC_EVENING,
    }


# ===========================================================================
# 5.  NESO data loading
# ===========================================================================

def load_neso():
    dfs = []
    for yr in NESO_YEARS:
        path = os.path.join(NESO_DATA_DIR, 'demanddata_{}.csv'.format(yr))
        if not os.path.isfile(path):
            raise RuntimeError('Missing NESO file: {}'.format(path))
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    raw.columns = [c.strip().upper() for c in raw.columns]

    date_col   = next(c for c in raw.columns if 'DATE' in c)
    period_col = next(c for c in raw.columns if 'PERIOD' in c)
    demand_col = 'ND' if 'ND' in raw.columns else 'TSD'

    raw[date_col]   = raw[date_col].astype(str).str.strip()
    raw[period_col] = pd.to_numeric(raw[period_col], errors='coerce')
    raw[demand_col] = pd.to_numeric(raw[demand_col], errors='coerce')
    raw = raw.dropna(subset=[date_col, period_col, demand_col])
    raw = raw[(raw[period_col] >= 1) & (raw[period_col] <= NESO_T)].copy()
    raw['period_idx'] = (raw[period_col] - 1).astype(int)

    pivot = raw.pivot_table(
        index=date_col, columns='period_idx',
        values=demand_col, aggfunc='mean')
    pivot = pivot.reindex(columns=range(NESO_T))
    pivot = pivot[pivot.notna().sum(axis=1) == NESO_T]

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
        records.append({
            'day_of_week'   : float(dow),
            'is_weekend'    : float(dow >= 5),
            'month'         : float(m),
            'season'        : float(_month_to_season(m)),
            'lag_daily_mean': lmean,
            'lag_morning'   : lmorn,
            'lag_evening'   : leve,
        })

    X_day = pd.DataFrame(records, index=dates)
    print('  [NESO] {} days, mean={:.0f} MW'.format(len(dates), Y_raw.mean()))

    return {
        'tag'     : 'neso',
        'X_np'    : X_day.to_numpy().astype(float),
        'Y_raw'   : Y_raw,
        'Y_adj'   : Y_adj,
        'diurnal' : diurnal,
        'dates'   : dates,
        'features': NESO_FEATURES,
        'T'       : T,
        't_grid'  : NESO_TGRID,
        'tlabels' : NESO_LABELS,
        'sample'  : NESO_SAMPLE,
        'ylabel'  : NESO_YLABEL,
        'morning' : NESO_MORNING,
        'evening' : NESO_EVENING,
    }


# ===========================================================================
# 6.  Plotting helpers
# ===========================================================================

def _xticks(ax, ds, sparse=False):
    T       = ds['T']
    tlabels = ds['tlabels']
    step    = max(1, T // 8) * (2 if sparse else 1)
    idxs    = list(range(0, T, step))
    ax.set_xticks(idxs)
    ax.set_xticklabels(
        [tlabels[i] for i in idxs],
        rotation=45, ha='right', fontsize=FS_TICK)
    ax.set_xlim(-0.5, T - 0.5)

def _shade(ax, ds):
    ax.axvspan(*ds['morning'], alpha=0.10, color='#4a90e2', zorder=0)
    ax.axvspan(*ds['evening'], alpha=0.10, color='#e24a4a', zorder=0)

def _top_k(mob, p, k=5):
    imps = np.array([float(np.sum(np.abs(mob[(i,)]))) for i in range(p)])
    return sorted(range(p), key=lambda i: imps[i], reverse=True)[:k]

def savefig(fig, name):
    path = os.path.join(BASE_PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print('  Saved: {}'.format(path))
    plt.close(fig)


# ===========================================================================
# 7.  Figure 1 -- Correlation matrix comparison (no annotations)
# ===========================================================================

def fig_correlation_matrices(ds_ihepc, ds_neso, K_ihepc, K_neso):
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    axes_heat = [fig.add_subplot(gs[0, j]) for j in range(2)]
    axes_row  = [fig.add_subplot(gs[1, j]) for j in range(2)]

    fig.suptitle(
        'Empirical cross-time correlation structure\n'
        'UCI IHEPC (single household) vs NESO GB Demand (national grid)',
        fontsize=FS_SUPTITLE, fontweight='bold')

    datasets = [
        ('ihepc', ds_ihepc, K_ihepc),
        ('neso',  ds_neso,  K_neso),
    ]

    for col, (tag, ds, K) in enumerate(datasets):
        T       = ds['T']
        tlabels = ds['tlabels']
        morning = ds['morning']
        evening = ds['evening']
        am_mid  = (morning[0] + morning[1]) // 2

        step     = max(1, T // 8)
        tick_i   = list(range(0, T, step))
        tick_lbl = [tlabels[i] for i in tick_i]

        # ── Heatmap ───────────────────────────────────────────────────────
        ax = axes_heat[col]
        im = ax.imshow(
            K, aspect='auto', origin='upper',
            cmap='RdBu_r', vmin=-0.2, vmax=1.0)
        ax.set_xticks(tick_i)
        ax.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=6)
        ax.set_yticks(tick_i)
        ax.set_yticklabels(tick_lbl, fontsize=6)
        ax.set_xlabel('Time $s$', fontsize=FS_AXIS)
        ax.set_ylabel('Time $t$', fontsize=FS_AXIS)
        ax.set_title(
            DS_LABEL[tag],
            fontsize=FS_TITLE, fontweight='bold', color=DS_COLOR[tag])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            labelsize=6)

        # ── AM row-slice ──────────────────────────────────────────────────
        ax2 = axes_row[col]
        t_vec = np.arange(T, dtype=float)
        ax2.plot(t_vec, K[am_mid, :],
                 color=DS_COLOR[tag], lw=2.2)
        ax2.axhline(0, color='gray', lw=0.5, ls=':')
        ax2.axvline(am_mid, color='gray', lw=0.8, ls=':', alpha=0.5)
        ax2.axvspan(*morning, alpha=0.12, color='#4a90e2')
        ax2.axvspan(*evening, alpha=0.12, color='#e24a4a')
        ax2.set_title(
            'Row $K(t_{{AM}}, s)$ — {}'.format(
                'IHEPC' if tag == 'ihepc' else 'NESO'),
            fontsize=FS_TITLE, fontweight='bold', color=DS_COLOR[tag])
        ax2.set_xticks(tick_i)
        ax2.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=6)
        ax2.set_xlim(-0.5, T - 0.5)
        ax2.set_xlabel('Time $s$', fontsize=FS_AXIS)
        ax2.set_ylabel('$K(t_{{AM}},\\ s)$', fontsize=FS_AXIS)
        ax2.tick_params(labelsize=FS_TICK)

    return fig


# ===========================================================================
# 8.  Figure 2 & 3 -- Main effects, identity kernel, pure/partial/full
#     3 rows (games) x 4 cols (pure/partial/full/importance bars)
#     Col 3 has extra width; legend shifted outside to the right.
# ===========================================================================

def fig_main_effects_ppf_identity(ds, mob_dict, shap_dict, top_k=5):
    tag      = ds['tag']
    features = ds['features']
    p        = len(features)
    T        = ds['T']
    t_grid   = ds['t_grid']
    ylabel   = ds['ylabel']
    K        = kernel_identity(T)

    fig, axes = plt.subplots(
        3, 4,
        figsize=(19, 4.0 * 3),
        gridspec_kw={'width_ratios': [3, 3, 3, 2.4]},
    )
    fig.suptitle(
        'Main effects — Identity kernel — pure / partial / full\n'
        '{}'.format(DS_LABEL[tag].replace('\n', '  ')),
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    for r, gtype in enumerate(GAME_TYPES):
        mob = mob_dict[gtype]

        pure_eff    = _pure_effects_e(mob, p, T)
        partial_eff = shap_dict[gtype]
        full_eff    = _full_effects_e(mob, p, T)

        effect_dicts = {
            'pure'   : pure_eff,
            'partial': partial_eff,
            'full'   : full_eff,
        }

        imps_partial = {
            i: float(np.sum(np.abs(partial_eff[i])))
            for i in range(p)
        }
        top = sorted(imps_partial, key=imps_partial.get, reverse=True)[:top_k]

        for c, etype in enumerate(_EFFECT_TYPES_E):
            ax  = axes[r, c]
            eff = effect_dicts[etype]

            for fi in top:
                curve = apply_kernel(eff[fi], K)
                ax.plot(t_grid, curve,
                        color=FEAT_COLORS[features[fi]],
                        lw=2.0,
                        label=features[fi] if c == 0 else '_')

            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _shade(ax, ds)
            _xticks(ax, ds)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.set_title(_XAI_LABELS_E[(gtype, etype)],
                         fontsize=FS_TITLE, fontweight='bold')

            if c == 0:
                ax.set_ylabel(ylabel[gtype], fontsize=FS_AXIS)
                ax.legend(fontsize=FS_LEGEND,
                          loc=_LEG_LOC_E[r], framealpha=0.85)

        ax_bar = axes[r, 3]
        imps_all = {
            etype: {
                i: float(np.sum(np.abs(effect_dicts[etype][i])))
                for i in range(p)
            }
            for etype in _EFFECT_TYPES_E
        }
        order = sorted(range(p),
                       key=lambda i: imps_all['partial'][i], reverse=True)
        y_pos   = np.arange(len(order))
        bar_h   = 0.25
        offsets = {'pure': -bar_h, 'partial': 0.0, 'full': bar_h}
        alphas  = {'pure': 0.55,   'partial': 0.90, 'full': 0.55}
        hatches = {'pure': '//',   'partial': '',   'full': '\\\\'}

        for etype in _EFFECT_TYPES_E:
            vals = [imps_all[etype][i] for i in order]
            ax_bar.barh(
                y_pos + offsets[etype], vals,
                height=bar_h,
                color=[FEAT_COLORS[features[i]] for i in order],
                alpha=alphas[etype],
                hatch=hatches[etype],
                label=etype,
            )

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([features[i] for i in order], fontsize=FS_TICK)
        ax_bar.axvline(0, color='gray', lw=0.8, ls=':')
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=FS_TICK)
        ax_bar.set_title('Integrated\nimportance\n(identity kernel)',
                         fontsize=FS_TITLE, fontweight='bold')
        # bar legend: just outside right edge, close to the bars
        ax_bar.legend(fontsize=FS_LEGEND, loc='upper right',
                      bbox_to_anchor=(1.22, 1.0), borderaxespad=0.)

    plt.tight_layout()
    return fig


# ===========================================================================
# 9.  Figure 4 -- Profile comparison
#     2 rows (identity / correlation) x 6 cols (3 IHEPC + 3 NESO)
#     Suptitle raised; plots enlarged; more hspace for legends.
# ===========================================================================

def fig_profiles_comparison(
        ds_ihepc, prof_ihepc,
        ds_neso,  prof_neso,
        K_ihepc,  K_neso):

    kernels_ordered = [
        ('Identity kernel',
         kernel_identity(ds_ihepc['T']),
         kernel_identity(ds_neso['T'])),
        ('Empirical correlation kernel', K_ihepc, K_neso),
    ]

    n_profiles = 3
    ncols      = 2 * n_profiles
    nrows      = 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.6 * ncols, 4.4 * nrows),
        gridspec_kw={'hspace': 0.65, 'wspace': 0.38})

    fig.suptitle(
        'Shapley curves — prediction game\n'
        'Identity kernel (top) vs Empirical correlation kernel (bottom)\n'
        'UCI IHEPC  |  NESO GB Demand',
        fontsize=FS_SUPTITLE, fontweight='bold',
        y=1.04)

    ihepc_titles = {
        'Typical weekday' : 'Typical weekday\n(Tue-Fri, mild)',
        'Weekend'         : 'Weekend\n(Sat-Sun)',
        'Cold winter day' : 'Cold winter day\n(Jan, high prior)',
    }
    neso_titles = {
        'Winter weekday'    : 'Winter weekday\n(Mon-Fri, Dec-Feb)',
        'Summer weekend'    : 'Summer weekend\n(Sat-Sun, Jun-Aug)',
        'Cold snap weekday' : 'Cold snap\n(winter, high lag)',
    }

    for row, (k_label, K_ih, K_ne) in enumerate(kernels_ordered):
        # ── IHEPC panels ─────────────────────────────────────────────────
        ds       = ds_ihepc
        features = ds['features']
        p        = len(features)
        t_grid   = ds['t_grid']

        imps_ih = np.zeros(p)
        for mob, _ in prof_ihepc.values():
            for i in range(p):
                imps_ih[i] += float(np.sum(np.abs(mob[(i,)])))
        top_ih = sorted(range(p), key=lambda i: imps_ih[i], reverse=True)[:4]

        for c, (lbl, (mob, shap)) in enumerate(prof_ihepc.items()):
            ax = axes[row, c]
            for fi in top_ih:
                curve = apply_kernel(shap[fi], K_ih)
                ax.plot(t_grid, curve,
                        color=FEAT_COLORS[features[fi]],
                        lw=2.0, label=features[fi])
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _shade(ax, ds)
            _xticks(ax, ds)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.set_title(ihepc_titles.get(lbl, lbl),
                         fontsize=FS_TITLE, color=DS_COLOR['ihepc'],
                         fontweight='bold')
            if c == 0:
                ax.set_ylabel(ds['ylabel']['prediction'], fontsize=FS_AXIS)
                ax.text(-0.30, 0.5, k_label,
                        transform=ax.transAxes,
                        fontsize=FS_AXIS, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')
            if c == 0 and row == 0:
                ax.legend(fontsize=FS_LEGEND, loc='upper left',
                          framealpha=0.9)

        # ── NESO panels ───────────────────────────────────────────────────
        ds       = ds_neso
        features = ds['features']
        p        = len(features)
        t_grid   = ds['t_grid']

        imps_ne = np.zeros(p)
        for mob, _ in prof_neso.values():
            for i in range(p):
                imps_ne[i] += float(np.sum(np.abs(mob[(i,)])))
        top_ne = sorted(range(p), key=lambda i: imps_ne[i], reverse=True)[:4]

        for c, (lbl, (mob, shap)) in enumerate(prof_neso.items()):
            ax = axes[row, c + n_profiles]
            for fi in top_ne:
                curve = apply_kernel(shap[fi], K_ne)
                ax.plot(t_grid, curve,
                        color=FEAT_COLORS[features[fi]],
                        lw=2.0, label=features[fi])
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _shade(ax, ds)
            _xticks(ax, ds)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.set_title(neso_titles.get(lbl, lbl),
                         fontsize=FS_TITLE, color=DS_COLOR['neso'],
                         fontweight='bold')
            if c == 0:
                ax.set_ylabel(ds['ylabel']['prediction'], fontsize=FS_AXIS)
            if c == 0 and row == 0:
                ax.legend(fontsize=FS_LEGEND, loc='upper left',
                          framealpha=0.9)

    fig.text(0.27, 1.02, 'UCI IHEPC (single household, kW)',
             ha='center', fontsize=FS_AXIS, fontweight='bold',
             color=DS_COLOR['ihepc'])
    fig.text(0.73, 1.02, 'NESO GB Demand (national grid, MW)',
             ha='center', fontsize=FS_AXIS, fontweight='bold',
             color=DS_COLOR['neso'])

    plt.tight_layout()
    return fig


# ===========================================================================
# 10. Figure 5 & 6 -- Pure/partial/full, correlation kernel
#     3 rows (games) x 4 cols (pure/partial/full/importance bars)
#     Plots enlarged; row-1 col-0 legend top-left;
#     col-3 legend shifted outside to the right.
# ===========================================================================

def fig_main_effects_ppf(ds, mob_dict, shap_dict, K, top_k=5,
                          legend_on_full=False):
    """
    Pure / partial / full main effects for one dataset, correlation kernel.

    legend_on_full=False  (default, fig 5 IHEPC):
        legend on col 0 (pure), using _leg_loc_ppf per row.
    legend_on_full=True   (fig 6 NESO):
        no legend on col 0; legend on col 2 (full), center-left.
    """
    tag      = ds['tag']
    features = ds['features']
    p        = len(features)
    T        = ds['T']
    t_grid   = ds['t_grid']
    ylabel   = ds['ylabel']

    fig, axes = plt.subplots(
        3, 4,
        figsize=(19, 4.4 * 3),
        gridspec_kw={'width_ratios': [3, 3, 3, 2.4]},
    )
    fig.suptitle(
        'Main effects — Empirical correlation kernel — pure / partial / full\n'
        '{}'.format(DS_LABEL[tag].replace('\n', '  ')),
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    # legend location for col-0 (used only when legend_on_full=False)
    _leg_loc_ppf = {0: 'upper left', 1: 'upper left', 2: 'lower center'}

    for r, gtype in enumerate(GAME_TYPES):
        mob = mob_dict[gtype]

        pure_eff    = _pure_effects_e(mob, p, T)
        partial_eff = shap_dict[gtype]
        full_eff    = _full_effects_e(mob, p, T)

        effect_dicts = {
            'pure'   : pure_eff,
            'partial': partial_eff,
            'full'   : full_eff,
        }

        imps_partial = {
            i: float(np.sum(np.abs(apply_kernel(partial_eff[i], K))))
            for i in range(p)
        }
        top = sorted(imps_partial, key=imps_partial.get, reverse=True)[:top_k]

        # ── Curve panels (cols 0-2) ───────────────────────────────────────
        for c, etype in enumerate(_EFFECT_TYPES_E):
            ax  = axes[r, c]
            eff = effect_dicts[etype]

            for fi in top:
                curve = apply_kernel(eff[fi], K)
                # always store label so either col can show the legend
                ax.plot(t_grid, curve,
                        color=FEAT_COLORS[features[fi]],
                        lw=2.0,
                        label=features[fi])

            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _shade(ax, ds)
            _xticks(ax, ds)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.set_title(_XAI_LABELS_E[(gtype, etype)],
                         fontsize=FS_TITLE, fontweight='bold')

            if c == 0:
                ax.set_ylabel(ylabel[gtype], fontsize=FS_AXIS)
                if not legend_on_full:
                    # fig 5 behaviour: legend on pure col
                    ax.legend(fontsize=FS_LEGEND,
                              loc=_leg_loc_ppf[r], framealpha=0.85)

            if c == 2 and legend_on_full:
                # fig 6 behaviour: legend on full col, center-left
                ax.legend(fontsize=FS_LEGEND,
                          loc='center left', framealpha=0.85)

        # ── Integrated importance bars (col 3) ────────────────────────────
        ax_bar = axes[r, 3]
        imps_all = {
            etype: {
                i: float(np.sum(np.abs(
                    apply_kernel(effect_dicts[etype][i], K))))
                for i in range(p)
            }
            for etype in _EFFECT_TYPES_E
        }
        order = sorted(range(p),
                       key=lambda i: imps_all['partial'][i], reverse=True)
        y_pos   = np.arange(len(order))
        bar_h   = 0.25
        offsets = {'pure': -bar_h, 'partial': 0.0, 'full': bar_h}
        alphas  = {'pure': 0.55,   'partial': 0.90, 'full': 0.55}
        hatches = {'pure': '//',   'partial': '',   'full': '\\\\'}

        for etype in _EFFECT_TYPES_E:
            vals = [imps_all[etype][i] for i in order]
            ax_bar.barh(
                y_pos + offsets[etype], vals,
                height=bar_h,
                color=[FEAT_COLORS[features[i]] for i in order],
                alpha=alphas[etype],
                hatch=hatches[etype],
                label=etype,
            )

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([features[i] for i in order], fontsize=FS_TICK)
        ax_bar.axvline(0, color='gray', lw=0.8, ls=':')
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=FS_TICK)
        ax_bar.set_title('Integrated\nimportance\n(corr. kernel)',
                         fontsize=FS_TITLE, fontweight='bold')
        ax_bar.legend(fontsize=FS_LEGEND, loc='upper right',
                      bbox_to_anchor=(1.22, 1.0), borderaxespad=0.)

    plt.tight_layout()
    return fig


# ===========================================================================
# 11. Figures 7 & 8 -- Sensitivity gap  Delta_tau_i(t)
#     Legends placed below each subplot.
# ===========================================================================

def fig_sensitivity_gap_e(ds, mob_sens, K, top_k=4):
    tag      = ds['tag']
    features = ds['features']
    p        = len(features)
    T        = ds['T']
    t_grid   = ds['t_grid']

    pure_eff = _pure_effects_e(mob_sens, p, T)
    full_eff = _full_effects_e(mob_sens, p, T)
    gap      = {i: full_eff[i] - pure_eff[i] for i in range(p)}

    gap_imp = {
        i: float(np.sum(np.abs(apply_kernel(gap[i], K))))
        for i in range(p)
    }
    top = sorted(gap_imp, key=gap_imp.get, reverse=True)[:top_k]

    fig, axes = plt.subplots(
        1, top_k,
        figsize=(4.5 * top_k, 4.5),
        sharey=False,
    )
    ds_label_flat = DS_LABEL[tag].replace('\n', '  ')
    fig.suptitle(
        r'Sensitivity gap  $\Delta\tau_i(t) = \bar{\tau}_i(t) - \tau^{\mathrm{cl}}_i(t)$'
        r'  —  Empirical correlation kernel'
        '\n'
        r'Total Sobol $\bar{\tau}_i$ minus Closed Sobol $\tau^{\mathrm{cl}}_i$:'
        r'  interaction contribution over time'
        '\n' + ds_label_flat,
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    for idx, (ax, fi) in enumerate(zip(axes, top)):
        col = FEAT_COLORS[features[fi]]

        pure_curve = apply_kernel(pure_eff[fi], K)
        full_curve = apply_kernel(full_eff[fi], K)
        gap_curve  = apply_kernel(gap[fi],      K)

        ax.fill_between(t_grid, pure_curve, full_curve,
                        color=col, alpha=0.18, label='gap region')
        ax.plot(t_grid, full_curve, color=col, lw=2.0, ls='-',
                label=r'Full  $\bar{\tau}_i$  (Total Sobol)')
        ax.plot(t_grid, pure_curve, color=col, lw=2.0, ls='--',
                label=r'Pure  $\tau^{\mathrm{cl}}_i$  (Closed Sobol)')
        ax.plot(t_grid, gap_curve,  color='black', lw=1.4, ls=':',
                alpha=0.7, label=r'Gap  $\Delta\tau_i$')

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _shade(ax, ds)
        _xticks(ax, ds)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_ylabel(ds['ylabel']['sensitivity'], fontsize=FS_AXIS)
        integ = float(np.trapz(np.abs(apply_kernel(gap[fi], K)), t_grid))
        ax.set_title(
            '{}\n'
            r'$\int|\Delta\tau_i|\,dt$ = {:.4f}'.format(features[fi], integ),
            fontsize=FS_TITLE, fontweight='bold',
            color=col,
        )
        # legend placed below the subplot
        ax.legend(
            fontsize=FS_LEGEND,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            framealpha=0.85,
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return fig


# ===========================================================================
# 12. Figure 0 -- Main-body summary
# ===========================================================================

def fig0_main_body(ds_ih, ds_ne,
                   mob_ih, shap_ih,
                   mob_ne, shap_ne,
                   K_ih,   K_ne):
    """
    Layout (3 rows):
      Row 0 (compact, centred): correlation heatmaps + AM row-slices
      Row 1 (tall):             pure + partial prediction curves, both datasets
      Row 2 (wide):             [IHEPC gap | IHEPC nets (2x2)] [NESO gap | NESO nets (2x2)]
    """
    fig = plt.figure(figsize=(22, 13))

    # ── Outer grid: 3 rows ────────────────────────────────────────────────
    gs_outer = GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.72, 1.55, 1.95],
        hspace=0.52,          # increased gap between row1 and row2
        top=0.91, bottom=0.07,
        left=0.05, right=0.98,
    )

    # ── Row 0: centred heatmaps + row-slices ─────────────────────────────
    gs0_wrap = GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=gs_outer[0],
        width_ratios=[0.06, 1.0, 0.06],
        wspace=0.0,
    )
    gs0_inner = GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=gs0_wrap[1],
        wspace=0.35,
        width_ratios=[1, 1.4, 1, 1.4],
    )
    ax_ih_heat = fig.add_subplot(gs0_inner[0])
    ax_ih_row  = fig.add_subplot(gs0_inner[1])
    ax_ne_heat = fig.add_subplot(gs0_inner[2])
    ax_ne_row  = fig.add_subplot(gs0_inner[3])

    # ── Row 1: prediction panels ──────────────────────────────────────────
    gs1 = GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=gs_outer[1],
        wspace=0.32,
    )
    ax_ih_pure = fig.add_subplot(gs1[0])
    ax_ih_part = fig.add_subplot(gs1[1])
    ax_ne_pure = fig.add_subplot(gs1[2])
    ax_ne_part = fig.add_subplot(gs1[3])

    # ── Row 2: [IHEPC gap | IHEPC nets] [NESO gap | NESO nets] ──────────
    gs2 = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs_outer[2],
        wspace=0.22,
    )

    # IHEPC half: gap panel + 2x2 networks
    gs2_ih = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs2[0],
        wspace=0.30,
        width_ratios=[1.1, 1.0],
    )
    ax_ih_gap = fig.add_subplot(gs2_ih[0])
    gs2_ih_nets = GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=gs2_ih[1],
        hspace=0.08,
        wspace=0.08,
    )
    ax_net_ih_sens_0 = fig.add_subplot(gs2_ih_nets[0, 0])
    ax_net_ih_sens_1 = fig.add_subplot(gs2_ih_nets[0, 1])
    ax_net_ih_pred_0 = fig.add_subplot(gs2_ih_nets[1, 0])
    ax_net_ih_pred_1 = fig.add_subplot(gs2_ih_nets[1, 1])

    # NESO half: gap panel + 2x2 networks
    gs2_ne = GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs2[1],
        wspace=0.30,
        width_ratios=[1.1, 1.0],
    )
    ax_ne_gap = fig.add_subplot(gs2_ne[0])
    gs2_ne_nets = GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=gs2_ne[1],
        hspace=0.08,
        wspace=0.08,
    )
    ax_net_ne_sens_0 = fig.add_subplot(gs2_ne_nets[0, 0])
    ax_net_ne_sens_1 = fig.add_subplot(gs2_ne_nets[0, 1])
    ax_net_ne_pred_0 = fig.add_subplot(gs2_ne_nets[1, 0])
    ax_net_ne_pred_1 = fig.add_subplot(gs2_ne_nets[1, 1])

    # ── Font sizes (enlarged for fig0) ───────────────────────────────────
    FS_SUP  = 15
    FS_T    = 13
    FS_AX   = 11.5
    FS_TK   = 10
    FS_LEG  = 10
    FS_RLAB = 11.5

    fig.suptitle(
        'Energy demand: correlation structure drives explanation shape\n'
        'UCI IHEPC (single household, kW)  vs  NESO GB Demand (national grid, MW)',
        fontsize=FS_SUP, fontweight='bold', y=0.975)

    ID_ALPHA, ID_LW = 0.25, 1.1
    MX_LW           = 2.4
    K_id_ih = kernel_identity(ds_ih['T'])
    K_id_ne = kernel_identity(ds_ne['T'])

    # ── Helper: compact heatmap ───────────────────────────────────────────
    def _heatmap(ax, ds, K, tag):
        T, tlabels = ds['T'], ds['tlabels']
        step  = max(1, T // 6)
        ticks = list(range(0, T, step))
        im = ax.imshow(K, aspect='auto', origin='upper',
                       cmap='RdBu_r', vmin=-0.2, vmax=1.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels([tlabels[i] for i in ticks],
                           rotation=45, ha='right', fontsize=6.5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([tlabels[i] for i in ticks], fontsize=6)
        ax.set_title(DS_LABEL[tag].replace('\n', ' '),
                     fontsize=FS_AX, fontweight='bold',
                     color=DS_COLOR[tag])
        am = (ds['morning'][0] + ds['morning'][1]) // 2
        ax.axhline(am, color='white', lw=0.8, ls='--', alpha=0.6)
        ax.axvline(am, color='white', lw=0.8, ls='--', alpha=0.6)
        plt.colorbar(im, ax=ax, fraction=0.06, pad=0.03).ax.tick_params(
            labelsize=6)

    # ── Helper: AM row-slice ──────────────────────────────────────────────
    def _rowslice(ax, ds, K, tag):
        T, tlabels       = ds['T'], ds['tlabels']
        morning, evening = ds['morning'], ds['evening']
        am    = (morning[0] + morning[1]) // 2
        step  = max(1, T // 6)
        ticks = list(range(0, T, step))
        ax.plot(np.arange(T), K[am, :], color=DS_COLOR[tag], lw=2.0)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.axvspan(*morning, alpha=0.12, color='#4a90e2')
        ax.axvspan(*evening, alpha=0.12, color='#e24a4a')
        ax.set_xticks(ticks)
        ax.set_xticklabels([tlabels[i] for i in ticks],
                           rotation=45, ha='right', fontsize=6.5)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylabel('$K(t_{AM}, s)$', fontsize=FS_AX - 1)
        ax.set_xlabel('Time $s$',        fontsize=FS_AX - 1)
        ax.tick_params(labelsize=FS_TK - 1)
        title = ('Structured (AM$\\leftrightarrow$PM)'
                 if tag == 'ihepc' else 'Uniform (regime-dominated)')
        ax.set_title(title, fontsize=FS_AX, color=DS_COLOR[tag],
                     fontweight='bold')

    # ── Row 0 ─────────────────────────────────────────────────────────────
    _heatmap( ax_ih_heat, ds_ih, K_ih, 'ihepc')
    _rowslice(ax_ih_row,  ds_ih, K_ih, 'ihepc')
    _heatmap( ax_ne_heat, ds_ne, K_ne, 'neso')
    _rowslice(ax_ne_row,  ds_ne, K_ne, 'neso')

    # ── Helper: prediction panel ──────────────────────────────────────────
    def _pred_panel(ax, ds, mob, shap, K_id, K_corr,
                    etype, title_str, tag, show_legend=False):
        features = ds['features']
        p, T     = len(features), ds['T']
        t_grid   = ds['t_grid']

        pure_eff    = _pure_effects_e(mob['prediction'], p, T)
        partial_eff = shap['prediction']
        eff = pure_eff if etype == 'pure' else partial_eff

        imps = {i: float(np.sum(np.abs(apply_kernel(partial_eff[i], K_corr))))
                for i in range(p)}
        top2 = sorted(imps, key=imps.get, reverse=True)[:2]

        for fi in top2:
            col = FEAT_COLORS[features[fi]]
            ls  = '-' if fi == top2[0] else '--'
            ax.plot(t_grid, apply_kernel(eff[fi], K_id),
                    color=col, lw=ID_LW, ls=ls, alpha=ID_ALPHA)
            ax.plot(t_grid, apply_kernel(eff[fi], K_corr),
                    color=col, lw=MX_LW, ls=ls, label=features[fi])

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _shade(ax, ds)
        _xticks(ax, ds, sparse=True)
        ax.tick_params(labelsize=FS_TK)
        ax.set_xlabel('Time', fontsize=FS_AX)
        ax.set_ylabel(ds['ylabel']['prediction'], fontsize=FS_AX)
        ax.set_title(title_str, fontsize=FS_T, fontweight='bold',
                     color=DS_COLOR[tag])

        if show_legend:
            handles = [
                Line2D([0],[0], color=FEAT_COLORS[features[top2[0]]],
                       lw=MX_LW, ls='-',
                       label=features[top2[0]] + ' (corr.)'),
                Line2D([0],[0], color=FEAT_COLORS[features[top2[1]]],
                       lw=MX_LW, ls='--',
                       label=features[top2[1]] + ' (corr.)'),
                Line2D([0],[0], color='gray', lw=ID_LW, ls='-',
                       alpha=0.6, label='identity (faded)'),
            ]
            ax.legend(handles=handles, fontsize=FS_LEG,
                      loc='upper left', framealpha=0.9)

    # ── Row 1 ─────────────────────────────────────────────────────────────
    _pred_panel(ax_ih_pure, ds_ih, mob_ih, shap_ih, K_id_ih, K_ih,
                'pure',    'Pure  $m_i \\equiv$ PDP',  'ihepc',
                show_legend=True)
    _pred_panel(ax_ih_part, ds_ih, mob_ih, shap_ih, K_id_ih, K_ih,
                'partial', 'Partial  $\\phi_i \\equiv$ SHAP', 'ihepc')
    _pred_panel(ax_ne_pure, ds_ne, mob_ne, shap_ne, K_id_ne, K_ne,
                'pure',    'Pure  $m_i \\equiv$ PDP',  'neso',
                show_legend=True)
    _pred_panel(ax_ne_part, ds_ne, mob_ne, shap_ne, K_id_ne, K_ne,
                'partial', 'Partial  $\\phi_i \\equiv$ SHAP', 'neso')

    ax_ih_pure.text(-0.20, 0.5, 'IHEPC',
                    transform=ax_ih_pure.transAxes,
                    fontsize=FS_RLAB, va='center', ha='right',
                    rotation=90, color=DS_COLOR['ihepc'], fontweight='bold')
    ax_ne_pure.text(-0.20, 0.5, 'NESO',
                    transform=ax_ne_pure.transAxes,
                    fontsize=FS_RLAB, va='center', ha='right',
                    rotation=90, color=DS_COLOR['neso'], fontweight='bold')

    # ── Helper: sensitivity gap panel ─────────────────────────────────────
    def _gap_panel(ax, ds, mob_sens, K_corr, tag):
        features = ds['features']
        p, T     = len(features), ds['T']
        t_grid   = ds['t_grid']

        pure_eff = _pure_effects_e(mob_sens, p, T)
        full_eff = _full_effects_e(mob_sens, p, T)
        gap      = {i: full_eff[i] - pure_eff[i] for i in range(p)}
        gap_imp  = {i: float(np.sum(np.abs(apply_kernel(gap[i], K_corr))))
                    for i in range(p)}
        fi  = max(gap_imp, key=gap_imp.get)
        col = FEAT_COLORS[features[fi]]

        pure_c = apply_kernel(pure_eff[fi], K_corr)
        full_c = apply_kernel(full_eff[fi], K_corr)
        gap_c  = apply_kernel(gap[fi],      K_corr)

        ax.fill_between(t_grid, pure_c, full_c,
                        color=col, alpha=0.20, label='gap region')
        ax.plot(t_grid, full_c, color=col, lw=2.0, ls='-',
                label=r'Full $\bar{\tau}_i$ (Total Sobol)')
        ax.plot(t_grid, pure_c, color=col, lw=2.0, ls='--',
                label=r'Pure $\tau^{\mathrm{cl}}_i$ (Closed Sobol)')
        ax.plot(t_grid, gap_c, color='black', lw=1.4, ls=':',
                alpha=0.7, label=r'Gap $\Delta\tau_i$')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _shade(ax, ds)
        _xticks(ax, ds, sparse=True)
        ax.tick_params(labelsize=FS_TK)
        ax.set_xlabel('Time', fontsize=FS_AX)
        ax.set_ylabel(ds['ylabel']['sensitivity'], fontsize=FS_AX)
        integ = float(np.trapz(np.abs(apply_kernel(gap[fi], K_corr)), t_grid))
        ax.set_title(
            'Sensitivity gap  —  {}  —  corr. kernel\n'
            'feature: {}    $\\int|\\Delta\\tau_i|\\,dt = {:.3g}$'.format(
                DS_LABEL[tag].split('\n')[0], features[fi], integ),
            fontsize=FS_T - 1, fontweight='bold', color=DS_COLOR[tag])
        # legend placed below the gap panel axes
        ax.legend(
            fontsize=FS_LEG,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.24),
            ncol=2,
            framealpha=0.85,
        )

    # ── Row 2: gap panels ─────────────────────────────────────────────────
    _gap_panel(ax_ih_gap, ds_ih, mob_ih['sensitivity'], K_ih, 'ihepc')
    _gap_panel(ax_ne_gap, ds_ne, mob_ne['sensitivity'], K_ne, 'neso')

    # ── Row 2: network panels ─────────────────────────────────────────────
    ih_net_specs = [
        (ax_net_ih_sens_0, 'sensitivity', 'pure',    'IHEPC sens.\npure'),
        (ax_net_ih_sens_1, 'sensitivity', 'partial', 'IHEPC sens.\npartial'),
        (ax_net_ih_pred_0, 'prediction',  'pure',    'IHEPC pred.\npure'),
        (ax_net_ih_pred_1, 'prediction',  'partial', 'IHEPC pred.\npartial'),
    ]
    for ax, gtype, etype, net_title in ih_net_specs:
        node_imp, edge_imp, node_sign = _network_importances(
            mob_ih[gtype], shap_ih[gtype],
            len(ds_ih['features']), ds_ih['T'], K_ih, etype)
        _draw_network(ax, ds_ih['features'],
                      node_imp, edge_imp, node_sign, net_title)

    ne_net_specs = [
        (ax_net_ne_sens_0, 'sensitivity', 'pure',    'NESO sens.\npure'),
        (ax_net_ne_sens_1, 'sensitivity', 'partial', 'NESO sens.\npartial'),
        (ax_net_ne_pred_0, 'prediction',  'pure',    'NESO pred.\npure'),
        (ax_net_ne_pred_1, 'prediction',  'partial', 'NESO pred.\npartial'),
    ]
    for ax, gtype, etype, net_title in ne_net_specs:
        node_imp, edge_imp, node_sign = _network_importances(
            mob_ne[gtype], shap_ne[gtype],
            len(ds_ne['features']), ds_ne['T'], K_ne, etype)
        _draw_network(ax, ds_ne['features'],
                      node_imp, edge_imp, node_sign, net_title)

    # ── Network legend: anchored to the bottom-right network axes ─────────
    # Place on ax_net_ih_pred_1 (bottom-right of IHEPC block) and
    # ax_net_ne_pred_1 (bottom-right of NESO block) independently
    net_handles = [
        Patch(facecolor=_NODE_POS, edgecolor='none', label='Positive effect'),
        Patch(facecolor=_NODE_NEG, edgecolor='none', label='Negative effect'),
    ]
    for net_ax in (ax_net_ih_pred_1, ax_net_ne_pred_1):
        net_ax.legend(
            handles=net_handles,
            loc='lower center',
            ncol=2,
            fontsize=FS_LEG - 1,
            framealpha=0.9,
            # place just below the axes box
            bbox_to_anchor=(0.5, -0.18),
            bbox_transform=net_ax.transAxes,
        )

    return fig

# ===========================================================================
# 13. Network helpers
# ===========================================================================

def _network_importances(mob, shap, p, T, K, effect_type):
    pure_eff = _pure_effects_e(mob, p, T)
    full_eff = _full_effects_e(mob, p, T)

    if effect_type == 'pure':
        eff = pure_eff
    elif effect_type == 'partial':
        eff = shap
    else:
        eff = full_eff

    t_grid = np.arange(T, dtype=float)

    node_imp  = np.array([
        float(np.sum(np.abs(apply_kernel(eff[i], K)))) for i in range(p)])
    node_sign = np.array([
        np.sign(float(np.trapz(apply_kernel(eff[i], K), t_grid)))
        for i in range(p)])

    edge_imp = {}
    for i in range(p):
        for j in range(i+1, p):
            raw      = mob.get((i, j), np.zeros(T))
            smoothed = apply_kernel(raw, K)
            val      = float(np.trapz(smoothed, t_grid))
            if abs(val) > 0:
                edge_imp[(i, j)] = val

    return node_imp, edge_imp, node_sign


def _draw_network(ax, features, node_imp, edge_imp, node_sign, title):
    """
    Draw one network panel.
    Tighter xlim/ylim (±1.32) to reduce inter-cell whitespace.
    """
    import math

    p     = len(features)
    angle = [math.pi / 2 - 2 * math.pi * i / p for i in range(p)]
    pos   = {i: (math.cos(a), math.sin(a)) for i, a in enumerate(angle)}

    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', pad=4)

    max_imp  = float(node_imp.max()) if node_imp.max() > 0 else 1.0
    node_r   = {i: 0.07 + 0.19 * (node_imp[i] / max_imp) for i in range(p)}
    max_edge = max((abs(v) for v in edge_imp.values()), default=1.0)
    max_edge = max_edge if max_edge > 0 else 1.0

    for (i, j), val in edge_imp.items():
        xi, yi = pos[i]
        xj, yj = pos[j]
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
        inner = plt.Circle((x, y), r * 0.52, color='white', ec='none',
                            zorder=3, alpha=0.95)
        ax.add_patch(inner)
        abbr = FEAT_ABBR.get(features[i], features[i][:3])
        ax.text(x, y, abbr, ha='center', va='center',
                fontsize=max(4.5, r * 22),
                fontweight='bold', color='#222', zorder=4)

    # tight limits: nodes on unit circle, largest node_r ≈ 0.26
    pad = 0.32
    ax.set_xlim(-1.0 - pad, 1.0 + pad)
    ax.set_ylim(-1.0 - pad, 1.0 + pad)


def fig_network_appendix(ds, mob_dict, shap_dict, K_corr):
    """
    3 rows x 3 cols — correlation kernel only:
      Row 0: prediction  -- pure | partial | full
      Row 1: sensitivity -- pure | partial | full
      Row 2: risk        -- pure | partial | full
    Two-entry Patch legend (positive / negative effect).
    """
    tag      = ds['tag']
    features = ds['features']
    p        = len(features)
    T        = ds['T']

    game_specs = [
        ('prediction',  'Prediction',  'PDP',          'SHAP',                'ICE-agg.'),
        ('sensitivity', 'Sensitivity', 'Closed Sobol', 'Shapley-sensitivity',  'Total Sobol'),
        ('risk',        'Risk (MSE)',  'Pure Risk',    'SAGE',                 'PFI'),
    ]

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(
        'Network plots — correlation kernel — all games\n'
        '{}'.format(DS_LABEL[tag].replace('\n', '  ')),
        fontsize=FS_SUPTITLE, fontweight='bold',
        y=1.01,
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.08,
        wspace=0.08,
        left=0.09, right=0.98,
        top=0.91, bottom=0.07,
    )

    col_etypes = ['pure', 'partial', 'full']

    for r, (gtype, glabel, lbl_pure, lbl_partial, lbl_full) in \
            enumerate(game_specs):
        col_labels = [
            'Pure  $m_i \\equiv$ {}'.format(lbl_pure),
            'Partial  $\\phi_i \\equiv$ {}'.format(lbl_partial),
            'Full  $\\Phi_i \\equiv$ {}'.format(lbl_full),
        ]
        mob  = mob_dict[gtype]
        shap = shap_dict[gtype]

        for c, etype in enumerate(col_etypes):
            ax = fig.add_subplot(gs[r, c])
            node_imp, edge_imp, node_sign = _network_importances(
                mob, shap, p, T, K_corr, etype)
            _draw_network(ax, features, node_imp, edge_imp, node_sign,
                          col_labels[c] if r == 0 else '')
            if c == 0:
                ax.text(-0.03, 0.5, glabel,
                        transform=ax.transAxes,
                        fontsize=FS_AXIS, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')

    leg_handles = [
        Patch(facecolor=_NODE_POS, edgecolor='none', label='Positive effect'),
        Patch(facecolor=_NODE_NEG, edgecolor='none', label='Negative effect'),
    ]
    fig.legend(
        handles=leg_handles,
        loc='lower center',
        ncol=2,
        fontsize=FS_LEGEND,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )
    return fig


# ===========================================================================
# 14. Run all games / profiles
# ===========================================================================

def run_games(ds, x_primary, y_primary):
    results = {}
    for gtype in GAME_TYPES:
        game = FunctionalGame(
            predict_fn  = ds['model'].predict,
            X_bg        = ds['X_np'],
            x_exp       = x_primary,
            T           = ds['T'],
            features    = ds['features'],
            game_type   = gtype,
            Y_obs       = y_primary,
            sample_size = ds['sample'][gtype],
            random_seed = RNG_SEED,
        )
        game.precompute()
        mob  = moebius_transform(game)
        shap = shapley_values(mob, game.p, game.T)
        results[gtype] = (mob, shap)
    return results


def run_profiles(ds, profile_defs):
    results = {}
    for label, x_prof, y_prof in profile_defs:
        print('  Profile: {} ...'.format(label))
        game = FunctionalGame(
            predict_fn  = ds['model'].predict,
            X_bg        = ds['X_np'],
            x_exp       = x_prof,
            T           = ds['T'],
            features    = ds['features'],
            game_type   = 'prediction',
            sample_size = ds['sample']['prediction'],
            random_seed = RNG_SEED,
        )
        game.precompute()
        mob  = moebius_transform(game)
        shap = shapley_values(mob, game.p, game.T)
        results[label] = (mob, shap)
    return results


# ===========================================================================
# 15. Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  Energy Combined Example  (IHEPC + NESO)')
    print('  Correlation kernel across data structures')
    print('=' * 60)

    _require_dir(BASE_PLOT_DIR)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print('\n[1] Loading data ...')
    ds_ih = load_ihepc()
    ds_ne = load_neso()

    # ── 2. Fit models ─────────────────────────────────────────────────────
    print('\n[2] Fitting models ...')
    for ds, name in [(ds_ih, 'IHEPC'), (ds_ne, 'NESO')]:
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            ds['X_np'], ds['Y_adj'],
            test_size=0.2, random_state=RNG_SEED)
        m = RFModel()
        m.fit(X_tr, Y_tr)
        r2 = m.evaluate(X_te, Y_te)
        print('  [{}] Test R²: {:.4f}'.format(name, r2))
        ds['model'] = m

    # ── 3. Correlation kernels ────────────────────────────────────────────
    print('\n[3] Building correlation kernels ...')
    K_ih = kernel_correlation(ds_ih['Y_raw'])
    K_ne = kernel_correlation(ds_ne['Y_raw'])

    am_ih = (ds_ih['morning'][0] + ds_ih['morning'][1]) // 2
    pm_ih = (ds_ih['evening'][0] + ds_ih['evening'][1]) // 2
    am_ne = (ds_ne['morning'][0] + ds_ne['morning'][1]) // 2
    pm_ne = (ds_ne['evening'][0] + ds_ne['evening'][1]) // 2
    print('  IHEPC K[AM, PM] = {:.3f}'.format(K_ih[am_ih, pm_ih]))
    print('  NESO  K[AM, PM] = {:.3f}'.format(K_ne[am_ne, pm_ne]))

    # ── 4. Select profiles ────────────────────────────────────────────────
    print('\n[4] Selecting profiles ...')

    X_ih = ds_ih['X_np']
    fn_ih = ds_ih['features']
    lag_p75_ih = float(np.percentile(
        X_ih[:, fn_ih.index('lag_daily_mean')], 75))

    def find_ih(conds, lbl):
        mask = np.ones(len(X_ih), dtype=bool)
        for f, (lo, hi) in conds.items():
            ci = fn_ih.index(f)
            mask &= (X_ih[:, ci] >= lo) & (X_ih[:, ci] <= hi)
        hits = X_ih[mask]
        if not len(hits):
            raise RuntimeError('No match: {}'.format(lbl))
        print('  IHEPC "{}": {} days'.format(lbl, len(hits)))
        return hits[len(hits) // 2]

    x_ih1 = find_ih({'is_weekend': (-0.1, 0.1),
                     'day_of_week': (0.9, 4.1)}, 'Typical weekday')
    x_ih2 = find_ih({'is_weekend': (0.9, 1.1)}, 'Weekend')
    x_ih3 = find_ih({'season': (0.9, 1.1), 'is_weekend': (-0.1, 0.1),
                     'lag_daily_mean': (lag_p75_ih, 9e9)}, 'Cold winter day')

    def _y_ih(xp):
        diffs = np.abs(X_ih - xp[None, :]).sum(axis=1)
        return ds_ih['Y_adj'][int(np.argmin(diffs))]

    ih_profile_defs = [
        ('Typical weekday', x_ih1, _y_ih(x_ih1)),
        ('Weekend',         x_ih2, _y_ih(x_ih2)),
        ('Cold winter day', x_ih3, _y_ih(x_ih3)),
    ]

    X_ne = ds_ne['X_np']
    fn_ne = ds_ne['features']
    lag_p75_ne = float(np.percentile(
        X_ne[:, fn_ne.index('lag_daily_mean')], 75))

    def find_ne(conds, lbl):
        mask = np.ones(len(X_ne), dtype=bool)
        for f, (lo, hi) in conds.items():
            ci = fn_ne.index(f)
            mask &= (X_ne[:, ci] >= lo) & (X_ne[:, ci] <= hi)
        hits = X_ne[mask]
        if not len(hits):
            raise RuntimeError('No match: {}'.format(lbl))
        print('  NESO "{}": {} days'.format(lbl, len(hits)))
        return hits[len(hits) // 2]

    x_ne1 = find_ne({'is_weekend': (-0.1, 0.1),
                     'season': (0.9, 1.1)}, 'Winter weekday')
    x_ne2 = find_ne({'is_weekend': (0.9, 1.1),
                     'season': (2.9, 3.1)}, 'Summer weekend')
    x_ne3 = find_ne({'is_weekend': (-0.1, 0.1), 'season': (0.9, 1.1),
                     'lag_daily_mean': (lag_p75_ne, 9e9)}, 'Cold snap weekday')

    def _y_ne(xp):
        diffs = np.abs(X_ne - xp[None, :]).sum(axis=1)
        return ds_ne['Y_adj'][int(np.argmin(diffs))]

    ne_profile_defs = [
        ('Winter weekday',    x_ne1, _y_ne(x_ne1)),
        ('Summer weekend',    x_ne2, _y_ne(x_ne2)),
        ('Cold snap weekday', x_ne3, _y_ne(x_ne3)),
    ]

    # ── 5. Run games ──────────────────────────────────────────────────────
    print('\n[5] Computing games ...')
    print('\n  IHEPC — primary profile (Typical weekday):')
    ih_games = run_games(ds_ih, x_ih1, _y_ih(x_ih1))
    mob_ih   = {gt: ih_games[gt][0] for gt in GAME_TYPES}
    shap_ih  = {gt: ih_games[gt][1] for gt in GAME_TYPES}

    print('\n  NESO — primary profile (Winter weekday):')
    ne_games = run_games(ds_ne, x_ne1, _y_ne(x_ne1))
    mob_ne   = {gt: ne_games[gt][0] for gt in GAME_TYPES}
    shap_ne  = {gt: ne_games[gt][1] for gt in GAME_TYPES}

    print('\n  IHEPC — all profiles (prediction game):')
    prof_ih = run_profiles(ds_ih, ih_profile_defs)

    print('\n  NESO — all profiles (prediction game):')
    prof_ne = run_profiles(ds_ne, ne_profile_defs)

    # ── 6. Generate figures ───────────────────────────────────────────────
    print('\n[6] Generating figures ...')

    savefig(
        fig0_main_body(ds_ih, ds_ne,
                       mob_ih, shap_ih,
                       mob_ne, shap_ne,
                       K_ih,   K_ne),
        'fig0_main_body.pdf')

    savefig(
        fig_correlation_matrices(ds_ih, ds_ne, K_ih, K_ne),
        'fig1_correlation_matrices.pdf')

    savefig(
        fig_main_effects_ppf_identity(ds_ih, mob_ih, shap_ih),
        'fig2_main_effects_ppf_identity_ihepc.pdf')

    savefig(
        fig_main_effects_ppf_identity(ds_ne, mob_ne, shap_ne),
        'fig3_main_effects_ppf_identity_neso.pdf')

    savefig(
        fig_profiles_comparison(
            ds_ih, prof_ih, ds_ne, prof_ne, K_ih, K_ne),
        'fig4_profiles_comparison.pdf')

    savefig(
        fig_main_effects_ppf(ds_ih, mob_ih, shap_ih, K_ih,
                             legend_on_full=False),
        'fig5_main_effects_ppf_ihepc.pdf')

    savefig(
        fig_main_effects_ppf(ds_ne, mob_ne, shap_ne, K_ne,
                             legend_on_full=True),
        'fig6_main_effects_ppf_neso.pdf')

    savefig(
        fig_sensitivity_gap_e(ds_ih, mob_ih['sensitivity'], K_ih),
        'fig7_sensitivity_gap_ihepc.pdf')

    savefig(
        fig_sensitivity_gap_e(ds_ne, mob_ne['sensitivity'], K_ne),
        'fig8_sensitivity_gap_neso.pdf')

    savefig(
        fig_network_appendix(ds_ih, mob_ih, shap_ih, K_ih),
        'fig_network_appendix_ihepc.pdf')

    savefig(
        fig_network_appendix(ds_ne, mob_ne, shap_ne, K_ne),
        'fig_network_appendix_neso.pdf')

    print('\n' + '=' * 60)
    print('  Done.  Figures in {}/'.format(BASE_PLOT_DIR))
    print('  fig0_main_body.pdf')
    print('  fig1_correlation_matrices.pdf')
    print('  fig2_main_effects_ppf_identity_ihepc.pdf')
    print('  fig3_main_effects_ppf_identity_neso.pdf')
    print('  fig4_profiles_comparison.pdf')
    print('  fig5_main_effects_ppf_ihepc.pdf')
    print('  fig6_main_effects_ppf_neso.pdf')
    print('  fig7_sensitivity_gap_ihepc.pdf')
    print('  fig8_sensitivity_gap_neso.pdf')
    print('  fig_network_appendix_ihepc.pdf')
    print('  fig_network_appendix_neso.pdf')
    print('=' * 60)