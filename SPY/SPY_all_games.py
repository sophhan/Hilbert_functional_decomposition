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

Core argument
-------------
Both datasets share the same formal structure (day-level features ->
daily demand trajectory) but the underlying generating process differs
fundamentally:

  IHEPC: One household's routine. The morning peak (07-09h) and evening
  peak (18-21h) are driven by partially independent triggers. The
  correlation matrix has a local maximum near each peak and modest
  off-diagonal structure. The correlation kernel produces explanations
  that respect this two-phase structure.

  NESO: Aggregate demand across ~30M consumers. Temperature and season
  dominate, lifting or suppressing the entire trajectory quasi-uniformly.
  The correlation matrix is nearly flat at ~0.8-0.9 everywhere. The
  correlation kernel correctly produces near-constant weighting across
  time -- reflecting that regime features (month, season) affect the
  entire day as a unit rather than specific phases.

The OU kernel with a fixed length-scale cannot adapt to either structure
automatically: it treats 07:00 and 19:00 as nearly independent regardless
of whether they are (IHEPC) or are not (NESO). The correlation kernel is
the only choice that reads the dependence structure from the data.

Models  : RandomForestRegressor, direct multi-output (no PCA, no t input)
Games   : prediction, sensitivity, risk
Kernels : Identity (baseline), Correlation (main argument)

Figures
-------
  fig1_correlation_matrices.pdf              -- side-by-side kernel comparison
  fig2_main_effects_ppf_identity_ihepc.pdf   -- pure/partial/full, identity kernel, IHEPC
  fig3_main_effects_ppf_identity_neso.pdf    -- pure/partial/full, identity kernel, NESO
  fig4_profiles_comparison.pdf               -- profile Shapley curves,
                                                identity vs correlation kernel
  fig5_main_effects_ppf_ihepc.pdf            -- pure/partial/full, corr. kernel, IHEPC
  fig6_main_effects_ppf_neso.pdf             -- pure/partial/full, corr. kernel, NESO
  fig7_sensitivity_gap_ihepc.pdf             -- functional Sobol gap, IHEPC
  fig8_sensitivity_gap_neso.pdf              -- functional Sobol gap, NESO
  fig9_summary_ihepc.pdf                     -- 2x4 summary (id vs corr), IHEPC
  fig10_summary_neso.pdf                     -- 2x4 summary (id vs corr), NESO
"""

import itertools
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
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

# Legend locations per row for pure/partial/full figures
_LEG_LOC_E = {0: 'upper left', 1: 'lower left', 2: 'lower center'}


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
    """Pure (main) effect m_{(i,)}(t) for each player i."""
    return {i: mob.get((i,), np.zeros(T)).copy()
            for i in range(p)}


def _full_effects_e(mob, p, T):
    """Full (superset) effect: sum_{S containing i} m_S(t)."""
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
# 7.  Figure 1 -- Correlation matrix comparison
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
        pm_mid  = (evening[0] + evening[1]) // 2

        step     = max(1, T // 8)
        tick_i   = list(range(0, T, step))
        tick_lbl = [tlabels[i] for i in tick_i]

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

        ax.annotate(
            'AM$\\times$PM\n$K={:.2f}$'.format(K[am_mid, pm_mid]),
            xy=(pm_mid, am_mid),
            xytext=(pm_mid + T * 0.05, am_mid - T * 0.12),
            fontsize=7, color='#333',
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

        for (r0, r1), (c0, c1), ec in [
            (morning, morning, '#4a90e2'),
            (evening, evening, '#e24a4a'),
            (morning, evening, '#9b59b6'),
            (evening, morning, '#9b59b6'),
        ]:
            rect = plt.Rectangle(
                (c0-0.5, r0-0.5), c1-c0, r1-r0,
                linewidth=1.2, edgecolor=ec,
                facecolor='none', zorder=3)
            ax.add_patch(rect)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            labelsize=6)

        ax2 = axes_row[col]
        t_vec = np.arange(T, dtype=float)
        ax2.plot(t_vec, K[am_mid, :],
                 color=DS_COLOR[tag], lw=2.2,
                 label='Empirical correlation')
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

        if tag == 'ihepc':
            ax2.annotate(
                'Secondary peak\n(AM$\\leftrightarrow$PM)',
                xy=(pm_mid, K[am_mid, pm_mid]),
                xytext=(pm_mid - T * 0.3, K[am_mid, pm_mid] + 0.15),
                fontsize=7, color='#9b59b6',
                arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.0))
        else:
            mid = T // 2
            ax2.annotate(
                'Nearly uniform\n(regime-dominated)',
                xy=(mid, K[am_mid, mid]),
                xytext=(mid - T * 0.25, K[am_mid, mid] - 0.15),
                fontsize=7, color='#555',
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

    return fig


# ===========================================================================
# 8.  Figures 2 & 3 -- Main effects (identity / correlation kernel)
#     2 rows (datasets) x 3 cols (games)
# ===========================================================================

def fig_main_effects_ppf_identity(ds, mob_dict, shap_dict, top_k=5):
    """
    Pure / partial / full main effects for one dataset, identity kernel.
    Mirrors fig_main_effects_ppf but uses K_id instead of K_corr.
    Rows: prediction / sensitivity / risk.
    Cols: pure / partial / full / integrated importance bars.
    """
    tag      = ds['tag']
    features = ds['features']
    p        = len(features)
    T        = ds['T']
    t_grid   = ds['t_grid']
    ylabel   = ds['ylabel']
    K        = kernel_identity(T)

    fig, axes = plt.subplots(
        3, 4,
        figsize=(18, 4.0 * 3),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]},
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

        # Rank by partial abs importance (identity kernel = raw integral)
        imps_partial = {
            i: float(np.sum(np.abs(partial_eff[i])))
            for i in range(p)
        }
        top = sorted(imps_partial, key=imps_partial.get, reverse=True)[:top_k]

        # ── Curve panels (cols 0-2) ───────────────────────────────────────
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

        # ── Integrated importance bars (col 3) ────────────────────────────
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
        ax_bar.legend(fontsize=FS_LEGEND, loc='upper right')

    plt.tight_layout()
    return fig


# ===========================================================================
# 9.  Figure 4 -- Profile comparison
#     2 rows (identity / correlation) x 6 cols (3 IHEPC + 3 NESO)
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
        figsize=(4.0 * ncols, 3.8 * nrows),
        gridspec_kw={'hspace': 0.55, 'wspace': 0.35})

    fig.suptitle(
        'Shapley curves — prediction game\n'
        'Identity kernel (top) vs Empirical correlation kernel (bottom)\n'
        'UCI IHEPC  |  NESO GB Demand',
        fontsize=FS_SUPTITLE, fontweight='bold')

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
        ds = ds_ihepc
        features = ds['features']
        p        = len(features)
        t_grid   = ds['t_grid']

        all_mob_ih = {lbl: mob for lbl, (mob, shap) in prof_ihepc.items()}
        imps_ih = np.zeros(p)
        for mob in all_mob_ih.values():
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
                ax.legend(fontsize=FS_LEGEND, loc='upper left')

        ds = ds_neso
        features = ds['features']
        p        = len(features)
        t_grid   = ds['t_grid']

        all_mob_ne = {lbl: mob for lbl, (mob, shap) in prof_neso.items()}
        imps_ne = np.zeros(p)
        for mob in all_mob_ne.values():
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
                ax.legend(fontsize=FS_LEGEND, loc='upper left')

    fig.text(0.27, 0.98, 'UCI IHEPC (single household, kW)',
             ha='center', fontsize=FS_AXIS, fontweight='bold',
             color=DS_COLOR['ihepc'])
    fig.text(0.73, 0.98, 'NESO GB Demand (national grid, MW)',
             ha='center', fontsize=FS_AXIS, fontweight='bold',
             color=DS_COLOR['neso'])

    return fig


# ===========================================================================
# 10. Figure 5/6 -- Pure / partial / full main effects (one dataset)
#     3 rows (games) x 4 cols (pure / partial / full / importance bars)
#     Kernel: empirical correlation kernel
# ===========================================================================

def fig_main_effects_ppf(ds, mob_dict, shap_dict, K, top_k=5):
    """
    Pure / partial / full main effects for one dataset.
    Rows: prediction / sensitivity / risk.
    Cols: pure / partial / full / integrated importance bars.
    Kernel: empirical correlation (passed as K).
    """
    tag      = ds['tag']
    features = ds['features']
    p        = len(features)
    T        = ds['T']
    t_grid   = ds['t_grid']
    ylabel   = ds['ylabel']

    fig, axes = plt.subplots(
        3, 4,
        figsize=(18, 4.0 * 3),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]},
    )
    fig.suptitle(
        'Main effects — Empirical correlation kernel — pure / partial / full\n'
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
        ax_bar.legend(fontsize=FS_LEGEND, loc='upper right')

    plt.tight_layout()
    return fig


# ===========================================================================
# 11. Figures 7/8 -- Sensitivity gap  Delta_tau_i(t)
#     Top-4 features by integrated |gap|, correlation kernel
# ===========================================================================

def fig_sensitivity_gap_e(ds, mob_sens, K, top_k=4):
    """
    Functional Sobol gap for sensitivity game, correlation kernel.
    """
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

    _leg_locs = ['lower left', 'lower left', 'lower center', 'lower center']

    fig, axes = plt.subplots(
        1, top_k,
        figsize=(4.5 * top_k, 4.5),
        sharey=False,
    )
    fig.suptitle(
        r'Sensitivity gap  $\Delta\tau_i(t) = \bar{\tau}_i(t) - \tau^{\mathrm{cl}}_i(t)$'
        r'  —  Empirical correlation kernel'
        '\n'
        r'Total Sobol $\bar{\tau}_i$ minus Closed Sobol $\tau^{\mathrm{cl}}_i$:'
        r'  interaction contribution over time'
        '\n{}'.format(DS_LABEL[tag].replace('\n', '  ')),
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
        ax.legend(fontsize=FS_LEGEND,
                  loc=_leg_locs[idx], framealpha=0.85)

    plt.tight_layout()
    return fig


# ===========================================================================
# 12. Figures 9/10 -- Summary figure  2 rows x 4 cols (one dataset)
#
#   Rows:  prediction game  |  risk game
#   Cols:  pure  |  partial  |  top interaction  |  importance bars
#
#   Within cols 0-2:
#     faded thin  = identity kernel  (baseline)
#     vivid thick = correlation kernel
#     top-2 features by prediction partial importance
# ===========================================================================

def fig_summary_e(ds, mob_dict, shap_dict, K_corr, top_k_bar=5):
    """
    2-row x 4-col summary for one dataset.
    Identity kernel curves overlaid (faded) with correlation kernel (vivid).
    """
    tag      = ds['tag']
    features = ds['features']
    p        = len(features)
    T        = ds['T']
    t_grid   = ds['t_grid']
    ylabel   = ds['ylabel']

    K_id = kernel_identity(T)

    # ── Pre-compute effects ───────────────────────────────────────────────
    pure_pred    = _pure_effects_e(mob_dict['prediction'], p, T)
    partial_pred = shap_dict['prediction']
    pure_risk    = _pure_effects_e(mob_dict['risk'],       p, T)
    partial_risk = shap_dict['risk']

    # Top-2 features by correlation-kernel partial prediction importance
    imps_pred = {
        i: float(np.sum(np.abs(apply_kernel(partial_pred[i], K_corr))))
        for i in range(p)
    }
    top2 = sorted(imps_pred, key=imps_pred.get, reverse=True)[:2]
    fi_1, fi_2 = top2[0], top2[1]
    c_1 = FEAT_COLORS[features[fi_1]]
    c_2 = FEAT_COLORS[features[fi_2]]

    # Top interaction pair by prediction game raw Möbius importance
    pair_imp = {
        (i, j): float(np.sum(np.abs(
            mob_dict['prediction'].get((i, j), np.zeros(T)))))
        for i in range(p) for j in range(i+1, p)
    }
    top_pair = max(pair_imp, key=pair_imp.get)
    fi_a, fi_b = top_pair

    # ── Visual constants ──────────────────────────────────────────────────
    ID_ALPHA, ID_LW = 0.30, 1.2
    MX_LW           = 2.2

    row_specs = [
        ('prediction', ylabel['prediction'],
         'Pure  $m_i$  $\\equiv$  PDP',
         'Partial  $\\phi_i$  $\\equiv$  Shapley (SHAP)',
         pure_pred, partial_pred),
        ('risk',       ylabel['risk'],
         'Pure  $m_i$  $\\equiv$  Pure Risk',
         'Partial  $\\phi_i$  $\\equiv$  SAGE',
         pure_risk, partial_risk),
    ]
    row_labels = ['Prediction game', 'Risk game']

    fig, axes = plt.subplots(
        2, 4,
        figsize=(17, 7.5),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]},
    )
    fig.suptitle(
        '{}: Kernel Choice, Game Type and Effect Decomposition'.format(
            'UCI IHEPC' if tag == 'ihepc' else 'NESO GB Demand'),
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    for r, (gtype, y_label, lbl_pure, lbl_partial,
            pure_eff, partial_eff) in enumerate(row_specs):

        # ── Col 0: pure ──────────────────────────────────────────────────
        ax = axes[r, 0]
        # identity (faded)
        ax.plot(t_grid, apply_kernel(pure_eff[fi_1], K_id),
                color=c_1, lw=ID_LW, ls='-',  alpha=ID_ALPHA)
        ax.plot(t_grid, apply_kernel(pure_eff[fi_2], K_id),
                color=c_2, lw=ID_LW, ls='--', alpha=ID_ALPHA)
        # correlation kernel (vivid)
        ax.plot(t_grid, apply_kernel(pure_eff[fi_1], K_corr),
                color=c_1, lw=MX_LW, ls='-',
                label=features[fi_1])
        ax.plot(t_grid, apply_kernel(pure_eff[fi_2], K_corr),
                color=c_2, lw=MX_LW, ls='--',
                label=features[fi_2])
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _shade(ax, ds)
        _xticks(ax, ds)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_ylabel(y_label, fontsize=FS_AXIS)
        ax.set_title(lbl_pure, fontsize=FS_TITLE, fontweight='bold')
        ax.text(-0.32, 0.5, row_labels[r],
                transform=ax.transAxes,
                fontsize=FS_AXIS, va='center', ha='right',
                rotation=90, color='#333', fontweight='bold')

        if r == 0:
            handles = [
                Line2D([0], [0], color=c_1, lw=MX_LW, ls='-',
                       label='{} (corr.)'.format(features[fi_1])),
                Line2D([0], [0], color=c_2, lw=MX_LW, ls='--',
                       label='{} (corr.)'.format(features[fi_2])),
                Line2D([0], [0], color='gray', lw=ID_LW, ls='-',
                       alpha=0.6, label='identity (faded)'),
            ]
            ax.legend(handles=handles, fontsize=FS_LEGEND,
                      loc='upper center', framealpha=0.9)

        # ── Col 1: partial ───────────────────────────────────────────────
        ax = axes[r, 1]
        ax.plot(t_grid, apply_kernel(partial_eff[fi_1], K_id),
                color=c_1, lw=ID_LW, ls='-',  alpha=ID_ALPHA)
        ax.plot(t_grid, apply_kernel(partial_eff[fi_2], K_id),
                color=c_2, lw=ID_LW, ls='--', alpha=ID_ALPHA)
        ax.plot(t_grid, apply_kernel(partial_eff[fi_1], K_corr),
                color=c_1, lw=MX_LW, ls='-')
        ax.plot(t_grid, apply_kernel(partial_eff[fi_2], K_corr),
                color=c_2, lw=MX_LW, ls='--')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _shade(ax, ds)
        _xticks(ax, ds)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_title(lbl_partial, fontsize=FS_TITLE, fontweight='bold')

        # pure->partial ratio annotation for top feature
        pure_int = float(np.sum(np.abs(
            apply_kernel(pure_eff[fi_1], K_corr))))
        part_int = float(np.sum(np.abs(
            apply_kernel(partial_eff[fi_1], K_corr))))
        ratio = part_int / pure_int if pure_int > 1e-12 else 1.0
        ax.text(0.03, 0.97,
                '{}: partial/pure\n= {:.2f}$\\times$'.format(
                    features[fi_1], ratio),
                transform=ax.transAxes,
                fontsize=FS_ANNOT - 1, va='top', color=c_1,
                bbox=dict(boxstyle='round,pad=0.25',
                          fc='white', ec='#ddd', alpha=0.9))

        # ── Col 2: top interaction ────────────────────────────────────────
        ax = axes[r, 2]
        raw    = mob_dict[gtype].get(top_pair, np.zeros(T))
        int_id = apply_kernel(raw, K_id)
        ax.plot(t_grid, int_id, color='#888', lw=ID_LW, alpha=ID_ALPHA)
        int_cx = apply_kernel(raw, K_corr)
        pos = np.where(int_cx >= 0, int_cx, 0.0)
        neg = np.where(int_cx <  0, int_cx, 0.0)
        ax.fill_between(t_grid, 0, pos, color='#2a9d8f', alpha=0.30)
        ax.fill_between(t_grid, 0, neg, color='#e63946', alpha=0.30)
        ax.plot(t_grid, int_cx, color='#333', lw=MX_LW - 0.4)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _shade(ax, ds)
        _xticks(ax, ds)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_title(
            'Interaction  $m_{{ij}}(t)$\n'
            '{} $\\times$ {}'.format(features[fi_a], features[fi_b]),
            fontsize=FS_TITLE, fontweight='bold')
        integ = float(np.trapz(raw, t_grid))
        ax.text(0.03, 0.97,
                r'$\int m_{{ij}}\,dt$ = {:.3f}'.format(integ),
                transform=ax.transAxes,
                fontsize=FS_ANNOT, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25',
                          fc='white', ec='#aaa', alpha=0.85))

        # ── Col 3: importance bars ────────────────────────────────────────
        ax = axes[r, 3]
        imps_part = {
            i: float(np.sum(np.abs(apply_kernel(partial_eff[i], K_corr))))
            for i in range(p)
        }
        imps_pure = {
            i: float(np.sum(np.abs(apply_kernel(pure_eff[i], K_corr))))
            for i in range(p)
        }
        order = sorted(range(p),
                       key=lambda i: imps_part[i], reverse=True)[:top_k_bar]
        y_pos = np.arange(len(order))
        bar_h = 0.35
        ax.barh(y_pos - bar_h / 2,
                [imps_pure[i] for i in order], height=bar_h,
                color=[FEAT_COLORS[features[i]] for i in order],
                alpha=0.45, hatch='//', label='pure')
        ax.barh(y_pos + bar_h / 2,
                [imps_part[i] for i in order], height=bar_h,
                color=[FEAT_COLORS[features[i]] for i in order],
                alpha=0.90, label='partial (Shapley)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([features[i] for i in order], fontsize=FS_TICK)
        ax.axvline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_title('Integrated\nimportance\n(corr. kernel)',
                     fontsize=FS_TITLE, fontweight='bold')
        ax.legend(fontsize=FS_LEGEND, loc='upper right')

    plt.tight_layout(rect=[0.04, 0, 1, 0.95])
    fig.subplots_adjust(top=0.92)
    return fig


# ===========================================================================
# 13. Run all games / profiles
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
# 14. Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  Energy Combined Example  (IHEPC + NESO)')
    print('  Correlation kernel across data structures')
    print('=' * 60)

    _require_dir(BASE_PLOT_DIR)

    # ── 1. Load both datasets ─────────────────────────────────────────────
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
    # identity kernels are built inside fig_summary_e and fig_main_effects_ppf_identity

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

    # Fig 1: correlation matrix comparison
    savefig(
        fig_correlation_matrices(ds_ih, ds_ne, K_ih, K_ne),
        'fig1_correlation_matrices.pdf')

    # Fig 2: IHEPC pure/partial/full, identity kernel
    savefig(
        fig_main_effects_ppf_identity(ds_ih, mob_ih, shap_ih),
        'fig2_main_effects_ppf_identity_ihepc.pdf')

    # Fig 3: NESO pure/partial/full, identity kernel
    savefig(
        fig_main_effects_ppf_identity(ds_ne, mob_ne, shap_ne),
        'fig3_main_effects_ppf_identity_neso.pdf')

    # Fig 4: profile comparison, identity vs correlation
    savefig(
        fig_profiles_comparison(
            ds_ih, prof_ih, ds_ne, prof_ne,
            K_ih, K_ne),
        'fig4_profiles_comparison.pdf')

    # Fig 5: IHEPC pure/partial/full, correlation kernel
    savefig(
        fig_main_effects_ppf(ds_ih, mob_ih, shap_ih, K_ih),
        'fig5_main_effects_ppf_ihepc.pdf')

    # Fig 6: NESO pure/partial/full, correlation kernel
    savefig(
        fig_main_effects_ppf(ds_ne, mob_ne, shap_ne, K_ne),
        'fig6_main_effects_ppf_neso.pdf')

    # Fig 7: IHEPC sensitivity gap
    savefig(
        fig_sensitivity_gap_e(ds_ih, mob_ih['sensitivity'], K_ih),
        'fig7_sensitivity_gap_ihepc.pdf')

    # Fig 8: NESO sensitivity gap
    savefig(
        fig_sensitivity_gap_e(ds_ne, mob_ne['sensitivity'], K_ne),
        'fig8_sensitivity_gap_neso.pdf')

    # Fig 9: IHEPC summary (identity vs corr, prediction + risk)
    savefig(
        fig_summary_e(ds_ih, mob_ih, shap_ih, K_ih),
        'fig9_summary_ihepc.pdf')

    # Fig 10: NESO summary (identity vs corr, prediction + risk)
    savefig(
        fig_summary_e(ds_ne, mob_ne, shap_ne, K_ne),
        'fig10_summary_neso.pdf')

    print('\n' + '=' * 60)
    print('  Done.  Figures in {}/'.format(BASE_PLOT_DIR))
    print('  fig1_correlation_matrices.pdf')
    print('  fig2_main_effects_ppf_identity_ihepc.pdf  -- pure/partial/full, identity, IHEPC')
    print('  fig3_main_effects_ppf_identity_neso.pdf   -- pure/partial/full, identity, NESO')
    print('  fig4_profiles_comparison.pdf')
    print('  fig5_main_effects_ppf_ihepc.pdf   -- appendix: pure/partial/full IHEPC')
    print('  fig6_main_effects_ppf_neso.pdf    -- appendix: pure/partial/full NESO')
    print('  fig7_sensitivity_gap_ihepc.pdf    -- appendix: Sobol gap IHEPC')
    print('  fig8_sensitivity_gap_neso.pdf     -- appendix: Sobol gap NESO')
    print('  fig9_summary_ihepc.pdf            -- summary: kernel x game IHEPC')
    print('  fig10_summary_neso.pdf            -- summary: kernel x game NESO')
    print('=' * 60)