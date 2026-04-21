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
  fig1_correlation_matrices.pdf    -- side-by-side kernel comparison
                                      (heatmaps + row slices)
  fig2_main_effects_identity.pdf   -- main effects, identity kernel,
                                      both datasets x all games
  fig3_main_effects_correlation.pdf-- same with correlation kernel
  fig4_profiles_comparison.pdf     -- profile Shapley curves,
                                      identity vs correlation kernel,
                                      both datasets
"""

import itertools
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

BASE_PLOT_DIR = os.path.join(
    'plots', 'energy_comparison')

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

IHEPC_MORNING = (6,  10)   # 06:00-09:00
IHEPC_EVENING = (17, 22)   # 17:00-21:00

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

NESO_MORNING = (12, 19)   # 06:00-09:30
NESO_EVENING = (34, 42)   # 17:00-21:00

NESO_SAMPLE = {'prediction': 150, 'sensitivity': 200, 'risk': 200}

NESO_YLABEL = {
    'prediction' : 'Effect on demand (MW)',
    'sensitivity': r'Var$[F(t)]$ (MW$^2$)',
    'risk'       : r'Effect on MSE (MW$^2$)',
}

# ── Shared visual settings ────────────────────────────────────────────────
GAME_TYPES  = ['prediction', 'sensitivity', 'risk']

GAME_TITLE = {
    'prediction' :
        r'Prediction  $v(S)(t)=\mathbb{E}[F(x)(t)\mid X_S]$',
    'sensitivity':
        r'Sensitivity  $v(S)(t)=\mathrm{Var}[F(x)(t)\mid X_S]$',
    'risk'       :
        r'Risk (MSE)  $v(S)(t)=\mathbb{E}[(Y(t)-F(x)(t))^2\mid X_S]$',
}

# Colours shared across both feature sets (overlapping names get same colour)
FEAT_COLORS = {
    'day_of_week'   : '#1f77b4',
    'is_weekend'    : '#ff7f0e',
    'month'         : '#2ca02c',
    'season'        : '#d62728',
    'lag_daily_mean': '#9467bd',
    'lag_morning'   : '#8c564b',
    'lag_evening'   : '#e377c2',
}

# Dataset display names and accent colours for panel headers
DS_LABEL = {
    'ihepc': 'UCI IHEPC\n(Single household, kW)',
    'neso' : 'NESO GB Demand\n(National grid, MW)',
}
DS_COLOR = {
    'ihepc': '#2a9d8f',
    'neso' : '#e76f51',
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
        self.nc  = len(self.coalitions)
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
# 3.  IHEPC data loading
# ===========================================================================

def load_ihepc():
    """Load IHEPC from parquet cache or UCI ML Repo. Returns dataset dict."""
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
        'tag'       : 'ihepc',
        'X_np'      : X_day.to_numpy().astype(float),
        'Y_raw'     : Y_raw,
        'Y_adj'     : Y_adj,
        'diurnal'   : diurnal,
        'dates'     : dates,
        'features'  : IHEPC_FEATURES,
        'T'         : T,
        't_grid'    : IHEPC_TGRID,
        'tlabels'   : IHEPC_LABELS,
        'sample'    : IHEPC_SAMPLE,
        'ylabel'    : IHEPC_YLABEL,
        'morning'   : IHEPC_MORNING,
        'evening'   : IHEPC_EVENING,
    }


# ===========================================================================
# 4.  NESO data loading
# ===========================================================================

def load_neso():
    """Load NESO CSV files. Returns dataset dict."""
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
        'tag'       : 'neso',
        'X_np'      : X_day.to_numpy().astype(float),
        'Y_raw'     : Y_raw,
        'Y_adj'     : Y_adj,
        'diurnal'   : diurnal,
        'dates'     : dates,
        'features'  : NESO_FEATURES,
        'T'         : T,
        't_grid'    : NESO_TGRID,
        'tlabels'   : NESO_LABELS,
        'sample'    : NESO_SAMPLE,
        'ylabel'    : NESO_YLABEL,
        'morning'   : NESO_MORNING,
        'evening'   : NESO_EVENING,
    }


# ===========================================================================
# 5.  Plotting helpers
# ===========================================================================

def _xticks(ax, ds, sparse=False):
    """Set time-axis ticks appropriate for T=24 or T=48."""
    T       = ds['T']
    tlabels = ds['tlabels']
    step    = max(1, T // 8) * (2 if sparse else 1)
    idxs    = list(range(0, T, step))
    ax.set_xticks(idxs)
    ax.set_xticklabels(
        [tlabels[i] for i in idxs],
        rotation=45, ha='right', fontsize=6)
    ax.set_xlim(-0.5, T - 0.5)

def _shade(ax, ds):
    """Shade morning and evening peak windows."""
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
# 6.  Figure 1 -- Correlation matrix comparison
#     Row 1: IHEPC heatmap | NESO heatmap
#     Row 2: IHEPC row slice at AM peak | NESO row slice at AM peak
# ===========================================================================

def fig_correlation_matrices(ds_ihepc, ds_neso, K_ihepc, K_neso):
    """
    Four-panel figure showing the empirical correlation matrices and
    row slices for both datasets side by side.  The contrast between
    the structured IHEPC matrix and the near-uniform NESO matrix is
    the central visual argument for the correlation kernel section.
    """
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.40, wspace=0.30)

    axes_heat = [fig.add_subplot(gs[0, j]) for j in range(2)]
    axes_row  = [fig.add_subplot(gs[1, j]) for j in range(2)]

    fig.suptitle(
        'Empirical cross-time correlation structure\n'
        'UCI IHEPC (single household) vs NESO GB Demand (national grid)',
        fontsize=12, fontweight='bold')

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

        # ── Heatmap ───────────────────────────────────────────────────────
        ax = axes_heat[col]
        im = ax.imshow(
            K, aspect='auto', origin='upper',
            cmap='RdBu_r', vmin=-0.2, vmax=1.0)
        ax.set_xticks(tick_i)
        ax.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=6)
        ax.set_yticks(tick_i)
        ax.set_yticklabels(tick_lbl, fontsize=6)
        ax.set_xlabel('Time $s$', fontsize=8)
        ax.set_ylabel('Time $t$', fontsize=8)
        ax.set_title(
            DS_LABEL[tag],
            fontsize=10, fontweight='bold', color=DS_COLOR[tag])

        # Annotate off-diagonal AM×PM value
        ax.annotate(
            'AM$\\times$PM\n$K={:.2f}$'.format(K[am_mid, pm_mid]),
            xy=(pm_mid, am_mid),
            xytext=(pm_mid + T * 0.05, am_mid - T * 0.12),
            fontsize=7, color='#333',
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

        # Mark morning and evening blocks
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

        # ── Row slice at AM peak ──────────────────────────────────────────
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
            fontsize=9, fontweight='bold', color=DS_COLOR[tag])
        ax2.set_xticks(tick_i)
        ax2.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=6)
        ax2.set_xlim(-0.5, T - 0.5)
        ax2.set_xlabel('Time $s$', fontsize=8)
        ax2.set_ylabel('$K(t_{{AM}},\\ s)$', fontsize=8)
        ax2.tick_params(labelsize=7)

        # Annotate PM secondary peak value for IHEPC;
        # annotate flatness for NESO
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
# 7.  Figure 2 & 3 -- Main effects (identity kernel / correlation kernel)
#     Layout: 2 rows (datasets) x 3 cols (games)
#     Each cell: curves (left) + importance bar (right) -- combined in one ax
# ===========================================================================

def _main_effects_figure(
        ds_ihepc, mob_ihepc,
        ds_neso,  mob_neso,
        K_ihepc,  K_neso,
        kernel_label, top_k=5):
    """
    2-row x 3-col figure.
    Row 0: IHEPC, Row 1: NESO.
    Cols: prediction / sensitivity / risk.
    kernel_label: 'Identity kernel' or 'Empirical correlation kernel'
    """
    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 8),
        gridspec_kw={'hspace': 0.50, 'wspace': 0.35})

    fig.suptitle(
        'Main effects $m_i(t)$ — {}\n'
        'UCI IHEPC (single household)  vs  NESO GB Demand (national grid)'.format(
            kernel_label),
        fontsize=11, fontweight='bold')

    for row, (tag, ds, mob, K) in enumerate([
        ('ihepc', ds_ihepc, mob_ihepc, K_ihepc),
        ('neso',  ds_neso,  mob_neso,  K_neso),
    ]):
        features = ds['features']
        p        = len(features)
        ylabel   = ds['ylabel']
        t_grid   = ds['t_grid']

        for col, gtype in enumerate(GAME_TYPES):
            ax = axes[row, col]
            m  = mob[gtype]

            top = _top_k(m, p, top_k)
            for fi in top:
                raw   = m[(fi,)]
                curve = apply_kernel(raw, K)
                ax.plot(t_grid, curve,
                        color=FEAT_COLORS[features[fi]],
                        lw=2.0, label=features[fi])

            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _shade(ax, ds)
            _xticks(ax, ds)
            ax.tick_params(labelsize=7)
            ax.set_xlabel('Time', fontsize=7)

            # Row label (dataset name) on leftmost column
            if col == 0:
                ax.set_ylabel(ylabel[gtype], fontsize=8)
                ax.text(
                    -0.28, 0.5, DS_LABEL[tag],
                    transform=ax.transAxes,
                    fontsize=8, va='center', ha='right',
                    rotation=90, color=DS_COLOR[tag],
                    fontweight='bold')

            # Game title on top row only
            if row == 0:
                ax.set_title(GAME_TITLE[gtype], fontsize=8.5)

            # Legend on first cell per row
            if col == 0:
                ax.legend(fontsize=6.5, loc='upper left',
                          framealpha=0.8)

    return fig


# ===========================================================================
# 8.  Figure 4 -- Profile comparison
#     Layout: 2 rows (identity / correlation kernel)
#             x 6 cols (3 IHEPC profiles | 3 NESO profiles)
# ===========================================================================

def fig_profiles_comparison(
        ds_ihepc, prof_ihepc,
        ds_neso,  prof_neso,
        K_ihepc,  K_neso):
    """
    Two rows: identity kernel (top) and correlation kernel (bottom).
    Six columns: 3 IHEPC profiles | 3 NESO profiles.
    Shows how the kernel choice changes the profile explanations
    differently for the two datasets.
    """
    kernels_ordered = [
        ('Identity kernel',              K_ihepc * 0 + np.eye(ds_ihepc['T']),
                                         K_neso  * 0 + np.eye(ds_neso['T'])),
        ('Empirical correlation kernel', K_ihepc, K_neso),
    ]

    n_profiles = 3
    ncols      = 2 * n_profiles   # 3 IHEPC + 3 NESO
    nrows      = 2                 # identity / correlation

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 3.8 * nrows),
        gridspec_kw={'hspace': 0.55, 'wspace': 0.35})

    fig.suptitle(
        'Shapley curves — prediction game\n'
        'Identity kernel (top) vs Empirical correlation kernel (bottom)\n'
        'UCI IHEPC  |  NESO GB Demand',
        fontsize=11, fontweight='bold')

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
        # ── IHEPC profiles (cols 0-2) ─────────────────────────────────────
        ds = ds_ihepc
        features = ds['features']
        p        = len(features)
        t_grid   = ds['t_grid']

        # Global top-4 across all IHEPC profiles
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
            ax.tick_params(labelsize=7)
            ax.set_xlabel('Time', fontsize=7)
            ax.set_title(ihepc_titles.get(lbl, lbl),
                         fontsize=8, color=DS_COLOR['ihepc'],
                         fontweight='bold')
            if c == 0:
                ax.set_ylabel(ds['ylabel']['prediction'], fontsize=8)
                ax.text(-0.30, 0.5, k_label,
                        transform=ax.transAxes,
                        fontsize=8, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')
            if c == 0 and row == 0:
                ax.legend(fontsize=6.5, loc='upper left')

        # ── NESO profiles (cols 3-5) ──────────────────────────────────────
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
            ax.tick_params(labelsize=7)
            ax.set_xlabel('Time', fontsize=7)
            ax.set_title(neso_titles.get(lbl, lbl),
                         fontsize=8, color=DS_COLOR['neso'],
                         fontweight='bold')
            if c == 0:
                ax.set_ylabel(ds['ylabel']['prediction'], fontsize=8)
            if c == 0 and row == 0:
                ax.legend(fontsize=6.5, loc='upper left')

    # Column group labels
    fig.text(0.27, 0.98, 'UCI IHEPC (single household, kW)',
             ha='center', fontsize=9, fontweight='bold',
             color=DS_COLOR['ihepc'])
    fig.text(0.73, 0.98, 'NESO GB Demand (national grid, MW)',
             ha='center', fontsize=9, fontweight='bold',
             color=DS_COLOR['neso'])

    return fig


# ===========================================================================
# 9.  Run all games for a dataset
# ===========================================================================

def run_games(ds, x_primary, y_primary):
    """Compute Möbius + Shapley for all three games on x_primary."""
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
    """Compute prediction-game Möbius + Shapley for a list of profiles."""
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
# 10.  Main
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
    K_id_ih = kernel_identity(ds_ih['T'])
    K_id_ne = kernel_identity(ds_ne['T'])

    # Report key values
    am_ih = (ds_ih['morning'][0] + ds_ih['morning'][1]) // 2
    pm_ih = (ds_ih['evening'][0] + ds_ih['evening'][1]) // 2
    am_ne = (ds_ne['morning'][0] + ds_ne['morning'][1]) // 2
    pm_ne = (ds_ne['evening'][0] + ds_ne['evening'][1]) // 2
    print('  IHEPC K[AM, PM] = {:.3f}'.format(K_ih[am_ih, pm_ih]))
    print('  NESO  K[AM, PM] = {:.3f}'.format(K_ne[am_ne, pm_ne]))

    # ── 4. Select profiles ────────────────────────────────────────────────
    print('\n[4] Selecting profiles ...')

    # IHEPC profiles
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

    # NESO profiles
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
        fig_correlation_matrices(ds_ih, ds_ne, K_ih, K_ne),
        'fig1_correlation_matrices.pdf')

    savefig(
        _main_effects_figure(
            ds_ih, mob_ih, ds_ne, mob_ne,
            K_id_ih, K_id_ne,
            'Identity kernel'),
        'fig2_main_effects_identity.pdf')

    savefig(
        _main_effects_figure(
            ds_ih, mob_ih, ds_ne, mob_ne,
            K_ih, K_ne,
            'Empirical correlation kernel'),
        'fig3_main_effects_correlation.pdf')

    savefig(
        fig_profiles_comparison(
            ds_ih, prof_ih, ds_ne, prof_ne,
            K_ih, K_ne),
        'fig4_profiles_comparison.pdf')

    print('\n' + '=' * 60)
    print('  Done.  Figures in {}/'.format(BASE_PLOT_DIR))
    print('  fig1_correlation_matrices.pdf')
    print('  fig2_main_effects_identity.pdf')
    print('  fig3_main_effects_correlation.pdf')
    print('  fig4_profiles_comparison.pdf')
    print('=' * 60)