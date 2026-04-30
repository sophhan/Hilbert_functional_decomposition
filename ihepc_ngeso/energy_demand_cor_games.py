"""
Functional Explanation Framework -- Combined Energy Example  (v7)
=================================================================
Changes vs v6:
  - fig0_main_body:
      * Heatmap colormap anchored at 0 via TwoSlopeNorm(vcenter=0,
        vmin=min(-0.2, K.min()), vmax=K.max()), so white = 0,
        blue = negative, red = positive, colors not misleading.
      * Removed \\Phi formulas from titles.
      * Network plots enlarged by stealing width on the heatmap side
        only (not moving toward the sensitivity panel) via per-axis
        position adjustment after layout.
      * Bumped font sizes globally for the figure, especially heatmap
        tick labels / colorbar and network legend.
      * Updated suptitle.
  - fig1 + fig2 bar plots:
      * Pure / partial / full now distinguished by three shades of the
        same feature color (light / medium / dark) instead of identical
        feature color with hatches; hatches removed; alpha = 1.
  - fig5_networks_global:
      * compute_global_effects now also returns avg_mob (full averaged
        Mobius dict over all subsets), accumulated from per-instance
        caches.  Cache files are NOT regenerated — full Mobius is
        already in cache, we just aggregate it on top.
      * _network_importances_global uses correct pure / partial / full
        edge formulas:
            pure edge (i,j)     = m_{ij}
            partial edge (i,j)  = sum_{S superseteq {i,j}} m_S / |S|
            full edge (i,j)     = sum_{S superseteq {i,j}} m_S
        This makes pure/partial/full sensitivity networks genuinely
        differ.
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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
CACHE_VERSION = 'v6'

_HERE    = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 42
RF_N_EST = 300
RF_JOBS  = -1

BASE_PLOT_DIR  = os.path.join('plots', 'energy_cor_games')
GAME_CACHE_DIR = os.path.join(_HERE, 'game_results_energy')

GLOBAL_N_INSTANCES = 30
GLOBAL_SAMPLE_SIZE = 100
N_FEAT_BINS        = 20   # max bins for continuous features in PDP
PDP_N_INSTANCES    = 120  # larger instance set for PDP plots, ensures all
                          # discrete categories (months, seasons, etc.) are
                          # represented; uses a separate cache namespace

# Continuous features — use uniform bins; all others use distinct values
CONTINUOUS_FEATURES = {'lag_daily_mean', 'lag_morning', 'lag_evening'}

FS_SUPTITLE = 13
FS_TITLE    = 11
FS_AXIS     = 10
FS_TICK     = 9
FS_LEGEND   = 8.5
FS_ANNOT    = 8.5

def _fs(bump=0):
    return SimpleNamespace(
        suptitle=FS_SUPTITLE+bump, title=FS_TITLE+bump,
        axis=FS_AXIS+bump,   tick=FS_TICK+bump,
        legend=FS_LEGEND+bump, annot=FS_ANNOT+bump)

IHEPC_DATA_DIR  = os.path.join(_HERE, 'data')
IHEPC_DATA_FILE = os.path.join(IHEPC_DATA_DIR, 'household_power_consumption.parquet')
IHEPC_T        = 24
IHEPC_LABELS   = ['{:02d}:00'.format(h) for h in range(IHEPC_T)]
IHEPC_TGRID    = np.arange(IHEPC_T, dtype=float)
IHEPC_FEATURES = ['day_of_week','is_weekend','month','season',
                  'lag_daily_mean','lag_morning']
IHEPC_MORNING  = (6, 10)
IHEPC_EVENING  = (17, 22)
IHEPC_SAMPLE   = {'prediction': 150, 'sensitivity': 200, 'risk': 200}
IHEPC_YLABEL   = {
    'prediction' : 'Effect on power (kW)',
    'sensitivity': r'Var$[F(t)]$ (kW$^2$)',
    'risk'       : r'Effect on MSE (kW$^2$)',
}

NESO_DATA_DIR  = os.path.join(_HERE, 'data')
NESO_YEARS     = [2018, 2019, 2020, 2021, 2022]
NESO_T         = 48
NESO_LABELS    = ['{:02d}:{:02d}'.format((i*30)//60,(i*30)%60) for i in range(NESO_T)]
NESO_TGRID     = np.arange(NESO_T, dtype=float)
NESO_FEATURES  = ['day_of_week','is_weekend','month','season',
                  'lag_daily_mean','lag_morning','lag_evening']
NESO_MORNING   = (12, 19)
NESO_EVENING   = (34, 42)
NESO_SAMPLE    = {'prediction': 150, 'sensitivity': 200, 'risk': 200}
NESO_YLABEL    = {
    'prediction' : 'Effect on demand (MW)',
    'sensitivity': r'Var$[F(t)]$ (MW$^2$)',
    'risk'       : r'Effect on MSE (MW$^2$)',
}

GAME_TYPES = ['prediction', 'sensitivity', 'risk']

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

FEAT_ABBR = {
    'day_of_week'   : 'DoW', 'is_weekend'    : 'WeD',
    'month'         : 'Mon', 'season'        : 'Sea',
    'lag_daily_mean': 'LDM', 'lag_morning'   : 'LMo',
    'lag_evening'   : 'LEv',
}

SHADE_AM_COLOR = '#4a90e2'
SHADE_PM_COLOR = '#e24a4a'
SHADE_ALPHA    = 0.16

_NODE_POS = '#2a9d8f'; _NODE_NEG = '#e63946'
_EDGE_SYN = '#2a9d8f'; _EDGE_RED = '#e63946'

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
# 0b.  Color shading helpers (for fig1+fig2 bar plots)
# ===========================================================================

def _hex_to_rgb01(c):
    return matplotlib.colors.to_rgb(c)

def _shade_color(c, factor):
    """factor>0: lighten toward white; factor<0: darken toward black."""
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
    if m in (12,1,2): return 1
    elif m in (3,4,5): return 2
    elif m in (6,7,8): return 3
    else: return 4


# ===========================================================================
# 2.  Cache helpers
# ===========================================================================

def _cache_path_global(ds_tag, game_type, instance_k):
    return os.path.join(
        GAME_CACHE_DIR,
        'global_{}_{}_{:04d}_{}.npz'.format(
            ds_tag, game_type, instance_k, CACHE_VERSION))

def _cache_path_local(ds_tag, label):
    safe = label.replace(' ', '_').replace('-', '_')
    return os.path.join(
        GAME_CACHE_DIR,
        'local_{}_{}_{}.npz'.format(ds_tag, safe, CACHE_VERSION))

def _save_cache(path, x_inst, pure, partial, full, moebius, n_players):
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
    def __init__(self, random_state=RNG_SEED):
        self.model = RandomForestRegressor(
            n_estimators=RF_N_EST, n_jobs=RF_JOBS,
            random_state=random_state)
    def fit(self, X, Y):
        self.model.fit(X, Y); return self
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X_te, Y_te):
        Yp = self.predict(X_te)
        return 1.0 - np.sum((Y_te-Yp)**2) / np.sum((Y_te-Y_te.mean())**2)


# ===========================================================================
# 4.  Cooperative game
# ===========================================================================

class FunctionalGame:
    def __init__(self, predict_fn, X_bg, x_exp, T, features,
                 game_type='prediction', Y_obs=None,
                 sample_size=150, random_seed=RNG_SEED):
        if game_type == 'risk' and Y_obs is None:
            raise ValueError('Y_obs required for risk.')
        self.predict_fn = predict_fn; self.X_bg = X_bg
        self.x_exp = x_exp; self.T = T
        self.game_type = game_type; self.Y_obs = Y_obs
        self.n = sample_size; self.seed = random_seed
        self.p = len(features); self.player_names = list(features)
        self.coalitions = np.array(
            list(itertools.product([False,True], repeat=self.p)), dtype=bool)
        self.nc = len(self.coalitions)
        self._idx = {tuple(c): i for i, c in enumerate(self.coalitions)}
        self.values = None

    def _impute(self, coal):
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(self.X_bg), size=self.n)
        X = self.X_bg[idx].copy()
        for j in range(self.p):
            if coal[j]: X[:, j] = self.x_exp[j]
        return X

    def value_function(self, coal):
        X = self._impute(coal); Yp = self.predict_fn(X)
        if self.game_type == 'prediction': return Yp.mean(axis=0)
        elif self.game_type == 'sensitivity': return Yp.var(axis=0)
        else: return ((self.Y_obs[None,:] - Yp)**2).mean(axis=0)

    def precompute(self):
        self.values = np.zeros((self.nc, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i+1) % 32 == 0 or i+1 == self.nc:
                print('      {}/{}'.format(i+1, self.nc))

    def __getitem__(self, c):
        return self.values[self._idx[c]]


# ===========================================================================
# 5.  Möbius + Shapley
# ===========================================================================

def moebius_transform(game):
    p = game.p
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
    shap = {i: np.zeros(T) for i in range(p)}
    for S, m in mob.items():
        if len(S) == 0: continue
        for i in S: shap[i] += m / len(S)
    return shap


# ===========================================================================
# 6.  Kernels
# ===========================================================================

def kernel_identity(T):
    return np.eye(T)

def kernel_correlation(Y_raw):
    C = np.cov(Y_raw.T); std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    return np.clip(C / np.outer(std, std), -1.0, 1.0)

def apply_kernel(effect, K, dt=1.0):
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return (K/rs) @ effect * dt


# ===========================================================================
# 7.  Pure / partial / full helpers
# ===========================================================================

def _pure(mob, p, T):
    return {i: mob.get((i,), np.zeros(T)).copy() for i in range(p)}

def _full(mob, p, T):
    f = {i: np.zeros(T) for i in range(p)}
    for S, m in mob.items():
        if len(S) == 0: continue
        for i in S: f[i] += m
    return f


# ===========================================================================
# 8.  Data loading
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
        print('  [IHEPC] Downloading from UCI ML Repo ...')
        ds = fetch_ucirepo(id=235); df = ds.data.features.copy()
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(
                df['Date']+' '+df['Time'], dayfirst=True, errors='coerce')
            df = df.drop(columns=['Date','Time'])
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
    hourly = hourly[hourly.notna().sum(axis=1) == T]
    Y_raw = hourly.values.astype(float); dates = hourly.index.tolist()
    diurnal = Y_raw.mean(axis=0); Y_adj = Y_raw - diurnal[None,:]
    records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str); m, dow = dt_obj.month, dt_obj.dayofweek
        lmean = (float(Y_raw.mean()) if i==0
                 else float(Y_raw[i-1].mean()))
        lmorn = (float(Y_raw[:,IHEPC_MORNING[0]:IHEPC_MORNING[1]].mean()) if i==0
                 else float(Y_raw[i-1,IHEPC_MORNING[0]:IHEPC_MORNING[1]].mean()))
        records.append({'day_of_week':float(dow),'is_weekend':float(dow>=5),
                        'month':float(m),'season':float(_month_to_season(m)),
                        'lag_daily_mean':lmean,'lag_morning':lmorn})
    X_day = pd.DataFrame(records, index=dates)
    print('  [IHEPC] {} days, mean={:.3f} kW'.format(len(dates), Y_raw.mean()))
    return {'tag':'ihepc', 'X_np':X_day.to_numpy().astype(float),
            'Y_raw':Y_raw, 'Y_adj':Y_adj, 'diurnal':diurnal, 'dates':dates,
            'features':IHEPC_FEATURES, 'T':T, 't_grid':IHEPC_TGRID,
            'tlabels':IHEPC_LABELS, 'sample':IHEPC_SAMPLE, 'ylabel':IHEPC_YLABEL,
            'morning':IHEPC_MORNING, 'evening':IHEPC_EVENING}


def load_neso():
    dfs = []
    for yr in NESO_YEARS:
        path = os.path.join(NESO_DATA_DIR, 'demanddata_{}.csv'.format(yr))
        if not os.path.isfile(path):
            raise RuntimeError('Missing NESO file: {}'.format(path))
        dfs.append(pd.read_csv(path, low_memory=False))
    raw = pd.concat(dfs, ignore_index=True)
    raw.columns = [c.strip().upper() for c in raw.columns]
    date_col   = next(c for c in raw.columns if 'DATE' in c)
    period_col = next(c for c in raw.columns if 'PERIOD' in c)
    demand_col = 'ND' if 'ND' in raw.columns else 'TSD'
    raw[date_col]   = raw[date_col].astype(str).str.strip()
    raw[period_col] = pd.to_numeric(raw[period_col], errors='coerce')
    raw[demand_col] = pd.to_numeric(raw[demand_col], errors='coerce')
    raw = raw.dropna(subset=[date_col, period_col, demand_col])
    raw = raw[(raw[period_col]>=1) & (raw[period_col]<=NESO_T)].copy()
    raw['period_idx'] = (raw[period_col]-1).astype(int)
    pivot = raw.pivot_table(index=date_col, columns='period_idx',
                            values=demand_col, aggfunc='mean')
    pivot = pivot.reindex(columns=range(NESO_T))
    pivot = pivot[pivot.notna().sum(axis=1) == NESO_T]
    Y_raw = pivot.values.astype(float); dates = pivot.index.tolist()
    diurnal = Y_raw.mean(axis=0); Y_adj = Y_raw - diurnal[None,:]
    T = NESO_T; records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str); m, dow = dt_obj.month, dt_obj.dayofweek
        if i == 0:
            lmean = float(Y_raw.mean())
            lmorn = float(Y_raw[:,NESO_MORNING[0]:NESO_MORNING[1]].mean())
            leve  = float(Y_raw[:,NESO_EVENING[0]:NESO_EVENING[1]].mean())
        else:
            lmean = float(Y_raw[i-1].mean())
            lmorn = float(Y_raw[i-1,NESO_MORNING[0]:NESO_MORNING[1]].mean())
            leve  = float(Y_raw[i-1,NESO_EVENING[0]:NESO_EVENING[1]].mean())
        records.append({'day_of_week':float(dow),'is_weekend':float(dow>=5),
                        'month':float(m),'season':float(_month_to_season(m)),
                        'lag_daily_mean':lmean,'lag_morning':lmorn,
                        'lag_evening':leve})
    X_day = pd.DataFrame(records, index=dates)
    print('  [NESO] {} days, mean={:.0f} MW'.format(len(dates), Y_raw.mean()))
    return {'tag':'neso', 'X_np':X_day.to_numpy().astype(float),
            'Y_raw':Y_raw, 'Y_adj':Y_adj, 'diurnal':diurnal, 'dates':dates,
            'features':NESO_FEATURES, 'T':T, 't_grid':NESO_TGRID,
            'tlabels':NESO_LABELS, 'sample':NESO_SAMPLE, 'ylabel':NESO_YLABEL,
            'morning':NESO_MORNING, 'evening':NESO_EVENING}


# ===========================================================================
# 9.  Game computation with caching
# ===========================================================================

def _run_or_load(predict_fn, X_bg, Y_adj, x_inst, game_type,
                 sample_size, seed, cache_path, features, T):
    n_players = len(features)
    if os.path.isfile(cache_path):
        return _load_cache(cache_path, n_players)
    y_inst = None
    if game_type == 'risk':
        diffs  = np.abs(X_bg - x_inst[None,:]).sum(axis=1)
        y_inst = Y_adj[int(np.argmin(diffs))]
    game = FunctionalGame(predict_fn=predict_fn, X_bg=X_bg, x_exp=x_inst,
                          T=T, features=features, game_type=game_type,
                          Y_obs=y_inst, sample_size=sample_size,
                          random_seed=seed)
    game.precompute()
    mob  = moebius_transform(game)
    shap = shapley_values(mob, n_players, T)
    pure = _pure(mob, n_players, T)
    full = _full(mob, n_players, T)
    _save_cache(cache_path, x_inst, pure, shap, full, mob, n_players)
    return x_inst, pure, shap, full, mob


def compute_global_effects(ds, game_type, n_instances, sample_size, seed):
    """Average pure/partial/full and full Mobius dict over n_instances.

    Returns
    -------
    avg_shap : dict[int -> ndarray(T)]
    avg_pure : dict[int -> ndarray(T)]
    avg_full : dict[int -> ndarray(T)]
    avg_pairs : dict[(i,j) -> ndarray(T)]    # 2-subset Mobius (legacy)
    avg_mob  : dict[tuple -> ndarray(T)]     # ALL subsets, averaged
    """
    _require_dir(GAME_CACHE_DIR)
    X_bg = ds['X_np']; Y_adj = ds['Y_adj']
    T = ds['T']; features = ds['features']; n_players = len(features)
    rng  = np.random.default_rng(seed)
    idxs = rng.choice(len(X_bg), size=n_instances, replace=False)
    sum_shap = {i: np.zeros(T) for i in range(n_players)}
    sum_pure = {i: np.zeros(T) for i in range(n_players)}
    sum_full = {i: np.zeros(T) for i in range(n_players)}
    sum_pairs = {(i,j): np.zeros(T)
                 for i in range(n_players) for j in range(i+1, n_players)}
    sum_mob = {}  # accumulate ALL Mobius subsets
    for k, idx in enumerate(idxs):
        cache = _cache_path_global(ds['tag'], game_type, k)
        x_inst = X_bg[idx]
        _, pure, shap, full, mob = _run_or_load(
            ds['model'].predict, X_bg, Y_adj,
            x_inst, game_type, sample_size, seed+k, cache, features, T)
        for i in range(n_players):
            sum_shap[i] += shap[i]
            sum_pure[i] += pure[i]
            sum_full[i] += full[i]
        for i in range(n_players):
            for j in range(i+1, n_players):
                sum_pairs[(i,j)] += mob.get((i,j), np.zeros(T))
        # Accumulate full Mobius dict
        for S, m in mob.items():
            if S not in sum_mob:
                sum_mob[S] = np.zeros(T)
            sum_mob[S] += m
        status = '(cached)' if os.path.isfile(cache) else ''
        print('    [{} {} global] {}/{} {}'.format(
            ds['tag'], game_type, k+1, n_instances, status))
    avg_pairs = {k: v/n_instances for k, v in sum_pairs.items()}
    avg_mob   = {S: m/n_instances for S, m in sum_mob.items()}
    return ({i: sum_shap[i]/n_instances for i in range(n_players)},
            {i: sum_pure[i]/n_instances for i in range(n_players)},
            {i: sum_full[i]/n_instances for i in range(n_players)},
            avg_pairs,
            avg_mob)


def compute_local_game(ds, x_profile, label):
    """Compute local prediction game for profile, with caching."""
    _require_dir(GAME_CACHE_DIR)
    features = ds['features']; T = ds['T']; n_players = len(features)
    cache_path = _cache_path_local(ds['tag'], label)
    if os.path.isfile(cache_path):
        print('  Loading local cache: {} {}'.format(ds['tag'], label))
        _, pure, shap, full, mob = _load_cache(cache_path, n_players)
        return mob, shap, pure, full
    print('  Computing local game: {} {}'.format(ds['tag'], label))
    results = {}
    for gtype in GAME_TYPES:
        x_inst = x_profile
        y_inst = None
        if gtype == 'risk':
            diffs  = np.abs(ds['X_np'] - x_inst[None,:]).sum(axis=1)
            y_inst = ds['Y_adj'][int(np.argmin(diffs))]
        game = FunctionalGame(
            predict_fn=ds['model'].predict, X_bg=ds['X_np'],
            x_exp=x_inst, T=T, features=features,
            game_type=gtype, Y_obs=y_inst,
            sample_size=ds['sample'][gtype], random_seed=RNG_SEED)
        game.precompute()
        mob  = moebius_transform(game)
        shap = shapley_values(mob, n_players, T)
        pure = _pure(mob, n_players, T)
        full = _full(mob, n_players, T)
        cache = _cache_path_local(ds['tag'], label + '_' + gtype)
        _save_cache(cache, x_inst, pure, shap, full, mob, n_players)
        results[gtype] = (mob, shap, pure, full)
    return results


def load_local_games(ds, x_profile, label):
    """Load or compute all three local games for a profile."""
    _require_dir(GAME_CACHE_DIR)
    features = ds['features']; T = ds['T']; n_players = len(features)
    results = {}
    for gtype in GAME_TYPES:
        cache = _cache_path_local(ds['tag'], label + '_' + gtype)
        x_inst = x_profile
        y_inst = None
        if gtype == 'risk':
            diffs  = np.abs(ds['X_np'] - x_inst[None,:]).sum(axis=1)
            y_inst = ds['Y_adj'][int(np.argmin(diffs))]
        _, pure, shap, full, mob = _run_or_load(
            ds['model'].predict, ds['X_np'], ds['Y_adj'],
            x_inst, gtype, ds['sample'][gtype], RNG_SEED,
            cache, features, T)
        results[gtype] = (mob, shap, pure, full)
    return results


def load_per_instance_effects(ds, seed):
    """Load or compute per-instance prediction effects for PDP plot."""
    _require_dir(GAME_CACHE_DIR)
    X_bg = ds['X_np']; Y_adj = ds['Y_adj']
    features = ds['features']; T = ds['T']; n_players = len(features)
    rng  = np.random.default_rng(seed)
    idxs = rng.choice(len(X_bg), size=GLOBAL_N_INSTANCES, replace=False)
    results = []
    for k, idx in enumerate(idxs):
        cache  = _cache_path_global(ds['tag'], 'prediction', k)
        x_inst = X_bg[idx]
        _, pure, shap, full, _ = _run_or_load(
            ds['model'].predict, X_bg, Y_adj,
            x_inst, 'prediction', ds['sample']['prediction'],
            seed+k, cache, features, T)
        results.append({'x': x_inst, 'pure': pure,
                        'partial': shap, 'full': full})
        print('    [{} per-instance pred] {}/{}'.format(
            ds['tag'], k+1, GLOBAL_N_INSTANCES))
    return results, X_bg[idxs]


def _cache_path_pdp(ds_tag, instance_k):
    """Separate cache namespace for the larger PDP instance set."""
    return os.path.join(
        GAME_CACHE_DIR,
        'pdp_{}_inst{:04d}_{}.npz'.format(ds_tag, instance_k, CACHE_VERSION))


def load_per_instance_effects_pdp(ds, seed):
    """Load or compute per-instance prediction effects for PDP plots."""
    _require_dir(GAME_CACHE_DIR)
    X_bg = ds['X_np']; Y_adj = ds['Y_adj']
    features = ds['features']; T = ds['T']; n_players = len(features)

    month_col = features.index('month')
    selected  = []
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
        _, pure, shap, full, _ = _run_or_load(
            ds['model'].predict, X_bg, Y_adj,
            x_inst, 'prediction', ds['sample']['prediction'],
            seed + k, cache, features, T)
        results.append({'x': x_inst, 'pure': pure,
                        'partial': shap, 'full': full})
        print('    [{} pdp pred] {}/{}'.format(
            ds['tag'], k+1, PDP_N_INSTANCES))
    return results, X_bg[idxs]


# ===========================================================================
# 10.  Plotting helpers
# ===========================================================================

def _xticks(ax, ds, sparse=False, fs=None):
    tick_fs = (fs.tick if fs else FS_TICK)
    T, tlabels = ds['T'], ds['tlabels']
    step = max(1, T//8) * (2 if sparse else 1)
    idxs = list(range(0, T, step))
    ax.set_xticks(idxs)
    ax.set_xticklabels([tlabels[i] for i in idxs],
                       rotation=45, ha='right', fontsize=tick_fs)
    ax.set_xlim(-0.5, T-0.5)

def _shade(ax, ds):
    ax.axvspan(*ds['morning'], alpha=SHADE_ALPHA,
               color=SHADE_AM_COLOR, zorder=10, lw=0)
    ax.axvspan(*ds['evening'], alpha=SHADE_ALPHA,
               color=SHADE_PM_COLOR, zorder=10, lw=0)

def savefig(fig, name):
    path = os.path.join(BASE_PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print('  Saved: {}'.format(path))
    plt.close(fig)

def _align_row(axes_list):
    ymin = min(ax.get_ylim()[0] for ax in axes_list)
    ymax = max(ax.get_ylim()[1] for ax in axes_list)
    for ax in axes_list:
        ax.set_ylim(ymin, ymax)

def _draw_bar(ax, effect_dicts, K, features, fs, legend_bbox=None,
              x_fmt=None):
    """Bar plot with three SHADES of the same feature color for
    pure / partial / full (light / medium / dark).  No hatches."""
    p = len(features)
    imps = {et: {i: float(np.sum(np.abs(apply_kernel(effect_dicts[et][i], K))))
                 for i in range(p)}
            for et in _EFFECT_TYPES}
    order   = sorted(range(p), key=lambda i: imps['partial'][i], reverse=True)
    y_pos   = np.arange(len(order))
    bar_h   = 0.25
    offsets = {'pure':-bar_h, 'partial':0.0, 'full':bar_h}
    # Light / medium / dark shades of feature color
    shade_factors = {'pure': 0.55, 'partial': 0.0, 'full': -0.40}
    for et in _EFFECT_TYPES:
        sf = shade_factors[et]
        bar_colors = [_shade_color(FEAT_COLORS[features[i]], sf) for i in order]
        ax.barh(y_pos+offsets[et],
                [imps[et][i] for i in order], height=bar_h,
                color=bar_colors, alpha=1.0, label=et,
                edgecolor='none')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([features[i] for i in order], fontsize=fs.tick)
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=fs.axis)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=fs.tick)
    ax.set_title('Time-aggregated', fontsize=fs.title, fontweight='bold')
    if x_fmt is not None:
        ax.xaxis.set_major_formatter(x_fmt)
    # Legend with neutral grey-shade swatches so it reads as a shading legend,
    # independent of any specific feature color.
    leg_handles = [
        Patch(facecolor=_shade_color('#888888',  0.55), edgecolor='none',
              label='pure (light)'),
        Patch(facecolor=_shade_color('#888888',  0.0),  edgecolor='none',
              label='partial (medium)'),
        Patch(facecolor=_shade_color('#888888', -0.40), edgecolor='none',
              label='full (dark)'),
    ]
    leg_kwargs = dict(fontsize=fs.legend, loc='upper left',
                      bbox_to_anchor=(1.02, 1.0),
                      bbox_transform=ax.transAxes,
                      borderaxespad=0., framealpha=0.9)
    ax.legend(handles=leg_handles, **leg_kwargs)

def _add_bottom_legends_energy(fig, features, top_feats, fs,
                                feat_bbox=(0.04, 0.025),
                                kern_bbox=(0.04, 0.0),
                                kern_labels=('Identity kernel',
                                             'Correlation kernel')):
    """Feature color legend + kernel linestyle legend stacked at bottom-left."""
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
# 11.  Network helpers
# ===========================================================================

def _network_importances(mob, shap, p, T, K, effect_type='partial'):
    pure_eff = _pure(mob, p, T); full_eff = _full(mob, p, T)
    eff = (pure_eff if effect_type=='pure'
           else (shap if effect_type=='partial' else full_eff))
    t_grid = np.arange(T, dtype=float)
    node_imp  = np.array([float(np.sum(np.abs(apply_kernel(eff[i],K))))
                          for i in range(p)])
    node_sign = np.array([np.sign(float(np.trapz(apply_kernel(eff[i],K), t_grid)))
                          for i in range(p)])
    edge_imp = {}
    for i in range(p):
        for j in range(i+1, p):
            if effect_type == 'pure':
                raw = mob.get((i,j), np.zeros(T))
            elif effect_type == 'partial':
                # Shapley interaction index (Grabisch & Roubens 1999):
                # I^Sh_{ij} = sum_{S supseteq {i,j}} m_S / (|S| - 1)
                raw = np.zeros(T)
                for S, m in mob.items():
                    if i in S and j in S:
                        raw = raw + m / (len(S) - 1)
            else:
                raw = np.zeros(T)
                for S, m in mob.items():
                    if i in S and j in S: raw = raw + m
            val = float(np.trapz(apply_kernel(raw, K), t_grid))
            if abs(val) > 0: edge_imp[(i,j)] = val
    return node_imp, edge_imp, node_sign

def _network_importances_global(avg_shap, avg_pure, avg_full, avg_pairs,
                                 p, T, K, effect_type='full',
                                 avg_mob=None):
    """Compute network node/edge importances from global averaged effects.

    Edges:
      pure    : raw m_{ij}                    (= avg_pairs[(i,j)])
      partial : Shapley interaction index (Grabisch & Roubens 1999):
                I^Sh_{ij} = sum_{S supseteq {i,j}} m_S / (|S| - 1)
      full    : sum_{S supseteq {i,j}} m_S

    For partial / full we need the FULL averaged Mobius dict (avg_mob).
    If avg_mob is None we fall back to pairwise-only (legacy behaviour).
    """
    if effect_type == 'pure':
        eff = avg_pure
    elif effect_type == 'partial':
        eff = avg_shap
    else:
        eff = avg_full
    t_grid = np.arange(T, dtype=float)
    node_imp  = np.array([float(np.sum(np.abs(apply_kernel(eff[i], K))))
                          for i in range(p)])
    node_sign = np.array([np.sign(float(np.trapz(apply_kernel(eff[i], K), t_grid)))
                          for i in range(p)])
    edge_imp = {}
    for i in range(p):
        for j in range(i+1, p):
            if effect_type == 'pure' or avg_mob is None:
                raw = avg_pairs.get((i,j), np.zeros(T))
            elif effect_type == 'partial':
                # Shapley interaction index (Grabisch & Roubens 1999):
                # I^Sh_{ij} = sum_{S supseteq {i,j}} m_S / (|S| - 1)
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
                edge_imp[(i,j)] = val
    return node_imp, edge_imp, node_sign


def _draw_network(ax, features, node_imp, edge_imp, node_sign,
                  title, fs_title=None, fs_label=None):
    import math
    fs_t = fs_title if fs_title is not None else FS_TITLE
    p = len(features)
    angle = [math.pi/2 - 2*math.pi*i/p for i in range(p)]
    pos   = {i: (math.cos(a), math.sin(a)) for i, a in enumerate(angle)}
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, fontsize=fs_t, fontweight='bold', pad=4)
    max_imp  = float(node_imp.max()) if node_imp.max() > 0 else 1.0
    # Slightly larger floor so labels never crowd small nodes
    node_r   = {i: 0.13 + 0.20*(node_imp[i]/max_imp) for i in range(p)}
    max_edge = max((abs(v) for v in edge_imp.values()), default=1.0)
    max_edge = max(max_edge, 1e-12)
    for (i,j), val in edge_imp.items():
        xi,yi = pos[i]; xj,yj = pos[j]
        lw  = 0.4 + 6.5*abs(val)/max_edge
        col = _EDGE_SYN if val > 0 else _EDGE_RED
        alph = 0.30 + 0.60*abs(val)/max_edge
        ax.plot([xi,xj],[yi,yj], color=col, lw=lw, alpha=alph,
                solid_capstyle='round', zorder=1)
    import matplotlib.patheffects as path_effects
    for i in range(p):
        x, y = pos[i]; r = node_r[i]
        fc = _NODE_POS if node_sign[i] >= 0 else _NODE_NEG
        # Solid colored disk only (no white inner circle).
        ax.add_patch(plt.Circle((x,y), r, color=fc, ec='white',
                                linewidth=1.2, zorder=2, alpha=0.92))
        abbr = FEAT_ABBR.get(features[i], features[i][:3])
        node_fs = fs_label if fs_label is not None else max(8.0, r*30)
        # Dark text with a thin white outline for readability on the
        # colored disk; works well on both green and red backgrounds.
        txt = ax.text(x, y, abbr, ha='center', va='center',
                      fontsize=node_fs, fontweight='bold',
                      color='#1a1a1a', zorder=4)
        txt.set_path_effects([
            path_effects.withStroke(linewidth=2.2, foreground='white')])
    pad = 0.36
    ax.set_xlim(-1.0-pad, 1.0+pad); ax.set_ylim(-1.0-pad, 1.0+pad)


# ===========================================================================
# 12.  FIG 0 — Single-row: heatmap + network + gap, per dataset
# ===========================================================================

def fig0_main_body(ds_ih, ds_ne, mob_ih, shap_ih, mob_ne, shap_ne,
                   K_ih, K_ne, global_sens_ih, global_sens_ne):
    """
    Single row, 6 panels:
    [IHEPC heatmap | IHEPC sensitivity network | IHEPC sensitivity gap |
     NESO heatmap  | NESO sensitivity network  | NESO sensitivity gap  ]

    v7 changes:
      * Heatmap: TwoSlopeNorm anchored at 0 (white=0, blue=neg, red=pos).
      * Removed \\Phi from titles.
      * Networks enlarged toward heatmaps only (not toward sens. panel).
      * Bumped font sizes globally for this figure.
    """
    # Bumped font sizes
    FS_SUP=20; FS_T=16; FS_AX=15; FS_TK=13; FS_LEG=13; FS_NODE=12

    ID_ALPHA=0.40; ID_LW=1.6; MX_LW=2.2

    fig = plt.figure(figsize=(28, 5.8))
    gs  = GridSpec(1, 7, figure=fig,
                   width_ratios=[1.25, 1.2, 1.6, 0.18, 1.25, 1.2, 1.6],
                   wspace=0.28, left=0.04, right=0.98,
                   top=0.78, bottom=0.20)

    ax_ih_heat = fig.add_subplot(gs[0])
    ax_ih_net  = fig.add_subplot(gs[1])
    ax_ih_gap  = fig.add_subplot(gs[2])
    ax_gap_spacer = fig.add_subplot(gs[3])
    ax_gap_spacer.set_visible(False)
    ax_ne_heat = fig.add_subplot(gs[4])
    ax_ne_net  = fig.add_subplot(gs[5])
    ax_ne_gap  = fig.add_subplot(gs[6])

    fig.suptitle(
        'Hilbert-valued explanation framework: energy demand — '
        'correlation structure drives explanation shape\n'
        'UCI IHEPC (single household, kW)  vs  NESO GB Demand '
        '(national grid, MW)',
        fontsize=FS_SUP, fontweight='bold', y=0.98)

    # ── Heatmaps (canonical correlation range -1..1, white at 0) ─────────
    def _heatmap(ax, ds, K, tag):
        T, tl = ds['T'], ds['tlabels']
        step  = max(1, T//6); ticks = list(range(0, T, step))
        # Symmetric colorbar in canonical correlation range. White sits at
        # 0; the colorbar is uniformly spaced so unit distances are honest.
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

    # ── Sensitivity networks (full/total Sobol, correlation kernel) ───────
    net_handles = [Patch(facecolor=_NODE_POS, edgecolor='none', label='Positive'),
                   Patch(facecolor=_NODE_NEG, edgecolor='none', label='Negative')]
    for ax, ds, tag in [
        (ax_ih_net, ds_ih, 'ihepc'),
        (ax_ne_net, ds_ne, 'neso'),
    ]:
        features = ds['features']; p, T = len(features), ds['T']
        K = K_ih if tag == 'ihepc' else K_ne
        g_sens = global_sens_ih if tag == 'ihepc' else global_sens_ne
        # Unpack 5-tuple (avg_shap, avg_pure, avg_full, avg_pairs, avg_mob)
        avg_shap, avg_pure, avg_full, avg_pairs, avg_mob = g_sens
        ni, ei, ns = _network_importances_global(
            avg_shap, avg_pure, avg_full, avg_pairs, p, T, K, 'full',
            avg_mob=avg_mob)
        _draw_network(ax, features, ni, ei, ns,
                      '{} sensitivity\nfull (corr.)'.format(tag.upper()),
                      fs_title=FS_T, fs_label=FS_NODE)
        ax.legend(handles=net_handles, loc='lower center', ncol=2,
                  fontsize=FS_LEG, framealpha=0.88,
                  bbox_to_anchor=(0.5,-0.10), bbox_transform=ax.transAxes,
                  borderpad=0.45, handlelength=1.4)

    # ── Sensitivity full-effect (total Sobol) panels ──────────────────────
    def _sens_full_panel(ax, ds, global_sens, K_corr, K_id, tag,
                          force_features=None):
        features = ds['features']; p, T = len(features), ds['T']
        t_grid = ds['t_grid']
        avg_full = global_sens[2]  # full (total Sobol)
        if force_features is None:
            imps = {i: float(np.sum(np.abs(apply_kernel(avg_full[i], K_corr))))
                    for i in range(p)}
            fis = sorted(imps, key=imps.get, reverse=True)[:2]
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
        ax.legend(handles=corr_handles + [extra_id],
                  fontsize=10, loc='upper center',
                  bbox_to_anchor=(0.5, -0.25), ncol=len(fis)+1,
                  framealpha=0.85)
        _shade(ax, ds)

    K_id_ih = kernel_identity(ds_ih['T'])
    K_id_ne = kernel_identity(ds_ne['T'])
    _sens_full_panel(ax_ih_gap, ds_ih, global_sens_ih, K_ih, K_id_ih,
                     'ihepc', force_features=None)
    _sens_full_panel(ax_ne_gap, ds_ne, global_sens_ne, K_ne, K_id_ne,
                     'neso',  force_features=['month', 'season'])

    # Force a draw so positions are computed before we shift axes.
    fig.canvas.draw()

    # Enlarge networks toward the heatmap side ONLY (not toward sens. panel).
    # We shift the network axis leftward AND widen it on the left, leaving
    # its right edge unchanged so the gap to the sensitivity panel is fixed.
    SHIFT_FRAC = 0.45  # how much of the heatmap-network gap to absorb
    for ax_heat, ax_net in [(ax_ih_heat, ax_ih_net),
                            (ax_ne_heat, ax_ne_net)]:
        bb_h = ax_heat.get_position()
        bb_n = ax_net.get_position()
        gap = bb_n.x0 - bb_h.x1
        if gap <= 0:
            continue
        delta = SHIFT_FRAC * gap
        new_x0 = bb_n.x0 - delta
        new_w  = bb_n.width + delta  # right edge unchanged
        ax_net.set_position([new_x0, bb_n.y0, new_w, bb_n.height])

    # Re-render to capture the moved positions in subsequent bbox queries.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ne_xmin = min(ax.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()).x0
        for ax in [ax_ne_heat, ax_ne_net, ax_ne_gap])
    x_split_ne = ne_xmin - 0.060

    def _bg_box_clipped(axes_list, color, xmin_clip, xmax_clip, pad=0.016,
                        extra_bottom=0.18, extra_top=0.07,
                        extra_left=0.0, extra_right=0.0):
        xmins, ymins, xmaxs, ymaxs = [], [], [], []
        for ax in axes_list:
            bb = ax.get_window_extent(renderer=renderer)
            bb_fig = bb.transformed(fig.transFigure.inverted())
            xmins.append(bb_fig.x0); ymins.append(bb_fig.y0)
            xmaxs.append(bb_fig.x1); ymaxs.append(bb_fig.y1)
        x0 = max(min(xmins) - pad - extra_left, xmin_clip)
        y0 = min(ymins) - pad - extra_bottom
        x1 = min(max(xmaxs) + pad + extra_right, xmax_clip)
        y1 = max(ymaxs) + pad + extra_top
        rect = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
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
# 13.  FIG 1 — Global Risk + Sensitivity (4 rows x 4 cols), per dataset
# ===========================================================================

def fig1_global_risk_sensitivity(ds, global_effects, fs=None):
    if fs is None: fs = _fs(0)
    features = ds['features']; p = len(features); T = ds['T']
    tag = ds['tag']
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
                             gridspec_kw={'width_ratios':[3,3,3,1.8]})
    fig.suptitle(
        'Global Sensitivity and Risk effects — pure / partial / full\n'
        '(averaged over {} instances) — {}'.format(
            GLOBAL_N_INSTANCES, DS_LABEL[tag].replace('\n','  ')),
        fontsize=fs.suptitle, fontweight='bold')

    top_k = 5; top_feats = None
    for r, (gtype, K, klabel, eff_labels, row_title) in enumerate(row_specs):
        # 5-tuple now: (avg_shap, avg_pure, avg_full, avg_pairs, avg_mob)
        avg_shap, avg_pure, avg_full, _, _ = global_effects[gtype]
        effect_dicts = {'pure':avg_pure, 'partial':avg_shap, 'full':avg_full}
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
# 14.  FIG 2 — Local Prediction (2 rows x 4 cols), per dataset
# ===========================================================================

def fig2_local_prediction(ds, local_games, fs=None):
    if fs is None: fs = _fs(0)
    features = ds['features']; p = len(features); T = ds['T']
    tag = ds['tag']
    K_id   = kernel_identity(T)
    K_corr = ds['K_corr']

    mob, shap, pure_eff, full_eff = (
        local_games['prediction'][0], local_games['prediction'][1],
        local_games['prediction'][2], local_games['prediction'][3])

    row_specs = [
        (K_id,   'identity',    'Identity kernel'),
        (K_corr, 'correlation', 'Correlation kernel'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 4.2*2),
                             gridspec_kw={'width_ratios':[3,3,3,1.8]})
    fig.suptitle(
        'Local Prediction effects — pure / partial / full\n{}'.format(
            DS_LABEL[tag].replace('\n','  ')),
        fontsize=fs.suptitle, fontweight='bold')

    top_k = 5; top_feats = None
    for r, (K, klabel, row_title) in enumerate(row_specs):
        effect_dicts = {'pure':pure_eff, 'partial':shap, 'full':full_eff}
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
                ax.text(-0.30, 0.5,
                        'Prediction\n{}'.format(row_title),
                        transform=ax.transAxes,
                        fontsize=fs.axis-1, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')
            _shade(ax, ds)

        neso_fmt = (matplotlib.ticker.FuncFormatter(
                       lambda x, _: '{:.0f}k'.format(x/1000))
                   if tag == 'neso' else None)
        _draw_bar(axes[r, 3], effect_dicts, K, features, fs,
                  x_fmt=neso_fmt)
        _align_row([axes[r, c] for c in range(3)])

    _add_bottom_legends_energy(fig, features, top_feats, fs,
                                feat_bbox=(0.04, 0.030),
                                kern_bbox=(0.04, -0.010))
    plt.tight_layout(rect=[0.04, 0.10, 1, 0.91])
    fig.subplots_adjust(hspace=0.65, top=0.88)
    return fig


# ===========================================================================
# 15.  FIG 3 — Global Prediction PDP-style (4 rows x 3 cols), per dataset
# ===========================================================================

def _top2_features(global_effects, features):
    p     = len(features)
    K_id  = kernel_identity(len(global_effects['prediction'][0][0]))
    avg_shap = global_effects['prediction'][0]
    imps = {i: float(np.sum(np.abs(apply_kernel(avg_shap[i], K_id))))
            for i in range(p)}
    return sorted(imps, key=imps.get, reverse=True)[:2]


def _pdp_panel_energy(ax, fi, etype, K, X_background, per_instance,
                      features, selected_t_idxs, t_cmap, ds, fs):
    feat_name  = features[fi]
    feat_vals  = X_background[:, fi]
    n_inst     = len(per_instance)

    if feat_name in CONTINUOUS_FEATURES:
        fmin, fmax = feat_vals.min(), feat_vals.max()
        bins        = np.linspace(fmin, fmax, N_FEAT_BINS+1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_idx     = np.clip(np.digitize(feat_vals, bins)-1,
                              0, N_FEAT_BINS-1)
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
        effects = np.array([
            apply_kernel(per_instance[k][etype][fi], K)
            for k in np.where(mask)[0]])
        bin_effects[b] = effects.mean(axis=0)

    valid  = ~np.isnan(bin_effects[:, 0])
    n_t    = len(selected_t_idxs)
    colors = [t_cmap(i/(n_t-1)) for i in range(n_t)]
    for ti, col in zip(selected_t_idxs, colors):
        ax.plot(bin_centers[valid], bin_effects[valid, ti],
                color=col, lw=1.4, ls='--', alpha=0.85,
                label=ds['tlabels'][ti])
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
                matplotlib.ticker.FuncFormatter(
                    lambda v, _: '{:.0f}'.format(v)))
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


def fig3_global_pdp(ds, global_effects, per_instance, X_background, fs=None):
    if fs is None: fs = _fs(0)
    features = ds['features']; tag = ds['tag']
    K_id   = kernel_identity(ds['T'])
    K_corr = ds['K_corr']

    top2 = _top2_features(global_effects, features)

    T = ds['T']
    selected_t_idxs = [0, T//5, 2*T//5, 3*T//5, T-1]
    t_cmap = cm.get_cmap('plasma', len(selected_t_idxs))

    feat_row_specs = []
    for fi in top2:
        feat_row_specs.append((fi, K_id,   'Identity kernel'))
        feat_row_specs.append((fi, K_corr, 'Correlation kernel'))

    fig, axes = plt.subplots(4, 3, figsize=(18, 3.8*4))
    fig.suptitle(
        'Global Prediction effects — PDP-style — pure / partial / full\n'
        '(per-instance effects, binned over feature range) — {}'.format(
            DS_LABEL[tag].replace('\n','  ')),
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
                ax.text(-0.22, 0.5,
                        '{}\n{}'.format(features[fi], klabel),
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
# 16.  FIG 4 — Local Interactions (2 rows x 2 cols), per dataset
# ===========================================================================

def fig4_interactions(ds, local_games, fs=None):
    if fs is None: fs = _fs(0)
    ax_tick_fs  = max(fs.tick-2, 6)
    ax_label_fs = max(fs.axis-1, 7)

    features = ds['features']; p = len(features); T = ds['T']
    tag = ds['tag']
    K_id   = kernel_identity(T)
    K_corr = ds['K_corr']

    mob = local_games['prediction'][0]
    PAIR_COLORS = ['#e63946','#2a9d8f','#8338ec','#fb8500','#457b9d']

    pair_imp = {}
    for i in range(p):
        for j in range(i+1, p):
            raw = mob.get((i,j), np.zeros(T))
            pair_imp[(i,j)] = float(np.sum(np.abs(apply_kernel(raw, K_corr))))
    top5 = sorted(pair_imp, key=pair_imp.get, reverse=True)[:5]

    fig, axes = plt.subplots(2, 2, figsize=(12, 4.2*2),
                             gridspec_kw={'width_ratios':[3,1.8]})
    fig.suptitle(
        'Local pairwise interaction effects \u2014 top-5 pairs'
        '\n{}'.format(DS_LABEL[tag].replace('\n','  ')),
        fontsize=fs.suptitle, fontweight='bold')

    row_specs = [
        (K_id,   'identity',    'Interactions — Identity kernel'),
        (K_corr, 'correlation', 'Interactions — Correlation kernel'),
    ]

    for r, (K, klabel, row_label) in enumerate(row_specs):
        ax = axes[r, 0]
        for pair_idx, (i, j) in enumerate(top5):
            raw  = mob.get((i,j), np.zeros(T))
            ls   = '--' if klabel == 'correlation' else '-'
            ax.plot(ds['t_grid'], apply_kernel(raw, K),
                    color=PAIR_COLORS[pair_idx], lw=1.8, ls=ls)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _shade(ax, ds)
        ax.set_xticks(list(range(0, T, max(1, T//8)*2)))
        ax.set_xticklabels([ds['tlabels'][i]
                            for i in range(0, T, max(1, T//8)*2)],
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
        ax_bar.barh(
            y_pos,
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
        for k, (i,j) in enumerate(top5)]
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
# 17.  FIG 5 — Network plots global risk + sensitivity, shared (4 rows x 3)
# ===========================================================================

def fig5_networks_global(ds_ih, ds_ne, global_ih, global_ne, K_ih, K_ne):
    """Sensitivity-only network plots: IHEPC + NESO, pure + partial.
    Uses correct Shapley interaction index edges via avg_mob."""
    row_specs = [
        (ds_ih, global_ih, K_ih, 'IHEPC Sensitivity'),
        (ds_ne, global_ne, K_ne, 'NESO Sensitivity'),
    ]
    col_labels = ['Pure', 'Partial']
    col_etypes = ['pure', 'partial']

    fig = plt.figure(figsize=(8.5, 9))
    fig.suptitle(
        'Global sensitivity network plots — correlation kernel\n'
        'IHEPC and NESO',
        fontsize=FS_SUPTITLE, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.12, wspace=0.10,
                           left=0.10, right=0.98,
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


# 18.  Main
# ===========================================================================

if __name__ == '__main__':
    print('\n'+'='*60)
    print('  Energy Combined Example  (IHEPC + NESO)  v7')
    print('='*60)

    _require_dir(BASE_PLOT_DIR)
    _require_dir(GAME_CACHE_DIR)

    print('\n[1] Loading data ...')
    ds_ih = load_ihepc()
    ds_ne = load_neso()

    print('\n[2] Fitting models ...')
    for ds, name in [(ds_ih,'IHEPC'), (ds_ne,'NESO')]:
        X_tr,X_te,Y_tr,Y_te = train_test_split(
            ds['X_np'], ds['Y_adj'], test_size=0.2, random_state=RNG_SEED)
        m = RFModel(); m.fit(X_tr, Y_tr)
        print('  [{}] Test R2: {:.4f}'.format(name, m.evaluate(X_te,Y_te)))
        ds['model'] = m

    print('\n[3] Building correlation kernels ...')
    K_ih = kernel_correlation(ds_ih['Y_raw']); ds_ih['K_corr'] = K_ih
    K_ne = kernel_correlation(ds_ne['Y_raw']); ds_ne['K_corr'] = K_ne

    print('\n[4] Selecting profiles ...')
    X_ih = ds_ih['X_np']; fn_ih = ds_ih['features']
    def find_ih(conds, lbl):
        mask = np.ones(len(X_ih), dtype=bool)
        for f,(lo,hi) in conds.items():
            ci = fn_ih.index(f)
            mask &= (X_ih[:,ci]>=lo)&(X_ih[:,ci]<=hi)
        hits = X_ih[mask]
        if not len(hits): raise RuntimeError('No match: {}'.format(lbl))
        print('  IHEPC "{}": {} days'.format(lbl, len(hits)))
        return hits[len(hits)//2]
    x_ih1 = find_ih({'is_weekend':(-0.1,0.1),'day_of_week':(0.9,4.1)},
                    'Typical weekday')

    X_ne = ds_ne['X_np']; fn_ne = ds_ne['features']
    def find_ne(conds, lbl):
        mask = np.ones(len(X_ne), dtype=bool)
        for f,(lo,hi) in conds.items():
            ci = fn_ne.index(f)
            mask &= (X_ne[:,ci]>=lo)&(X_ne[:,ci]<=hi)
        hits = X_ne[mask]
        if not len(hits): raise RuntimeError('No match: {}'.format(lbl))
        print('  NESO "{}": {} days'.format(lbl, len(hits)))
        return hits[len(hits)//2]
    x_ne1 = find_ne({'is_weekend':(-0.1,0.1),'season':(0.9,1.1)},
                    'Winter weekday')
    print(dict(zip(fn_ih, x_ih1)))
    print(dict(zip(fn_ne, x_ne1)))

    print('\n[5] Loading / computing local games ...')
    local_ih = load_local_games(ds_ih, x_ih1, 'Typical_weekday')
    local_ne = load_local_games(ds_ne, x_ne1, 'Winter_weekday')

    mob_ih  = {gt: local_ih[gt][0] for gt in GAME_TYPES}
    shap_ih = {gt: local_ih[gt][1] for gt in GAME_TYPES}
    mob_ne  = {gt: local_ne[gt][0] for gt in GAME_TYPES}
    shap_ne = {gt: local_ne[gt][1] for gt in GAME_TYPES}

    print('\n[6] Computing / loading global effects ({} instances) ...'.format(
        GLOBAL_N_INSTANCES))
    global_ih = {}; global_ne = {}
    for gtype in GAME_TYPES:
        print('\n  IHEPC global {} ...'.format(gtype))
        global_ih[gtype] = compute_global_effects(
            ds_ih, gtype, GLOBAL_N_INSTANCES, GLOBAL_SAMPLE_SIZE, RNG_SEED)
        print('\n  NESO global {} ...'.format(gtype))
        global_ne[gtype] = compute_global_effects(
            ds_ne, gtype, GLOBAL_N_INSTANCES, GLOBAL_SAMPLE_SIZE, RNG_SEED)

    print('\n[7] Loading / computing per-instance prediction effects (PDP) ...')
    per_inst_ih, X_pdp_ih = load_per_instance_effects_pdp(ds_ih, RNG_SEED)
    per_inst_ne, X_pdp_ne = load_per_instance_effects_pdp(ds_ne, RNG_SEED)

    print('\n[8] Generating figures ...')

    savefig(
        fig0_main_body(ds_ih, ds_ne, mob_ih, shap_ih, mob_ne, shap_ne,
                       K_ih, K_ne,
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

    for ds, global_eff, per_inst, X_pdp, tag in [
        (ds_ih, global_ih, per_inst_ih, X_pdp_ih, 'ihepc'),
        (ds_ne, global_ne, per_inst_ne, X_pdp_ne, 'neso'),
    ]:
        savefig(
            fig3_global_pdp(ds, global_eff, per_inst, X_pdp, fs=_fs(3)),
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

    print('\n'+'='*60)
    print('  Done.  Figures in {}/'.format(BASE_PLOT_DIR))
    print('  Game cache in {}/'.format(GAME_CACHE_DIR))
    print('='*60)