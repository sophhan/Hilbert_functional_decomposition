"""
Functional Explanation Framework -- Combined Energy Example  (v5)
=================================================================
Changes vs v4:
  - Game result caching: .npz per instance per game type per dataset.
    On re-run only plots are regenerated. Bump CACHE_VERSION to recompute.
  - Per-instance prediction effects cached for PDP plots (fig3).
  - New figure set matching SPY v7 structure:
      fig0 : single-row layout (heatmap, network, gap x2 datasets)
      fig1  : global risk + sensitivity (4 rows x 4 cols), per dataset
      fig2  : local prediction (2 rows x 4 cols), per dataset
      fig3  : global prediction PDP-style (4 rows x 3 cols), per dataset
      fig4  : local interactions (2 rows x 2 cols), per dataset
      fig5  : network plots global risk+sensitivity shared IHEPC+NESO
  - Discrete features use actual distinct values as PDP bin centers.
  - Font-size bundle _fs(bump) for per-figure font scaling.
"""

import itertools
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
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
CACHE_VERSION = 'v5'

_HERE    = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 42
RF_N_EST = 300
RF_JOBS  = -1

BASE_PLOT_DIR  = os.path.join('plots', 'energy_cor_games')
GAME_CACHE_DIR = os.path.join(_HERE, 'game_results_energy')

GLOBAL_N_INSTANCES = 30
GLOBAL_SAMPLE_SIZE = 100
N_FEAT_BINS        = 20   # max bins for continuous features in PDP

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
    'pure'   : r'Pure $m_i(t)$',
    'partial': r'Partial $\phi_i(t)$',
    'full'   : r'Full $\Phi_i(t)$',
}
_GLOBAL_PRED_LABELS = {
    'pure'   : r'Pure $m_i(x_j)$ = PDP',
    'partial': r'Partial $\phi_i(x_j)$ = global SHAP',
    'full'   : r'Full $\Phi_i(x_j)$',
}
_EFFECT_TYPES = ['pure', 'partial', 'full']


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
    """Average pure/partial/full over n_instances, using per-instance cache."""
    _require_dir(GAME_CACHE_DIR)
    X_bg = ds['X_np']; Y_adj = ds['Y_adj']
    T = ds['T']; features = ds['features']; n_players = len(features)
    rng  = np.random.default_rng(seed)
    idxs = rng.choice(len(X_bg), size=n_instances, replace=False)
    sum_shap = {i: np.zeros(T) for i in range(n_players)}
    sum_pure = {i: np.zeros(T) for i in range(n_players)}
    sum_full = {i: np.zeros(T) for i in range(n_players)}
    for k, idx in enumerate(idxs):
        cache = _cache_path_global(ds['tag'], game_type, k)
        x_inst = X_bg[idx]
        _, pure, shap, full, _ = _run_or_load(
            ds['model'].predict, X_bg, Y_adj,
            x_inst, game_type, sample_size, seed+k, cache, features, T)
        for i in range(n_players):
            sum_shap[i] += shap[i]
            sum_pure[i] += pure[i]
            sum_full[i] += full[i]
        status = '(cached)' if os.path.isfile(cache) else ''
        print('    [{} {} global] {}/{} {}'.format(
            ds['tag'], game_type, k+1, n_instances, status))
    return ({i: sum_shap[i]/n_instances for i in range(n_players)},
            {i: sum_pure[i]/n_instances for i in range(n_players)},
            {i: sum_full[i]/n_instances for i in range(n_players)})


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
    # Compute all three games for local (needed for figs 2 and 4)
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
        # Cache each game type separately
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
    p = len(features)
    imps = {et: {i: float(np.sum(np.abs(apply_kernel(effect_dicts[et][i], K))))
                 for i in range(p)}
            for et in _EFFECT_TYPES}
    order   = sorted(range(p), key=lambda i: imps['partial'][i], reverse=True)
    y_pos   = np.arange(len(order))
    bar_h   = 0.25
    offsets = {'pure':-bar_h, 'partial':0.0, 'full':bar_h}
    alphas  = {'pure':0.55,   'partial':0.90, 'full':0.55}
    hatches = {'pure':'//',   'partial':'',   'full':'\\\\'}
    for et in _EFFECT_TYPES:
        ax.barh(y_pos+offsets[et],
                [imps[et][i] for i in order], height=bar_h,
                color=[FEAT_COLORS[features[i]] for i in order],
                alpha=alphas[et], hatch=hatches[et], label=et)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([features[i] for i in order], fontsize=fs.tick)
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=fs.axis)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=fs.tick)
    ax.set_title('Time-aggregated', fontsize=fs.title, fontweight='bold')
    if x_fmt is not None:
        ax.xaxis.set_major_formatter(x_fmt)
    leg_kwargs = dict(fontsize=fs.legend, loc='upper left',
                      bbox_to_anchor=(1.02, 1.0),
                      bbox_transform=ax.transAxes,
                      borderaxespad=0., framealpha=0.9)
    ax.legend(**leg_kwargs)

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
                raw = np.zeros(T)
                for S, m in mob.items():
                    if i in S and j in S: raw = raw + m/len(S)
            else:
                raw = np.zeros(T)
                for S, m in mob.items():
                    if i in S and j in S: raw = raw + m
            val = float(np.trapz(apply_kernel(raw, K), t_grid))
            if abs(val) > 0: edge_imp[(i,j)] = val
    return node_imp, edge_imp, node_sign

def _draw_network(ax, features, node_imp, edge_imp, node_sign,
                  title, fs_title=None):
    import math
    fs_t = fs_title if fs_title is not None else FS_TITLE
    p = len(features)
    angle = [math.pi/2 - 2*math.pi*i/p for i in range(p)]
    pos   = {i: (math.cos(a), math.sin(a)) for i, a in enumerate(angle)}
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title, fontsize=fs_t, fontweight='bold', pad=4)
    max_imp  = float(node_imp.max()) if node_imp.max() > 0 else 1.0
    node_r   = {i: 0.07 + 0.19*(node_imp[i]/max_imp) for i in range(p)}
    max_edge = max((abs(v) for v in edge_imp.values()), default=1.0)
    max_edge = max(max_edge, 1e-12)
    for (i,j), val in edge_imp.items():
        xi,yi = pos[i]; xj,yj = pos[j]
        lw  = 0.4 + 6.5*abs(val)/max_edge
        col = _EDGE_SYN if val > 0 else _EDGE_RED
        alph = 0.30 + 0.60*abs(val)/max_edge
        ax.plot([xi,xj],[yi,yj], color=col, lw=lw, alpha=alph,
                solid_capstyle='round', zorder=1)
    for i in range(p):
        x, y = pos[i]; r = node_r[i]
        fc = _NODE_POS if node_sign[i] >= 0 else _NODE_NEG
        ax.add_patch(plt.Circle((x,y), r, color=fc, ec='white',
                                linewidth=1.2, zorder=2, alpha=0.88))
        ax.add_patch(plt.Circle((x,y), r*0.52, color='white', ec='none',
                                zorder=3, alpha=0.95))
        abbr = FEAT_ABBR.get(features[i], features[i][:3])
        ax.text(x, y, abbr, ha='center', va='center',
                fontsize=max(4.5, r*22), fontweight='bold',
                color='#222', zorder=4)
    pad = 0.32
    ax.set_xlim(-1.0-pad, 1.0+pad); ax.set_ylim(-1.0-pad, 1.0+pad)


# ===========================================================================
# 12.  FIG 0 — Single-row: heatmap + network + gap, per dataset
# ===========================================================================

def _add_bg_box(fig, axes_list, color, pad=0.014,
                extra_bottom=0.0, extra_top=0.0):
    """Draw colored background box. extra_bottom/extra_top extend in
    figure-fraction units to cover below-axes legends and above-axes titles."""
    renderer = fig.canvas.get_renderer()
    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    for ax in axes_list:
        bb = ax.get_window_extent(renderer=renderer)
        bb_fig = bb.transformed(fig.transFigure.inverted())
        xmins.append(bb_fig.x0); ymins.append(bb_fig.y0)
        xmaxs.append(bb_fig.x1); ymaxs.append(bb_fig.y1)
    x0 = min(xmins) - pad
    y0 = min(ymins) - pad - extra_bottom
    w  = max(xmaxs) - x0 + pad
    h  = max(ymaxs) - y0 + pad + extra_top
    rect = FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.006',
                          linewidth=1.4, edgecolor=color, facecolor=color,
                          alpha=0.10, transform=fig.transFigure,
                          zorder=0, clip_on=False)
    fig.add_artist(rect)


def fig0_main_body(ds_ih, ds_ne, mob_ih, shap_ih, mob_ne, shap_ne,
                   K_ih, K_ne, global_sens_ih, global_sens_ne):
    """
    Single row, 6 panels:
    [IHEPC heatmap | IHEPC sensitivity network | IHEPC sensitivity gap |
     NESO heatmap  | NESO sensitivity network  | NESO sensitivity gap  ]
    """
    FS_SUP=17; FS_T=14; FS_AX=13; FS_TK=11; FS_LEG=11

    ID_ALPHA=0.40; ID_LW=1.6; MX_LW=2.2

    fig = plt.figure(figsize=(28, 5.5))
    gs  = GridSpec(1, 7, figure=fig,
                   width_ratios=[1.4, 1.0, 1.6, 0.18, 1.4, 1.0, 1.6],
                   wspace=0.28, left=0.04, right=0.98,
                   top=0.78, bottom=0.20)

    ax_ih_heat = fig.add_subplot(gs[0])
    ax_ih_net  = fig.add_subplot(gs[1])
    ax_ih_gap  = fig.add_subplot(gs[2])
    ax_gap_spacer = fig.add_subplot(gs[3])  # invisible spacer
    ax_gap_spacer.set_visible(False)
    ax_ne_heat = fig.add_subplot(gs[4])
    ax_ne_net  = fig.add_subplot(gs[5])
    ax_ne_gap  = fig.add_subplot(gs[6])

    fig.suptitle(
        'Energy demand: correlation structure drives explanation shape\n'
        'UCI IHEPC (single household, kW)  vs  NESO GB Demand (national grid, MW)',
        fontsize=FS_SUP, fontweight='bold', y=0.98)

    # ── Heatmaps ──────────────────────────────────────────────────────────
    def _heatmap(ax, ds, K, tag):
        T, tl = ds['T'], ds['tlabels']
        step  = max(1, T//6); ticks = list(range(0, T, step))
        im = ax.imshow(K, aspect='equal', origin='upper',
                       cmap='RdBu_r', vmin=-0.2, vmax=1.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels([tl[i] for i in ticks],
                           rotation=45, ha='right', fontsize=9.5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([tl[i] for i in ticks], fontsize=9.0)
        ax.set_title(DS_LABEL[tag].replace('\n',' ') + '\ncorrelation kernel $K$',
                     fontsize=FS_AX, fontweight='bold', color=DS_COLOR[tag])
        am = (ds['morning'][0]+ds['morning'][1])//2
        ax.axhline(am, color='white', lw=0.8, ls='--', alpha=0.6)
        ax.axvline(am, color='white', lw=0.8, ls='--', alpha=0.6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=9.0)

    _heatmap(ax_ih_heat, ds_ih, K_ih, 'ihepc')
    _heatmap(ax_ne_heat, ds_ne, K_ne, 'neso')

    # ── Sensitivity networks (partial, correlation kernel) ─────────────
    net_handles = [Patch(facecolor=_NODE_POS, edgecolor='none', label='Positive'),
                   Patch(facecolor=_NODE_NEG, edgecolor='none', label='Negative')]
    for ax, mob, shap, ds, tag in [
        (ax_ih_net, mob_ih['sensitivity'], shap_ih['sensitivity'], ds_ih, 'ihepc'),
        (ax_ne_net, mob_ne['sensitivity'], shap_ne['sensitivity'], ds_ne, 'neso'),
    ]:
        features = ds['features']; p, T = len(features), ds['T']
        K = K_ih if tag == 'ihepc' else K_ne
        ni, ei, ns = _network_importances(mob, shap, p, T, K, 'partial')
        _draw_network(ax, features, ni, ei, ns,
                      '{} sens.\npartial (corr)'.format(tag.upper()),
                      fs_title=FS_T)
        ax.legend(handles=net_handles, loc='lower center', ncol=2,
                  fontsize=FS_LEG-1.5, framealpha=0.88,
                  bbox_to_anchor=(0.5,-0.12), bbox_transform=ax.transAxes,
                  borderpad=0.4, handlelength=1.2)

    # ── Sensitivity gap panels ─────────────────────────────────────────
    def _gap_panel(ax, ds, mob_sens, K_corr, K_id, tag):
        features = ds['features']; p, T = len(features), ds['T']
        t_grid = ds['t_grid']
        pure_eff = _pure(mob_sens, p, T)
        full_eff = _full(mob_sens, p, T)
        gap = {i: full_eff[i] - pure_eff[i] for i in range(p)}
        gap_imp = {i: float(np.sum(np.abs(apply_kernel(gap[i], K_corr))))
                   for i in range(p)}
        fi  = max(gap_imp, key=gap_imp.get)
        col = FEAT_COLORS[features[fi]]
        pure_id = apply_kernel(pure_eff[fi], K_id)
        full_id = apply_kernel(full_eff[fi], K_id)
        pure_c  = apply_kernel(pure_eff[fi], K_corr)
        full_c  = apply_kernel(full_eff[fi], K_corr)
        gap_c   = apply_kernel(gap[fi], K_corr)
        # Identity kernel — faded
        ax.plot(t_grid, full_id, color=col, lw=ID_LW, ls='-', alpha=ID_ALPHA, zorder=2)
        ax.plot(t_grid, pure_id, color=col, lw=ID_LW, ls='--', alpha=ID_ALPHA, zorder=2)
        # Correlation kernel — full opacity
        ax.fill_between(t_grid, pure_c, full_c,
                        color=col, alpha=0.20, label='gap region', zorder=1)
        ax.plot(t_grid, full_c, color=col, lw=MX_LW, ls='-',
                label=r'Full (Total Sobol)', zorder=3)
        ax.plot(t_grid, pure_c, color=col, lw=MX_LW, ls='--',
                label=r'Pure (Closed Sobol)', zorder=3)
        ax.plot(t_grid, gap_c, color='black', lw=1.4, ls=':', alpha=0.7,
                label=r'Gap $\Delta\tau_i$', zorder=3)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _xticks(ax, ds, sparse=True)
        ax.tick_params(labelsize=FS_TK)
        ax.set_xlabel('Time', fontsize=FS_AX)
        ax.set_ylabel(ds['ylabel']['sensitivity'], fontsize=FS_AX)
        integ = float(np.trapz(np.abs(gap_c), t_grid))
        ax.set_title(
            'Sensitivity gap — corr. kernel\n'
            '{} — {} | $\\int|\\Delta\\tau_i|\\,dt={:.3g}$'.format(
                DS_LABEL[tag].split('\n')[0], features[fi], integ),
            fontsize=FS_T-1, fontweight='bold', color=DS_COLOR[tag])
        extra_id = Line2D([0],[0], color='gray', lw=ID_LW, ls='-',
                          alpha=ID_ALPHA, label='identity (faded)')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles+[extra_id], labels+['identity (faded)'],
                  fontsize=FS_LEG, loc='upper center',
                  bbox_to_anchor=(0.5,-0.22), ncol=3, framealpha=0.85)
        _shade(ax, ds)

    def _sens_partial_panel(ax, ds, global_sens, K_corr, K_id, tag,
                             force_features=None):
        """Show global Shapley-sensitivity partial effect for 1-2 features:
        identity kernel faded, correlation kernel full opacity."""
        features = ds['features']; p, T = len(features), ds['T']
        t_grid = ds['t_grid']
        avg_shap = global_sens[0]  # partial (Shapley-sensitivity)
        if force_features is None:
            # Pick highest-importance feature under correlation kernel
            imps = {i: float(np.sum(np.abs(apply_kernel(avg_shap[i], K_corr))))
                    for i in range(p)}
            fis = [max(imps, key=imps.get)]
        else:
            fis = [features.index(f) for f in force_features]

        for fi in fis:
            col      = FEAT_COLORS[features[fi]]
            eff_id   = apply_kernel(avg_shap[fi], K_id)
            eff_corr = apply_kernel(avg_shap[fi], K_corr)
            ls = '-' if fi == fis[0] else '--'
            ax.plot(t_grid, eff_id, color=col, lw=ID_LW, ls=ls,
                    alpha=ID_ALPHA, zorder=2)
            ax.plot(t_grid, eff_corr, color=col, lw=MX_LW, ls=ls,
                    label=features[fi] + ' (corr.)', zorder=3)

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _xticks(ax, ds, sparse=True)
        ax.tick_params(labelsize=FS_TK)
        ax.set_xlabel('Time', fontsize=FS_AX)
        # Suppress scientific notation offset; fold scale into label
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
            r'Partial $\phi_i$ = Shapley-sens. — corr. kernel' + '\n'
            '{} — {}'.format(DS_LABEL[tag].split('\n')[0], feat_str),
            fontsize=FS_T-1, fontweight='bold', color=DS_COLOR[tag])
        # Build legend: corr lines + one faded identity line
        corr_handles = [
            Line2D([0],[0], color=FEAT_COLORS[features[fi]],
                   lw=MX_LW, ls='-' if fi == fis[0] else '--',
                   label=features[fi] + ' (corr.)')
            for fi in fis]
        extra_id = Line2D([0],[0], color='gray', lw=ID_LW, ls='-',
                          alpha=ID_ALPHA, label='identity (faded)')
        ax.legend(handles=corr_handles + [extra_id],
                  fontsize=FS_LEG, loc='upper center',
                  bbox_to_anchor=(0.5, -0.23), ncol=len(fis)+1,
                  framealpha=0.85)
        _shade(ax, ds)

    K_id_ih = kernel_identity(ds_ih['T'])
    K_id_ne = kernel_identity(ds_ne['T'])
    _sens_partial_panel(ax_ih_gap, ds_ih, global_sens_ih, K_ih, K_id_ih,
                        'ihepc', force_features=['lag_daily_mean', 'lag_morning'])
    _sens_partial_panel(ax_ne_gap, ds_ne, global_sens_ne, K_ne, K_id_ne,
                        'neso',  force_features=['month', 'season'])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get NESO left edge for box clipping
    ne_xmin = min(ax.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()).x0
        for ax in [ax_ne_heat, ax_ne_net, ax_ne_gap])
    x_split_ne = ne_xmin - 0.018   # NESO box left edge

    def _bg_box_clipped(axes_list, color, xmin_clip, xmax_clip, pad=0.016,
                        extra_bottom=0.18, extra_top=0.07, extra_right=0.0):
        """Box covering axes group, clipped to [xmin_clip, xmax_clip],
        extended downward to cover below-axes legends and upward for titles."""
        xmins, ymins, xmaxs, ymaxs = [], [], [], []
        for ax in axes_list:
            bb = ax.get_window_extent(renderer=renderer)
            bb_fig = bb.transformed(fig.transFigure.inverted())
            xmins.append(bb_fig.x0); ymins.append(bb_fig.y0)
            xmaxs.append(bb_fig.x1); ymaxs.append(bb_fig.y1)
        x0 = max(min(xmins) - pad, xmin_clip)
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
                    DS_COLOR['neso'], xmin_clip=x_split_ne, xmax_clip=1.0)
    return fig


# ===========================================================================
# 13.  FIG 1 — Global Risk + Sensitivity (4 rows x 4 cols), per dataset
# ===========================================================================

def fig1_global_risk_sensitivity(ds, global_effects, fs=None):
    """
    4 rows x 4 cols:
      row 0: risk,        identity kernel
      row 1: risk,        correlation kernel
      row 2: sensitivity, identity kernel
      row 3: sensitivity, correlation kernel
    """
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
        avg_shap, avg_pure, avg_full = global_effects[gtype]
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

    # Stacked bottom legends
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
    """
    2 rows x 4 cols:
      row 0: prediction, identity kernel
      row 1: prediction, correlation kernel
    """
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
    """Select top-2 features by global SHAP importance under identity kernel.
    Identity kernel gives well-distributed importance across all feature types
    and avoids empty bins in PDP plots caused by correlation-kernel weighting."""
    p     = len(features)
    K_id  = kernel_identity(len(global_effects['prediction'][0][0]))
    avg_shap = global_effects['prediction'][0]
    imps = {i: float(np.sum(np.abs(apply_kernel(avg_shap[i], K_id))))
            for i in range(p)}
    return sorted(imps, key=imps.get, reverse=True)[:2]


def _pdp_panel_energy(ax, fi, etype, K, X_background, per_instance,
                      features, selected_t_idxs, t_cmap, ds, fs):
    """PDP-style panel. Discrete features use distinct values as bin centers."""
    feat_name = features[fi]
    feat_vals = X_background[:, fi]
    n_inst    = len(per_instance)

    if feat_name in CONTINUOUS_FEATURES:
        fmin, fmax = feat_vals.min(), feat_vals.max()
        bins        = np.linspace(fmin, fmax, N_FEAT_BINS+1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_idx     = np.clip(np.digitize(feat_vals, bins)-1, 0, N_FEAT_BINS-1)
        n_bins      = N_FEAT_BINS
    else:
        unique_vals = np.sort(np.unique(feat_vals))
        bin_centers = unique_vals
        n_bins      = len(unique_vals)
        bin_idx     = np.array([np.argmin(np.abs(unique_vals - v))
                                for v in feat_vals])

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
    # Lock x-axis to actual data range — critical for discrete features
    # (prevents any subsequent call from expanding the axis)
    if valid.any():
        xlo = bin_centers[valid].min()
        xhi = bin_centers[valid].max()
        xrange = xhi - xlo
        if feat_name in CONTINUOUS_FEATURES:
            # Small proportional margin for continuous features
            margin = xrange * 0.03
        else:
            # Fixed half-step margin for discrete features
            margin = 0.3
        ax.set_xlim(xlo - margin, xhi + margin)
        ax.autoscale(False, axis='x')
    ax.tick_params(labelsize=fs.tick)
    ax.xaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(nbins=6, prune='both'))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha('right'); lbl.set_fontsize(fs.tick)
    ax.set_xlabel(feat_name, fontsize=fs.axis)
    # No shading — x-axis is feature values, not time


def fig3_global_pdp(ds, global_effects, per_instance, X_background, fs=None):
    """
    4 rows x 3 cols (no bar panel):
      row 0: feature 1, identity kernel
      row 1: feature 1, correlation kernel
      row 2: feature 2, identity kernel
      row 3: feature 2, correlation kernel
    """
    if fs is None: fs = _fs(0)
    features = ds['features']; tag = ds['tag']
    K_id   = kernel_identity(ds['T'])
    K_corr = ds['K_corr']

    top2 = _top2_features(global_effects, features)

    # Selected time points: spread across the day
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

    # Legend: time points + time-aggregated — bottom center
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
    """
    2 rows x 2 cols:
      row 0: identity kernel
      row 1: correlation kernel
    Top-5 pairs as lines. Solid = identity, dashed = correlation.
    """
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
        'Local pairwise interaction effects $m_{{ij}}(t)$ \u2014 top-5 pairs'
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
        ax.set_title(r'$m_{ij}(t)$', fontsize=fs.title, fontweight='bold')
        ax.text(-0.18, 0.5, row_label, transform=ax.transAxes,
                fontsize=ax_label_fs-1, va='center', ha='right',
                rotation=90, color='#333', fontweight='bold')
        _align_row([axes[r, 0]])

        # Bar panel
        ax_bar = axes[r, 1]
        pair_imps, pair_lbls = [], []
        for i, j in top5:
            raw = mob.get((i,j), np.zeros(T))
            pair_imps.append(
                float(np.sum(np.abs(apply_kernel(raw, K)))))
            pair_lbls.append('{} x {}'.format(features[i], features[j]))
        y_pos = np.arange(len(top5))
        ax_bar.barh(y_pos, pair_imps,
                    color=PAIR_COLORS[:len(top5)], alpha=0.85)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(pair_lbls, fontsize=ax_tick_fs-1)
        ax_bar.set_xlabel(r'$\int|m_{ij}|\,dt$', fontsize=ax_label_fs)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=ax_tick_fs)
        ax_bar.set_title('Time-aggregated', fontsize=fs.title, fontweight='bold')

    # Stacked legends at bottom
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
    """
    4 rows x 3 cols:
      row 0: IHEPC sensitivity (partial, corr kernel)
      row 1: IHEPC risk        (partial, corr kernel)
      row 2: NESO  sensitivity (partial, corr kernel)
      row 3: NESO  risk        (partial, corr kernel)
    Columns: pure / partial / full
    """
    row_specs = [
        (ds_ih, global_ih, 'sensitivity', K_ih, 'IHEPC Sensitivity'),
        (ds_ih, global_ih, 'risk',        K_ih, 'IHEPC Risk'),
        (ds_ne, global_ne, 'sensitivity', K_ne, 'NESO Sensitivity'),
        (ds_ne, global_ne, 'risk',        K_ne, 'NESO Risk'),
    ]
    col_labels = [r'Pure $m_i$', r'Partial $\phi_i$', r'Full $\Phi_i$']

    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(
        'Global network plots — correlation kernel\n'
        'Sensitivity and Risk games — IHEPC and NESO',
        fontsize=FS_SUPTITLE, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.10, wspace=0.08,
                           left=0.09, right=0.98,
                           top=0.93, bottom=0.06)

    for r, (ds, g_eff, gtype, K, row_label) in enumerate(row_specs):
        features = ds['features']; p = len(features); T = ds['T']
        avg_shap, avg_pure, avg_full = g_eff[gtype]

        # Build pseudo-mob from global averages for network
        # (we only need partial for the network display)
        pseudo_shap = avg_shap
        pseudo_mob  = {(i,): avg_pure[i] for i in range(p)}
        pseudo_mob[()] = np.zeros(T)

        for c, etype in enumerate(_EFFECT_TYPES):
            ax = fig.add_subplot(gs[r, c])
            if etype == 'pure':
                eff = avg_pure
            elif etype == 'partial':
                eff = avg_shap
            else:
                eff = avg_full
            t_grid   = np.arange(T, dtype=float)
            node_imp = np.array([float(np.sum(np.abs(apply_kernel(eff[i], K))))
                                 for i in range(p)])
            node_sign= np.array([np.sign(float(np.trapz(
                apply_kernel(eff[i], K), t_grid))) for i in range(p)])
            # Edge importance: use global partial shap for pairwise
            edge_imp = {}
            for i in range(p):
                for j in range(i+1, p):
                    # Use product of node importances as proxy for edge
                    val = float(np.trapz(
                        apply_kernel(eff[i], K) * apply_kernel(eff[j], K),
                        t_grid))
                    if abs(val) > 0: edge_imp[(i,j)] = val
            _draw_network(ax, features, node_imp, edge_imp, node_sign,
                          col_labels[c] if r == 0 else '',
                          fs_title=FS_TITLE)
            if c == 0:
                ax.text(-0.03, 0.5, row_label,
                        transform=ax.transAxes,
                        fontsize=FS_AXIS, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')

    leg_handles = [
        Patch(facecolor=_NODE_POS, edgecolor='none', label='Positive effect'),
        Patch(facecolor=_NODE_NEG, edgecolor='none', label='Negative effect'),
    ]
    fig.legend(handles=leg_handles, loc='lower center', ncol=2,
               fontsize=FS_LEGEND, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))
    return fig


# ===========================================================================
# 18.  Main
# ===========================================================================

if __name__ == '__main__':
    print('\n'+'='*60)
    print('  Energy Combined Example  (IHEPC + NESO)  v5')
    print('='*60)

    _require_dir(BASE_PLOT_DIR)
    _require_dir(GAME_CACHE_DIR)

    # ── 1. Data ────────────────────────────────────────────────────────────
    print('\n[1] Loading data ...')
    ds_ih = load_ihepc()
    ds_ne = load_neso()

    # ── 2. Models ──────────────────────────────────────────────────────────
    print('\n[2] Fitting models ...')
    for ds, name in [(ds_ih,'IHEPC'), (ds_ne,'NESO')]:
        X_tr,X_te,Y_tr,Y_te = train_test_split(
            ds['X_np'], ds['Y_adj'], test_size=0.2, random_state=RNG_SEED)
        m = RFModel(); m.fit(X_tr, Y_tr)
        print('  [{}] Test R2: {:.4f}'.format(name, m.evaluate(X_te,Y_te)))
        ds['model'] = m

    # ── 3. Correlation kernels ─────────────────────────────────────────────
    print('\n[3] Building correlation kernels ...')
    K_ih = kernel_correlation(ds_ih['Y_raw']); ds_ih['K_corr'] = K_ih
    K_ne = kernel_correlation(ds_ne['Y_raw']); ds_ne['K_corr'] = K_ne

    # ── 4. Profiles ────────────────────────────────────────────────────────
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

    # ── 5. Local games ─────────────────────────────────────────────────────
    print('\n[5] Loading / computing local games ...')
    local_ih = load_local_games(ds_ih, x_ih1, 'Typical_weekday')
    local_ne = load_local_games(ds_ne, x_ne1, 'Winter_weekday')

    # Unpack for fig0 (uses mob/shap per game)
    mob_ih  = {gt: local_ih[gt][0] for gt in GAME_TYPES}
    shap_ih = {gt: local_ih[gt][1] for gt in GAME_TYPES}
    mob_ne  = {gt: local_ne[gt][0] for gt in GAME_TYPES}
    shap_ne = {gt: local_ne[gt][1] for gt in GAME_TYPES}

    # ── 6. Global games ────────────────────────────────────────────────────
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

    # ── 7. Per-instance prediction effects for PDP plots ──────────────────
    print('\n[7] Loading / computing per-instance prediction effects ...')
    per_inst_ih, X_glob_ih = load_per_instance_effects(ds_ih, RNG_SEED)
    per_inst_ne, X_glob_ne = load_per_instance_effects(ds_ne, RNG_SEED)

    # ── 8. Generate figures ────────────────────────────────────────────────
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

    for ds, global_eff, per_inst, X_glob, tag in [
        (ds_ih, global_ih, per_inst_ih, X_glob_ih, 'ihepc'),
        (ds_ne, global_ne, per_inst_ne, X_glob_ne, 'neso'),
    ]:
        savefig(
            fig3_global_pdp(ds, global_eff, per_inst, X_glob, fs=_fs(3)),
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