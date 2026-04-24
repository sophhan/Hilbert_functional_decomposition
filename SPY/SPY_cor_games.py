"""
Functional Explanation Framework -- Intraday SPY Volatility (Random Forest)
============================================================================
v5 changes vs v4:
  - Game result caching: results saved to game_results/ as .npz per instance;
    on re-run plots are regenerated without recomputing games if cache exists.
    Bump CACHE_VERSION to force recomputation.
  - fig2 interaction rows: up to 5 pairs plotted as lines per panel (same
    pairs across pure/partial/full columns).
  - Nomenclature: local prediction titles have no XAI equivalences; global
    prediction keeps "= PDP" / "= global SHAP"; sensitivity/risk keep their
    equivalences. All use plain "=" not the equiv sign.
  - Bar panel titles: "Time-aggregated" (not "Integrated").
  - Y-axes: aligned within each row only (cols 0-2), never across rows.
  - Plot 3: 3 columns only, no bar panel.
"""

import itertools
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0.  Settings
# ---------------------------------------------------------------------------
CACHE_VERSION = 'v5'          # bump to force recomputation of all games (unchanged)

_HERE          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(_HERE, 'data')
CACHE_PATH     = os.path.join(DATA_DIR, 'spy_5min_cache.csv')
VIX_CACHE_PATH = os.path.join(DATA_DIR, 'vix_daily_cache.csv')
GAME_CACHE_DIR = os.path.join(_HERE, 'game_results')

VIX_LOOKBACK_DAYS = 5
RNG_SEED  = 42
T_BARS    = 78
dt        = 1.0

RF_N_EST  = 300
RF_JOBS   = -1

SAMPLE_SIZE = {
    'prediction' : 200,
    'sensitivity': 600,
    'risk'       : 600,
}

GLOBAL_N_INSTANCES = 30
GLOBAL_SAMPLE_SIZE = 100
N_FEAT_BINS        = 20      # bins for PDP-style global prediction plots

_open_min  = 9 * 60 + 30
BAR_LABELS = [
    '{:02d}:{:02d}'.format(
        (_open_min + i * 5) // 60,
        (_open_min + i * 5) % 60,
    )
    for i in range(T_BARS)
]
t_grid = np.arange(T_BARS, dtype=float)

BASE_PLOT_DIR = os.path.join('plots', 'SPY_cor_games')

DAY_FEATURE_NAMES = [
    'vix_prev',
    'overnight_ret',
    'ann_indicator',
    'day_of_week',
    'trailing_rv',
    'month',
]

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
# 1.  Cache helpers
# ===========================================================================

def _require_dir(path):
    os.makedirs(path, exist_ok=True)

def _cache_path_global(game_type, instance_k):
    """Path for one global instance cache file."""
    return os.path.join(
        GAME_CACHE_DIR,
        'global_{}_inst{:04d}_{}.npz'.format(game_type, instance_k, CACHE_VERSION))

def _cache_path_local(label):
    """Path for local game cache (one per profile label)."""
    safe = label.replace(' ', '_').replace('-', '_')
    return os.path.join(
        GAME_CACHE_DIR,
        'local_prediction_{}_{}.npz'.format(safe, CACHE_VERSION))

def _save_instance_cache(path, x_inst, pure, partial, full, moebius, n_players):
    """Save one instance's effects + full Möbius dict to .npz."""
    # Flatten moebius dict: key "(i,j,...)" -> array
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
    """Load one instance cache; return (x, pure, partial, full, moebius)."""
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
        if key_str == 'empty':
            S = ()
        else:
            S = tuple(int(x) for x in key_str.split('_'))
        moebius[S] = arr
    return x_inst, pure, partial, full, moebius


# ===========================================================================
# 2.  Data loading
# ===========================================================================

def _validate_bars(bars):
    required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    missing  = required - set(bars.columns)
    if missing:
        raise RuntimeError('Bar cache missing columns: {}'.format(missing))
    if len(bars) == 0:
        raise RuntimeError('Bar cache is empty.')
    if bars['close'].isna().mean() > 0.01:
        raise RuntimeError('More than 1 pct of close prices are NaN.')

def _validate_pivot(pivot):
    if len(pivot) < 100:
        raise RuntimeError('Only {} complete trading days.'.format(len(pivot)))

def _validate_feature_matrix(X_day):
    for col in DAY_FEATURE_NAMES:
        if col not in X_day.columns:
            raise RuntimeError('Missing feature: {}'.format(col))
        if X_day[col].nunique() <= 1 or X_day[col].std() < 1e-8:
            raise RuntimeError('Feature {} is constant.'.format(col))

def _resolve_close_column(df, label):
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
    if os.path.isfile(CACHE_PATH):
        print('  Loading bar cache ...')
        bars = pd.read_csv(CACHE_PATH)
        bars['timestamp'] = (
            pd.to_datetime(bars['timestamp'], utc=True)
            .dt.tz_convert('America/New_York'))
        _validate_bars(bars)
        print('  Loaded {:,} rows.'.format(len(bars)))
        return bars
    raise RuntimeError('No bar cache found at: {}'.format(CACHE_PATH))

def _resolve_vix_prev(vix_dict, date_str):
    dt_obj = pd.Timestamp(date_str)
    for d in range(1, 15):
        key = str((dt_obj - pd.Timedelta(d, 'D')).date())
        if key in vix_dict:
            return vix_dict[key]
    if date_str in vix_dict:
        return vix_dict[date_str]
    raise RuntimeError('No VIX within 14 days of {}'.format(date_str))

def load_and_aggregate():
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
        Y_pred = self.predict(X_test)
        ss_res = np.sum((Y_test - Y_pred) ** 2)
        ss_tot = np.sum((Y_test - Y_test.mean()) ** 2)
        return 1.0 - ss_res / ss_tot


# ===========================================================================
# 4.  Cooperative game
# ===========================================================================

class FunctionalGame:
    def __init__(self, predict_fn, X_background, x_explain,
                 game_type='prediction', Y_obs=None,
                 sample_size=200, random_seed=RNG_SEED):
        if game_type == 'risk' and Y_obs is None:
            raise ValueError('Y_obs required for risk game.')
        self.predict_fn   = predict_fn
        self.X_background = X_background
        self.x_explain    = x_explain
        self.game_type    = game_type
        self.Y_obs        = Y_obs
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
        if self.game_type == 'prediction':
            return Y_pred.mean(axis=0)
        elif self.game_type == 'sensitivity':
            return Y_pred.var(axis=0)
        else:
            res = (self.Y_obs[None, :] - Y_pred) ** 2
            return res.mean(axis=0)

    def precompute(self):
        self.values = np.zeros((self.n_coalitions, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i + 1) % 16 == 0 or i + 1 == self.n_coalitions:
                print('    {}/{} coalitions done.'.format(
                    i + 1, self.n_coalitions))

    def __getitem__(self, coalition):
        return self.values[self._idx[coalition]]


# ===========================================================================
# 5.  Mobius + Shapley
# ===========================================================================

def functional_moebius_transform(game):
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
    return np.eye(len(t))

def kernel_ou(t, length_scale=8.0):
    return np.exp(-np.abs(t[:, None] - t[None, :]) / length_scale)

def kernel_causal(t, length_scale=8.0):
    d = t[:, None] - t[None, :]
    return np.where(d >= 0, np.exp(-d / length_scale), 0.0)

def _normalize_kernel(K):
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return K / rs

def apply_kernel(effect, K):
    return _normalize_kernel(K) @ effect * dt

def get_feature_kernel(fi, pnames):
    """Return (ktype_label, K) under the mixed kernel."""
    if pnames[fi] == 'ann_indicator':
        return ('causal', kernel_causal(t_grid, length_scale=8.0))
    return ('ou', kernel_ou(t_grid, length_scale=8.0))


# ===========================================================================
# 7.  Pure / partial / full effect helpers
# ===========================================================================

def _pure_effects(moebius, n_players):
    return {i: moebius.get((i,), np.zeros(T_BARS)).copy()
            for i in range(n_players)}

def _full_effects(moebius, n_players):
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

def _run_or_load_instance(predict_fn, X_background, Y_adj,
                           x_inst, game_type, sample_size,
                           random_seed, cache_path, n_players):
    """
    Run FunctionalGame for one instance and cache, or load from cache.
    Returns (x_inst, pure, partial, full, moebius).
    """
    if os.path.isfile(cache_path):
        return _load_instance_cache(cache_path, n_players)

    y_inst = None
    if game_type == 'risk':
        diffs  = np.abs(X_background - x_inst[None, :]).sum(axis=1)
        y_inst = Y_adj[int(np.argmin(diffs))]

    game = FunctionalGame(
        predict_fn   = predict_fn,
        X_background = X_background,
        x_explain    = x_inst,
        game_type    = game_type,
        Y_obs        = y_inst,
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


def compute_global_effects(predict_fn, X_background, Y_adj,
                            game_type, n_instances, sample_size, seed):
    """
    Average pure/partial/full effects over n_instances randomly drawn
    instances, using per-instance cache files.
    Returns (avg_shapley, avg_pure, avg_full).
    """
    _require_dir(GAME_CACHE_DIR)
    rng  = np.random.default_rng(seed)
    idxs = rng.choice(len(X_background), size=n_instances, replace=False)
    n_players = len(DAY_FEATURE_NAMES)

    sum_shapley = {i: np.zeros(T_BARS) for i in range(n_players)}
    sum_pure    = {i: np.zeros(T_BARS) for i in range(n_players)}
    sum_full    = {i: np.zeros(T_BARS) for i in range(n_players)}

    for k, idx in enumerate(idxs):
        cache = _cache_path_global(game_type, k)
        x_inst = X_background[idx]
        _, pure, shap, full, _ = _run_or_load_instance(
            predict_fn, X_background, Y_adj,
            x_inst, game_type, sample_size,
            seed + k, cache, n_players)
        for i in range(n_players):
            sum_shapley[i] += shap[i]
            sum_pure[i]    += pure[i]
            sum_full[i]    += full[i]
        cached = '(cached)' if os.path.isfile(cache) else ''
        print('    global {} instance {}/{} {}'.format(
            game_type, k + 1, n_instances, cached))

    avg_shapley = {i: sum_shapley[i] / n_instances for i in range(n_players)}
    avg_pure    = {i: sum_pure[i]    / n_instances for i in range(n_players)}
    avg_full    = {i: sum_full[i]    / n_instances for i in range(n_players)}
    return avg_shapley, avg_pure, avg_full


def compute_local_prediction(predict_fn, X_background, x_hv, label):
    """
    Compute local prediction game for the High-VIX profile, with caching.
    Returns (moebius, shapley).
    """
    _require_dir(GAME_CACHE_DIR)
    n_players  = len(DAY_FEATURE_NAMES)
    cache_path = _cache_path_local(label)

    if os.path.isfile(cache_path):
        print('  Loading local prediction cache for "{}" ...'.format(label))
        _, pure, shap, full, mob = _load_instance_cache(cache_path, n_players)
        return mob, shap

    print('  Computing local prediction game for "{}" ...'.format(label))
    game = FunctionalGame(
        predict_fn   = predict_fn,
        X_background = X_background,
        x_explain    = x_hv,
        game_type    = 'prediction',
        sample_size  = SAMPLE_SIZE['prediction'],
        random_seed  = RNG_SEED,
    )
    game.precompute()
    mob  = functional_moebius_transform(game)
    shap = shapley_from_moebius(mob, n_players)
    pure = _pure_effects(mob, n_players)
    full = _full_effects(mob, n_players)
    _save_instance_cache(cache_path, x_hv, pure, shap, full, mob, n_players)
    return mob, shap


def load_per_instance_effects(predict_fn, X_background, Y_adj,
                               glob_idxs, seed):
    """
    Load or compute per-instance prediction effects for the PDP plot.
    Uses the same cache files as compute_global_effects with game_type='prediction'.
    Returns list of dicts: {'x', 'pure', 'partial', 'full'}.
    """
    _require_dir(GAME_CACHE_DIR)
    n_players = len(DAY_FEATURE_NAMES)
    results   = []
    for k, idx in enumerate(glob_idxs):
        cache  = _cache_path_global('prediction', k)
        x_inst = X_background[idx]
        _, pure, shap, full, _ = _run_or_load_instance(
            predict_fn, X_background, Y_adj,
            x_inst, 'prediction', SAMPLE_SIZE['prediction'],
            seed + k, cache, n_players)
        results.append({'x': x_inst, 'pure': pure, 'partial': shap, 'full': full})
        print('    per-instance pred {}/{}'.format(k + 1, len(glob_idxs)))
    return results


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

# ---------------------------------------------------------------------------
# Font-size bundle: pass fs=_fs(3) into plot functions that need larger text.
# ---------------------------------------------------------------------------
from types import SimpleNamespace

def _fs(bump=0):
    return SimpleNamespace(
        suptitle = FS_SUPTITLE + bump,
        title    = FS_TITLE    + bump,
        axis     = FS_AXIS     + bump,
        tick     = FS_TICK     + bump,
        legend   = FS_LEGEND   + bump,
        annot    = FS_ANNOT    + bump,
    )

FEAT_COLORS = {
    'vix_prev'     : '#1f77b4',
    'overnight_ret': '#2ca02c',
    'ann_indicator': '#ff7f0e',
    'day_of_week'  : '#9467bd',
    'trailing_rv'  : '#8c564b',
    'month'        : '#d62728',
}

GAME_YLABEL = {
    'prediction' : 'Effect on vol (%)',
    'sensitivity': r'Var$[F(t)]$ ($\%^2 \times 10^4$)',
    'risk'       : r'Effect on MSE ($\%^2 \times 10^4$)',
}

_EFFECT_TYPES = ['pure', 'partial', 'full']

# Local prediction: no XAI equivalences
_LOCAL_PRED_LABELS = {
    'pure'   : r'Pure $m_i(t)$',
    'partial': r'Partial $\phi_i(t)$',
    'full'   : r'Full $\Phi_i(t)$',
}
# Global prediction: keep "= PDP" etc., plain = sign
_GLOBAL_PRED_LABELS = {
    'pure'   : r'Pure $m_i(x_j)$ = PDP',
    'partial': r'Partial $\phi_i(x_j)$ = global SHAP',
    'full'   : r'Full $\Phi_i(x_j)$',
}
# Sensitivity (global): plain = sign
_SENS_LABELS = {
    'pure'   : r'Pure = closed Sobol',
    'partial': r'Partial = Shapley-sens.',
    'full'   : r'Full = total Sobol',
}
# Risk (global): plain = sign
_RISK_LABELS = {
    'pure'   : r'Pure = pure risk',
    'partial': r'Partial = SAGE',
    'full'   : r'Full = PFI',
}

def _scale(game_type):
    return 100.0 if game_type == 'prediction' else 1e4

def _set_time_axis(ax, sparse=False):
    step = 2 if sparse else 1
    ax.set_xticks(XTICK_IDXS[::step])
    ax.set_xticklabels(XTICK_LABELS[::step],
                       rotation=45, ha='right', fontsize=FS_TICK)
    ax.set_xlim(-0.5, T_BARS - 0.5)

def _period_shade(ax):
    ax.axvspan(0,  6,  alpha=0.08, color='#ffd699', zorder=0)
    ax.axvspan(72, 78, alpha=0.08, color='#ffd699', zorder=0)

def _ann_vline(ax):
    ax.axvline(54, color='#555', lw=0.9, ls='--', alpha=0.6)

def savefig(fig, name):
    path = os.path.join(BASE_PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print('  Saved: {}'.format(path))
    plt.close(fig)

def _align_row_ylims(axes_row):
    """Align y-limits across a list of axes (one row, cols 0-2)."""
    ymin = min(ax.get_ylim()[0] for ax in axes_row)
    ymax = max(ax.get_ylim()[1] for ax in axes_row)
    for ax in axes_row:
        ax.set_ylim(ymin, ymax)

def _draw_bar_panel(ax, effect_dicts, kern_fn, pnames, sc,
                    title='Time-aggregated', fs=None):
    """Horizontal bar panel: time-aggregated pure/partial/full importances."""
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
    offsets = {'pure': -bar_h, 'partial': 0.0, 'full': bar_h}
    alphas  = {'pure': 0.55,   'partial': 0.90, 'full': 0.55}
    hatches = {'pure': '//',   'partial': '',   'full': '\\\\'}
    for etype in _EFFECT_TYPES:
        ax.barh(y_pos + offsets[etype],
                [imps_all[etype][i] for i in order],
                height=bar_h,
                color=[FEAT_COLORS[pnames[i]] for i in order],
                alpha=alphas[etype], hatch=hatches[etype], label=etype)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([pnames[i] for i in order], fontsize=fs.tick)
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=fs.axis)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=fs.tick)
    ax.set_title(title, fontsize=fs.title, fontweight='bold')
    ax.legend(fontsize=fs.legend, loc='upper right')


def _add_fig_legend(fig, pnames, top_features, fs, bbox=(0.01, 0.98)):
    """
    Add a single shared feature legend to the figure, top-left near suptitle.
    All feature lines drawn solid — linestyle in plots encodes kernel choice.
    """
    handles = []
    for fi in top_features:
        handles.append(Line2D([0], [0],
                              color=FEAT_COLORS[pnames[fi]],
                              lw=1.8, ls='-',
                              label=pnames[fi]))
    fig.legend(handles=handles, fontsize=fs.legend,
               loc='upper left',
               bbox_to_anchor=bbox,
               framealpha=0.9, ncol=len(top_features))


def _add_kernel_legend(fig, fs, bbox=(0.01, 0.0)):
    """
    Kernel linestyle legend: solid = OU, dashed = Causal.
    Left-bounded, placed below the feature legend.
    """
    handles = [
        Line2D([0], [0], color='#555', lw=1.8, ls='-',  label='OU kernel'),
        Line2D([0], [0], color='#555', lw=1.8, ls='--', label='Causal kernel'),
    ]
    fig.legend(handles=handles, fontsize=fs.legend,
               loc='lower left',
               bbox_to_anchor=bbox,
               framealpha=0.9, ncol=2)


# ===========================================================================
# 10.  PLOT 1 — Sensitivity + Risk (global, 4 rows x 4 cols)
# ===========================================================================

def plot_sensitivity_risk_global(global_sens, global_risk, pnames,
                                  fs=None):
    """
    4 rows x 4 columns:
      row 0: risk,        identity kernel
      row 1: risk,        mixed kernel
      row 2: sensitivity, identity kernel
      row 3: sensitivity, mixed kernel
    Single shared feature legend + kernel legend below. Y-axes per row only.
    """
    if fs is None:
        fs = _fs(0)
    n_players = len(pnames)
    K_id      = kernel_identity(t_grid)

    def kern_id(fi):  return K_id
    def kern_mix(fi): return get_feature_kernel(fi, pnames)[1]

    row_specs = [
        ('risk',        global_risk, kern_id,  'identity', _RISK_LABELS,
         'Risk — Identity kernel'),
        ('risk',        global_risk, kern_mix, 'mixed',    _RISK_LABELS,
         'Risk — Mixed kernel (OU + causal)'),
        ('sensitivity', global_sens, kern_id,  'identity', _SENS_LABELS,
         'Sensitivity — Identity kernel'),
        ('sensitivity', global_sens, kern_mix, 'mixed',    _SENS_LABELS,
         'Sensitivity — Mixed kernel (OU + causal)'),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(20, 4.2 * 4),
                             gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Global Sensitivity and Risk effects — pure / partial / full\n'
        '(averaged over {} instances) — Random Forest'.format(GLOBAL_N_INSTANCES),
        fontsize=fs.suptitle, fontweight='bold')

    top_k     = 5
    top_feats = None

    for r, (gtype, g_eff, kern_fn, klabel, eff_labels, row_title) in \
            enumerate(row_specs):
        sc = _scale(gtype)
        avg_shap, avg_pure, avg_full = g_eff
        effect_dicts = {'pure': avg_pure, 'partial': avg_shap, 'full': avg_full}

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
                ls = '--' if pnames[fi] == 'ann_indicator' \
                             and klabel == 'mixed' else '-'
                ax.plot(t_grid, apply_kernel(eff[fi], K) * sc,
                        color=FEAT_COLORS[pnames[fi]], lw=1.8, ls=ls)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax); _ann_vline(ax); _set_time_axis(ax, sparse=True)
            ax.tick_params(labelsize=fs.tick)
            ax.set_xlabel('Time', fontsize=fs.axis)
            ax.set_title(eff_labels[etype], fontsize=fs.title, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=fs.axis)
                ax.text(-0.30, 0.5, row_title, transform=ax.transAxes,
                        fontsize=fs.axis - 1, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')

        _draw_bar_panel(axes[r, 3], effect_dicts, kern_fn, pnames, sc,
                        fs=fs)
        _align_row_ylims([axes[r, c] for c in range(3)])

    # Feature legend top-left, kernel legend below it
    _add_fig_legend(fig, pnames, top_feats, fs=fs, bbox=(0.01, 0.995))
    _add_kernel_legend(fig, fs=fs, bbox=(0.01, 0.96))
    plt.tight_layout(rect=[0.04, 0, 1, 0.94])
    fig.subplots_adjust(hspace=0.65, top=0.92)
    return fig



# ===========================================================================
# 11.  PLOT 2 — Local Prediction (2 rows x 4 cols)
# ===========================================================================

def plot_local_prediction(moebius_hv, shapley_hv, pnames, fs=None):
    """
    2 rows x 4 columns:
      row 0: prediction, identity kernel — pure/partial/full + bar
      row 1: prediction, mixed kernel    — same
    Single shared feature legend + kernel legend below. Y-axes per row only.
    Interaction effects moved to plot_interactions (fig4).
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

    top_k        = 5
    row_specs    = [
        (kern_id,  'Identity kernel',            'identity'),
        (kern_mix, 'Mixed kernel (OU + causal)', 'mixed'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 4.2 * 2),
                             gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Local Prediction effects — pure / partial / full\n'
        'High-VIX Announcement profile (2022-07-13) — Random Forest',
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
                             and kkey == 'mixed' else '-'
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
                ax.text(-0.30, 0.5,
                        'Prediction\n{}'.format(klabel),
                        transform=ax.transAxes,
                        fontsize=fs.axis - 1, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')

        _draw_bar_panel(axes[r, 3], effect_dicts, kern_fn, pnames, sc, fs=fs)
        _align_row_ylims([axes[r, c] for c in range(3)])

    _add_fig_legend(fig, pnames, top_feats, fs=fs, bbox=(0.01, 0.995))
    _add_kernel_legend(fig, fs=fs, bbox=(0.01, 0.960))
    plt.tight_layout(rect=[0.04, 0, 1, 0.91])
    fig.subplots_adjust(hspace=0.65, top=0.88)
    return fig


# ===========================================================================
# 11b.  PLOT 4 — Interactions (2 rows x 2 cols)
# ===========================================================================

def plot_interactions(moebius_hv, pnames, fs=None):
    """
    2 rows x 2 columns:
      row 0: identity kernel — interaction line plot | time-aggregated bar
      row 1: mixed kernel    — same (causal pairs dashed, OU pairs solid)
    Pair color legend + kernel legend stacked at bottom-left.
    """
    if fs is None:
        fs = _fs(0)
    n_players  = len(pnames)
    K_id       = kernel_identity(t_grid)
    K_ou       = kernel_ou(t_grid, 8.0)
    moebius    = moebius_hv['prediction']
    sc         = _scale('prediction')
    PAIR_COLORS = ['#e63946', '#2a9d8f', '#8338ec', '#fb8500', '#457b9d']

    pair_imp = {}
    for i in range(n_players):
        for j in range(i + 1, n_players):
            raw = moebius.get((i, j), np.zeros(T_BARS))
            pair_imp[(i, j)] = float(np.sum(np.abs(apply_kernel(raw, K_ou))))
    top5 = sorted(pair_imp, key=pair_imp.get, reverse=True)[:5]

    fig, axes = plt.subplots(2, 2, figsize=(12, 4.2 * 2),
                             gridspec_kw={'width_ratios': [3, 1.8]})
    fig.suptitle(
        r'Local pairwise interaction effects $m_{ij}(t)$ — top-5 pairs'
        '\nHigh-VIX Announcement profile (2022-07-13) — Random Forest',
        fontsize=fs.suptitle, fontweight='bold')

    inter_row_specs = [
        (K_id,  'identity', 'Interactions — Identity kernel'),
        (K_ou,  'mixed',    'Interactions — Mixed kernel (OU + causal)'),
    ]

    for r, (K_base, kkey, row_label) in enumerate(inter_row_specs):
        ax = axes[r, 0]
        fi_ann = pnames.index('ann_indicator')
        for pair_idx, (i, j) in enumerate(top5):
            raw = moebius.get((i, j), np.zeros(T_BARS))
            if kkey == 'mixed':
                is_causal = (i == fi_ann or j == fi_ann)
                K_use = kernel_causal(t_grid, 8.0) if is_causal else K_ou
                ls    = '--' if is_causal else '-'
            else:
                K_use = K_id
                ls    = '-'
            curve = apply_kernel(raw, K_use) * sc
            ax.plot(t_grid, curve, color=PAIR_COLORS[pair_idx], lw=1.8, ls=ls)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax, sparse=True)
        ax.tick_params(labelsize=fs.tick)
        ax.set_xlabel('Time', fontsize=fs.axis)
        ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=fs.axis)
        ax.set_title(r'$m_{ij}(t)$', fontsize=fs.title, fontweight='bold')
        ax.text(-0.18, 0.5, row_label, transform=ax.transAxes,
                fontsize=fs.axis - 1, va='center', ha='right',
                rotation=90, color='#333', fontweight='bold')
        _align_row_ylims([axes[r, 0]])

        ax_bar = axes[r, 1]
        pair_imps, pair_lbls = [], []
        for pair_idx, (i, j) in enumerate(top5):
            raw = moebius.get((i, j), np.zeros(T_BARS))
            if kkey == 'mixed':
                K_use = kernel_causal(t_grid, 8.0)                         if (i == fi_ann or j == fi_ann) else K_ou
            else:
                K_use = K_id
            pair_imps.append(
                float(np.sum(np.abs(apply_kernel(raw, K_use)))) * sc)
            pair_lbls.append('{} x {}'.format(pnames[i], pnames[j]))
        y_pos = np.arange(len(top5))
        ax_bar.barh(y_pos, pair_imps, color=PAIR_COLORS[:len(top5)], alpha=0.85)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(pair_lbls, fontsize=fs.tick - 1)
        ax_bar.set_xlabel(r'$\int|m_{ij}|\,dt$', fontsize=fs.axis)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=fs.tick)
        ax_bar.set_title('Time-aggregated', fontsize=fs.title, fontweight='bold')

    # Legend 1 (top): pair colors — bottom-left
    pair_handles = [
        Line2D([0], [0], color=PAIR_COLORS[k], lw=1.8,
               label='{} x {}'.format(pnames[i], pnames[j]))
        for k, (i, j) in enumerate(top5)]
    leg1 = fig.legend(handles=pair_handles, fontsize=fs.legend,
                      loc='lower left', ncol=len(top5),
                      bbox_to_anchor=(0.04, 0.03), framealpha=0.9)
    fig.add_artist(leg1)

    # Legend 2 (below pairs): kernel linestyle — bottom-left, stacked
    kern_handles = [
        Line2D([0], [0], color='#555', lw=1.8, ls='-',  label='OU kernel'),
        Line2D([0], [0], color='#555', lw=1.8, ls='--', label='Causal kernel'),
    ]
    fig.legend(handles=kern_handles, fontsize=fs.legend,
               loc='lower left', ncol=2,
               bbox_to_anchor=(0.04, -0.03), framealpha=0.9)

    plt.tight_layout(rect=[0.04, 0.12, 1, 0.93])
    fig.subplots_adjust(hspace=0.55, top=0.88)
    return fig

# ===========================================================================
# 12.  PLOT 3 — Global Prediction PDP-style (4 rows x 3 cols, no bar panel)
# ===========================================================================

def _top2_global_features(global_pred, pnames):
    """Top-2 features by integrated |global SHAP| under OU kernel."""
    n_players = len(pnames)
    avg_shap, _, _ = global_pred
    K_ou = kernel_ou(t_grid, 8.0)
    imps = {i: float(np.sum(np.abs(apply_kernel(avg_shap[i], K_ou))))
            for i in range(n_players)}
    return sorted(imps, key=imps.get, reverse=True)[:2]


def _pdp_panel(ax, fi, etype, kern_fn, X_background,
               per_instance_effects, pnames, sc,
               selected_t_idxs, t_cmap, fs, n_bins=N_FEAT_BINS):
    """
    One PDP-style panel:
      x-axis  = observed values of feature fi (binned)
      y-axis  = kernel-smoothed effect
      lines   = dashed, one per selected time point (colored)
      black   = solid time-aggregated mean across all T
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

    pdp = np.nanmean(bin_effects, axis=1) * sc
    ax.plot(bin_centers[valid], pdp[valid],
            color='black', lw=2.5, ls='-', zorder=5, label='time-agg.')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.tick_params(labelsize=fs.tick)
    ax.set_xlabel(pnames[fi], fontsize=fs.axis)


def plot_global_prediction_pdp(per_instance_effects, global_pred,
                                X_background, pnames, fs=None):
    """
    4 rows x 3 columns (no bar panel):
      row 0: feature 1, identity kernel
      row 1: feature 1, OU kernel
      row 2: feature 2, identity kernel
      row 3: feature 2, OU kernel
    Time-point lines: dashed. Time-aggregated PDP: solid black, thick.
    Shared legend bottom-center. Y-axes aligned within each row only.
    """
    if fs is None:
        fs = _fs(0)
    n_players = len(pnames)
    K_id      = kernel_identity(t_grid)
    K_ou      = kernel_ou(t_grid, 8.0)

    def kern_id(fi):  return K_id
    def kern_ou_fn(fi): return K_ou

    sc   = _scale('prediction')
    top2 = _top2_global_features(global_pred, pnames)

    selected_t_idxs = [0, 18, 42, 54, 77]
    t_cmap = cm.get_cmap('plasma', len(selected_t_idxs))

    feat_row_specs = []
    for fi in top2:
        feat_row_specs.append((fi, kern_id,    'Identity kernel'))
        feat_row_specs.append((fi, kern_ou_fn, 'OU kernel'))

    fig, axes = plt.subplots(4, 3, figsize=(18, 3.8 * 4))
    fig.suptitle(
        'Global Prediction effects — PDP-style — pure / partial / full\n'
        '(per-instance effects, binned over feature range) — Random Forest',
        fontsize=fs.suptitle, fontweight='bold')

    for r, (fi, kern_fn, klabel) in enumerate(feat_row_specs):
        for c, etype in enumerate(_EFFECT_TYPES):
            ax = axes[r, c]
            _pdp_panel(ax, fi, etype, kern_fn,
                       X_background, per_instance_effects,
                       pnames, sc, selected_t_idxs, t_cmap, fs=fs)
            ax.set_title(_GLOBAL_PRED_LABELS[etype],
                         fontsize=fs.title, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=fs.axis)
                row_label = '{}\n{}'.format(pnames[fi], klabel)
                ax.text(-0.22, 0.5, row_label,
                        transform=ax.transAxes,
                        fontsize=fs.axis - 1, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')

        _align_row_ylims([axes[r, c] for c in range(3)])

    # Shared legend for time points + time-agg — bottom center
    t_handles = [
        Line2D([0], [0],
               color=t_cmap(i / (len(selected_t_idxs) - 1)),
               lw=1.4, ls='--', label=BAR_LABELS[ti])
        for i, ti in enumerate(selected_t_idxs)
    ] + [Line2D([0], [0], color='black', lw=2.5, ls='-', label='time-agg.')]
    fig.legend(handles=t_handles, fontsize=fs.legend,
               loc='lower center', ncol=len(t_handles),
               bbox_to_anchor=(0.5, 0.0),
               framealpha=0.9)

    plt.tight_layout(rect=[0.04, 0.04, 1, 0.93])
    fig.subplots_adjust(hspace=0.70, top=0.91)
    return fig


# ===========================================================================
# 13.  fig0 — Main body summary (local pred rows 0-1, global risk row 2)
# ===========================================================================

def plot_main_body_summary(moebius_hv, shapley_hv, pnames):
    """
    3 rows x 4 columns:
      row 0: local prediction, identity kernel
      row 1: local prediction, mixed kernel
      row 2: global risk,      mixed kernel
    Focuses on vix_prev + ann_indicator for prediction rows; top-3 for risk.
    Y-axes aligned within each row (cols 0-2) only.
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
        # (gtype, y_label, pure, partial, full, kern_fn, row_label,
        #  lbl_pure, lbl_partial, lbl_full, K_inter, sc)
        ('prediction', GAME_YLABEL['prediction'],
         pure_pred, partial_pred, full_pred, kern_id,
         'Prediction\n(Identity)',
         r'Pure $m_i(t)$', r'Partial $\phi_i(t)$', r'Full $\Phi_i(t)$',
         K_id, _scale('prediction')),
        ('prediction', GAME_YLABEL['prediction'],
         pure_pred, partial_pred, full_pred, kern_mix,
         'Prediction\n(Mixed)',
         r'Pure $m_i(t)$', r'Partial $\phi_i(t)$', r'Full $\Phi_i(t)$',
         K_causal, _scale('prediction')),
    ]

    fig, axes = plt.subplots(
        2, 4, figsize=(19, 2.9 * 2),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'H-FD Framework: Kernel Choice and Effect Decomposition\n'
        'High-VIX Announcement Profile — Local Prediction',
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

        # col 0: pure — legend upper-left for row 0, lower-left for row 1
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

        # col 1: partial
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
                bbox=dict(boxstyle='round,pad=0.25',
                          fc='white', ec='#ddd', alpha=0.9))
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax, sparse=True)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_title(lbl_partial, fontsize=FS_TITLE - 1, fontweight='bold')

        # col 2: interaction — no backslash before underscores
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
                bbox=dict(boxstyle='round,pad=0.25',
                          fc='white', ec='#aaa', alpha=0.85))
        ax.set_title(
            r'Interaction $m_{ij}$ — vix_prev x ann_indicator',
            fontsize=FS_TITLE - 1, fontweight='bold')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax, sparse=True)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)

        # col 3: bar
        effect_dicts_bar = {
            'pure': pure_eff, 'partial': partial_eff, 'full': full_eff}
        _draw_bar_panel(axes[r, 3], effect_dicts_bar, kern_fn, pnames, sc)

        _align_row_ylims([axes[r, c] for c in range(3)])

    plt.tight_layout(rect=[0.04, 0, 1, 0.91])
    fig.subplots_adjust(top=0.86, hspace=0.70)
    return fig


# ===========================================================================
# 14.  Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  SPY Intraday Vol  —  RF  (v6)')
    print('=' * 60)

    _require_dir(BASE_PLOT_DIR)
    _require_dir(DATA_DIR)
    _require_dir(GAME_CACHE_DIR)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print('\n[1] Loading data ...')
    X_day, Y_day, Y_adj, diurnal_mean = load_and_aggregate()
    X_day_np = X_day.to_numpy().astype(float)

    # ── 2. Model ──────────────────────────────────────────────────────────
    print('\n[2] Fitting Random Forest ...')
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_day_np, Y_adj, test_size=0.2, random_state=RNG_SEED)
    model = RFModel()
    model.fit(X_tr, Y_tr)
    r2 = model.evaluate(X_te, Y_te)
    print('  Test R2 (trajectory-level): {:.4f}'.format(r2))
    pnames = list(DAY_FEATURE_NAMES)

    # ── 3. Global games (prediction, sensitivity, risk) ───────────────────
    print('\n[3] Global games ({} instances each) ...'.format(
        GLOBAL_N_INSTANCES))
    global_effects = {}
    for gtype in ('prediction', 'sensitivity', 'risk'):
        print('\n  global {} ...'.format(gtype))
        avg_shap, avg_pure, avg_full = compute_global_effects(
            predict_fn   = model.predict,
            X_background = X_day_np,
            Y_adj        = Y_adj,
            game_type    = gtype,
            n_instances  = GLOBAL_N_INSTANCES,
            sample_size  = GLOBAL_SAMPLE_SIZE,
            seed         = RNG_SEED,
        )
        global_effects[gtype] = (avg_shap, avg_pure, avg_full)

    # ── 4. High-VIX Announcement profile ──────────────────────────────────
    print('\n[4] Selecting High-VIX Announcement profile ...')
    vix_col = DAY_FEATURE_NAMES.index('vix_prev')
    vix_p75 = float(np.percentile(X_day_np[:, vix_col], 75))

    def find_profile(conditions):
        mask = np.ones(len(X_day_np), dtype=bool)
        for feat, (lo, hi) in conditions.items():
            ci = DAY_FEATURE_NAMES.index(feat)
            mask &= (X_day_np[:, ci] >= lo) & (X_day_np[:, ci] <= hi)
        hits = X_day_np[mask]
        if len(hits) == 0:
            raise RuntimeError('No day matches: {}'.format(conditions))
        print('  {} matching days.'.format(len(hits)))
        return hits[len(hits) // 2]

    x_hv = find_profile(
        {'ann_indicator': (0.9, 1.1), 'vix_prev': (vix_p75, 999)})

    # ── 5. Local prediction game ───────────────────────────────────────────
    print('\n[5] Local prediction game (High-VIX profile) ...')
    mob_hv, shap_hv = compute_local_prediction(
        model.predict, X_day_np, x_hv, 'High_VIX_Announcement')
    moebius_hv = {'prediction': mob_hv}
    shapley_hv = {'prediction': shap_hv}

    # ── 6. Per-instance prediction effects for PDP plot ───────────────────
    print('\n[6] Per-instance prediction effects for PDP plot ...')
    rng_glob  = np.random.default_rng(RNG_SEED)
    glob_idxs = rng_glob.choice(
        len(X_day_np), size=GLOBAL_N_INSTANCES, replace=False)
    X_glob_subset       = X_day_np[glob_idxs]
    per_instance_effects = load_per_instance_effects(
        model.predict, X_day_np, Y_adj, glob_idxs, RNG_SEED)

    # ── 7. Generate figures ───────────────────────────────────────────────
    print('\n[7] Generating figures ...')

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
            global_pred=global_effects['prediction'],
            X_background=X_glob_subset,
            pnames=pnames,
            fs=_fs(3)),
        'fig3_global_prediction_pdp.pdf')

    savefig(
        plot_interactions(moebius_hv, pnames, fs=_fs(3)),
        'fig4_interactions.pdf')

    print('\n' + '=' * 60)
    print('  Done.  Figures in {}/'.format(BASE_PLOT_DIR))
    print('  Game cache in {}/'.format(GAME_CACHE_DIR))
    print('=' * 60)