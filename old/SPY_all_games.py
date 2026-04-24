"""
Functional Explanation Framework -- Intraday SPY Volatility (Random Forest)
============================================================================
Changes vs previous version:
  1. Dropped fig1 (operator sweep), fig3b (sensitivity gap),
     figA_network_prediction_risk.
  2. fig2, fig2b, figA_network titles now state "local analogue".
  3. fig2b legend in col-0 row-0 uses smaller font.
  4. Added fig2_global and fig2b_global: global analogues of fig2/fig2b
     obtained by averaging local Shapley/Möbius curves over a random
     sample of background days (GLOBAL_N_INSTANCES days).
  5. Global computation reuses a shared precomputed coalition-value cache
     keyed by (game_type, instance_index) to avoid redundant model calls.
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0.  Settings
# ---------------------------------------------------------------------------
TICKER         = 'SPY'
_HERE          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(_HERE, 'data')
CACHE_PATH     = os.path.join(DATA_DIR, 'spy_5min_cache.csv')
VIX_CACHE_PATH = os.path.join(DATA_DIR, 'vix_daily_cache.csv')

POLYGON_API_KEY   = ''
FETCH_START       = '2022-01-01'
FETCH_END         = '2024-04-01'
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

# Number of instances to average over for global explanations.
# Keep small (≤30) for tractability; each instance requires 64 coalition evals.
GLOBAL_N_INSTANCES  = 30
GLOBAL_SAMPLE_SIZE  = 100   # background draws per coalition for global sweep

_open_min  = 9 * 60 + 30
BAR_LABELS = [
    '{:02d}:{:02d}'.format(
        (_open_min + i * 5) // 60,
        (_open_min + i * 5) % 60,
    )
    for i in range(T_BARS)
]
t_grid = np.arange(T_BARS, dtype=float)

BASE_PLOT_DIR = os.path.join('plots', 'SPY_all_games')

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
# 1.  Data loading
# ===========================================================================

def _require_dir(path):
    os.makedirs(path, exist_ok=True)

def _validate_bars(bars):
    required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    missing  = required - set(bars.columns)
    if missing:
        raise RuntimeError('Bar cache missing columns: {}'.format(missing))
    if len(bars) == 0:
        raise RuntimeError('Bar cache is empty.')
    if bars['close'].isna().mean() > 0.01:
        raise RuntimeError('More than 1% of close prices are NaN.')

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
# 2.  Model
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
# 3.  Cooperative game
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

    @property
    def empty_value(self):
        return self[tuple([False] * self.n_players)]

    @property
    def grand_value(self):
        return self[tuple([True] * self.n_players)]


# ===========================================================================
# 4.  Mobius + Shapley
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
# 5.  Kernels
# ===========================================================================

def kernel_identity(t):
    return np.eye(len(t))

def kernel_ou(t, length_scale=8.0):
    return np.exp(-np.abs(t[:, None] - t[None, :]) / length_scale)

def kernel_causal(t, length_scale=8.0):
    d = t[:, None] - t[None, :]
    return np.where(d >= 0, np.exp(-d / length_scale), 0.0)

def kernel_output_correlation(Y_day):
    C   = np.cov(Y_day.T)
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    K   = np.clip(C / np.outer(std, std), -1.0, 1.0)
    print('  Correlation kernel: off-diag mean={:.3f}'.format(
        (K - np.eye(T_BARS)).mean()))
    return K

def _normalize_kernel(K):
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return K / rs

def apply_kernel(effect, K):
    return _normalize_kernel(K) @ effect * dt


# ===========================================================================
# 5b.  Per-profile, per-feature kernel assignment
# ===========================================================================

KERNEL_TYPE_LABEL = {
    'identity': 'Identity',
    'ou'      : 'OU ($\\ell=8$)',
    'causal'  : 'Causal ($\\ell=8$)',
}

def get_profile_kernels(profile_label, pnames):
    n = len(pnames)
    K_id  = kernel_identity(t_grid)
    K_ou  = kernel_ou(t_grid, 8.0)
    K_cau = kernel_causal(t_grid, 8.0)

    if profile_label == 'High-VIX Announcement':
        assignment = {
            'vix_prev'     : ('ou',      K_ou),
            'overnight_ret': ('ou',      K_ou),
            'ann_indicator': ('causal',  K_cau),
            'day_of_week'  : ('ou',      K_ou),
            'trailing_rv'  : ('ou',      K_ou),
            'month'        : ('ou',      K_ou),
        }
    elif profile_label == 'Quiet Low-VIX':
        assignment = {
            'vix_prev'     : ('ou',      K_ou),
            'overnight_ret': ('ou',      K_ou),
            'ann_indicator': ('identity',K_id),
            'day_of_week'  : ('identity',K_id),
            'trailing_rv'  : ('ou',      K_ou),
            'month'        : ('identity',K_id),
        }
    elif profile_label == 'Monday Gap':
        assignment = {
            'vix_prev'     : ('ou',      K_ou),
            'overnight_ret': ('causal',  K_cau),
            'ann_indicator': ('identity',K_id),
            'day_of_week'  : ('identity',K_id),
            'trailing_rv'  : ('ou',      K_ou),
            'month'        : ('identity',K_id),
        }
    else:
        assignment = {name: ('ou', K_ou) for name in pnames}

    return {pnames.index(name): (ktype, K)
            for name, (ktype, K) in assignment.items()}


# ===========================================================================
# 6.  Pure / partial / full effect helpers
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
# 7.  Plotting helpers
# ===========================================================================

XTICK_IDXS   = list(range(0, T_BARS, 6))
XTICK_LABELS = [BAR_LABELS[i] for i in XTICK_IDXS]

FS_SUPTITLE = 15
FS_TITLE    = 13
FS_AXIS     = 12
FS_TICK     = 11
FS_LEGEND   = 10.5
FS_ANNOT    = 10.5

FEAT_COLORS = {
    'vix_prev'     : '#1f77b4',
    'overnight_ret': '#2ca02c',
    'ann_indicator': '#ff7f0e',
    'day_of_week'  : '#9467bd',
    'trailing_rv'  : '#8c564b',
    'month'        : '#d62728',
}

GAME_LABELS = {
    'prediction' : 'Prediction  $v(S)(t) = \\mathbb{E}[F(x)(t)\\mid X_S]$',
    'sensitivity': 'Sensitivity  $v(S)(t) = \\mathrm{Var}[F(x)(t)\\mid X_S]$',
    'risk'       : 'Risk (MSE)  $v(S)(t) = \\mathbb{E}[(Y(t)-F(x)(t))^2\\mid X_S]$',
}

GAME_YLABEL = {
    'prediction' : 'Effect on vol (%)',
    'sensitivity': r'Var$[F(t)]$ ($\%^2 \times 10^4$)',
    'risk'       : r'Effect on MSE ($\%^2 \times 10^4$)',
}

# ── Local XAI labels (corrected: no global method names) ─────────────────
_XAI_LABELS_LOCAL = {
    ('prediction',  'pure')   : 'Pure  $m_i(t)$  $\\equiv$  local PDP',
    ('prediction',  'partial'): 'Partial  $\\phi_i(t)$  $\\equiv$  SHAP',
    ('prediction',  'full')   : 'Full  $\\Phi_i(t)$  $\\equiv$  local ICE-agg.',
    ('sensitivity', 'pure')   : 'Pure  $\\equiv$  local closed Sobol',
    ('sensitivity', 'partial'): 'Partial  $\\equiv$  local Shapley-sens.',
    ('sensitivity', 'full')   : 'Full  $\\equiv$  local total Sobol',
    ('risk',        'pure')   : 'Pure  $\\equiv$  local pure risk',
    ('risk',        'partial'): 'Partial  $\\equiv$  local SAGE-style',
    ('risk',        'full')   : 'Full  $\\equiv$  local PFI-style',
}

# ── Global XAI labels ─────────────────────────────────────────────────────
_XAI_LABELS_GLOBAL = {
    ('prediction',  'pure')   : 'Pure  $m_i(t)$  $\\equiv$  PDP',
    ('prediction',  'partial'): 'Partial  $\\phi_i(t)$  $\\equiv$  global SHAP',
    ('prediction',  'full')   : 'Full  $\\Phi_i(t)$  $\\equiv$  ICE-agg.',
    ('sensitivity', 'pure')   : 'Pure  $\\equiv$  closed Sobol',
    ('sensitivity', 'partial'): 'Partial  $\\equiv$  Shapley-sens.',
    ('sensitivity', 'full')   : 'Full  $\\equiv$  total Sobol',
    ('risk',        'pure')   : 'Pure  $\\equiv$  pure risk',
    ('risk',        'partial'): 'Partial  $\\equiv$  SAGE',
    ('risk',        'full')   : 'Full  $\\equiv$  PFI',
}

_EFFECT_TYPES = ['pure', 'partial', 'full']

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

def _top_features(moebius_dict, n_players, top_k):
    imps = np.zeros(n_players)
    for mob in moebius_dict.values():
        for i in range(n_players):
            imps[i] += float(np.sum(np.abs(mob[(i,)])))
    return sorted(range(n_players),
                  key=lambda i: imps[i], reverse=True)[:top_k]


# ===========================================================================
# 8.  Figure 2 -- Main effects, pure/partial/full, identity kernel, all games
#     (LOCAL analogue)
# ===========================================================================

def plot_main_effects_all_games(moebius_dict, shapley_dict, pnames, top_k=5):
    game_types = ['prediction', 'sensitivity', 'risk']
    K_id       = kernel_identity(t_grid)
    n_players  = len(pnames)
    _leg_loc   = {0: 'upper center', 1: 'upper left', 2: 'lower center'}

    fig, axes = plt.subplots(
        len(game_types), 4,
        figsize=(20, 4.5 * len(game_types)),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Local main effects — Identity kernel — pure / partial / full\n'
        'High-VIX Announcement profile  —  Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold')

    for r, gtype in enumerate(game_types):
        sc      = _scale(gtype)
        moebius = moebius_dict[gtype]
        pure_eff    = _pure_effects(moebius, n_players)
        partial_eff = shapley_dict[gtype]
        full_eff    = _full_effects(moebius, n_players)
        effect_dicts = {'pure': pure_eff, 'partial': partial_eff, 'full': full_eff}

        imps_partial = {i: float(np.sum(np.abs(partial_eff[i])))
                        for i in range(n_players)}
        top = sorted(imps_partial, key=imps_partial.get, reverse=True)[:top_k]

        for c, etype in enumerate(_EFFECT_TYPES):
            ax  = axes[r, c]
            eff = effect_dicts[etype]
            for fi in top:
                ax.plot(t_grid, apply_kernel(eff[fi], K_id) * sc,
                        color=FEAT_COLORS[pnames[fi]], lw=2.0,
                        label=pnames[fi] if c == 0 else '_')
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.set_title(_XAI_LABELS_LOCAL[(gtype, etype)],
                         fontsize=FS_TITLE, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=FS_AXIS)
                ax.legend(fontsize=FS_LEGEND, loc=_leg_loc[r], framealpha=0.85)

        ax_bar = axes[r, 3]
        imps_all = {
            etype: {i: float(np.sum(np.abs(effect_dicts[etype][i]))) * sc
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
            ax_bar.barh(y_pos + offsets[etype],
                        [imps_all[etype][i] for i in order],
                        height=bar_h,
                        color=[FEAT_COLORS[pnames[i]] for i in order],
                        alpha=alphas[etype], hatch=hatches[etype], label=etype)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([pnames[i] for i in order], fontsize=FS_TICK)
        ax_bar.axvline(0, color='gray', lw=0.8, ls=':')
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=FS_TICK)
        ax_bar.set_title('Integrated\nimportance', fontsize=FS_TITLE,
                         fontweight='bold')
        ax_bar.legend(fontsize=FS_LEGEND, loc='upper right')

    plt.tight_layout()
    return fig


# ===========================================================================
# 9.  Figure 2b -- Main effects, mixed kernel, all games  (LOCAL)
# ===========================================================================

def _feature_kernel_hv(fi, pnames):
    if pnames[fi] == 'ann_indicator':
        return kernel_causal(t_grid, length_scale=8.0)
    return kernel_ou(t_grid, length_scale=8.0)


def plot_mixed_kernel_effects(moebius_dict, shapley_dict, pnames, top_k=5):
    game_types = ['prediction', 'sensitivity', 'risk']
    n_players  = len(pnames)
    feat_kernels = {fi: _feature_kernel_hv(fi, pnames)
                    for fi in range(n_players)}
    _leg_loc = {0: 'upper center', 1: 'upper left', 2: 'lower center'}

    fig, axes = plt.subplots(
        len(game_types), 4,
        figsize=(20, 4.5 * len(game_types)),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Local main effects — mixed kernel (OU + causal) — pure / partial / full\n'
        'ann_indicator: causal ($\\ell=8$)  |  all others: OU ($\\ell=8$)  '
        '—  High-VIX Announcement profile  —  Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold')

    for r, gtype in enumerate(game_types):
        sc      = _scale(gtype)
        moebius = moebius_dict[gtype]
        pure_eff    = _pure_effects(moebius, n_players)
        partial_eff = shapley_dict[gtype]
        full_eff    = _full_effects(moebius, n_players)
        effect_dicts = {'pure': pure_eff, 'partial': partial_eff, 'full': full_eff}

        imps_partial = {
            i: float(np.sum(np.abs(apply_kernel(partial_eff[i], feat_kernels[i]))))
            for i in range(n_players)}
        top = sorted(imps_partial, key=imps_partial.get, reverse=True)[:top_k]

        for c, etype in enumerate(_EFFECT_TYPES):
            ax  = axes[r, c]
            eff = effect_dicts[etype]
            # ── smaller legend only in col-0, row-0 ─────────────────────
            leg_fs = FS_LEGEND - 2.5 if (r == 0 and c == 0) else FS_LEGEND
            for fi in top:
                ls = '--' if pnames[fi] == 'ann_indicator' else '-'
                ax.plot(t_grid,
                        apply_kernel(eff[fi], feat_kernels[fi]) * sc,
                        color=FEAT_COLORS[pnames[fi]], lw=2.0, ls=ls,
                        label=pnames[fi] if c == 0 else '_')
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.set_title(_XAI_LABELS_LOCAL[(gtype, etype)],
                         fontsize=FS_TITLE, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=FS_AXIS)
                ax.legend(fontsize=leg_fs, loc=_leg_loc[r], framealpha=0.85)

        ax_bar = axes[r, 3]
        imps_all = {
            etype: {
                i: float(np.sum(np.abs(
                    apply_kernel(effect_dicts[etype][i], feat_kernels[i])))) * sc
                for i in range(n_players)}
            for etype in _EFFECT_TYPES}
        order   = sorted(range(n_players),
                         key=lambda i: imps_all['partial'][i], reverse=True)
        y_pos   = np.arange(len(order))
        bar_h   = 0.25
        offsets = {'pure': -bar_h, 'partial': 0.0, 'full': bar_h}
        alphas  = {'pure': 0.55,   'partial': 0.90, 'full': 0.55}
        hatches = {'pure': '//',   'partial': '',   'full': '\\\\'}
        for etype in _EFFECT_TYPES:
            ax_bar.barh(y_pos + offsets[etype],
                        [imps_all[etype][i] for i in order],
                        height=bar_h,
                        color=[FEAT_COLORS[pnames[i]] for i in order],
                        alpha=alphas[etype], hatch=hatches[etype], label=etype)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([pnames[i] for i in order], fontsize=FS_TICK)
        ax_bar.axvline(0, color='gray', lw=0.8, ls=':')
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=FS_TICK)
        ax_bar.set_title('Integrated\nimportance\n(mixed kernel)',
                         fontsize=FS_TITLE, fontweight='bold')
        ax_bar.legend(fontsize=FS_LEGEND, loc='upper right')

    plt.tight_layout()
    return fig


# ===========================================================================
# 10. Figure 4 -- Pairwise interactions, prediction + risk, OU kernel
# ===========================================================================

def plot_interactions(moebius_dict, pnames):
    K_ou = kernel_ou(t_grid, 8.0)
    n    = len(pnames)
    game_rows = [
        ('prediction', 'Prediction  $\\equiv$  local Shapley interaction (2-SII)'),
        ('risk',       'Risk  $\\equiv$  local SAGE-style interaction'),
    ]
    moebius_pred = moebius_dict['prediction']
    pair_imp = {
        (i, j): float(np.sum(np.abs(
            moebius_pred.get((i, j), np.zeros(T_BARS)))))
        for i in range(n) for j in range(i + 1, n)
    }
    top3        = sorted(pair_imp, key=pair_imp.get, reverse=True)[:3]
    pair_colors = ['#e63946', '#2a9d8f', '#8338ec']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.0), sharey=False)
    fig.suptitle(
        'Local pairwise interaction effects  $m_{ij}(t)$  —  OU kernel ($\\ell=8$ bars)\n'
        'Top-3 pairs ranked by prediction game  —  High-VIX Announcement profile',
        fontsize=FS_SUPTITLE, fontweight='bold')

    for r, (gtype, row_label) in enumerate(game_rows):
        sc      = _scale(gtype)
        moebius = moebius_dict[gtype]
        for c, ((i, j), col) in enumerate(zip(top3, pair_colors)):
            ax    = axes[r, c]
            raw   = moebius.get((i, j), np.zeros(T_BARS))
            curve = apply_kernel(raw, K_ou) * sc
            ax.plot(t_grid, curve, color=col, lw=2.2)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            if r == 0:
                ax.set_title('{} $\\times$ {}'.format(pnames[i], pnames[j]),
                             fontsize=FS_TITLE, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=FS_AXIS)
                ax.text(-0.30, 0.5, row_label, transform=ax.transAxes,
                        fontsize=FS_AXIS - 1, va='center', ha='right',
                        rotation=90, color='#333')
            integ = float(np.trapz(raw, t_grid)) * sc
            loc   = (0.03, 0.97) if (r == 0 and c in (0, 1)) or (r == 1 and c == 0) else (0.97, 0.97)
            ha    = 'left' if loc[0] < 0.5 else 'right'
            ax.text(loc[0], loc[1],
                    r'$\int m_{{ij}}\,dt$ = {:.3f}'.format(integ),
                    transform=ax.transAxes, fontsize=FS_ANNOT,
                    va='top', ha=ha,
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', ec='#aaa', alpha=0.8))
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    return fig

# ===========================================================================
# 11. Figure 5 -- Profile comparison
# ===========================================================================

def plot_profiles_comparison(profile_results, pnames):
    n_prof    = len(profile_results)
    sc        = _scale('prediction')
    n_players = len(pnames)

    all_mob = {k: v[0] for k, v in profile_results.items()}
    top4    = _top_features(all_mob, n_players, top_k=4)

    profile_titles = {
        'High-VIX Announcement':
            'High-VIX Announcement\n(CPI + Beige Book, VIX$\\approx$27)',
        'Quiet Low-VIX':
            'Quiet Low-VIX\n(non-announcement Mon, VIX$\\approx$12)',
        'Monday Gap':
            'Monday Gap\n(overnight gap +0.87%, VIX$\\approx$22)',
    }

    fig, axes = plt.subplots(1, n_prof, figsize=(6.0 * n_prof, 5.0),
                             sharey=False)
    fig.suptitle(
        'Local Shapley curves — per-feature kernels — prediction game\n'
        'Three market-regime profiles  —  Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold')

    for p_idx, (ax, (label, (moebius, shapley))) in enumerate(
            zip(axes, profile_results.items())):

        pkernels = get_profile_kernels(label, pnames)

        for fi in top4:
            ktype, K = pkernels[fi]
            curve = apply_kernel(shapley[fi], K) * sc
            ls = '--' if ktype == 'causal' else (':' if ktype == 'identity' else '-')
            ax.plot(t_grid, curve,
                    color=FEAT_COLORS[pnames[fi]], lw=2.0, ls=ls,
                    label=pnames[fi])

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
        ax.set_title(profile_titles.get(label, label),
                     fontsize=FS_TITLE, fontweight='bold')
        ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=FS_AXIS)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)

        if p_idx == 1:
            ax.legend(fontsize=FS_LEGEND, loc='lower right')

    ls_handles = [
        Line2D([0], [0], color='#555', lw=2.0, ls='-',  label='OU kernel'),
        Line2D([0], [0], color='#555', lw=2.0, ls='--', label='Causal kernel'),
        Line2D([0], [0], color='#555', lw=2.0, ls=':',  label='Identity kernel'),
    ]
    axes[2].legend(
        handles=ls_handles,
        loc='lower center',
        ncol=1,
        fontsize=FS_LEGEND,
        framealpha=0.9,
        title='Kernel (linestyle)',
        title_fontsize=FS_LEGEND,
    )

    plt.tight_layout()
    return fig


# ===========================================================================
# 12. Figure 0 -- Main body summary (LOCAL)
# ===========================================================================

def plot_main_body_summary_v2(moebius_dict, shapley_dict, pnames):
    n_players = len(pnames)
    fi_vix    = pnames.index('vix_prev')
    fi_ann    = pnames.index('ann_indicator')

    K_id     = kernel_identity(t_grid)
    K_ou     = kernel_ou(t_grid,     length_scale=8.0)
    K_causal = kernel_causal(t_grid, length_scale=8.0)

    def K_mixed(fi):
        return K_causal if pnames[fi] == 'ann_indicator' else K_ou

    pure_pred    = _pure_effects(moebius_dict['prediction'], n_players)
    partial_pred = shapley_dict['prediction']
    pure_risk    = _pure_effects(moebius_dict['risk'],       n_players)
    partial_risk = shapley_dict['risk']

    def _int_curve(game_type, kern):
        raw = moebius_dict[game_type].get((fi_vix, fi_ann), np.zeros(T_BARS))
        return apply_kernel(raw, kern)

    row_specs = [
        ('prediction', GAME_YLABEL['prediction'],
         'Pure  $m_i$  $\\equiv$  local PDP',
         'Partial  $\\phi_i$  $\\equiv$  SHAP',
         pure_pred, partial_pred, lambda fi: K_id),
        ('prediction', GAME_YLABEL['prediction'],
         'Pure  $m_i$  $\\equiv$  local PDP  (mixed kernel)',
         'Partial  $\\phi_i$  $\\equiv$  SHAP  (mixed kernel)',
         pure_pred, partial_pred, K_mixed),
        ('risk', GAME_YLABEL['risk'],
         'Pure  $m_i$  $\\equiv$  local pure risk  (mixed kernel)',
         'Partial  $\\phi_i$  $\\equiv$  local SAGE-style  (mixed kernel)',
         pure_risk, partial_risk, K_mixed),
    ]
    row_labels = ['Prediction\n(Identity)', 'Prediction\n(Mixed)', 'Risk\n(Mixed)']

    c_vix = FEAT_COLORS['vix_prev']
    c_ann = FEAT_COLORS['ann_indicator']

    # Slightly reduced individual plot title font size
    title_fs = FS_TITLE - 1

    # Per-row annotation settings for col 1:
    #   r=0 → upper left   (y=0.97, va='top')
    #   r=1 → center left  (y=0.50, va='center')
    #   r=2 → upper left   (y=0.97, va='top')
    annot_col1 = {
        0: (0.97, 'top'),
        1: (0.50, 'center'),
        2: (0.97, 'top'),
    }

    n_rows = 3
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(19, 2.8 * n_rows),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]},
    )
    fig.suptitle(
        'High-VIX Announcement Profile (local): '
        'Kernel Choice, Game Type and Effect Decomposition',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    for r, (gtype, y_label, lbl_pure, lbl_partial,
            pure_eff, partial_eff, kern_fn) in enumerate(row_specs):
        sc = _scale(gtype)

        # ── col 0: pure effects ───────────────────────────────────────────
        ax = axes[r, 0]
        ax.plot(t_grid, apply_kernel(pure_eff[fi_vix], kern_fn(fi_vix)) * sc,
                color=c_vix, lw=2.2, ls='-',  label='vix_prev')
        ax.plot(t_grid, apply_kernel(pure_eff[fi_ann], kern_fn(fi_ann)) * sc,
                color=c_ann, lw=2.2, ls='--', label='ann_indicator')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_ylabel(y_label, fontsize=FS_AXIS)
        ax.set_title(lbl_pure, fontsize=title_fs, fontweight='bold')
        ax.text(-0.32, 0.5, row_labels[r], transform=ax.transAxes,
                fontsize=FS_AXIS, va='center', ha='right',
                rotation=90, color='#333', fontweight='bold')
        # Legend only in row 0, col 0 → upper left
        if r == 0:
            ax.legend(fontsize=FS_LEGEND, loc='upper left', framealpha=0.9)

        # ── col 1: partial effects ────────────────────────────────────────
        ax = axes[r, 1]
        ax.plot(t_grid, apply_kernel(partial_eff[fi_vix], kern_fn(fi_vix)) * sc,
                color=c_vix, lw=2.2, ls='-')
        ax.plot(t_grid, apply_kernel(partial_eff[fi_ann], kern_fn(fi_ann)) * sc,
                color=c_ann, lw=2.2, ls='--')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_title(lbl_partial, fontsize=title_fs, fontweight='bold')
        pure_int = float(np.sum(np.abs(
            apply_kernel(pure_eff[fi_ann], kern_fn(fi_ann))))) * sc
        part_int = float(np.sum(np.abs(
            apply_kernel(partial_eff[fi_ann], kern_fn(fi_ann))))) * sc
        ratio = part_int / pure_int if pure_int > 1e-12 else 1.0
        annot_y, annot_va = annot_col1[r]
        ax.text(0.03, annot_y,
                'ann_indicator: partial/pure\n= {:.2f}$\\times$'.format(ratio),
                transform=ax.transAxes, fontsize=FS_ANNOT - 1,
                va=annot_va, ha='left',
                color=c_ann,
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#ddd', alpha=0.9))

        # ── col 2: interaction ────────────────────────────────────────────
        ax  = axes[r, 2]
        kern_for_pair = K_id if r == 0 else K_causal
        int_mx = _int_curve(gtype, kern_for_pair) * sc
        pos = np.where(int_mx >= 0, int_mx, 0.0)
        neg = np.where(int_mx <  0, int_mx, 0.0)
        ax.fill_between(t_grid, 0, pos, color='#2a9d8f', alpha=0.30)
        ax.fill_between(t_grid, 0, neg, color='#e63946', alpha=0.30)
        ax.plot(t_grid, int_mx, color='#333', lw=1.8)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        # No backslash before underscore
        ax.set_title(
            r'Interaction  $m_{ij}(t)$  — vix_prev $\times$ ann_indicator',
            fontsize=title_fs, fontweight='bold')
        integ = float(np.trapz(
            moebius_dict[gtype].get((fi_vix, fi_ann), np.zeros(T_BARS)),
            t_grid)) * sc
        ax.text(0.03, 0.97,
                r'$\int m_{{ij}}\,dt$ = {:.3f}'.format(integ),
                transform=ax.transAxes, fontsize=FS_ANNOT,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25',
                          fc='white', ec='#aaa', alpha=0.85))

        # ── col 3: bar chart ──────────────────────────────────────────────
        ax_bar = axes[r, 3]
        imps_part = {i: float(np.sum(np.abs(
            apply_kernel(partial_eff[i], kern_fn(i))))) * sc
            for i in range(n_players)}
        imps_pure = {i: float(np.sum(np.abs(
            apply_kernel(pure_eff[i], kern_fn(i))))) * sc
            for i in range(n_players)}
        order = sorted(range(n_players),
                       key=lambda i: imps_part[i], reverse=True)
        y_pos = np.arange(len(order)); bar_h = 0.35
        ax_bar.barh(y_pos - bar_h / 2,
                    [imps_pure[i] for i in order], height=bar_h,
                    color=[FEAT_COLORS[pnames[i]] for i in order],
                    alpha=0.45, hatch='//', label='pure')
        ax_bar.barh(y_pos + bar_h / 2,
                    [imps_part[i] for i in order], height=bar_h,
                    color=[FEAT_COLORS[pnames[i]] for i in order],
                    alpha=0.90, label='partial')
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([pnames[i] for i in order], fontsize=FS_TICK)
        ax_bar.axvline(0, color='gray', lw=0.8, ls=':')
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=FS_TICK)
        ax_bar.set_title('Integrated\nimportance', fontsize=title_fs,
                         fontweight='bold')
        ax_bar.legend(fontsize=FS_LEGEND, loc='upper right')

    # ── shared y-limits across time-series columns ────────────────────────
    for r in range(n_rows):
        ymin = min(axes[r, c].get_ylim()[0] for c in range(3))
        ymax = max(axes[r, c].get_ylim()[1] for c in range(3))
        for c in range(3):
            axes[r, c].set_ylim(ymin, ymax)

    plt.tight_layout(rect=[0.04, 0, 1, 0.95])
    fig.subplots_adjust(top=0.92, hspace=0.70)   # row whitespace
    return fig


# ===========================================================================
# 13. Network -- all games, pure/partial/full  (LOCAL)
# ===========================================================================

_SPY_ABBR = {
    'vix_prev'     : 'VIX',
    'overnight_ret': 'ONR',
    'ann_indicator': 'Ann',
    'day_of_week'  : 'DoW',
    'trailing_rv'  : 'TRV',
    'month'        : 'Mon',
}

_NODE_POS_SPY = '#2a9d8f'
_NODE_NEG_SPY = '#e63946'
_EDGE_SYN_SPY = '#2a9d8f'
_EDGE_RED_SPY = '#e63946'


def _network_importances_spy(moebius, shapley, n_players, K, effect_type):
    pure_eff = _pure_effects(moebius, n_players)
    full_eff = _full_effects(moebius, n_players)

    if effect_type == 'pure':
        eff = pure_eff
    elif effect_type == 'partial':
        eff = shapley
    else:
        eff = full_eff

    node_imp  = np.array([
        float(np.sum(np.abs(apply_kernel(eff[i], K))))
        for i in range(n_players)])
    node_sign = np.array([
        np.sign(float(np.trapz(apply_kernel(eff[i], K), t_grid)))
        for i in range(n_players)])

    edge_imp = {}
    for i in range(n_players):
        for j in range(i + 1, n_players):
            if effect_type == 'pure':
                raw = moebius.get((i, j), np.zeros(T_BARS))
            elif effect_type == 'partial':
                raw = np.zeros(T_BARS)
                for S, m in moebius.items():
                    if i in S and j in S:
                        raw = raw + m / len(S)
            else:
                raw = np.zeros(T_BARS)
                for S, m in moebius.items():
                    if i in S and j in S:
                        raw = raw + m
            val = float(np.trapz(apply_kernel(raw, K), t_grid))
            if abs(val) > 1e-10:
                edge_imp[(i, j)] = val

    return node_imp, edge_imp, node_sign


def _draw_network_spy(ax, pnames, node_imp, edge_imp, node_sign, title):
    import math
    p     = len(pnames)
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
        xi, yi = pos[i]; xj, yj = pos[j]
        lw   = 0.4 + 6.5 * abs(val) / max_edge
        col  = _EDGE_SYN_SPY if val > 0 else _EDGE_RED_SPY
        alph = 0.30 + 0.60 * abs(val) / max_edge
        ax.plot([xi, xj], [yi, yj], color=col, lw=lw, alpha=alph,
                solid_capstyle='round', zorder=1)

    for i in range(p):
        x, y = pos[i]; r = node_r[i]
        fc   = _NODE_POS_SPY if node_sign[i] >= 0 else _NODE_NEG_SPY
        ax.add_patch(plt.Circle((x, y), r, color=fc, ec='white',
                                linewidth=1.2, zorder=2, alpha=0.88))
        ax.add_patch(plt.Circle((x, y), r * 0.52, color='white', ec='none',
                                zorder=3, alpha=0.95))
        abbr = _SPY_ABBR.get(pnames[i], pnames[i][:3])
        ax.text(x, y, abbr, ha='center', va='center',
                fontsize=max(5.5, r * 24), fontweight='bold',
                color='#222', zorder=4)

    pad = 0.32
    ax.set_xlim(-1.0 - pad, 1.0 + pad)
    ax.set_ylim(-1.0 - pad, 1.0 + pad)


def plot_network_all_games_ppf(moebius_dict, shapley_dict, pnames):
    """All 3 games x pure/partial/full, OU kernel (3x3 grid) — LOCAL."""
    K_ou      = kernel_ou(t_grid, 8.0)
    n_players = len(pnames)

    game_specs = [
        ('prediction',  'Prediction',  'PDP',               'SHAP',              'ICE-agg.'),
        ('sensitivity', 'Sensitivity', 'local closed Sobol', 'local Shapley-sens','local total Sobol'),
        ('risk',        'Risk (MSE)',  'local pure risk',    'local SAGE-style',  'local PFI-style'),
    ]

    fig = plt.figure(figsize=(11, 10.5))
    fig.suptitle(
        'Local network plots — OU kernel ($\\ell=8$ bars) — all games\n'
        'High-VIX Announcement profile  —  SPY Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.10, wspace=0.08,
                           left=0.08, right=0.98,
                           top=0.91, bottom=0.06)

    for r, (gtype, glabel, lp, lpa, lf) in enumerate(game_specs):
        col_titles = [
            'Pure  $m_i \\equiv$ {}'.format(lp),
            'Partial  $\\phi_i \\equiv$ {}'.format(lpa),
            'Full  $\\Phi_i \\equiv$ {}'.format(lf),
        ]
        for c, etype in enumerate(['pure', 'partial', 'full']):
            ax = fig.add_subplot(gs[r, c])
            node_imp, edge_imp, node_sign = _network_importances_spy(
                moebius_dict[gtype], shapley_dict[gtype],
                n_players, K_ou, etype)
            _draw_network_spy(ax, pnames, node_imp, edge_imp, node_sign,
                              col_titles[c] if r == 0 else '')
            if c == 0:
                ax.text(-0.03, 0.5, glabel, transform=ax.transAxes,
                        fontsize=FS_AXIS, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')

    leg_handles = [
        Patch(facecolor=_NODE_POS_SPY, edgecolor='none', label='Positive effect'),
        Patch(facecolor=_NODE_NEG_SPY, edgecolor='none', label='Negative effect'),
    ]
    fig.legend(handles=leg_handles, loc='lower center', ncol=2,
               fontsize=FS_LEGEND, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))
    return fig


# ===========================================================================
# 14. Global computation helpers
# ===========================================================================

def compute_global_shapley(predict_fn, X_background, Y_adj,
                            game_type, n_instances, sample_size, seed):
    """
    Average Shapley (partial) and Möbius pure/full effects over `n_instances`
    randomly drawn instances from X_background.

    Returns
    -------
    avg_shapley : dict  {feature_idx -> np.array shape (T,)}
    avg_pure    : dict  {feature_idx -> np.array shape (T,)}
    avg_full    : dict  {feature_idx -> np.array shape (T,)}
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X_background), size=n_instances, replace=False)
    n_players = len(DAY_FEATURE_NAMES)

    sum_shapley = {i: np.zeros(T_BARS) for i in range(n_players)}
    sum_pure    = {i: np.zeros(T_BARS) for i in range(n_players)}
    sum_full    = {i: np.zeros(T_BARS) for i in range(n_players)}

    for k, idx in enumerate(idxs):
        x_inst = X_background[idx]
        y_inst = Y_adj[idx] if game_type == 'risk' else None
        game = FunctionalGame(
            predict_fn   = predict_fn,
            X_background = X_background,
            x_explain    = x_inst,
            game_type    = game_type,
            Y_obs        = y_inst,
            sample_size  = sample_size,
            random_seed  = seed + k,   # vary seed per instance
        )
        game.precompute()
        mob  = functional_moebius_transform(game)
        shap = shapley_from_moebius(mob, n_players)
        pure = _pure_effects(mob, n_players)
        full = _full_effects(mob, n_players)
        for i in range(n_players):
            sum_shapley[i] += shap[i]
            sum_pure[i]    += pure[i]
            sum_full[i]    += full[i]
        print('    instance {}/{} done.'.format(k + 1, n_instances))

    avg_shapley = {i: sum_shapley[i] / n_instances for i in range(n_players)}
    avg_pure    = {i: sum_pure[i]    / n_instances for i in range(n_players)}
    avg_full    = {i: sum_full[i]    / n_instances for i in range(n_players)}
    return avg_shapley, avg_pure, avg_full


# ===========================================================================
# 15. Figure 2_global -- Global analogue of fig2 (identity kernel)
# ===========================================================================

def plot_global_main_effects(global_effects, pnames, top_k=5):
    """
    global_effects: dict  game_type -> (avg_shapley, avg_pure, avg_full)
    """
    game_types = ['prediction', 'sensitivity', 'risk']
    K_id       = kernel_identity(t_grid)
    n_players  = len(pnames)
    _leg_loc   = {0: 'upper center', 1: 'lower center', 2: 'lower center'}  

    fig, axes = plt.subplots(
        len(game_types), 4,
        figsize=(20, 4.5 * len(game_types)),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Global main effects — Identity kernel — pure / partial / full\n'
        '(averaged over {} instances)  —  Random Forest'.format(GLOBAL_N_INSTANCES),
        fontsize=FS_SUPTITLE, fontweight='bold')

    for r, gtype in enumerate(game_types):
        sc = _scale(gtype)
        avg_shapley, avg_pure, avg_full = global_effects[gtype]
        effect_dicts = {
            'pure'   : avg_pure,
            'partial': avg_shapley,
            'full'   : avg_full,
        }

        imps_partial = {i: float(np.sum(np.abs(avg_shapley[i])))
                        for i in range(n_players)}
        top = sorted(imps_partial, key=imps_partial.get, reverse=True)[:top_k]

        for c, etype in enumerate(_EFFECT_TYPES):
            ax  = axes[r, c]
            eff = effect_dicts[etype]
            for fi in top:
                ax.plot(t_grid, apply_kernel(eff[fi], K_id) * sc,
                        color=FEAT_COLORS[pnames[fi]], lw=2.0,
                        label=pnames[fi] if c == 0 else '_')
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.set_title(_XAI_LABELS_GLOBAL[(gtype, etype)],
                         fontsize=FS_TITLE, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=FS_AXIS)
                ax.legend(fontsize=FS_LEGEND, loc=_leg_loc[r], framealpha=0.85)

        ax_bar = axes[r, 3]
        imps_all = {
            etype: {i: float(np.sum(np.abs(effect_dicts[etype][i]))) * sc
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
            ax_bar.barh(y_pos + offsets[etype],
                        [imps_all[etype][i] for i in order],
                        height=bar_h,
                        color=[FEAT_COLORS[pnames[i]] for i in order],
                        alpha=alphas[etype], hatch=hatches[etype], label=etype)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([pnames[i] for i in order], fontsize=FS_TICK)
        ax_bar.axvline(0, color='gray', lw=0.8, ls=':')
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=FS_TICK)
        ax_bar.set_title('Integrated\nimportance', fontsize=FS_TITLE,
                         fontweight='bold')
        ax_bar.legend(fontsize=FS_LEGEND, loc='upper right')

    plt.tight_layout()
    return fig


# ===========================================================================
# 16. Figure 2b_global -- Global analogue of fig2b (mixed kernel)
# ===========================================================================

def plot_global_mixed_kernel_effects(global_effects, pnames, top_k=5):
    game_types   = ['prediction', 'sensitivity', 'risk']
    n_players    = len(pnames)
    feat_kernels = {fi: _feature_kernel_hv(fi, pnames)
                    for fi in range(n_players)}
    _leg_loc = {0: 'upper center', 1: 'lower center', 2: 'lower center'}  

    fig, axes = plt.subplots(
        len(game_types), 4,
        figsize=(20, 4.5 * len(game_types)),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]})
    fig.suptitle(
        'Global main effects — mixed kernel (OU + causal) — pure / partial / full\n'
        'ann_indicator: causal ($\\ell=8$)  |  all others: OU ($\\ell=8$)  '
        '—  averaged over {} instances  —  Random Forest'.format(GLOBAL_N_INSTANCES),
        fontsize=FS_SUPTITLE, fontweight='bold')

    for r, gtype in enumerate(game_types):
        sc = _scale(gtype)
        avg_shapley, avg_pure, avg_full = global_effects[gtype]
        effect_dicts = {
            'pure'   : avg_pure,
            'partial': avg_shapley,
            'full'   : avg_full,
        }

        imps_partial = {
            i: float(np.sum(np.abs(apply_kernel(avg_shapley[i], feat_kernels[i]))))
            for i in range(n_players)}
        top = sorted(imps_partial, key=imps_partial.get, reverse=True)[:top_k]

        for c, etype in enumerate(_EFFECT_TYPES):
            ax  = axes[r, c]
            eff = effect_dicts[etype]
            leg_fs = FS_LEGEND - 2.5 if (r == 0 and c == 0) else FS_LEGEND
            for fi in top:
                ls = '--' if pnames[fi] == 'ann_indicator' else '-'
                ax.plot(t_grid,
                        apply_kernel(eff[fi], feat_kernels[fi]) * sc,
                        color=FEAT_COLORS[pnames[fi]], lw=2.0, ls=ls,
                        label=pnames[fi] if c == 0 else '_')
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.set_title(_XAI_LABELS_GLOBAL[(gtype, etype)],
                         fontsize=FS_TITLE, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=FS_AXIS)
                ax.legend(fontsize=leg_fs, loc=_leg_loc[r], framealpha=0.85)

        ax_bar = axes[r, 3]
        imps_all = {
            etype: {
                i: float(np.sum(np.abs(
                    apply_kernel(effect_dicts[etype][i], feat_kernels[i])))) * sc
                for i in range(n_players)}
            for etype in _EFFECT_TYPES}
        order   = sorted(range(n_players),
                         key=lambda i: imps_all['partial'][i], reverse=True)
        y_pos   = np.arange(len(order))
        bar_h   = 0.25
        offsets = {'pure': -bar_h, 'partial': 0.0, 'full': bar_h}
        alphas  = {'pure': 0.55,   'partial': 0.90, 'full': 0.55}
        hatches = {'pure': '//',   'partial': '',   'full': '\\\\'}
        for etype in _EFFECT_TYPES:
            ax_bar.barh(y_pos + offsets[etype],
                        [imps_all[etype][i] for i in order],
                        height=bar_h,
                        color=[FEAT_COLORS[pnames[i]] for i in order],
                        alpha=alphas[etype], hatch=hatches[etype], label=etype)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([pnames[i] for i in order], fontsize=FS_TICK)
        ax_bar.axvline(0, color='gray', lw=0.8, ls=':')
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=FS_TICK)
        ax_bar.set_title('Integrated\nimportance\n(mixed kernel)',
                         fontsize=FS_TITLE, fontweight='bold')
        ax_bar.legend(fontsize=FS_LEGEND, loc='upper right')

    plt.tight_layout()
    return fig


# ===========================================================================
# 17. Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  SPY Intraday Vol  —  RF direct  (v3: global + local fixes)')
    print('=' * 60)

    _require_dir(BASE_PLOT_DIR)
    _require_dir(DATA_DIR)

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
    print('  Test R² (trajectory-level): {:.4f}'.format(r2))

    # ── 3. Profiles ───────────────────────────────────────────────────────
    print('\n[3] Selecting profiles ...')
    vix_col  = DAY_FEATURE_NAMES.index('vix_prev')
    vix_p25  = float(np.percentile(X_day_np[:, vix_col], 25))
    vix_p75  = float(np.percentile(X_day_np[:, vix_col], 75))

    def find_profile(conditions):
        mask = np.ones(len(X_day_np), dtype=bool)
        for feat, (lo, hi) in conditions.items():
            ci = DAY_FEATURE_NAMES.index(feat)
            mask &= (X_day_np[:, ci] >= lo) & (X_day_np[:, ci] <= hi)
        hits = X_day_np[mask]
        if len(hits) == 0:
            raise RuntimeError('No day matches: {}'.format(conditions))
        print('  {} matching days; picking median.'.format(len(hits)))
        return hits[len(hits) // 2]

    x_p1 = find_profile(
        {'ann_indicator': (0.9, 1.1), 'vix_prev': (vix_p75, 999)})
    x_p2 = find_profile(
        {'ann_indicator': (-0.1, 0.1), 'vix_prev': (0, vix_p25)})
    x_p3 = find_profile({'day_of_week': (-0.1, 0.1)})

    def _y_obs(x_prof):
        diffs = np.abs(X_day_np - x_prof[None, :]).sum(axis=1)
        return Y_adj[int(np.argmin(diffs))]

    profiles = [
        ('High-VIX Announcement', x_p1, _y_obs(x_p1)),
        ('Quiet Low-VIX',         x_p2, _y_obs(x_p2)),
        ('Monday Gap',            x_p3, _y_obs(x_p3)),
    ]
    pnames = list(DAY_FEATURE_NAMES)

    # ── 4. Local games (High-VIX profile, all 3 games) ────────────────────
    print('\n[4] Computing local games for High-VIX Announcement profile ...')
    x_hv, y_hv = x_p1, _y_obs(x_p1)
    moebius_hv = {}
    shapley_hv = {}

    for gtype in ('prediction', 'sensitivity', 'risk'):
        print('\n  game: {} ...'.format(gtype))
        game = FunctionalGame(
            predict_fn   = model.predict,
            X_background = X_day_np,
            x_explain    = x_hv,
            game_type    = gtype,
            Y_obs        = y_hv,
            sample_size  = SAMPLE_SIZE[gtype],
            random_seed  = RNG_SEED,
        )
        game.precompute()
        moebius_hv[gtype] = functional_moebius_transform(game)
        shapley_hv[gtype] = shapley_from_moebius(
            moebius_hv[gtype], game.n_players)

    # ── 5. Local games (other profiles, prediction only) ──────────────────
    print('\n[5] Computing local prediction game for remaining profiles ...')
    profile_results = {}
    for label, x_prof, y_prof in profiles:
        print('\n  profile: {} ...'.format(label))
        game = FunctionalGame(
            predict_fn   = model.predict,
            X_background = X_day_np,
            x_explain    = x_prof,
            game_type    = 'prediction',
            sample_size  = SAMPLE_SIZE['prediction'],
            random_seed  = RNG_SEED,
        )
        game.precompute()
        mob  = functional_moebius_transform(game)
        shap = shapley_from_moebius(mob, game.n_players)
        profile_results[label] = (mob, shap)

    # ── 6. Global games (all 3 game types) ────────────────────────────────
    print('\n[6] Computing global effects ({} instances per game) ...'.format(
        GLOBAL_N_INSTANCES))
    global_effects = {}
    for gtype in ('prediction', 'sensitivity', 'risk'):
        print('\n  global game: {} ...'.format(gtype))
        avg_shap, avg_pure, avg_full = compute_global_shapley(
            predict_fn   = model.predict,
            X_background = X_day_np,
            Y_adj        = Y_adj,
            game_type    = gtype,
            n_instances  = GLOBAL_N_INSTANCES,
            sample_size  = GLOBAL_SAMPLE_SIZE,
            seed         = RNG_SEED,
        )
        global_effects[gtype] = (avg_shap, avg_pure, avg_full)

    # ── 7. Generate figures ───────────────────────────────────────────────
    print('\n[7] Generating figures ...')

    savefig(plot_main_body_summary_v2(moebius_hv, shapley_hv, pnames),
            'fig0_main_body_summary_v2.pdf')

    savefig(plot_main_effects_all_games(moebius_hv, shapley_hv, pnames, top_k=5),
            'fig2_local_main_effects_all_games.pdf')

    savefig(plot_mixed_kernel_effects(moebius_hv, shapley_hv, pnames, top_k=5),
            'fig2b_local_mixed_kernel_effects.pdf')

    savefig(plot_global_main_effects(global_effects, pnames, top_k=5),
            'fig2_global_main_effects_all_games.pdf')

    savefig(plot_global_mixed_kernel_effects(global_effects, pnames, top_k=5),
            'fig2b_global_mixed_kernel_effects.pdf')

    savefig(plot_interactions(moebius_hv, pnames),
            'fig4_interactions.pdf')

    savefig(plot_profiles_comparison(profile_results, pnames),
            'fig5_profiles_comparison.pdf')

    savefig(plot_network_all_games_ppf(moebius_hv, shapley_hv, pnames),
            'figA_network_all_games_ppf.pdf')

    print('\n' + '=' * 60)
    print('  Done.  Figures saved under {}/'.format(BASE_PLOT_DIR))
    print('=' * 60)