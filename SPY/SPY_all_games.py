"""
Functional Explanation Framework -- Intraday SPY Volatility (Random Forest)
============================================================================
Model:  F^H : R^6 -> R^78
  RandomForestRegressor (multi-output, direct, no PCA)
  t is NEVER an input feature.

Plots generated (focused on paper storyline):
  1. fig1_operator_sweep.pdf          -- 3x4 grid: games x kernels, vix_prev + ann_indicator
  2. fig2_main_effects_all_games.pdf  -- 3x4 grid: games x (pure/partial/full + importance bars)
                                         identity kernel, XAI equiv. labelled in headers
  3. fig3_ann_causal_comparison.pdf   -- ann_indicator under identity/OU/causal kernels,
                                         prediction + risk game side by side
  3b.fig3b_sensitivity_gap.pdf        -- NEW: Δτ_i(t) = total Sobol − closed Sobol, top-4 features
  4. fig4_interactions.pdf            -- 2x3 grid: prediction + risk game, top-3 pairs, OU kernel
  5. fig5_profiles_comparison.pdf     -- Shapley curves (OU kernel), all 3 profiles,
                                         prediction game only

Kernels used:
  Identity    -- recovers classical pointwise SHAP (special case)
  OU          -- AR(1)-type temporal smoothing, ell=8 bars (~40 min)
  Correlation -- empirical bar-correlation kernel, data-adaptive
  Causal      -- one-sided exponential, enforces forward causality for ann_indicator

Games:
  prediction  -- v(S)(t) = E[F(X)(t) | X_S]         (= pointwise SHAP under identity)
  sensitivity -- v(S)(t) = Var[F(X)(t) | X_S]        (novel; no existing XAI equivalent)
  risk        -- v(S)(t) = E[(Y(t)-F(X)(t))^2 | X_S] (novel; MSE attribution)

Pure / partial / full effect taxonomy (Fumagalli et al. 2025):
  pure    -- m_i(t)                       prediction=PDP,  sensitivity=Closed Sobol,  risk=Pure Risk
  partial -- phi_i(t) (Shapley)           prediction=SHAP, sensitivity=Shapley-sens,  risk=SAGE
  full    -- sum_{S ni i} m_S(t)          prediction=ICE-agg, sensitivity=Total Sobol, risk=PFI
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

_open_min  = 9 * 60 + 30
BAR_LABELS = [
    '{:02d}:{:02d}'.format(
        (_open_min + i * 5) // 60,
        (_open_min + i * 5) % 60,
    )
    for i in range(T_BARS)
]
t_grid = np.arange(T_BARS, dtype=float)

BASE_PLOT_DIR = os.path.join(
    'plots', 'SPY_all_games')

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
        raise RuntimeError(
            'Only {} complete trading days.'.format(len(pivot)))


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
        return {str(k): float(v)
                for k, v in df['vix'].dropna().items()}
    import yfinance as yf
    vix_start = (pd.Timestamp(first_date)
                 - pd.Timedelta(days=VIX_LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    end_ex = (pd.Timestamp(last_date)
              + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    raw    = yf.download('^VIX', start=vix_start,
                         end=end_ex, progress=False, auto_adjust=True)
    close  = _resolve_close_column(raw, 'yfinance ^VIX')
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
    if not POLYGON_API_KEY:
        raise RuntimeError(
            'No bar cache and POLYGON_API_KEY not set.\n'
            'Place a CSV at: {}'.format(CACHE_PATH))
    raise RuntimeError('Polygon fetch not implemented in this script.')


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
    bars = bars[
        (bars['bar_idx'] >= 0) & (bars['bar_idx'] < T_BARS)
    ].copy()
    bars['open']  = pd.to_numeric(bars['open'],  errors='coerce')
    bars['close'] = pd.to_numeric(bars['close'], errors='coerce')
    bars = bars[(bars['open'] > 0) & (bars['close'] > 0)].copy()
    bars['abs_log_ret'] = np.abs(
        np.log(bars['close'] / bars['open']))

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
        dt_obj   = pd.Timestamp(date_str)
        vix_prev = _resolve_vix_prev(vix_dict, date_str)
        day_bars  = bars[bars['date'] == date_str].sort_values('bar_idx')
        prev_bars = bars[
            bars['date'] < date_str].sort_values(['date', 'bar_idx'])
        this_open = float(day_bars.iloc[0]['open'])
        overnight_ret = (
            0.0 if len(prev_bars) == 0
            else float(np.log(
                this_open / float(prev_bars.iloc[-1]['close'])))
        )
        recent      = list(range(max(0, i - 5), i))
        trailing_rv = (
            float(Y_day[recent].mean()) if recent
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
# 2.  Model  -- Random Forest, direct multi-output, no PCA
# ===========================================================================

class RFModel:
    """
    RandomForestRegressor mapping X: (N,6) -> Y: (N,78) directly.
    sklearn RF natively supports multi-output regression.
    No PCA, no t as input.
    """
    def __init__(self, n_estimators=RF_N_EST,
                 n_jobs=RF_JOBS, random_state=RNG_SEED):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)   # (N, T)

    def evaluate(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        ss_res = np.sum((Y_test - Y_pred) ** 2)
        ss_tot = np.sum((Y_test - Y_test.mean()) ** 2)
        return 1.0 - ss_res / ss_tot


# ===========================================================================
# 3.  Cooperative game
# ===========================================================================

class FunctionalGame:
    """
    game_type:
      prediction  -- v(S)(t) = E[F(X)(t) | X_S]
      sensitivity -- v(S)(t) = Var[F(X)(t) | X_S]  (empirical over bg samples)
      risk        -- v(S)(t) = E[(Y_obs(t) - F(X)(t))^2 | X_S]
    """

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
            list(itertools.product(
                [False, True], repeat=self.n_players)),
            dtype=bool)
        self.n_coalitions = len(self.coalitions)
        self._idx = {
            tuple(c): i for i, c in enumerate(self.coalitions)}
        self.values = None

    def _impute(self, coalition):
        rng = np.random.default_rng(self.random_seed)
        idx = rng.integers(
            0, len(self.X_background), size=self.sample_size)
        X = self.X_background[idx].copy()
        for j in range(self.n_players):
            if coalition[j]:
                X[:, j] = self.x_explain[j]
        return X

    def value_function(self, coalition):
        X      = self._impute(coalition)
        Y_pred = self.predict_fn(X)       # (sample_size, T)
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
        itertools.combinations(range(p), r)
        for r in range(p + 1)))
    moebius = {}
    for S in all_S:
        m = np.zeros(game.T)
        for L in itertools.chain.from_iterable(
            itertools.combinations(S, r)
            for r in range(len(S) + 1)
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


# ===========================================================================
# 6.  Kernel application
# ===========================================================================

def _normalize_kernel(K):
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return K / rs

def apply_kernel(effect, K):
    return _normalize_kernel(K) @ effect * dt


# ===========================================================================
# 7.  Pure / partial / full effect helpers
# ===========================================================================

def _pure_effects(moebius, n_players):
    """
    Pure (main) effect for player i: m_{(i,)}(t).
    Functional analogue of PDP / Closed Sobol / Pure Risk.
    """
    return {i: moebius.get((i,), np.zeros(T_BARS)).copy()
            for i in range(n_players)}


def _full_effects(moebius, n_players):
    """
    Full (superset) effect for player i:
      Phi_i^full(t) = sum_{S containing i} m_S(t)
    Absorbs ALL interaction mass touching i, no fairness redistribution.
    Functional analogue of ICE-aggregate / Total Sobol / PFI.
    """
    full = {i: np.zeros(T_BARS) for i in range(n_players)}
    for S, m in moebius.items():
        if len(S) == 0:
            continue
        for i in S:
            full[i] += m
    return full


# ===========================================================================
# 8.  Plotting helpers
# ===========================================================================

XTICK_IDXS   = list(range(0, T_BARS, 6))
XTICK_LABELS = [BAR_LABELS[i] for i in XTICK_IDXS]

# ── Global font sizes (applied consistently across all figures) ───────────
FS_SUPTITLE = 13    # figure-level suptitle
FS_TITLE    = 11    # subplot title
FS_AXIS     = 10    # axis labels (x/y)
FS_TICK     = 9     # tick labels
FS_LEGEND   = 8.5   # legend text
FS_ANNOT    = 8.5   # in-plot text annotations

# Feature colours -- consistent across all plots
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

KERNEL_LABELS = {
    'Identity'   : 'Identity\n(pointwise SHAP)',
    'OU'         : 'OU  ($\\ell=8$ bars)',
    'Correlation': 'Empirical\ncorrelation',
    'Causal'     : 'Causal\n($\\ell=8$ bars)',
}

# XAI equivalence labels for pure/partial/full x game combinations
_XAI_LABELS = {
    ('prediction',  'pure')   : 'Pure  $m_i(t)$  $\\equiv$  PDP',
    ('prediction',  'partial'): 'Partial  $\\phi_i(t)$  $\\equiv$  Shapley (SHAP)',
    ('prediction',  'full')   : 'Full  $\\Phi_i(t)$  $\\equiv$  ICE-aggregate',
    ('sensitivity', 'pure')   : 'Pure  $\\equiv$  Closed Sobol  $\\tau^{\\mathrm{cl}}_i$',
    ('sensitivity', 'partial'): 'Partial  $\\equiv$  Shapley-sensitivity',
    ('sensitivity', 'full')   : 'Full  $\\equiv$  Total Sobol  $\\bar{\\tau}_i$',
    ('risk',        'pure')   : 'Pure  $\\equiv$  Pure Risk',
    ('risk',        'partial'): 'Partial  $\\equiv$  SAGE',
    ('risk',        'full')   : 'Full  $\\equiv$  PFI',
}

_EFFECT_TYPES = ['pure', 'partial', 'full']

def _scale(game_type):
    return 100.0 if game_type == 'prediction' else 1e4

def _set_time_axis(ax, sparse=False):
    step = 2 if sparse else 1
    ax.set_xticks(XTICK_IDXS[::step])
    ax.set_xticklabels(
        XTICK_LABELS[::step],
        rotation=45, ha='right', fontsize=FS_TICK)
    ax.set_xlim(-0.5, T_BARS - 0.5)

def _period_shade(ax):
    ax.axvspan(0,  6,  alpha=0.08, color='#ffd699', zorder=0)
    ax.axvspan(72, 78, alpha=0.08, color='#ffd699', zorder=0)

def _ann_vline(ax):
    """Mark bar 54 = 14:00 ET (Beige Book / FOMC announcement time)."""
    ax.axvline(54, color='#555', lw=0.9, ls='--', alpha=0.6)

def savefig(fig, name):
    path = os.path.join(BASE_PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print('  Saved: {}'.format(path))
    plt.close(fig)

def _top_features(moebius_dict, n_players, top_k):
    """
    Rank features by mean integrated |m_i(t)| across all games/profiles
    stored in moebius_dict (values are moebius dicts keyed by subset tuple).
    """
    imps = np.zeros(n_players)
    for mob in moebius_dict.values():
        for i in range(n_players):
            imps[i] += float(np.sum(np.abs(mob[(i,)])))
    return sorted(range(n_players),
                  key=lambda i: imps[i], reverse=True)[:top_k]


# ===========================================================================
# 9.  Figure 1 -- Operator sweep  (central paper figure)
#     Rows: prediction / sensitivity / risk
#     Cols: Identity / OU / Correlation / Causal
#     Each cell: Shapley curves for vix_prev and ann_indicator
# ===========================================================================

def plot_operator_sweep(games_moebius, shapley_dict, kernels, pnames):
    """
    games_moebius : dict  game_type -> moebius
    shapley_dict  : dict  game_type -> shapley
    kernels       : ordered dict  kernel_name -> K matrix
    pnames        : list of feature names
    """
    game_types  = ['prediction', 'sensitivity', 'risk']
    kernel_names = list(kernels.keys())   # Identity, OU, Correlation, Causal
    nrows, ncols = len(game_types), len(kernel_names)

    # Features to highlight: vix_prev and ann_indicator
    fi_vix = pnames.index('vix_prev')
    fi_ann = pnames.index('ann_indicator')
    focus  = [fi_vix, fi_ann]
    colors = [FEAT_COLORS['vix_prev'], FEAT_COLORS['ann_indicator']]
    labels = ['vix_prev', 'ann_indicator']

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 3.2 * nrows),
        sharey='row',
    )
    fig.suptitle(
        'Operator sweep: Shapley curves across games and kernels\n'
        'High-VIX Announcement profile  —  Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    for r, gtype in enumerate(game_types):
        sc      = _scale(gtype)
        shapley = shapley_dict[gtype]

        for c, kname in enumerate(kernel_names):
            ax = axes[r, c]
            K  = kernels[kname]

            for fi, col, lbl in zip(focus, colors, labels):
                curve = apply_kernel(shapley[fi], K) * sc
                ax.plot(t_grid, curve,
                        color=col, lw=2.0,
                        label=lbl if (r == 0 and c == 0) else '_')

            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            _ann_vline(ax)
            _set_time_axis(ax, sparse=True)

            # Column headers (top row only)
            if r == 0:
                ax.set_title(
                    KERNEL_LABELS[kname],
                    fontsize=FS_TITLE, fontweight='bold')

            # Row labels (left column only)
            if c == 0:
                ax.set_ylabel(
                    GAME_YLABEL[gtype], fontsize=FS_AXIS)
                ax.text(
                    -0.38, 0.5,
                    GAME_LABELS[gtype],
                    transform=ax.transAxes,
                    fontsize=FS_AXIS - 1, va='center',
                    ha='right', rotation=90,
                    color='#333',
                )

            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.tick_params(axis='y', labelsize=FS_TICK)

            # Annotate the identity+prediction cell
            if r == 0 and c == 0:
                ax.text(
                    0.97, 0.97,
                    '= pointwise\nSHAP',
                    transform=ax.transAxes,
                    fontsize=FS_ANNOT - 1, va='top', ha='right',
                    color='#555',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', ec='#aaa', alpha=0.8),
                )

    # Legend: upper-left of row-0 col-0 (Identity / prediction panel)
    handles = [
        plt.Line2D([0], [0], color=FEAT_COLORS['vix_prev'],
                   lw=2, label='vix_prev'),
        plt.Line2D([0], [0], color=FEAT_COLORS['ann_indicator'],
                   lw=2, label='ann_indicator'),
        plt.Line2D([0], [0], color='#555', lw=0.9, ls='--',
                   alpha=0.6, label='14:00 ET'),
    ]
    axes[0, 0].legend(
        handles=handles,
        loc='upper left',
        fontsize=FS_LEGEND,
        framealpha=0.9,
    )

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    return fig


# ===========================================================================
# 10. Figure 2 -- Main effects, pure / partial / full, all games
#     Rows: prediction / sensitivity / risk
#     Cols: pure (PDP/Closed Sobol/Pure Risk) /
#           partial (SHAP/Shapley-sens/SAGE) /
#           full (ICE-agg/Total Sobol/PFI) /
#           integrated importance bars (all three overlaid)
# ===========================================================================

def plot_main_effects_all_games(moebius_dict, shapley_dict, pnames, top_k=5):
    """
    moebius_dict  : dict  game_type -> moebius
    shapley_dict  : dict  game_type -> shapley  (partial effects)
    pnames        : list of feature names
    top_k         : number of features to show in curve panels
    """
    game_types = ['prediction', 'sensitivity', 'risk']
    K_id       = kernel_identity(t_grid)
    n_players  = len(pnames)

    fig, axes = plt.subplots(
        len(game_types), 4,
        figsize=(18, 4.0 * len(game_types)),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]},
    )
    fig.suptitle(
        'Main effects — Identity kernel — pure / partial / full\n'
        'High-VIX Announcement profile  —  Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    # Legend location for col-0 curve panel per row
    _leg_loc = {0: 'upper center', 1: 'upper left', 2: 'lower center'}

    for r, gtype in enumerate(game_types):
        sc      = _scale(gtype)
        moebius = moebius_dict[gtype]

        pure_eff    = _pure_effects(moebius, n_players)
        partial_eff = shapley_dict[gtype]       # already computed Shapley values
        full_eff    = _full_effects(moebius, n_players)

        effect_dicts = {
            'pure'   : pure_eff,
            'partial': partial_eff,
            'full'   : full_eff,
        }

        # Rank features by partial (Shapley) integrated abs importance
        # and keep ranking consistent across all three effect columns
        imps_partial = {i: float(np.sum(np.abs(partial_eff[i])))
                        for i in range(n_players)}
        top = sorted(imps_partial, key=imps_partial.get, reverse=True)[:top_k]

        # ── curve panels (cols 0-2) ───────────────────────────────────────
        for c, etype in enumerate(_EFFECT_TYPES):
            ax  = axes[r, c]
            eff = effect_dicts[etype]

            for fi in top:
                curve = apply_kernel(eff[fi], K_id) * sc
                ax.plot(t_grid, curve,
                        color=FEAT_COLORS[pnames[fi]],
                        lw=2.0,
                        label=pnames[fi] if c == 0 else '_')

            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            _ann_vline(ax)
            _set_time_axis(ax)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.set_title(_XAI_LABELS[(gtype, etype)],
                         fontsize=FS_TITLE, fontweight='bold')

            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=FS_AXIS)
                ax.legend(fontsize=FS_LEGEND,
                          loc=_leg_loc[r], framealpha=0.85)

        # ── integrated importance bars (col 3) ────────────────────────────
        ax_bar = axes[r, 3]
        imps_all = {
            etype: {i: float(np.sum(np.abs(effect_dicts[etype][i]))) * sc
                    for i in range(n_players)}
            for etype in _EFFECT_TYPES
        }
        order = sorted(range(n_players),
                       key=lambda i: imps_all['partial'][i], reverse=True)
        y_pos   = np.arange(len(order))
        bar_h   = 0.25
        offsets = {'pure': -bar_h, 'partial': 0.0, 'full': bar_h}
        alphas  = {'pure': 0.55,   'partial': 0.90, 'full': 0.55}
        hatches = {'pure': '//',   'partial': '',   'full': '\\\\'}

        for etype in _EFFECT_TYPES:
            vals = [imps_all[etype][i] for i in order]
            ax_bar.barh(
                y_pos + offsets[etype], vals,
                height=bar_h,
                color=[FEAT_COLORS[pnames[i]] for i in order],
                alpha=alphas[etype],
                hatch=hatches[etype],
                label=etype,
            )

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
# 11. Figure 3 -- ann_indicator causal kernel comparison
#     Rows: prediction game / risk game
#     Cols: Identity / OU / Causal (ls=4) / Causal (ls=8) / Causal (ls=16)
# ===========================================================================

def plot_ann_causal_comparison(moebius_dict, pnames):
    ann_idx = pnames.index('ann_indicator')
    game_types = ['prediction', 'risk']

    kernels_ordered = {
        'Identity'      : kernel_identity(t_grid),
        'OU ($\\ell$=8)': kernel_ou(t_grid, 8.0),
        'Causal $\\ell$=4' : kernel_causal(t_grid, 4.0),
        'Causal $\\ell$=8' : kernel_causal(t_grid, 8.0),
        'Causal $\\ell$=16': kernel_causal(t_grid, 16.0),
    }
    knames = list(kernels_ordered.keys())
    ncols  = len(knames)

    causal_palette = plt.get_cmap('YlOrRd')(
        np.linspace(0.45, 0.85, 3))

    fig, axes = plt.subplots(
        len(game_types), ncols,
        figsize=(3.2 * ncols, 3.5 * len(game_types)),
        sharey='row',
    )
    fig.suptitle(
        'ann_indicator — symmetric vs causal kernel\n'
        'High-VIX Announcement profile  —  Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    for r, gtype in enumerate(game_types):
        sc  = _scale(gtype)
        raw = moebius_dict[gtype][(ann_idx,)]

        for c, (kname, K) in enumerate(kernels_ordered.items()):
            ax    = axes[r, c]
            is_id = kname == 'Identity'
            is_ou = kname.startswith('OU')

            if is_id:
                col = '#444444'
            elif is_ou:
                col = '#457b9d'
            else:
                ci  = c - 2
                col = causal_palette[ci]

            curve = apply_kernel(raw, K) * sc
            ax.plot(t_grid, curve, color=col, lw=2.2)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            _ann_vline(ax)
            _set_time_axis(ax, sparse=True)
            ax.tick_params(axis='y', labelsize=7)

            if r == 0:
                ax.set_title(kname, fontsize=8.5, fontweight='bold')

            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
                ax.text(
                    -0.35, 0.5,
                    GAME_LABELS[gtype],
                    transform=ax.transAxes,
                    fontsize=7.5, va='center',
                    ha='right', rotation=90, color='#333',
                )

            ax.set_xlabel('Time', fontsize=7)

            if is_ou and r == 0:
                ax.text(
                    0.05, 0.97,
                    'leakage\nbefore 14:00',
                    transform=ax.transAxes,
                    fontsize=6.5, va='top', color='#c0392b',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='#ffeaea', ec='#c0392b', alpha=0.9),
                )

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    return fig


# ===========================================================================
# 12. Figure 3b -- Sensitivity gap  Δτ_i(t) = τ̄_i(t) − τ_i(t)
#     Standalone panel showing when interactions are temporally active.
#     Functional analogue of the Total Sobol − Closed Sobol gap.
#     One subplot per top-k feature (ranked by integrated |gap|).
# ===========================================================================

def plot_sensitivity_gap(moebius_sens, pnames, top_k=4):
    """
    moebius_sens : moebius dict for the sensitivity game
    pnames       : list of feature names
    top_k        : number of features to display
    """
    n_players = len(pnames)
    K_id      = kernel_identity(t_grid)
    sc        = _scale('sensitivity')

    pure_eff = _pure_effects(moebius_sens, n_players)
    full_eff = _full_effects(moebius_sens, n_players)

    # Gap trajectory: full − pure = interaction contribution to sensitivity
    gap = {i: full_eff[i] - pure_eff[i] for i in range(n_players)}

    # Rank features by integrated |gap|
    gap_imp = {i: float(np.sum(np.abs(gap[i]))) for i in range(n_players)}
    top     = sorted(gap_imp, key=gap_imp.get, reverse=True)[:top_k]

    fig, axes = plt.subplots(
        1, top_k,
        figsize=(4.5 * top_k, 4.2),
        sharey=False,
    )
    fig.suptitle(
        r'Sensitivity gap  $\Delta\tau_i(t) = \bar{\tau}_i(t) - \tau^{\mathrm{cl}}_i(t)$'
        '  —  Identity kernel\n'
        r'Total Sobol $\bar{\tau}_i$ minus Closed Sobol $\tau^{\mathrm{cl}}_i$:'
        '  interaction contribution over time\n'
        'High-VIX Announcement profile  —  Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    # Legend location per subplot: first two lower-left, last two lower-center
    _leg_locs = ['lower left', 'lower left', 'upper center', 'lower center']

    for idx, (ax, fi) in enumerate(zip(axes, top)):
        col = FEAT_COLORS[pnames[fi]]

        pure_curve = apply_kernel(pure_eff[fi], K_id) * sc
        full_curve = apply_kernel(full_eff[fi], K_id) * sc
        gap_curve  = apply_kernel(gap[fi],      K_id) * sc

        # Shaded region between pure and full visualises the gap
        ax.fill_between(t_grid, pure_curve, full_curve,
                        color=col, alpha=0.18, label='gap region')
        ax.plot(t_grid, full_curve, color=col, lw=2.0, ls='-',
                label=r'Full  $\bar{\tau}_i$  (Total Sobol)')
        ax.plot(t_grid, pure_curve, color=col, lw=2.0, ls='--',
                label=r'Pure  $\tau^{\mathrm{cl}}_i$  (Closed Sobol)')
        ax.plot(t_grid, gap_curve,  color='black', lw=1.4, ls=':',
                alpha=0.7, label=r'Gap  $\Delta\tau_i$')

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax)
        _ann_vline(ax)
        _set_time_axis(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_ylabel(GAME_YLABEL['sensitivity'], fontsize=FS_AXIS)
        ax.set_title(
            '{}\n'
            r'$\int|\Delta\tau_i|\,dt$ = {:.4f}'.format(
                pnames[fi],
                float(np.trapz(np.abs(gap[fi]), t_grid)) * sc),
            fontsize=FS_TITLE, fontweight='bold',
            color=col,
        )
        ax.legend(fontsize=FS_LEGEND,
                  loc=_leg_locs[idx], framealpha=0.85)

    plt.tight_layout()
    return fig


# ===========================================================================
# 13. Figure 4 -- Pairwise interaction effects, prediction + risk game
#     Row 0: prediction game  ≡  Shapley interaction (2-SII)
#     Row 1: risk game        ≡  SAGE interaction
#     Cols: top-3 pairs ranked by prediction game importance
#     OU kernel throughout
# ===========================================================================

def plot_interactions(moebius_dict, pnames):
    """
    moebius_dict : dict  game_type -> moebius
    pnames       : list of feature names
    """
    K_ou  = kernel_ou(t_grid, 8.0)
    n     = len(pnames)

    game_rows = [
        ('prediction', 'Prediction  $\\equiv$  Shapley interaction (2-SII)'),
        ('risk',       'Risk  $\\equiv$  SAGE interaction'),
    ]

    # Rank pairs by prediction game integrated |m_ij|
    moebius_pred = moebius_dict['prediction']
    pair_imp = {
        (i, j): float(np.sum(np.abs(moebius_pred.get((i, j), np.zeros(T_BARS)))))
        for i in range(n) for j in range(i + 1, n)
    }
    top3        = sorted(pair_imp, key=pair_imp.get, reverse=True)[:3]
    pair_colors = ['#e63946', '#2a9d8f', '#8338ec']

    fig, axes = plt.subplots(
        2, 3,
        figsize=(13, 7.0),
        sharey=False,
    )
    fig.suptitle(
        'Pairwise interaction effects  $m_{ij}(t)$  —  OU kernel ($\\ell=8$ bars)\n'
        'Top-3 pairs ranked by prediction game  —  High-VIX Announcement profile',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    for r, (gtype, row_label) in enumerate(game_rows):
        sc      = _scale(gtype)
        moebius = moebius_dict[gtype]

        for c, ((i, j), col) in enumerate(zip(top3, pair_colors)):
            ax  = axes[r, c]
            raw = moebius.get((i, j), np.zeros(T_BARS))
            curve = apply_kernel(raw, K_ou) * sc

            ax.plot(t_grid, curve, color=col, lw=2.2)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            _ann_vline(ax)
            _set_time_axis(ax)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)

            # Pair name as column title (top row only)
            if r == 0:
                ax.set_title(
                    '{} $\\times$ {}'.format(pnames[i], pnames[j]),
                    fontsize=FS_TITLE, fontweight='bold')

            # y-label + XAI row label on left column
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=FS_AXIS)
                ax.text(
                    -0.30, 0.5, row_label,
                    transform=ax.transAxes,
                    fontsize=FS_AXIS - 1, va='center', ha='right',
                    rotation=90, color='#333',
                )

            # Integral annotation: upper-left for row0 cols 0&1 and row1 col0, upper-right elsewhere
            integ = float(np.trapz(raw, t_grid)) * sc
            if (r == 0 and c in (0, 1)) or (r == 1 and c == 0):
                ax.text(
                    0.03, 0.97,
                    r'$\int m_{{ij}}\,dt$ = {:.3f}'.format(integ),
                    transform=ax.transAxes,
                    fontsize=FS_ANNOT, va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', ec='#aaa', alpha=0.8),
                )
            else:
                ax.text(
                    0.97, 0.97,
                    r'$\int m_{{ij}}\,dt$ = {:.3f}'.format(integ),
                    transform=ax.transAxes,
                    fontsize=FS_ANNOT, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', ec='#aaa', alpha=0.8),
                )

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    return fig


# ===========================================================================
# 14. Figure 5 -- Profile comparison
#     Prediction game, OU kernel, all 3 profiles, top-4 features
# ===========================================================================

def plot_profiles_comparison(profile_results, pnames):
    """
    profile_results : dict  label -> (moebius, shapley)
    """
    K_ou    = kernel_ou(t_grid, 8.0)
    n_prof  = len(profile_results)
    sc      = _scale('prediction')

    # Determine globally consistent feature ranking
    all_mob = {k: v[0] for k, v in profile_results.items()}
    top4    = _top_features(all_mob, len(pnames), top_k=4)

    fig, axes = plt.subplots(
        1, n_prof,
        figsize=(5.5 * n_prof, 4.2),
        sharey=False,
    )
    fig.suptitle(
        'Shapley curves — OU kernel ($\\ell=8$ bars) — prediction game\n'
        'Three market-regime profiles  —  Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    profile_titles = {
        'High-VIX Announcement':
            'High-VIX Announcement\n(CPI release + Beige Book, VIX≈27)',
        'Quiet Low-VIX':
            'Quiet Low-VIX\n(non-announcement Monday, VIX≈12)',
        'Monday Gap':
            'Monday Gap\n(positive overnight gap +0.87%, VIX≈22)',
    }

    for p_idx, (ax, (label, (moebius, shapley))) in enumerate(
            zip(axes, profile_results.items())):
        for fi in top4:
            curve = apply_kernel(shapley[fi], K_ou) * sc
            ax.plot(t_grid, curve,
                    color=FEAT_COLORS[pnames[fi]],
                    lw=2.0, label=pnames[fi])
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax)
        _ann_vline(ax)
        _set_time_axis(ax)
        ax.set_title(
            profile_titles.get(label, label),
            fontsize=FS_TITLE, fontweight='bold')
        ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=FS_AXIS)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        # Legend only in the middle panel, lower right
        if p_idx == 1:
            ax.legend(fontsize=FS_LEGEND, loc='lower right')

    plt.tight_layout()
    return fig


# ===========================================================================
# 15. Figure 2b -- Mixed-kernel main effects, pure / partial / full
#     Same layout as Fig 2 but applies a per-feature kernel:
#       ann_indicator -> causal kernel (ell=8)  [no backward leakage]
#       all other features -> OU kernel (ell=8) [temporal smoothing]
#     Rows: prediction / sensitivity / risk
#     Cols: pure / partial / full / integrated importance bars
# ===========================================================================

def _feature_kernel(fi, pnames, t):
    """
    Return the appropriate kernel matrix for feature fi.
    ann_indicator -> causal (one-sided exponential, ell=8)
    all others    -> OU (symmetric exponential, ell=8)
    """
    if pnames[fi] == 'ann_indicator':
        return kernel_causal(t, length_scale=8.0)
    else:
        return kernel_ou(t, length_scale=8.0)


def plot_mixed_kernel_effects(moebius_dict, shapley_dict, pnames, top_k=5):
    """
    Analogue of plot_main_effects_all_games but with per-feature kernels:
      ann_indicator -> causal (ell=8)
      all others    -> OU (ell=8)

    moebius_dict  : dict  game_type -> moebius
    shapley_dict  : dict  game_type -> shapley  (partial effects)
    pnames        : list of feature names
    top_k         : number of features to show in curve panels
    """
    game_types = ['prediction', 'sensitivity', 'risk']
    n_players  = len(pnames)

    # Pre-build per-feature kernel matrices (reused across rows/columns)
    feat_kernels = {fi: _feature_kernel(fi, pnames, t_grid)
                    for fi in range(n_players)}

    fig, axes = plt.subplots(
        len(game_types), 4,
        figsize=(18, 4.0 * len(game_types)),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]},
    )
    fig.suptitle(
        'Main effects — mixed kernel (OU + causal) — pure / partial / full\n'
        'ann_indicator: causal ($\\ell=8$)  |  all others: OU ($\\ell=8$)  '
        '—  High-VIX Announcement profile  —  Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    # Legend location for col-0 curve panel per row
    _leg_loc = {0: 'upper center', 1: 'upper left', 2: 'lower center'}

    for r, gtype in enumerate(game_types):
        sc      = _scale(gtype)
        moebius = moebius_dict[gtype]

        pure_eff    = _pure_effects(moebius, n_players)
        partial_eff = shapley_dict[gtype]
        full_eff    = _full_effects(moebius, n_players)

        effect_dicts = {
            'pure'   : pure_eff,
            'partial': partial_eff,
            'full'   : full_eff,
        }

        # Rank features by kernel-smoothed partial integrated abs importance
        imps_partial = {
            i: float(np.sum(np.abs(apply_kernel(partial_eff[i], feat_kernels[i]))))
            for i in range(n_players)
        }
        top = sorted(imps_partial, key=imps_partial.get, reverse=True)[:top_k]

        # ── curve panels (cols 0-2) ───────────────────────────────────────
        for c, etype in enumerate(_EFFECT_TYPES):
            ax  = axes[r, c]
            eff = effect_dicts[etype]

            for fi in top:
                K     = feat_kernels[fi]
                curve = apply_kernel(eff[fi], K) * sc
                # Mark ann_indicator with a distinct linestyle
                ls = '--' if pnames[fi] == 'ann_indicator' else '-'
                ax.plot(t_grid, curve,
                        color=FEAT_COLORS[pnames[fi]],
                        lw=2.0, ls=ls,
                        label=pnames[fi] if c == 0 else '_')

            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            _ann_vline(ax)
            _set_time_axis(ax)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.set_title(_XAI_LABELS[(gtype, etype)],
                         fontsize=FS_TITLE, fontweight='bold')

            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=FS_AXIS)
                ax.legend(fontsize=FS_LEGEND,
                          loc=_leg_loc[r], framealpha=0.85)

        # ── integrated importance bars (col 3) ────────────────────────────
        # Importance computed after applying per-feature kernel
        ax_bar = axes[r, 3]
        imps_all = {
            etype: {
                i: float(np.sum(np.abs(
                    apply_kernel(effect_dicts[etype][i], feat_kernels[i])
                ))) * sc
                for i in range(n_players)
            }
            for etype in _EFFECT_TYPES
        }
        order = sorted(range(n_players),
                       key=lambda i: imps_all['partial'][i], reverse=True)
        y_pos   = np.arange(len(order))
        bar_h   = 0.25
        offsets = {'pure': -bar_h, 'partial': 0.0, 'full': bar_h}
        alphas  = {'pure': 0.55,   'partial': 0.90, 'full': 0.55}
        hatches = {'pure': '//',   'partial': '',   'full': '\\\\'}

        for etype in _EFFECT_TYPES:
            vals = [imps_all[etype][i] for i in order]
            ax_bar.barh(
                y_pos + offsets[etype], vals,
                height=bar_h,
                color=[FEAT_COLORS[pnames[i]] for i in order],
                alpha=alphas[etype],
                hatch=hatches[etype],
                label=etype,
            )

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
# 16. Figure: Main-body summary -- 2 rows x 4 columns
#
#   Rows
#   -----
#   Row 0  Prediction game  -- identity (faded) overlaid with mixed kernel (vivid)
#   Row 1  Risk game        -- identity (faded) overlaid with mixed kernel (vivid)
#
#   Columns
#   --------
#   Col 0  Pure  m_i(t)        -- PDP / Pure Risk
#   Col 1  Partial  phi_i(t)   -- Shapley (SHAP) / SAGE
#   Col 2  Interaction m_ij(t) -- vix_prev x ann_indicator
#   Col 3  Integrated importance bars (pure + partial, mixed kernel)
#
#   Within cols 0-2:
#     faded thin line  = identity kernel  (baseline)
#     vivid thick line = mixed kernel     (OU for vix_prev, causal for ann_indicator)
#     solid   = vix_prev
#     dashed  = ann_indicator
# ===========================================================================

def plot_main_body_summary(moebius_dict, shapley_dict, pnames):
    """
    Self-contained main-body figure (2 rows x 4 cols).

    moebius_dict : dict  game_type -> moebius
    shapley_dict : dict  game_type -> shapley
    pnames       : list of feature names
    """
    n_players = len(pnames)
    fi_vix    = pnames.index('vix_prev')
    fi_ann    = pnames.index('ann_indicator')

    # ── Kernels ──────────────────────────────────────────────────────────────
    K_id     = kernel_identity(t_grid)
    K_ou     = kernel_ou(t_grid,     length_scale=8.0)
    K_causal = kernel_causal(t_grid, length_scale=8.0)

    def K_mixed(fi):
        """Per-feature mixed kernel: causal for ann_indicator, OU for all others."""
        return K_causal if pnames[fi] == 'ann_indicator' else K_ou

    # ── Pre-compute effects ───────────────────────────────────────────────────
    pure_pred    = _pure_effects(moebius_dict['prediction'], n_players)
    partial_pred = shapley_dict['prediction']
    pure_risk    = _pure_effects(moebius_dict['risk'],       n_players)
    partial_risk = shapley_dict['risk']

    # ── Interaction curves (OU-smoothed raw Mobius term) ─────────────────────
    def _int_curve(game_type, kern):
        raw = moebius_dict[game_type].get(
            (fi_vix, fi_ann), np.zeros(T_BARS))
        return apply_kernel(raw, kern)

    # ── Row spec ──────────────────────────────────────────────────────────────
    # Each entry: (game_type, y_label, col_title_pure, col_title_partial,
    #              pure_eff, partial_eff)
    row_specs = [
        ('prediction',
         GAME_YLABEL['prediction'],
         'Pure  $m_i$  $\\equiv$  PDP',
         'Partial  $\\phi_i$  $\\equiv$  Shapley (SHAP)',
         pure_pred, partial_pred),
        ('risk',
         GAME_YLABEL['risk'],
         'Pure  $m_i$  $\\equiv$  Pure Risk',
         'Partial  $\\phi_i$  $\\equiv$  SAGE',
         pure_risk, partial_risk),
    ]

    row_labels = ['Prediction game', 'Risk game']

    # ── Visual constants ──────────────────────────────────────────────────────
    c_vix   = FEAT_COLORS['vix_prev']
    c_ann   = FEAT_COLORS['ann_indicator']
    FS_TITLE = 10.0   # subplot title fontsize
    FS_AXIS  = 9.0    # axis label fontsize
    FS_TICK  = 8.0    # tick label fontsize
    FS_ANNOT = 8.0    # annotation text fontsize
    FS_LEG   = 8.0    # legend fontsize

    # identity style: faded, thin
    ID_ALPHA = 0.30
    ID_LW    = 1.2
    # mixed-kernel style: vivid, thick
    MX_ALPHA = 1.0
    MX_LW    = 2.2

    fig, axes = plt.subplots(
        2, 4,
        figsize=(17, 7.5),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]},
    )

    fig.suptitle(
        'High-VIX Announcement Profile: '
        'Kernel Choice, Game Type and Effect Decomposition',
        fontsize=12, fontweight='bold',
    )

    for r, (gtype, y_label, lbl_pure, lbl_partial,
            pure_eff, partial_eff) in enumerate(row_specs):

        sc = _scale(gtype)

        # ── Col 0: pure ──────────────────────────────────────────────────────
        ax = axes[r, 0]

        # identity (faded background)
        ax.plot(t_grid,
                apply_kernel(pure_eff[fi_vix], K_id) * sc,
                color=c_vix, lw=ID_LW, ls='-',  alpha=ID_ALPHA)
        ax.plot(t_grid,
                apply_kernel(pure_eff[fi_ann], K_id) * sc,
                color=c_ann, lw=ID_LW, ls='--', alpha=ID_ALPHA)
        # mixed kernel (vivid foreground)
        ax.plot(t_grid,
                apply_kernel(pure_eff[fi_vix], K_mixed(fi_vix)) * sc,
                color=c_vix, lw=MX_LW, ls='-',
                label='vix_prev')
        ax.plot(t_grid,
                apply_kernel(pure_eff[fi_ann], K_mixed(fi_ann)) * sc,
                color=c_ann, lw=MX_LW, ls='--',
                label='ann_indicator')

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax)
        _ann_vline(ax)
        _set_time_axis(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_ylabel(y_label, fontsize=FS_AXIS)
        ax.set_title(lbl_pure, fontsize=FS_TITLE, fontweight='bold')

        # Legend only in top-left panel
        if r == 0:
            # Add identity/mixed entries
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color=c_vix, lw=MX_LW, ls='-',
                       label='vix_prev (mixed)'),
                Line2D([0], [0], color=c_ann, lw=MX_LW, ls='--',
                       label='ann_indicator (mixed)'),
                Line2D([0], [0], color='gray', lw=ID_LW, ls='-',
                       alpha=0.6, label='identity (faded)'),
            ]
            ax.legend(handles=handles, fontsize=FS_LEG,
                      loc='upper center', framealpha=0.9)

        # Row label on far left
        ax.text(-0.32, 0.5, row_labels[r],
                transform=ax.transAxes,
                fontsize=FS_AXIS, va='center', ha='right',
                rotation=90, color='#333', fontweight='bold')

        # ── Col 1: partial ───────────────────────────────────────────────────
        ax = axes[r, 1]

        # identity (faded)
        ax.plot(t_grid,
                apply_kernel(partial_eff[fi_vix], K_id) * sc,
                color=c_vix, lw=ID_LW, ls='-',  alpha=ID_ALPHA)
        ax.plot(t_grid,
                apply_kernel(partial_eff[fi_ann], K_id) * sc,
                color=c_ann, lw=ID_LW, ls='--', alpha=ID_ALPHA)
        # mixed kernel (vivid)
        ax.plot(t_grid,
                apply_kernel(partial_eff[fi_vix], K_mixed(fi_vix)) * sc,
                color=c_vix, lw=MX_LW, ls='-')
        ax.plot(t_grid,
                apply_kernel(partial_eff[fi_ann], K_mixed(fi_ann)) * sc,
                color=c_ann, lw=MX_LW, ls='--')

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax)
        _ann_vline(ax)
        _set_time_axis(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_title(lbl_partial, fontsize=FS_TITLE, fontweight='bold')

        # pure->partial ratio annotation for ann_indicator
        pure_int = float(np.sum(np.abs(
            apply_kernel(pure_eff[fi_ann], K_mixed(fi_ann))))) * sc
        part_int = float(np.sum(np.abs(
            apply_kernel(partial_eff[fi_ann], K_mixed(fi_ann))))) * sc
        ratio = part_int / pure_int if pure_int > 1e-12 else 1.0
        ax.text(0.03, 0.97,
                'ann_indicator: partial/pure\n'
                '= {:.2f}$\\times$ (mixed kernel)'.format(ratio),
                transform=ax.transAxes,
                fontsize=FS_ANNOT - 1, va='top', color=c_ann,
                bbox=dict(boxstyle='round,pad=0.25',
                          fc='white', ec='#ddd', alpha=0.9))

        # ── Col 2: interaction vix_prev x ann_indicator ──────────────────────
        ax = axes[r, 2]

        # identity interaction (faded)
        int_id = _int_curve(gtype, K_id) * sc
        ax.plot(t_grid, int_id, color='#888', lw=ID_LW, alpha=ID_ALPHA)

        # mixed-kernel interaction (vivid, signed fill)
        int_mx = _int_curve(gtype, K_ou) * sc   # OU for the pair
        pos = np.where(int_mx >= 0, int_mx, 0.0)
        neg = np.where(int_mx <  0, int_mx, 0.0)
        ax.fill_between(t_grid, 0, pos, color='#2a9d8f', alpha=0.30)
        ax.fill_between(t_grid, 0, neg, color='#e63946', alpha=0.30)
        ax.plot(t_grid, int_mx, color='#333', lw=MX_LW - 0.4)

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax)
        _ann_vline(ax)
        _set_time_axis(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_title(
            'Interaction  $m_{{ij}}(t)$  —  '
            'vix_prev $\\times$ ann_indicator',
            fontsize=FS_TITLE, fontweight='bold')

        # Integral annotation — upper left
        integ = float(np.trapz(
            moebius_dict[gtype].get(
                (fi_vix, fi_ann), np.zeros(T_BARS)),
            t_grid)) * sc
        ax.text(0.03, 0.97,
                r'$\int m_{{ij}}\,dt$ = {:.3f}'.format(integ),
                transform=ax.transAxes,
                fontsize=FS_ANNOT, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25',
                          fc='white', ec='#aaa', alpha=0.85))

        # ── Col 3: integrated importance bars ────────────────────────────────
        ax = axes[r, 3]

        imps_partial_mx = {
            i: float(np.sum(np.abs(
                apply_kernel(partial_eff[i], K_mixed(i))))) * sc
            for i in range(n_players)
        }
        imps_pure_mx = {
            i: float(np.sum(np.abs(
                apply_kernel(pure_eff[i], K_mixed(i))))) * sc
            for i in range(n_players)
        }
        order = sorted(range(n_players),
                       key=lambda i: imps_partial_mx[i], reverse=True)

        y_pos = np.arange(len(order))
        bar_h = 0.35
        ax.barh(y_pos - bar_h / 2,
                [imps_pure_mx[i]    for i in order],
                height=bar_h,
                color=[FEAT_COLORS[pnames[i]] for i in order],
                alpha=0.45, hatch='//', label='pure')
        ax.barh(y_pos + bar_h / 2,
                [imps_partial_mx[i] for i in order],
                height=bar_h,
                color=[FEAT_COLORS[pnames[i]] for i in order],
                alpha=0.90, label='partial (Shapley)')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([pnames[i] for i in order], fontsize=FS_TICK)
        ax.axvline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_title('Integrated\nimportance\n(mixed kernel)',
                     fontsize=FS_TITLE, fontweight='bold')
        ax.legend(fontsize=FS_LEG, loc='upper right')

    plt.tight_layout(rect=[0.04, 0, 1, 0.95])
    fig.subplots_adjust(top=0.92)   # small gap below suptitle, not too tight
    return fig


# Alias: the previous 3-row version is kept as plot_main_body_summary_v2
# by calling plot_main_body_summary with the old 3-row structure.
# We expose it as a separate function so both can be saved.

def plot_main_body_summary_v2(moebius_dict, shapley_dict, pnames):
    """
    Three-row variant of the main-body summary:
      Row 0  Identity kernel  —  prediction game
      Row 1  Mixed kernel     —  prediction game
      Row 2  Mixed kernel     —  risk game
    Separate rows make per-kernel curves easier to read when the
    identity and mixed-kernel amplitudes differ greatly.
    """
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
         'Pure  $m_i$  $\\equiv$  PDP',
         'Partial  $\\phi_i$  $\\equiv$  Shapley (SHAP)',
         pure_pred, partial_pred, lambda fi: K_id),
        ('prediction', GAME_YLABEL['prediction'],
         'Pure  $m_i$  $\\equiv$  PDP  (mixed kernel)',
         'Partial  $\\phi_i$  $\\equiv$  SHAP  (mixed kernel)',
         pure_pred, partial_pred, K_mixed),
        ('risk', GAME_YLABEL['risk'],
         'Pure  $m_i$  $\\equiv$  Pure Risk  (mixed kernel)',
         'Partial  $\\phi_i$  $\\equiv$  SAGE  (mixed kernel)',
         pure_risk, partial_risk, K_mixed),
    ]
    row_labels   = ['Prediction\n(Identity)', 'Prediction\n(Mixed)', 'Risk\n(Mixed)']
    _leg_loc_col0 = {0: 'upper center', 1: 'upper center', 2: 'lower center'}

    c_vix = FEAT_COLORS['vix_prev']
    c_ann = FEAT_COLORS['ann_indicator']

    fig, axes = plt.subplots(
        3, 4,
        figsize=(17, 11.0),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]},
    )
    fig.suptitle(
        'High-VIX Announcement Profile: '
        'Kernel Choice, Game Type and Effect Decomposition',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    for r, (gtype, y_label, lbl_pure, lbl_partial,
            pure_eff, partial_eff, kern_fn) in enumerate(row_specs):
        sc = _scale(gtype)

        # Col 0: pure
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
        ax.set_title(lbl_pure, fontsize=FS_TITLE, fontweight='bold')
        ax.legend(fontsize=FS_LEGEND, loc=_leg_loc_col0[r], framealpha=0.9)
        ax.text(-0.32, 0.5, row_labels[r], transform=ax.transAxes,
                fontsize=FS_AXIS, va='center', ha='right',
                rotation=90, color='#333', fontweight='bold')

        # Col 1: partial
        ax = axes[r, 1]
        ax.plot(t_grid, apply_kernel(partial_eff[fi_vix], kern_fn(fi_vix)) * sc,
                color=c_vix, lw=2.2, ls='-')
        ax.plot(t_grid, apply_kernel(partial_eff[fi_ann], kern_fn(fi_ann)) * sc,
                color=c_ann, lw=2.2, ls='--')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_title(lbl_partial, fontsize=FS_TITLE, fontweight='bold')
        pure_int = float(np.sum(np.abs(
            apply_kernel(pure_eff[fi_ann], kern_fn(fi_ann))))) * sc
        part_int = float(np.sum(np.abs(
            apply_kernel(partial_eff[fi_ann], kern_fn(fi_ann))))) * sc
        ratio = part_int / pure_int if pure_int > 1e-12 else 1.0
        ax.text(0.03, 0.97,
                'ann_indicator: partial/pure\n= {:.2f}$\\times$'.format(ratio),
                transform=ax.transAxes, fontsize=FS_ANNOT - 1, va='top',
                color=c_ann,
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#ddd', alpha=0.9))
        # Row 1 col 1: no legend (shared with col0 legend)

        # Col 2: interaction
        ax = axes[r, 2]
        kern_for_pair = K_id if r == 0 else K_ou
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
        ax.set_title('Interaction  $m_{{ij}}(t)$  —  '
                     'vix_prev $\\times$ ann_indicator',
                     fontsize=FS_TITLE, fontweight='bold')
        integ = float(np.trapz(
            moebius_dict[gtype].get((fi_vix, fi_ann), np.zeros(T_BARS)),
            t_grid)) * sc
        ax.text(0.03, 0.97,
                r'$\int m_{{ij}}\,dt$ = {:.3f}'.format(integ),
                transform=ax.transAxes,
                fontsize=FS_ANNOT, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25',
                          fc='white', ec='#aaa', alpha=0.85))

        # Col 3: importance bars
        ax = axes[r, 3]
        imps_part = {i: float(np.sum(np.abs(
            apply_kernel(partial_eff[i], kern_fn(i))))) * sc
            for i in range(n_players)}
        imps_pure = {i: float(np.sum(np.abs(
            apply_kernel(pure_eff[i], kern_fn(i))))) * sc
            for i in range(n_players)}
        order = sorted(range(n_players),
                       key=lambda i: imps_part[i], reverse=True)
        y_pos = np.arange(len(order)); bar_h = 0.35
        ax.barh(y_pos - bar_h / 2,
                [imps_pure[i] for i in order], height=bar_h,
                color=[FEAT_COLORS[pnames[i]] for i in order],
                alpha=0.45, hatch='//', label='pure')
        ax.barh(y_pos + bar_h / 2,
                [imps_part[i] for i in order], height=bar_h,
                color=[FEAT_COLORS[pnames[i]] for i in order],
                alpha=0.90, label='partial')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([pnames[i] for i in order], fontsize=FS_TICK)
        ax.axvline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_title('Integrated\nimportance', fontsize=FS_TITLE,
                     fontweight='bold')
        ax.legend(fontsize=FS_LEGEND, loc='upper right')

    # ── Enforce shared y-scale across cols 0-2 within each row ──────────
    # Col 3 (importance bars) keeps its own scale.
    for r in range(3):
        ymin = min(axes[r, c].get_ylim()[0] for c in range(3))
        ymax = max(axes[r, c].get_ylim()[1] for c in range(3))
        for c in range(3):
            axes[r, c].set_ylim(ymin, ymax)

    plt.tight_layout(rect=[0.04, 0, 1, 0.95])
    fig.subplots_adjust(top=0.92)
    return fig


# ===========================================================================
# 17. Network appendix figures for SPY
#
#   Two figures:
#     figA_network_prediction_risk.pdf  -- prediction + risk, partial (Shapley),
#                                          OU kernel, side by side
#     figA_network_all_games_ppf.pdf    -- all 3 games x pure/partial/full,
#                                          OU kernel (3x3 grid)
#
#   Node:  teal = positive integrated effect, red = negative
#          inner white circle with 3-letter abbreviation
#   Edge:  teal = synergy (+), red = redundancy (-)
# ===========================================================================

# SPY feature abbreviations for network labels
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
    """Compute node importances and signed edge weights for one network panel."""
    pure_eff = _pure_effects(moebius, n_players)
    full_eff = _full_effects(moebius, n_players)
    if effect_type == 'pure':
        eff = pure_eff
    elif effect_type == 'partial':
        eff = shapley
    else:
        eff = full_eff

    node_imp  = np.array([
        float(np.sum(np.abs(apply_kernel(eff[i], K)))) for i in range(n_players)])
    node_sign = np.array([
        np.sign(float(np.trapz(apply_kernel(eff[i], K), t_grid)))
        for i in range(n_players)])
    edge_imp = {}
    for i in range(n_players):
        for j in range(i + 1, n_players):
            raw = moebius.get((i, j), np.zeros(T_BARS))
            val = float(np.trapz(apply_kernel(raw, K), t_grid))
            if abs(val) > 0:
                edge_imp[(i, j)] = val
    return node_imp, edge_imp, node_sign


def _draw_network_spy(ax, pnames, node_imp, edge_imp, node_sign, title):
    """Draw a single network panel with tighter node layout."""
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
        x, y = pos[i]
        r    = node_r[i]
        fc   = _NODE_POS_SPY if node_sign[i] >= 0 else _NODE_NEG_SPY
        circle = plt.Circle((x, y), r, color=fc, ec='white',
                             linewidth=1.2, zorder=2, alpha=0.88)
        ax.add_patch(circle)
        inner = plt.Circle((x, y), r * 0.52, color='white', ec='none',
                            zorder=3, alpha=0.95)
        ax.add_patch(inner)
        abbr = _SPY_ABBR.get(pnames[i], pnames[i][:3])
        ax.text(x, y, abbr, ha='center', va='center',
                fontsize=max(4.5, r * 22),
                fontweight='bold', color='#222', zorder=4)

    # tighter limits: nodes sit on unit circle, largest node_r ≈ 0.26
    pad = 0.32
    ax.set_xlim(-1.0 - pad, 1.0 + pad)
    ax.set_ylim(-1.0 - pad, 1.0 + pad)


def plot_network_prediction_risk(moebius_dict, shapley_dict, pnames):
    """
    Appendix figure A1: prediction + risk networks, partial (Shapley), OU kernel.
    """
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec

    K_ou      = kernel_ou(t_grid, 8.0)
    n_players = len(pnames)

    fig = plt.figure(figsize=(8, 4.2))
    fig.suptitle(
        'Network plots — Partial (Shapley) — OU kernel ($\\ell=8$ bars)\n'
        'High-VIX Announcement profile  —  SPY Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold',
        y=1.01,
    )

    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        wspace=0.05,
        left=0.02, right=0.98,
        top=0.88, bottom=0.10,
    )

    game_titles = {
        'prediction': 'Prediction  $\\phi_i \\equiv$ SHAP',
        'risk':       'Risk (MSE)  $\\phi_i \\equiv$ SAGE',
    }
    for col, gtype in enumerate(['prediction', 'risk']):
        ax = fig.add_subplot(gs[0, col])
        node_imp, edge_imp, node_sign = _network_importances_spy(
            moebius_dict[gtype], shapley_dict[gtype],
            n_players, K_ou, 'partial')
        _draw_network_spy(ax, pnames, node_imp, edge_imp, node_sign,
                          game_titles[gtype])

    leg_handles = [
        Patch(facecolor=_NODE_POS_SPY, edgecolor='none', label='Positive effect'),
        Patch(facecolor=_NODE_NEG_SPY, edgecolor='none', label='Negative effect'),
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


def plot_network_all_games_ppf(moebius_dict, shapley_dict, pnames):
    """
    Appendix figure A2: all 3 games x pure/partial/full, OU kernel (3x3 grid).
    """
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec

    K_ou      = kernel_ou(t_grid, 8.0)
    n_players = len(pnames)

    game_specs = [
        ('prediction',  'Prediction',  'PDP',          'SHAP',         'ICE-agg.'),
        ('sensitivity', 'Sensitivity', 'Closed Sobol', 'Shapley-sens', 'Total Sobol'),
        ('risk',        'Risk (MSE)',  'Pure Risk',    'SAGE',         'PFI'),
    ]

    fig = plt.figure(figsize=(9, 9.2))
    fig.suptitle(
        'Network plots — OU kernel ($\\ell=8$ bars) — all games\n'
        'High-VIX Announcement profile  —  SPY Random Forest',
        fontsize=FS_SUPTITLE, fontweight='bold',
        y=1.01,
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.08,
        wspace=0.08,
        left=0.08, right=0.98,
        top=0.91, bottom=0.06,
    )

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
            _draw_network_spy(
                ax, pnames, node_imp, edge_imp, node_sign,
                col_titles[c] if r == 0 else '',
            )
            if c == 0:
                ax.text(
                    -0.03, 0.5, glabel,
                    transform=ax.transAxes,
                    fontsize=FS_AXIS, va='center', ha='right',
                    rotation=90, color='#333', fontweight='bold',
                )

    leg_handles = [
        Patch(facecolor=_NODE_POS_SPY, edgecolor='none', label='Positive effect'),
        Patch(facecolor=_NODE_NEG_SPY, edgecolor='none', label='Negative effect'),
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


if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  SPY Intraday Vol  —  RF direct  (paper figures)')
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
        X_day_np, Y_adj,
        test_size=0.2, random_state=RNG_SEED)
    model = RFModel()
    model.fit(X_tr, Y_tr)
    r2 = model.evaluate(X_te, Y_te)
    print('  Test R² (trajectory-level): {:.4f}'.format(r2))

    # ── 3. Kernels ────────────────────────────────────────────────────────
    print('\n[3] Building kernels ...')
    K_corr = kernel_output_correlation(Y_day)
    kernels = {
        'Identity'   : kernel_identity(t_grid),
        'OU'         : kernel_ou(t_grid, length_scale=8.0),
        'Correlation': K_corr,
        'Causal'     : kernel_causal(t_grid, length_scale=8.0),
    }

    # ── 4. Profiles ───────────────────────────────────────────────────────
    print('\n[4] Selecting profiles ...')
    vix_col = DAY_FEATURE_NAMES.index('vix_prev')
    vix_p25 = float(np.percentile(X_day_np[:, vix_col], 25))
    vix_p75 = float(np.percentile(X_day_np[:, vix_col], 75))

    def find_profile(conditions):
        mask = np.ones(len(X_day_np), dtype=bool)
        for feat, (lo, hi) in conditions.items():
            ci = DAY_FEATURE_NAMES.index(feat)
            mask &= (X_day_np[:, ci] >= lo) & (X_day_np[:, ci] <= hi)
        hits = X_day_np[mask]
        if len(hits) == 0:
            raise RuntimeError(
                'No day matches: {}'.format(conditions))
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

    for lbl, xp, _ in profiles:
        print('  {}: {}'.format(lbl, '  '.join(
            '{}={:.3f}'.format(n, xp[j])
            for j, n in enumerate(DAY_FEATURE_NAMES))))

    # ── 5. Games  (High-VIX profile, all 3 games) ─────────────────────────
    print('\n[5] Computing games for High-VIX Announcement profile ...')
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

    # ── 6. Games  (other profiles, prediction only) ───────────────────────
    print('\n[6] Computing prediction game for remaining profiles ...')
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

    # ── 7. Generate figures ───────────────────────────────────────────────
    print('\n[7] Generating figures ...')

    # Fig 0: main-body summary v1 (2-row: identity+mixed overlaid)
    savefig(
        plot_main_body_summary(moebius_hv, shapley_hv, pnames),
        'fig0_main_body_summary_v1.pdf',
    )

    # Fig 0v2: main-body summary v2 (3-row: identity / mixed-pred / mixed-risk)
    savefig(
        plot_main_body_summary_v2(moebius_hv, shapley_hv, pnames),
        'fig0_main_body_summary_v2.pdf',
    )

    # Fig 1: operator sweep
    savefig(
        plot_operator_sweep(
            moebius_hv, shapley_hv, kernels, pnames),
        'fig1_operator_sweep.pdf',
    )

    # Fig 2: main effects, pure/partial/full, all games, identity kernel
    savefig(
        plot_main_effects_all_games(
            moebius_hv, shapley_hv, pnames, top_k=5),
        'fig2_main_effects_all_games.pdf',
    )

    # Fig 2b: main effects, pure/partial/full, all games, mixed kernel
    savefig(
        plot_mixed_kernel_effects(
            moebius_hv, shapley_hv, pnames, top_k=5),
        'fig2b_mixed_kernel_effects.pdf',
    )

    # Fig 3: ann_indicator causal vs symmetric kernel
    savefig(
        plot_ann_causal_comparison(moebius_hv, pnames),
        'fig3_ann_causal_comparison.pdf',
    )

    # Fig 3b: sensitivity gap (standalone)
    savefig(
        plot_sensitivity_gap(
            moebius_hv['sensitivity'], pnames, top_k=4),
        'fig3b_sensitivity_gap.pdf',
    )

    # Fig 4: pairwise interactions, prediction + risk game, OU kernel
    savefig(
        plot_interactions(moebius_hv, pnames),
        'fig4_interactions.pdf',
    )

    # Fig 5: profile comparison, prediction game, OU kernel
    savefig(
        plot_profiles_comparison(profile_results, pnames),
        'fig5_profiles_comparison.pdf',
    )

    # Fig A1: network -- prediction + risk, partial, OU kernel
    savefig(
        plot_network_prediction_risk(moebius_hv, shapley_hv, pnames),
        'figA_network_prediction_risk.pdf',
    )

    # Fig A2: network -- all games x pure/partial/full, OU kernel
    savefig(
        plot_network_all_games_ppf(moebius_hv, shapley_hv, pnames),
        'figA_network_all_games_ppf.pdf',
    )

    print('\n' + '=' * 60)
    print('  Done.  Figures saved under {}/'.format(BASE_PLOT_DIR))
    print('  fig0_main_body_summary_v1.pdf        -- MAIN BODY option A: 2-row overlaid')
    print('  fig0_main_body_summary_v2.pdf        -- MAIN BODY option B: 3-row separate (shared y per row)')
    print('  fig1_operator_sweep.pdf              -- operator sweep')
    print('  fig2_main_effects_all_games.pdf      -- pure/partial/full, identity kernel')
    print('  fig2b_mixed_kernel_effects.pdf       -- pure/partial/full, mixed kernel')
    print('  fig3_ann_causal_comparison.pdf       -- causal vs symmetric kernel')
    print('  fig3b_sensitivity_gap.pdf            -- Total - Closed Sobol gap')
    print('  fig4_interactions.pdf                -- prediction + risk, top-3 pairs')
    print('  fig5_profiles_comparison.pdf         -- three market profiles')
    print('  figA_network_prediction_risk.pdf     -- APPENDIX: network pred+risk')
    print('  figA_network_all_games_ppf.pdf       -- APPENDIX: network all games ppf')
    print('=' * 60)