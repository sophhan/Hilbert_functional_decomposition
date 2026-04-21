"""
Functional Explanation Framework -- Intraday SPY Volatility (Random Forest)
============================================================================
Model:  F^H : R^6 -> R^78
  RandomForestRegressor (multi-output, direct, no PCA)
  t is NEVER an input feature.

Plots generated (focused on paper storyline):
  1. operator_sweep.pdf          -- 3x4 grid: games x kernels, vix_prev + ann_indicator
                                    This is the central paper figure.
  2. main_effects_highvix.pdf    -- Top-5 main effects, identity kernel, all 3 games
  3. ann_causal_comparison.pdf   -- ann_indicator under identity/OU/causal kernels,
                                    prediction + risk game side by side
  4. interaction_highvix.pdf     -- Top-3 pairwise Mobius terms, OU kernel, prediction game
  5. profiles_comparison.pdf     -- Shapley curves (OU kernel), all 3 profiles,
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
    'sensitivity': 300,
    'risk'       : 300,
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
    'plots', 'SPY_intraday_rf')

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
# 1.  Data loading  (identical logic to original script)
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
# 7.  Plotting helpers
# ===========================================================================

XTICK_IDXS   = list(range(0, T_BARS, 6))
XTICK_LABELS = [BAR_LABELS[i] for i in XTICK_IDXS]

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

def _scale(game_type):
    return 100.0 if game_type == 'prediction' else 1e4

def _set_time_axis(ax, sparse=False):
    step = 2 if sparse else 1
    ax.set_xticks(XTICK_IDXS[::step])
    ax.set_xticklabels(
        XTICK_LABELS[::step],
        rotation=45, ha='right', fontsize=6)
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
# 8.  Figure 1 -- Operator sweep  (central paper figure)
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
    labels = ['vix\\_prev', 'ann\\_indicator']

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 3.2 * nrows),
        sharey='row',
    )
    fig.suptitle(
        'Operator sweep: Shapley curves across games and kernels\n'
        'High-VIX Announcement profile  —  Random Forest',
        fontsize=12, fontweight='bold',
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
                    fontsize=9, fontweight='bold')

            # Row labels (left column only)
            if c == 0:
                ax.set_ylabel(
                    GAME_YLABEL[gtype], fontsize=8)
                # Game label as text on left margin
                ax.text(
                    -0.38, 0.5,
                    GAME_LABELS[gtype],
                    transform=ax.transAxes,
                    fontsize=7.5, va='center',
                    ha='right', rotation=90,
                    color='#333',
                )

            ax.set_xlabel('Time', fontsize=7)
            ax.tick_params(axis='y', labelsize=7)

            # Annotate the identity+prediction cell
            if r == 0 and c == 0:
                ax.text(
                    0.97, 0.97,
                    '= pointwise\nSHAP',
                    transform=ax.transAxes,
                    fontsize=6.5, va='top', ha='right',
                    color='#555',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', ec='#aaa', alpha=0.8),
                )

    # Single legend top-right
    handles = [
        plt.Line2D([0], [0], color=FEAT_COLORS['vix_prev'],
                   lw=2, label='vix_prev'),
        plt.Line2D([0], [0], color=FEAT_COLORS['ann_indicator'],
                   lw=2, label='ann_indicator'),
        plt.Line2D([0], [0], color='#555', lw=0.9, ls='--',
                   alpha=0.6, label='14:00 ET'),
    ]
    fig.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    return fig


# ===========================================================================
# 9.  Figure 2 -- Main effects, all games, identity kernel
#     3 rows (games) x 1 col of curves + 1 col of importance bars
# ===========================================================================

def plot_main_effects_all_games(moebius_dict, pnames, top_k=5):
    game_types = ['prediction', 'sensitivity', 'risk']
    K_id       = kernel_identity(t_grid)

    fig, axes = plt.subplots(
        len(game_types), 2,
        figsize=(12, 3.8 * len(game_types)),
        gridspec_kw={'width_ratios': [3, 1.5]},
    )
    fig.suptitle(
        'Main effects  $m_i(t)$  —  Identity kernel\n'
        'High-VIX Announcement profile  —  Random Forest',
        fontsize=11, fontweight='bold',
    )

    for r, gtype in enumerate(game_types):
        sc      = _scale(gtype)
        moebius = moebius_dict[gtype]
        pnames_ = list(pnames)

        imps = {i: float(np.sum(np.abs(moebius[(i,)])))
                for i in range(len(pnames_))}
        top  = sorted(imps, key=imps.get, reverse=True)[:top_k]

        # Left: curves
        ax = axes[r, 0]
        for fi in top:
            ax.plot(t_grid, moebius[(fi,)] * sc,
                    color=FEAT_COLORS[pnames_[fi]],
                    lw=2.0, label=pnames_[fi])
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax)
        _ann_vline(ax)
        _set_time_axis(ax)
        ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
        ax.set_xlabel('Time', fontsize=8)
        ax.set_title(GAME_LABELS[gtype], fontsize=9)
        if r == 0:
            ax.legend(fontsize=7, loc='upper right')

        # Right: importance bars
        ax2 = axes[r, 1]
        order = sorted(range(len(pnames_)),
                       key=lambda i: imps[i], reverse=True)
        vals  = [imps[i] * sc for i in order]
        names = [pnames_[i] for i in order]
        cols  = [FEAT_COLORS[n] for n in names]
        ax2.barh(range(len(names)), vals, color=cols, alpha=0.85)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=7)
        ax2.axvline(0, color='gray', lw=0.8, ls=':')
        ax2.set_xlabel(r'$\int|m_i(t)|\,dt$', fontsize=8)
        ax2.set_title('Integrated importance', fontsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


# ===========================================================================
# 10.  Figure 3 -- ann_indicator causal kernel comparison
#      Rows: prediction game / risk game
#      Cols: Identity / OU / Causal (ls=4) / Causal (ls=8) / Causal (ls=16)
#      Shows how symmetric kernels produce pre-announcement leakage
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
        'ann\\_indicator — symmetric vs causal kernel\n'
        'High-VIX Announcement profile  —  Random Forest',
        fontsize=11, fontweight='bold',
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
                ci  = c - 2   # 0,1,2 for the three causal
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

            # Annotate pre-14:00 leakage on OU panel
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
# 11.  Figure 4 -- Top pairwise interaction effects
#      Prediction game, OU kernel, top-3 pairs
# ===========================================================================

def plot_interactions(moebius, pnames, game_type='prediction'):
    sc       = _scale(game_type)
    K_ou     = kernel_ou(t_grid, 8.0)
    n        = len(pnames)
    pair_imp = {
        (i, j): float(np.sum(np.abs(moebius.get((i, j), np.zeros(T_BARS)))))
        for i in range(n) for j in range(i + 1, n)
    }
    top3 = sorted(pair_imp, key=pair_imp.get, reverse=True)[:3]

    fig, axes = plt.subplots(
        1, 3, figsize=(13, 3.8), sharey=False)
    fig.suptitle(
        'Top-3 pairwise interaction effects  $m_{ij}(t)$\n'
        'OU kernel ($\\ell=8$ bars)  —  prediction game  —  High-VIX day',
        fontsize=11, fontweight='bold',
    )

    pair_colors = ['#e63946', '#2a9d8f', '#8338ec']

    for ax, (i, j), col in zip(axes, top3, pair_colors):
        raw   = moebius.get((i, j), np.zeros(T_BARS))
        curve = apply_kernel(raw, K_ou) * sc
        ax.plot(t_grid, curve, color=col, lw=2.2)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax)
        _ann_vline(ax)
        _set_time_axis(ax)
        ax.set_title(
            '{} × {}'.format(pnames[i], pnames[j]),
            fontsize=9, fontweight='bold')
        ax.set_ylabel(GAME_YLABEL[game_type], fontsize=8)
        ax.set_xlabel('Time', fontsize=8)
        ax.tick_params(labelsize=7)
        # Integrated value annotation
        integ = float(np.trapz(raw, t_grid)) * sc
        ax.text(
            0.97, 0.97,
            r'$\int m_{{ij}}\,dt$ = {:.3f}'.format(integ),
            transform=ax.transAxes,
            fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.2',
                      fc='white', ec='#aaa', alpha=0.8),
        )

    plt.tight_layout()
    return fig


# ===========================================================================
# 12.  Figure 5 -- Profile comparison
#      Prediction game, OU kernel, all 3 profiles, top-4 features
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
        fontsize=11, fontweight='bold',
    )

    profile_titles = {
        'High-VIX Announcement':
            'High-VIX Announcement\n(CPI release + Beige Book, VIX≈27)',
        'Quiet Low-VIX':
            'Quiet Low-VIX\n(non-announcement Monday, VIX≈12)',
        'Monday Gap':
            'Monday Gap\n(positive overnight gap +0.87%, VIX≈22)',
    }

    for ax, (label, (moebius, shapley)) in zip(axes, profile_results.items()):
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
            fontsize=9, fontweight='bold')
        ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=8)
        ax.set_xlabel('Time', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


# ===========================================================================
# 13.  Main
# ===========================================================================

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

    # Fig 1: operator sweep (central paper figure)
    savefig(
        plot_operator_sweep(
            moebius_hv, shapley_hv, kernels, pnames),
        'fig1_operator_sweep.pdf',
    )

    # Fig 2: main effects, all games, identity kernel
    savefig(
        plot_main_effects_all_games(moebius_hv, pnames, top_k=5),
        'fig2_main_effects_all_games.pdf',
    )

    # Fig 3: ann_indicator causal vs symmetric kernel
    savefig(
        plot_ann_causal_comparison(moebius_hv, pnames),
        'fig3_ann_causal_comparison.pdf',
    )

    # Fig 4: pairwise interactions, prediction game, OU kernel
    savefig(
        plot_interactions(
            moebius_hv['prediction'], pnames,
            game_type='prediction'),
        'fig4_interactions_prediction.pdf',
    )

    # Fig 5: profile comparison, prediction game, OU kernel
    savefig(
        plot_profiles_comparison(profile_results, pnames),
        'fig5_profiles_comparison.pdf',
    )

    print('\n' + '=' * 60)
    print('  Done.  Figures saved under {}/'.format(BASE_PLOT_DIR))
    print('  fig1_operator_sweep.pdf       -- central paper figure')
    print('  fig2_main_effects_all_games.pdf')
    print('  fig3_ann_causal_comparison.pdf')
    print('  fig4_interactions_prediction.pdf')
    print('  fig5_profiles_comparison.pdf')
    print('=' * 60)