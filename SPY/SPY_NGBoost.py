"""
Functional Explanation Framework -- Intraday SPY Volatility (NGBoost version)
==============================================================================
Replaces the long-format GBT with a genuinely multivariate NGBoost model:

  Model:  F^H : R^6 -> R^78
    Step 1 -- PCA: Y_adj (n_days, 78) -> scores (n_days, K)
    Step 2 -- NGBoost: X_day (n_days, 6) -> MultivariateNormal(K) over scores
    Predict: E[Y | X] = pca.inverse_transform( ngb.predict(X) )

  t is NEVER an input feature.  The model is a genuine map X -> R^T.
  This directly instantiates Section 4.5 (PCA representation) of the paper.

  Sensitivity game uses the NGBoost predictive covariance:
    Var[F^H(X)(t) | X_S] propagated from score-space covariance via PCA.

Data source : Polygon.io Starter subscription -- SPY 5-minute bars, 2 years.
VIX         : yfinance (daily, free).

Games     : prediction, sensitivity, risk
Kernels   : Identity, Gaussian, OU, Correlation, Causal (ann_indicator only)
Profiles  : High-VIX Announcement, Quiet Low-VIX, Monday Gap
"""

import itertools
import os
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal

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

RNG_SEED   = 42
T_BARS     = 78
dt         = 1.0

# NGBoost / PCA hyperparameters
N_PCA        = 8      # PCA components; K=8 captures the bulk of structured variation
N_ESTIMATORS = 300    # NGBoost trees
LEARNING_RATE = 0.05

# Sample sizes for game evaluation
SAMPLE_SIZE = {
    'prediction' : 200,
    'sensitivity': 400,
    'risk'       : 400,
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

BASE_PLOT_DIR = os.path.join('plots', 'intraday_ngboost')
PLOT_DIRS = {
    'prediction'  : os.path.join(BASE_PLOT_DIR, 'prediction'),
    'sensitivity' : os.path.join(BASE_PLOT_DIR, 'sensitivity'),
    'risk'        : os.path.join(BASE_PLOT_DIR, 'risk'),
}

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

DAY_FEATURE_NAMES = [
    'vix_prev',
    'overnight_ret',
    'ann_indicator',
    'day_of_week',
    'trailing_rv',
    'month',
]

INTRADAY_PERIODS = {
    'open'     : (0,   6),
    'morning'  : (6,  24),
    'midday'   : (24, 54),
    'afternoon': (54, 72),
    'close'    : (72, 78),
}


# ===========================================================================
# 1.  Validation helpers
# ===========================================================================

def _require_dir(path):
    os.makedirs(path, exist_ok=True)


def _validate_vix_dict(vix_dict, trading_dates):
    missing = []
    for date_str in trading_dates:
        dt_obj    = pd.Timestamp(date_str)
        prev_days = [
            str((dt_obj - pd.Timedelta(d, 'D')).date())
            for d in range(1, 15)
        ]
        if not any(d in vix_dict for d in prev_days):
            missing.append(date_str)
    if missing:
        threshold = max(1, int(0.01 * len(trading_dates)))
        if len(missing) > threshold:
            raise RuntimeError(
                'VIX prior-day data missing for {}/{} dates. '
                'First 10: {}'.format(len(missing), len(trading_dates), missing[:10])
            )


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
            'Only {} complete trading days after filtering.'.format(len(pivot))
        )


def _validate_feature_matrix(X_day):
    for col in DAY_FEATURE_NAMES:
        if col not in X_day.columns:
            raise RuntimeError('Missing feature column: {}'.format(col))
        if X_day[col].nunique() <= 1 or X_day[col].std() < 1e-8:
            raise RuntimeError('Feature {} is constant.'.format(col))


# ===========================================================================
# 2.  VIX loading
# ===========================================================================

def _vix_fetch_start(first_trading_date):
    return (
        pd.Timestamp(first_trading_date) - pd.Timedelta(days=VIX_LOOKBACK_DAYS)
    ).strftime('%Y-%m-%d')


def _resolve_close_column(df, source_label):
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
    raise RuntimeError('{}: no recognisable Close column.'.format(source_label))


def _fetch_vix_yfinance(start, end):
    import yfinance as yf
    end_ex = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    print('    Downloading ^VIX  {} -> {} ...'.format(start, end))
    raw = yf.download('^VIX', start=start, end=end_ex, progress=False,
                      auto_adjust=True)
    close = _resolve_close_column(raw, 'yfinance ^VIX')
    result = {
        (str(idx.date()) if hasattr(idx, 'date') else str(idx)): float(val)
        for idx, val in close.items()
    }
    print('    VIX loaded: {} days  [{:.1f} - {:.1f}]'.format(
        len(result), min(result.values()), max(result.values())))
    return result


def load_vix(first_trading_date, last_trading_date):
    vix_start = _vix_fetch_start(first_trading_date)
    if os.path.isfile(VIX_CACHE_PATH):
        print('    Reading VIX cache ...')
        df_cache = pd.read_csv(VIX_CACHE_PATH, index_col=0)
        result = {str(k): float(v) for k, v in df_cache['vix'].dropna().items()}
        print('    VIX cache OK: {} days'.format(len(result)))
        return result
    result = _fetch_vix_yfinance(vix_start, last_trading_date)
    pd.DataFrame({'vix': result}).to_csv(VIX_CACHE_PATH)
    return result


# ===========================================================================
# 3.  SPY bar loading
# ===========================================================================

def _fetch_polygon_5min(api_key):
    import requests
    base, chunks, windows = 'https://api.polygon.io/v2/aggs/ticker', [], []
    t, end = pd.Timestamp(FETCH_START), pd.Timestamp(FETCH_END)
    while t < end:
        t_end = min(t + pd.DateOffset(months=6), end)
        windows.append((t.strftime('%Y-%m-%d'), t_end.strftime('%Y-%m-%d')))
        t = t_end
    for i, (w_start, w_end) in enumerate(windows):
        print('    [{}/{}]  {} -> {}'.format(i+1, len(windows), w_start, w_end))
        url    = '{}/{}/range/5/minute/{}/{}'.format(base, TICKER, w_start, w_end)
        params = {'adjusted': 'true', 'sort': 'asc', 'limit': 50000,
                  'apiKey': api_key}
        while url:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            chunks.extend(data.get('results', []))
            url    = data.get('next_url')
            params = {'apiKey': api_key}
        if i < len(windows) - 1:
            time.sleep(1)
    df = pd.DataFrame(chunks)
    df['timestamp'] = (pd.to_datetime(df['t'], unit='ms', utc=True)
                       .dt.tz_convert('America/New_York'))
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    return df[['timestamp','open','high','low','close','volume']].sort_values('timestamp')


def load_bars():
    if os.path.isfile(CACHE_PATH):
        print('  Loading bar cache ...')
        bars = pd.read_csv(CACHE_PATH)
        bars['timestamp'] = (pd.to_datetime(bars['timestamp'], utc=True)
                             .dt.tz_convert('America/New_York'))
        _validate_bars(bars)
        print('  Loaded {:,} rows.'.format(len(bars)))
        return bars
    if not POLYGON_API_KEY:
        raise RuntimeError(
            'No bar cache found and POLYGON_API_KEY is not set.\n'
            'Place a CSV at: {}'.format(CACHE_PATH)
        )
    bars = _fetch_polygon_5min(POLYGON_API_KEY)
    _validate_bars(bars)
    bars.to_csv(CACHE_PATH, index=False)
    return bars


# ===========================================================================
# 4.  Feature construction
# ===========================================================================

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
    print('  Processing {:,} bars ...'.format(len(bars)))
    bars = bars.copy()
    bars['date']    = bars['timestamp'].dt.date.astype(str)
    bars['bar_idx'] = (
        (bars['timestamp'].dt.hour * 60 + bars['timestamp'].dt.minute - 570) // 5
    ).astype(int)
    bars = bars[(bars['bar_idx'] >= 0) & (bars['bar_idx'] < T_BARS)].copy()
    bars['open']  = pd.to_numeric(bars['open'],  errors='coerce')
    bars['close'] = pd.to_numeric(bars['close'], errors='coerce')
    bars = bars[(bars['open'] > 0) & (bars['close'] > 0)].copy()
    bars['abs_log_ret'] = np.abs(np.log(bars['close'] / bars['open']))

    pivot = bars.pivot_table(
        index='date', columns='bar_idx', values='abs_log_ret', aggfunc='mean'
    )
    pivot = pivot.reindex(columns=range(T_BARS))
    pivot = pivot[pivot.notna().sum(axis=1) >= 70].fillna(pivot.mean())
    _validate_pivot(pivot)

    Y_day        = pivot.values.astype(float)
    dates        = pivot.index.tolist()
    diurnal_mean = Y_day.mean(axis=0)
    Y_adj        = Y_day - diurnal_mean[None, :]

    print('  Complete trading days: {}'.format(len(dates)))
    vix_dict = load_vix(dates[0], dates[-1])
    _validate_vix_dict(vix_dict, dates)

    records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str)
        vix_prev = _resolve_vix_prev(vix_dict, date_str)

        day_bars  = bars[bars['date'] == date_str].sort_values('bar_idx')
        prev_bars = bars[bars['date'] < date_str].sort_values(['date', 'bar_idx'])

        this_open = float(day_bars.iloc[0]['open'])
        if len(prev_bars) == 0:
            overnight_ret = 0.0
        else:
            prev_close    = float(prev_bars.iloc[-1]['close'])
            overnight_ret = float(np.log(this_open / prev_close))

        recent      = list(range(max(0, i - 5), i))
        trailing_rv = float(Y_day[recent].mean()) if recent else float(Y_day.mean())

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
        int(X_day['ann_indicator'].sum()), X_day['ann_indicator'].mean() * 100))
    return X_day, Y_day, Y_adj, diurnal_mean


# ===========================================================================
# 5.  NGBoost + PCA model
#     Genuine map F^H: R^6 -> R^78  (no t as input)
# ===========================================================================

class NGBoostFunctionalModel:
    """
    Genuinely multivariate probabilistic model for functional outputs.

    Architecture
    ------------
    1. PCA: Y_adj (n, T) -> scores (n, K)   [captures temporal structure]
    2. NGBoost: X (n, p) -> MvNormal(K)     [models score distribution]
    3. Predict: E[Y|X] = pca.inverse_transform( ngb.predict(X) )

    Sensitivity output
    ------------------
    NGBoost gives a per-sample covariance Sigma_i (K, K) over scores.
    Trajectory variance is propagated as:
      Var[F(t) | X=x_i] = V[:,t]^T Sigma_i V[:,t]
    where V = pca.components_ (K, T).

    This is dimensionally exact and connects to Section 4.5 of the paper.
    """

    def __init__(self, n_pca=N_PCA, n_estimators=N_ESTIMATORS,
                 learning_rate=LEARNING_RATE, random_state=RNG_SEED):
        self.n_pca         = n_pca
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.pca           = None
        self.ngb           = None
        self._var_explained = None

    def fit(self, X, Y):
        """X: (n, p),  Y: (n, T) diurnal-adjusted trajectories."""
        self.pca    = PCA(n_components=self.n_pca)
        scores      = self.pca.fit_transform(Y)
        self._var_explained = float(self.pca.explained_variance_ratio_.sum())
        print('  PCA: K={} components, {:.1f}% variance explained'.format(
            self.n_pca, self._var_explained * 100))

        self.ngb = NGBRegressor(
            Dist          = MultivariateNormal(self.n_pca),
            n_estimators  = self.n_estimators,
            learning_rate = self.learning_rate,
            random_state  = self.random_state,
            verbose       = False,
        )
        self.ngb.fit(X, scores)
        return self

    def predict_mean(self, X):
        """Point prediction E[Y|X]. X: (N, p) -> (N, T)."""
        pred_scores = self.ngb.predict(X)           # (N, K)
        return self.pca.inverse_transform(pred_scores)  # (N, T)

    def predict_var(self, X):
        """
        Predictive variance Var[F(t)|X] for each sample.
        X: (N, p) -> (N, T)

        Propagates covariance from score space to trajectory space:
          Var[F(t)|X=x] = V[:,t]^T Sigma(x) V[:,t]
        where V = pca.components_ (K, T).
        """
        dist = self.ngb.pred_dist(X)        # NGBoost distribution object
        Sigma = dist.cov                    # (N, K, K)
        V     = self.pca.components_        # (K, T)
        # Einsum: for each sample i and time t:
        # var[i,t] = sum_k sum_l V[k,t] * Sigma[i,k,l] * V[l,t]
        var = np.einsum('kt,ikl,lt->it', V, Sigma, V)  # (N, T)
        return np.maximum(var, 0.0)  # numerical safety

    def evaluate(self, X_test, Y_test):
        """Compute R2 on held-out data."""
        Y_pred = self.predict_mean(X_test)
        ss_res = np.sum((Y_test - Y_pred) ** 2)
        ss_tot = np.sum((Y_test - Y_test.mean()) ** 2)
        return 1.0 - ss_res / ss_tot

    @property
    def variance_explained(self):
        return self._var_explained


# ===========================================================================
# 6.  Cooperative game
# ===========================================================================

class IntradayFunctionalGame:
    """
    Cooperative functional game over day-level features.

    game_type : 'prediction'  -- v(S)(t) = E[F(X)(t) | X_S]
                'sensitivity' -- v(S)(t) = Var[F(X)(t) | X_S]
                'risk'        -- v(S)(t) = E[(Y_obs(t) - F(X)(t))^2 | X_S]

    The model is called as:
      prediction  -> model.predict_mean(X_imp)          (N, T)
      sensitivity -> model.predict_var(X_imp)           (N, T)  [mean over N]
      risk        -> residual^2 averaged over N

    For sensitivity the value function is the mean predictive variance,
    averaging over background samples X_bg with X_S fixed.  This measures
    how much residual uncertainty remains about F(t) after fixing X_S.
    """

    def __init__(self, model, X_background, x_explain,
                 game_type='prediction', Y_day_row=None,
                 sample_size=None, random_seed=RNG_SEED):

        if game_type not in ('prediction', 'sensitivity', 'risk'):
            raise ValueError('Unknown game_type: {}'.format(game_type))
        if game_type == 'risk' and Y_day_row is None:
            raise ValueError('Y_day_row required for risk game.')

        self.model        = model
        self.X_background = X_background
        self.x_explain    = x_explain
        self.game_type    = game_type
        self.Y_day_row    = Y_day_row
        self.random_seed  = random_seed
        self.sample_size  = (sample_size or SAMPLE_SIZE)[game_type]

        self.T          = T_BARS
        self.n_players  = len(DAY_FEATURE_NAMES)
        self.player_names = list(DAY_FEATURE_NAMES)

        self.coalitions   = np.array(
            list(itertools.product([False, True], repeat=self.n_players)),
            dtype=bool,
        )
        self.n_coalitions = len(self.coalitions)
        self._idx         = {tuple(c): i for i, c in enumerate(self.coalitions)}
        self.values       = None

    def _impute(self, coalition):
        rng = np.random.default_rng(self.random_seed)
        idx = rng.integers(0, len(self.X_background), size=self.sample_size)
        X   = self.X_background[idx].copy()
        for j in range(self.n_players):
            if coalition[j]:
                X[:, j] = self.x_explain[j]
        return X

    def value_function(self, coalition):
        X = self._impute(coalition)  # (sample_size, p)

        if self.game_type == 'prediction':
            return self.model.predict_mean(X).mean(axis=0)  # (T,)

        elif self.game_type == 'sensitivity':
            # Mean predictive variance over background samples
            # E_{X_-S}[ Var[F(t) | X_S, X_-S] ]
            return self.model.predict_var(X).mean(axis=0)   # (T,)

        elif self.game_type == 'risk':
            Y_pred    = self.model.predict_mean(X)          # (N, T)
            residuals = (self.Y_day_row[None, :] - Y_pred) ** 2
            return residuals.mean(axis=0)                   # (T,)

    def precompute(self):
        print('  [{}] {} coalitions x {} samples x {} bars ...'.format(
            self.game_type, self.n_coalitions, self.sample_size, self.T))
        self.values = np.zeros((self.n_coalitions, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i + 1) % 16 == 0 or (i + 1) == self.n_coalitions:
                print('    {}/{} done.'.format(i + 1, self.n_coalitions))

    def __getitem__(self, coalition):
        return self.values[self._idx[coalition]]

    @property
    def empty_value(self):
        return self[tuple([False] * self.n_players)]

    @property
    def grand_value(self):
        return self[tuple([True] * self.n_players)]


# ===========================================================================
# 7.  Möbius transform + Shapley values
# ===========================================================================

def functional_moebius_transform(game):
    p     = game.n_players
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(p), r) for r in range(p + 1)
    ))
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
# 8.  Kernels
# ===========================================================================

def kernel_identity(t):
    return np.eye(len(t))

def kernel_gaussian(t, sigma=6.0):
    d = t[:, None] - t[None, :]
    return np.exp(-0.5 * (d / sigma) ** 2)

def kernel_ou(t, length_scale=8.0):
    return np.exp(-np.abs(t[:, None] - t[None, :]) / length_scale)

def kernel_causal(t, length_scale=8.0):
    """One-sided exponential: K(t,s) = exp(-(t-s)/ls) for t>=s, else 0."""
    d = t[:, None] - t[None, :]
    return np.where(d >= 0, np.exp(-d / length_scale), 0.0)

def kernel_output_correlation(Y_day):
    """
    Empirical correlation kernel from observed intraday vol trajectories.
    Estimated from raw Y_day (before diurnal subtraction) to capture
    the true cross-bar dependence structure.
    """
    C   = np.cov(Y_day.T)
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    K   = np.clip(C / np.outer(std, std), -1.0, 1.0)
    print('  Output correlation kernel: off-diag mean={:.3f}  min={:.3f}  max={:.3f}'.format(
        (K - np.eye(T_BARS)).mean(), K.min(), K.max()))
    return K

def kernel_pca(model):
    """
    Kernel derived from the model's PCA covariance operator.
    K(t,s) = sum_k lambda_k * phi_k(t) * phi_k(s)
    where lambda_k are PCA eigenvalues and phi_k are eigenfunctions.
    This is the covariance kernel of the PCA-reconstructed process.
    It connects directly to Proposition 3 (diagonalization) in the paper.
    """
    V      = model.pca.components_      # (K, T)
    lambdas = model.pca.explained_variance_  # (K,)
    return np.einsum('k,kt,ks->ts', lambdas, V, V)  # (T, T)


# ===========================================================================
# 9.  Explanation helpers
# ===========================================================================

def _normalize_kernel(K, dt):
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return K / rs

def apply_kernel(effect, K, dt):
    return _normalize_kernel(K, dt) @ effect * dt

def integrated_explanation(effect, K, dt):
    return float(np.sum(_normalize_kernel(K, dt) @ effect) * dt ** 2)


# ===========================================================================
# 10.  Plotting helpers
# ===========================================================================

XTICK_IDXS   = list(range(0, T_BARS, 6))
XTICK_LABELS = [BAR_LABELS[i] for i in XTICK_IDXS]

def get_colors(n):
    return [plt.get_cmap('tab10')(i % 10) for i in range(n)]

def _set_time_axis(ax):
    ax.set_xticks(XTICK_IDXS)
    ax.set_xticklabels(XTICK_LABELS, rotation=45, ha='right', fontsize=7)
    ax.set_xlim(-0.3, T_BARS - 0.7)

def _period_shade(ax):
    ax.axvspan(0,  6,  alpha=0.10, color='#ffd699', zorder=0)
    ax.axvspan(72, 78, alpha=0.10, color='#ffd699', zorder=0)

def _game_ylabel(game_type):
    return {
        'prediction' : 'Effect on vol (%)',
        'sensitivity': 'Mean predictive Var[F(t)] (%^2)',
        'risk'       : 'Effect on MSE (%^2)',
    }[game_type]

def _game_title_suffix(game_type):
    return {
        'prediction' : 'Prediction game',
        'sensitivity': 'Sensitivity game (NGBoost MvN variance)',
        'risk'       : 'Risk (MSE) game',
    }[game_type]

def _scale(game_type):
    return 100 if game_type == 'prediction' else 1e4


# ===========================================================================
# 11.  Plots (identical interface to original script)
# ===========================================================================

def plot_model_diagnostics(model, X_day_np, Y_adj):
    """
    New figure specific to NGBoost+PCA:
      Panel 1: PCA cumulative variance explained
      Panel 2: First 6 PCA eigenfunctions (intraday basis)
      Panel 3: Predicted vs actual trajectory for 3 sample days
      Panel 4: R2 per bar (marginal prediction quality)
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        'NGBoost+PCA model diagnostics  (K={}, n_est={})'.format(
            model.n_pca, model.n_estimators),
        fontsize=11, fontweight='bold',
    )

    # Panel 1: cumulative variance
    ax = axes[0]
    cumvar = np.cumsum(model.pca.explained_variance_ratio_)
    ax.plot(range(1, model.n_pca + 1), cumvar * 100, 'o-',
            color='steelblue', lw=2, ms=6)
    ax.axhline(95, color='gray', lw=0.8, ls='--', label='95%')
    ax.set_xlabel('K components', fontsize=9)
    ax.set_ylabel('Cumulative variance (%)', fontsize=9)
    ax.set_title('PCA variance explained', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: eigenfunctions
    ax = axes[1]
    colors = get_colors(min(model.n_pca, 6))
    for k in range(min(model.n_pca, 6)):
        ax.plot(t_grid, model.pca.components_[k] * 100,
                color=colors[k], lw=1.8,
                label='PC{} ({:.1f}%)'.format(
                    k+1, model.pca.explained_variance_ratio_[k]*100))
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    _set_time_axis(ax)
    _period_shade(ax)
    ax.set_ylabel('Eigenfunction amplitude (%)', fontsize=9)
    ax.set_title('PCA eigenfunctions (intraday basis)', fontsize=10)
    ax.legend(fontsize=6.5, ncol=2)

    # Panel 3: sample predictions
    ax = axes[2]
    rng      = np.random.default_rng(RNG_SEED)
    idx_days = rng.choice(len(X_day_np), size=3, replace=False)
    cols3    = ['#e05c2a', '#2a9d8f', '#8338ec']
    Y_pred   = model.predict_mean(X_day_np[idx_days])
    for k, (i, col) in enumerate(zip(idx_days, cols3)):
        ax.plot(t_grid, Y_adj[i] * 100,  color=col, lw=1.2, alpha=0.5)
        ax.plot(t_grid, Y_pred[k] * 100, color=col, lw=2.0,
                ls='--', label='Day {}'.format(i))
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    _set_time_axis(ax)
    _period_shade(ax)
    ax.set_ylabel('Diurnal-adjusted vol (%)', fontsize=9)
    ax.set_title('Sample predictions (solid=actual, dashed=pred)', fontsize=9)
    ax.legend(fontsize=7)

    # Panel 4: R2 per bar
    ax = axes[3]
    Y_all_pred = model.predict_mean(X_day_np)
    r2_per_bar = np.array([
        r2_score(Y_adj[:, t], Y_all_pred[:, t]) for t in range(T_BARS)
    ])
    ax.plot(t_grid, r2_per_bar, color='#457b9d', lw=2)
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.fill_between(t_grid, 0, r2_per_bar,
                    where=r2_per_bar > 0, alpha=0.2, color='#457b9d')
    _set_time_axis(ax)
    _period_shade(ax)
    ax.set_ylabel('R² per bar', fontsize=9)
    ax.set_title('Marginal R² per 5-min bar', fontsize=10)

    plt.tight_layout()
    return fig


def plot_diurnal_and_trajectory(diurnal_mean, game, x_explain):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Diurnal baseline and explained-day trajectory',
                 fontsize=11, fontweight='bold')
    ax = axes[0]
    ax.plot(t_grid, diurnal_mean * 100, color='steelblue', lw=2)
    ax.fill_between(t_grid, 0, diurnal_mean * 100, alpha=0.2, color='steelblue')
    ax.set_ylabel('Mean |log-return| (%)')
    ax.set_title('Diurnal baseline', fontsize=10)
    _set_time_axis(ax); _period_shade(ax)
    ax2 = axes[1]
    sc = _scale(game.game_type)
    ax2.plot(t_grid, game.grand_value * sc, color='#e05c2a', lw=2.5,
             label='F(x)(t) -- grand coalition')
    ax2.plot(t_grid, game.empty_value * sc, color='gray', lw=1.5, ls='--',
             label='f0(t) -- empty coalition')
    ax2.fill_between(t_grid, game.empty_value*sc, game.grand_value*sc,
                     alpha=0.2, color='#e05c2a')
    ax2.axhline(0, color='gray', lw=0.8, ls=':')
    ax2.set_ylabel('Diurnal-adjusted vol (%)')
    ax2.legend(fontsize=8)
    ax2.set_title('NGBoost predicted trajectory  F(x) in R^{}'.format(T_BARS),
                  fontsize=10)
    _set_time_axis(ax2); _period_shade(ax2)
    feat_str = '  |  '.join('{}={:.2f}'.format(n, x_explain[j])
                             for j, n in enumerate(DAY_FEATURE_NAMES))
    fig.text(0.5, -0.02, feat_str, ha='center', fontsize=7, color='gray')
    plt.tight_layout()
    return fig


def plot_grand_vs_empty(game):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sc   = _scale(game.game_type)
    unit = '%' if game.game_type == 'prediction' else '(%^2 x 1e4)'
    ax.plot(t_grid, game.grand_value * sc, color='#e05c2a', lw=2.5,
            label='Grand coalition  v(N)(t)')
    ax.plot(t_grid, game.empty_value * sc, color='gray', lw=1.8, ls='--',
            label='Empty coalition  v(empty)(t)')
    gap = (game.grand_value - game.empty_value) * sc
    ax.fill_between(t_grid, game.empty_value*sc, game.grand_value*sc,
                    alpha=0.25, color='#e05c2a',
                    label='Gap  mean={:.4f}  max={:.4f}'.format(
                        gap.mean(), gap.max()))
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    _period_shade(ax); _set_time_axis(ax)
    ax.set_ylabel('Value ({})'.format(unit))
    ax.set_title('Grand vs empty coalition -- {}'.format(
        _game_title_suffix(game.game_type)), fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_main_effects(moebius, game, top_k=5):
    pnames = game.player_names
    sc     = _scale(game.game_type)
    imps   = {i: float(np.sum(np.abs(moebius[(i,)])))
              for i in range(game.n_players)}
    top    = sorted(imps, key=imps.get, reverse=True)[:top_k]
    colors = get_colors(top_k)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Top-{} main effects  [{}]'.format(
        top_k, _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold')
    ax = axes[0]
    for rank, fi in enumerate(top):
        ax.plot(t_grid, moebius[(fi,)] * sc, color=colors[rank],
                lw=2, label=pnames[fi])
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    _period_shade(ax); _set_time_axis(ax)
    ax.set_ylabel(_game_ylabel(game.game_type))
    ax.legend(fontsize=7)
    ax.set_title('Main-effect curves m(xi)(t)', fontsize=10)
    ax2 = axes[1]
    sp = sorted(imps.items(), key=lambda x: abs(x[1]), reverse=True)
    fi_list = [f for f, _ in sp]
    ax2.barh(range(len(fi_list)),
             [imps[f] * sc for f in fi_list],
             color=get_colors(len(fi_list)), alpha=0.8)
    ax2.set_yticks(range(len(fi_list)))
    ax2.set_yticklabels([pnames[f] for f in fi_list], fontsize=9)
    ax2.set_xlabel('Integrated |m(xi)(t)|')
    ax2.set_title('Feature importance (Identity kernel)', fontsize=10)
    plt.tight_layout()
    return fig


def plot_shapley_curves(shapley, game, kernels):
    pnames = game.player_names
    colors = get_colors(game.n_players)
    nk     = len(kernels)
    sc     = _scale(game.game_type)
    fig, axes = plt.subplots(1, nk, figsize=(3.5 * nk, 4))
    if nk == 1:
        axes = [axes]
    fig.suptitle('Shapley value curves  phi_i(t)  [{}]'.format(
        _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold')
    for k_idx, (k_name, K) in enumerate(kernels.items()):
        ax = axes[k_idx]
        for i in range(game.n_players):
            ax.plot(t_grid, apply_kernel(shapley[i], K, dt) * sc,
                    color=colors[i], lw=1.8, label=pnames[i])
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        _period_shade(ax); _set_time_axis(ax)
        ax.set_title(k_name, fontsize=9, fontweight='bold')
        ax.set_xlabel('Time', fontsize=8)
        if k_idx == 0:
            ax.set_ylabel(_game_ylabel(game.game_type), fontsize=8)
            ax.legend(fontsize=7)
    plt.tight_layout()
    return fig


def plot_kernel_comparison(moebius, game, kernels, top_k=4):
    pnames = game.player_names
    imps   = {i: float(np.sum(np.abs(moebius[(i,)])))
              for i in range(game.n_players)}
    top    = sorted(imps, key=imps.get, reverse=True)[:top_k]
    colors = get_colors(top_k)
    nk     = len(kernels)
    sc     = _scale(game.game_type)
    fig, axes = plt.subplots(top_k, nk, figsize=(3.2*nk, 3.0*top_k), sharey='row')
    fig.suptitle('Kernel-weighted main effects  [{}]'.format(
        _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold')
    for k_idx, (k_name, K) in enumerate(kernels.items()):
        for f_idx, feat_idx in enumerate(top):
            ax = axes[f_idx, k_idx]
            ax.plot(t_grid,
                    apply_kernel(moebius[(feat_idx,)], K, dt) * sc,
                    color=colors[f_idx], lw=1.8)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            ax.set_xlim(0, T_BARS - 1)
            ax.set_xticks(XTICK_IDXS[::2])
            ax.set_xticklabels(XTICK_LABELS[::2], rotation=45, ha='right', fontsize=6)
            if f_idx == 0:
                ax.set_title(k_name, fontsize=9, fontweight='bold')
            if k_idx == 0:
                ax.set_ylabel(pnames[feat_idx], fontsize=8)
            if f_idx == top_k - 1:
                ax.set_xlabel('Time', fontsize=7)
    plt.tight_layout()
    return fig


def plot_local_explanations(moebius, game, kernels, bars_of_interest=None):
    if bars_of_interest is None:
        bars_of_interest = [2, 24, 60]
    pnames   = game.player_names
    n_players = game.n_players
    n_bars   = len(bars_of_interest)
    nk       = len(kernels)
    colors   = get_colors(n_players)
    sc       = _scale(game.game_type)
    fig, axes = plt.subplots(n_bars, nk, figsize=(3.2*nk, 3.2*n_bars))
    fig.suptitle('Local explanations at selected bars  [{}]'.format(
        _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold')
    bar_wall = {2: '09:40', 24: '11:30', 60: '14:30'}
    for h_idx, t0 in enumerate(bars_of_interest):
        for k_idx, (k_name, K) in enumerate(kernels.items()):
            ax = axes[h_idx, k_idx]
            lv = [float(apply_kernel(moebius[(f,)], K, dt)[int(t0)])
                  for f in range(n_players)]
            tri = sorted(zip(pnames, lv, colors),
                         key=lambda x: abs(x[1]), reverse=True)
            ns, vs, cs = zip(*tri)
            bc = [c if v >= 0 else '#c0392b' for v, c in zip(vs, cs)]
            ax.barh(range(len(ns)), [v * sc for v in vs], color=bc, alpha=0.85)
            ax.set_yticks(range(len(ns)))
            ax.set_yticklabels(ns, fontsize=7)
            ax.axvline(0, color='gray', lw=0.8, ls=':')
            ax.set_xlabel(_game_ylabel(game.game_type), fontsize=7)
            if h_idx == 0:
                ax.set_title(k_name, fontsize=9, fontweight='bold')
            if k_idx == 0:
                ax.set_ylabel('t = {}'.format(bar_wall.get(t0, str(t0))),
                              fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_profiles_comparison(profiles, kernels, game_type='prediction'):
    K_id  = kernels['Identity']
    n     = len(profiles)
    sc    = _scale(game_type)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle('Shapley curves across profiles  [{}]'.format(
        _game_title_suffix(game_type)),
        fontsize=11, fontweight='bold')
    for ax, (label, (gp, shap, pnames)) in zip(axes, profiles.items()):
        colors = get_colors(gp.n_players)
        order  = sorted(range(gp.n_players),
                        key=lambda i: float(np.sum(np.abs(shap[i]))),
                        reverse=True)
        for rank, fi in enumerate(order[:5]):
            ax.plot(t_grid, apply_kernel(shap[fi], K_id, dt) * sc,
                    color=colors[rank], lw=2, label=pnames[fi])
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        _period_shade(ax)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_ylabel(_game_ylabel(game_type))
        ax.legend(fontsize=7)
        _set_time_axis(ax)
    plt.tight_layout()
    return fig


def plot_interaction_effects(moebius, game, kernels, top_pairs=3):
    pnames    = game.player_names
    np_       = game.n_players
    sc        = _scale(game.game_type)
    pair_imps = {
        (i, j): float(np.sum(np.abs(moebius.get((i, j), np.zeros(T_BARS)))))
        for i in range(np_) for j in range(i+1, np_)
    }
    if not pair_imps:
        return None
    top_list = sorted(pair_imps, key=pair_imps.get, reverse=True)[:top_pairs]
    colors   = get_colors(top_pairs)
    nk       = len(kernels)
    fig, axes = plt.subplots(top_pairs, nk, figsize=(3.2*nk, 3.0*top_pairs),
                              sharey='row')
    if top_pairs == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Pairwise interaction effects  [{}]'.format(
        _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold')
    for p_idx, S in enumerate(top_list):
        i, j  = S
        label = '{}  x  {}'.format(pnames[i], pnames[j])
        for k_idx, (k_name, K) in enumerate(kernels.items()):
            ax = axes[p_idx, k_idx]
            ax.plot(t_grid,
                    apply_kernel(moebius.get(S, np.zeros(T_BARS)), K, dt) * sc,
                    color=colors[p_idx], lw=2)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            ax.set_xlim(0, T_BARS-1)
            ax.set_xticks(XTICK_IDXS[::2])
            ax.set_xticklabels(XTICK_LABELS[::2], rotation=45, ha='right', fontsize=6)
            if p_idx == 0:
                ax.set_title(k_name, fontsize=9, fontweight='bold')
            if k_idx == 0:
                ax.set_ylabel(label, fontsize=8)
            if p_idx == top_pairs - 1:
                ax.set_xlabel('Time', fontsize=7)
    plt.tight_layout()
    return fig


def plot_ann_indicator_causal(moebius, game, kernels_standard,
                               causal_length_scales=(4, 8, 16)):
    ann_idx      = DAY_FEATURE_NAMES.index('ann_indicator')
    raw          = moebius[(ann_idx,)]
    sc           = _scale(game.game_type)
    causal_kernels = {
        'Causal ls={}'.format(ls): kernel_causal(t_grid, ls)
        for ls in causal_length_scales
    }
    all_kernels  = dict(kernels_standard)
    all_kernels.update(causal_kernels)
    nk           = len(all_kernels)
    causal_colors = plt.get_cmap('Oranges')(
        np.linspace(0.4, 0.9, len(causal_length_scales)))
    fig, axes = plt.subplots(1, nk, figsize=(3.2*nk, 3.8), sharey=True)
    fig.suptitle('ann_indicator -- causal kernel comparison  [{}]'.format(
        _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold')
    for k_idx, (k_name, K) in enumerate(all_kernels.items()):
        ax     = axes[k_idx]
        is_cau = k_name.startswith('Causal')
        c_idx  = list(causal_kernels.keys()).index(k_name) if is_cau else 0
        color  = causal_colors[c_idx] if is_cau else 'steelblue'
        ax.plot(t_grid, apply_kernel(raw, K, dt) * sc, color=color, lw=2)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.axvline(54, color='#888', lw=0.8, ls='--', alpha=0.5, label='14:00')
        _period_shade(ax); _set_time_axis(ax)
        ax.set_title(k_name, fontsize=9, fontweight='bold')
        ax.set_xlabel('Time', fontsize=8)
        if k_idx == 0:
            ax.set_ylabel(_game_ylabel(game.game_type), fontsize=8)
    plt.tight_layout()
    return fig


def plot_kernel_hyperparams(moebius, game, top_k=4):
    pnames        = game.player_names
    imps          = {i: float(np.sum(np.abs(moebius[(i,)])))
                     for i in range(game.n_players)}
    top           = sorted(imps, key=imps.get, reverse=True)[:top_k]
    sc            = _scale(game.game_type)
    gauss_sigmas  = [3, 6, 12]
    ou_scales     = [4, 8, 16]
    causal_scales = [4, 8, 16]
    gauss_colors  = plt.get_cmap('Blues')(np.linspace(0.4, 0.9, len(gauss_sigmas)))
    ou_colors     = plt.get_cmap('Oranges')(np.linspace(0.4, 0.9, len(ou_scales)))
    causal_colors = plt.get_cmap('Greens')(np.linspace(0.4, 0.9, len(causal_scales)))
    K_id          = kernel_identity(t_grid)
    fig, axes = plt.subplots(top_k, 4, figsize=(17, 3.0*top_k), sharey='row')
    fig.suptitle('Kernel hyperparameter sweep  [{}]'.format(
        _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold')
    for col, ct in enumerate(['Identity (ref)', 'Gaussian', 'OU', 'Causal']):
        axes[0, col].set_title(ct, fontsize=10, fontweight='bold')
    for row, feat_idx in enumerate(top):
        raw      = moebius[(feat_idx,)]
        id_curve = apply_kernel(raw, K_id, dt) * sc
        axes[row, 0].plot(t_grid, id_curve, color='black', lw=2, label='Identity')
        axes[row, 0].axhline(0, color='gray', lw=0.6, ls=':')
        axes[row, 0].set_ylabel(pnames[feat_idx], fontsize=9)
        _period_shade(axes[row, 0]); _set_time_axis(axes[row, 0])
        for ci, sig in enumerate(gauss_sigmas):
            axes[row, 1].plot(t_grid,
                apply_kernel(raw, kernel_gaussian(t_grid, sig), dt) * sc,
                color=gauss_colors[ci], lw=2, label='sigma={}'.format(sig))
        for ci, ls_val in enumerate(ou_scales):
            axes[row, 2].plot(t_grid,
                apply_kernel(raw, kernel_ou(t_grid, ls_val), dt) * sc,
                color=ou_colors[ci], lw=2, label='ls={}'.format(ls_val))
        for ci, ls_val in enumerate(causal_scales):
            axes[row, 3].plot(t_grid,
                apply_kernel(raw, kernel_causal(t_grid, ls_val), dt) * sc,
                color=causal_colors[ci], lw=2, label='ls={}'.format(ls_val))
        for col in range(1, 4):
            axes[row, col].plot(t_grid, id_curve, color='black',
                                lw=1, ls='--', alpha=0.4)
            axes[row, col].axhline(0, color='gray', lw=0.6, ls=':')
            _period_shade(axes[row, col]); _set_time_axis(axes[row, col])
            if row == 0:
                axes[row, col].legend(fontsize=7)
        if row == top_k - 1:
            for col in range(4):
                axes[row, col].set_xlabel('Time', fontsize=7)
    plt.tight_layout()
    return fig


# ===========================================================================
# 12.  Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  Functional Explanation -- Intraday SPY  (NGBoost+PCA)')
    print('=' * 60)

    for d in [DATA_DIR] + list(PLOT_DIRS.values()):
        _require_dir(d)

    def save(fig, game_type, name):
        path = os.path.join(PLOT_DIRS[game_type], name)
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print('  Saved: {}'.format(path))
        plt.close(fig)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print('\n[1] Loading data ...')
    X_day, Y_day, Y_adj, diurnal_mean = load_and_aggregate()
    X_day_np = X_day.to_numpy().astype(float)

    # ── 2. Model ──────────────────────────────────────────────────────────
    print('\n[2] Fitting NGBoost+PCA model ...')
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_day_np, Y_adj, test_size=0.2, random_state=RNG_SEED
    )
    model = NGBoostFunctionalModel(
        n_pca=N_PCA, n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE, random_state=RNG_SEED,
    )
    model.fit(X_tr, Y_tr)
    r2_test = model.evaluate(X_te, Y_te)
    print('  Test R² (trajectory-level): {:.4f}'.format(r2_test))

    # ── 3. Model diagnostics ──────────────────────────────────────────────
    print('\n[3] Model diagnostics ...')
    for game_type in PLOT_DIRS:
        save(
            plot_model_diagnostics(model, X_day_np, Y_adj),
            game_type,
            'model_diagnostics.pdf',
        )

    # ── 4. Day profiles ───────────────────────────────────────────────────
    print('\n[4] Selecting day profiles ...')
    vix_col = DAY_FEATURE_NAMES.index('vix_prev')
    vix_p25 = float(np.percentile(X_day_np[:, vix_col], 25))
    vix_p75 = float(np.percentile(X_day_np[:, vix_col], 75))

    def find_day_profile(conditions, pool):
        mask = np.ones(len(pool), dtype=bool)
        for feat, (lo, hi) in conditions.items():
            ci = DAY_FEATURE_NAMES.index(feat)
            mask &= (pool[:, ci] >= lo) & (pool[:, ci] <= hi)
        hits = pool[mask]
        if len(hits) == 0:
            raise RuntimeError('No day matches profile: {}'.format(conditions))
        print('    {} matching days; picking median.'.format(len(hits)))
        return hits[len(hits) // 2]

    x_p1 = find_day_profile(
        {'ann_indicator': (0.9, 1.1), 'vix_prev': (vix_p75, 999)}, X_day_np)
    x_p2 = find_day_profile(
        {'ann_indicator': (-0.1, 0.1), 'vix_prev': (0, vix_p25)}, X_day_np)
    x_p3 = find_day_profile({'day_of_week': (-0.1, 0.1)}, X_day_np)

    def _find_y_row(x_profile):
        diffs = np.abs(X_day_np - x_profile[None, :]).sum(axis=1)
        return Y_adj[int(np.argmin(diffs))]

    profile_defs = [
        ('High-VIX Announcement', x_p1, _find_y_row(x_p1)),
        ('Quiet Low-VIX',         x_p2, _find_y_row(x_p2)),
        ('Monday Gap',            x_p3, _find_y_row(x_p3)),
    ]
    for lbl, xp, _ in profile_defs:
        print('  {}: {}'.format(lbl, '  '.join(
            '{}={:.3f}'.format(n, xp[j]) for j, n in enumerate(DAY_FEATURE_NAMES))))

    # ── 5. Kernels ────────────────────────────────────────────────────────
    print('\n[5] Building kernels ...')
    K_corr   = kernel_output_correlation(Y_day)
    K_ou     = kernel_ou(t_grid, length_scale=8.0)
    K_causal = kernel_causal(t_grid, length_scale=8.0)
    K_pca    = kernel_pca(model)
    kernels  = {
        'Identity'   : kernel_identity(t_grid),
        'Gaussian'   : kernel_gaussian(t_grid, sigma=6.0),
        'OU'         : K_ou,
        'Correlation': K_corr,
        'PCA cov'    : K_pca,   # new: connects to Prop. 3 in paper
    }
    print('  PCA covariance kernel: range [{:.2e}, {:.2e}]'.format(
        K_pca.min(), K_pca.max()))

    # ── 6. Games loop ─────────────────────────────────────────────────────
    for game_type in ('prediction', 'sensitivity', 'risk'):
        print('\n' + '='*60)
        print('  GAME: {}'.format(game_type.upper()))
        print('='*60)
        all_profiles = {}

        for prof_label, x_explain, y_explain in profile_defs:
            print('\n  Profile: {}  game: {}'.format(prof_label, game_type))
            game = IntradayFunctionalGame(
                model       = model,
                X_background= X_day_np,
                x_explain   = x_explain,
                game_type   = game_type,
                Y_day_row   = y_explain,
                random_seed = RNG_SEED,
            )
            game.precompute()
            print('  Grand-empty gap max = {:.4e}'.format(
                np.abs(game.grand_value - game.empty_value).max()))

            moebius = functional_moebius_transform(game)
            shapley = shapley_from_moebius(moebius, game.n_players)
            slug    = prof_label.lower().replace(' ', '_').replace('-', '')

            # Full set of plots for Profile 1
            if prof_label == 'High-VIX Announcement':
                if game_type == 'prediction':
                    save(plot_diurnal_and_trajectory(diurnal_mean, game, x_explain),
                         game_type, 'diurnal_and_trajectory.pdf')
                save(plot_grand_vs_empty(game),
                     game_type, 'grand_vs_empty_{}.pdf'.format(slug))
                save(plot_main_effects(moebius, game),
                     game_type, 'main_effects_{}.pdf'.format(slug))
                save(plot_shapley_curves(shapley, game, kernels),
                     game_type, 'shapley_curves_{}.pdf'.format(slug))
                fig_int = plot_interaction_effects(moebius, game, kernels, top_pairs=3)
                if fig_int is not None:
                    save(fig_int, game_type,
                         'interaction_effects_{}.pdf'.format(slug))
                save(plot_ann_indicator_causal(moebius, game, kernels,
                                               causal_length_scales=(4, 8, 16)),
                     game_type, 'ann_indicator_causal_{}.pdf'.format(slug))
                save(plot_kernel_hyperparams(moebius, game, top_k=4),
                     game_type, 'kernel_hyperparams_{}.pdf'.format(slug))

            # Kernel comparison + local explanations for all profiles
            save(plot_kernel_comparison(moebius, game, kernels, top_k=4),
                 game_type, 'kernel_comparison_{}.pdf'.format(slug))
            save(plot_local_explanations(moebius, game, kernels,
                                         bars_of_interest=[2, 24, 60]),
                 game_type, 'local_explanations_{}.pdf'.format(slug))

            all_profiles[prof_label] = (game, shapley, game.player_names)

        save(plot_profiles_comparison(all_profiles, kernels, game_type=game_type),
             game_type, 'profiles_comparison.pdf')

    print('\n' + '='*60)
    print('  Done.  Plots saved under {}'.format(BASE_PLOT_DIR))
    print('='*60)