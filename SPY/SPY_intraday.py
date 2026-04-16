"""
Functional Explanation Framework -- Intraday Financial Returns Example
======================================================================
Data source : Polygon.io Starter subscription -- SPY 5-minute bars, 2 years.
VIX         : yfinance (daily, free).

Changes vs original:
  - Removed Constant and Periodic kernels throughout
  - Added output correlation kernel (estimated from observed Y_day)
  - Added causal (one-sided exponential) kernel
  - New figure: ann_indicator causal kernel comparison (separate)
  - New figure: kernel hyperparameter sweep (Gaussian / OU / causal)
  - Added sensitivity game (Var[f(x)|S]) and risk game (MSE vs observed)
  - Kernel comparison + local explanation plots for Profiles 2 and 3
  - Output organised into intraday/prediction/, intraday/sensitivity/,
    intraday/risk/ subdirectories
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0.  Settings
# ---------------------------------------------------------------------------
TICKER         = 'SPY'
_HERE          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(_HERE, 'data')
CACHE_PATH     = os.path.join(DATA_DIR, 'spy_5min_cache.csv')
VIX_CACHE_PATH = os.path.join(DATA_DIR, 'vix_daily_cache.csv')

POLYGON_API_KEY = ''
FETCH_START     = '2022-01-01'
FETCH_END       = '2024-04-01'
VIX_LOOKBACK_DAYS = 5

RNG_SEED    = 42
# Larger sample sizes for sensitivity/risk since variance/MSE estimation
# converges more slowly than mean estimation.
SAMPLE_SIZE = {
    'prediction' : 200,
    'sensitivity': 400,
    'risk'       : 400,
}
T_BARS      = 78
dt          = 1.0

_open_min  = 9 * 60 + 30
BAR_LABELS = [
    '{:02d}:{:02d}'.format(
        (_open_min + i * 5) // 60,
        (_open_min + i * 5) % 60
    )
    for i in range(T_BARS)
]
t_grid = np.arange(T_BARS, dtype=float)

BASE_PLOT_DIR = os.path.join('plots', 'intraday')
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

INTRADAY_PERIODS = {
    'open'     : (0,   6),
    'morning'  : (6,  24),
    'midday'   : (24, 54),
    'afternoon': (54, 72),
    'close'    : (72, 78),
}

DAY_FEATURE_NAMES = [
    'vix_prev',
    'overnight_ret',
    'ann_indicator',
    'day_of_week',
    'trailing_rv',
    'month',
]

INTERACTION_FEATURES = [
    'vix_prev',
    'overnight_ret',
    'ann_indicator',
    'trailing_rv',
    'day_of_week',
]


# ===========================================================================
# 1.  Validation helpers  (unchanged)
# ===========================================================================

def _require_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            'Cannot create directory {}: {}'.format(path, exc)
        ) from exc


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
                'VIX prior-day data is missing for {}/{} trading dates '
                '(threshold: {}).\nFirst 10 missing: {}\n'
                'Fix: delete {} and re-run.'.format(
                    len(missing), len(trading_dates), threshold,
                    missing[:10], VIX_CACHE_PATH
                )
            )
        print(
            '  Note: {} trading date(s) have no prior VIX within '
            '14 calendar days.  Affected: {}{}'.format(
                len(missing),
                missing[:5],
                ' ...' if len(missing) > 5 else ''
            )
        )


def _validate_bars(bars):
    required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    missing  = required - set(bars.columns)
    if missing:
        raise RuntimeError(
            'Bar cache is missing columns: {}.\n'
            'Delete {} and re-fetch.'.format(missing, CACHE_PATH)
        )
    if len(bars) == 0:
        raise RuntimeError('Bar cache is empty ({}).'.format(CACHE_PATH))
    nan_frac = bars['close'].isna().mean()
    if nan_frac > 0.01:
        raise RuntimeError(
            '{:.1f}% of close prices are NaN.'.format(nan_frac * 100)
        )


def _validate_pivot(pivot):
    if len(pivot) < 100:
        raise RuntimeError(
            'Only {} complete trading days survived filtering '
            '(need >= 100).'.format(len(pivot))
        )


def _validate_feature_matrix(X_day):
    for col in DAY_FEATURE_NAMES:
        if col not in X_day.columns:
            raise RuntimeError(
                'Expected feature column "{}" is absent.'.format(col)
            )
        n_unique = X_day[col].nunique()
        std      = float(X_day[col].std())
        if n_unique <= 1 or std < 1e-8:
            raise RuntimeError(
                'Feature "{}" is constant (nunique={}, std={:.2e}).'.format(
                    col, n_unique, std
                )
            )


# ===========================================================================
# 2.  VIX loading  (unchanged)
# ===========================================================================

def _vix_fetch_start(first_trading_date):
    return (
        pd.Timestamp(first_trading_date)
        - pd.Timedelta(days=VIX_LOOKBACK_DAYS)
    ).strftime('%Y-%m-%d')


def _resolve_close_column(df, source_label):
    if isinstance(df.columns, pd.MultiIndex):
        for field in ('Close', 'Adj Close'):
            for ticker in ('^VIX', 'VIX', ''):
                if (field, ticker) in df.columns:
                    series = df[(field, ticker)].dropna()
                    if len(series) > 0:
                        return series
        close_like = [c for c in df.columns if c[0] in ('Close', 'Adj Close')]
        if close_like:
            series = df[close_like[0]].dropna()
            if len(series) > 0:
                return series
        raise RuntimeError(
            '{}: no recognisable Close column.'.format(source_label)
        )
    else:
        for col in ('Close', 'Adj Close', 'close', 'adj close'):
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    return series
        raise RuntimeError(
            '{}: no recognisable Close column.'.format(source_label)
        )


def _fetch_vix_yfinance(start, end):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('yfinance not installed.') from exc

    end_exclusive = (
        pd.Timestamp(end) + pd.Timedelta(days=1)
    ).strftime('%Y-%m-%d')

    print('    Downloading ^VIX via yfinance  {} -> {} ...'.format(
        start, end))
    try:
        raw = yf.download('^VIX', start=start, end=end_exclusive,
                          progress=False, auto_adjust=True)
    except Exception as exc:
        raise RuntimeError('yfinance.download failed: {}'.format(exc)) from exc

    if raw is None or len(raw) == 0:
        raise RuntimeError('yfinance returned empty DataFrame for ^VIX.')

    close = _resolve_close_column(raw, 'yfinance ^VIX')
    if len(close) < 200:
        raise RuntimeError(
            'yfinance returned only {} VIX observations.'.format(len(close))
        )

    result = {
        (str(idx.date()) if hasattr(idx, 'date') else str(idx)): float(val)
        for idx, val in close.items()
    }
    print('    VIX loaded: {} days  [{:.1f} - {:.1f}]'.format(
        len(result), min(result.values()), max(result.values())))
    return result


def load_vix(first_trading_date, last_trading_date):
    vix_start = _vix_fetch_start(first_trading_date)
    vix_end   = last_trading_date

    if os.path.isfile(VIX_CACHE_PATH):
        print('    Reading VIX cache {} ...'.format(VIX_CACHE_PATH))
        try:
            df_cache = pd.read_csv(VIX_CACHE_PATH, index_col=0)
        except Exception as exc:
            raise RuntimeError(
                'VIX cache cannot be read: {}'.format(exc)
            ) from exc

        if 'vix' not in df_cache.columns:
            raise RuntimeError('VIX cache missing "vix" column.')

        result = {
            str(k): float(v)
            for k, v in df_cache['vix'].dropna().items()
        }
        if not result:
            raise RuntimeError('VIX cache contains no usable rows.')

        cached_start = min(result)
        cached_end   = max(result)

        if cached_start > vix_start or cached_end < vix_end:
            raise RuntimeError(
                'VIX cache covers {} -> {} but required window is '
                '{} -> {}.\nDelete {} and re-run.'.format(
                    cached_start, cached_end, vix_start, vix_end,
                    VIX_CACHE_PATH
                )
            )

        print('    VIX cache OK: {} days  [{:.1f} - {:.1f}]'.format(
            len(result), min(result.values()), max(result.values())))
        return result

    result = _fetch_vix_yfinance(vix_start, vix_end)
    pd.DataFrame({'vix': result}).to_csv(VIX_CACHE_PATH)
    print('    VIX cached to {}'.format(VIX_CACHE_PATH))
    return result


# ===========================================================================
# 3.  SPY bar loading  (unchanged)
# ===========================================================================

def _fetch_polygon_5min(api_key):
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError('requests not installed.') from exc

    base    = 'https://api.polygon.io/v2/aggs/ticker'
    chunks  = []
    windows = []
    t   = pd.Timestamp(FETCH_START)
    end = pd.Timestamp(FETCH_END)
    while t < end:
        t_end = min(t + pd.DateOffset(months=6), end)
        windows.append((t.strftime('%Y-%m-%d'), t_end.strftime('%Y-%m-%d')))
        t = t_end

    for i, (w_start, w_end) in enumerate(windows):
        print('    [{}/{}]  {} -> {} ...'.format(
            i + 1, len(windows), w_start, w_end))
        url    = '{}/{}/range/5/minute/{}/{}'.format(
            base, TICKER, w_start, w_end)
        params = {'adjusted': 'true', 'sort': 'asc',
                  'limit': 50000, 'apiKey': api_key}
        while url:
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
            except Exception as exc:
                raise RuntimeError(
                    'Polygon.io request failed: {}'.format(exc)
                ) from exc

            data   = resp.json()
            status = data.get('status')
            if status not in ('OK', 'DELAYED'):
                raise RuntimeError(
                    'Polygon.io status="{}".'.format(status)
                )
            results = data.get('results')
            if results is None:
                raise RuntimeError('Polygon.io response has no "results".')
            chunks.extend(results)
            url    = data.get('next_url')
            params = {'apiKey': api_key}

        if i < len(windows) - 1:
            time.sleep(1)

    if not chunks:
        raise RuntimeError('Polygon.io returned zero bars.')

    df = pd.DataFrame(chunks)
    df['timestamp'] = (
        pd.to_datetime(df['t'], unit='ms', utc=True)
        .dt.tz_convert('America/New_York')
    )
    df = df.rename(columns={
        'o': 'open', 'h': 'high', 'l': 'low',
        'c': 'close', 'v': 'volume',
    })
    df = (
        df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        .sort_values('timestamp')
        .reset_index(drop=True)
    )
    print('    Fetched {:,} 5-min bars.'.format(len(df)))
    return df


def load_bars():
    if os.path.isfile(CACHE_PATH):
        print('  Loading bar cache from {} ...'.format(CACHE_PATH))
        try:
            bars = pd.read_csv(CACHE_PATH)
        except Exception as exc:
            raise RuntimeError(
                'Bar cache cannot be read: {}'.format(exc)
            ) from exc

        try:
            bars['timestamp'] = (
                pd.to_datetime(bars['timestamp'], utc=True)
                .dt.tz_convert('America/New_York')
            )
        except Exception as exc:
            raise RuntimeError(
                'Bar cache has unparseable timestamp: {}'.format(exc)
            ) from exc

        _validate_bars(bars)
        print('  Loaded {:,} rows.'.format(len(bars)))
        return bars

    if not POLYGON_API_KEY:
        raise RuntimeError(
            'No bar cache found and POLYGON_API_KEY is not set.\n'
            'Either set POLYGON_API_KEY or place a CSV at:\n'
            '  {}'.format(CACHE_PATH)
        )

    print('  Fetching SPY 5-min bars from Polygon.io ...')
    bars = _fetch_polygon_5min(POLYGON_API_KEY)
    _validate_bars(bars)
    bars.to_csv(CACHE_PATH, index=False)
    print('  Bar cache saved to {}'.format(CACHE_PATH))
    return bars


# ===========================================================================
# 4.  Feature construction  (unchanged)
# ===========================================================================

def _resolve_vix_prev(vix_dict, date_str, same_day_fallback=True):
    dt_obj = pd.Timestamp(date_str)
    for d in range(1, 15):
        key = str((dt_obj - pd.Timedelta(d, 'D')).date())
        if key in vix_dict:
            return vix_dict[key]
    if same_day_fallback and date_str in vix_dict:
        return vix_dict[date_str]
    raise RuntimeError(
        'No VIX value found within 14 calendar days before {}.'.format(
            date_str)
    )


def load_and_aggregate():
    bars = load_bars()
    print('  Processing {:,} bars ...'.format(len(bars)))
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

    if len(bars) == 0:
        raise RuntimeError('All bars dropped after numeric filtering.')

    bars['abs_log_ret'] = np.abs(np.log(bars['close'] / bars['open']))

    pivot = bars.pivot_table(
        index='date', columns='bar_idx',
        values='abs_log_ret', aggfunc='mean',
    )
    pivot = pivot.reindex(columns=range(T_BARS))
    pivot = pivot[pivot.notna().sum(axis=1) >= 70].fillna(pivot.mean())
    _validate_pivot(pivot)

    Y_day        = pivot.values.astype(float)
    dates        = pivot.index.tolist()
    diurnal_mean = Y_day.mean(axis=0)
    Y_adj        = Y_day - diurnal_mean[None, :]

    print('  Complete trading days : {}'.format(len(dates)))
    print('  Y_day  mean={:.4f}%  std={:.4f}%'.format(
        Y_day.mean() * 100, Y_day.std() * 100))

    print('  Loading VIX ...')
    vix_dict = load_vix(dates[0], dates[-1])
    _validate_vix_dict(vix_dict, dates)

    records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str)
        allow_same_day = (i == 0)
        vix_prev = _resolve_vix_prev(
            vix_dict, date_str, same_day_fallback=allow_same_day
        )

        day_bars  = bars[bars['date'] == date_str].sort_values('bar_idx')
        prev_bars = (
            bars[bars['date'] < date_str]
            .sort_values(['date', 'bar_idx'])
        )

        if len(day_bars) == 0:
            raise RuntimeError(
                'No bars for trading date {}.'.format(date_str)
            )

        this_open = float(day_bars.iloc[0]['open'])

        if len(prev_bars) == 0:
            if i == 0:
                overnight_ret = 0.0
            else:
                raise RuntimeError(
                    'No prior bars for {} (index {}).'.format(date_str, i)
                )
        else:
            prev_close = float(prev_bars.iloc[-1]['close'])
            if prev_close <= 0:
                raise RuntimeError(
                    'Non-positive prior close for {}.'.format(date_str)
                )
            overnight_ret = float(np.log(this_open / prev_close))

        recent      = list(range(max(0, i - 5), i))
        trailing_rv = (
            float(Y_day[recent].mean()) if recent
            else float(Y_day.mean())
        )

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

    print('  Announcement days : {} ({:.1f}%)'.format(
        int(X_day['ann_indicator'].sum()),
        X_day['ann_indicator'].mean() * 100))

    df_long = _build_long_format(X_day, Y_adj)
    return X_day, Y_day, Y_adj, diurnal_mean, df_long


def _build_long_format(X_day, Y_adj):
    vals = X_day.values
    cols = list(X_day.columns)
    rows = []
    for i in range(len(X_day)):
        base = {c: vals[i, j] for j, c in enumerate(cols)}
        for t in range(T_BARS):
            row            = base.copy()
            row['bar_idx'] = float(t)
            row['y']       = float(Y_adj[i, t])
            rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# 5.  Feature augmentation  (unchanged)
# ===========================================================================

def _augment_dataframe(df):
    df = df.copy()
    df['bar_sin'] = np.sin(2 * np.pi * df['bar_idx'] / T_BARS)
    df['bar_cos'] = np.cos(2 * np.pi * df['bar_idx'] / T_BARS)

    interaction_cols = []
    for feat in INTERACTION_FEATURES:
        if feat not in df.columns:
            continue
        for pname, (lo, hi) in INTRADAY_PERIODS.items():
            col  = '{}_{}'.format(feat, pname)
            mask = (
                (df['bar_idx'] >= lo) & (df['bar_idx'] < hi)
            ).astype(float)
            df[col] = df[feat] * mask
            interaction_cols.append(col)

    model_cols = (
        DAY_FEATURE_NAMES
        + interaction_cols
        + ['bar_sin', 'bar_cos', 'bar_idx']
    )
    return df, model_cols


def _build_X_matrix(X_day_np, bar, model_feature_cols):
    n   = X_day_np.shape[0]
    out = np.zeros((n, len(model_feature_cols)), dtype=float)

    bar_sin = np.sin(2 * np.pi * bar / T_BARS)
    bar_cos = np.cos(2 * np.pi * bar / T_BARS)

    for ci, col in enumerate(model_feature_cols):
        if col in DAY_FEATURE_NAMES:
            src        = DAY_FEATURE_NAMES.index(col)
            out[:, ci] = X_day_np[:, src]
        elif col == 'bar_sin':
            out[:, ci] = bar_sin
        elif col == 'bar_cos':
            out[:, ci] = bar_cos
        elif col == 'bar_idx':
            out[:, ci] = bar
        else:
            matched = False
            for feat in INTERACTION_FEATURES:
                for pname, (lo, hi) in INTRADAY_PERIODS.items():
                    if col == '{}_{}'.format(feat, pname):
                        in_win = float(lo <= bar < hi)
                        if feat in DAY_FEATURE_NAMES:
                            src        = DAY_FEATURE_NAMES.index(feat)
                            out[:, ci] = X_day_np[:, src] * in_win
                        matched = True
                        break
                if matched:
                    break

    return out


# ===========================================================================
# 6.  Model  (unchanged)
# ===========================================================================

def train_model(df_long, random_seed=RNG_SEED):
    df_aug, feature_cols = _augment_dataframe(df_long)

    X_model = df_aug[feature_cols].to_numpy().astype(float)
    y_model = df_aug['y'].to_numpy().astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_model,
        test_size=0.2, random_state=random_seed, shuffle=True,
    )

    print('  Training GBT  ({} features, {:,} training rows) ...'.format(
        len(feature_cols), len(X_train)))
    model = GradientBoostingRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.04,
        subsample=0.8, min_samples_leaf=15, random_state=random_seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    print('  Test RMSE: {:.5f}   R2: {:.4f}'.format(rmse, r2))

    return model, feature_cols, X_train, X_test, y_train, y_test


# ===========================================================================
# 7.  Functional prediction  (unchanged)
# ===========================================================================

def functional_predict(model, X_day_np, model_feature_cols, t=None):
    if t is None:
        t = t_grid
    n = X_day_np.shape[0]
    Y = np.zeros((n, len(t)))
    for t_idx, bar in enumerate(t):
        X_h         = _build_X_matrix(X_day_np, bar, model_feature_cols)
        Y[:, t_idx] = model.predict(X_h)
    return Y


# ===========================================================================
# 8.  Functional Games
# ===========================================================================

class IntradayFunctionalGame:
    """
    Cooperative functional game over day-level features.

    game_type : 'prediction'  -- v(S)(t) = E[f(x)(t) | x_S = x_explain_S]
                'sensitivity' -- v(S)(t) = Var[f(x)(t) | x_S = x_explain_S]
                'risk'        -- v(S)(t) = E[(Y(t)-f(x)(t))^2 | x_S=x_explain_S]

    For 'risk', Y_day_row must be provided (the observed trajectory for
    x_explain, shape (T_BARS,)).
    """

    def __init__(self, model, X_background, x_explain,
                 model_feature_cols, game_type='prediction',
                 Y_day_row=None,
                 sample_size=SAMPLE_SIZE, random_seed=RNG_SEED):

        if X_background.shape[1] != len(DAY_FEATURE_NAMES):
            raise RuntimeError('X_background column mismatch.')
        if x_explain.shape[0] != len(DAY_FEATURE_NAMES):
            raise RuntimeError('x_explain length mismatch.')
        if game_type not in ('prediction', 'sensitivity', 'risk'):
            raise RuntimeError(
                'game_type must be prediction/sensitivity/risk.'
            )
        if game_type == 'risk' and Y_day_row is None:
            raise RuntimeError(
                'Y_day_row required for risk game.'
            )

        self.model               = model
        self.X_background        = X_background
        self.x_explain           = x_explain
        self.model_feature_cols  = model_feature_cols
        self.game_type           = game_type
        self.Y_day_row           = Y_day_row
        self.sample_size         = sample_size
        self.random_seed         = random_seed
        self.T                   = T_BARS
        self.n_players           = len(DAY_FEATURE_NAMES)
        self.player_names        = list(DAY_FEATURE_NAMES)

        self.coalitions = np.array(
            list(itertools.product([False, True], repeat=self.n_players)),
            dtype=bool,
        )
        self.n_coalitions = len(self.coalitions)
        self._idx = {
            tuple(c): i for i, c in enumerate(self.coalitions)
        }
        self.values = None

    def _impute(self, coalition):
        rng = np.random.default_rng(self.random_seed)
        idx = rng.integers(0, len(self.X_background), size=self.sample_size)
        X   = self.X_background[idx].copy()
        for p in range(self.n_players):
            if coalition[p]:
                X[:, p] = self.x_explain[p]
        return X

    def value_function(self, coalition):
        X    = self._impute(coalition)
        Y_pred = functional_predict(
            self.model, X, self.model_feature_cols, t_grid
        )   # (sample_size, T)

        if self.game_type == 'prediction':
            return Y_pred.mean(axis=0)

        elif self.game_type == 'sensitivity':
            return Y_pred.var(axis=0)

        elif self.game_type == 'risk':
            # E[(Y_obs(t) - f(x)(t))^2]  over marginalised samples
            residuals = (self.Y_day_row[None, :] - Y_pred) ** 2
            return residuals.mean(axis=0)

    def precompute(self):
        print('  [{}] Evaluating {} coalitions x {} samples x {} bars ...'.format(
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
# 9.  Mobius transform + Shapley values  (unchanged)
# ===========================================================================

def functional_moebius_transform(game):
    p     = game.n_players
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(p), r)
        for r in range(p + 1)
    ))
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
# 10.  Kernels
# ===========================================================================

def kernel_identity(t):
    return np.eye(len(t))

def kernel_gaussian(t, sigma=6.0):
    d = t[:, None] - t[None, :]
    return np.exp(-0.5 * (d / sigma) ** 2)

def kernel_ou(t, length_scale=8.0):
    return np.exp(-np.abs(t[:, None] - t[None, :]) / length_scale)

def kernel_causal(t, length_scale=8.0):
    """
    One-sided exponential: smoothed value at t is a weighted average
    of raw effects at s <= t (backward-looking / causal).
    K(t, s) = exp(-(t-s)/ls) if t >= s else 0.
    Row i accumulates only past bars j <= i, so no future information
    leaks backward. Appropriate for features with a known release time.
    """
    d = t[:, None] - t[None, :]   # d[i,j] = t[i] - t[j]
    return np.where(d >= 0, np.exp(-d / length_scale), 0.0)

def kernel_output_correlation(Y_day):
    """
    Correlation kernel estimated from observed outputs Y_day (n_days, T).
    Uses empirical cross-bar correlations of realised intraday volatility,
    independent of any model.
    """
    C   = np.cov(Y_day.T)   # (T, T)
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    K   = np.clip(C / np.outer(std, std), -1.0, 1.0)
    print(
        '  Output correlation kernel: off-diag mean={:.3f}  '
        'min={:.3f}  max={:.3f}'.format(
            (K - np.eye(T_BARS)).mean(), K.min(), K.max()
        )
    )
    return K


# ===========================================================================
# 11.  Explanation helpers  (unchanged)
# ===========================================================================

def _normalize_kernel(K, dt):
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return K / rs

def apply_kernel(effect, K, dt):
    return _normalize_kernel(K, dt) @ effect * dt

def integrated_explanation(effect, K, dt):
    return float(
        np.sum(_normalize_kernel(K, dt) @ effect) * dt ** 2
    )


# ===========================================================================
# 12.  Plotting helpers  (unchanged)
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
        'sensitivity': 'Effect on Var[f(x)] (%^2)',
        'risk'       : 'Effect on MSE (%^2)',
    }[game_type]

def _game_title_suffix(game_type):
    return {
        'prediction' : 'Prediction game',
        'sensitivity': 'Sensitivity game',
        'risk'       : 'Risk (MSE) game',
    }[game_type]


# ===========================================================================
# 13.  Plots
# ===========================================================================

def plot_diurnal_and_trajectory(diurnal_mean, game, x_explain):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(
        'Diurnal volatility baseline and explained-day trajectory',
        fontsize=11, fontweight='bold',
    )

    ax = axes[0]
    ax.plot(t_grid, diurnal_mean * 100, color='steelblue', lw=2)
    ax.fill_between(t_grid, 0, diurnal_mean * 100,
                    alpha=0.2, color='steelblue')
    ax.set_ylabel('Mean |log-return| (%)')
    ax.set_title('Diurnal baseline  sigma_diurnal(t)', fontsize=10)
    _set_time_axis(ax)
    _period_shade(ax)

    ax2 = axes[1]
    ax2.plot(t_grid, game.grand_value * 100,
             color='#e05c2a', lw=2.5, label='F(x)(t) -- grand coalition')
    ax2.plot(t_grid, game.empty_value * 100,
             color='gray', lw=1.5, ls='--',
             label='f0(t) -- empty coalition')
    ax2.fill_between(
        t_grid, game.empty_value * 100, game.grand_value * 100,
        alpha=0.2, color='#e05c2a',
    )
    ax2.axhline(0, color='gray', lw=0.8, ls=':')
    ax2.set_ylabel('Diurnal-adjusted vol (%)')
    ax2.legend(fontsize=8)
    ax2.set_title('Predicted trajectory  F(x) in R^78', fontsize=10)
    _set_time_axis(ax2)
    _period_shade(ax2)

    feat_str = '  |  '.join(
        '{}={:.2f}'.format(n, x_explain[j])
        for j, n in enumerate(DAY_FEATURE_NAMES)
    )
    fig.text(0.5, -0.02, feat_str, ha='center', fontsize=7, color='gray')
    plt.tight_layout()
    return fig


def plot_grand_vs_empty(game):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    scale   = 100 if game.game_type == 'prediction' else 1e4
    unit    = '%' if game.game_type == 'prediction' else '(%^2 x 1e4)'
    ax.plot(t_grid, game.grand_value * scale,
            color='#e05c2a', lw=2.5,
            label='Grand coalition  v(N)(t)')
    ax.plot(t_grid, game.empty_value * scale,
            color='gray', lw=1.8, ls='--',
            label='Empty coalition  v(empty)(t)')
    gap = (game.grand_value - game.empty_value) * scale
    ax.fill_between(
        t_grid, game.empty_value * scale, game.grand_value * scale,
        alpha=0.25, color='#e05c2a',
        label='Gap  mean={:.4f}  max={:.4f}'.format(
            gap.mean(), gap.max()),
    )
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    _period_shade(ax)
    _set_time_axis(ax)
    ax.set_ylabel('Value ({})'.format(unit))
    ax.set_title(
        'Grand vs empty coalition -- {} -- gap is what Mobius must recover'
        .format(_game_title_suffix(game.game_type)),
        fontsize=10,
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_main_effects(moebius, game, top_k=5):
    pnames = game.player_names
    scale  = 100 if game.game_type == 'prediction' else 1e4
    imps   = {
        i: float(np.sum(np.abs(moebius[(i,)])))
        for i in range(game.n_players)
    }
    top    = sorted(imps, key=imps.get, reverse=True)[:top_k]
    colors = get_colors(top_k)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(
        'Top-{} feature main effects  m(xi)(t)  [{}]'.format(
            top_k, _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold',
    )

    ax = axes[0]
    for rank, fi in enumerate(top):
        ax.plot(t_grid, moebius[(fi,)] * scale,
                color=colors[rank], lw=2, label=pnames[fi])
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    _period_shade(ax)
    _set_time_axis(ax)
    ax.set_ylabel(_game_ylabel(game.game_type))
    ax.legend(fontsize=7)
    ax.set_title('Main-effect curves  m({i})(t)', fontsize=10)

    ax2 = axes[1]
    sp      = sorted(imps.items(), key=lambda x: abs(x[1]), reverse=True)
    fi_list = [f for f, _ in sp]
    fv_list = [imps[f] for f in fi_list]
    fn_list = [pnames[f] for f in fi_list]
    ax2.barh(
        range(len(fn_list)),
        [v * scale for v in fv_list],
        color=get_colors(len(fn_list)),
        alpha=0.8,
    )
    ax2.set_yticks(range(len(fn_list)))
    ax2.set_yticklabels(fn_list, fontsize=9)
    ax2.set_xlabel('Integrated |m({i})(t)|')
    ax2.set_title('Feature importance (Identity kernel)', fontsize=10)
    plt.tight_layout()
    return fig


def plot_shapley_curves(shapley, game, kernels):
    pnames = game.player_names
    colors = get_colors(game.n_players)
    nk     = len(kernels)
    scale  = 100 if game.game_type == 'prediction' else 1e4

    fig, axes = plt.subplots(1, nk, figsize=(3.5 * nk, 4))
    if nk == 1:
        axes = [axes]
    fig.suptitle(
        'Shapley value curves  phi_i(t)  [{}]'.format(
            _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold',
    )

    for k_idx, (k_name, K) in enumerate(kernels.items()):
        ax = axes[k_idx]
        for i in range(game.n_players):
            phi_t = apply_kernel(shapley[i], K, dt) * scale
            ax.plot(t_grid, phi_t,
                    color=colors[i], lw=1.8, label=pnames[i])
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        ax.set_title(k_name, fontsize=9, fontweight='bold')
        ax.set_xlabel('Time', fontsize=8)
        if k_idx == 0:
            ax.set_ylabel(_game_ylabel(game.game_type), fontsize=8)
            ax.legend(fontsize=7)
    plt.tight_layout()
    return fig


def plot_kernel_comparison(moebius, game, kernels, top_k=4):
    pnames = game.player_names
    imps   = {
        i: float(np.sum(np.abs(moebius[(i,)])))
        for i in range(game.n_players)
    }
    top    = sorted(imps, key=imps.get, reverse=True)[:top_k]
    colors = get_colors(top_k)
    nk     = len(kernels)
    scale  = 100 if game.game_type == 'prediction' else 1e4

    fig, axes = plt.subplots(
        top_k, nk,
        figsize=(3.2 * nk, 3.0 * top_k),
        sharey='row',
    )
    fig.suptitle(
        'Kernel-weighted main effects  (K * m_i)(t)  [{}]'.format(
            _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold',
    )

    for k_idx, (k_name, K) in enumerate(kernels.items()):
        for f_idx, feat_idx in enumerate(top):
            ax = axes[f_idx, k_idx]
            ax.plot(
                t_grid,
                apply_kernel(moebius[(feat_idx,)], K, dt) * scale,
                color=colors[f_idx], lw=1.8,
            )
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            ax.set_xlim(0, T_BARS - 1)
            ax.set_xticks(XTICK_IDXS[::2])
            ax.set_xticklabels(
                XTICK_LABELS[::2], rotation=45, ha='right', fontsize=6
            )
            if f_idx == 0:
                ax.set_title(k_name, fontsize=9, fontweight='bold')
            if k_idx == 0:
                ax.set_ylabel(pnames[feat_idx], fontsize=8)
            if f_idx == top_k - 1:
                ax.set_xlabel('Time', fontsize=7)
    plt.tight_layout()
    return fig


def plot_local_explanations(moebius, game, kernels,
                             bars_of_interest=None):
    if bars_of_interest is None:
        bars_of_interest = [2, 24, 60]

    pnames    = game.player_names
    n_players = game.n_players
    n_bars    = len(bars_of_interest)
    nk        = len(kernels)
    colors    = get_colors(n_players)
    scale     = 100 if game.game_type == 'prediction' else 1e4

    fig, axes = plt.subplots(
        n_bars, nk, figsize=(3.2 * nk, 3.2 * n_bars)
    )
    fig.suptitle(
        'Local explanations  (K * m_i)(t0)  at selected bars  [{}]'.format(
            _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold',
    )

    bar_wall = {2: '09:40', 24: '11:30', 60: '14:30'}
    for h_idx, t0 in enumerate(bars_of_interest):
        for k_idx, (k_name, K) in enumerate(kernels.items()):
            ax  = axes[h_idx, k_idx]
            lv  = [
                float(apply_kernel(moebius[(f,)], K, dt)[int(t0)])
                for f in range(n_players)
            ]
            tri = sorted(
                zip(pnames, lv, colors),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            ns, vs, cs = zip(*tri)
            bc = [c if v >= 0 else '#c0392b' for v, c in zip(vs, cs)]
            ax.barh(range(len(ns)), [v * scale for v in vs],
                    color=bc, alpha=0.85)
            ax.set_yticks(range(len(ns)))
            ax.set_yticklabels(ns, fontsize=7)
            ax.axvline(0, color='gray', lw=0.8, ls=':')
            ax.set_xlabel(_game_ylabel(game.game_type), fontsize=7)
            if h_idx == 0:
                ax.set_title(k_name, fontsize=9, fontweight='bold')
            if k_idx == 0:
                ax.set_ylabel(
                    't = {}'.format(bar_wall.get(t0, str(t0))),
                    fontsize=9, fontweight='bold',
                )
    plt.tight_layout()
    return fig


def plot_profiles_comparison(profiles, kernels, game_type='prediction'):
    K_id  = kernels['Identity']
    n     = len(profiles)
    scale = 100 if game_type == 'prediction' else 1e4
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        'Shapley value curves  phi_i(t)  across day profiles  [{}]'.format(
            _game_title_suffix(game_type)),
        fontsize=11, fontweight='bold',
    )

    for ax, (label, (gp, shap, pnames)) in zip(axes, profiles.items()):
        colors = get_colors(gp.n_players)
        order  = sorted(
            range(gp.n_players),
            key=lambda i: float(np.sum(np.abs(shap[i]))),
            reverse=True,
        )
        for rank, fi in enumerate(order[:5]):
            curve = apply_kernel(shap[fi], K_id, dt) * scale
            ax.plot(t_grid, curve,
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
    scale     = 100 if game.game_type == 'prediction' else 1e4
    pair_imps = {
        (i, j): float(np.sum(np.abs(
            moebius.get((i, j), np.zeros(T_BARS))
        )))
        for i in range(np_) for j in range(i + 1, np_)
    }
    if not pair_imps:
        return None

    top_list = sorted(pair_imps, key=pair_imps.get, reverse=True)[:top_pairs]
    colors   = get_colors(top_pairs)
    nk       = len(kernels)

    fig, axes = plt.subplots(
        top_pairs, nk,
        figsize=(3.2 * nk, 3.0 * top_pairs),
        sharey='row',
    )
    if top_pairs == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(
        'Pairwise interaction effects  m({{xi, xj}})(t)  [{}]'.format(
            _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold',
    )

    for p_idx, S in enumerate(top_list):
        i, j  = S
        label = '{}  x  {}'.format(pnames[i], pnames[j])
        for k_idx, (k_name, K) in enumerate(kernels.items()):
            ax = axes[p_idx, k_idx]
            ax.plot(
                t_grid,
                apply_kernel(
                    moebius.get(S, np.zeros(T_BARS)), K, dt
                ) * scale,
                color=colors[p_idx], lw=2,
            )
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            ax.set_xlim(0, T_BARS - 1)
            ax.set_xticks(XTICK_IDXS[::2])
            ax.set_xticklabels(
                XTICK_LABELS[::2], rotation=45, ha='right', fontsize=6
            )
            if p_idx == 0:
                ax.set_title(k_name, fontsize=9, fontweight='bold')
            if k_idx == 0:
                ax.set_ylabel(label, fontsize=8)
            if p_idx == top_pairs - 1:
                ax.set_xlabel('Time', fontsize=7)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# NEW PLOTS
# ---------------------------------------------------------------------------

def plot_ann_indicator_causal(moebius, game, kernels_standard,
                               causal_length_scales=(4, 8, 16)):
    """
    Separate figure for ann_indicator only, comparing:
      - all standard kernels (Identity, Gaussian, OU, Correlation)
      - causal kernel at multiple length scales
    """
    ann_idx = DAY_FEATURE_NAMES.index('ann_indicator')
    raw     = moebius[(ann_idx,)]
    scale   = 100 if game.game_type == 'prediction' else 1e4

    causal_kernels = {
        'Causal ls={}'.format(ls): kernel_causal(t_grid, ls)
        for ls in causal_length_scales
    }
    all_kernels = dict(kernels_standard)
    all_kernels.update(causal_kernels)
    nk = len(all_kernels)

    fig, axes = plt.subplots(
        1, nk, figsize=(3.2 * nk, 3.8), sharey=True
    )
    fig.suptitle(
        'ann_indicator main effect  m(ann)(t) -- kernel comparison\n'
        'including causal kernel  [{}]'.format(
            _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold',
    )

    causal_colors = plt.get_cmap('Oranges')(
        np.linspace(0.4, 0.9, len(causal_length_scales))
    )

    for k_idx, (k_name, K) in enumerate(all_kernels.items()):
        ax     = axes[k_idx]
        is_cau = k_name.startswith('Causal')
        c_idx  = list(causal_kernels.keys()).index(k_name) \
                 if is_cau else 0
        color  = causal_colors[c_idx] if is_cau else 'steelblue'

        ax.plot(t_grid, apply_kernel(raw, K, dt) * scale,
                color=color, lw=2)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.axvline(54, color='#888', lw=0.8, ls='--', alpha=0.5,
                   label='14:00')
        _period_shade(ax)
        _set_time_axis(ax)
        ax.set_title(k_name, fontsize=9, fontweight='bold')
        ax.set_xlabel('Time', fontsize=8)
        if k_idx == 0:
            ax.set_ylabel(_game_ylabel(game.game_type), fontsize=8)

    plt.tight_layout()
    return fig


def plot_kernel_hyperparams(moebius, game, top_k=4):
    """
    Hyperparameter sweep for Gaussian, OU, and causal kernels.
    Identity kernel is shown as a reference in each panel.
    Each feature gets one row; curves are overlaid per kernel family.
    """
    pnames = game.player_names
    imps   = {
        i: float(np.sum(np.abs(moebius[(i,)])))
        for i in range(game.n_players)
    }
    top    = sorted(imps, key=imps.get, reverse=True)[:top_k]
    scale  = 100 if game.game_type == 'prediction' else 1e4

    gauss_sigmas  = [3, 6, 12]
    ou_scales     = [4, 8, 16]
    causal_scales = [4, 8, 16]

    gauss_colors  = plt.get_cmap('Blues')(np.linspace(0.4, 0.9, len(gauss_sigmas)))
    ou_colors     = plt.get_cmap('Oranges')(np.linspace(0.4, 0.9, len(ou_scales)))
    causal_colors = plt.get_cmap('Greens')(np.linspace(0.4, 0.9, len(causal_scales)))
    identity_color = 'black'

    # 4 columns: Identity (reference), Gaussian, OU, Causal
    fig, axes = plt.subplots(
        top_k, 4,
        figsize=(17, 3.0 * top_k),
        sharey='row',
    )
    fig.suptitle(
        'Kernel hyperparameter sweep  [{}]  '
        '(Identity shown as reference in each panel)'.format(
            _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold',
    )

    col_titles = ['Identity (ref)', 'Gaussian', 'OU', 'Causal']
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=10, fontweight='bold')

    K_id = kernel_identity(t_grid)

    for row, feat_idx in enumerate(top):
        raw = moebius[(feat_idx,)]
        id_curve = apply_kernel(raw, K_id, dt) * scale

        # Col 0: Identity reference only
        ax = axes[row, 0]
        ax.plot(t_grid, id_curve, color=identity_color, lw=2, label='Identity')
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        ax.set_ylabel(pnames[feat_idx], fontsize=9)
        if row == 0:
            ax.legend(fontsize=7)

        # Col 1: Gaussian + identity reference
        ax = axes[row, 1]
        ax.plot(t_grid, id_curve, color=identity_color, lw=1,
                ls='--', alpha=0.4, label='Identity')
        for ci, sig in enumerate(gauss_sigmas):
            K = kernel_gaussian(t_grid, sigma=sig)
            ax.plot(t_grid, apply_kernel(raw, K, dt) * scale,
                    color=gauss_colors[ci], lw=2,
                    label='sigma={}'.format(sig))
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        if row == 0:
            ax.legend(fontsize=7)

        # Col 2: OU + identity reference
        ax = axes[row, 2]
        ax.plot(t_grid, id_curve, color=identity_color, lw=1,
                ls='--', alpha=0.4, label='Identity')
        for ci, ls_val in enumerate(ou_scales):
            K = kernel_ou(t_grid, length_scale=ls_val)
            ax.plot(t_grid, apply_kernel(raw, K, dt) * scale,
                    color=ou_colors[ci], lw=2,
                    label='ls={}'.format(ls_val))
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        if row == 0:
            ax.legend(fontsize=7)

        # Col 3: Causal + identity reference
        ax = axes[row, 3]
        ax.plot(t_grid, id_curve, color=identity_color, lw=1,
                ls='--', alpha=0.4, label='Identity')
        for ci, ls_val in enumerate(causal_scales):
            K = kernel_causal(t_grid, length_scale=ls_val)
            ax.plot(t_grid, apply_kernel(raw, K, dt) * scale,
                    color=causal_colors[ci], lw=2,
                    label='ls={}'.format(ls_val))
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        if row == 0:
            ax.legend(fontsize=7)

        for col in range(4):
            if row == top_k - 1:
                axes[row, col].set_xlabel('Time', fontsize=7)

    plt.tight_layout()
    return fig


def plot_feature_specific_kernels(moebius, game, K_causal, K_ou, top_k=4):
    """
    Feature-specific kernel application:
      ann_indicator  -> causal kernel  (respects 14:00 release timing)
      all others     -> OU kernel      (symmetric smoothing)
    Shows all top_k features on the same axes for direct comparison.
    """
    pnames = game.player_names
    imps   = {
        i: float(np.sum(np.abs(moebius[(i,)])))
        for i in range(game.n_players)
    }
    top    = sorted(imps, key=imps.get, reverse=True)[:top_k]
    scale  = 100 if game.game_type == 'prediction' else 1e4
    colors = get_colors(top_k)
    ann_idx = DAY_FEATURE_NAMES.index('ann_indicator')

    fig, axes = plt.subplots(1, top_k, figsize=(4.5 * top_k, 4), sharey=False)
    fig.suptitle(
        'Feature-specific kernel application  [{}]\n'
        'ann_indicator: causal kernel  |  all others: OU kernel'.format(
            _game_title_suffix(game.game_type)),
        fontsize=11, fontweight='bold',
    )

    for rank, feat_idx in enumerate(top):
        raw    = moebius[(feat_idx,)]
        K      = K_causal if feat_idx == ann_idx else K_ou
        k_name = 'Causal' if feat_idx == ann_idx else 'OU'
        curve  = apply_kernel(raw, K, dt) * scale

        ax = axes[rank]
        ax.plot(t_grid, curve, color=colors[rank], lw=2)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        ax.set_title(
            '{}\n({})'.format(pnames[feat_idx], k_name),
            fontsize=9, fontweight='bold',
        )
        ax.set_xlabel('Time', fontsize=8)
        if rank == 0:
            ax.set_ylabel(_game_ylabel(game.game_type), fontsize=8)

    plt.tight_layout()
    return fig


# ===========================================================================
# 14.  Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  Functional Explanation -- Intraday SPY Volatility')
    print('=' * 60)

    _require_dir(DATA_DIR)
    for d in PLOT_DIRS.values():
        _require_dir(d)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print('\n[1] Loading data ...')
    (X_day, Y_day, Y_adj,
     diurnal_mean, df_long) = load_and_aggregate()
    X_day_np = X_day.to_numpy().astype(float)

    # ── 2. Model ──────────────────────────────────────────────────────────
    print('\n[2] Training model ...')
    (model, feature_cols,
     X_train, X_test,
     y_train, y_test) = train_model(df_long)

    # ── 3. Profiles ───────────────────────────────────────────────────────
    print('\n[3] Selecting day profiles ...')

    vix_col = DAY_FEATURE_NAMES.index('vix_prev')
    vix_p25 = float(np.percentile(X_day_np[:, vix_col], 25))
    vix_p75 = float(np.percentile(X_day_np[:, vix_col], 75))

    def find_day_profile(conditions, pool):
        mask = np.ones(len(pool), dtype=bool)
        for feat, (lo, hi) in conditions.items():
            ci    = DAY_FEATURE_NAMES.index(feat)
            mask &= (pool[:, ci] >= lo) & (pool[:, ci] <= hi)
        hits = pool[mask]
        if len(hits) == 0:
            raise RuntimeError(
                'No trading day matches profile: {}'.format(conditions)
            )
        print('    {} matching days; picking median row.'.format(len(hits)))
        return hits[len(hits) // 2]

    x_p1 = find_day_profile(
        {'ann_indicator': (0.9, 1.1), 'vix_prev': (vix_p75, 999),
         'trailing_rv': (1e-4, 999)},
        X_day_np,
    )
    x_p2 = find_day_profile(
        {'ann_indicator': (-0.1, 0.1), 'vix_prev': (0, vix_p25)},
        X_day_np,
    )
    x_p3 = find_day_profile(
        {'day_of_week': (-0.1, 0.1)},
        X_day_np,
    )

    # Find observed Y row for each profile (for risk game)
    def _find_y_row(x_profile):
        diffs = np.abs(X_day_np - x_profile[None, :]).sum(axis=1)
        idx   = int(np.argmin(diffs))
        return Y_adj[idx]

    y_p1 = _find_y_row(x_p1)
    y_p2 = _find_y_row(x_p2)
    y_p3 = _find_y_row(x_p3)

    profile_defs = [
        ('High-VIX Announcement', x_p1, y_p1),
        ('Quiet Low-VIX',         x_p2, y_p2),
        ('Monday Gap',            x_p3, y_p3),
    ]

    for lbl, xp, _ in profile_defs:
        desc = '  '.join(
            '{}={:.4f}'.format(n, xp[j])
            for j, n in enumerate(DAY_FEATURE_NAMES)
        )
        print('  {}:\n    {}'.format(lbl, desc))

    # ── 4. Kernels ────────────────────────────────────────────────────────
    print('\n[4] Building kernels ...')
    K_corr  = kernel_output_correlation(Y_day)
    K_ou    = kernel_ou(t_grid, length_scale=8.0)
    K_causal = kernel_causal(t_grid, length_scale=8.0)
    kernels = {
        'Identity'   : kernel_identity(t_grid),
        'Gaussian'   : kernel_gaussian(t_grid, sigma=6.0),
        'OU'         : K_ou,
        'Correlation': K_corr,
    }

    # ── 5. Games loop ─────────────────────────────────────────────────────

    def save(fig, game_type, name):
        path = os.path.join(PLOT_DIRS[game_type], name)
        fig.savefig(path, bbox_inches='tight', dpi=150)
        print('  Saved: {}'.format(path))
        plt.close(fig)

    game_types = ('prediction', 'sensitivity', 'risk')

    for game_type in game_types:
        print('\n' + '=' * 60)
        print('  GAME TYPE: {}'.format(game_type.upper()))
        print('=' * 60)

        all_profiles = {}

        for prof_label, x_explain, y_explain in profile_defs:
            print('\n  Profile: {}  game: {}'.format(prof_label, game_type))

            # Build game
            game = IntradayFunctionalGame(
                model              = model,
                X_background       = X_day_np,
                x_explain          = x_explain,
                model_feature_cols = feature_cols,
                game_type          = game_type,
                Y_day_row          = y_explain,
                sample_size        = SAMPLE_SIZE[game_type],
                random_seed        = RNG_SEED,
            )
            game.precompute()

            gap_max = np.abs(game.grand_value - game.empty_value).max()
            print('  Grand-empty gap max = {:.4e}'.format(gap_max))

            # Mobius + Shapley
            moebius = functional_moebius_transform(game)
            shapley = shapley_from_moebius(moebius, game.n_players)

            # Slug for filenames
            slug = prof_label.lower().replace(' ', '_').replace('-', '')

            # Plots for Profile 1 only: diurnal/trajectory, grand_vs_empty,
            # main_effects, shapley_curves, interaction_effects,
            # ann_indicator causal, hyperparameter sweep
            if prof_label == 'High-VIX Announcement':
                if game_type == 'prediction':
                    save(
                        plot_diurnal_and_trajectory(
                            diurnal_mean, game, x_explain),
                        game_type,
                        'diurnal_and_trajectory.pdf',
                    )

                save(
                    plot_grand_vs_empty(game),
                    game_type,
                    'grand_vs_empty_{}.pdf'.format(slug),
                )
                save(
                    plot_main_effects(moebius, game),
                    game_type,
                    'main_effects_{}.pdf'.format(slug),
                )
                save(
                    plot_shapley_curves(shapley, game, kernels),
                    game_type,
                    'shapley_curves_{}.pdf'.format(slug),
                )
                fig_int = plot_interaction_effects(
                    moebius, game, kernels, top_pairs=3
                )
                if fig_int is not None:
                    save(fig_int, game_type,
                         'interaction_effects_{}.pdf'.format(slug))

                save(
                    plot_ann_indicator_causal(
                        moebius, game, kernels,
                        causal_length_scales=(4, 8, 16),
                    ),
                    game_type,
                    'ann_indicator_causal_{}.pdf'.format(slug),
                )
                save(
                    plot_kernel_hyperparams(moebius, game, top_k=4),
                    game_type,
                    'kernel_hyperparams_{}.pdf'.format(slug),
                )
                save(
                    plot_feature_specific_kernels(
                        moebius, game,
                        K_causal=K_causal,
                        K_ou=K_ou,
                        top_k=4,
                    ),
                    game_type,
                    'feature_specific_kernels_{}.pdf'.format(slug),
                )

            # Kernel comparison + local explanations for ALL profiles
            save(
                plot_kernel_comparison(moebius, game, kernels, top_k=4),
                game_type,
                'kernel_comparison_{}.pdf'.format(slug),
            )
            save(
                plot_local_explanations(
                    moebius, game, kernels,
                    bars_of_interest=[2, 24, 60],
                ),
                game_type,
                'local_explanations_{}.pdf'.format(slug),
            )

            all_profiles[prof_label] = (game, shapley, game.player_names)

        # Cross-profile comparison
        save(
            plot_profiles_comparison(
                all_profiles, kernels, game_type=game_type
            ),
            game_type,
            'profiles_comparison.pdf',
        )

    print('\n' + '=' * 60)
    print('  Done.  Plots saved under plots/intraday/')
    print('=' * 60)
    
    
    
    
   