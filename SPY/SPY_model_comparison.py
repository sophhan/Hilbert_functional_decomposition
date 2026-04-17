"""
SPY Intraday Volatility — Multi-Model Comparison
=================================================
Compares five genuinely multivariate models (no t as input):

  M1: Random Forest           (sklearn MultiOutput, direct X->R^78)
  M2: Random Forest + PCA     (RF on K PCA scores, reconstruct)
  M3: MLP                     (bottleneck MLP, direct X->R^78)
  M4: MLP + PCA               (MLP on K PCA scores, reconstruct)
  M5: NGBoost + PCA           (MultivariateNormal on K scores, Section 4.5)

Note: NGBoost direct (78 independent Normal models) is excluded because
fitting 78 × NGBRegressor(n_est=150) takes ~60s per game evaluation call,
making the 64-coalition × 3-game × 3-profile loop impractical (~25h).
RF and MLP direct are included because their inference is fast enough
(~0.2s per coalition eval) to make the game loop feasible.

Scope: focused on the High-VIX Announcement profile only.
       Three game types: prediction, sensitivity, risk.
       Three output figures per game type:
         1. main_effects.pdf       — m(xi)(t) identity kernel
         2. kernel_comparison.pdf  — top-4 features × 4 kernels
         3. profiles_comparison.pdf — Shapley curves, all models side-by-side

Data   : SPY 5-min bars (Polygon.io cache) + VIX (yfinance)
Outputs: plots/intraday_model_comparison/{prediction,sensitivity,risk}/
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
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

RNG_SEED      = 42
T_BARS        = 78
dt            = 1.0

# Model hyperparameters
N_PCA         = 8
RF_N_EST      = 300
RF_JOBS       = -1         # use all cores for RF
MLP_EPOCHS    = 800
MLP_LR        = 1e-3
MLP_PATIENCE  = 60
MLP_DEVICE    = 'cpu'      # change to 'cuda' if available
NGB_N_EST     = 300
NGB_LR        = 0.05

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

BASE_PLOT_DIR = os.path.join('Hilbert_functional_decomposition','plots', 'SPY_intraday_model_comparison')
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

# Model display names and colours
MODEL_TAGS = ['rf', 'rf_pca', 'mlp', 'mlp_pca', 'ngboost_pca']

MODEL_LABELS = {
    'rf':          'Random Forest',
    'rf_pca':      'Random Forest + PCA',
    'mlp':         'MLP',
    'mlp_pca':     'MLP + PCA',
    'ngboost_pca': 'NGBoost + PCA',
}

MODEL_COLORS = {
    'rf':          '#457b9d',
    'rf_pca':      '#1d3557',
    'mlp':         '#f4a261',
    'mlp_pca':     '#e76f51',
    'ngboost_pca': '#2a9d8f',
}

MODEL_LS = {
    'rf':          '--',
    'rf_pca':      '-.',
    'mlp':         (0, (5, 2)),
    'mlp_pca':     (0, (1, 1)),
    'ngboost_pca': '-',
}

MODEL_MARKERS = {
    'rf':          's',
    'rf_pca':      'P',
    'mlp':         'v',
    'mlp_pca':     '*',
    'ngboost_pca': 'o',
}


# ===========================================================================
# 1.  Validation helpers
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
    if len(missing) > max(1, int(0.01 * len(trading_dates))):
        raise RuntimeError(
            'VIX missing for {}/{} dates.'.format(
                len(missing), len(trading_dates)))


# ===========================================================================
# 2.  Data loading (identical to spy_ngboost.py)
# ===========================================================================

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
    vix_start = (pd.Timestamp(first_date)
                 - pd.Timedelta(days=VIX_LOOKBACK_DAYS)
                 ).strftime('%Y-%m-%d')
    if os.path.isfile(VIX_CACHE_PATH):
        print('    Reading VIX cache ...')
        df = pd.read_csv(VIX_CACHE_PATH, index_col=0)
        return {str(k): float(v)
                for k, v in df['vix'].dropna().items()}
    import yfinance as yf
    end_ex = (pd.Timestamp(last_date)
              + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    raw    = yf.download('^VIX', start=vix_start,
                         end=end_ex, progress=False,
                         auto_adjust=True)
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
    raise RuntimeError(
        'No VIX within 14 days of {}'.format(date_str))


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
    pivot = pivot[pivot.notna().sum(axis=1) >= 70].fillna(
        pivot.mean())
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
        dt_obj    = pd.Timestamp(date_str)
        vix_prev  = _resolve_vix_prev(vix_dict, date_str)
        day_bars  = bars[
            bars['date'] == date_str].sort_values('bar_idx')
        prev_bars = bars[
            bars['date'] < date_str
        ].sort_values(['date', 'bar_idx'])
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
# 3.  Model classes — all map X: (N,6) -> Y_hat: (N,78), no t input
# ===========================================================================

# ---- PCA wrapper (shared by RF+PCA and MLP+PCA) --------------------------

class PCAWrapper:
    """
    Wraps any model with a predict(X)->(N,K) interface so that the
    full pipeline is:
      fit:     PCA(Y) -> scores;  base.fit(X, scores)
      predict: base.predict(X) -> scores;  pca.inverse_transform -> Y
    """
    def __init__(self, base, n_pca):
        self.base  = base
        self.n_pca = n_pca
        self.pca   = None

    def fit(self, X, Y):
        self.pca   = PCA(n_components=self.n_pca)
        scores     = self.pca.fit_transform(Y)
        self.base.fit(X, scores)
        return self

    def predict(self, X):
        scores = self.base.predict(X)
        return self.pca.inverse_transform(scores)

    @property
    def variance_explained(self):
        return float(self.pca.explained_variance_ratio_.sum()) \
               if self.pca else None


# ---- M1: Random Forest (direct) -----------------------------------------

class RFModel:
    """
    sklearn RandomForestRegressor with MultiOutput support.
    Fits a single forest predicting all T=78 outputs jointly.
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
        return self.model.predict(X)


# ---- M2: Random Forest + PCA --------------------------------------------

class RFPCAModel:
    """RF on PCA scores — uses PCAWrapper around MultiOutputRegressor."""
    def __init__(self, n_estimators=RF_N_EST, n_pca=N_PCA,
                 n_jobs=RF_JOBS, random_state=RNG_SEED):
        base = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
        self._wrapper = PCAWrapper(base, n_pca)

    def fit(self, X, Y):
        self._wrapper.fit(X, Y)
        return self

    def predict(self, X):
        return self._wrapper.predict(X)

    @property
    def variance_explained(self):
        return self._wrapper.variance_explained


# ---- MLP backbone --------------------------------------------------------

class _MLPNet(nn.Module):
    def __init__(self, p, hidden, out_dim):
        super().__init__()
        layers, in_d = [], p
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.Tanh()]
            in_d = h
        layers.append(nn.Linear(in_d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _fit_mlp(net, X, Y, lr, n_epochs, patience,
             batch_size, device_str):
    """Generic MLP training loop with early stopping."""
    device = torch.device(device_str)
    net    = net.to(device)
    opt    = optim.Adam(net.parameters(), lr=lr,
                        weight_decay=1e-4)
    sched  = optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=15, factor=0.5)
    loss_fn = nn.MSELoss()

    n     = len(X)
    n_val = max(2, int(0.15 * n))
    idx   = np.random.permutation(n)
    vi, ti = idx[:n_val], idx[n_val:]

    def _t(a):
        return torch.tensor(a, dtype=torch.float32).to(device)

    Xtr, Ytr = _t(X[ti]), _t(Y[ti])
    Xvl, Yvl = _t(X[vi]), _t(Y[vi])
    loader   = DataLoader(
        TensorDataset(Xtr, Ytr),
        batch_size=min(batch_size, len(ti)),
        shuffle=True,
    )

    best_val, best_state, no_imp = float('inf'), None, 0
    net.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(Xvl), Yvl).item()
        net.train()
        sched.step(vl)
        if vl < best_val - 1e-7:
            best_val   = vl
            best_state = {k: v.clone()
                          for k, v in net.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    return net


def _mlp_predict(net, X, device_str):
    device = torch.device(device_str)
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor(
            X, dtype=torch.float32).to(device))
    return out.cpu().numpy()


# ---- M3: MLP (direct) ---------------------------------------------------

class MLPModel:
    """
    Bottleneck MLP: X -> (hidden layers) -> R^T directly.
    Hidden architecture is fixed; bottleneck at 32 units.
    """
    def __init__(self, T=T_BARS, hidden=(64, 128, 64),
                 n_epochs=MLP_EPOCHS, lr=MLP_LR,
                 patience=MLP_PATIENCE,
                 batch_size=64, device=MLP_DEVICE,
                 random_state=RNG_SEED):
        torch.manual_seed(random_state)
        self.net      = _MLPNet(6, hidden, T)
        self.n_epochs = n_epochs
        self.lr       = lr
        self.patience = patience
        self.batch    = batch_size
        self.device   = device

    def fit(self, X, Y):
        self.net = _fit_mlp(
            self.net, X, Y,
            lr=self.lr, n_epochs=self.n_epochs,
            patience=self.patience,
            batch_size=self.batch,
            device_str=self.device,
        )
        return self

    def predict(self, X):
        return _mlp_predict(self.net, X, self.device)


# ---- M4: MLP + PCA ------------------------------------------------------

class MLPPCAModel:
    """MLP predicting K PCA scores, then reconstructed."""
    def __init__(self, n_pca=N_PCA, hidden=(64, 64),
                 n_epochs=MLP_EPOCHS, lr=MLP_LR,
                 patience=MLP_PATIENCE,
                 batch_size=64, device=MLP_DEVICE,
                 random_state=RNG_SEED):
        torch.manual_seed(random_state)
        base_net      = _MLPNet(6, hidden, n_pca)
        self._base_net = base_net
        self.n_pca    = n_pca
        self.n_epochs = n_epochs
        self.lr       = lr
        self.patience = patience
        self.batch    = batch_size
        self.device   = device
        self.pca      = None

    def fit(self, X, Y):
        self.pca = PCA(n_components=self.n_pca)
        scores   = self.pca.fit_transform(Y)
        self._base_net = _fit_mlp(
            self._base_net, X, scores,
            lr=self.lr, n_epochs=self.n_epochs,
            patience=self.patience,
            batch_size=self.batch,
            device_str=self.device,
        )
        return self

    def predict(self, X):
        scores = _mlp_predict(
            self._base_net, X, self.device)
        return self.pca.inverse_transform(scores)

    @property
    def variance_explained(self):
        return float(
            self.pca.explained_variance_ratio_.sum()
        ) if self.pca else None


# ---- M5: NGBoost + PCA --------------------------------------------------

class NGBoostPCAModel:
    """
    NGBoost with MultivariateNormal(K) on PCA scores.
    See spy_ngboost.py for full documentation.
    """
    def __init__(self, n_pca=N_PCA,
                 n_estimators=NGB_N_EST,
                 learning_rate=NGB_LR,
                 random_state=RNG_SEED):
        self.n_pca = n_pca
        self.n_est = n_estimators
        self.lr    = learning_rate
        self.rs    = random_state
        self.pca   = None
        self.ngb   = None

    def fit(self, X, Y):
        self.pca = PCA(n_components=self.n_pca)
        scores   = self.pca.fit_transform(Y)
        self.ngb = NGBRegressor(
            Dist          = MultivariateNormal(self.n_pca),
            n_estimators  = self.n_est,
            learning_rate = self.lr,
            random_state  = self.rs,
            verbose       = False,
        )
        self.ngb.fit(X, scores)
        return self

    def predict(self, X):
        scores = self.ngb.pred_dist(X).loc
        return self.pca.inverse_transform(scores)

    @property
    def variance_explained(self):
        return float(
            self.pca.explained_variance_ratio_.sum()
        ) if self.pca else None


# ===========================================================================
# 4.  Model factory
# ===========================================================================

def build_models():
    """
    Returns dict: tag -> unfitted model instance.
    All models share the same random state for comparability.
    """
    return {
        'rf':          RFModel(random_state=RNG_SEED),
        'rf_pca':      RFPCAModel(random_state=RNG_SEED),
        'mlp':         MLPModel(random_state=RNG_SEED),
        'mlp_pca':     MLPPCAModel(random_state=RNG_SEED),
        'ngboost_pca': NGBoostPCAModel(random_state=RNG_SEED),
    }


def fit_all_models(models, X_train, Y_train):
    """Fit all models, print timing per model."""
    for tag, model in models.items():
        t0 = time.time()
        model.fit(X_train, Y_train)
        elapsed = time.time() - t0
        var_exp = getattr(model, 'variance_explained', None)
        ve_str  = (f'  PCA var={var_exp:.3f}'
                   if var_exp is not None else '')
        print(f'  [{tag:15s}] fit: {elapsed:.1f}s{ve_str}')
    return models


# ===========================================================================
# 5.  Cooperative game  (prediction / sensitivity / risk)
# ===========================================================================

class FunctionalGame:
    """
    Cooperative game for a single (model, profile, game_type) triple.

    game_type:
      prediction  -- v(S)(t) = E[F(X)(t) | X_S]
      sensitivity -- v(S)(t) = Var[F(X)(t) | X_S]  (empirical over bg samples)
      risk        -- v(S)(t) = E[(Y_obs(t)-F(X)(t))^2 | X_S]
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

        self.T          = T_BARS
        self.n_players  = len(DAY_FEATURE_NAMES)
        self.player_names = list(DAY_FEATURE_NAMES)
        self.coalitions = np.array(
            list(itertools.product(
                [False, True], repeat=self.n_players)),
            dtype=bool)
        self.n_coalitions = len(self.coalitions)
        self._idx = {
            tuple(c): i
            for i, c in enumerate(self.coalitions)
        }
        self.values = None

    def _impute(self, coalition):
        rng = np.random.default_rng(self.random_seed)
        idx = rng.integers(
            0, len(self.X_background),
            size=self.sample_size)
        X   = self.X_background[idx].copy()
        for j in range(self.n_players):
            if coalition[j]:
                X[:, j] = self.x_explain[j]
        return X

    def value_function(self, coalition):
        X      = self._impute(coalition)
        Y_pred = self.predict_fn(X)   # (sample_size, T)
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
                print(f'    {i+1}/{self.n_coalitions} coalitions done.')

    def __getitem__(self, coalition):
        return self.values[self._idx[coalition]]

    @property
    def empty_value(self):
        return self[tuple([False] * self.n_players)]

    @property
    def grand_value(self):
        return self[tuple([True] * self.n_players)]


# ===========================================================================
# 6.  Möbius + Shapley
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
# 7.  Kernels + explanation helpers
# ===========================================================================

def kernel_identity(t):
    return np.eye(len(t))

def kernel_gaussian(t, sigma=6.0):
    d = t[:, None] - t[None, :]
    return np.exp(-0.5 * (d / sigma) ** 2)

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
# 8.  Plotting helpers
# ===========================================================================

XTICK_IDXS   = list(range(0, T_BARS, 6))
XTICK_LABELS = [BAR_LABELS[i] for i in XTICK_IDXS]


def _set_time_axis(ax):
    ax.set_xticks(XTICK_IDXS)
    ax.set_xticklabels(
        XTICK_LABELS, rotation=45, ha='right', fontsize=7)
    ax.set_xlim(-0.3, T_BARS - 0.7)


def _period_shade(ax):
    ax.axvspan(0,  6,  alpha=0.10, color='#ffd699', zorder=0)
    ax.axvspan(72, 78, alpha=0.10, color='#ffd699', zorder=0)


def _game_ylabel(game_type):
    return {
        'prediction' : 'Effect on vol (%)',
        'sensitivity': 'Var[F(t)] (%^2)',
        'risk'       : 'Effect on MSE (%^2)',
    }[game_type]


def _scale(game_type):
    return 100 if game_type == 'prediction' else 1e4


def savefig(fig, game_type, name):
    path = os.path.join(PLOT_DIRS[game_type], name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print('  Saved: {}'.format(path))
    plt.close(fig)


def get_top_features(moebius_dict, n_players, top_k=4):
    """
    Return indices of top_k features by mean integrated |m(xi)(t)|
    across all models, so the same features appear in every panel.
    """
    imps = np.zeros(n_players)
    for tag, moebius in moebius_dict.items():
        for i in range(n_players):
            imps[i] += float(np.sum(np.abs(moebius[(i,)])))
    return sorted(range(n_players),
                  key=lambda i: imps[i], reverse=True)[:top_k]


# ===========================================================================
# 9.  Figures
# ===========================================================================

def plot_main_effects(moebius_dict, game_type, pnames, top_k=5):
    """
    One panel per model.  Each panel: top_k feature main-effect curves
    under the identity kernel.
    Models are columns; all share the same y-scale per feature.
    """
    sc      = _scale(game_type)
    n_models = len(MODEL_TAGS)
    top      = get_top_features(
        moebius_dict, len(pnames), top_k=top_k)
    colors   = [plt.get_cmap('tab10')(i) for i in range(top_k)]

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(4.0 * n_models, 4.5),
        sharey=True,
    )
    fig.suptitle(
        'Main effects  m(xi)(t)  —  Identity kernel\n'
        f'High-VIX Announcement profile  [{game_type}]',
        fontsize=11, fontweight='bold',
    )

    for col, tag in enumerate(MODEL_TAGS):
        ax      = axes[col]
        moebius = moebius_dict[tag]
        for rank, fi in enumerate(top):
            ax.plot(
                t_grid,
                moebius[(fi,)] * sc,
                color=colors[rank],
                lw=2.0,
                label=pnames[fi],
            )
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        ax.set_title(
            MODEL_LABELS[tag], fontsize=9,
            fontweight='bold',
            color=MODEL_COLORS[tag],
        )
        ax.set_xlabel('Time', fontsize=8)
        if col == 0:
            ax.set_ylabel(_game_ylabel(game_type), fontsize=9)
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    return fig


def plot_kernel_comparison(moebius_dict, game_type,
                            pnames, kernels, top_k=4):
    """
    Grid: rows = top_k features,  columns = models × kernels.
    Each cell: (K m_i)(t) for one (model, kernel) combination.

    Layout: models are grouped in super-columns; within each
    super-column the four kernel columns are shown.
    For readability we limit to 2 kernels: Identity and OU.
    Full kernel sweep (Gaussian, Causal, Correlation) is
    available but would make the figure too wide for the paper.
    """
    sc       = _scale(game_type)
    top      = get_top_features(
        moebius_dict, len(pnames), top_k=top_k)
    row_colors = [plt.get_cmap('tab10')(i) for i in range(top_k)]

    # Use a focused 2-kernel comparison: Identity vs OU
    k_subset = {
        'Identity': kernels['Identity'],
        'OU':       kernels['OU'],
    }
    nk       = len(k_subset)
    n_models = len(MODEL_TAGS)
    n_cols   = nk * n_models

    fig, axes = plt.subplots(
        top_k, n_cols,
        figsize=(2.8 * n_cols, 2.8 * top_k),
        sharey='row',
    )
    fig.suptitle(
        'Kernel-weighted main effects  (K m_i)(t)\n'
        f'High-VIX Announcement profile  [{game_type}]  '
        '— Identity vs OU kernel',
        fontsize=11, fontweight='bold',
    )

    for m_idx, tag in enumerate(MODEL_TAGS):
        moebius = moebius_dict[tag]
        for k_idx, (k_name, K) in enumerate(k_subset.items()):
            col = m_idx * nk + k_idx
            # Column header on row 0
            axes[0, col].set_title(
                f'{MODEL_LABELS[tag]}\n{k_name}',
                fontsize=7.5, fontweight='bold',
                color=MODEL_COLORS[tag],
            )
            for row, fi in enumerate(top):
                ax = axes[row, col]
                ax.plot(
                    t_grid,
                    apply_kernel(moebius[(fi,)], K) * sc,
                    color=row_colors[row], lw=1.8,
                )
                ax.axhline(
                    0, color='gray', lw=0.5, ls=':')
                _period_shade(ax)
                ax.set_xlim(0, T_BARS - 1)
                ax.set_xticks(XTICK_IDXS[::2])
                ax.set_xticklabels(
                    XTICK_LABELS[::2],
                    rotation=45, ha='right', fontsize=5.5)
                if col == 0:
                    ax.set_ylabel(
                        pnames[fi], fontsize=8)
                if row == top_k - 1:
                    ax.set_xlabel('Time', fontsize=6)

    plt.tight_layout()
    return fig


def plot_profiles_comparison(shapley_dict, moebius_dict,
                              game_type, pnames, kernels,
                              top_k=4):
    """
    One panel per model (columns).
    Each panel shows the top_k Shapley curves under the OU kernel,
    so temporal smoothing is applied consistently across models.

    Below each Shapley panel: integrated importance bar chart
    (identity kernel) for direct cross-model comparison.
    """
    sc       = _scale(game_type)
    K_ou     = kernels['OU']
    K_id     = kernels['Identity']
    n_models = len(MODEL_TAGS)
    top      = get_top_features(
        moebius_dict, len(pnames), top_k=top_k)
    colors   = [plt.get_cmap('tab10')(i) for i in range(top_k)]

    fig, axes = plt.subplots(
        2, n_models,
        figsize=(4.2 * n_models, 8.5),
        gridspec_kw={'height_ratios': [3, 1.5]},
    )
    fig.suptitle(
        'Shapley curves (OU kernel) and integrated importance\n'
        f'High-VIX Announcement profile  [{game_type}]',
        fontsize=11, fontweight='bold',
    )

    for col, tag in enumerate(MODEL_TAGS):
        shapley = shapley_dict[tag]
        moebius = moebius_dict[tag]

        # Row 0: Shapley curves under OU kernel
        ax = axes[0, col]
        for rank, fi in enumerate(top):
            ax.plot(
                t_grid,
                apply_kernel(shapley[fi], K_ou) * sc,
                color=colors[rank],
                lw=2.0,
                label=pnames[fi],
            )
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        ax.set_title(
            MODEL_LABELS[tag], fontsize=9,
            fontweight='bold',
            color=MODEL_COLORS[tag],
        )
        if col == 0:
            ax.set_ylabel(
                _game_ylabel(game_type), fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('Time', fontsize=8)

        # Row 1: integrated importance (identity kernel)
        ax2 = axes[1, col]
        feat_order = sorted(
            range(len(pnames)),
            key=lambda i: abs(float(np.trapz(
                moebius[(i,)], t_grid))),
            reverse=True,
        )[:top_k]
        vals  = [
            float(np.trapz(moebius[(fi,)], t_grid)) * sc
            for fi in feat_order
        ]
        names = [pnames[fi] for fi in feat_order]
        bar_colors = [
            colors[top.index(fi)]
            if fi in top else '#cccccc'
            for fi in feat_order
        ]
        ax2.barh(
            range(len(names)), vals,
            color=bar_colors, alpha=0.85,
        )
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=7)
        ax2.axvline(0, color='gray', lw=0.8, ls=':')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(labelsize=7)
        if col == 0:
            ax2.set_ylabel(
                r'$\int m_i(t)\,dt$', fontsize=8)
        ax2.set_xlabel(_game_ylabel(game_type), fontsize=7)

    plt.tight_layout()
    return fig


# ===========================================================================
# 10.  Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  SPY Multi-Model Comparison  (RF / MLP / NGBoost+PCA)')
    print('=' * 60)

    for d in [DATA_DIR] + list(PLOT_DIRS.values()):
        _require_dir(d)

    def save(fig, game_type, name):
        savefig(fig, game_type, name)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print('\n[1] Loading data ...')
    X_day, Y_day, Y_adj, diurnal_mean = load_and_aggregate()
    X_day_np = X_day.to_numpy().astype(float)

    # ── 2. Train / test split and model fitting ───────────────────────────
    print('\n[2] Fitting models ...')
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_day_np, Y_adj,
        test_size=0.2, random_state=RNG_SEED,
    )

    models = build_models()
    fit_all_models(models, X_tr, Y_tr)

    # Quick R² diagnostics
    print('\n  R² on held-out test set (trajectory-level):')
    for tag, model in models.items():
        Y_pred = model.predict(X_te)
        ss_res = np.sum((Y_te - Y_pred) ** 2)
        ss_tot = np.sum((Y_te - Y_te.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot
        print(f'    {tag:15s}: R² = {r2:.4f}')

    # ── 3. Kernels ────────────────────────────────────────────────────────
    print('\n[3] Building kernels ...')
    K_corr  = kernel_output_correlation(Y_day)
    kernels = {
        'Identity'   : kernel_identity(t_grid),
        'Gaussian'   : kernel_gaussian(t_grid, sigma=6.0),
        'OU'         : kernel_ou(t_grid, length_scale=8.0),
        'Causal'     : kernel_causal(t_grid, length_scale=8.0),
        'Correlation': K_corr,
    }

    # ── 4. Select High-VIX Announcement profile ───────────────────────────
    print('\n[4] Selecting High-VIX Announcement profile ...')
    vix_col = DAY_FEATURE_NAMES.index('vix_prev')
    vix_p75 = float(np.percentile(X_day_np[:, vix_col], 75))

    ann_mask = (X_day_np[:, DAY_FEATURE_NAMES.index('ann_indicator')]
                > 0.5)
    vix_mask = X_day_np[:, vix_col] >= vix_p75
    hits     = X_day_np[ann_mask & vix_mask]
    if len(hits) == 0:
        raise RuntimeError(
            'No days match High-VIX Announcement criteria.')
    print(f'    {len(hits)} matching days; picking median.')
    x_explain = hits[len(hits) // 2]

    diffs   = np.abs(X_day_np - x_explain[None, :]).sum(axis=1)
    Y_obs   = Y_adj[int(np.argmin(diffs))]

    desc = '  '.join(
        f'{n}={x_explain[j]:.3f}'
        for j, n in enumerate(DAY_FEATURE_NAMES))
    print(f'    Profile: {desc}')

    # ── 5. Games loop ─────────────────────────────────────────────────────
    pnames = list(DAY_FEATURE_NAMES)

    for game_type in ('prediction', 'sensitivity', 'risk'):
        print(f'\n{"="*60}')
        print(f'  GAME: {game_type.upper()}')
        print(f'{"="*60}')

        sample_size = SAMPLE_SIZE[game_type]
        moebius_all = {}
        shapley_all = {}

        for tag, model in models.items():
            print(f'\n  [{tag}] game={game_type} ...')
            game = FunctionalGame(
                predict_fn   = model.predict,
                X_background = X_day_np,
                x_explain    = x_explain,
                game_type    = game_type,
                Y_obs        = Y_obs,
                sample_size  = sample_size,
                random_seed  = RNG_SEED,
            )
            game.precompute()
            moebius_all[tag] = functional_moebius_transform(game)
            shapley_all[tag] = shapley_from_moebius(
                moebius_all[tag], game.n_players)

        # ── Figures ───────────────────────────────────────────────────
        print(f'\n  Generating figures [{game_type}] ...')

        save(
            plot_main_effects(
                moebius_all, game_type, pnames, top_k=5),
            game_type,
            'main_effects_highvix.pdf',
        )

        save(
            plot_kernel_comparison(
                moebius_all, game_type, pnames,
                kernels, top_k=4),
            game_type,
            'kernel_comparison_highvix.pdf',
        )

        save(
            plot_profiles_comparison(
                shapley_all, moebius_all,
                game_type, pnames, kernels, top_k=4),
            game_type,
            'profiles_comparison_highvix.pdf',
        )

    print(f'\n{"="*60}')
    print(f'  Done.  Outputs in {BASE_PLOT_DIR}/')
    print(f'{"="*60}')