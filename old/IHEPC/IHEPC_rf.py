"""
Functional Explanation Framework -- UCI IHEPC Household Power Consumption
=========================================================================
Dataset : UCI Individual Household Electric Power Consumption (id=235)
          Fetched automatically via:
              pip install ucimlrepo
              from ucimlrepo import fetch_ucirepo
          On first run the script downloads the data (~125 MB) and caches
          it as a parquet file at ./data/ihepc/household_power_consumption.parquet
          for fast re-loading on subsequent runs.
          Requires:  pip install ucimlrepo pyarrow

Output  : 24-hour Global_active_power trajectory (kW, hourly means).
          T = 24 hours.

Model   : F^H : R^6 -> R^24
  RandomForestRegressor, direct multi-output (no PCA, no t as input).
  Six day-level features, all known before the day begins:
    day_of_week     integer 0 (Mon) - 6 (Sun)
    is_weekend      binary: 1 if Sat or Sun
    month           integer 1-12 (seasonal patterns)
    season          integer 1 (winter) - 4 (autumn)
    lag_daily_mean  previous day's mean Global_active_power (kW)
    lag_morning     previous day's 06:00-09:00 mean (kW)

Why the correlation kernel is the central result here
------------------------------------------------------
Household electricity demand has a bimodal daily curve: a morning peak
(~07:00-09:00, kettle / shower / heating startup) and an evening peak
(~18:00-21:00, cooking / lighting / evening heating). These two peaks
co-vary strongly across days because both reflect the same household's
activity regime: a cold winter day elevates both peaks together; a
weekend shifts the morning peak later while the evening peak persists.

The OU kernel with ell=3h treats 07:00 and 19:00 as nearly independent
(K_OU(7,19) = exp(-12/3) ~ 0.02). The empirical correlation kernel has
a visible off-diagonal block connecting the two peaks, discovered
automatically from the data. Applying the correlation kernel to the
`is_weekend` and `month` main effects produces qualitatively different
attributions from OU: effects that the OU kernel assigns separately to
the morning and evening phases are correctly understood as reflecting a
single underlying household-regime variable.

Games     : prediction, sensitivity, risk
Kernels   : Identity, OU (ell=3h), Correlation, Causal (ell=3h)
Profiles  : Typical weekday / Weekend / Cold winter weekday

Figures
-------
  fig1_operator_sweep.pdf         -- 3x4 games x kernels (month + is_weekend)
  fig2_correlation_kernel.pdf     -- kernel matrix + OU comparison + row slice
  fig3_main_effects_all_games.pdf -- top-5 main effects, identity kernel
  fig4_weekend_causal.pdf         -- is_weekend under symmetric vs causal kernels
  fig5_profiles_comparison.pdf    -- Shapley curves, 3 profiles, OU kernel
"""

import itertools
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0.  Settings
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(_HERE, 'data', 'ihepc')
DATA_FILE = os.path.join(DATA_DIR, 'household_power_consumption.parquet')

BASE_PLOT_DIR = os.path.join(
    'plots', 'ihepc')

RNG_SEED = 42
T_HOURS  = 24
dt       = 1.0          # time step = 1 hour

RF_N_EST = 300
RF_JOBS  = -1

SAMPLE_SIZE = {
    'prediction' : 150,
    'sensitivity': 200,
    'risk'       : 200,
}

HOUR_LABELS = ['{:02d}:00'.format(h) for h in range(T_HOURS)]
t_grid      = np.arange(T_HOURS, dtype=float)

# Day-level features
DAY_FEATURE_NAMES = [
    'day_of_week',      # 0=Mon ... 6=Sun
    'is_weekend',       # binary: Sat or Sun  (the ann_indicator analogue)
    'month',            # 1-12: seasonal heating/cooling
    'season',           # 1=winter 2=spring 3=summer 4=autumn
    'lag_daily_mean',   # previous day's mean kW  (regime persistence)
    'lag_morning',      # previous day's 06-09h mean kW
]

# Intraday phase windows (hour indices)
MORNING_PEAK = (6,  10)   # 06:00-09:00
EVENING_PEAK = (17, 22)   # 17:00-21:00
OVERNIGHT    = (0,  5)    # 00:00-04:00

# Feature colours
FEAT_COLORS = {
    'day_of_week'   : '#1f77b4',
    'is_weekend'    : '#ff7f0e',
    'month'         : '#2ca02c',
    'season'        : '#d62728',
    'lag_daily_mean': '#9467bd',
    'lag_morning'   : '#8c564b',
}

GAME_LABELS = {
    'prediction' :
        r'Prediction  $v(S)(t)=\mathbb{E}[F(x)(t)\mid X_S]$',
    'sensitivity':
        r'Sensitivity  $v(S)(t)=\mathrm{Var}[F(x)(t)\mid X_S]$',
    'risk'       :
        r'Risk (MSE)  $v(S)(t)=\mathbb{E}[(Y(t)-F(x)(t))^2\mid X_S]$',
}

GAME_YLABEL = {
    'prediction' : 'Effect on power (kW)',
    'sensitivity': r'Var$[F(t)]$ (kW$^2$)',
    'risk'       : r'Effect on MSE (kW$^2$)',
}

KERNEL_LABELS = {
    'Identity'   : 'Identity\n(pointwise SHAP)',
    'OU'         : r'OU  ($\ell=3$ h)',
    'Correlation': 'Empirical\ncorrelation',
    'Causal'     : r'Causal  ($\ell=3$ h)',
}


# ===========================================================================
# 1.  Data loading and feature construction
# ===========================================================================

def _require_dir(path):
    os.makedirs(path, exist_ok=True)


def _month_to_season(m):
    """1=winter(Dec-Feb), 2=spring, 3=summer, 4=autumn."""
    if m in (12, 1, 2):
        return 1
    elif m in (3, 4, 5):
        return 2
    elif m in (6, 7, 8):
        return 3
    else:
        return 4


def _fetch_and_cache():
    """
    Download IHEPC from UCI ML Repository via ucimlrepo and cache
    the raw minute-level DataFrame to DATA_FILE as a parquet file
    for fast re-loading on subsequent runs.

    Returns a DataFrame with columns:
        datetime (pd.Timestamp), Global_active_power (float),
        date (str), hour (int)
    """
    import importlib
    if importlib.util.find_spec('ucimlrepo') is None:
        raise RuntimeError(
            'ucimlrepo is not installed.\n'
            'Run:  pip install ucimlrepo')

    from ucimlrepo import fetch_ucirepo

    print('  Fetching IHEPC from UCI ML Repository (id=235) ...')
    print('  This downloads ~125 MB and may take a minute.')
    dataset = fetch_ucirepo(id=235)

    # dataset.data.features contains all measurement columns;
    # the Date and Time columns are part of the feature set.
    raw = dataset.data.features.copy()

    # Print metadata once so the user can inspect variable definitions
    print('\n  Dataset metadata:')
    print('    name     :', dataset.metadata.get('name', 'N/A'))
    print('    n_rows   :', len(raw))
    print('    columns  :', list(raw.columns))

    # Parse datetime -- ucimlrepo returns Date as 'dd/mm/yyyy'
    # and Time as 'HH:MM:SS' as separate columns
    if 'Date' in raw.columns and 'Time' in raw.columns:
        raw['datetime'] = pd.to_datetime(
            raw['Date'] + ' ' + raw['Time'],
            dayfirst=True, errors='coerce')
        raw = raw.drop(columns=['Date', 'Time'])
    elif 'datetime' in raw.columns:
        raw['datetime'] = pd.to_datetime(raw['datetime'], errors='coerce')
    else:
        # Fallback: index may already be a DatetimeIndex
        raw = raw.reset_index()
        raw.columns = ['datetime'] + list(raw.columns[1:])
        raw['datetime'] = pd.to_datetime(raw['datetime'], errors='coerce')

    raw = raw.dropna(subset=['datetime'])

    # ucimlrepo returns measurement columns as object dtype with '?'
    # for missing values. Coerce every non-datetime/derived column to
    # float so pyarrow can serialise without ArrowTypeError.
    skip = {'datetime', 'date', 'hour'}
    for col in [c for c in raw.columns if c not in skip]:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')

    raw = raw.dropna(subset=['Global_active_power'])
    raw['date'] = raw['datetime'].dt.date.astype(str)
    raw['hour'] = raw['datetime'].dt.hour

    # Cache to parquet for fast subsequent loads
    _require_dir(DATA_DIR)
    raw.to_parquet(DATA_FILE, index=False)
    print('  Cached to: {}'.format(DATA_FILE))
    return raw


def load_and_aggregate():
    """
    Load IHEPC data (from local parquet cache or UCI ML Repository),
    aggregate to daily 24h power curves, and engineer day-level features.

    On first run: downloads from UCI via ucimlrepo and caches locally.
    On subsequent runs: loads from parquet cache (~2s vs ~30s).

    Returns
    -------
    X_day   : pd.DataFrame  (n_days, 6)
    Y_raw   : np.ndarray    (n_days, 24)  hourly mean kW
    Y_adj   : np.ndarray    (n_days, 24)  diurnal-subtracted
    diurnal : np.ndarray    (24,)
    dates   : list of str
    """
    if os.path.isfile(DATA_FILE):
        print('  Loading cached IHEPC data from {} ...'.format(DATA_FILE))
        df = pd.read_parquet(DATA_FILE)
        # Ensure derived columns exist in case cache predates this version
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(
                df['datetime']).dt.date.astype(str)
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    else:
        df = _fetch_and_cache()

    print('  {:,} minute-level rows.'.format(len(df)))

    # Aggregate to hourly means per day
    hourly = (
        df.groupby(['date', 'hour'])['Global_active_power']
        .mean()
        .unstack('hour')
        .reindex(columns=range(T_HOURS))
    )

    # Keep only days with all 24 hours observed
    hourly = hourly[hourly.notna().sum(axis=1) == T_HOURS]
    Y_raw  = hourly.values.astype(float)
    dates  = hourly.index.tolist()

    print('  Complete days: {}  ({} - {})'.format(
        len(dates), dates[0], dates[-1]))

    diurnal = Y_raw.mean(axis=0)
    Y_adj   = Y_raw - diurnal[None, :]

    # ── Day-level features ────────────────────────────────────────────────
    records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str)
        m      = dt_obj.month
        dow    = dt_obj.dayofweek        # 0=Mon, 6=Sun
        is_wkd = int(dow >= 5)           # Sat or Sun

        # Lagged features: require previous day to exist in data
        if i == 0:
            lag_mean = float(Y_raw.mean())
            lag_morn = float(Y_raw[:, MORNING_PEAK[0]:MORNING_PEAK[1]].mean())
        else:
            lag_mean = float(Y_raw[i - 1].mean())
            lag_morn = float(
                Y_raw[i - 1, MORNING_PEAK[0]:MORNING_PEAK[1]].mean())

        records.append({
            'day_of_week'   : float(dow),
            'is_weekend'    : float(is_wkd),
            'month'         : float(m),
            'season'        : float(_month_to_season(m)),
            'lag_daily_mean': lag_mean,
            'lag_morning'   : lag_morn,
        })

    X_day = pd.DataFrame(records, index=dates)

    # Validation
    for col in DAY_FEATURE_NAMES:
        assert col in X_day.columns
        assert X_day[col].std() > 1e-6, 'Constant: {}'.format(col)

    print('  Weekend days: {} ({:.1f}%)'.format(
        int(X_day['is_weekend'].sum()),
        X_day['is_weekend'].mean() * 100))
    print('  Mean daily consumption: {:.3f} kW'.format(Y_raw.mean()))

    return X_day, Y_raw, Y_adj, diurnal, dates


# ===========================================================================
# 2.  Model
# ===========================================================================

class RFModel:
    """RandomForestRegressor: X (N,6) -> Y (N,24). No t, no PCA."""

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
        return self.model.predict(X)    # (N, 24)

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
    prediction  : v(S)(t) = E[F(X)(t) | X_S]
    sensitivity : v(S)(t) = Var[F(X)(t) | X_S]
    risk        : v(S)(t) = E[(Y_obs(t) - F(X)(t))^2 | X_S]
    """

    def __init__(self, predict_fn, X_background, x_explain,
                 game_type='prediction', Y_obs=None,
                 sample_size=150, random_seed=RNG_SEED):
        if game_type == 'risk' and Y_obs is None:
            raise ValueError('Y_obs required for risk game.')
        self.predict_fn   = predict_fn
        self.X_background = X_background
        self.x_explain    = x_explain
        self.game_type    = game_type
        self.Y_obs        = Y_obs
        self.sample_size  = sample_size
        self.random_seed  = random_seed
        self.T            = T_HOURS
        self.n_players    = len(DAY_FEATURE_NAMES)
        self.player_names = list(DAY_FEATURE_NAMES)
        self.coalitions   = np.array(
            list(itertools.product(
                [False, True], repeat=self.n_players)),
            dtype=bool)
        self.n_coalitions = len(self.coalitions)
        self._idx = {
            tuple(c): i
            for i, c in enumerate(self.coalitions)}
        self.values = None

    def _impute(self, coalition):
        rng = np.random.default_rng(self.random_seed)
        idx = rng.integers(
            0, len(self.X_background),
            size=self.sample_size)
        X = self.X_background[idx].copy()
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
        print('  [{}]  {} coalitions x {} samples ...'.format(
            self.game_type, self.n_coalitions, self.sample_size))
        self.values = np.zeros((self.n_coalitions, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i + 1) % 16 == 0 or i + 1 == self.n_coalitions:
                print('    {}/{} done.'.format(
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
            coal = tuple(i in L for i in range(p))
            m   += (-1) ** (len(S) - len(L)) * game[coal]
        moebius[S] = m
    return moebius


def shapley_from_moebius(moebius, n_players):
    shapley = {i: np.zeros(T_HOURS) for i in range(n_players)}
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

def kernel_ou(t, length_scale=3.0):
    """
    OU kernel: ell=3h means correlation decays to e^{-1}~0.37 at 3h lag.
    At 12h lag (morning to evening peak): K = exp(-4) ~ 0.02.
    Effectively treats the two peaks as independent.
    """
    return np.exp(-np.abs(t[:, None] - t[None, :]) / length_scale)

def kernel_causal(t, length_scale=3.0):
    """One-sided: K(t,s) = exp(-(t-s)/ell) for t>=s, else 0."""
    d = t[:, None] - t[None, :]
    return np.where(d >= 0, np.exp(-d / length_scale), 0.0)

def kernel_output_correlation(Y_raw):
    """
    Empirical cross-hour correlation kernel from observed power curves.
    Uses raw (non-diurnal-subtracted) data to capture the true cross-hour
    dependence structure including morning/evening peak coupling.

    Key property: K[7, 19] >> 0 (morning and evening peaks co-vary)
    even though |7 - 19| = 12 >> ell, so OU kernel misses this entirely.
    """
    C   = np.cov(Y_raw.T)           # (24, 24)
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    K   = np.clip(C / np.outer(std, std), -1.0, 1.0)
    offdiag = K.copy()
    np.fill_diagonal(offdiag, 0.0)
    print('  Correlation kernel:')
    print('    off-diagonal mean = {:.3f}'.format(offdiag.mean()))
    print('    K[07:00, 19:00]   = {:.3f}  (morning-evening coupling)'.format(
        K[7, 19]))
    print('    K[02:00, 14:00]   = {:.3f}  (overnight-midday, should be low)'.format(
        K[2, 14]))
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

XTICK_IDXS   = list(range(0, T_HOURS, 3))
XTICK_LABELS = [HOUR_LABELS[i] for i in XTICK_IDXS]

def _set_time_axis(ax, sparse=False):
    step = 2 if sparse else 1
    ax.set_xticks(XTICK_IDXS[::step])
    ax.set_xticklabels(
        XTICK_LABELS[::step],
        rotation=45, ha='right', fontsize=6)
    ax.set_xlim(-0.5, T_HOURS - 0.5)

def _period_shade(ax):
    """Shade morning and evening consumption peaks."""
    ax.axvspan(MORNING_PEAK[0], MORNING_PEAK[1],
               alpha=0.10, color='#4a90e2', zorder=0)
    ax.axvspan(EVENING_PEAK[0], EVENING_PEAK[1],
               alpha=0.10, color='#e24a4a', zorder=0)

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
# 8.  Figure 1 -- Operator sweep
#     Rows: prediction / sensitivity / risk
#     Cols: Identity / OU / Correlation / Causal
#     Focus features: month (slow seasonal) + is_weekend (sharp binary)
# ===========================================================================

def plot_operator_sweep(moebius_hv, shapley_hv, kernels, pnames):
    game_types   = ['prediction', 'sensitivity', 'risk']
    kernel_names = list(kernels.keys())
    nrows, ncols = len(game_types), len(kernel_names)

    fi_month = pnames.index('month')
    fi_wkd   = pnames.index('is_weekend')
    focus    = [fi_month, fi_wkd]
    colors   = [FEAT_COLORS['month'], FEAT_COLORS['is_weekend']]
    flabels  = ['month', 'is\\_weekend']

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 3.2 * nrows),
        sharey='row',
    )
    fig.suptitle(
        'Operator sweep: Shapley curves across games and kernels\n'
        'Typical weekday profile  —  Random Forest  —  UCI IHEPC',
        fontsize=12, fontweight='bold',
    )

    for r, gtype in enumerate(game_types):
        shapley = shapley_hv[gtype]

        for c, kname in enumerate(kernel_names):
            ax = axes[r, c]
            K  = kernels[kname]

            for fi, col, lbl in zip(focus, colors, flabels):
                curve = apply_kernel(shapley[fi], K)
                ax.plot(t_grid, curve, color=col, lw=2.0, label=lbl)

            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            _set_time_axis(ax, sparse=True)
            ax.tick_params(axis='y', labelsize=7)

            if r == 0:
                ax.set_title(
                    KERNEL_LABELS[kname],
                    fontsize=9, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
                ax.text(
                    -0.42, 0.5, GAME_LABELS[gtype],
                    transform=ax.transAxes,
                    fontsize=7.5, va='center', ha='right',
                    rotation=90, color='#333')
            ax.set_xlabel('Hour', fontsize=7)

            # Mark identity+prediction as classical SHAP
            if r == 0 and c == 0:
                ax.text(
                    0.97, 0.97, '= pointwise\nSHAP',
                    transform=ax.transAxes,
                    fontsize=6.5, va='top', ha='right', color='#555',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', ec='#aaa', alpha=0.8))

    handles = [
        plt.Line2D([0], [0], color=FEAT_COLORS['month'],
                   lw=2, label='month'),
        plt.Line2D([0], [0], color=FEAT_COLORS['is_weekend'],
                   lw=2, label='is\\_weekend'),
        plt.matplotlib.patches.Patch(
            color='#4a90e2', alpha=0.3, label='Morning peak (06-09h)'),
        plt.matplotlib.patches.Patch(
            color='#e24a4a', alpha=0.3, label='Evening peak (17-21h)'),
    ]
    fig.legend(handles=handles, loc='upper right',
               bbox_to_anchor=(0.99, 0.99),
               fontsize=8, framealpha=0.9)
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    return fig


# ===========================================================================
# 9.  Figure 2 -- Correlation kernel structure
#     Panel A: 24x24 empirical correlation heatmap (annotated blocks)
#     Panel B: 24x24 OU kernel heatmap (same scale)
#     Panel C: Row slice at hour 7 (07:00) -- shows AM/PM coupling
#     Panel D: Row slice at hour 2 (02:00) -- shows overnight isolation
# ===========================================================================

def plot_correlation_kernel_structure(K_corr, K_ou):
    fig = plt.figure(figsize=(16, 5.0))
    gs  = fig.add_gridspec(
        1, 5,
        width_ratios=[1, 1, 0.06, 1.1, 1.1],
        wspace=0.40)

    ax_corr = fig.add_subplot(gs[0])
    ax_ou   = fig.add_subplot(gs[1])
    ax_cb   = fig.add_subplot(gs[2])
    ax_r7   = fig.add_subplot(gs[3])
    ax_r2   = fig.add_subplot(gs[4])

    fig.suptitle(
        'Cross-hour correlation structure in household power consumption\n'
        'Empirical kernel discovers morning/evening peak coupling; '
        'OU kernel misses it',
        fontsize=11, fontweight='bold')

    tick_h   = list(range(0, T_HOURS, 3))
    tick_lbl = ['{:02d}:00'.format(h) for h in tick_h]
    vmin, vmax = -0.2, 1.0

    # ── Panel A: empirical correlation ────────────────────────────────────
    im = ax_corr.imshow(
        K_corr, aspect='auto', origin='upper',
        cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax_corr.set_title(
        'Empirical correlation\n$K(t,s)=\\mathrm{Corr}(Y_t,Y_s)$',
        fontsize=9, fontweight='bold')

    for ax_ in [ax_corr, ax_ou]:
        ax_.set_xticks(tick_h)
        ax_.set_xticklabels(
            tick_lbl, rotation=45, ha='right', fontsize=6)
        ax_.set_yticks(tick_h)
        ax_.set_yticklabels(tick_lbl, fontsize=6)
        ax_.set_xlabel('Hour $s$', fontsize=8)
        ax_.set_ylabel('Hour $t$', fontsize=8)

    # Annotate the three meaningful blocks
    block_annotations = [
        (MORNING_PEAK, MORNING_PEAK, '#4a90e2', 'AM×AM'),
        (EVENING_PEAK, EVENING_PEAK, '#e24a4a', 'PM×PM'),
        (MORNING_PEAK, EVENING_PEAK, '#9b59b6', 'AM×PM\ncoupling'),
        (EVENING_PEAK, MORNING_PEAK, '#9b59b6', ''),
    ]
    for (r0, r1), (c0, c1), col, lbl in block_annotations:
        rect = plt.Rectangle(
            (c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
            linewidth=1.5, edgecolor=col,
            facecolor='none', zorder=3)
        ax_corr.add_patch(rect)
        if lbl:
            ax_corr.text(
                (c0 + c1) / 2, (r0 + r1) / 2, lbl,
                fontsize=5.5, ha='center', va='center',
                color=col, fontweight='bold')

    # ── Panel B: OU kernel ────────────────────────────────────────────────
    ax_ou.imshow(
        K_ou, aspect='auto', origin='upper',
        cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax_ou.set_title(
        'OU kernel\n$K(t,s)=e^{-|t-s|/3}$',
        fontsize=9, fontweight='bold')
    ax_ou.text(
        0.5, 0.02,
        'No AM$\\leftrightarrow$PM coupling\n'
        '$K(07:00, 19:00)=e^{-4}\\approx 0.02$',
        transform=ax_ou.transAxes,
        fontsize=7, ha='center', va='bottom', color='#888',
        bbox=dict(boxstyle='round,pad=0.2',
                  fc='white', ec='#ccc', alpha=0.9))

    # ── Colorbar ─────────────────────────────────────────────────────────
    plt.colorbar(im, cax=ax_cb)
    ax_cb.set_ylabel('Correlation', fontsize=7)
    ax_cb.tick_params(labelsize=7)

    # ── Panel C: row at 07:00 ─────────────────────────────────────────────
    row_h = 7
    _plot_kernel_row(ax_r7, K_corr, K_ou, row_h,
                     title='Kernel row at $t=07$:00\n'
                           '(morning peak hour)',
                     annotate_pm=True)

    # ── Panel D: row at 02:00 (overnight -- should be isolated) ──────────
    row_h2 = 2
    _plot_kernel_row(ax_r2, K_corr, K_ou, row_h2,
                     title='Kernel row at $t=02$:00\n'
                           '(overnight -- both kernels agree)',
                     annotate_pm=False)

    plt.tight_layout()
    return fig


def _plot_kernel_row(ax, K_corr, K_ou, row_h,
                     title='', annotate_pm=False):
    ax.plot(t_grid, K_corr[row_h, :],
            color='#2a9d8f', lw=2.2, label='Empirical correlation')
    ax.plot(t_grid, K_ou[row_h, :],
            color='#e76f51', lw=2.2, ls='--',
            label='OU ($\\ell=3$h)')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    for s, e, col in [
        (MORNING_PEAK[0], MORNING_PEAK[1], '#4a90e2'),
        (EVENING_PEAK[0], EVENING_PEAK[1], '#e24a4a'),
    ]:
        ax.axvspan(s, e, alpha=0.12, color=col)
    ax.axvline(row_h, color='gray', lw=0.8, ls=':', alpha=0.5)
    ax.set_xticks(XTICK_IDXS)
    ax.set_xticklabels(XTICK_LABELS, rotation=45, ha='right', fontsize=6)
    ax.set_xlim(-0.5, T_HOURS - 0.5)
    ax.set_xlabel('Hour $s$', fontsize=8)
    ax.set_ylabel('$K({:02d}:00,\\ s)$'.format(row_h), fontsize=8)
    ax.set_title(title, fontsize=8.5, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.tick_params(labelsize=7)

    if annotate_pm:
        pm_mid = (EVENING_PEAK[0] + EVENING_PEAK[1]) // 2
        val    = K_corr[row_h, pm_mid]
        ax.annotate(
            'AM$\\leftrightarrow$PM\ncoupling\n(corr. only)',
            xy=(pm_mid, val),
            xytext=(pm_mid - 4, val + 0.08),
            fontsize=7, color='#9b59b6',
            arrowprops=dict(
                arrowstyle='->', color='#9b59b6', lw=1.2))


# ===========================================================================
# 10.  Figure 3 -- Main effects, all three games, identity kernel
# ===========================================================================

def plot_main_effects_all_games(moebius_dict, pnames, top_k=5):
    game_types = ['prediction', 'sensitivity', 'risk']

    fig, axes = plt.subplots(
        len(game_types), 2,
        figsize=(12, 3.8 * len(game_types)),
        gridspec_kw={'width_ratios': [3, 1.5]})
    fig.suptitle(
        'Main effects $m_i(t)$ — Identity kernel\n'
        'Typical weekday profile  —  UCI IHEPC',
        fontsize=11, fontweight='bold')

    for r, gtype in enumerate(game_types):
        moebius = moebius_dict[gtype]
        imps    = {i: float(np.sum(np.abs(moebius[(i,)])))
                   for i in range(len(pnames))}
        top     = sorted(imps, key=imps.get, reverse=True)[:top_k]

        # Curves
        ax = axes[r, 0]
        for fi in top:
            ax.plot(t_grid, moebius[(fi,)],
                    color=FEAT_COLORS[pnames[fi]],
                    lw=2.0, label=pnames[fi])
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
        ax.set_xlabel('Hour', fontsize=8)
        ax.set_title(GAME_LABELS[gtype], fontsize=9)
        if r == 0:
            ax.legend(fontsize=7, loc='upper left')

        # Importance bars
        ax2 = axes[r, 1]
        order = sorted(range(len(pnames)),
                       key=lambda i: imps[i], reverse=True)
        ax2.barh(
            range(len(pnames)),
            [imps[i] for i in order],
            color=[FEAT_COLORS[pnames[i]] for i in order],
            alpha=0.85)
        ax2.set_yticks(range(len(pnames)))
        ax2.set_yticklabels([pnames[i] for i in order], fontsize=7)
        ax2.axvline(0, color='gray', lw=0.8, ls=':')
        ax2.set_xlabel(r'$\int|m_i(t)|\,dt$', fontsize=8)
        ax2.set_title('Integrated importance', fontsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


# ===========================================================================
# 11.  Figure 4 -- is_weekend causal kernel comparison
#
#      The `is_weekend` effect is concentrated in the morning peak: on
#      weekends the morning routine is delayed (~09:00-10:00 vs ~07:00)
#      and commuter demand disappears entirely.  Before the morning peak
#      (~00:00-06:00) consumption cannot yet be affected by the day-type.
#
#      Under a symmetric kernel (OU), attribution smears backward into
#      overnight hours -- physically incoherent.  The causal kernel
#      enforces that overnight hours cannot "know" about the day-type
#      and correctly restricts attribution to the morning peak onward.
# ===========================================================================

def plot_weekend_causal(moebius_dict, pnames):
    fi_wkd = pnames.index('is_weekend')
    game_types = ['prediction', 'risk']

    kernels_ordered = {
        'Identity'          : kernel_identity(t_grid),
        'OU ($\\ell$=3h)'   : kernel_ou(t_grid, 3.0),
        'Causal $\\ell$=2h' : kernel_causal(t_grid, 2.0),
        'Causal $\\ell$=3h' : kernel_causal(t_grid, 3.0),
        'Causal $\\ell$=6h' : kernel_causal(t_grid, 6.0),
    }
    knames     = list(kernels_ordered.keys())
    causal_pal = plt.get_cmap('YlOrRd')(np.linspace(0.45, 0.85, 3))

    fig, axes = plt.subplots(
        len(game_types), len(knames),
        figsize=(3.2 * len(knames), 3.5 * len(game_types)),
        sharey='row')
    fig.suptitle(
        'is\\_weekend — symmetric vs causal kernel\n'
        'OU kernel smears attribution into overnight hours '
        '(before morning peak); causal kernel prevents this\n'
        'UCI IHEPC',
        fontsize=11, fontweight='bold')

    for r, gtype in enumerate(game_types):
        raw = moebius_dict[gtype][(fi_wkd,)]

        for c, (kname, K) in enumerate(kernels_ordered.items()):
            ax    = axes[r, c]
            is_id = kname == 'Identity'
            is_ou = kname.startswith('OU')
            ci    = c - 2 if c >= 2 else 0

            col = ('#444444' if is_id
                   else '#457b9d' if is_ou
                   else causal_pal[ci])

            ax.plot(t_grid, apply_kernel(raw, K),
                    color=col, lw=2.2)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)

            # Mark the overnight window where leakage would occur
            ax.axvspan(OVERNIGHT[0], OVERNIGHT[1],
                       alpha=0.08, color='#555555', zorder=0)

            _set_time_axis(ax, sparse=True)
            ax.tick_params(axis='y', labelsize=7)

            if r == 0:
                ax.set_title(kname, fontsize=8.5, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
                ax.text(
                    -0.42, 0.5, GAME_LABELS[gtype],
                    transform=ax.transAxes,
                    fontsize=7.5, va='center', ha='right',
                    rotation=90, color='#333')
            ax.set_xlabel('Hour', fontsize=7)

            # Annotate OU leakage
            if is_ou and r == 0:
                ax.text(
                    0.5, 0.97,
                    'overnight\nleakage\n(before AM peak)',
                    transform=ax.transAxes,
                    fontsize=6.5, va='top', ha='center',
                    color='#c0392b',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='#ffeaea', ec='#c0392b', alpha=0.9))

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    return fig


# ===========================================================================
# 12.  Figure 5 -- Profile comparison
#      OU kernel, prediction game, three household-regime profiles
# ===========================================================================

def plot_profiles_comparison(profile_results, pnames):
    """
    profile_results : dict  label -> (moebius, shapley)
    Three profiles:
      Typical weekday  -- workday, mild temp, average lag
      Weekend          -- is_weekend=1, mild temp
      Cold winter day  -- winter month, high lag_daily_mean (high prev day)
    """
    K_ou = kernel_ou(t_grid, 3.0)
    n    = len(profile_results)

    all_mob = {k: v[0] for k, v in profile_results.items()}
    top4    = _top_features(all_mob, len(pnames), top_k=4)

    profile_titles = {
        'Typical weekday' :
            'Typical weekday\n(working day, mild temp)',
        'Weekend'         :
            'Weekend\n(Saturday/Sunday)',
        'Cold winter day' :
            'Cold winter day\n(January, high prior consumption)',
    }

    fig, axes = plt.subplots(
        1, n, figsize=(5.5 * n, 4.2), sharey=False)
    fig.suptitle(
        'Shapley curves — OU kernel ($\\ell=3$h) — prediction game\n'
        'Three household-regime profiles  —  UCI IHEPC',
        fontsize=11, fontweight='bold')

    for ax, (label, (moebius, shapley)) in zip(axes, profile_results.items()):
        for fi in top4:
            curve = apply_kernel(shapley[fi], K_ou)
            ax.plot(t_grid, curve,
                    color=FEAT_COLORS[pnames[fi]],
                    lw=2.0, label=pnames[fi])
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        ax.set_title(
            profile_titles.get(label, label),
            fontsize=9, fontweight='bold')
        ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=8)
        ax.set_xlabel('Hour', fontsize=8)
        ax.legend(fontsize=7, loc='upper left')
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


# ===========================================================================
# 13.  Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  UCI IHEPC  —  RF direct  (paper figures)')
    print('=' * 60)

    _require_dir(BASE_PLOT_DIR)
    _require_dir(DATA_DIR)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print('\n[1] Loading and aggregating data ...')
    X_day, Y_raw, Y_adj, diurnal, dates = load_and_aggregate()
    X_day_np = X_day.to_numpy().astype(float)

    print('\n  Diurnal mean power curve:')
    for h in [0, 7, 12, 19, 23]:
        print('    {:02d}:00  {:.3f} kW'.format(h, diurnal[h]))

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
    K_corr = kernel_output_correlation(Y_raw)
    K_ou_k = kernel_ou(t_grid, length_scale=3.0)
    kernels = {
        'Identity'   : kernel_identity(t_grid),
        'OU'         : K_ou_k,
        'Correlation': K_corr,
        'Causal'     : kernel_causal(t_grid, length_scale=3.0),
    }

    # ── 4. Profiles ───────────────────────────────────────────────────────
    print('\n[4] Selecting day profiles ...')
    pnames  = list(DAY_FEATURE_NAMES)
    fi_wkd  = pnames.index('is_weekend')
    fi_dow  = pnames.index('day_of_week')
    fi_mon  = pnames.index('month')
    fi_lag  = pnames.index('lag_daily_mean')
    fi_sea  = pnames.index('season')

    lag_p75 = float(np.percentile(X_day_np[:, fi_lag], 75))

    def find_profile(conditions, label):
        mask = np.ones(len(X_day_np), dtype=bool)
        for feat, (lo, hi) in conditions.items():
            ci    = pnames.index(feat)
            mask &= (X_day_np[:, ci] >= lo) & (X_day_np[:, ci] <= hi)
        hits = X_day_np[mask]
        if len(hits) == 0:
            raise RuntimeError(
                'No day matches profile "{}": {}'.format(
                    label, conditions))
        print('  "{}": {} matching days; picking median.'.format(
            label, len(hits)))
        return hits[len(hits) // 2]

    x_weekday = find_profile(
        {'is_weekend': (-0.1, 0.1),
         'day_of_week': (0.9, 4.1)},   # Tue-Fri (avoid Monday effects)
        'Typical weekday')

    x_weekend = find_profile(
        {'is_weekend': (0.9, 1.1)},
        'Weekend')

    x_winter = find_profile(
        {'season': (0.9, 1.1),          # winter
         'is_weekend': (-0.1, 0.1),     # weekday
         'lag_daily_mean': (lag_p75, 999)},   # high prior consumption
        'Cold winter day')

    def _y_obs(x_prof):
        diffs = np.abs(X_day_np - x_prof[None, :]).sum(axis=1)
        return Y_adj[int(np.argmin(diffs))]

    profile_defs = [
        ('Typical weekday', x_weekday, _y_obs(x_weekday)),
        ('Weekend',         x_weekend, _y_obs(x_weekend)),
        ('Cold winter day', x_winter,  _y_obs(x_winter)),
    ]

    for lbl, xp, _ in profile_defs:
        print('  {}: {}'.format(lbl, '  '.join(
            '{}={:.3f}'.format(n, xp[j])
            for j, n in enumerate(pnames))))

    # ── 5. Games for primary profile (Typical weekday) ────────────────────
    print('\n[5] Computing all three games (Typical weekday) ...')
    x_prim, y_prim = x_weekday, _y_obs(x_weekday)

    moebius_prim = {}
    shapley_prim = {}

    for gtype in ('prediction', 'sensitivity', 'risk'):
        print('\n  game: {} ...'.format(gtype))
        game = FunctionalGame(
            predict_fn   = model.predict,
            X_background = X_day_np,
            x_explain    = x_prim,
            game_type    = gtype,
            Y_obs        = y_prim,
            sample_size  = SAMPLE_SIZE[gtype],
            random_seed  = RNG_SEED,
        )
        game.precompute()
        moebius_prim[gtype] = functional_moebius_transform(game)
        shapley_prim[gtype] = shapley_from_moebius(
            moebius_prim[gtype], game.n_players)

    # ── 6. Prediction game for all profiles ───────────────────────────────
    print('\n[6] Computing prediction game for all profiles ...')
    profile_results = {}

    for label, x_prof, y_prof in profile_defs:
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

    # ── 7. Figures ────────────────────────────────────────────────────────
    print('\n[7] Generating figures ...')

    savefig(
        plot_operator_sweep(
            moebius_prim, shapley_prim, kernels, pnames),
        'fig1_operator_sweep.pdf')

    savefig(
        plot_correlation_kernel_structure(K_corr, K_ou_k),
        'fig2_correlation_kernel.pdf')

    savefig(
        plot_main_effects_all_games(moebius_prim, pnames),
        'fig3_main_effects_all_games.pdf')

    savefig(
        plot_weekend_causal(moebius_prim, pnames),
        'fig4_weekend_causal.pdf')

    savefig(
        plot_profiles_comparison(profile_results, pnames),
        'fig5_profiles_comparison.pdf')

    print('\n' + '=' * 60)
    print('  Done.  Figures saved under {}/'.format(BASE_PLOT_DIR))
    print('  fig1_operator_sweep.pdf        -- central paper figure')
    print('  fig2_correlation_kernel.pdf    -- kernel structure motivation')
    print('  fig3_main_effects_all_games.pdf')
    print('  fig4_weekend_causal.pdf')
    print('  fig5_profiles_comparison.pdf')
    print('=' * 60)