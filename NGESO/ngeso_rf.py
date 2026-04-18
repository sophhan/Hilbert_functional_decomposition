"""
Functional Explanation Framework -- NESO GB Historic Demand
===========================================================
Dataset : National Energy System Operator (NESO) Historic Demand Data
          Half-hourly GB National Demand (ND) in MW, 2018-2022.
          Place CSV files in ./data/ngeso/:
            demanddata_2018.csv ... demanddata_2022.csv
          Download from: https://www.neso.energy/data-portal/historic-demand-data

Output  : 48-period half-hourly demand trajectory per day (T=48).
          Each period is 30 minutes; period 1 = 00:00-00:30,
          period 48 = 23:30-00:00.

Model   : F^H : R^7 -> R^48
  RandomForestRegressor, direct multi-output (no PCA, no t as input).
  Seven day-level features, all known before the day begins:
    day_of_week     integer 0 (Mon) - 6 (Sun)
    is_weekend      binary: 1 if Sat or Sun
    month           integer 1-12
    season          integer 1=winter 2=spring 3=summer 4=autumn
    lag_daily_mean  previous day mean ND (MW)
    lag_morning     previous day 06:00-09:00 mean ND (MW)
    lag_evening     previous day 17:00-20:00 mean ND (MW)

Why the correlation kernel matters here
---------------------------------------
GB national demand has a pronounced bimodal daily structure: a morning
peak (~07:00-09:00) driven by household wakeup + industrial startup, and
an evening peak (~17:00-20:00) driven by household return + cooking +
lighting. Both peaks co-vary strongly across days because both reflect the
same underlying demand regime (temperature, season, day type). On a cold
winter weekday both peaks are elevated; on a summer weekend both are
suppressed. The OU kernel with ell=4 treats 07:30 and 18:00 as nearly
independent (K_OU ~ exp(-21/4) ~ 0.005). The empirical correlation kernel
has a visible off-diagonal block connecting the two peaks, discovered
automatically from the data. This produces qualitatively different
attributions for `month`, `season`, and `is_weekend` -- features whose
effects span both peaks as a coherent unit rather than two isolated events.

Games     : prediction, sensitivity, risk
Kernels   : Identity, OU (ell=4 periods = 2h), Correlation, Causal (ell=4)
Profiles  : Typical winter weekday / Summer weekend / Cold snap weekday

Figures
-------
  fig1_operator_sweep.pdf         -- 3x4 games x kernels (month + is_weekend)
  fig2_correlation_kernel.pdf     -- kernel matrices + row slices
  fig3_main_effects_all_games.pdf -- top-5 main effects, identity kernel
  fig4_weekend_causal.pdf         -- is_weekend symmetric vs causal kernels
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
DATA_DIR  = os.path.join(_HERE, 'data')
YEARS     = [2018, 2019, 2020, 2021, 2022]

BASE_PLOT_DIR = os.path.join(
    'plots', 'neso')

RNG_SEED = 42
T        = 48           # half-hourly periods per day
dt       = 1.0          # time step = 1 settlement period (30 min)

RF_N_EST = 300
RF_JOBS  = -1

SAMPLE_SIZE = {
    'prediction' : 150,
    'sensitivity': 200,
    'risk'       : 200,
}

# Period labels: '00:00', '00:30', '01:00', ...
PERIOD_LABELS = [
    '{:02d}:{:02d}'.format((i * 30) // 60, (i * 30) % 60)
    for i in range(T)
]
t_grid = np.arange(T, dtype=float)

# Day-level features
DAY_FEATURE_NAMES = [
    'day_of_week',    # 0=Mon ... 6=Sun
    'is_weekend',     # binary: Sat or Sun
    'month',          # 1-12
    'season',         # 1=winter 2=spring 3=summer 4=autumn
    'lag_daily_mean', # previous day mean ND (MW)
    'lag_morning',    # previous day 06:00-09:00 mean ND (MW)
    'lag_evening',    # previous day 17:00-20:00 mean ND (MW)
]

# Intraday phase windows (period indices, 0-based)
# Period i covers [i*30min, (i+1)*30min)
MORNING_PEAK  = (12, 19)   # 06:00-09:30
EVENING_PEAK  = (34, 42)   # 17:00-21:00
OVERNIGHT     = (0,  8)    # 00:00-04:00

# Consistent feature colours
FEAT_COLORS = {
    'day_of_week'   : '#1f77b4',
    'is_weekend'    : '#ff7f0e',
    'month'         : '#2ca02c',
    'season'        : '#d62728',
    'lag_daily_mean': '#9467bd',
    'lag_morning'   : '#8c564b',
    'lag_evening'   : '#e377c2',
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
    'prediction' : 'Effect on demand (MW)',
    'sensitivity': r'Var$[F(t)]$ (MW$^2$)',
    'risk'       : r'Effect on MSE (MW$^2$)',
}

KERNEL_LABELS = {
    'Identity'   : 'Identity\n(pointwise SHAP)',
    'OU'         : r'OU  ($\ell=4$ periods = 2h)',
    'Correlation': 'Empirical\ncorrelation',
    'Causal'     : r'Causal  ($\ell=4$ periods)',
}


# ===========================================================================
# 1.  Data loading and feature construction
# ===========================================================================

def _require_dir(path):
    os.makedirs(path, exist_ok=True)


def _month_to_season(m):
    if m in (12, 1, 2):  return 1   # winter
    elif m in (3, 4, 5): return 2   # spring
    elif m in (6, 7, 8): return 3   # summer
    else:                return 4   # autumn


def _period_to_hour_range(p_start, p_end):
    """Convert period index range to readable string."""
    return '{}-{}'.format(PERIOD_LABELS[p_start], PERIOD_LABELS[p_end])


def load_and_aggregate():
    """
    Load NESO CSV files, aggregate to daily 48-period demand curves,
    and engineer day-level features.

    Returns
    -------
    X_day   : pd.DataFrame  (n_days, 7)
    Y_raw   : np.ndarray    (n_days, 48)  ND in MW per period
    Y_adj   : np.ndarray    (n_days, 48)  diurnal-subtracted
    diurnal : np.ndarray    (48,)
    dates   : list of str
    """
    dfs = []
    for yr in YEARS:
        path = os.path.join(DATA_DIR, 'demanddata_{}.csv'.format(yr))
        if not os.path.isfile(path):
            raise RuntimeError(
                'Missing: {}\n'
                'Download from neso.energy and place in {}'.format(
                    path, DATA_DIR))
        df = pd.read_csv(path, low_memory=False)
        dfs.append(df)
        print('  Loaded {}: {:,} rows, columns: {}'.format(
            os.path.basename(path), len(df),
            list(df.columns[:6])))

    raw = pd.concat(dfs, ignore_index=True)
    print('  Total rows: {:,}'.format(len(raw)))

    # Normalise column names -- NESO files use uppercase
    raw.columns = [c.strip().upper() for c in raw.columns]

    # Identify date and period columns robustly
    date_col   = next(c for c in raw.columns if 'DATE' in c)
    period_col = next(c for c in raw.columns if 'PERIOD' in c)

    # Use ND (National Demand) as output; fall back to TSD
    if 'ND' in raw.columns:
        demand_col = 'ND'
    elif 'TSD' in raw.columns:
        demand_col = 'TSD'
    else:
        raise RuntimeError(
            'Neither ND nor TSD column found. '
            'Available: {}'.format(list(raw.columns)))

    print('  Using demand column: {}'.format(demand_col))

    raw[date_col]   = raw[date_col].astype(str).str.strip()
    raw[period_col] = pd.to_numeric(raw[period_col], errors='coerce')
    raw[demand_col] = pd.to_numeric(raw[demand_col], errors='coerce')
    raw = raw.dropna(subset=[date_col, period_col, demand_col])

    # Keep only periods 1-48
    raw = raw[(raw[period_col] >= 1) & (raw[period_col] <= T)].copy()
    raw['period_idx'] = (raw[period_col] - 1).astype(int)  # 0-based

    # Pivot to (n_days, 48)
    pivot = raw.pivot_table(
        index=date_col,
        columns='period_idx',
        values=demand_col,
        aggfunc='mean')
    pivot = pivot.reindex(columns=range(T))

    # Keep days with all 48 periods present
    pivot = pivot[pivot.notna().sum(axis=1) == T]
    Y_raw  = pivot.values.astype(float)
    dates  = pivot.index.tolist()

    print('  Complete days: {}  ({} to {})'.format(
        len(dates), dates[0], dates[-1]))
    print('  Mean ND: {:.0f} MW  '
          'Peak period mean: {:.0f} MW  '
          'Trough period mean: {:.0f} MW'.format(
              Y_raw.mean(),
              Y_raw[:, MORNING_PEAK[0]:MORNING_PEAK[1]].mean(),
              Y_raw[:, OVERNIGHT[0]:OVERNIGHT[1]].mean()))

    diurnal = Y_raw.mean(axis=0)
    Y_adj   = Y_raw - diurnal[None, :]

    # ── Day-level features ────────────────────────────────────────────────
    records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str)
        m      = dt_obj.month
        dow    = dt_obj.dayofweek

        if i == 0:
            lag_mean = float(Y_raw.mean())
            lag_morn = float(Y_raw[:, MORNING_PEAK[0]:MORNING_PEAK[1]].mean())
            lag_eve  = float(Y_raw[:, EVENING_PEAK[0]:EVENING_PEAK[1]].mean())
        else:
            lag_mean = float(Y_raw[i - 1].mean())
            lag_morn = float(Y_raw[i - 1, MORNING_PEAK[0]:MORNING_PEAK[1]].mean())
            lag_eve  = float(Y_raw[i - 1, EVENING_PEAK[0]:EVENING_PEAK[1]].mean())

        records.append({
            'day_of_week'   : float(dow),
            'is_weekend'    : float(dow >= 5),
            'month'         : float(m),
            'season'        : float(_month_to_season(m)),
            'lag_daily_mean': lag_mean,
            'lag_morning'   : lag_morn,
            'lag_evening'   : lag_eve,
        })

    X_day = pd.DataFrame(records, index=dates)

    for col in DAY_FEATURE_NAMES:
        assert X_day[col].std() > 1e-6, 'Constant feature: {}'.format(col)

    print('  Weekend days: {} ({:.1f}%)'.format(
        int(X_day['is_weekend'].sum()),
        X_day['is_weekend'].mean() * 100))
    print('  Winter days (season=1): {} ({:.1f}%)'.format(
        int((X_day['season'] == 1).sum()),
        (X_day['season'] == 1).mean() * 100))

    return X_day, Y_raw, Y_adj, diurnal, dates


# ===========================================================================
# 2.  Model
# ===========================================================================

class RFModel:
    """RandomForestRegressor: X (N,7) -> Y (N,48). No t, no PCA."""

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
        self.T            = T
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
            0, len(self.X_background), size=self.sample_size)
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
        print('  [{}]  {} coalitions x {} samples x {} periods ...'.format(
            self.game_type, self.n_coalitions,
            self.sample_size, self.T))
        self.values = np.zeros((self.n_coalitions, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i + 1) % 32 == 0 or i + 1 == self.n_coalitions:
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
    shapley = {i: np.zeros(T) for i in range(n_players)}
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

def kernel_ou(t, length_scale=4.0):
    """
    OU kernel with ell=4 periods (2 hours).
    At 21 periods lag (morning to evening peak ~10.5h):
    K = exp(-21/4) ~ 0.005 -- peaks effectively independent under OU.
    """
    return np.exp(-np.abs(t[:, None] - t[None, :]) / length_scale)

def kernel_causal(t, length_scale=4.0):
    d = t[:, None] - t[None, :]
    return np.where(d >= 0, np.exp(-d / length_scale), 0.0)

def kernel_output_correlation(Y_raw):
    """
    Empirical cross-period correlation kernel.
    Key property: strong AM×PM off-diagonal block because both peaks
    co-vary with temperature, season, and day-type across thousands of days.
    """
    C   = np.cov(Y_raw.T)
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    K   = np.clip(C / np.outer(std, std), -1.0, 1.0)

    # Report key correlations
    am_mid = (MORNING_PEAK[0] + MORNING_PEAK[1]) // 2
    pm_mid = (EVENING_PEAK[0] + EVENING_PEAK[1]) // 2
    ov_mid = (OVERNIGHT[0] + OVERNIGHT[1]) // 2

    offdiag = K.copy()
    np.fill_diagonal(offdiag, 0.0)
    print('  Correlation kernel diagnostics:')
    print('    off-diagonal mean          = {:.3f}'.format(offdiag.mean()))
    print('    K[AM peak, PM peak]        = {:.3f}  ({} <-> {})'.format(
        K[am_mid, pm_mid],
        PERIOD_LABELS[am_mid], PERIOD_LABELS[pm_mid]))
    print('    K[overnight, AM peak]      = {:.3f}  ({} <-> {})'.format(
        K[ov_mid, am_mid],
        PERIOD_LABELS[ov_mid], PERIOD_LABELS[am_mid]))
    print('    K[AM peak, overnight]      = {:.3f}  ({} <-> {})'.format(
        K[am_mid, ov_mid],
        PERIOD_LABELS[am_mid], PERIOD_LABELS[ov_mid]))
    print('    OU K[AM, PM] by comparison = {:.3f}'.format(
        float(np.exp(-abs(am_mid - pm_mid) / 4.0))))
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

# Sparse ticks: every 4 periods = 2 hours
XTICK_IDXS   = list(range(0, T, 4))
XTICK_LABELS = [PERIOD_LABELS[i] for i in XTICK_IDXS]

def _set_time_axis(ax, sparse=False):
    step = 2 if sparse else 1
    ax.set_xticks(XTICK_IDXS[::step])
    ax.set_xticklabels(
        XTICK_LABELS[::step],
        rotation=45, ha='right', fontsize=6)
    ax.set_xlim(-0.5, T - 0.5)

def _period_shade(ax):
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
        sharey='row')
    fig.suptitle(
        'Operator sweep: Shapley curves across games and kernels\n'
        'Typical winter weekday  —  Random Forest  —  NESO GB Demand',
        fontsize=12, fontweight='bold')

    for r, gtype in enumerate(game_types):
        shapley = shapley_hv[gtype]
        for c, kname in enumerate(kernel_names):
            ax = axes[r, c]
            K  = kernels[kname]
            for fi, col, lbl in zip(focus, colors, flabels):
                ax.plot(t_grid, apply_kernel(shapley[fi], K),
                        color=col, lw=2.0, label=lbl)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            _set_time_axis(ax, sparse=True)
            ax.tick_params(axis='y', labelsize=7)
            if r == 0:
                ax.set_title(KERNEL_LABELS[kname],
                             fontsize=9, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
                ax.text(-0.42, 0.5, GAME_LABELS[gtype],
                        transform=ax.transAxes,
                        fontsize=7.5, va='center', ha='right',
                        rotation=90, color='#333')
            ax.set_xlabel('Time', fontsize=7)
            if r == 0 and c == 0:
                ax.text(0.97, 0.97, '= pointwise\nSHAP',
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
            color='#e24a4a', alpha=0.3, label='Evening peak (17-20h)'),
    ]
    fig.legend(handles=handles, loc='upper right',
               bbox_to_anchor=(0.99, 0.99),
               fontsize=8, framealpha=0.9)
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    return fig


# ===========================================================================
# 9.  Figure 2 -- Correlation kernel structure
# ===========================================================================

def plot_correlation_kernel_structure(K_corr, K_ou):
    fig = plt.figure(figsize=(17, 5.0))
    gs  = fig.add_gridspec(
        1, 5, width_ratios=[1, 1, 0.06, 1.1, 1.1], wspace=0.40)
    ax_corr = fig.add_subplot(gs[0])
    ax_ou   = fig.add_subplot(gs[1])
    ax_cb   = fig.add_subplot(gs[2])
    ax_ram  = fig.add_subplot(gs[3])   # row at AM peak
    ax_rov  = fig.add_subplot(gs[4])   # row at overnight

    fig.suptitle(
        'Cross-period correlation structure — GB National Demand\n'
        'Empirical kernel discovers AM$\\leftrightarrow$PM coupling; '
        'OU kernel misses it',
        fontsize=11, fontweight='bold')

    tick_i   = list(range(0, T, 8))
    tick_lbl = [PERIOD_LABELS[i] for i in tick_i]
    vmin, vmax = -0.2, 1.0

    # ── Empirical correlation heatmap ─────────────────────────────────────
    im = ax_corr.imshow(K_corr, aspect='auto', origin='upper',
                        cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax_corr.set_title(
        'Empirical correlation\n$K(t,s)=\\mathrm{Corr}(Y_t,Y_s)$',
        fontsize=9, fontweight='bold')

    for ax_ in [ax_corr, ax_ou]:
        ax_.set_xticks(tick_i)
        ax_.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=6)
        ax_.set_yticks(tick_i)
        ax_.set_yticklabels(tick_lbl, fontsize=6)
        ax_.set_xlabel('Period $s$', fontsize=8)
        ax_.set_ylabel('Period $t$', fontsize=8)

    am_s, am_e = MORNING_PEAK
    pm_s, pm_e = EVENING_PEAK
    for (r0, r1), (c0, c1), col, lbl in [
        (MORNING_PEAK, MORNING_PEAK, '#4a90e2', 'AM×AM'),
        (EVENING_PEAK, EVENING_PEAK, '#e24a4a', 'PM×PM'),
        (MORNING_PEAK, EVENING_PEAK, '#9b59b6', 'AM×PM'),
        (EVENING_PEAK, MORNING_PEAK, '#9b59b6', ''),
    ]:
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

    # ── OU kernel heatmap ─────────────────────────────────────────────────
    ax_ou.imshow(K_ou, aspect='auto', origin='upper',
                 cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax_ou.set_title(
        'OU kernel\n$K(t,s)=e^{-|t-s|/4}$ (2h)',
        fontsize=9, fontweight='bold')
    am_mid = (am_s + am_e) // 2
    pm_mid = (pm_s + pm_e) // 2
    ax_ou.text(0.5, 0.02,
               'AM$\\leftrightarrow$PM: '
               '$K=e^{{-{}/4}}\\approx{:.3f}$'.format(
                   abs(am_mid - pm_mid),
                   float(np.exp(-abs(am_mid - pm_mid) / 4.0))),
               transform=ax_ou.transAxes,
               fontsize=7, ha='center', va='bottom', color='#888',
               bbox=dict(boxstyle='round,pad=0.2',
                         fc='white', ec='#ccc', alpha=0.9))

    # Colorbar
    plt.colorbar(im, cax=ax_cb)
    ax_cb.set_ylabel('Correlation', fontsize=7)
    ax_cb.tick_params(labelsize=7)

    # ── Row slice at AM peak ───────────────────────────────────────────────
    _plot_kernel_row(
        ax_ram, K_corr, K_ou, am_mid,
        title='Row at $t={}$ (AM peak)\n'
              'Corr. kernel shows PM coupling'.format(
                  PERIOD_LABELS[am_mid]),
        annotate_pm=True)

    # ── Row slice at overnight ─────────────────────────────────────────────
    ov_mid = (OVERNIGHT[0] + OVERNIGHT[1]) // 2
    _plot_kernel_row(
        ax_rov, K_corr, K_ou, ov_mid,
        title='Row at $t={}$ (overnight)\n'
              'Both kernels agree: isolated'.format(
                  PERIOD_LABELS[ov_mid]),
        annotate_pm=False)

    plt.tight_layout()
    return fig


def _plot_kernel_row(ax, K_corr, K_ou, row_i,
                     title='', annotate_pm=False):
    ax.plot(t_grid, K_corr[row_i, :],
            color='#2a9d8f', lw=2.2, label='Empirical correlation')
    ax.plot(t_grid, K_ou[row_i, :],
            color='#e76f51', lw=2.2, ls='--',
            label='OU ($\\ell=4$)')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    for s, e, col in [
        (MORNING_PEAK[0], MORNING_PEAK[1], '#4a90e2'),
        (EVENING_PEAK[0], EVENING_PEAK[1], '#e24a4a'),
    ]:
        ax.axvspan(s, e, alpha=0.12, color=col)
    ax.axvline(row_i, color='gray', lw=0.8, ls=':', alpha=0.5)
    ax.set_xticks(XTICK_IDXS)
    ax.set_xticklabels(XTICK_LABELS, rotation=45, ha='right', fontsize=6)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xlabel('Period $s$', fontsize=8)
    ax.set_ylabel('$K(t={},\\ s)$'.format(PERIOD_LABELS[row_i]), fontsize=8)
    ax.set_title(title, fontsize=8.5, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.tick_params(labelsize=7)
    if annotate_pm:
        pm_mid = (EVENING_PEAK[0] + EVENING_PEAK[1]) // 2
        val    = K_corr[row_i, pm_mid]
        ax.annotate(
            'AM$\\leftrightarrow$PM\ncoupling\n(corr. only)',
            xy=(pm_mid, val),
            xytext=(pm_mid - 6, val + 0.08),
            fontsize=7, color='#9b59b6',
            arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.2))


# ===========================================================================
# 10.  Figure 3 -- Main effects, all games, identity kernel
# ===========================================================================

def plot_main_effects_all_games(moebius_dict, pnames, top_k=5):
    game_types = ['prediction', 'sensitivity', 'risk']
    fig, axes  = plt.subplots(
        len(game_types), 2,
        figsize=(13, 3.8 * len(game_types)),
        gridspec_kw={'width_ratios': [3, 1.5]})
    fig.suptitle(
        'Main effects $m_i(t)$ — Identity kernel\n'
        'Typical winter weekday  —  NESO GB Demand',
        fontsize=11, fontweight='bold')

    for r, gtype in enumerate(game_types):
        moebius = moebius_dict[gtype]
        imps    = {i: float(np.sum(np.abs(moebius[(i,)])))
                   for i in range(len(pnames))}
        top     = sorted(imps, key=imps.get, reverse=True)[:top_k]

        ax = axes[r, 0]
        for fi in top:
            ax.plot(t_grid, moebius[(fi,)],
                    color=FEAT_COLORS[pnames[fi]],
                    lw=2.0, label=pnames[fi])
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
        ax.set_xlabel('Time', fontsize=8)
        ax.set_title(GAME_LABELS[gtype], fontsize=9)
        if r == 0:
            ax.legend(fontsize=7, loc='upper left')

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
# ===========================================================================

def plot_weekend_causal(moebius_dict, pnames):
    fi_wkd     = pnames.index('is_weekend')
    game_types = ['prediction', 'risk']
    kernels_ordered = {
        'Identity'          : kernel_identity(t_grid),
        'OU ($\\ell$=4)'    : kernel_ou(t_grid, 4.0),
        'Causal $\\ell$=2'  : kernel_causal(t_grid, 2.0),
        'Causal $\\ell$=4'  : kernel_causal(t_grid, 4.0),
        'Causal $\\ell$=8'  : kernel_causal(t_grid, 8.0),
    }
    knames     = list(kernels_ordered.keys())
    causal_pal = plt.get_cmap('YlOrRd')(np.linspace(0.45, 0.85, 3))

    fig, axes = plt.subplots(
        len(game_types), len(knames),
        figsize=(3.2 * len(knames), 3.5 * len(game_types)),
        sharey='row')
    fig.suptitle(
        'is\\_weekend — symmetric vs causal kernel\n'
        'OU kernel smears attribution into overnight hours; '
        'causal kernel prevents this  —  NESO GB Demand',
        fontsize=11, fontweight='bold')

    for r, gtype in enumerate(game_types):
        raw = moebius_dict[gtype][(fi_wkd,)]
        for c, (kname, K) in enumerate(kernels_ordered.items()):
            ax    = axes[r, c]
            is_id = kname == 'Identity'
            is_ou = kname.startswith('OU')
            ci    = c - 2 if c >= 2 else 0
            col   = ('#444444' if is_id
                     else '#457b9d' if is_ou
                     else causal_pal[ci])
            ax.plot(t_grid, apply_kernel(raw, K), color=col, lw=2.2)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _period_shade(ax)
            ax.axvspan(OVERNIGHT[0], OVERNIGHT[1],
                       alpha=0.08, color='#555555', zorder=0)
            _set_time_axis(ax, sparse=True)
            ax.tick_params(axis='y', labelsize=7)
            if r == 0:
                ax.set_title(kname, fontsize=8.5, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
                ax.text(-0.42, 0.5, GAME_LABELS[gtype],
                        transform=ax.transAxes,
                        fontsize=7.5, va='center', ha='right',
                        rotation=90, color='#333')
            ax.set_xlabel('Time', fontsize=7)
            if is_ou and r == 0:
                ax.text(0.5, 0.97,
                        'overnight\nleakage',
                        transform=ax.transAxes,
                        fontsize=6.5, va='top', ha='center',
                        color='#c0392b',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  fc='#ffeaea', ec='#c0392b', alpha=0.9))

    plt.tight_layout(rect=[0.04, 0, 1, 1])
    return fig


# ===========================================================================
# 12.  Figure 5 -- Profile comparison
# ===========================================================================

def plot_profiles_comparison(profile_results, pnames):
    K_ou = kernel_ou(t_grid, 4.0)
    n    = len(profile_results)
    all_mob = {k: v[0] for k, v in profile_results.items()}
    top4    = _top_features(all_mob, len(pnames), top_k=4)

    profile_titles = {
        'Winter weekday' :
            'Typical winter weekday\n(Mon-Fri, Dec-Feb)',
        'Summer weekend' :
            'Summer weekend\n(Sat-Sun, Jun-Aug)',
        'Cold snap weekday':
            'Cold snap weekday\n(winter, high prior demand)',
    }

    fig, axes = plt.subplots(
        1, n, figsize=(5.5 * n, 4.2), sharey=False)
    fig.suptitle(
        'Shapley curves — OU kernel ($\\ell=4$ periods = 2h) — prediction game\n'
        'Three demand-regime profiles  —  NESO GB Demand',
        fontsize=11, fontweight='bold')

    for ax, (label, (moebius, shapley)) in zip(axes, profile_results.items()):
        for fi in top4:
            ax.plot(t_grid, apply_kernel(shapley[fi], K_ou),
                    color=FEAT_COLORS[pnames[fi]],
                    lw=2.0, label=pnames[fi])
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax)
        _set_time_axis(ax)
        ax.set_title(profile_titles.get(label, label),
                     fontsize=9, fontweight='bold')
        ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=8)
        ax.set_xlabel('Time', fontsize=8)
        ax.legend(fontsize=7, loc='upper left')
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


# ===========================================================================
# 13.  Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  NESO GB Demand  —  RF direct  (paper figures)')
    print('=' * 60)

    _require_dir(BASE_PLOT_DIR)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print('\n[1] Loading data ...')
    X_day, Y_raw, Y_adj, diurnal, dates = load_and_aggregate()
    X_day_np = X_day.to_numpy().astype(float)
    pnames   = list(DAY_FEATURE_NAMES)

    print('\n  Diurnal mean ND profile (selected periods):')
    for i in [0, 6, 12, 16, 24, 34, 38, 47]:
        print('    {}  {:.0f} MW'.format(PERIOD_LABELS[i], diurnal[i]))

    # ── 2. Model ──────────────────────────────────────────────────────────
    print('\n[2] Fitting Random Forest ...')
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_day_np, Y_adj, test_size=0.2, random_state=RNG_SEED)
    model = RFModel()
    model.fit(X_tr, Y_tr)
    r2 = model.evaluate(X_te, Y_te)
    print('  Test R² (trajectory-level): {:.4f}'.format(r2))

    # ── 3. Kernels ────────────────────────────────────────────────────────
    print('\n[3] Building kernels ...')
    K_corr = kernel_output_correlation(Y_raw)
    K_ou_k = kernel_ou(t_grid, length_scale=4.0)
    kernels = {
        'Identity'   : kernel_identity(t_grid),
        'OU'         : K_ou_k,
        'Correlation': K_corr,
        'Causal'     : kernel_causal(t_grid, length_scale=4.0),
    }

    # ── 4. Profiles ───────────────────────────────────────────────────────
    print('\n[4] Selecting profiles ...')
    fi_wkd = pnames.index('is_weekend')
    fi_sea = pnames.index('season')
    fi_lag = pnames.index('lag_daily_mean')

    lag_p75 = float(np.percentile(X_day_np[:, fi_lag], 75))

    def find_profile(conditions, label):
        mask = np.ones(len(X_day_np), dtype=bool)
        for feat, (lo, hi) in conditions.items():
            ci    = pnames.index(feat)
            mask &= (X_day_np[:, ci] >= lo) & (X_day_np[:, ci] <= hi)
        hits = X_day_np[mask]
        if len(hits) == 0:
            raise RuntimeError(
                'No day matches "{}": {}'.format(label, conditions))
        print('  "{}": {} matching days.'.format(label, len(hits)))
        return hits[len(hits) // 2]

    x_winter = find_profile(
        {'is_weekend': (-0.1, 0.1), 'season': (0.9, 1.1)},
        'Winter weekday')

    x_summer_wkd = find_profile(
        {'is_weekend': (0.9, 1.1), 'season': (2.9, 3.1)},
        'Summer weekend')

    x_coldsnap = find_profile(
        {'is_weekend': (-0.1, 0.1),
         'season': (0.9, 1.1),
         'lag_daily_mean': (lag_p75, 9e9)},
        'Cold snap weekday')

    def _y_obs(x_prof):
        diffs = np.abs(X_day_np - x_prof[None, :]).sum(axis=1)
        return Y_adj[int(np.argmin(diffs))]

    profile_defs = [
        ('Winter weekday',   x_winter,     _y_obs(x_winter)),
        ('Summer weekend',   x_summer_wkd, _y_obs(x_summer_wkd)),
        ('Cold snap weekday', x_coldsnap,  _y_obs(x_coldsnap)),
    ]

    for lbl, xp, _ in profile_defs:
        print('  {}: {}'.format(lbl, '  '.join(
            '{}={:.1f}'.format(n, xp[j])
            for j, n in enumerate(pnames))))

    # ── 5. All three games for primary profile ────────────────────────────
    print('\n[5] Computing all games (Winter weekday) ...')
    x_prim, y_prim = x_winter, _y_obs(x_winter)
    moebius_prim   = {}
    shapley_prim   = {}

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
    print('\n[6] Prediction game for all profiles ...')
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

    savefig(plot_operator_sweep(
        moebius_prim, shapley_prim, kernels, pnames),
        'fig1_operator_sweep.pdf')

    savefig(plot_correlation_kernel_structure(K_corr, K_ou_k),
        'fig2_correlation_kernel.pdf')

    savefig(plot_main_effects_all_games(moebius_prim, pnames),
        'fig3_main_effects_all_games.pdf')

    savefig(plot_weekend_causal(moebius_prim, pnames),
        'fig4_weekend_causal.pdf')

    savefig(plot_profiles_comparison(profile_results, pnames),
        'fig5_profiles_comparison.pdf')

    print('\n' + '=' * 60)
    print('  Done.  Figures saved under {}/'.format(BASE_PLOT_DIR))
    print('  fig1_operator_sweep.pdf        -- central paper figure')
    print('  fig2_correlation_kernel.pdf    -- AM/PM coupling motivation')
    print('  fig3_main_effects_all_games.pdf')
    print('  fig4_weekend_causal.pdf')
    print('  fig5_profiles_comparison.pdf')
    print('=' * 60)