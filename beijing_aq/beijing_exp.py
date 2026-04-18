"""
Functional Explanation Framework -- Beijing PM2.5 Air Quality
=============================================================
Dataset : UCI Beijing Multi-Site Air Quality Data (id=501)
          Fetched automatically via ucimlrepo.
          12 stations, hourly PM2.5 + meteorology, 2013-03-01 to 2017-02-28.
          Cached to ./data/beijing/beijing_air_quality.parquet

Station : US Embassy Beijing (single station, id=381, 2010-2014)
          Columns: pm2.5, TEMP, PRES, DEWP, cbwd, Iws, Is, Ir

Output  : F^H : R^8 -> R^24
  Daily 24-hour log(PM2.5+1) trajectory, hourly means, T=24.
  Log-transform applied before modelling because PM2.5 is right-skewed
  with extreme pollution events. Diurnal mean subtracted after log.
  Model: RandomForestRegressor, direct multi-output, no PCA, no t input.

Features (cooperative game players, all from previous day):
  month            integer 1-12  (seasonal heating/cooling cycle)
  is_heating       binary 1 if Nov-Mar (Beijing coal heating season)
                   -- the ann_indicator analogue for this dataset.
                   Heating is switched on ~Nov 15 city-wide, dramatically
                   elevating overnight and morning PM2.5 via coal combustion
  day_of_week      integer 0-6 (weekday traffic vs weekend)
  lag_pm25_mean    previous day mean log(PM2.5+1) -- pollution persistence
  lag_pm25_morning previous day 06-09h mean -- morning rush persistence
  wind_sin         sin(wind_direction_prev) -- NW wind clears Beijing
  wind_cos         cos(wind_direction_prev) -- S/SW brings Hebei pollution
  wspm_prev        previous day mean wind speed (m/s) -- dispersion

Why the correlation kernel is interesting here
----------------------------------------------
PM2.5 in Beijing has a pollution-regime-dependent intraday structure.
On high-pollution days (stagnant air, heating season): overnight and
morning hours are highly elevated and strongly correlated because stagnant
air traps pollution uniformly throughout the night and early morning.
Daytime hours show weaker relief due to limited photochemical breakdown.

On low-pollution days (high wind, summer): a clear rush-hour pattern
emerges (06:00-09:00 and 17:00-20:00 traffic peaks), similar to NESO
or IHEPC but driven by combustion rather than electricity demand.

The empirical correlation kernel estimated from all days will reflect the
dominant regime -- overnight and morning hours clustering together,
midday having weaker coupling. This produces qualitatively different
attributions for regime features (is_heating, lag_pm25_mean) compared
to the identity kernel.

The causal kernel argument for is_heating
------------------------------------------
The heating season indicator is binary and city-wide: coal heating is
switched on uniformly across Beijing. Its effect on PM2.5 is concentrated
in overnight and early morning hours (stagnant cold air + heating demand),
not in the afternoon when mixing height increases. A symmetric kernel
(OU, Gaussian) would smear attribution backward or forward past the
diurnal window where the heating effect is physically active. The causal
kernel correctly restricts attribution to the hours when overnight
heating emissions have accumulated.

Games     : prediction, sensitivity, risk
Kernels   : Identity, Correlation, Causal (ell=4h, on is_heating)
Profiles  : Heavy pollution winter day / Clean summer day /
            Heating season onset (first heating day)

Figures
-------
  fig1_diurnal_and_correlation.pdf  -- mean PM2.5 curve + correlation matrix
  fig2_main_effects.pdf             -- identity vs correlation kernel,
                                        all three games
  fig3_is_heating_causal.pdf        -- is_heating under identity / correlation
                                        / causal kernels, prediction + risk
  fig4_profiles_comparison.pdf      -- 3 profiles, identity vs correlation
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
DATA_FILE = os.path.join(DATA_DIR, 'beijing_pm25_embassy.parquet')

BASE_PLOT_DIR = os.path.join(
    'plots', 'beijing')

RNG_SEED = 42
T        = 24
dt       = 1.0

RF_N_EST = 300
RF_JOBS  = -1

# Use Dongsi station -- central urban, well-studied
STATION = 'US_Embassy'  # single-station dataset, label only

SAMPLE_SIZE = {'prediction': 150, 'sensitivity': 200, 'risk': 200}

HOUR_LABELS = ['{:02d}:00'.format(h) for h in range(T)]
t_grid      = np.arange(T, dtype=float)

# Beijing heating season: Nov (11) through Mar (3)
HEATING_MONTHS = {11, 12, 1, 2, 3}

DAY_FEATURE_NAMES = [
    'month',            # 1-12: seasonal cycle
    'is_heating',       # binary: Nov-Mar coal heating season
    'day_of_week',      # 0-6
    'lag_pm25_mean',    # previous day mean log(PM2.5+1)
    'lag_pm25_morning', # previous day 06-09h mean
    'wind_sin',         # sin(prev day wind direction in radians)
    'wind_cos',         # cos(prev day wind direction in radians)
    'wspm_prev',        # previous day mean wind speed (m/s)
]

# Intraday phase windows (hour indices)
OVERNIGHT   = (0,  6)    # 00:00-05:00 -- stagnant air accumulation
MORNING     = (6,  10)   # 06:00-09:00 -- morning rush + heating peak
AFTERNOON   = (11, 16)   # 11:00-15:00 -- mixed layer, lower pollution
EVENING     = (17, 21)   # 17:00-20:00 -- evening rush

FEAT_COLORS = {
    'month'            : '#2ca02c',
    'is_heating'       : '#ff7f0e',
    'day_of_week'      : '#1f77b4',
    'lag_pm25_mean'    : '#d62728',
    'lag_pm25_morning' : '#9467bd',
    'wind_sin'         : '#8c564b',
    'wind_cos'         : '#e377c2',
    'wspm_prev'        : '#17becf',
}

GAME_YLABEL = {
    'prediction' : r'Effect on log(PM2.5+1)',
    'sensitivity': r'Var$[F(t)]$',
    'risk'       : r'Effect on MSE',
}

GAME_TITLE = {
    'prediction' :
        r'Prediction  $v(S)(t)=\mathbb{E}[F(x)(t)\mid X_S]$',
    'sensitivity':
        r'Sensitivity  $v(S)(t)=\mathrm{Var}[F(x)(t)\mid X_S]$',
    'risk'       :
        r'Risk (MSE)  $v(S)(t)=\mathbb{E}[(Y(t)-F(x)(t))^2\mid X_S]$',
}


# ===========================================================================
# 1.  Data loading
# ===========================================================================

def _require_dir(path):
    os.makedirs(path, exist_ok=True)


def _wind_dir_to_radians(wd_series):
    """
    Convert categorical wind direction strings to radians.
    Returns (sin, cos) as two float series.
    Handles NA as (0, 0) -- calm/no wind.
    """
    dir_map = {
        'N':   0.0,   'NNE': np.pi/8,   'NE':  np.pi/4,
        'ENE': 3*np.pi/8, 'E': np.pi/2, 'ESE': 5*np.pi/8,
        'SE':  3*np.pi/4, 'SSE': 7*np.pi/8, 'S': np.pi,
        'SSW': 9*np.pi/8, 'SW': 5*np.pi/4,  'WSW': 11*np.pi/8,
        'W':  3*np.pi/2,  'WNW': 13*np.pi/8,'NW':  7*np.pi/4,
        'NNW': 15*np.pi/8,
    }
    rads = wd_series.map(dir_map).fillna(0.0)
    return np.sin(rads), np.cos(rads)


def _fetch_and_cache():
    import importlib
    if importlib.util.find_spec('ucimlrepo') is None:
        raise RuntimeError('pip install ucimlrepo')
    from ucimlrepo import fetch_ucirepo
    print('  Fetching Beijing PM2.5 US Embassy (id=381) ...')
    print('  Single station, 2010-2014, ~17500 rows.')
    ds = fetch_ucirepo(id=381)

    # Features (meteorology) + Targets (PM2.5) must be joined
    features = ds.data.features.copy()
    targets  = ds.data.targets.copy()

    print('  Feature columns : {}'.format(list(features.columns)))
    print('  Target  columns : {}'.format(list(targets.columns)))

    raw = pd.concat([features, targets], axis=1)

    # Rename to consistent names used in the rest of the script
    rename = {
        'pm2.5': 'PM2.5',
        'cbwd' : 'wd',
        'Iws'  : 'WSPM',
        'TEMP' : 'TEMP',
        'PRES' : 'PRES',
        'DEWP' : 'DEWP',
    }
    raw = raw.rename(columns={k: v for k, v in rename.items()
                               if k in raw.columns})

    # Coerce numeric columns
    skip = {'wd'}
    for col in raw.columns:
        if col not in skip:
            raw[col] = pd.to_numeric(raw[col], errors='coerce')

    if 'WSPM' in raw.columns:
        raw['WSPM'] = raw['WSPM'].clip(lower=0)

    # Verify PM2.5 exists after join + rename
    if 'PM2.5' not in raw.columns:
        raise RuntimeError(
            'PM2.5 column not found after join.\n'
            'All columns: {}\n'
            'Check ds.data.targets column names above.'.format(
                list(raw.columns))
        )

    print('  Downloaded: {:,} rows, columns: {}'.format(
        len(raw), list(raw.columns)))
    _require_dir(DATA_DIR)
    raw.to_parquet(DATA_FILE, index=False)
    print('  Cached to {}'.format(DATA_FILE))
    return raw


def load_and_aggregate():
    """
    Load Beijing air quality, filter to STATION, aggregate to daily
    24h log(PM2.5+1) curves, engineer day-level features.

    Returns
    -------
    X_day   : pd.DataFrame  (n_days, 8)
    Y_raw   : np.ndarray    (n_days, 24)  hourly mean log(PM2.5+1)
    Y_adj   : np.ndarray    (n_days, 24)  diurnal-subtracted
    diurnal : np.ndarray    (24,)
    dates   : list of str
    """
    if os.path.isfile(DATA_FILE):
        print('  Loading parquet cache ...')
        df = pd.read_parquet(DATA_FILE)
    else:
        df = _fetch_and_cache()

    # ── Defensive: normalise column names regardless of cache state ──────
    rename_map = {
        'pm2.5': 'PM2.5',
        'cbwd' : 'wd',
        'Iws'  : 'WSPM',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items()
                             if k in df.columns})

    # ── Verify required columns exist ────────────────────────────────────
    required = ['year', 'month', 'day', 'hour', 'PM2.5']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            'Missing columns after rename: {}.\n'
            'Available: {}\n'
            'Delete the cache and re-run:\n'
            '  rm {}'.format(missing, list(df.columns), DATA_FILE)
        )

    print('  Single-station dataset: {:,} rows'.format(len(df)))

    # Build datetime and date/hour columns
    df['datetime'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour']]
        .rename(columns={'hour': 'hour'}),
        errors='coerce')
    df = df.dropna(subset=['datetime', 'PM2.5'])
    df['date'] = df['datetime'].dt.date.astype(str)
    df['hour'] = df['datetime'].dt.hour

    # Log-transform PM2.5
    df['logpm25'] = np.log1p(df['PM2.5'].clip(lower=0))

    # Aggregate hourly means of logpm25 and met variables
    agg = df.groupby(['date', 'hour']).agg(
        logpm25 =('logpm25', 'mean'),
        TEMP    =('TEMP',    'mean'),
        PRES    =('PRES',    'mean'),
        WSPM    =('WSPM',    'mean'),
        wd      =('wd',      lambda x: x.mode()[0] if len(x) > 0 else np.nan),
    ).reset_index()

    # Pivot log PM2.5 to (n_days, 24)
    pivot = agg.pivot(index='date', columns='hour', values='logpm25')
    pivot = pivot.reindex(columns=range(T))
    pivot = pivot[pivot.notna().sum(axis=1) >= 20]   # allow up to 4 missing
    pivot = pivot.fillna(pivot.mean())

    Y_raw  = pivot.values.astype(float)
    dates  = pivot.index.tolist()
    diurnal = Y_raw.mean(axis=0)
    Y_adj  = Y_raw - diurnal[None, :]

    print('  Complete days: {}  ({} to {})'.format(
        len(dates), dates[0], dates[-1]))
    print('  Mean log(PM2.5+1): {:.3f}  '
          '(= PM2.5 ~ {:.0f} ug/m3)'.format(
              diurnal.mean(),
              np.expm1(diurnal.mean())))

    # ── Day-level features ────────────────────────────────────────────────
    # Build daily met summaries for lagged features
    daily_wspm = agg.groupby('date')['WSPM'].mean()
    daily_wd   = agg.groupby('date')['wd'].agg(
        lambda x: x.mode()[0] if len(x) > 0 else 'N')

    records = []
    for i, date_str in enumerate(dates):
        dt_obj = pd.Timestamp(date_str)
        m, dow = dt_obj.month, dt_obj.dayofweek

        if i == 0:
            lag_mean = float(Y_raw.mean())
            lag_morn = float(Y_raw[:, MORNING[0]:MORNING[1]].mean())
            lag_wspm = float(daily_wspm.mean())
            lag_wd   = 'N'
        else:
            prev = dates[i - 1]
            lag_mean = float(Y_raw[i - 1].mean())
            lag_morn = float(Y_raw[i - 1,
                                   MORNING[0]:MORNING[1]].mean())
            lag_wspm = float(daily_wspm.get(prev, daily_wspm.mean()))
            lag_wd   = daily_wd.get(prev, 'N')

        w_sin_val, w_cos_val = _wind_dir_to_radians(
            pd.Series([lag_wd]))
        records.append({
            'month'            : float(m),
            'is_heating'       : float(m in HEATING_MONTHS),
            'day_of_week'      : float(dow),
            'lag_pm25_mean'    : lag_mean,
            'lag_pm25_morning' : lag_morn,
            'wind_sin'         : float(w_sin_val.iloc[0]),
            'wind_cos'         : float(w_cos_val.iloc[0]),
            'wspm_prev'        : lag_wspm,
        })

    X_day = pd.DataFrame(records, index=dates)
    for col in DAY_FEATURE_NAMES:
        assert col in X_day.columns
        assert X_day[col].std() > 1e-6, 'Constant: {}'.format(col)

    print('  Heating season days: {} ({:.1f}%)'.format(
        int(X_day['is_heating'].sum()),
        X_day['is_heating'].mean() * 100))

    return X_day, Y_raw, Y_adj, diurnal, dates

# ===========================================================================
# 2.  Model
# ===========================================================================

class RFModel:
    def __init__(self, random_state=RNG_SEED):
        self.model = RandomForestRegressor(
            n_estimators=RF_N_EST, n_jobs=RF_JOBS,
            random_state=random_state)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_te, Y_te):
        Yp = self.predict(X_te)
        return 1.0 - (np.sum((Y_te-Yp)**2) /
                      np.sum((Y_te-Y_te.mean())**2))


# ===========================================================================
# 3.  Cooperative game
# ===========================================================================

class FunctionalGame:
    def __init__(self, predict_fn, X_bg, x_exp,
                 game_type='prediction', Y_obs=None,
                 sample_size=150, random_seed=RNG_SEED):
        if game_type == 'risk' and Y_obs is None:
            raise ValueError('Y_obs required for risk.')
        self.predict_fn = predict_fn
        self.X_bg       = X_bg
        self.x_exp      = x_exp
        self.game_type  = game_type
        self.Y_obs      = Y_obs
        self.n          = sample_size
        self.seed       = random_seed
        self.T          = T
        self.p          = len(DAY_FEATURE_NAMES)
        self.player_names = list(DAY_FEATURE_NAMES)
        self.coalitions = np.array(
            list(itertools.product([False, True], repeat=self.p)),
            dtype=bool)
        self.nc   = len(self.coalitions)
        self._idx = {tuple(c): i for i, c in enumerate(self.coalitions)}
        self.values = None

    def _impute(self, coal):
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(self.X_bg), size=self.n)
        X   = self.X_bg[idx].copy()
        for j in range(self.p):
            if coal[j]:
                X[:, j] = self.x_exp[j]
        return X

    def value_function(self, coal):
        X  = self._impute(coal)
        Yp = self.predict_fn(X)
        if self.game_type == 'prediction':
            return Yp.mean(axis=0)
        elif self.game_type == 'sensitivity':
            return Yp.var(axis=0)
        else:
            return ((self.Y_obs[None, :] - Yp)**2).mean(axis=0)

    def precompute(self):
        print('  [{}]  {} coalitions x {} samples ...'.format(
            self.game_type, self.nc, self.n))
        self.values = np.zeros((self.nc, self.T))
        for i, c in enumerate(self.coalitions):
            self.values[i] = self.value_function(tuple(c))
            if (i+1) % 32 == 0 or i+1 == self.nc:
                print('    {}/{}'.format(i+1, self.nc))

    def __getitem__(self, c):
        return self.values[self._idx[c]]


# ===========================================================================
# 4.  Mobius + Shapley
# ===========================================================================

def moebius_transform(game):
    p     = game.p
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(p), r)
        for r in range(p+1)))
    mob = {}
    for S in all_S:
        m = np.zeros(game.T)
        for L in itertools.chain.from_iterable(
            itertools.combinations(S, r)
            for r in range(len(S)+1)
        ):
            c = tuple(i in L for i in range(p))
            m += (-1)**(len(S)-len(L)) * game[c]
        mob[S] = m
    return mob


def shapley_values(mob, p):
    shap = {i: np.zeros(T) for i in range(p)}
    for S, m in mob.items():
        if len(S) == 0:
            continue
        for i in S:
            shap[i] += m / len(S)
    return shap


# ===========================================================================
# 5.  Kernels
# ===========================================================================

def kernel_identity(t):
    return np.eye(len(t))

def kernel_ou(t, length_scale=4.0):
    return np.exp(-np.abs(t[:, None] - t[None, :]) / length_scale)

def kernel_causal(t, length_scale=4.0):
    d = t[:, None] - t[None, :]
    return np.where(d >= 0, np.exp(-d / length_scale), 0.0)

def kernel_correlation(Y_raw):
    C   = np.cov(Y_raw.T)
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    K   = np.clip(C / np.outer(std, std), -1.0, 1.0)
    ov  = (OVERNIGHT[0] + OVERNIGHT[1]) // 2
    mo  = (MORNING[0]   + MORNING[1])   // 2
    af  = (AFTERNOON[0] + AFTERNOON[1]) // 2
    print('  Correlation kernel diagnostics:')
    print('    K[overnight, morning]   = {:.3f}'.format(K[ov, mo]))
    print('    K[overnight, afternoon] = {:.3f}'.format(K[ov, af]))
    print('    K[morning, afternoon]   = {:.3f}'.format(K[mo, af]))
    return K

def apply_kernel(effect, K):
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return (K / rs) @ effect * dt


# ===========================================================================
# 6.  Plotting helpers
# ===========================================================================

XTICK_IDXS   = list(range(0, T, 3))
XTICK_LABELS = [HOUR_LABELS[i] for i in XTICK_IDXS]

def _set_time_axis(ax, sparse=False):
    step = 2 if sparse else 1
    ax.set_xticks(XTICK_IDXS[::step])
    ax.set_xticklabels(
        XTICK_LABELS[::step], rotation=45, ha='right', fontsize=6)
    ax.set_xlim(-0.5, T - 0.5)

def _phase_shade(ax):
    ax.axvspan(*OVERNIGHT,  alpha=0.10, color='#555555', zorder=0)
    ax.axvspan(*MORNING,    alpha=0.10, color='#4a90e2', zorder=0)
    ax.axvspan(*AFTERNOON,  alpha=0.10, color='#ffbb78', zorder=0)
    ax.axvspan(*EVENING,    alpha=0.10, color='#e24a4a', zorder=0)

def savefig(fig, name):
    path = os.path.join(BASE_PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print('  Saved: {}'.format(path))
    plt.close(fig)

def _top_k(mob, p, k=5):
    imps = np.array([float(np.sum(np.abs(mob[(i,)]))) for i in range(p)])
    return sorted(range(p), key=lambda i: imps[i], reverse=True)[:k]


# ===========================================================================
# 7.  Figure 1 -- Diurnal PM2.5 curve + correlation matrix
# ===========================================================================

def fig_diurnal_and_correlation(diurnal, Y_raw, K_corr):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        'Daily PM2.5 cycle — Beijing US Embassy station 2010-2014\n'
        'Overnight/morning stagnation cluster motivates the correlation kernel',
        fontsize=11, fontweight='bold')

    # Panel 1: diurnal curve
    ax = axes[0]
    pm25_diurnal = np.expm1(diurnal)
    ax.plot(t_grid, pm25_diurnal, color='#8B0000', lw=2.5)
    ax.fill_between(t_grid, 0, pm25_diurnal, alpha=0.15, color='#8B0000')
    _phase_shade(ax)
    _set_time_axis(ax)
    ax.set_ylabel(r'Mean PM2.5 ($\mu$g/m$^3$)', fontsize=9)
    ax.set_xlabel('Hour', fontsize=8)
    ax.set_title('Mean diurnal PM2.5 curve\n({} days)'.format(len(Y_raw)),
                 fontsize=9, fontweight='bold')
    tmin = int(np.argmin(pm25_diurnal))
    tmax = int(np.argmax(pm25_diurnal))
    ax.annotate('Min\n{:.0f}'.format(pm25_diurnal[tmin]),
                xy=(tmin, pm25_diurnal[tmin]),
                xytext=(tmin + 2, pm25_diurnal[tmin] - 8),
                fontsize=7.5, color='steelblue',
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.0))
    ax.annotate('Max\n{:.0f}'.format(pm25_diurnal[tmax]),
                xy=(tmax, pm25_diurnal[tmax]),
                xytext=(tmax - 4, pm25_diurnal[tmax] + 5),
                fontsize=7.5, color='#8B0000',
                arrowprops=dict(arrowstyle='->', color='#8B0000', lw=1.0))
    ax.tick_params(labelsize=7)

    # Panel 2: empirical correlation matrix
    ax2 = axes[1]
    tick_i   = list(range(0, T, 4))
    tick_lbl = [HOUR_LABELS[i] for i in tick_i]
    im = ax2.imshow(K_corr, aspect='auto', origin='upper',
                    cmap='RdBu_r', vmin=-0.2, vmax=1.0)
    ax2.set_xticks(tick_i)
    ax2.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=6)
    ax2.set_yticks(tick_i)
    ax2.set_yticklabels(tick_lbl, fontsize=6)
    ax2.set_xlabel('Hour $s$', fontsize=8)
    ax2.set_ylabel('Hour $t$', fontsize=8)
    ax2.set_title('Empirical cross-hour correlation\n'
                  '$K(t,s)=\\mathrm{Corr}(Y_t, Y_s)$',
                  fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(
        labelsize=6)

    # Annotate overnight-morning block
    for (r0,r1),(c0,c1),ec,lbl in [
        (OVERNIGHT, OVERNIGHT, '#555555', 'ON×ON'),
        (MORNING,   MORNING,   '#4a90e2', 'AM×AM'),
        (OVERNIGHT, MORNING,   '#9b59b6', 'ON×AM'),
        (MORNING,   OVERNIGHT, '#9b59b6', ''),
    ]:
        rect = plt.Rectangle(
            (c0-0.5, r0-0.5), c1-c0, r1-r0,
            linewidth=1.2, edgecolor=ec, facecolor='none', zorder=3)
        ax2.add_patch(rect)
        if lbl:
            ax2.text((c0+c1)/2, (r0+r1)/2, lbl,
                     fontsize=5.5, ha='center', va='center',
                     color=ec, fontweight='bold')

    # Panel 3: row slice at overnight hour
    ax3 = axes[2]
    ov_mid = (OVERNIGHT[0] + OVERNIGHT[1]) // 2
    mo_mid = (MORNING[0]   + MORNING[1])   // 2
    af_mid = (AFTERNOON[0] + AFTERNOON[1]) // 2

    ax3.plot(t_grid, K_corr[ov_mid, :],
             color='#555555', lw=2.2,
             label='Row $t={}$ (overnight)'.format(
                 HOUR_LABELS[ov_mid]))
    ax3.plot(t_grid, K_corr[af_mid, :],
             color='#ffbb78', lw=2.2, ls='--',
             label='Row $t={}$ (afternoon)'.format(
                 HOUR_LABELS[af_mid]))
    ax3.axhline(0, color='gray', lw=0.5, ls=':')
    _phase_shade(ax3)
    _set_time_axis(ax3)
    ax3.set_xlabel('Hour $s$', fontsize=8)
    ax3.set_ylabel('$K(t,\\ s)$', fontsize=8)
    ax3.set_title('Kernel row slices\nOvernight vs afternoon',
                  fontsize=9, fontweight='bold')
    ax3.legend(fontsize=7.5)
    ax3.tick_params(labelsize=7)
    ax3.annotate(
        'ON high-corr\nwith morning\n(stagnation regime)',
        xy=(mo_mid, K_corr[ov_mid, mo_mid]),
        xytext=(mo_mid + 3, K_corr[ov_mid, mo_mid] + 0.07),
        fontsize=7, color='#9b59b6',
        arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.0))

    plt.tight_layout()
    return fig


# ===========================================================================
# 8.  Figure 2 -- Main effects: identity vs correlation kernel, all games
# ===========================================================================

def fig_main_effects(mob, pnames, K_corr):
    kernels = {
        'Identity\n(pointwise SHAP)': kernel_identity(t_grid),
        'Empirical correlation'      : K_corr,
    }
    game_types = ['prediction', 'sensitivity', 'risk']
    nrows, ncols = len(game_types), len(kernels)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, 3.5 * nrows),
        sharey='row')
    fig.suptitle(
        'Main effects $m_i(t)$ — Identity vs Correlation kernel\n'
        'Beijing US Embassy — heavy pollution winter day',
        fontsize=11, fontweight='bold')

    p   = len(pnames)
    top = _top_k(mob['prediction'], p, k=5)

    for r, gtype in enumerate(game_types):
        m = mob[gtype]
        for c, (kname, K) in enumerate(kernels.items()):
            ax = axes[r, c]
            for fi in top:
                curve = apply_kernel(m[(fi,)], K)
                ax.plot(t_grid, curve,
                        color=FEAT_COLORS[pnames[fi]],
                        lw=2.0, label=pnames[fi])
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _phase_shade(ax)
            _set_time_axis(ax)
            ax.tick_params(axis='y', labelsize=7)
            ax.set_xlabel('Hour', fontsize=7)
            if r == 0:
                ax.set_title(kname, fontsize=9.5, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
                ax.text(-0.22, 0.5, GAME_TITLE[gtype],
                        transform=ax.transAxes,
                        fontsize=7.5, va='center', ha='right',
                        rotation=90, color='#333')
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc='upper right')
                ax.text(0.02, 0.97, '= pointwise SHAP',
                        transform=ax.transAxes,
                        fontsize=6.5, va='top', color='#555',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  fc='white', ec='#aaa', alpha=0.8))

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    return fig


# ===========================================================================
# 9.  Figure 3 -- is_heating causal kernel comparison
#     Rows: prediction / risk
#     Cols: Identity / Correlation / Causal ell=3 / Causal ell=6
# ===========================================================================

def fig_is_heating_causal(mob, pnames):
    fi_h = pnames.index('is_heating')
    kernels_ordered = {
        'Identity'            : kernel_identity(t_grid),
        'Correlation'         : None,   # filled below
        'Causal\n$\\ell=3$h'  : kernel_causal(t_grid, 3.0),
        'Causal\n$\\ell=6$h'  : kernel_causal(t_grid, 6.0),
    }
    causal_pal = plt.get_cmap('YlOrRd')(np.linspace(0.45, 0.85, 2))

    fig, axes = plt.subplots(
        2, len(kernels_ordered),
        figsize=(3.5 * len(kernels_ordered), 3.5 * 2),
        sharey='row')
    fig.suptitle(
        'is\\_heating — Identity vs Correlation vs Causal kernel\n'
        'Heating season elevates overnight/morning PM2.5; '
        'symmetric kernels leak attribution to afternoon\n'
        'Beijing Dongsi',
        fontsize=11, fontweight='bold')

    return fig, kernels_ordered, fi_h, causal_pal


def _fill_is_heating_causal(fig, axes, mob, pnames, K_corr):
    fi_h = pnames.index('is_heating')
    kernels_ordered = {
        'Identity'           : kernel_identity(t_grid),
        'Correlation'        : K_corr,
        'Causal $\\ell$=3h'  : kernel_causal(t_grid, 3.0),
        'Causal $\\ell$=6h'  : kernel_causal(t_grid, 6.0),
    }
    causal_pal = plt.get_cmap('YlOrRd')(np.linspace(0.45, 0.85, 2))
    game_types = ['prediction', 'risk']

    for r, gtype in enumerate(game_types):
        raw = mob[gtype][(fi_h,)]
        for c, (kname, K) in enumerate(kernels_ordered.items()):
            ax     = axes[r, c]
            is_id  = kname == 'Identity'
            is_cor = kname == 'Correlation'
            ci     = c - 2 if c >= 2 else 0
            col    = ('#444444' if is_id
                      else '#2a9d8f' if is_cor
                      else causal_pal[ci])
            curve  = apply_kernel(raw, K)
            ax.plot(t_grid, curve, color=col, lw=2.2)
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _phase_shade(ax)
            _set_time_axis(ax, sparse=True)
            ax.tick_params(axis='y', labelsize=7)
            ax.set_xlabel('Hour', fontsize=7)
            if r == 0:
                ax.set_title(kname, fontsize=9, fontweight='bold')
            if c == 0:
                ax.set_ylabel(GAME_YLABEL[gtype], fontsize=8)
                ax.text(-0.32, 0.5, GAME_TITLE[gtype],
                        transform=ax.transAxes,
                        fontsize=7.5, va='center', ha='right',
                        rotation=90, color='#333')
            # Annotate afternoon leakage on correlation panel
            if is_cor and r == 0:
                ax.text(0.5, 0.97,
                        'afternoon\nleakage',
                        transform=ax.transAxes,
                        fontsize=6.5, va='top', ha='center',
                        color='#c0392b',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  fc='#ffeaea', ec='#c0392b',
                                  alpha=0.9))
    return fig


# ===========================================================================
# 10.  Figure 4 -- Profile comparison
#      Identity (top) vs Correlation (bottom), 3 profiles
# ===========================================================================

def fig_profiles_comparison(prof_results, pnames, K_corr):
    K_id = kernel_identity(t_grid)
    kernels_rows = [
        ('Identity kernel',             K_id),
        ('Empirical correlation kernel', K_corr),
    ]
    n_prof = len(prof_results)
    nrows  = len(kernels_rows)

    # Global top-4 across profiles
    all_mob = {lbl: mob for lbl, (mob, shap) in prof_results.items()}
    p    = len(pnames)
    imps = np.zeros(p)
    for mob in all_mob.values():
        for i in range(p):
            imps[i] += float(np.sum(np.abs(mob[(i,)])))
    top4 = sorted(range(p), key=lambda i: imps[i], reverse=True)[:4]

    fig, axes = plt.subplots(
        nrows, n_prof,
        figsize=(5.0 * n_prof, 3.8 * nrows),
        gridspec_kw={'hspace': 0.50, 'wspace': 0.35})
    fig.suptitle(
        'Shapley curves — prediction game\n'
        'Identity kernel (top) vs Empirical correlation kernel (bottom)\n'
        'Beijing US Embassy — three pollution-regime profiles',
        fontsize=11, fontweight='bold')

    profile_titles = {
        'Heavy pollution winter':
            'Heavy pollution\n(winter, heating season, low wind)',
        'Clean summer':
            'Clean summer day\n(Jun-Aug, high wind)',
        'Heating onset':
            'Heating season onset\n(first heating day, Nov)',
    }

    for row, (k_label, K) in enumerate(kernels_rows):
        for col, (lbl, (mob, shap)) in enumerate(prof_results.items()):
            ax = axes[row, col]
            for fi in top4:
                curve = apply_kernel(shap[fi], K)
                ax.plot(t_grid, curve,
                        color=FEAT_COLORS[pnames[fi]],
                        lw=2.0, label=pnames[fi])
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _phase_shade(ax)
            _set_time_axis(ax)
            ax.tick_params(labelsize=7)
            ax.set_xlabel('Hour', fontsize=7)
            ax.set_title(profile_titles.get(lbl, lbl),
                         fontsize=8.5, fontweight='bold')
            if col == 0:
                ax.set_ylabel(GAME_YLABEL['prediction'], fontsize=8)
                ax.text(-0.28, 0.5, k_label,
                        transform=ax.transAxes,
                        fontsize=8, va='center', ha='right',
                        rotation=90, color='#333', fontweight='bold')
            if col == 0 and row == 0:
                ax.legend(fontsize=6.5, loc='upper right')

    return fig


# ===========================================================================
# 11.  Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  Beijing Air Quality  —  RF direct  (paper figures)')
    print('=' * 60)

    _require_dir(BASE_PLOT_DIR)
    _require_dir(DATA_DIR)

    # ── 1. Data ───────────────────────────────────────────────────────────
    print('\n[1] Loading data ...')
    X_day, Y_raw, Y_adj, diurnal, dates = load_and_aggregate()
    X_day_np = X_day.to_numpy().astype(float)
    pnames   = list(DAY_FEATURE_NAMES)

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
    K_corr = kernel_correlation(Y_raw)

    # ── 4. Profiles ───────────────────────────────────────────────────────
    print('\n[4] Selecting profiles ...')
    fi_heat = pnames.index('is_heating')
    fi_lag  = pnames.index('lag_pm25_mean')
    fi_wspm = pnames.index('wspm_prev')
    fi_mon  = pnames.index('month')

    lag_p75  = float(np.percentile(X_day_np[:, fi_lag],  75))
    lag_p25  = float(np.percentile(X_day_np[:, fi_lag],  25))
    wspm_p75 = float(np.percentile(X_day_np[:, fi_wspm], 75))

    def find_profile(conds, lbl):
        mask = np.ones(len(X_day_np), dtype=bool)
        for f, (lo, hi) in conds.items():
            ci = pnames.index(f)
            mask &= (X_day_np[:, ci] >= lo) & (X_day_np[:, ci] <= hi)
        hits = X_day_np[mask]
        if not len(hits):
            raise RuntimeError('No match: {}'.format(lbl))
        print('  "{}": {} days'.format(lbl, len(hits)))
        return hits[len(hits) // 2]

    # Heavy pollution winter: heating season, high lag PM2.5, low wind
    x_heavy = find_profile(
        {'is_heating'    : (0.9, 1.1),
         'lag_pm25_mean' : (lag_p75, 99),
         'wspm_prev'     : (0.0, 2.0)},
        'Heavy pollution winter')

    # Clean summer: Jun-Aug, low lag PM2.5, high wind
    x_clean = find_profile(
        {'month'         : (5.9, 8.1),
         'lag_pm25_mean' : (0.0, lag_p25),
         'wspm_prev'     : (wspm_p75, 99)},
        'Clean summer')

    # Heating onset: first heating month (November), previously non-heating
    x_onset = find_profile(
        {'month'      : (10.9, 11.1),
         'is_heating' : (0.9, 1.1)},
        'Heating onset')

    def _y_obs(xp):
        diffs = np.abs(X_day_np - xp[None, :]).sum(axis=1)
        return Y_adj[int(np.argmin(diffs))]

    profile_defs = [
        ('Heavy pollution winter', x_heavy, _y_obs(x_heavy)),
        ('Clean summer',           x_clean, _y_obs(x_clean)),
        ('Heating onset',          x_onset, _y_obs(x_onset)),
    ]
    for lbl, xp, _ in profile_defs:
        print('  {}: {}'.format(lbl, '  '.join(
            '{}={:.2f}'.format(n, xp[j])
            for j, n in enumerate(pnames))))

    # ── 5. Games (heavy pollution primary profile) ────────────────────────
    print('\n[5] Computing all games (heavy pollution winter day) ...')
    x_prim, y_prim = x_heavy, _y_obs(x_heavy)
    mob_prim, shap_prim = {}, {}

    for gtype in ('prediction', 'sensitivity', 'risk'):
        print('\n  game: {} ...'.format(gtype))
        game = FunctionalGame(
            predict_fn  = model.predict,
            X_bg        = X_day_np,
            x_exp       = x_prim,
            game_type   = gtype,
            Y_obs       = y_prim,
            sample_size = SAMPLE_SIZE[gtype],
            random_seed = RNG_SEED,
        )
        game.precompute()
        mob_prim[gtype]  = moebius_transform(game)
        shap_prim[gtype] = shapley_values(mob_prim[gtype], game.p)

    # ── 6. Prediction game for all profiles ───────────────────────────────
    print('\n[6] Prediction game for all profiles ...')
    prof_results = {}
    for label, x_prof, y_prof in profile_defs:
        print('  profile: {} ...'.format(label))
        game = FunctionalGame(
            predict_fn  = model.predict,
            X_bg        = X_day_np,
            x_exp       = x_prof,
            game_type   = 'prediction',
            sample_size = SAMPLE_SIZE['prediction'],
            random_seed = RNG_SEED,
        )
        game.precompute()
        mob  = moebius_transform(game)
        shap = shapley_values(mob, game.p)
        prof_results[label] = (mob, shap)

    # ── 7. Figures ────────────────────────────────────────────────────────
    print('\n[7] Generating figures ...')

    savefig(
        fig_diurnal_and_correlation(diurnal, Y_raw, K_corr),
        'fig1_diurnal_and_correlation.pdf')

    savefig(
        fig_main_effects(mob_prim, pnames, K_corr),
        'fig2_main_effects.pdf')

    # Figure 3: is_heating causal comparison
    fig3, axes3 = plt.subplots(
        2, 4,
        figsize=(3.5 * 4, 3.5 * 2),
        sharey='row')
    fig3.suptitle(
        'is\\_heating — Identity vs Correlation vs Causal kernel\n'
        'Heating elevates overnight/morning PM2.5; '
        'causal kernel correctly restricts attribution\n'
        'Beijing Dongsi',
        fontsize=11, fontweight='bold')
    fig3 = _fill_is_heating_causal(fig3, axes3, mob_prim, pnames, K_corr)
    plt.tight_layout(rect=[0.04, 0, 1, 1])
    savefig(fig3, 'fig3_is_heating_causal.pdf')

    savefig(
        fig_profiles_comparison(prof_results, pnames, K_corr),
        'fig4_profiles_comparison.pdf')

    print('\n' + '=' * 60)
    print('  Done.  Figures in {}/'.format(BASE_PLOT_DIR))
    print('  fig1_diurnal_and_correlation.pdf')
    print('  fig2_main_effects.pdf')
    print('  fig3_is_heating_causal.pdf')
    print('  fig4_profiles_comparison.pdf')
    print('=' * 60)