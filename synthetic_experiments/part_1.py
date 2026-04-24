"""
Part 1 — Ground Truth Validation: Synthetic Experiments
=========================================================
ICU model with one pairwise interaction:

  F(x)(t) = x1*phi1(t) + x2*phi2(t) + x3*phi3(t)
           + alpha*(x1-mu)*(x2-mu)*phi12(t)

  phi1(t)  = exp(-0.2*t)
  phi2(t)  = exp(-(t-10)^2 / 2)
  phi3(t)  = exp(-(t-18)^2 / 2)
  phi12(t) = exp(-(t-5)^2  / 2)

  Xi ~ Uniform[0,1],  mu = 0.5,  alpha = 1.0

Models compared:
  M1: Ridge                (linear, misspecified)
  M2: Random Forest
  M3: NGBoost              (independent Normal per time point)
  M4: MLP                  (direct trajectory prediction)

Plus Oracle: Mobius on the TRUE model with n background samples.

Error metrics:
  - Normalised L2 error  (time-resolved curve recovery)
  - Relative aggregated error (scalar integral recovery)

Outputs saved to plots/synthetic_experiments/part_1/

Usage:
  python part1_synthetic_experiments.py
         [--n_runs 30] [--seed 0] [--latent_dim 8] [--n_est 200]
         [--n_jobs 32] [--rf_jobs 4] [--device cpu]
         [--gpu_id 0] [--quick] [--plots_only]
         [--cache_dir plots/synthetic_experiments/part_1/cache]
"""

import os
import time
import argparse
import itertools
import warnings
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from ngboost import NGBRegressor
from ngboost.distns import Normal

from joblib import Parallel, delayed

import pandas as pd

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_runs',     type=int, default=30)
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--latent_dim', type=int, default=8)
    p.add_argument('--n_est',      type=int, default=200)
    p.add_argument('--n_jobs',     type=int, default=32)
    p.add_argument('--rf_jobs',    type=int, default=4)
    p.add_argument('--device',     type=str, default='cpu',
                   choices=['cpu', 'cuda'])
    p.add_argument('--gpu_id',     type=int, default=0)
    p.add_argument('--quick',      action='store_true')
    p.add_argument('--cache_dir',  type=str,
                   default=os.path.join(
                       'plots', 'synthetic_experiments',
                       'part_1', 'cache'))
    p.add_argument('--plots_only', action='store_true')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

PLOT_DIR = os.path.join(
    'Hilbert_functional_decomposition',
    'plots', 'synthetic_experiments', 'part_1'
)
os.makedirs(PLOT_DIR, exist_ok=True)

T_MAX    = 24.0
T_POINTS = 240
t_grid   = np.linspace(0, T_MAX, T_POINTS)

MU    = 0.5
ALPHA = 1.0

N_VALUES_FULL  = [50, 100, 250, 500, 1000, 2000, 5000, 10000]
N_VALUES_QUICK = [50, 250, 1000, 2000]
REP_N = 1000

NOISE_ELL = 2.0
X_STAR    = np.array([0.8, 0.9, 0.7])

REPORT_SUBSETS = [(1,), (2,), (3,), (1, 2)]

SUBSET_LABELS = {
    (1,):   r'$f_{\{X_1\}}$',
    (2,):   r'$f_{\{X_2\}}$',
    (3,):   r'$f_{\{X_3\}}$',
    (1, 2): r'$f_{\{X_1,X_2\}}$ (interaction)',
}

SUBSET_LABELS_PLAIN = {
    (1,):   'f_{X1}',
    (2,):   'f_{X2}',
    (3,):   'f_{X3}',
    (1, 2): 'f_{X1,X2}',
}

SUBSET_COLORS = {
    (1,):   '#c1121f',
    (2,):   '#2a9d8f',
    (3,):   '#e9c46a',
    (1, 2): '#8338ec',
}

ALL_MODELS = ['oracle', 'ridge', 'rf', 'ngboost', 'mlp']

MODEL_LABELS = {
    'oracle':  'Oracle (true model)',
    'ridge':   'Ridge',
    'rf':      'Random Forest',
    'ngboost': 'NGBoost',
    'mlp':     'MLP',
}

MODEL_COLORS = {
    'oracle':  '#1b2631',
    'ridge':   '#a8dadc',
    'rf':      '#457b9d',
    'ngboost': '#f4d35e',
    'mlp':     '#f4a261',
}

MODEL_LS = {
    'oracle':  '-',
    'ridge':   ':',
    'rf':      '--',
    'ngboost': (0, (3, 1)),
    'mlp':     (0, (5, 2)),
}

MODEL_MARKERS = {
    'oracle':  'D',
    'ridge':   'x',
    'rf':      's',
    'ngboost': 'o',
    'mlp':     'v',
}

# ---------------------------------------------------------------------------
# Global font sizes
# ---------------------------------------------------------------------------

FS_SUPTITLE = 14
FS_TITLE    = 13
FS_LABEL    = 12
FS_TICK     = 11
FS_LEGEND   = 10
FS_ANNOT    = 10

matplotlib.rcParams.update({
    'font.size':       FS_TICK,
    'axes.titlesize':  FS_TITLE,
    'axes.labelsize':  FS_LABEL,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'legend.fontsize': FS_LEGEND,
})

# ===========================================================================
# 1.  True model and analytical ground truth
# ===========================================================================

def phi1(t):  return np.exp(-0.2 * t)
def phi2(t):  return np.exp(-0.5 * (t - 10.0) ** 2)
def phi3(t):  return np.exp(-0.5 * (t - 18.0) ** 2)
def phi12(t): return np.exp(-0.5 * (t -  5.0) ** 2)


def model_true(X, t):
    X = np.atleast_2d(X)
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (
        x1[:, None] * phi1(t)[None, :]
      + x2[:, None] * phi2(t)[None, :]
      + x3[:, None] * phi3(t)[None, :]
      + ALPHA * (x1 - MU)[:, None] * (x2 - MU)[:, None]
               * phi12(t)[None, :]
    )


def analytical_pure_effects(t):
    x1, x2, x3 = X_STAR
    return {
        (1,):      (x1 - MU) * phi1(t),
        (2,):      (x2 - MU) * phi2(t),
        (3,):      (x3 - MU) * phi3(t),
        (1, 2):    ALPHA * (x1 - MU) * (x2 - MU) * phi12(t),
        (1, 3):    np.zeros_like(t),
        (2, 3):    np.zeros_like(t),
        (1, 2, 3): np.zeros_like(t),
    }


def analytical_aggregated(effects_dict, t):
    return {S: float(np.trapezoid(eff, t))
            for S, eff in effects_dict.items()}


def compute_signal_variance(t, n_mc=20000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (n_mc, 3))
    return float(model_true(X, t).var(axis=0).mean())


def compute_variance_decomposition(alpha, t, n_mc=50000, seed=0):
    rng = np.random.default_rng(seed)
    X   = rng.uniform(0, 1, (n_mc, 3))
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    terms = {
        'f1':  x1[:, None] * phi1(t)[None, :],
        'f2':  x2[:, None] * phi2(t)[None, :],
        'f3':  x3[:, None] * phi3(t)[None, :],
        'f12': alpha * (x1 - MU)[:, None] * (x2 - MU)[:, None]
               * phi12(t)[None, :],
    }
    variances = {k: float(np.mean(np.var(v, axis=0)))
                 for k, v in terms.items()}
    vtot      = sum(variances.values())
    fractions = {k: v / vtot for k, v in variances.items()}
    fractions['vtot'] = vtot
    return fractions

# ===========================================================================
# 2.  Noise generation
# ===========================================================================

def make_ou_cov(t, sigma2, ell):
    return sigma2 * np.exp(-np.abs(t[:, None] - t[None, :]) / ell)


def sample_noise(t, sigma2, ell, n, rng):
    K = make_ou_cov(t, sigma2, ell)
    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(t)))
    return (L @ rng.standard_normal((len(t), n))).T


def generate_training_data(n, t, sigma2, ell, rng):
    X   = rng.uniform(0, 1, (n, 3))
    F   = model_true(X, t)
    eps = sample_noise(t, sigma2, ell, n, rng)
    return X, F + eps

# ===========================================================================
# 3.  Model factories
# ===========================================================================

def make_ridge():
    return MultiOutputRegressor(Ridge(alpha=1.0))


def make_rf(n_estimators, rf_jobs, random_state):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=rf_jobs,
        random_state=random_state,
    )


def make_ngboost_direct(n_est, random_state):
    return MultiOutputRegressor(
        NGBRegressor(
            Dist=Normal, n_estimators=n_est,
            learning_rate=0.05, random_state=random_state,
            verbose=False,
        )
    )


def get_ngboost_n_est(n, base):
    if n <= 500:  return base
    if n <= 2000: return max(50, base // 2)
    return max(30, base // 4)


class _MLPNet(nn.Module):
    def __init__(self, p, latent_dim, out_dim, hidden_enc, hidden_dec):
        super().__init__()
        enc, in_d = [], p
        for h in hidden_enc:
            enc += [nn.Linear(in_d, h), nn.Tanh()]
            in_d = h
        enc.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc)
        dec, in_d = [], latent_dim
        for h in hidden_dec:
            dec += [nn.Linear(in_d, h), nn.Tanh()]
            in_d = h
        dec.append(nn.Linear(in_d, out_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class _MLPModel:
    def __init__(self, p, latent_dim, hidden_enc, hidden_dec,
                 lr, n_epochs, batch_size, patience,
                 weight_decay, random_state, device='cpu'):
        self.p            = p
        self.latent_dim   = latent_dim
        self.hidden_enc   = hidden_enc
        self.hidden_dec   = hidden_dec
        self.lr           = lr
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.patience     = patience
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device       = torch.device(device)
        self.net          = None

    def fit(self, X, Y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        out_dim  = Y.shape[1]
        self.net = _MLPNet(
            self.p, self.latent_dim, out_dim,
            self.hidden_enc, self.hidden_dec
        ).to(self.device)
        optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr,
            weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, factor=0.5)
        criterion = nn.MSELoss()
        n     = len(X)
        n_val = max(2, int(0.15 * n))
        idx   = np.random.permutation(n)
        vi, ti = idx[:n_val], idx[n_val:]

        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32).to(self.device)

        Xtr, Ytr = _t(X[ti]), _t(Y[ti])
        Xvl, Yvl = _t(X[vi]), _t(Y[vi])
        loader = DataLoader(
            TensorDataset(Xtr, Ytr),
            batch_size=min(self.batch_size, len(ti)),
            shuffle=True,
            pin_memory=(self.device.type == 'cuda'),
        )
        best_val, best_state, no_imp = float('inf'), None, 0
        self.net.train()
        for _ in range(self.n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(self.net(xb), yb).backward()
                optimizer.step()
            self.net.eval()
            with torch.no_grad():
                vl = criterion(self.net(Xvl), Yvl).item()
            self.net.train()
            scheduler.step(vl)
            if vl < best_val - 1e-7:
                best_val   = vl
                best_state = {k: v.clone()
                              for k, v in self.net.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            out = self.net(
                torch.tensor(X, dtype=torch.float32).to(self.device))
        return out.cpu().numpy()


def _mlp_capacity(n):
    if n < 200:
        return (16,),        (32,),          300,  16, 25
    elif n < 500:
        return (32, 32),     (64, 64),        500,  32, 35
    elif n < 2000:
        return (64, 64),     (128, 128),      700,  64, 50
    elif n < 5000:
        return (64, 64, 64), (128, 256),      800, 128, 60
    else:
        return (64, 64, 64), (128, 256, 256), 1000, 256, 80


def make_mlp_direct(n, T, latent_dim, rng, device='cpu'):
    henc, hdec, epochs, batch, patience = _mlp_capacity(n)
    return _MLPModel(
        p=3, latent_dim=latent_dim,
        hidden_enc=henc, hidden_dec=hdec,
        lr=1e-3, n_epochs=epochs,
        batch_size=batch, patience=patience,
        weight_decay=1e-4,
        random_state=int(rng.integers(0, 2**31)),
        device=device,
    )

# ===========================================================================
# 4.  Mobius estimation
# ===========================================================================

def subset_to_mask(S, p=3):
    mask = np.zeros(p, dtype=bool)
    for j in S:
        mask[j - 1] = True
    return mask


def impute_x(x_star, X_bg, S_mask):
    x_imp = X_bg.copy()
    for j in range(len(x_star)):
        if S_mask[j]:
            x_imp[:, j] = x_star[j]
    return x_imp


def compute_game_values(predict_fn, x_star, X_bg, t):
    p     = len(x_star)
    v     = {}
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(1, p + 1), r)
        for r in range(0, p + 1)
    ))
    for S in all_S:
        if len(S) == 0:
            v[()] = predict_fn(X_bg).mean(axis=0)
        else:
            mask  = subset_to_mask(S, p)
            x_imp = impute_x(x_star, X_bg, mask)
            v[S]  = predict_fn(x_imp).mean(axis=0)
    return v


def mobius_transform(v_dict, p=3):
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(1, p + 1), r)
        for r in range(0, p + 1)
    ))
    m = {}
    for S in all_S:
        val = None
        for L in itertools.chain.from_iterable(
            itertools.combinations(S, r)
            for r in range(len(S) + 1)
        ):
            sign = (-1) ** (len(S) - len(L))
            term = v_dict[L] if L != () else v_dict[()]
            val  = sign * term if val is None else val + sign * term
        m[S] = val
    return m


def estimate_mobius(predict_fn, x_star, X_bg, t):
    v = compute_game_values(predict_fn, x_star, X_bg, t)
    return mobius_transform(v)

# ===========================================================================
# 5.  Error metrics
# ===========================================================================

def l2_error_normalized(est, truth, t):
    norm_truth = np.sqrt(np.trapezoid(truth ** 2, t))
    norm_err   = np.sqrt(np.trapezoid((est - truth) ** 2, t))
    return float(norm_err / max(norm_truth, 1e-10))


def aggregated_error(est_scalar, truth_scalar):
    return float(
        abs(est_scalar - truth_scalar)
        / max(abs(truth_scalar), 1e-10)
    )

# ===========================================================================
# 6.  Single run
# ===========================================================================

def run_one(n, seed, t, sigma2, latent_dim, base_n_est,
            rf_jobs, device):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    warnings.filterwarnings('ignore')

    rng = np.random.default_rng(seed)
    T   = len(t)

    truth_effects = analytical_pure_effects(t)
    truth_agg     = analytical_aggregated(truth_effects, t)

    X_bg             = rng.uniform(0, 1, (n, 3))
    X_train, Y_train = generate_training_data(n, t, sigma2, NOISE_ELL, rng)

    n_est = get_ngboost_n_est(n, base_n_est)
    rs    = int(rng.integers(0, 2**31))

    ridge_model = make_ridge()
    ridge_model.fit(X_train, Y_train)

    rf_model = make_rf(n_estimators=200, rf_jobs=rf_jobs, random_state=rs)
    rf_model.fit(X_train, Y_train)

    ngb_model = make_ngboost_direct(n_est=n_est, random_state=rs)
    ngb_model.fit(X_train, Y_train)

    mlp_model = make_mlp_direct(n=n, T=T, latent_dim=latent_dim,
                                rng=rng, device=device)
    mlp_model.fit(X_train, Y_train)

    def oracle_fn(X): return model_true(X, t)

    predict_fns = {
        'oracle':  oracle_fn,
        'ridge':   ridge_model.predict,
        'rf':      rf_model.predict,
        'ngboost': ngb_model.predict,
        'mlp':     mlp_model.predict,
    }

    mobius = {}
    for tag, fn in predict_fns.items():
        mobius[tag] = estimate_mobius(fn, X_STAR, X_bg, t)

    out = {
        '_truth_effects': truth_effects,
        '_truth_agg':     truth_agg,
        '_mobius':        mobius,
    }

    for tag in ALL_MODELS:
        m = mobius[tag]
        for S in REPORT_SUBSETS:
            truth    = truth_effects[S]
            truth_sc = truth_agg[S]
            est      = m.get(S, np.zeros_like(t))
            est_sc   = float(np.trapezoid(est, t))
            out[(tag, S)] = {
                'l2_err':  l2_error_normalized(est, truth, t),
                'agg_err': aggregated_error(est_sc, truth_sc),
                'effect':  est.copy(),
            }

    return out

# ===========================================================================
# 7.  Parallel experiment loop
# ===========================================================================

def run_experiments(n_runs, base_seed, t, sigma2, latent_dim, n_est,
                    n_values, n_jobs, rf_jobs, device, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    rng_master = np.random.default_rng(base_seed)
    seeds      = rng_master.integers(0, 2**31, size=n_runs)
    jobs       = [(n, int(seeds[i]))
                  for n in n_values for i in range(n_runs)]

    log.info(f'Total jobs      : {len(jobs)}')
    log.info(f'Parallel workers: {n_jobs}')

    t0  = time.time()
    raw = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
        delayed(run_one)(n, seed, t, sigma2, latent_dim,
                         n_est, rf_jobs, device)
        for n, seed in jobs
    )
    log.info(f'Experiments finished in {(time.time()-t0)/60:.1f} min')

    l2_errors  = {n: {tag: {S: [] for S in REPORT_SUBSETS}
                       for tag in ALL_MODELS} for n in n_values}
    agg_errors = {n: {tag: {S: [] for S in REPORT_SUBSETS}
                       for tag in ALL_MODELS} for n in n_values}
    representative = None

    for (n, _seed), res in zip(jobs, raw):
        for tag in ALL_MODELS:
            for S in REPORT_SUBSETS:
                l2_errors[n][tag][S].append(res[(tag, S)]['l2_err'])
                agg_errors[n][tag][S].append(res[(tag, S)]['agg_err'])
        if n == REP_N and REP_N in n_values:
            representative = res

    for n in n_values:
        for tag in ALL_MODELS:
            for S in REPORT_SUBSETS:
                l2_errors[n][tag][S]  = np.array(l2_errors[n][tag][S])
                agg_errors[n][tag][S] = np.array(agg_errors[n][tag][S])

    _save_cache(l2_errors, agg_errors, representative, n_values, cache_dir)
    return l2_errors, agg_errors, representative

# ===========================================================================
# 8.  Cache helpers
# ===========================================================================

def _save_cache(l2_errors, agg_errors, representative,
                n_values, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    for n in n_values:
        d = {}
        for tag in ALL_MODELS:
            for S in REPORT_SUBSETS:
                key = f'{tag}_{"_".join(map(str, S))}'
                d[f'l2_{key}']  = l2_errors[n][tag][S]
                d[f'agg_{key}'] = agg_errors[n][tag][S]
        np.savez(os.path.join(cache_dir, f'results_n{n}.npz'), **d)

    if representative is not None:
        rep_d = {}
        for tag in ALL_MODELS:
            for S in REPORT_SUBSETS:
                key = f'{tag}_{"_".join(map(str, S))}_effect'
                rep_d[key] = representative[(tag, S)]['effect']
        for S in REPORT_SUBSETS:
            key = f'truth_{"_".join(map(str, S))}'
            rep_d[key] = representative['_truth_effects'][S]
        np.savez(os.path.join(cache_dir, 'representative.npz'), **rep_d)

    log.info(f'Results cached to {cache_dir}')


def _load_cache(n_values, cache_dir):
    l2_errors  = {n: {tag: {S: None for S in REPORT_SUBSETS}
                       for tag in ALL_MODELS} for n in n_values}
    agg_errors = {n: {tag: {S: None for S in REPORT_SUBSETS}
                       for tag in ALL_MODELS} for n in n_values}

    for n in n_values:
        path = os.path.join(cache_dir, f'results_n{n}.npz')
        d    = np.load(path)
        for tag in ALL_MODELS:
            for S in REPORT_SUBSETS:
                key = f'{tag}_{"_".join(map(str, S))}'
                l2_errors[n][tag][S]  = d[f'l2_{key}']
                agg_errors[n][tag][S] = d[f'agg_{key}']

    rep_path = os.path.join(cache_dir, 'representative.npz')
    rep_d    = np.load(rep_path)
    representative = {}
    for tag in ALL_MODELS:
        for S in REPORT_SUBSETS:
            key = f'{tag}_{"_".join(map(str, S))}_effect'
            representative[(tag, S)] = {'effect': rep_d[key]}
    representative['_truth_effects'] = {}
    for S in REPORT_SUBSETS:
        key = f'truth_{"_".join(map(str, S))}'
        representative['_truth_effects'][S] = rep_d[key]

    log.info(f'Results loaded from {cache_dir}')
    return l2_errors, agg_errors, representative

# ===========================================================================
# 9.  Plotting helpers
# ===========================================================================

def savefig(fig, name, plot_dir):
    path = os.path.join(plot_dir, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def _style_ax(ax, ylabel=None, xlabel='Time (h)'):
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.tick_params(labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.axvspan(8,  12, alpha=0.05, color='#2a9d8f', zorder=0)
    ax.axvspan(16, 20, alpha=0.05, color='#e9c46a', zorder=0)


def _n_recovery_panel(ax, error_dict, n_values, tag_list,
                      S, xtl, show_legend=False,
                      ylabel=None, xlabel=True):
    for tag in tag_list:
        means = np.array([error_dict[n][tag][S].mean() for n in n_values])
        stds  = np.array([error_dict[n][tag][S].std()  for n in n_values])
        ax.plot(
            n_values, means,
            color=MODEL_COLORS[tag],
            ls=MODEL_LS[tag],
            marker=MODEL_MARKERS[tag],
            lw=2.0, ms=5.5,
            label=MODEL_LABELS[tag],
        )
        ax.fill_between(
            n_values,
            np.clip(means - stds, 0, None),
            means + stds,
            alpha=0.10,
            color=MODEL_COLORS[tag],
        )

    ax.axhline(1.0, color='#e63946', lw=1.0, ls=':', alpha=0.7)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.grid(True, linestyle=':', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    ax.set_xscale('log')
    ax.set_xticks(n_values)
    ax.set_xticklabels(xtl, fontsize=FS_TICK - 1, rotation=30)
    ax.tick_params(labelsize=FS_TICK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if xlabel:
        ax.set_xlabel('$n$ (training samples)', fontsize=FS_LABEL)
    if show_legend:
        ax.legend(fontsize=FS_LEGEND, ncol=2,
                  loc='upper right', framealpha=0.9)

# ===========================================================================
# 10.  Fig 1a — N-recovery L2 (standalone)
# ===========================================================================

def plot_n_recovery_l2(l2_errors, n_values, n_runs,
                       plot_dir, all_models, model_labels,
                       model_colors, model_ls, model_markers,
                       report_subsets, subset_labels, subset_colors):
    n_cols = len(report_subsets)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(5.0 * n_cols, 5.0))
    fig.suptitle(
        f'N-Recovery: Normalised L2 error\n'
        f'Mean $\\pm$ 1 std over {n_runs} runs',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )
    xtl = [str(n) if n < 1000 else f'{n // 1000}k'
           for n in n_values]

    for col, S in enumerate(report_subsets):
        ax = axes[col]
        _n_recovery_panel(
            ax, l2_errors, n_values, all_models, S, xtl,
            show_legend=(col == 0),
            ylabel='Normalised L2 error' if col == 0 else None,
        )
        ax.set_title(subset_labels[S], fontsize=FS_TITLE,
                     fontweight='bold', color=subset_colors[S])

    plt.tight_layout()
    savefig(fig, 'fig1a_n_recovery_l2.pdf', plot_dir)


# ===========================================================================
# 10b.  Fig 1b — N-recovery aggregated (standalone)
# ===========================================================================

def plot_n_recovery_agg(agg_errors, n_values, n_runs,
                        plot_dir, all_models, model_labels,
                        model_colors, model_ls, model_markers,
                        report_subsets, subset_labels, subset_colors):
    n_cols = len(report_subsets)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(5.0 * n_cols, 5.0))
    fig.suptitle(
        f'N-Recovery: Relative aggregated error\n'
        f'Mean $\\pm$ 1 std over {n_runs} runs',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )
    xtl = [str(n) if n < 1000 else f'{n // 1000}k'
           for n in n_values]

    for col, S in enumerate(report_subsets):
        ax = axes[col]
        _n_recovery_panel(
            ax, agg_errors, n_values, all_models, S, xtl,
            show_legend=(col == 0),
            ylabel='Relative aggregated error' if col == 0 else None,
        )
        ax.set_title(subset_labels[S], fontsize=FS_TITLE,
                     fontweight='bold', color=subset_colors[S])

    plt.tight_layout()
    savefig(fig, 'fig1b_n_recovery_agg.pdf', plot_dir)


# ===========================================================================
# 10c.  Fig 1c — Combined N-recovery: L2 (row 0) + Agg (row 1)
# ===========================================================================

def plot_n_recovery_combined(l2_errors, agg_errors, n_values, n_runs,
                             plot_dir, all_models,
                             report_subsets, subset_labels, subset_colors):
    n_cols = len(report_subsets)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(5.0 * n_cols, 11.0),
        gridspec_kw={'hspace': 0.55, 'wspace': 0.28},
    )
    fig.suptitle(
        f'N-Recovery: Normalised L2 error (top) and '
        f'Relative aggregated error (bottom)\n'
        f'Mean $\\pm$ 1 std over {n_runs} runs',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )
    xtl = [str(n) if n < 1000 else f'{n // 1000}k'
           for n in n_values]

    row_labels  = ['Normalised L2 error', 'Relative aggregated error']
    error_dicts = [l2_errors, agg_errors]

    for row, (err_dict, row_label) in enumerate(
            zip(error_dicts, row_labels)):
        for col, S in enumerate(report_subsets):
            ax = axes[row, col]
            _n_recovery_panel(
                ax, err_dict, n_values, all_models, S, xtl,
                show_legend=False,
                ylabel=row_label if col == 0 else None,
                xlabel=(row == 1),
            )
            if row == 0:
                ax.set_title(subset_labels[S], fontsize=FS_TITLE,
                             fontweight='bold', color=subset_colors[S])
            if col == n_cols - 1:
                ax.text(1.03, 0.5, row_label,
                        transform=ax.transAxes,
                        fontsize=FS_LABEL - 1, va='center',
                        rotation=270, color='gray')

    # single one-line legend centred below the full top row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        fontsize=FS_LEGEND,
        ncol=len(handles),
        loc='upper center',
        bbox_to_anchor=(0.5, 0.505),
        framealpha=0.9,
        borderaxespad=0.0,
        handlelength=2.0,
        columnspacing=1.2,
    )

    savefig(fig, 'fig1c_n_recovery_combined.pdf', plot_dir)


# ===========================================================================
# 11.  Effect comparison helpers (shared by standalone + combined)
# ===========================================================================

def _effect_comparison_panels(axes, representative, t,
                               all_models, model_labels,
                               model_colors, model_ls,
                               report_subsets, subset_labels,
                               subset_colors):
    tags = [tg for tg in all_models if tg != 'oracle']
    truth_effects = representative['_truth_effects']

    for col, S in enumerate(report_subsets):
        ax    = axes[col]
        truth = truth_effects[S]

        for tag in tags:
            est = representative[(tag, S)]['effect']
            ax.plot(t, est,
                    color=model_colors[tag],
                    ls=model_ls[tag],
                    lw=2.0, alpha=0.85,
                    label=model_labels[tag])

        ax.plot(t, truth,
                color='#888888', lw=1.4, ls='--',
                dashes=(4, 3), label='Analytical', zorder=10)

        _style_ax(ax, ylabel='Effect (a.u.)' if col == 0 else None)
        ax.set_title(subset_labels[S], fontsize=FS_TITLE,
                     fontweight='bold', color=subset_colors[S])
        if col == 0:
            ax.legend(fontsize=FS_LEGEND, loc='upper right')


def plot_effect_comparison(representative, t, n_values,
                           plot_dir, all_models, model_labels,
                           model_colors, model_ls,
                           report_subsets, subset_labels,
                           subset_colors, rep_n):
    n_cols = len(report_subsets)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(5.0 * n_cols, 5.0))
    fig.suptitle(
        f'Time-resolved pure effects at '
        f'$\\mathbf{{x}}^* = (0.8, 0.9, 0.7)$\n'
        f'Representative run, $n = {rep_n}$',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )
    _effect_comparison_panels(
        axes, representative, t,
        all_models, model_labels, model_colors, model_ls,
        report_subsets, subset_labels, subset_colors,
    )
    plt.tight_layout()
    savefig(fig, 'fig2_effect_comparison.pdf', plot_dir)


# ===========================================================================
# 12.  Fig 3 — Aggregated scalar effects (standalone)
# ===========================================================================

def _aggregated_effects_panel(ax, representative, t):
    truth_agg = {
        S: float(np.trapezoid(representative['_truth_effects'][S], t))
        for S in REPORT_SUBSETS
    }
    x_pos        = np.arange(len(REPORT_SUBSETS))
    w            = 0.12
    tags_ordered = ['oracle', 'ridge', 'rf', 'ngboost', 'mlp']
    n_bars       = 1 + len(tags_ordered)
    offsets      = np.linspace(
        -(n_bars - 1) / 2 * w,
         (n_bars - 1) / 2 * w,
        n_bars,
    )

    ax.bar(x_pos + offsets[0],
           [truth_agg[S] for S in REPORT_SUBSETS],
           width=w, color='#1b2631', alpha=0.9,
           label='Analytical')

    for i, tag in enumerate(tags_ordered):
        vals = [float(np.trapezoid(
                    representative[(tag, S)]['effect'], t))
                for S in REPORT_SUBSETS]
        ec = 'black' if tag == 'oracle' else 'none'
        lw = 1.5     if tag == 'oracle' else 0.0
        ax.bar(x_pos + offsets[i + 1], vals,
               width=w,
               color=MODEL_COLORS[tag],
               edgecolor=ec, linewidth=lw,
               alpha=0.85,
               label=MODEL_LABELS[tag])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [SUBSET_LABELS[S] for S in REPORT_SUBSETS],
        fontsize=FS_LABEL)
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(r'$\Phi_S$', fontsize=FS_LABEL + 1)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, ncol=3, loc='upper right')


def plot_aggregated_effects(representative, t):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.suptitle(
        r'Time-aggregated effects $\Phi_S = \int f_S(t)\,dt$'
        '\n(representative run)',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )
    _aggregated_effects_panel(ax, representative, t)
    plt.tight_layout()
    savefig(fig, 'fig3_aggregated_effects.pdf', PLOT_DIR)


# ===========================================================================
# 12b.  Fig 2+3 combined: effect curves (row 0) + bar chart (row 1)
# ===========================================================================

def plot_effects_combined(representative, t, n_values,
                          plot_dir, all_models, model_labels,
                          model_colors, model_ls,
                          report_subsets, subset_labels,
                          subset_colors, rep_n):
    import matplotlib.gridspec as gridspec
    n_cols = len(report_subsets)

    fig = plt.figure(figsize=(5.0 * n_cols, 10.5))
    fig.suptitle(
        f'Time-resolved and time-aggregated effects at '
        f'$\\mathbf{{x}}^* = (0.8, 0.9, 0.7)$\n'
        f'Representative run, $n = {rep_n}$',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    gs = gridspec.GridSpec(
        2, n_cols, figure=fig,
        height_ratios=[1.0, 0.85],
        hspace=0.48, wspace=0.30,
        left=0.07, right=0.97,
        top=0.90, bottom=0.07,
    )

    # Row 0: time-resolved panels
    axes_top = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
    _effect_comparison_panels(
        axes_top, representative, t,
        all_models, model_labels, model_colors, model_ls,
        report_subsets, subset_labels, subset_colors,
    )
    axes_top[-1].text(
        1.03, 0.5, 'Time-resolved',
        transform=axes_top[-1].transAxes,
        fontsize=FS_LABEL - 1, va='center',
        rotation=270, color='gray',
    )

    # Row 1: aggregated bar chart (spanning all columns)
    ax_bar = fig.add_subplot(gs[1, :])
    _aggregated_effects_panel(ax_bar, representative, t)
    ax_bar.set_title(
        r'Time-aggregated effects $\Phi_S = \int f_S(t)\,dt$',
        fontsize=FS_TITLE, fontweight='bold', pad=10,
    )
    ax_bar.text(
        1.01, 0.5, 'Time-aggregated',
        transform=ax_bar.transAxes,
        fontsize=FS_LABEL - 1, va='center',
        rotation=270, color='gray',
    )

    savefig(fig, 'fig23_effects_combined.pdf', plot_dir)


# ===========================================================================
# 13.  Tables
# ===========================================================================

def _render_df_to_ax(ax, df, title, fontsize=8.5):
    ax.axis('off')
    col_labels = list(df.columns)
    row_labels  = [
        ' / '.join(str(i) for i in idx)
        if isinstance(idx, tuple) else str(idx)
        for idx in df.index
    ]
    table = ax.table(
        cellText  = df.values,
        rowLabels = row_labels,
        colLabels = col_labels,
        cellLoc   = 'center',
        rowLoc    = 'right',
        loc       = 'center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.auto_set_column_width(list(range(len(col_labels))))

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2a9d8f')
        table[0, j].set_text_props(color='white', fontweight='bold')

    for i in range(len(row_labels)):
        fc = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(fc)

    for i in range(len(row_labels)):
        table[i + 1, -1].set_facecolor('#e8f4f8')

    ax.set_title(title, fontsize=fontsize + 2,
                 fontweight='bold', pad=12)


def build_summary_table(l2_errors, agg_errors, n_values):
    n_large = n_values[-1]
    rows    = []
    for tag in ALL_MODELS:
        row = {'Model': MODEL_LABELS[tag]}
        for S in REPORT_SUBSETS:
            sl  = SUBSET_LABELS_PLAIN[S]
            al2 = l2_errors[n_large][tag][S]
            aa  = agg_errors[n_large][tag][S]
            row[f'L2 {sl}']  = f'{al2.mean():.3f} ± {al2.std():.3f}'
            row[f'Agg {sl}'] = f'{aa.mean():.3f} ± {aa.std():.3f}'
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')


def build_full_table(error_dict, n_values, metric_name):
    rows = []
    for tag in ALL_MODELS:
        for S in REPORT_SUBSETS:
            row = {'Model': MODEL_LABELS[tag],
                   'Subset': SUBSET_LABELS_PLAIN[S]}
            for n in n_values:
                arr = error_dict[n][tag][S]
                col = str(n) if n < 1000 else f'{n//1000}k'
                row[col] = f'{arr.mean():.3f} ± {arr.std():.3f}'
            rows.append(row)
    return pd.DataFrame(rows).set_index(['Model', 'Subset'])


def build_convergence_table(l2_errors, agg_errors, n_values):
    n_select = [n for n in [50, 500, 2000, n_values[-1]]
                if n in n_values]
    rows = []
    for tag in ALL_MODELS:
        for S in REPORT_SUBSETS:
            row = {'Model': MODEL_LABELS[tag],
                   'Subset': SUBSET_LABELS_PLAIN[S]}
            for n in n_select:
                col = str(n) if n < 1000 else f'{n//1000}k'
                al2 = l2_errors[n][tag][S]
                aa  = agg_errors[n][tag][S]
                row[f'L2 n={col}']  = f'{al2.mean():.3f}±{al2.std():.3f}'
                row[f'Agg n={col}'] = f'{aa.mean():.3f}±{aa.std():.3f}'
            rows.append(row)
    return pd.DataFrame(rows).set_index(['Model', 'Subset'])


def plot_tables(l2_errors, agg_errors, n_values, n_runs):
    df_summary     = build_summary_table(l2_errors, agg_errors, n_values)
    df_l2_full     = build_full_table(l2_errors,  n_values, 'L2')
    df_agg_full    = build_full_table(agg_errors, n_values, 'Agg')
    df_convergence = build_convergence_table(l2_errors, agg_errors, n_values)

    n_large = n_values[-1]
    nl_str  = str(n_large) if n_large < 1000 else f'{n_large//1000}k'

    for df, fname, title in [
        (df_summary,
         'fig_table_summary.pdf',
         f'Summary: L2 and aggregated error at n={nl_str}\n'
         f'(mean ± 1 std, {n_runs} runs)'),
        (df_l2_full,
         'fig_table_l2_full.pdf',
         f'Full results: Normalised L2 error\n'
         f'(mean ± 1 std, {n_runs} runs)'),
        (df_agg_full,
         'fig_table_agg_full.pdf',
         f'Full results: Relative aggregated error\n'
         f'(mean ± 1 std, {n_runs} runs)'),
        (df_convergence,
         'fig_table_convergence.pdf',
         f'Convergence overview: L2 and aggregated error\n'
         f'(mean ± 1 std, {n_runs} runs)'),
    ]:
        nr, nc = df.shape
        fig, ax = plt.subplots(
            figsize=(max(10, nc * 2.0 + 3),
                     max(4,  nr * 0.55 + 2.0)))
        _render_df_to_ax(ax, df, title=title, fontsize=8.5)
        plt.tight_layout()
        savefig(fig, fname, PLOT_DIR)

    csv_dir = os.path.join(PLOT_DIR, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    df_summary.to_csv(os.path.join(csv_dir, 'table_summary.csv'))
    df_l2_full.to_csv(os.path.join(csv_dir, 'table_l2_full.csv'))
    df_agg_full.to_csv(os.path.join(csv_dir, 'table_agg_full.csv'))
    df_convergence.to_csv(os.path.join(csv_dir, 'table_convergence.csv'))
    log.info(f'CSV tables saved to {csv_dir}/')

# ===========================================================================
# 14.  Main
# ===========================================================================

def main():
    args       = parse_args()
    n_runs     = 1 if args.quick else args.n_runs
    latent_dim = args.latent_dim
    n_est      = args.n_est
    n_jobs     = args.n_jobs
    rf_jobs    = args.rf_jobs
    device     = args.device
    n_values   = N_VALUES_QUICK if args.quick else N_VALUES_FULL

    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        if not torch.cuda.is_available():
            log.warning('CUDA not available — falling back to CPU')
            device = 'cpu'
        else:
            log.info(f'Using GPU {args.gpu_id}: '
                     f'{torch.cuda.get_device_name(args.gpu_id)}')

    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    log.info('=' * 60)
    log.info('Part 1 — Synthetic Experiments: Model Comparison')
    log.info(f'  n_runs   : {n_runs}')
    log.info(f'  n_values : {n_values}')
    log.info(f'  device   : {device}')
    log.info(f'  plot_dir : {PLOT_DIR}')
    log.info('=' * 60)

    vd = compute_variance_decomposition(ALPHA, t_grid)
    log.info('DGP variance decomposition (alpha=%.1f):', ALPHA)
    log.info(f"  f_{{X1}}    : {vd['f1']:.1%}")
    log.info(f"  f_{{X2}}    : {vd['f2']:.1%}")
    log.info(f"  f_{{X3}}    : {vd['f3']:.1%}")
    log.info(f"  f_{{X1,X2}} : {vd['f12']:.1%}  <-- interaction")
    log.info(f"  Total var : {vd['vtot']:.4f}")
    log.info('-' * 60)

    if args.plots_only:
        log.info('--plots_only flag: loading results from cache')
        l2_errors, agg_errors, rep = _load_cache(n_values, cache_dir)
    else:
        sig_var = compute_signal_variance(
            t_grid, rng=np.random.default_rng(args.seed))
        sigma2  = 0.20 * sig_var
        log.info(f'Signal variance : {sig_var:.4f}')
        log.info(f'Noise sigma²    : {sigma2:.4f}  '
                 f'(SNR ~ {sig_var/sigma2:.1f})')
        l2_errors, agg_errors, rep = run_experiments(
            n_runs=n_runs, base_seed=args.seed,
            t=t_grid, sigma2=sigma2,
            latent_dim=latent_dim, n_est=n_est,
            n_values=n_values, n_jobs=n_jobs,
            rf_jobs=rf_jobs, device=device,
            cache_dir=cache_dir,
        )

    log.info('Generating figures ...')

    # Standalone figures
    plot_effect_comparison(
        rep, t_grid, n_values,
        plot_dir=PLOT_DIR,
        all_models=ALL_MODELS, model_labels=MODEL_LABELS,
        model_colors=MODEL_COLORS, model_ls=MODEL_LS,
        report_subsets=REPORT_SUBSETS, subset_labels=SUBSET_LABELS,
        subset_colors=SUBSET_COLORS, rep_n=REP_N,
    )
    plot_n_recovery_l2(
        l2_errors, n_values, n_runs,
        plot_dir=PLOT_DIR,
        all_models=ALL_MODELS, model_labels=MODEL_LABELS,
        model_colors=MODEL_COLORS, model_ls=MODEL_LS,
        model_markers=MODEL_MARKERS,
        report_subsets=REPORT_SUBSETS, subset_labels=SUBSET_LABELS,
        subset_colors=SUBSET_COLORS,
    )
    plot_n_recovery_agg(
        agg_errors, n_values, n_runs,
        plot_dir=PLOT_DIR,
        all_models=ALL_MODELS, model_labels=MODEL_LABELS,
        model_colors=MODEL_COLORS, model_ls=MODEL_LS,
        model_markers=MODEL_MARKERS,
        report_subsets=REPORT_SUBSETS, subset_labels=SUBSET_LABELS,
        subset_colors=SUBSET_COLORS,
    )
    plot_aggregated_effects(rep, t_grid)

    # Combined figures
    plot_n_recovery_combined(
        l2_errors, agg_errors, n_values, n_runs,
        plot_dir=PLOT_DIR,
        all_models=ALL_MODELS,
        report_subsets=REPORT_SUBSETS,
        subset_labels=SUBSET_LABELS,
        subset_colors=SUBSET_COLORS,
    )
    plot_effects_combined(
        rep, t_grid, n_values,
        plot_dir=PLOT_DIR,
        all_models=ALL_MODELS, model_labels=MODEL_LABELS,
        model_colors=MODEL_COLORS, model_ls=MODEL_LS,
        report_subsets=REPORT_SUBSETS, subset_labels=SUBSET_LABELS,
        subset_colors=SUBSET_COLORS, rep_n=REP_N,
    )

    log.info('Generating tables ...')
    plot_tables(l2_errors, agg_errors, n_values, n_runs)

    log.info(f'All outputs saved to {PLOT_DIR}/')


if __name__ == '__main__':
    main()