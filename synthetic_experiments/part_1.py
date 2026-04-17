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

  Xi ~ Uniform[0,1],  mu = 0.5,  alpha = 0.5

Models compared:
  M1: Ridge                (linear, misspecified)
  M2: Random Forest        (no PCA)
  M3: Random Forest + PCA
  M4: NGBoost              (no PCA, independent Normal per time point)
  M5: NGBoost + PCA        (MultivariateNormal on PCA scores, Section 4.5)
  M6: MLP                  (no PCA, direct trajectory prediction)
  M7: MLP + PCA            (bottleneck MLP on PCA scores)

Plus Oracle: Möbius on the TRUE model with n background samples.

Error metrics:
  - Normalised L2 error  (time-resolved curve recovery)
  - Relative aggregated error (scalar integral recovery)

Outputs saved to plots/synthetic_experiments/part_1/

Usage:
  python part1_synthetic_experiments.py
         [--n_runs 30] [--seed 0] [--n_pca 8] [--n_est 200]
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

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from ngboost import NGBRegressor
from ngboost.distns import MultivariateNormal, Normal

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
    p.add_argument('--n_runs',    type=int, default=30,
                   help='Monte Carlo repetitions (default 30)')
    p.add_argument('--seed',      type=int, default=0,
                   help='Base random seed (default 0)')
    p.add_argument('--n_pca',     type=int, default=8,
                   help='PCA components (default 8)')
    p.add_argument('--n_est',     type=int, default=200,
                   help='Base n_estimators for RF/NGBoost (default 200)')
    p.add_argument('--n_jobs',    type=int, default=32,
                   help='Parallel workers for outer run loop (default 32)')
    p.add_argument('--rf_jobs',   type=int, default=4,
                   help='n_jobs inside each RF fit (default 4)')
    p.add_argument('--device',    type=str, default='cpu',
                   choices=['cpu', 'cuda'],
                   help='Device for MLP training (default cpu)')
    p.add_argument('--gpu_id',    type=int, default=0,
                   help='Which GPU to use if device=cuda (default 0)')
    p.add_argument('--quick',     action='store_true',
                   help='Quick test: 1 run, reduced N_VALUES')
    p.add_argument('--cache_dir', type=str,
                   default=os.path.join(
                       'plots', 'synthetic_experiments',
                       'part_1', 'cache'),
                   help='Directory to cache raw results')
    p.add_argument('--plots_only', action='store_true',
                   help='Skip experiments, load cache, regenerate plots')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

PLOT_DIR = os.path.join('Hilbert_functional_decomposition','plots', 'synthetic_experiments', 'part_1')
os.makedirs(PLOT_DIR, exist_ok=True)

T_MAX    = 24.0
T_POINTS = 240
t_grid   = np.linspace(0, T_MAX, T_POINTS)

MU    = 0.5
ALPHA = 0.5

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

ALL_MODELS = [
    'oracle',
    'ridge',
    'rf',
    'rf_pca',
    'ngboost',
    'ngboost_pca',
    'mlp',
    'mlp_pca',
]

MODEL_LABELS = {
    'oracle':      'Oracle (true model)',
    'ridge':       'Ridge',
    'rf':          'Random Forest',
    'rf_pca':      'Random Forest + PCA',
    'ngboost':     'NGBoost',
    'ngboost_pca': 'NGBoost + PCA',
    'mlp':         'MLP',
    'mlp_pca':     'MLP + PCA',
}

MODEL_COLORS = {
    'oracle':      '#1b2631',
    'ridge':       '#a8dadc',
    'rf':          '#457b9d',
    'rf_pca':      '#1d3557',
    'ngboost':     '#f4d35e',
    'ngboost_pca': '#e9c46a',
    'mlp':         '#f4a261',
    'mlp_pca':     '#e76f51',
}

MODEL_LS = {
    'oracle':      '-',
    'ridge':       ':',
    'rf':          '--',
    'rf_pca':      '-.',
    'ngboost':     (0, (3, 1)),
    'ngboost_pca': (0, (3, 1, 1, 1)),
    'mlp':         (0, (5, 2)),
    'mlp_pca':     (0, (1, 1)),
}

MODEL_MARKERS = {
    'oracle':      'D',
    'ridge':       'x',
    'rf':          's',
    'rf_pca':      'P',
    'ngboost':     'o',
    'ngboost_pca': '^',
    'mlp':         'v',
    'mlp_pca':     '*',
}

PCA_PAIRS = [
    ('rf',      'rf_pca',      'Random Forest'),
    ('ngboost', 'ngboost_pca', 'NGBoost'),
    ('mlp',     'mlp_pca',     'MLP'),
]

# ===========================================================================
# 1.  True model and analytical ground truth
# ===========================================================================

def phi1(t):  return np.exp(-0.2 * t)
def phi2(t):  return np.exp(-0.5 * (t - 10.0) ** 2)
def phi3(t):  return np.exp(-0.5 * (t - 18.0) ** 2)
def phi12(t): return np.exp(-0.5 * (t -  5.0) ** 2)


def model_true(X, t):
    """
    True functional model  F: R^3 -> R^T.
    X : (N, 3)   t : (T,)
    Returns (N, T)
    """
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
    """
    Closed-form H-FD pure effects at x* = (0.8, 0.9, 0.7).
    Returns dict: subset_tuple -> (T,) array.
    """
    x1, x2, x3 = X_STAR
    return {
        (1,):     (x1 - MU) * phi1(t),
        (2,):     (x2 - MU) * phi2(t),
        (3,):     (x3 - MU) * phi3(t),
        (1, 2):   ALPHA * (x1 - MU) * (x2 - MU) * phi12(t),
        (1, 3):   np.zeros_like(t),
        (2, 3):   np.zeros_like(t),
        (1, 2, 3):np.zeros_like(t),
    }


def analytical_aggregated(effects_dict, t):
    """Scalar integrated effects Phi_S = int f_S(t) dt."""
    return {S: float(np.trapezoid(eff, t))
            for S, eff in effects_dict.items()}


def compute_signal_variance(t, n_mc=20000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (n_mc, 3))
    return float(model_true(X, t).var(axis=0).mean())

# ===========================================================================
# 2.  Noise generation (OU covariance)
# ===========================================================================

def make_ou_cov(t, sigma2, ell):
    return sigma2 * np.exp(
        -np.abs(t[:, None] - t[None, :]) / ell
    )


def sample_noise(t, sigma2, ell, n, rng):
    K = make_ou_cov(t, sigma2, ell)
    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(t)))
    return (L @ rng.standard_normal((len(t), n))).T  # (n, T)


def generate_training_data(n, t, sigma2, ell, rng):
    X   = rng.uniform(0, 1, (n, 3))
    F   = model_true(X, t)
    eps = sample_noise(t, sigma2, ell, n, rng)
    return X, F + eps   # X, Y_noisy

# ===========================================================================
# 3.  PCA wrapper  (shared by RF+PCA and MLP+PCA)
# ===========================================================================

class PCAWrapper:
    """
    Wraps any sklearn-compatible regressor with PCA output compression.

    fit(X, Y):
        1. Fit PCA on Y  ->  scores (n, K)
        2. Fit base_model on (X, scores)

    predict(X):
        1. base_model predicts scores  (N, K)
        2. PCA inverse transform       (N, T)

    Because PCA is a linear operator it commutes with the Möbius
    transform (Section 4.5, Proposition 2), so the pure effects
    recovered via Möbius on the wrapped model are projections of
    the true Hilbert-valued pure effects onto the PCA subspace.
    """

    def __init__(self, base_model, n_pca):
        self.base_model = base_model
        self.n_pca      = n_pca
        self.pca        = None

    def fit(self, X, Y):
        self.pca   = PCA(n_components=self.n_pca)
        scores     = self.pca.fit_transform(Y)   # (n, K)
        self.base_model.fit(X, scores)
        return self

    def predict(self, X):
        scores = self.base_model.predict(X)      # (N, K)
        return self.pca.inverse_transform(scores) # (N, T)

    @property
    def variance_explained(self):
        if self.pca is None:
            return None
        return float(self.pca.explained_variance_ratio_.sum())

# ===========================================================================
# 4.  Model factories
# ===========================================================================

# ---- M1: Ridge ----

def make_ridge():
    """
    Independent Ridge regression at each time point.
    Linear in features — CANNOT capture the X1*X2 interaction.
    Included as a deliberately misspecified baseline.
    The framework should recover m_{(1,2)} ≈ 0 regardless of n.
    """
    return MultiOutputRegressor(Ridge(alpha=1.0))


# ---- M2: Random Forest (no PCA) ----

def make_rf(n_estimators, rf_jobs, random_state):
    """
    Multi-output Random Forest.
    sklearn handles (n, T) output natively via shared tree structure.
    Captures interactions through tree splits.
    rf_jobs controls threads per RF instance — keep small when many
    parallel outer workers are running to avoid thread contention.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=rf_jobs,
        random_state=random_state,
    )


# ---- M3: Random Forest + PCA ----

def make_rf_pca(n_estimators, n_pca, rf_jobs, random_state):
    """
    Random Forest on PCA scores.
    Fits K independent RF regressors (one per PCA component)
    then reconstructs via PCA inverse transform.
    Tests whether compressing the output first helps RF.
    """
    base = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=rf_jobs,
            random_state=random_state,
        )
    )
    return PCAWrapper(base, n_pca)


# ---- M4: NGBoost (no PCA) ----

def make_ngboost_direct(n_est, random_state):
    """
    NGBoost without PCA.
    Fits T independent NGBRegressor(Normal) models, one per time point,
    via MultiOutputRegressor.

    Limitation: treats each time point as an independent scalar output.
    No temporal correlation is modelled between time points.
    The mean predictions are still valid for Möbius estimation.

    Contrast with NGBoost+PCA which models the full K-dimensional
    covariance via MultivariateNormal.
    """
    return MultiOutputRegressor(
        NGBRegressor(
            Dist          = Normal,
            n_estimators  = n_est,
            learning_rate = 0.05,
            random_state  = random_state,
            verbose       = False,
        )
    )


# ---- M5: NGBoost + PCA ----

class _NGBoostPCAModel:
    """
    NGBoost with MultivariateNormal distribution on PCA-reduced scores.

    This is the main probabilistic model of the paper.
    Maps X -> MultivariateNormal(mu in R^K, Sigma in R^{KxK})
    where K = n_pca << T = 240.

    Why PCA is necessary:
      - MultivariateNormal(T=240) requires a 240x240 covariance matrix
        with 240^2 / 2 ≈ 29k parameters — infeasible for n < 10000
      - PCA reduces this to K x K = 8 x 8 = 64 parameters
      - The trajectories lie on a low-dimensional manifold spanned by
        phi1, phi2, phi3, phi12 — K=8 is already generous

    Connects directly to Section 4.5 of the paper.
    """

    def __init__(self, n_pca, n_est, random_state, lr=0.05):
        self.n_pca = n_pca
        self.n_est = n_est
        self.rs    = random_state
        self.lr    = lr
        self.pca   = None
        self.ngb   = None

    def fit(self, X, Y):
        self.pca = PCA(n_components=self.n_pca)
        scores   = self.pca.fit_transform(Y)   # (n, K)
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
        scores = self.ngb.pred_dist(X).loc     # (N, K) mean
        return self.pca.inverse_transform(scores)  # (N, T)

    @property
    def variance_explained(self):
        if self.pca is None:
            return None
        return float(self.pca.explained_variance_ratio_.sum())


def make_ngboost_pca(n_est, n_pca, random_state):
    return _NGBoostPCAModel(
        n_pca=n_pca, n_est=n_est, random_state=random_state
    )


def get_ngboost_n_est(n, base):
    """Scale NGBoost estimators with n to control runtime."""
    if n <= 500:  return base
    if n <= 2000: return max(50, base // 2)
    return max(30, base // 4)


# ---- M6 / M7: MLP (with / without PCA) ----

class _MLPNet(nn.Module):
    """
    Bottleneck MLP: X -> encoder -> z in R^K -> decoder -> R^out_dim.

    When used without PCA: out_dim = T = 240 (direct trajectory)
    When used with PCA:    out_dim = K = n_pca (PCA scores)

    Tanh activations produce smooth outputs appropriate for
    the smooth basis functions phi1, phi2, phi3, phi12.
    """

    def __init__(self, p, latent_dim, out_dim,
                 hidden_enc, hidden_dec):
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
    """
    Trains _MLPNet with early stopping and LR scheduling.

    Used for both MLP variants:
      - Direct (no PCA): target shape (n, T), latent_dim acts as
        internal bottleneck only
      - Via PCAWrapper:  target shape (n, K), latent_dim = K

    Parameters
    ----------
    p           : input dimension (3 features)
    latent_dim  : bottleneck dimension
    hidden_enc  : tuple of encoder hidden layer sizes
    hidden_dec  : tuple of decoder hidden layer sizes
    device      : 'cpu' or 'cuda'
    """

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
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, factor=0.5)
        criterion = nn.MSELoss()

        # Train / validation split
        n     = len(X)
        n_val = max(2, int(0.15 * n))
        idx   = np.random.permutation(n)
        vi, ti = idx[:n_val], idx[n_val:]

        def _t(arr):
            return torch.tensor(
                arr, dtype=torch.float32
            ).to(self.device)

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
                best_state = {
                    k: v.clone()
                    for k, v in self.net.state_dict().items()
                }
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
                torch.tensor(
                    X, dtype=torch.float32
                ).to(self.device)
            )
        return out.cpu().numpy()


def _mlp_capacity(n):
    """
    Return (hidden_enc, hidden_dec, epochs, batch, patience)
    scaled to training size n.

    Small n: shallow, heavily regularised
    Large n: deeper, less regularised
    The crossover region (n ~ 500-2000) is where the neural
    network transitions from under- to over-performing classical
    models — itself an interesting empirical finding.
    """
    if n < 200:
        return (16,),        (32,),           300,  16,  25
    elif n < 500:
        return (32, 32),     (64, 64),         500,  32,  35
    elif n < 2000:
        return (64, 64),     (128, 128),        700,  64,  50
    elif n < 5000:
        return (64, 64, 64), (128, 256),        800, 128,  60
    else:
        return (64, 64, 64), (128, 256, 256),  1000, 256,  80


def make_mlp_direct(n, T, n_pca, rng, device='cpu'):
    """
    MLP without PCA.
    Directly predicts the full T=240 trajectory.
    latent_dim = n_pca keeps the bottleneck comparable to MLP+PCA.
    """
    henc, hdec, epochs, batch, patience = _mlp_capacity(n)
    return _MLPModel(
        p=3, latent_dim=n_pca,
        hidden_enc=henc, hidden_dec=hdec,
        lr=1e-3, n_epochs=epochs,
        batch_size=batch, patience=patience,
        weight_decay=1e-4,
        random_state=int(rng.integers(0, 2**31)),
        device=device,
    )


def make_mlp_pca(n, n_pca, rng, device='cpu'):
    """
    MLP + PCA.
    Predicts K=n_pca PCA scores then reconstructs via PCA inverse.
    The base MLP has latent_dim = n_pca — same bottleneck as MLP direct
    but output layer is K not T.
    """
    henc, hdec, epochs, batch, patience = _mlp_capacity(n)
    base = _MLPModel(
        p=3, latent_dim=n_pca,
        hidden_enc=henc, hidden_dec=hdec,
        lr=1e-3, n_epochs=epochs,
        batch_size=batch, patience=patience,
        weight_decay=1e-4,
        random_state=int(rng.integers(0, 2**31)),
        device=device,
    )
    return PCAWrapper(base, n_pca)

# ===========================================================================
# 5.  Möbius estimation
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
    """
    Local prediction game values v(S)(t) for all S.
    predict_fn : (N, p) -> (N, T)
    Returns dict: subset_tuple -> (T,) array
    """
    p     = len(x_star)
    v     = {}
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(1, p+1), r)
        for r in range(0, p+1)
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
    """
    Möbius inversion: m(S) = sum_{L subset S} (-1)^{|S|-|L|} v(L).
    Returns dict: subset_tuple -> (T,) array.
    """
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(1, p+1), r)
        for r in range(0, p+1)
    ))
    m = {}
    for S in all_S:
        val = None
        for L in itertools.chain.from_iterable(
            itertools.combinations(S, r)
            for r in range(len(S)+1)
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
# 6.  Error metrics
# ===========================================================================

def l2_error_normalized(est, truth, t):
    """
    Normalised L2 error: ||f_hat - f||_2 / ||f||_2.
    Measures curve shape recovery.
    0 = perfect, 1 = error as large as signal.
    """
    norm_truth = np.sqrt(np.trapezoid(truth**2, t))
    norm_err   = np.sqrt(np.trapezoid((est - truth)**2, t))
    return float(norm_err / max(norm_truth, 1e-10))


def aggregated_error(est_scalar, truth_scalar):
    """
    Relative aggregated error: |Phi_hat - Phi| / |Phi|.
    Measures scalar integral recovery.
    """
    return float(
        abs(est_scalar - truth_scalar)
        / max(abs(truth_scalar), 1e-10)
    )

# ===========================================================================
# 7.  Single run  (one seed, one n)
#     Designed to be called in parallel — fully self-contained,
#     no shared mutable state, returns only plain arrays.
# ===========================================================================

def run_one(n, seed, t, sigma2, n_pca, base_n_est,
            rf_jobs, device):
    """
    One Monte Carlo repetition for a given (n, seed).

    Returns
    -------
    dict with keys:
        (tag, S) -> {'l2_err': float, 'agg_err': float,
                     'effect': (T,) array}
        '_truth_effects' -> dict S -> (T,) array
        '_truth_agg'     -> dict S -> float
        '_mobius'        -> dict tag -> dict S -> (T,) array
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    warnings.filterwarnings('ignore')

    rng = np.random.default_rng(seed)
    T   = len(t)

    truth_effects = analytical_pure_effects(t)
    truth_agg     = analytical_aggregated(truth_effects, t)

    # Background samples for Möbius (same size as training)
    X_bg             = rng.uniform(0, 1, (n, 3))
    X_train, Y_train = generate_training_data(
        n, t, sigma2, NOISE_ELL, rng
    )

    n_est = get_ngboost_n_est(n, base_n_est)
    rs    = int(rng.integers(0, 2**31))

    # ------------------------------------------------------------------
    # Fit all models
    # ------------------------------------------------------------------

    ridge_model = make_ridge()
    ridge_model.fit(X_train, Y_train)

    rf_model = make_rf(
        n_estimators=200, rf_jobs=rf_jobs, random_state=rs
    )
    rf_model.fit(X_train, Y_train)

    rf_pca_model = make_rf_pca(
        n_estimators=200, n_pca=n_pca,
        rf_jobs=rf_jobs, random_state=rs
    )
    rf_pca_model.fit(X_train, Y_train)

    ngb_model = make_ngboost_direct(n_est=n_est, random_state=rs)
    ngb_model.fit(X_train, Y_train)

    ngb_pca_model = make_ngboost_pca(
        n_est=n_est, n_pca=n_pca, random_state=rs
    )
    ngb_pca_model.fit(X_train, Y_train)

    mlp_model = make_mlp_direct(
        n=n, T=T, n_pca=n_pca, rng=rng, device=device
    )
    mlp_model.fit(X_train, Y_train)

    mlp_pca_model = make_mlp_pca(
        n=n, n_pca=n_pca, rng=rng, device=device
    )
    mlp_pca_model.fit(X_train, Y_train)

    # ------------------------------------------------------------------
    # Möbius for oracle + all ML models
    # ------------------------------------------------------------------

    def oracle_fn(X): return model_true(X, t)

    predict_fns = {
        'oracle':      oracle_fn,
        'ridge':       ridge_model.predict,
        'rf':          rf_model.predict,
        'rf_pca':      rf_pca_model.predict,
        'ngboost':     ngb_model.predict,
        'ngboost_pca': ngb_pca_model.predict,
        'mlp':         mlp_model.predict,
        'mlp_pca':     mlp_pca_model.predict,
    }

    mobius = {}
    for tag, fn in predict_fns.items():
        mobius[tag] = estimate_mobius(fn, X_STAR, X_bg, t)

    # ------------------------------------------------------------------
    # Compute errors and package output
    # ------------------------------------------------------------------

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
# 8.  Parallel experiment loop
# ===========================================================================

def run_experiments(n_runs, base_seed, t, sigma2,
                    n_pca, n_est, n_values,
                    n_jobs, rf_jobs, device, cache_dir):
    """
    Parallelises over all (n, seed) pairs using joblib.

    All 240 jobs (8 n-values × 30 runs) are submitted at once.
    joblib/loky manages the worker pool automatically.

    Thread budget:
      n_jobs workers × rf_jobs RF threads = total RF threads
      e.g. 32 × 4 = 128 RF threads — matches your 128-core server
      leaving headroom for MLP and NGBoost.
    """
    os.makedirs(cache_dir, exist_ok=True)

    rng_master = np.random.default_rng(base_seed)
    seeds      = rng_master.integers(0, 2**31, size=n_runs)

    jobs = [
        (n, int(seeds[i]))
        for n in n_values
        for i in range(n_runs)
    ]

    log.info(f'Total jobs      : {len(jobs)}')
    log.info(f'Parallel workers: {n_jobs}')
    log.info(f'RF threads/job  : {rf_jobs}')
    log.info(f'MLP device      : {device}')

    t0  = time.time()
    raw = Parallel(
        n_jobs=n_jobs,
        verbose=10,
        backend='loky',
    )(
        delayed(run_one)(
            n, seed, t, sigma2, n_pca, n_est,
            rf_jobs, device
        )
        for n, seed in jobs
    )
    elapsed = time.time() - t0
    log.info(f'Experiments finished in {elapsed/60:.1f} min')

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------

    l2_errors  = {
        n: {tag: {S: [] for S in REPORT_SUBSETS}
            for tag in ALL_MODELS}
        for n in n_values
    }
    agg_errors = {
        n: {tag: {S: [] for S in REPORT_SUBSETS}
            for tag in ALL_MODELS}
        for n in n_values
    }
    representative = None

    for (n, _seed), res in zip(jobs, raw):
        for tag in ALL_MODELS:
            for S in REPORT_SUBSETS:
                l2_errors[n][tag][S].append(
                    res[(tag, S)]['l2_err'])
                agg_errors[n][tag][S].append(
                    res[(tag, S)]['agg_err'])
        if n == REP_N and REP_N in n_values:
            representative = res

    # Convert to numpy arrays
    for n in n_values:
        for tag in ALL_MODELS:
            for S in REPORT_SUBSETS:
                l2_errors[n][tag][S]  = np.array(
                    l2_errors[n][tag][S])
                agg_errors[n][tag][S] = np.array(
                    agg_errors[n][tag][S])

    _save_cache(l2_errors, agg_errors, representative,
                n_values, cache_dir)

    return l2_errors, agg_errors, representative

# ===========================================================================
# 9.  Cache helpers
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
        np.savez(
            os.path.join(cache_dir, f'results_n{n}.npz'), **d
        )

    if representative is not None:
        rep_d = {}
        for tag in ALL_MODELS:
            for S in REPORT_SUBSETS:
                key = f'{tag}_{"_".join(map(str, S))}_effect'
                rep_d[key] = representative[(tag, S)]['effect']
        for S in REPORT_SUBSETS:
            key = f'truth_{"_".join(map(str, S))}'
            rep_d[key] = representative['_truth_effects'][S]
        np.savez(
            os.path.join(cache_dir, 'representative.npz'),
            **rep_d
        )

    log.info(f'Results cached to {cache_dir}')


def _load_cache(n_values, cache_dir):
    l2_errors  = {
        n: {tag: {S: None for S in REPORT_SUBSETS}
            for tag in ALL_MODELS}
        for n in n_values
    }
    agg_errors = {
        n: {tag: {S: None for S in REPORT_SUBSETS}
            for tag in ALL_MODELS}
        for n in n_values
    }

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
# 10.  Plotting helpers
# ===========================================================================

def savefig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    log.info(f'Saved: {path}')


def _style_ax(ax, ylabel=None, xlabel='Time (h)'):
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    ax.axvspan(8,  12, alpha=0.05, color='#2a9d8f', zorder=0)
    ax.axvspan(16, 20, alpha=0.05, color='#e9c46a', zorder=0)

# ===========================================================================
# 11.  Fig 1 — N-recovery
# ===========================================================================

def plot_n_recovery(l2_errors, agg_errors, n_values, n_runs):
    """
    2 x 4 grid.
    Rows:    normalised L2 error / relative aggregated error
    Columns: four subsets
    Lines:   all 8 models (oracle + 7 ML)

    Y-axis: linear scale with sensible limits per subset.
    Ridge interaction term shown with dashed reference line at 1.0.
    """
    n_cols = len(REPORT_SUBSETS)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(4.5 * n_cols, 9)
    )
    fig.suptitle(
        f'N-Recovery: mean ± 1 std over {n_runs} runs\n'
        'All models vs analytical ground truth',
        fontsize=11, fontweight='bold',
    )
    xtl = [
        str(n) if n < 1000 else f'{n//1000}k'
        for n in n_values
    ]

    # Compute y-axis limits per (row, col) dynamically
    # so each panel uses the full available space
    def get_ylim(row, S, metric_dict, pad=0.05):
        """
        Compute sensible y limits for a panel.
        Excludes Ridge on interaction (always 1.0) from
        the limit computation so other models are visible.
        """
        vals = []
        for tag in ALL_MODELS:
            # Skip Ridge on interaction — it dominates the scale
            if tag == 'ridge' and S == (1, 2):
                continue
            means = np.array([metric_dict[n][tag][S].mean()
                              for n in n_values])
            stds  = np.array([metric_dict[n][tag][S].std()
                              for n in n_values])
            vals.extend((means + stds).tolist())
            vals.extend(means.tolist())

        if not vals:
            return 0.0, 1.0

        vmax = max(vals)
        vmin = max(0.0, min(vals) - pad)
        # Add padding at top
        vmax = vmax + pad * vmax
        # Cap at 1.5 for readability — annotate anything above
        return vmin, min(vmax, 1.5)

    for col, S in enumerate(REPORT_SUBSETS):
        for row, (metric, ylabel) in enumerate([
            (l2_errors,  'Normalised L2 error'),
            (agg_errors, 'Relative agg. error'),
        ]):
            ax    = axes[row, col]
            ymin, ymax = get_ylim(row, S, metric)

            for tag in ALL_MODELS:
                means = np.array([
                    metric[n][tag][S].mean() for n in n_values
                ])
                stds  = np.array([
                    metric[n][tag][S].std()  for n in n_values
                ])
                ax.plot(
                    n_values, means,
                    color=MODEL_COLORS[tag],
                    ls=MODEL_LS[tag],
                    marker=MODEL_MARKERS[tag],
                    lw=1.8, ms=5,
                    label=MODEL_LABELS[tag],
                )
                ax.fill_between(
                    n_values,
                    np.clip(means - stds, 0, None),
                    means + stds,
                    alpha=0.10,
                    color=MODEL_COLORS[tag],
                )

            # Reference line at 1.0 (= error as large as signal)
            ax.axhline(
                1.0, color='#e63946', lw=1.0,
                ls=':', alpha=0.7,
                label='Error = signal size' if col == 0 and row == 0
                      else None
            )

            # Annotate Ridge on interaction panel
            if S == (1, 2) and tag == 'ridge':
                ax.annotate(
                    'Ridge ≡ 1.0\n(predicts zero)',
                    xy=(n_values[len(n_values)//2], 1.0),
                    xytext=(n_values[1], 1.15),
                    fontsize=7,
                    color=MODEL_COLORS['ridge'],
                    arrowprops=dict(
                        arrowstyle='->', color='gray',
                        lw=0.8
                    ),
                )

            # Y-axis: linear scale with dynamic limits
            ax.set_ylim(ymin, ymax)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter('%.2f')
            )

            # Add horizontal grid lines for readability
            ax.yaxis.grid(
                True, linestyle=':', alpha=0.4, color='gray'
            )
            ax.set_axisbelow(True)

            # X-axis: log scale (n values span orders of magnitude)
            ax.set_xscale('log')
            ax.set_xticks(n_values)
            ax.set_xticklabels(xtl, fontsize=7, rotation=30)
            ax.tick_params(labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if row == 0:
                ax.set_title(
                    SUBSET_LABELS[S], fontsize=9,
                    fontweight='bold',
                    color=SUBSET_COLORS[S],
                )
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 1:
                ax.set_xlabel('n (training samples)', fontsize=8)
            if col == 0 and row == 0:
                ax.legend(
                    fontsize=6.5, ncol=2,
                    loc='upper right',
                    framealpha=0.9,
                )

    plt.tight_layout()
    savefig(fig, 'fig1_n_recovery.pdf')

# ===========================================================================
# 12.  Fig 2 — PCA impact per model family
# ===========================================================================

def plot_pca_impact(l2_errors, n_values, n_runs):
    """
    Three panels (RF / NGBoost / MLP).
    Each panel: no-PCA vs PCA + oracle as reference.
    Focuses on interaction term f_{X1,X2} — hardest subset.
    """
    S   = (1, 2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        r'Impact of PCA on interaction term $f_{\{X_1,X_2\}}$'
        f'\n(mean ± 1 std, {n_runs} runs)  '
        '— Oracle shown as reference',
        fontsize=11, fontweight='bold',
    )
    xtl = [
        str(n) if n < 1000 else f'{n//1000}k'
        for n in n_values
    ]

    for ax, (tag_no, tag_pca, title) in zip(axes, PCA_PAIRS):
        # Oracle reference
        om = np.array([
            l2_errors[n]['oracle'][S].mean() for n in n_values
        ])
        os_ = np.array([
            l2_errors[n]['oracle'][S].std()  for n in n_values
        ])
        ax.plot(
            n_values, om,
            color=MODEL_COLORS['oracle'],
            ls=MODEL_LS['oracle'],
            marker=MODEL_MARKERS['oracle'],
            lw=2.0, ms=5,
            label=MODEL_LABELS['oracle'],
        )
        ax.fill_between(
            n_values, om - os_, om + os_,
            alpha=0.08, color=MODEL_COLORS['oracle'],
        )

        # Model pair
        for tag in [tag_no, tag_pca]:
            means = np.array([
                l2_errors[n][tag][S].mean() for n in n_values
            ])
            stds  = np.array([
                l2_errors[n][tag][S].std()  for n in n_values
            ])
            ax.plot(
                n_values, means,
                color=MODEL_COLORS[tag],
                ls=MODEL_LS[tag],
                marker=MODEL_MARKERS[tag],
                lw=2.2, ms=6,
                label=MODEL_LABELS[tag],
            )
            ax.fill_between(
                n_values, means - stds, means + stds,
                alpha=0.12, color=MODEL_COLORS[tag],
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(n_values)
        ax.set_xticklabels(xtl, fontsize=7.5, rotation=30)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('n (training samples)', fontsize=9)
        ax.set_ylabel('Normalised L2 error', fontsize=9)
        ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, 'fig2_pca_impact.pdf')

# ===========================================================================
# 13.  Fig 3 — Time-resolved effect curves
# ===========================================================================

def plot_effect_comparison(representative, t, n_values):
    """
    2 x 4 grid of time-resolved effect curves.
    Row 0: without PCA  (analytical + oracle + ridge + rf + ngboost + mlp)
    Row 1: with PCA     (analytical + oracle + rf_pca + ngboost_pca + mlp_pca)
    """
    n_cols = len(REPORT_SUBSETS)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(4.5 * n_cols, 9)
    )
    fig.suptitle(
        f'Time-resolved pure effects at x* = (0.8, 0.9, 0.7)\n'
        f'Representative run, n = {max(n_values)}',
        fontsize=11, fontweight='bold',
    )

    truth_effects = representative['_truth_effects']

    row0_tags = ['oracle', 'ridge', 'rf',     'ngboost', 'mlp']
    row1_tags = ['oracle', 'rf_pca', 'ngboost_pca', 'mlp_pca']

    for col, S in enumerate(REPORT_SUBSETS):
        truth = truth_effects[S]
        for row, tags in enumerate([row0_tags, row1_tags]):
            ax = axes[row, col]

            # Analytical ground truth always shown
            ax.plot(
                t, truth,
                color='#1b2631', lw=2.8,
                label='Analytical', zorder=10,
            )
            for tag in tags:
                est = representative[(tag, S)]['effect']
                ax.plot(
                    t, est,
                    color=MODEL_COLORS[tag],
                    ls=MODEL_LS[tag],
                    lw=1.8, alpha=0.85,
                    label=MODEL_LABELS[tag],
                )

            _style_ax(
                ax,
                ylabel='Effect (a.u.)' if col == 0 else None,
            )
            if row == 0:
                ax.set_title(
                    SUBSET_LABELS[S], fontsize=9,
                    fontweight='bold',
                    color=SUBSET_COLORS[S],
                )
            if col == 0:
                ax.legend(fontsize=7, loc='upper right')
            if col == n_cols - 1:
                label = ('Without PCA' if row == 0
                         else 'With PCA')
                ax.text(
                    1.02, 0.5, label,
                    transform=ax.transAxes,
                    fontsize=9, va='center',
                    rotation=270, color='gray',
                )

    plt.tight_layout()
    savefig(fig, 'fig3_effect_comparison.pdf')

# ===========================================================================
# 14.  Fig 4 — Aggregated scalar effects
# ===========================================================================

def plot_aggregated_effects(representative, t):
    """
    Grouped bar chart.
    Groups: four subsets.
    Bars:   analytical (dark) + oracle (outlined) + 7 ML models.
    """
    truth_agg = {
        S: float(np.trapezoid(
            representative['_truth_effects'][S], t
        ))
        for S in REPORT_SUBSETS
    }

    x_pos          = np.arange(len(REPORT_SUBSETS))
    w              = 0.09
    tags_ordered   = ['oracle'] + [
        tg for tg in ALL_MODELS if tg != 'oracle'
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        r'Time-aggregated effects $\Phi_S = \int f_S(t)\,dt$'
        '\n(representative run)',
        fontsize=11, fontweight='bold',
    )

    # Analytical reference
    ax.bar(
        x_pos - 4.5 * w,
        [truth_agg[S] for S in REPORT_SUBSETS],
        width=w, color='#1b2631', alpha=0.9,
        label='Analytical',
    )

    for i, tag in enumerate(tags_ordered):
        vals = [
            float(np.trapezoid(
                representative[(tag, S)]['effect'], t
            ))
            for S in REPORT_SUBSETS
        ]
        ec  = 'black' if tag == 'oracle' else 'none'
        lw  = 1.5     if tag == 'oracle' else 0.0
        ax.bar(
            x_pos + (i - 3.5) * w, vals,
            width=w,
            color=MODEL_COLORS[tag],
            edgecolor=ec, linewidth=lw,
            alpha=0.85,
            label=MODEL_LABELS[tag],
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [SUBSET_LABELS[S] for S in REPORT_SUBSETS],
        fontsize=9,
    )
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(r'$\Phi_S$', fontsize=10)
    ax.legend(fontsize=7.5, ncol=3, loc='upper right')

    plt.tight_layout()
    savefig(fig, 'fig4_aggregated_effects.pdf')

# ===========================================================================
# 15.  Tables
# ===========================================================================

def _render_df_to_ax(ax, df, title, fontsize=7.5):
    """Render a pandas DataFrame as a styled matplotlib table."""
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

    # Header row styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2a9d8f')
        table[0, j].set_text_props(
            color='white', fontweight='bold'
        )

    # Alternating row colours
    for i in range(len(row_labels)):
        fc = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(fc)

    # Row label styling
    for i in range(len(row_labels)):
        table[i + 1, -1].set_facecolor('#e8f4f8')

    ax.set_title(
        title, fontsize=fontsize + 1.5,
        fontweight='bold', pad=12,
    )


def build_summary_table(l2_errors, agg_errors, n_values):
    """
    Compact summary: one row per model, one column per subset.
    Values: mean ± std L2 error AND agg error at the largest n.
    Suitable for the paper body.
    """
    n_large = n_values[-1]
    rows    = []

    for tag in ALL_MODELS:
        row = {'Model': MODEL_LABELS[tag]}
        for S in REPORT_SUBSETS:
            sl  = SUBSET_LABELS_PLAIN[S]
            al2 = l2_errors[n_large][tag][S]
            aa  = agg_errors[n_large][tag][S]
            row[f'L2 {sl}']  = (
                f'{al2.mean():.3f} ± {al2.std():.3f}'
            )
            row[f'Agg {sl}'] = (
                f'{aa.mean():.3f} ± {aa.std():.3f}'
            )
        rows.append(row)

    return pd.DataFrame(rows).set_index('Model')


def build_full_table(error_dict, n_values, metric_name):
    """
    Full table: rows = (Model, Subset), columns = n values.
    Values: mean ± std.
    """
    rows = []
    for tag in ALL_MODELS:
        for S in REPORT_SUBSETS:
            row = {
                'Model':  MODEL_LABELS[tag],
                'Subset': SUBSET_LABELS_PLAIN[S],
            }
            for n in n_values:
                arr = error_dict[n][tag][S]
                col = (str(n) if n < 1000
                       else f'{n//1000}k')
                row[col] = (
                    f'{arr.mean():.3f} ± {arr.std():.3f}'
                )
            rows.append(row)

    return pd.DataFrame(rows).set_index(['Model', 'Subset'])


def build_convergence_table(l2_errors, agg_errors, n_values):
    """
    Combined convergence overview.
    Rows = (Model, Subset).
    Columns = selected n values, interleaved L2 and Agg.
    """
    n_select = [n for n in [50, 500, 2000, n_values[-1]]
                if n in n_values]
    rows     = []

    for tag in ALL_MODELS:
        for S in REPORT_SUBSETS:
            row = {
                'Model':  MODEL_LABELS[tag],
                'Subset': SUBSET_LABELS_PLAIN[S],
            }
            for n in n_select:
                col = (str(n) if n < 1000
                       else f'{n//1000}k')
                al2 = l2_errors[n][tag][S]
                aa  = agg_errors[n][tag][S]
                row[f'L2 n={col}']  = (
                    f'{al2.mean():.3f}±{al2.std():.3f}'
                )
                row[f'Agg n={col}'] = (
                    f'{aa.mean():.3f}±{aa.std():.3f}'
                )
            rows.append(row)

    return pd.DataFrame(rows).set_index(['Model', 'Subset'])


def plot_tables(l2_errors, agg_errors, n_values, n_runs):
    """
    Export four table PDFs and matching CSVs.

    fig_table_summary.pdf
        Compact: mean L2 + Agg error at largest n.
        One row per model, two columns per subset.
        Suitable for paper body.

    fig_table_l2_full.pdf
        Full L2 error table: all models × subsets × n values.
        Appendix.

    fig_table_agg_full.pdf
        Full aggregated error table. Appendix.

    fig_table_convergence.pdf
        Both metrics at selected n values side by side.
        Convergence overview.

    CSV copies saved to plots/synthetic_experiments/part_1/csv/.
    """
    df_summary     = build_summary_table(
        l2_errors, agg_errors, n_values
    )
    df_l2_full     = build_full_table(
        l2_errors, n_values, 'L2'
    )
    df_agg_full    = build_full_table(
        agg_errors, n_values, 'Agg'
    )
    df_convergence = build_convergence_table(
        l2_errors, agg_errors, n_values
    )

    n_large = n_values[-1]
    nl_str  = str(n_large) if n_large < 1000 else f'{n_large//1000}k'

    # ---- Table 1: summary ----
    nr, nc = df_summary.shape
    fig, ax = plt.subplots(
        figsize=(max(10, nc * 2.0 + 3),
                 max(4,  nr * 0.55 + 2.0))
    )
    _render_df_to_ax(
        ax, df_summary,
        title=(
            f'Summary: L2 and aggregated error at n={nl_str}\n'
            f'(mean ± 1 std, {n_runs} runs)'
        ),
        fontsize=8,
    )
    plt.tight_layout()
    savefig(fig, 'fig_table_summary.pdf')

    # ---- Table 2: full L2 ----
    nr, nc = df_l2_full.shape
    fig, ax = plt.subplots(
        figsize=(max(14, nc * 2.2 + 4),
                 max(6,  nr * 0.42 + 2.5))
    )
    _render_df_to_ax(
        ax, df_l2_full,
        title=(
            f'Full results: Normalised L2 error\n'
            f'(mean ± 1 std, {n_runs} runs)'
        ),
        fontsize=7,
    )
    plt.tight_layout()
    savefig(fig, 'fig_table_l2_full.pdf')

    # ---- Table 3: full aggregated ----
    fig, ax = plt.subplots(
        figsize=(max(14, nc * 2.2 + 4),
                 max(6,  nr * 0.42 + 2.5))
    )
    _render_df_to_ax(
        ax, df_agg_full,
        title=(
            f'Full results: Relative aggregated error\n'
            f'(mean ± 1 std, {n_runs} runs)'
        ),
        fontsize=7,
    )
    plt.tight_layout()
    savefig(fig, 'fig_table_agg_full.pdf')

    # ---- Table 4: convergence overview ----
    nr, nc = df_convergence.shape
    fig, ax = plt.subplots(
        figsize=(max(16, nc * 2.0 + 4),
                 max(6,  nr * 0.42 + 2.5))
    )
    _render_df_to_ax(
        ax, df_convergence,
        title=(
            f'Convergence overview: L2 and aggregated error\n'
            f'(mean ± 1 std, {n_runs} runs)'
        ),
        fontsize=6.5,
    )
    plt.tight_layout()
    savefig(fig, 'fig_table_convergence.pdf')

    # ---- CSV exports ----
    csv_dir = os.path.join(PLOT_DIR, 'csv')
    os.makedirs(csv_dir, exist_ok=True)

    df_summary.to_csv(
        os.path.join(csv_dir, 'table_summary.csv')
    )
    df_l2_full.to_csv(
        os.path.join(csv_dir, 'table_l2_full.csv')
    )
    df_agg_full.to_csv(
        os.path.join(csv_dir, 'table_agg_full.csv')
    )
    df_convergence.to_csv(
        os.path.join(csv_dir, 'table_convergence.csv')
    )
    log.info(f'CSV tables saved to {csv_dir}/')

# ===========================================================================
# 16.  Main
# ===========================================================================

def main():
    args     = parse_args()
    n_runs   = 1 if args.quick else args.n_runs
    n_pca    = args.n_pca
    n_est    = args.n_est
    n_jobs   = args.n_jobs
    rf_jobs  = args.rf_jobs
    device   = args.device
    n_values = N_VALUES_QUICK if args.quick else N_VALUES_FULL

    # GPU setup
    if device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        if not torch.cuda.is_available():
            log.warning(
                'CUDA not available — falling back to CPU'
            )
            device = 'cpu'
        else:
            log.info(
                f'Using GPU {args.gpu_id}: '
                f'{torch.cuda.get_device_name(args.gpu_id)}'
            )

    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    log.info('=' * 60)
    log.info('Part 1 — Synthetic Experiments: Model Comparison')
    log.info(f'  Models   : {ALL_MODELS}')
    log.info(f'  n_runs   : {n_runs}')
    log.info(f'  n_values : {n_values}')
    log.info(f'  n_pca    : {n_pca}')
    log.info(f'  n_est    : {n_est}  (base, scaled with n)')
    log.info(f'  n_jobs   : {n_jobs}  (outer parallelism)')
    log.info(f'  rf_jobs  : {rf_jobs} (RF inner threads)')
    log.info(f'  device   : {device}')
    log.info(f'  plot_dir : {PLOT_DIR}')
    log.info(f'  cache    : {cache_dir}')
    log.info('=' * 60)

    if args.plots_only:
        log.info('--plots_only flag: loading results from cache')
        l2_errors, agg_errors, rep = _load_cache(
            n_values, cache_dir
        )
    else:
        # Calibrate noise level to 20% of signal variance
        sig_var = compute_signal_variance(
            t_grid, rng=np.random.default_rng(args.seed)
        )
        sigma2 = 0.20 * sig_var
        log.info(f'Signal variance : {sig_var:.4f}')
        log.info(
            f'Noise sigma²    : {sigma2:.4f}  '
            f'(SNR ~ {sig_var/sigma2:.1f})'
        )

        l2_errors, agg_errors, rep = run_experiments(
            n_runs    = n_runs,
            base_seed = args.seed,
            t         = t_grid,
            sigma2    = sigma2,
            n_pca     = n_pca,
            n_est     = n_est,
            n_values  = n_values,
            n_jobs    = n_jobs,
            rf_jobs   = rf_jobs,
            device    = device,
            cache_dir = cache_dir,
        )

    log.info('Generating figures ...')
    plot_n_recovery(l2_errors, agg_errors, n_values, n_runs)
    plot_pca_impact(l2_errors, n_values, n_runs)
    plot_effect_comparison(rep, t_grid, n_values)
    plot_aggregated_effects(rep, t_grid)

    log.info('Generating tables ...')
    plot_tables(l2_errors, agg_errors, n_values, n_runs)

    log.info(f'All outputs saved to {PLOT_DIR}/')


if __name__ == '__main__':
    main()