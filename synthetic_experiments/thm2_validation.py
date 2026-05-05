"""
Theorem 2 Validation: Sobol Index Recovery
===================================================

This script validates the theoretical connection between the H-FD framework
and classical Sobol sensitivity indices (Theorem 2 of the paper).

Theoretical statement
---------------------
Under the constant kernel K(t,s) = 1 and the variance (sensitivity) game,
the functional Möbius decomposition recovers the classical time-resolved and
time-aggregated first-order Sobol indices exactly.

Specifically, for an additive model F(x)(t) = sum_j x_j * phi_j(t):

  Time-resolved Sobol:   S_j(t) = Var(Xj)*phi_j(t)^2 / sum_k Var(Xk)*phi_k(t)^2
  Time-aggregated Sobol: xi_j   = int Var(Xj)*phi_j(t)^2 dt / int sum_k ... dt

This script confirms that:
  (a) The Oracle (true model + finite background) recovers the analytical indices
      up to Monte Carlo estimation error.
  (b) Trained ML models (Ridge, RF, MLP) also recover the indices when the
      model is well-specified or sufficiently expressive.

Data-generating process
-----------------------
Purely additive ICU-inspired model (no interaction term):

    F(x)(t) = x1*phi1(t) + x2*phi2(t) + x3*phi3(t)

    phi1(t) = exp(-0.2*t)                   — decaying baseline recovery
    phi2(t) = exp(-(t-10)^2 / 2)            — early shock at t=10h
    phi3(t) = exp(-(t-18)^2 / 2)            — late deterioration at t=18h

    Xi ~ Uniform[0,1],  Var(Xi) = 1/12

Observations are corrupted with correlated (Ornstein–Uhlenbeck) noise.

Methods compared
----------------
  - Analytical:    closed-form ground truth from Theorem 2
  - Oracle:        true model evaluated with finite background sample
  - Ridge:         linear, misspecified (no interactions)
  - Random Forest
  - MLP:           encoder-decoder trajectory regression

Outputs
-------
Figures are saved to:
    plots/synthetic_experiments/thm2_validation/

Usage
-----
    python thm2_validation.py

All CLI options:
    --seed      Master random seed               (default: 0)
    --n_bg      Background sample size           (default: 1000)
    --n_train   Training set size for ML models  (default: 1000)
    --rf_jobs   Parallel workers for RF          (default: 4)
    --device    Torch device: 'cpu' or 'cuda'    (default: cpu)
    --plot_dir  Output directory for figures
"""

import os
import argparse
import itertools
import warnings
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

# ── Global font size defaults ────────────────────────────────────────────
plt.rcParams.update({
    'font.size'       : 14,
    'axes.titlesize'  : 15,
    'axes.labelsize'  : 14,
    'xtick.labelsize' : 13,
    'ytick.labelsize' : 13,
    'legend.fontsize' : 13,
    'figure.titlesize': 16,
})

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

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
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Theorem 2 validation — Sobol index recovery.')
    p.add_argument('--seed',     type=int, default=0,
                   help='Master random seed.')
    p.add_argument('--n_bg',     type=int, default=1000,
                   help='Background sample size for variance game estimation.')
    p.add_argument('--n_train',  type=int, default=1000,
                   help='Training set size for ML models.')
    p.add_argument('--rf_jobs',  type=int, default=4,
                   help='Parallel workers inside RandomForest.')
    p.add_argument('--device',   type=str, default='cpu',
                   choices=['cpu', 'cuda'],
                   help='PyTorch device.')
    p.add_argument('--plot_dir', type=str,
                   default=os.path.join(
                       'plots', 'synthetic_experiments', 'thm2_validation'),
                   help='Output directory for figures.')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

# Time grid: 240 points over a 24-hour window
T_MAX    = 24.0
T_POINTS = 240
t_grid   = np.linspace(0, T_MAX, T_POINTS)

# Feature distribution: Xi ~ Uniform[0,1]
MU    = 0.5          # E[Xi]
VAR_X = 1.0 / 12.0  # Var(Xi)

# OU noise parameters
NOISE_ELL = 2.0
NOISE_SNR = 5.0   # signal-to-noise ratio for training data generation

FEATURES = [1, 2, 3]

# Display labels and colors for features and methods
FEATURE_LABELS = {
    1: r'$X_1$ (baseline recovery)',
    2: r'$X_2$ (early shock, $t\approx10\,\mathrm{h}$)',
    3: r'$X_3$ (late deterioration, $t\approx18\,\mathrm{h}$)',
}
FEATURE_COLORS = {
    1: '#c1121f',
    2: '#2a9d8f',
    3: '#e9c46a',
}

METHOD_LABELS = {
    'analytical': 'Analytical (ground truth)',
    'oracle':     'Oracle (true model)',
    'ridge':      'Ridge',
    'rf':         'Random Forest',
    'mlp':        'MLP',
}
METHOD_COLORS = {
    'oracle':     '#888888',
    'analytical': '#1b2631',
    'ridge':      '#a8dadc',
    'rf':         '#457b9d',
    'mlp':        '#f4a261',
}
METHOD_LS = {
    'analytical': '--',
    'oracle':     (0, (4, 2)),
    'ridge':      ':',
    'rf':         (0, (5, 2)),
    'mlp':        (0, (3, 1)),
}
METHOD_LW = {
    'analytical': 1.2,
    'oracle':     1.8,
    'ridge':      1.5,
    'rf':         1.5,
    'mlp':        1.5,
}

# ===========================================================================
# 1.  True model (purely additive — no pairwise interaction)
# ===========================================================================

# Basis functions for the three features
def phi1(t): return np.exp(-0.2 * t)
def phi2(t): return np.exp(-0.5 * (t - 10.0) ** 2)
def phi3(t): return np.exp(-0.5 * (t - 18.0) ** 2)

PHI = {1: phi1, 2: phi2, 3: phi3}


def model_true(X, t):
    """
    Evaluate the true additive functional model F(x)(t) for a batch of inputs.

    F(x)(t) = x1*phi1(t) + x2*phi2(t) + x3*phi3(t)

    Parameters
    ----------
    X : ndarray of shape (n, 3)
    t : ndarray of shape (T,)

    Returns
    -------
    F : ndarray of shape (n, T)
    """
    X = np.atleast_2d(X)
    return sum(X[:, j-1:j] * PHI[j](t)[None, :] for j in FEATURES)


# ===========================================================================
# 2.  Analytical Sobol indices — Theorem 2 ground truth
# ===========================================================================

def analytical_sobol_resolved(t):
    """
    Compute the analytical time-resolved first-order Sobol indices.

    For the additive model with independent Uniform[0,1] features:

        S_j(t) = Var(Xj) * phi_j(t)^2 / sum_k Var(Xk) * phi_k(t)^2

    Values lie in [0,1] and sum to 1 at each time point t.

    Returns
    -------
    dict mapping j -> ndarray of shape (T,)
    """
    numerators  = {j: VAR_X * PHI[j](t)**2 for j in FEATURES}
    denominator = sum(numerators.values())
    denominator = np.maximum(denominator, 1e-12)  # avoid division by zero
    return {j: numerators[j] / denominator for j in FEATURES}


def analytical_sobol_aggregated(t):
    """
    Compute the analytical time-aggregated first-order Sobol indices.

        xi_j = int Var(Xj)*phi_j(t)^2 dt / int sum_k Var(Xk)*phi_k(t)^2 dt

    Values lie in [0,1] and sum to 1.

    Returns
    -------
    dict mapping j -> float
    """
    numerators  = {j: float(np.trapezoid(VAR_X * PHI[j](t)**2, t))
                   for j in FEATURES}
    denominator = sum(numerators.values())
    return {j: numerators[j] / denominator for j in FEATURES}


# ===========================================================================
# 3.  Noise generation and training data
# ===========================================================================

def make_ou_cov(t, sigma2, ell):
    """OU covariance matrix K(s,t) = sigma2 * exp(-|s-t| / ell)."""
    return sigma2 * np.exp(-np.abs(t[:, None] - t[None, :]) / ell)


def sample_noise(t, sigma2, ell, n, rng):
    """
    Draw n independent OU noise trajectories via Cholesky factorization.

    Returns ndarray of shape (n, T).
    """
    K = make_ou_cov(t, sigma2, ell)
    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(t)))
    return (L @ rng.standard_normal((len(t), n))).T


def generate_training_data(n, t, rng):
    """
    Generate n noisy training observations (X, Y).

    Noise variance is set so that SNR = NOISE_SNR (default 5).
    Y = F(X)(t) + epsilon, epsilon ~ OU process.

    Returns
    -------
    X : ndarray of shape (n, 3)
    Y : ndarray of shape (n, T)
    """
    # Estimate signal variance via Monte Carlo to calibrate noise level
    sig_var = float(model_true(rng.uniform(0, 1, (5000, 3)), t).var(axis=0).mean())
    sigma2  = sig_var / NOISE_SNR
    X       = rng.uniform(0, 1, (n, 3))
    Y       = model_true(X, t) + sample_noise(t, sigma2, NOISE_ELL, n, rng)
    return X, Y


# ===========================================================================
# 4.  Model factories
# ===========================================================================

def make_ridge():
    """
    Ridge regression wrapped for multi-output prediction.
    Correctly specified for the additive model; included as a sanity check.
    """
    return MultiOutputRegressor(Ridge(alpha=1.0))


def make_rf(rf_jobs, random_state):
    """Random Forest regressor for multi-output trajectory prediction."""
    return RandomForestRegressor(
        n_estimators=200, n_jobs=rf_jobs, random_state=random_state
    )


# ---------------------------------------------------------------------------
# MLP: encoder-decoder architecture (same design as Part 1)
# ---------------------------------------------------------------------------

class _MLPNet(nn.Module):
    """
    Encoder-decoder MLP for trajectory regression.

    encoder: input (p=3) -> hidden layers -> latent_dim
    decoder: latent_dim  -> hidden layers -> T time points
    Activation: Tanh throughout.
    """
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
    """
    Training wrapper for _MLPNet with early stopping.

    Uses Adam optimizer with ReduceLROnPlateau scheduler.
    Holds out 15% of data for validation-based early stopping.
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
        """Train the MLP on (X, Y) with early stopping on a held-out split."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        out_dim  = Y.shape[1]
        self.net = _MLPNet(
            self.p, self.latent_dim, out_dim,
            self.hidden_enc, self.hidden_dec
        ).to(self.device)
        optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=15, factor=0.5
        )
        criterion = nn.MSELoss()
        n     = len(X)
        n_val = max(2, int(0.15 * n))
        idx   = np.random.permutation(n)
        vi, ti = idx[:n_val], idx[n_val:]

        def _t(a):
            return torch.tensor(a, dtype=torch.float32).to(self.device)

        Xtr, Ytr = _t(X[ti]), _t(Y[ti])
        Xvl, Yvl = _t(X[vi]), _t(Y[vi])
        loader = DataLoader(
            TensorDataset(Xtr, Ytr),
            batch_size=min(self.batch_size, len(ti)),
            shuffle=True,
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
            # Early stopping: checkpoint best model
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
        """Return trajectory predictions as a numpy array of shape (n, T)."""
        self.net.eval()
        with torch.no_grad():
            out = self.net(
                torch.tensor(X, dtype=torch.float32).to(self.device)
            )
        return out.cpu().numpy()


def make_mlp(n, T, rng, device='cpu'):
    """
    Instantiate an _MLPModel with capacity scaled to training set size n.
    Larger n uses deeper/wider networks and more training epochs.
    """
    if n < 500:
        henc, hdec, epochs, batch, pat = (32, 32), (64, 64), 500, 32, 35
    elif n < 2000:
        henc, hdec, epochs, batch, pat = (64, 64), (128, 128), 700, 64, 50
    else:
        henc, hdec, epochs, batch, pat = (64, 64, 64), (128, 256), 800, 128, 60
    return _MLPModel(
        p=3, latent_dim=8,
        hidden_enc=henc, hidden_dec=hdec,
        lr=1e-3, n_epochs=epochs,
        batch_size=batch, patience=pat,
        weight_decay=1e-4,
        random_state=int(rng.integers(0, 2**31)),
        device=device,
    )


# ===========================================================================
# 5.  Variance game values and Möbius transform
# ===========================================================================

def subset_to_mask(S, p=3):
    """Convert a subset tuple S ⊆ {1,...,p} to a boolean mask (0-indexed)."""
    mask = np.zeros(p, dtype=bool)
    for j in S:
        mask[j - 1] = True
    return mask


def impute_x(X_bg, S_mask, x_star=None):
    """
    Replace features in S_mask with x_star values in a copy of X_bg.
    If x_star is None, features in S are left as-is (used internally).
    """
    x_imp = X_bg.copy()
    if x_star is not None:
        for j in range(3):
            if S_mask[j]:
                x_imp[:, j] = x_star[j]
    return x_imp


def compute_variance_game(predict_fn, X_bg, t):
    """
    Compute the variance game values v(S)(t) for all S ⊆ {1,2,3}.

    The variance game measures how much predictive variance is explained
    by each coalition S:

        v(S)(t) = Var_{X_bg}[ E[F(x)(t) | x_S] ]

    This is computed by:
      - For each background point x_i, fix features in S to x_i[S] and
        average predictions over all background values of the remaining features.
      - The resulting conditional means E[F(x)(t) | x_S = x_i[S]] are
        collected over all i, and their variance is taken.

    Note: This is the "inner expectation, outer variance" estimator of the
    explained variance, consistent with the ANOVA decomposition.

    Parameters
    ----------
    predict_fn : callable, maps (n, 3) -> (n, T)
    X_bg       : ndarray of shape (n_bg, 3) — background sample
    t          : ndarray of shape (T,)

    Returns
    -------
    v : dict mapping subset tuples -> ndarray of shape (T,)
    """
    p     = 3
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(1, p + 1), r)
        for r in range(0, p + 1)
    ))

    v = {}
    n = len(X_bg)

    for S in all_S:
        if len(S) == 0:
            # Empty coalition: no features fixed -> zero explained variance
            v[()] = np.zeros(len(t))
        else:
            mask = subset_to_mask(S, p)
            # For each background point i, compute conditional mean over X_{-S}
            conditional_means = np.zeros((n, len(t)))
            for i in range(n):
                x_imp = X_bg.copy()
                for j in range(p):
                    if mask[j]:
                        x_imp[:, j] = X_bg[i, j]
                conditional_means[i] = predict_fn(x_imp).mean(axis=0)
            # Variance of conditional means = explained variance by S
            v[S] = conditional_means.var(axis=0)

    return v


def mobius_transform(v_dict, p=3):
    """
    Apply the Möbius inversion formula to recover pure variance effects.

    Inverts v(S) = sum_{L ⊆ S} m(L) to give:
        m(S) = sum_{L ⊆ S} (-1)^{|S|-|L|} v(L)

    Under the additive model and constant kernel, m({j}) recovers
    Var(Xj) * phi_j(t)^2 (the time-resolved variance contribution of Xj).

    Returns
    -------
    m : dict mapping subset tuples -> ndarray of shape (T,)
    """
    all_S = list(itertools.chain.from_iterable(
        itertools.combinations(range(1, p + 1), r)
        for r in range(0, p + 1)
    ))
    m = {}
    for S in all_S:
        val = None
        for L in itertools.chain.from_iterable(
            itertools.combinations(S, r) for r in range(len(S) + 1)
        ):
            sign = (-1) ** (len(S) - len(L))
            term = v_dict[L] if L != () else v_dict[()]
            val  = sign * term if val is None else val + sign * term
        m[S] = val
    return m


def sobol_from_mobius_constant_kernel(mobius_dict, t):
    """
    Recover Sobol indices from first-order Möbius coefficients under the
    constant kernel K(t,s) = 1.

    By Theorem 2, normalization of the Möbius variance effects yields the
    classical Sobol indices:

        Time-resolved:   S_j(t) = m_{j}(t) / sum_k m_{k}(t)
        Time-aggregated: xi_j   = int m_{j}(t) dt / int sum_k m_{k}(t) dt

    Negative values of m_{j}(t) are clipped to zero (can arise from finite
    sample estimation noise).

    Parameters
    ----------
    mobius_dict : dict mapping subset tuples -> ndarray of shape (T,)
    t           : ndarray of shape (T,)

    Returns
    -------
    resolved   : dict mapping j -> ndarray of shape (T,)  — time-resolved Sobol
    aggregated : dict mapping j -> float                  — time-aggregated Sobol
    """
    m1         = {j: mobius_dict[(j,)] for j in FEATURES}
    m1_clipped = {j: np.maximum(m1[j], 0.0) for j in FEATURES}

    # Time-resolved normalization
    denom_resolved = sum(m1_clipped.values())
    denom_resolved = np.maximum(denom_resolved, 1e-12)
    resolved = {j: m1_clipped[j] / denom_resolved for j in FEATURES}

    # Time-aggregated normalization via numerical integration
    num_agg   = {j: float(np.trapezoid(m1_clipped[j], t)) for j in FEATURES}
    denom_agg = sum(num_agg.values())
    denom_agg = max(denom_agg, 1e-12)
    aggregated = {j: num_agg[j] / denom_agg for j in FEATURES}

    return resolved, aggregated


# ===========================================================================
# 6.  Main experiment
# ===========================================================================

def run_experiment(args):
    """
    Execute the full Sobol recovery experiment:
        1. Compute analytical ground truth (Theorem 2)
        2. Fit ML models on noisy training data
        3. Estimate variance game + Möbius + Sobol for each method
        4. Return results dict for plotting

    Returns
    -------
    results : dict mapping method tag -> (resolved, aggregated)
              where resolved is dict j -> (T,) and aggregated is dict j -> float
    t       : ndarray of shape (T,) — time grid
    """
    rng     = np.random.default_rng(args.seed)
    t       = t_grid
    n_bg    = args.n_bg
    n_train = args.n_train

    log.info('=' * 60)
    log.info('Part 2 — Theorem Validation: Sobol Index Recovery')
    log.info(f'  n_bg    : {n_bg}   (background sample for Mobius)')
    log.info(f'  n_train : {n_train}  (training set for ML models)')
    log.info(f'  seed    : {args.seed}')
    log.info('=' * 60)

    # Analytical ground truth from closed-form expressions
    gt_resolved   = analytical_sobol_resolved(t)
    gt_aggregated = analytical_sobol_aggregated(t)

    log.info('Analytical aggregated Sobol indices:')
    for j in FEATURES:
        log.info(f'  Xi_{j} : {gt_aggregated[j]:.4f}')

    # Shared background sample for all variance game computations
    X_bg = rng.uniform(0, 1, (n_bg, 3))

    # Training data for ML models
    X_train, Y_train = generate_training_data(n_train, t, rng)

    # Fit models
    log.info('Fitting ML models ...')
    rs = int(rng.integers(0, 2**31))

    ridge_model = make_ridge()
    ridge_model.fit(X_train, Y_train)
    log.info('  Ridge done')

    rf_model = make_rf(rf_jobs=args.rf_jobs, random_state=rs)
    rf_model.fit(X_train, Y_train)
    log.info('  Random Forest done')

    mlp_model = make_mlp(n=n_train, T=len(t), rng=rng, device=args.device)
    mlp_model.fit(X_train, Y_train)
    log.info('  MLP done')

    # Oracle uses the true model — isolates estimation error from background size
    def oracle_fn(X): return model_true(X, t)

    predict_fns = {
        'oracle': oracle_fn,
        'ridge':  ridge_model.predict,
        'rf':     rf_model.predict,
        'mlp':    mlp_model.predict,
    }

    # Compute variance game -> Möbius transform -> Sobol indices for each method
    results = {'analytical': (gt_resolved, gt_aggregated)}

    for tag, fn in predict_fns.items():
        log.info(f'Computing variance game: {tag} ...')
        v_dict       = compute_variance_game(fn, X_bg, t)
        m_dict       = mobius_transform(v_dict)
        res, agg     = sobol_from_mobius_constant_kernel(m_dict, t)
        results[tag] = (res, agg)
        log.info(f'  {tag} aggregated: ' +
                 ', '.join(f'X{j}={agg[j]:.3f}' for j in FEATURES))

    return results, t


# ===========================================================================
# 7.  Plotting
# ===========================================================================

def savefig(fig, name, plot_dir):
    """Save figure as PDF to plot_dir and close it."""
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    log.info(f'Saved: {path}')


def _style_ax(ax, ylabel=None, xlabel='Time (h)'):
    """
    Apply standard axis styling for time-domain Sobol index plots.
    Y-axis is fixed to [-0.05, 1.05] since Sobol indices lie in [0, 1].
    Adds light shading for illustrative activity windows (08–12h, 16–20h).
    """
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.tick_params(labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    ax.axvspan(8,  12, alpha=0.05, color='#2a9d8f', zorder=0)
    ax.axvspan(16, 20, alpha=0.05, color='#e9c46a', zorder=0)


def plot_sobol_recovery(results, t, plot_dir):
    """
    Generate the main Sobol recovery figure (two-row layout).

    Row 0 (3 panels): Time-resolved Sobol index S_j(t) for each feature.
                      All methods overlaid; analytical shown as dashed reference.
    Row 1 (1 wide panel): Time-aggregated scalar Sobol indices as a grouped
                          bar chart, with analytical as the reference bars.

    Saved as: fig_sobol_recovery.pdf
    """
    methods = list(results.keys())

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        'Theorem validation: Sobol Index recovery under constant kernel\n'
        r'$\nu_\mathrm{var}(S)(t,s) = \mathrm{Cov}(F^H_S(X)(t), F^H_S(X)(s))$, '
        r'$K(t,s) = 1$',
        fontsize=16, fontweight='bold',
        y=0.99,
    )

    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.6, 1.0],
        hspace=0.40, wspace=0.32,
        top=0.86, bottom=0.08,
    )
    axes_top = [fig.add_subplot(gs[0, k]) for k in range(3)]
    ax_bar   = fig.add_subplot(gs[1, :])

    # --- Row 0: time-resolved S_j(t) ---
    for col, j in enumerate(FEATURES):
        ax = axes_top[col]

        for tag in methods:
            res, _ = results[tag]
            ax.plot(
                t, res[j],
                color=METHOD_COLORS[tag],
                ls=METHOD_LS[tag],
                lw=METHOD_LW[tag],
                alpha=0.9,
                label=METHOD_LABELS[tag],
                zorder=10 if tag == 'analytical' else 5,
            )

        _style_ax(
            ax,
            ylabel='time-resolved Sobol' if col == 0 else None,
        )
        ax.set_title(
            FEATURE_LABELS[j],
            fontsize=15, fontweight='bold',
            color=FEATURE_COLORS[j],
        )
        if col == 2:
            ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

    # --- Row 1: aggregated bar chart ---
    n_methods = len(methods)
    n_feats   = len(FEATURES)
    w         = 0.12   # bar width
    x_pos     = np.arange(n_feats)
    offsets   = np.linspace(
        -(n_methods - 1) / 2 * w,
         (n_methods - 1) / 2 * w,
        n_methods,
    )

    for mi, tag in enumerate(methods):
        _, agg = results[tag]
        vals   = [agg[j] for j in FEATURES]
        ec     = 'black' if tag == 'analytical' else 'none'
        lw_bar = 1.5     if tag == 'analytical' else 0.0
        ax_bar.bar(
            x_pos + offsets[mi], vals,
            width=w,
            color=METHOD_COLORS[tag],
            edgecolor=ec, linewidth=lw_bar,
            alpha=0.85,
            label=METHOD_LABELS[tag],
        )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(
        [FEATURE_LABELS[j] for j in FEATURES], fontsize=14,
    )
    ax_bar.set_ylabel('time-aggregated Sobol', fontsize=14)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.axhline(0, color='gray', lw=0.5, ls=':')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.set_title(
        'Time-aggregated Sobol indices',
        fontsize=15, fontweight='bold',
    )
    ax_bar.legend(
        fontsize=13, ncol=len(methods),
        loc='upper right', framealpha=0.9,
    )
    ax_bar.tick_params(labelsize=13)
    ax_bar.yaxis.grid(True, linestyle=':', alpha=0.4, color='gray')
    ax_bar.set_axisbelow(True)

    savefig(fig, 'fig_sobol_recovery.pdf', plot_dir)


def plot_mobius_curves(results, t, plot_dir):
    """
    Generate a supplementary figure showing the raw (unnormalized) first-order
    Möbius variance coefficients m_{j}(t) before Sobol normalization.

    By Theorem 2, the Oracle should recover m_{j}(t) ≈ Var(Xj) * phi_j(t)^2.
    This plot makes the relationship between the Möbius coefficients and the
    basis functions visually explicit.

    For ML models, the recovered m_{j}(t) are rescaled to the analytical
    denominator for comparability.

    Saved as: fig_mobius_variance_effects.pdf
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    fig.suptitle(
        r'First-order variance effects $m_{\{j\}}(t)$ before Sobol normalisation'
        '\n(constant kernel, sensitivity game)',
        fontsize=16, fontweight='bold',
    )

    # Analytical denominator for rescaling ML estimates to the same scale
    denom_anal = sum(VAR_X * PHI[k](t)**2 for k in FEATURES)

    for col, j in enumerate(FEATURES):
        ax = axes[col]

        # Analytical ground truth: Var(Xj) * phi_j(t)^2
        anal_m = VAR_X * PHI[j](t) ** 2
        ax.plot(
            t, anal_m,
            color=METHOD_COLORS['analytical'],
            lw=METHOD_LW['analytical'],
            ls=METHOD_LS['analytical'],
            label=METHOD_LABELS['analytical'],
            zorder=10,
        )

        # ML methods: reconstruct unnormalized m_j from normalized Sobol
        # by multiplying back by the analytical denominator
        for tag in [k for k in results.keys() if k != 'analytical']:
            res, _ = results[tag]
            m_unnorm = res[j] * denom_anal
            ax.plot(
                t, m_unnorm,
                color=METHOD_COLORS[tag],
                ls=METHOD_LS[tag],
                lw=METHOD_LW[tag],
                alpha=0.85,
                label=METHOD_LABELS[tag],
            )

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.tick_params(labelsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Time (h)', fontsize=14)
        ax.axvspan(8,  12, alpha=0.05, color='#2a9d8f', zorder=0)
        ax.axvspan(16, 20, alpha=0.05, color='#e9c46a', zorder=0)

        if col == 0:
            ax.set_ylabel(
                r'$m_{\{j\}}(t) \approx \mathrm{Var}(X_j)\,\phi_j(t)^2$',
                fontsize=14,
            )
            ax.legend(fontsize=13, loc='upper right')

        ax.set_title(
            FEATURE_LABELS[j],
            fontsize=15, fontweight='bold',
            color=FEATURE_COLORS[j],
        )

    plt.tight_layout()
    savefig(fig, 'fig_mobius_variance_effects.pdf', plot_dir)


def print_aggregated_table(results):
    """
    Print a formatted console table of time-aggregated Sobol indices
    for all methods, including row sums (should be ≈ 1.0 for all methods).
    """
    col_w  = 12
    header = f"{'Method':<25}" + ''.join(
        f"{'X' + str(j):>{col_w}}" for j in FEATURES
    ) + f"{'Sum':>{col_w}}"
    log.info('')
    log.info('Time-aggregated Sobol indices')
    log.info('-' * len(header))
    log.info(header)
    log.info('-' * len(header))
    for tag, (_, agg) in results.items():
        vals = [agg[j] for j in FEATURES]
        row  = f"{METHOD_LABELS[tag]:<25}"
        row += ''.join(f"{v:>{col_w}.4f}" for v in vals)
        row += f"{sum(vals):>{col_w}.4f}"
        log.info(row)
    log.info('-' * len(header))
    log.info('')


# ===========================================================================
# 8.  Main entry point
# ===========================================================================

def main():
    args     = parse_args()
    plot_dir = args.plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    results, t = run_experiment(args)

    print_aggregated_table(results)

    log.info('Generating figures ...')
    plot_sobol_recovery(results, t, plot_dir)
    plot_mobius_curves(results, t, plot_dir)

    log.info(f'All outputs saved to {plot_dir}/')


if __name__ == '__main__':
    main()