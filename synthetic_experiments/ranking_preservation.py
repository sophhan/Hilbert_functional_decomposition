"""
Ranking Preservation Across Games and Kernels
=======================================================

This script tests whether time-aggregated feature importance rankings are
preserved across kernel choices for all three cooperative game formulations
supported by the functional decomposition framework.

Scientific context
------------------
A key desideratum for functional feature attribution is that the relative
importance ranking of features should not depend on the analyst's choice
of output kernel K — the kernel should refine the temporal structure of
attribution curves, not reverse who matters most.

This script verifies that property analytically for the ICU additive model
under three game types and three kernels.

Game formulations
-----------------
The three cooperative game types differ in what quantity v(S) measures:

  1. Prediction game:
       v(S)(t) = E[F(x)(t) | X_S = x*_S]   — conditional expectation
       Pure effect: m_{j}(t) = (x*_j - mu) * phi_j(t)
       Measures how much feature j shifts the predicted trajectory.

  2. Sensitivity game:
       v(S)(t,s) = Cov(F_S(X)(t), F_S(X)(s))   — output covariance
       Pure effect: m_{j}(t) = Var(Xj) * phi_j(t)^2
       Measures how much of total trajectory variance is attributable to j.

  3. Risk / MSE game:
       v(S)(t) = E[(Y(t) - F(x)(t))^2 | X_S]   — conditional MSE
       Pure effect: m_{j}(t) = -Var(Xj) * phi_j(t)^2
       Measures how much j reduces prediction error.
       (Mirror image of the sensitivity game — sign-flipped.)

For each game, the time-aggregated importance is:
    Phi_S = integral (K f_S)(t) dt

where (K f_S)(t) is the row-normalised kernel application of f_S.

Kernels compared
----------------
  - Identity:     K(t,s) = delta(t-s) — pointwise, no temporal coupling
  - OU (ell=4h):  K(t,s) = exp(-|t-s|/ell) — local symmetric smoothing
  - Correlation:  K derived from the ICU model's output covariance structure

Expected result
---------------
Rankings are preserved across all three kernels for all three games,
confirming that kernel choice affects the shape and temporal structure of
attribution curves but not the relative ordering of features by importance.

Data-generating process
-----------------------
Additive ICU model (no pairwise interaction):

    F(x)(t) = X1*phi1(t) + X2*phi2(t) + X3*phi3(t)

    phi1(t) = exp(-0.2*t)                — decaying baseline recovery
    phi2(t) = exp(-(t-10)^2 / 2)         — early shock at t=10h
    phi3(t) = exp(-(t-18)^2 / 2)         — late deterioration at t=18h

    Xi ~ Uniform[0,1],  x* = (0.8, 0.9, 0.7)

Outputs
-------
    plots/synthetic_experiments/kernel_guidance/fig6_ranking_preservation_games.pdf

    Layout: 2 rows x 3 columns
        Row 0: Time-resolved pure effects f_S(t) per game
        Row 1: Time-aggregated bar charts with ranking preservation status

Usage
-----
    python ranking_games.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

PLOT_DIR = os.path.join('plots', 'synthetic_experiments', 'kernel_guidance')
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------

FS_SUPTITLE = 17
FS_TITLE    = 14
FS_LABEL    = 13
FS_TICK     = 12
FS_LEGEND   = 12
FS_ANNOT    = 13

matplotlib.rcParams.update({
    'font.size':       FS_TICK,
    'axes.titlesize':  FS_TITLE,
    'axes.labelsize':  FS_LABEL,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'legend.fontsize': FS_LEGEND,
})

# ---------------------------------------------------------------------------
# Colors — Wong (2011) CVD-safe palette
# ---------------------------------------------------------------------------

C_X1 = '#0072B2'   # blue      — X1 (baseline recovery)
C_X2 = '#D55E00'   # vermillon — X2 (early shock)
C_X3 = '#CC79A7'   # purple    — X3 (late deterioration)

C_ID   = '#888888'   # Identity kernel
C_OU   = '#E69F00'   # OU kernel (amber)
C_CORR = '#009E73'   # Correlation kernel (dark teal)

FEAT_COLORS = {'X1': C_X1, 'X2': C_X2, 'X3': C_X3}

# ---------------------------------------------------------------------------
# Model and time grid
# ---------------------------------------------------------------------------

T_MAX, T_POINTS = 24.0, 240
t    = np.linspace(0, T_MAX, T_POINTS)
dt   = t[1] - t[0]
MU   = 0.5           # E[Xi] for Uniform[0,1]
VAR_X = 1.0 / 12.0  # Var(Xi) for Uniform[0,1]
X_STAR = np.array([0.8, 0.9, 0.7])   # evaluation point x*

# Basis functions
def phi1(t): return np.exp(-0.2 * t)
def phi2(t): return np.exp(-0.5 * (t - 10.0)**2)
def phi3(t): return np.exp(-0.5 * (t - 18.0)**2)

PHI        = {'X1': phi1, 'X2': phi2, 'X3': phi3}
FEAT_NAMES = ['X1', 'X2', 'X3']

# ===========================================================================
# Game-specific pure effects (first-order Möbius coefficients)
# ===========================================================================

def pure_effects_pred(x_star=X_STAR):
    """
    Prediction game: pure effect of feature j at x* under the additive model.

        m_{j}(t) = (x*_j - mu) * phi_j(t)

    This is the contribution of feature j to the centered prediction:
    F(x*)(t) - E[F(X)(t)] = sum_j (x*_j - mu) * phi_j(t).

    Returns
    -------
    dict mapping feature name -> ndarray of shape (T,)
    """
    return {
        'X1': (x_star[0] - MU) * phi1(t),
        'X2': (x_star[1] - MU) * phi2(t),
        'X3': (x_star[2] - MU) * phi3(t),
    }


def sensitivity_pure_effect(fn):
    """
    Sensitivity game: first-order Möbius coefficient for feature j.

    For an additive model with independent features:
        m_{j}(t) = Var(Xj) * phi_j(t)^2

    This equals the marginal variance contribution of Xj to the total
    trajectory variance at each time t. Under the constant kernel and
    normalization, this recovers the classical Sobol index.

    Parameters
    ----------
    fn : str — feature name ('X1', 'X2', or 'X3')

    Returns
    -------
    ndarray of shape (T,)
    """
    return VAR_X * PHI[fn](t)**2


def risk_pure_effect(fn):
    """
    Risk / MSE game: first-order Möbius coefficient for feature j.

    For the MSE game, knowing feature j reduces prediction error in
    proportion to its variance contribution:
        m_{j}(t) = -Var(Xj) * phi_j(t)^2

    The negative sign reflects that features reduce (rather than increase)
    the conditional MSE. Rankings by absolute magnitude match the
    sensitivity game exactly; only the sign differs.

    Parameters
    ----------
    fn : str — feature name

    Returns
    -------
    ndarray of shape (T,)
    """
    return -VAR_X * PHI[fn](t)**2

# ===========================================================================
# Kernel constructors
# ===========================================================================

def kernel_identity(t):
    """Identity kernel: K(t,s) = delta(t-s). No temporal coupling."""
    return np.eye(len(t))


def kernel_ou(t, ell=4.0):
    """
    OU (exponential) kernel: K(t,s) = exp(-|t-s| / ell).
    Symmetric local smoothing with correlation length ell.
    """
    return np.exp(-np.abs(t[:,None] - t[None,:]) / ell)


def kernel_correlation(t):
    """
    Data-adaptive correlation kernel derived from the ICU model's output covariance.

    Constructs C(t,s) = Var(X) * sum_j phi_j(t)*phi_j(s), then normalises
    to a correlation matrix. Ties together times that co-vary across patients
    according to the model structure, without a bandwidth parameter.
    """
    C   = VAR_X * (np.outer(phi1(t), phi1(t)) +
                   np.outer(phi2(t), phi2(t)) +
                   np.outer(phi3(t), phi3(t)))
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)   # avoid division by zero at tails
    return C / np.outer(std, std)

# ===========================================================================
# Kernel application and aggregation
# ===========================================================================

def apply_kernel_rowwise(effect, K):
    """
    Apply kernel K to a functional effect via row-normalised convolution.

        (Kf)(t) = [sum_s K(t,s) * f(s) * dt] / [sum_s K(t,s) * dt]

    Row normalisation makes attribution magnitudes comparable across kernels
    with different total mass. For the identity kernel this is a no-op.

    Parameters
    ----------
    effect : ndarray of shape (T,)
    K      : ndarray of shape (T, T)

    Returns
    -------
    ke : ndarray of shape (T,)
    """
    if np.allclose(K, np.eye(K.shape[0]), atol=1e-10):
        return effect.copy()
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return (K / rs) @ effect * dt


def time_agg_pred(effect, K):
    """
    Compute scalar time-aggregated importance: Phi_S = integral (Kf_S)(t) dt.

    Uses the trapezoidal rule (or a simple sum for the identity kernel).

    Parameters
    ----------
    effect : ndarray of shape (T,)
    K      : ndarray of shape (T, T)

    Returns
    -------
    float
    """
    ke = apply_kernel_rowwise(effect, K)
    if np.allclose(K, np.eye(K.shape[0]), atol=1e-10):
        return float(np.sum(ke) * dt)
    return float(np.trapezoid(ke, dx=dt))


def time_agg_sensitivity(fn, K):
    """Time-aggregated importance for the sensitivity game."""
    return time_agg_pred(sensitivity_pure_effect(fn), K)


def time_agg_risk(fn, K):
    """Time-aggregated importance for the risk / MSE game."""
    return time_agg_pred(risk_pure_effect(fn), K)

# ===========================================================================
# Precompute kernel matrices and importance values
# ===========================================================================

K_ID   = kernel_identity(t)
K_OU   = kernel_ou(t, ell=4.0)
K_CORR = kernel_correlation(t)

# Each entry: (display label, kernel matrix, color)
KERNELS = [
    ('Identity',           K_ID,   C_ID),
    ('OU ($\\ell=4$h)',    K_OU,   C_OU),
    ('Correlation',        K_CORR, C_CORR),
]

GAMES = ['Prediction', 'Sensitivity', 'Risk (MSE)']


def get_importances():
    """
    Compute time-aggregated importance for all (game, kernel, feature)
    combinations.

    Returns
    -------
    importances : nested dict [game][kernel_label][feature] -> float
    """
    out = {g: {kl: {} for kl, _, _ in KERNELS} for g in GAMES}
    pred_effects = pure_effects_pred()

    for kl, K, _ in KERNELS:
        for fn in FEAT_NAMES:
            out['Prediction'][kl][fn]  = time_agg_pred(pred_effects[fn], K)
            out['Sensitivity'][kl][fn] = time_agg_sensitivity(fn, K)
            out['Risk (MSE)'][kl][fn]  = time_agg_risk(fn, K)

    return out


def normalise(d):
    """
    Normalise a dict of floats so that absolute values sum to 1.
    Returns zeroes if all values are negligible.
    """
    total = sum(abs(v) for v in d.values())
    if total < 1e-14:
        return {k: 0.0 for k in d}
    return {k: v / total for k, v in d.items()}


def ranking(d):
    """Return feature names sorted by descending absolute importance."""
    return sorted(d.keys(), key=lambda k: -abs(d[k]))

# ===========================================================================
# Plotting helpers
# ===========================================================================

def savefig(fig, name):
    """Save figure as PDF to PLOT_DIR and close it."""
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


def _spine(ax):
    """Remove top and right spines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ===========================================================================
# Main figure
# ===========================================================================

def make_ranking_games_figure():
    """
    Generate fig6_ranking_preservation_games.pdf.

    Two-row layout:
        Row 0 (3 panels): Time-resolved pure effects f_S(t) per game.
                          Illustrates why the effects have the same shape
                          for sensitivity and risk (up to sign).
        Row 1 (3 panels): Time-aggregated normalized importance bars
                          for each (game, kernel) combination, with a
                          ranking preservation status annotation.

    Also prints a ranking table to stdout for quick verification.
    """
    importances = get_importances()

    # ------------------------------------------------------------------
    # Print ranking table to stdout
    # ------------------------------------------------------------------
    print('\nTime-aggregated importance rankings')
    print('=' * 62)
    for game in GAMES:
        print(f'\n  {game}')
        rankings_match = True
        ref_rank = None
        for kl, _, _ in KERNELS:
            imp  = normalise(importances[game][kl])
            rank = ranking(imp)
            s    = '  '.join(f'{f}({imp[f]:.3f})' for f in rank)
            print(f'    {kl:20s}: {s}')
            if ref_rank is None:
                ref_rank = rank
            elif rank != ref_rank:
                rankings_match = False
        status = '✓ PRESERVED' if rankings_match else '✗ CHANGED'
        print(f'    Ranking: {status}')
    print()

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 8.5))
    fig.suptitle(
        'Ranking preservation across kernels — all three games\n'
        r'ICU model: $F(\mathbf{x})(t)=X_1 e^{-0.2t}+'
        r'X_2 e^{-(t-10)^2/2}+X_3 e^{-(t-18)^2/2}$,'
        r'  $\mathbf{x}^*=(0.8,\,0.9,\,0.7)$',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.0, 0.85],
        hspace=0.42, wspace=0.32,
        left=0.07, right=0.96,
        top=0.88, bottom=0.09,
    )

    # Pre-compute per-game pure effect dicts for Row 0
    game_effects = {
        'Prediction':  pure_effects_pred(),
        'Sensitivity': {fn: sensitivity_pure_effect(fn) for fn in FEAT_NAMES},
        'Risk (MSE)':  {fn: risk_pure_effect(fn) for fn in FEAT_NAMES},
    }

    # ------------------------------------------------------------------
    # Row 0: time-resolved pure effects — one panel per game
    # Shows the raw m_{j}(t) curves before kernel application or aggregation.
    # Note: Sensitivity and Risk effects are identical up to sign.
    # ------------------------------------------------------------------
    for col, game in enumerate(GAMES):
        ax  = fig.add_subplot(gs[0, col])
        eff = game_effects[game]

        all_vals = np.concatenate([eff[fn] for fn in FEAT_NAMES])
        ymax = max(all_vals.max(), 0) * 1.28 + 0.01
        ymin = min(all_vals.min(), 0) * 1.28 - 0.01

        for fn in FEAT_NAMES:
            ax.plot(t, eff[fn], color=FEAT_COLORS[fn], lw=2.0,
                    label=fn, alpha=0.92)

        ax.axhline(0, color='gray', lw=0.5, ls=':')
        # Light shading for the two activity windows
        ax.axvspan(8,  12, alpha=0.07, color=C_X2, zorder=0)
        ax.axvspan(16, 20, alpha=0.07, color=C_X3, zorder=0)
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
        ax.set_xticklabels([str(v) for v in range(0, 25, 4)],
                           fontsize=FS_TICK)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=FS_TICK)
        _spine(ax)
        ax.set_xlabel('Time (h)', fontsize=FS_LABEL)
        if col == 0:
            ax.set_ylabel('Time-resolved pure effects', fontsize=FS_LABEL)
        ax.set_title(f'{game} game — pure effects',
                     fontsize=FS_TITLE, fontweight='bold')
        if col == 0:
            ax.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.85)

    # ------------------------------------------------------------------
    # Row 1: time-aggregated bar charts — one panel per game
    # Shows normalized Phi_S (sums to 1) for each kernel as grouped bars.
    # A status annotation confirms whether the ranking is preserved.
    # ------------------------------------------------------------------
    n_k     = len(KERNELS)
    n_f     = len(FEAT_NAMES)
    bar_w   = 0.65 / n_k
    offsets = np.linspace(-(n_k - 1) / 2, (n_k - 1) / 2, n_k) * bar_w
    x_pos   = np.arange(n_f)

    for col, game in enumerate(GAMES):
        ax = fig.add_subplot(gs[1, col])

        for k_idx, (kl, K, kc) in enumerate(KERNELS):
            imp  = importances[game][kl]
            nimp = normalise(imp)
            vals = [nimp[fn] for fn in FEAT_NAMES]
            ax.bar(x_pos + offsets[k_idx], vals,
                   width=bar_w * 0.88, color=kc, alpha=0.85, label=kl)

        # Ranking preservation check: compare all kernels to the first
        ref_rank = ranking(normalise(importances[game][KERNELS[0][0]]))
        all_same = all(
            ranking(normalise(importances[game][kl])) == ref_rank
            for kl, _, _ in KERNELS
        )
        # Green tick = preserved; orange cross = changed
        status_color = '#009E73' if all_same else '#D55E00'
        status_text  = '✓ Ranking preserved' if all_same else '✗ Ranking changed'
        # Position annotation at bottom-right for col 2 (avoids bar overlap)
        ypos   = 0.03 if col == 2 else 0.97
        va_pos = 'bottom' if col == 2 else 'top'
        ax.text(
            0.97, ypos, status_text,
            transform=ax.transAxes, fontsize=FS_ANNOT,
            va=va_pos, ha='right',
            color=status_color, fontweight='bold',
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(FEAT_NAMES, fontsize=FS_LABEL)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        all_bar_vals = [normalise(importances[game][kl])[fn]
                        for kl, _, _ in KERNELS for fn in FEAT_NAMES]
        ylo = min(min(all_bar_vals) * 1.3, -0.05)
        yhi = max(max(all_bar_vals) * 1.3,  0.05)
        ax.set_ylim(ylo, yhi)
        ax.yaxis.grid(True, linestyle=':', alpha=0.4, color='gray')
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=FS_TICK)
        _spine(ax)
        if col == 0:
            ax.set_ylabel(r'Normalized $\Phi_S$ (sums to 1)',
                          fontsize=FS_LABEL)
        ax.set_title(f'{game} game — time-aggregated importance',
                     fontsize=FS_TITLE, fontweight='bold')
        if col == 0:
            ax.legend(fontsize=FS_LEGEND, loc='center right', framealpha=0.85)

    # Row labels on the right edge of the last-column axes
    for ax_last, label in [(fig.axes[2], 'Time-resolved'),
                           (fig.axes[5], 'Time-aggregated')]:
        ax_last.text(1.02, 0.5, label,
                     transform=ax_last.transAxes, fontsize=FS_TITLE,
                     va='center', rotation=270, color='gray')

    savefig(fig, 'fig6_ranking_preservation_games.pdf')


# ===========================================================================
# Main entry point
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  Ranking Preservation Across Games')
    print('=' * 60)
    make_ranking_games_figure()
    print(f'\nFigure saved to {PLOT_DIR}/')