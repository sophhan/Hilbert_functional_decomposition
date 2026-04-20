"""
Part 3 — Ranking Preservation Across Games and Kernels
=======================================================
Tests whether time-aggregated feature importance rankings are preserved
across kernel choices for all three cooperative game types:

  1. Prediction game:   v(S)(t) = E[F(x)(t) | X_S]
  2. Sensitivity game:  v(S)(t,s) = Cov(F_S(X)(t), F_S(X)(s))
  3. Risk/MSE game:     v(S)(t) = E[(Y(t) - F(x)(t))^2 | X_S]

ICU model (additive, no interaction):
  F(x)(t) = x1*phi1(t) + x2*phi2(t) + x3*phi3(t)
  phi1(t) = exp(-0.2*t)
  phi2(t) = exp(-(t-10)^2 / 2)
  phi3(t) = exp(-(t-18)^2 / 2)
  Xi ~ Uniform[0,1], x* = (0.8, 0.9, 0.7)

For each game:
  - Compute time-aggregated importance Phi_S = int (K f_S)(t) dt
    under Identity, OU, and Correlation kernels
  - Report feature ranking and whether it is preserved

Output: one figure with 3 rows (games) x 3 columns (kernels)
        plus a summary bar chart comparing aggregated importance
        across all games and kernels.

Usage:
  python part3_ranking_games.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

PLOT_DIR = os.path.join(
    'plots', 'synthetic_experiments', 'part_3',
)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colors — Wong (2011) CVD-safe palette
# ---------------------------------------------------------------------------

C_X1 = '#0072B2'   # blue
C_X2 = '#D55E00'   # vermillion
C_X3 = '#CC79A7'   # reddish purple

C_ID   = '#888888'
C_OU   = '#E69F00'   # amber
C_CORR = '#009E73'   # dark-teal

FEAT_COLORS = {'X1': C_X1, 'X2': C_X2, 'X3': C_X3}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

T_MAX, T_POINTS = 24.0, 240
t = np.linspace(0, T_MAX, T_POINTS)
dt = t[1] - t[0]
MU = 0.5
VAR_X = 1.0 / 12.0
X_STAR = np.array([0.8, 0.9, 0.7])

def phi1(t): return np.exp(-0.2 * t)
def phi2(t): return np.exp(-0.5 * (t - 10.0)**2)
def phi3(t): return np.exp(-0.5 * (t - 18.0)**2)

PHI = {'X1': phi1, 'X2': phi2, 'X3': phi3}
FEAT_NAMES = ['X1', 'X2', 'X3']

# Pure effects at x* (prediction game, identity kernel)
def pure_effects_pred(x_star=X_STAR):
    return {
        'X1': (x_star[0] - MU) * phi1(t),
        'X2': (x_star[1] - MU) * phi2(t),
        'X3': (x_star[2] - MU) * phi3(t),
    }

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

def kernel_identity(t):
    return np.eye(len(t))

def kernel_ou(t, ell=4.0):
    return np.exp(-np.abs(t[:,None] - t[None,:]) / ell)

def kernel_correlation(t):
    """Analytical correlation kernel for the ICU model."""
    C = VAR_X * (np.outer(phi1(t), phi1(t)) +
                 np.outer(phi2(t), phi2(t)) +
                 np.outer(phi3(t), phi3(t)))
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    return C / np.outer(std, std)

# ---------------------------------------------------------------------------
# Kernel application — row-normalised (prediction / risk games)
# ---------------------------------------------------------------------------

def apply_kernel_rowwise(effect, K):
    """Row-normalised: (Kf)(t) = [int K(t,s) ds]^{-1} int K(t,s) f(s) ds"""
    if np.allclose(K, np.eye(K.shape[0]), atol=1e-10):
        return effect.copy()
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return (K / rs) @ effect * dt

def time_agg_pred(effect, K):
    """Phi_S = int (K f_S)(t) dt  for prediction / risk game."""
    ke = apply_kernel_rowwise(effect, K)
    if np.allclose(K, np.eye(K.shape[0]), atol=1e-10):
        return float(np.sum(ke) * dt)
    return float(np.trapezoid(ke, dx=dt))

# ---------------------------------------------------------------------------
# Sensitivity game
# ---------------------------------------------------------------------------

def sensitivity_pure_effect(fn):
    """
    First-order variance effect (Mobius coeff for singleton {j}):
      m_{j}(t) = Var(Xj) * phi_j(t)^2
    This is the pure variance contribution of feature j to the
    covariance surface nu_var({j})(t,s) = Var(Xj)*phi_j(t)*phi_j(s).
    """
    return VAR_X * PHI[fn](t)**2

def time_agg_sensitivity(fn, K):
    """
    Under the constant kernel K(t,s)=1:
      Phi_j = int int K(t,s) * Var(Xj)*phi_j(t)*phi_j(s) ds dt
            = Var(Xj) * (int phi_j(t) dt)^2

    Under a general kernel K:
      Phi_j = int (K m_j)(t) dt
    where m_j(t) = Var(Xj)*phi_j(t)^2 is the diagonal of the covariance surface.

    This is the standard approach: apply the kernel to the variance
    effect curve m_j(t) and integrate.
    Note: for the covariance kernel this corresponds to the Shi (2018)
    style aggregation.
    """
    m_j = sensitivity_pure_effect(fn)
    return time_agg_pred(m_j, K)

# ---------------------------------------------------------------------------
# Risk / MSE game
# ---------------------------------------------------------------------------

def risk_pure_effect(fn):
    """
    Correct closed-form Mobius coefficient for the risk game singleton {j}.

    Risk game value function:
      v_risk(S)(t) = E_{X_{-S}}[ (Y(t) - F_S(x*_S, X_{-S})(t))^2 ]

    For the additive model F(x)(t) = sum_j xj*phi_j(t) with Y(t)=F(x*)(t)
    and Xi ~ Uniform[0,1] independently:

      Y(t) - F_S(x*_S, X_{-S})(t) = sum_{j not in S} (x*_j - X_j) phi_j(t)

    Taking expectation over X_{-S}:
      v_risk(S)(t) = sum_{j not in S} Var(Xj) * phi_j(t)^2

    Mobius inversion gives the singleton pure effect:
      m_{j}(t) = v_risk({j})(t) - v_risk({})(t)
               = [sum_{k != j} Var(Xk)*phi_k^2] - [sum_k Var(Xk)*phi_k^2]
               = -Var(Xj) * phi_j(t)^2

    This is always NEGATIVE: fixing X_j reduces the MSE by Var(Xj)*phi_j(t)^2.
    The more negative, the more informative the feature.
    Note: the result does not depend on x* — it measures epistemic uncertainty
    reduction, not the deviation of x* from the mean.
    """
    return -VAR_X * PHI[fn](t)**2

def time_agg_risk(fn, K):
    r = risk_pure_effect(fn)
    return time_agg_pred(r, K)

# ---------------------------------------------------------------------------
# Compute all importances
# ---------------------------------------------------------------------------

K_ID   = kernel_identity(t)
K_OU   = kernel_ou(t, ell=4.0)
K_CORR = kernel_correlation(t)

KERNELS = [
    ('Identity',    K_ID,   C_ID),
    ('OU ($\\ell=4$h)',    K_OU,   C_OU),
    ('Correlation', K_CORR, C_CORR),
]

GAMES = ['Prediction', 'Sensitivity', 'Risk (MSE)']

def get_importances():
    """
    Returns dict:
      importances[game][kernel_label][feat] = float
    """
    out = {g: {kl: {} for kl, _, _ in KERNELS} for g in GAMES}

    pred_effects = pure_effects_pred()

    for kl, K, _ in KERNELS:
        # Prediction
        for fn in FEAT_NAMES:
            out['Prediction'][kl][fn] = time_agg_pred(pred_effects[fn], K)

        # Sensitivity
        for fn in FEAT_NAMES:
            out['Sensitivity'][kl][fn] = time_agg_sensitivity(fn, K)

        # Risk
        for fn in FEAT_NAMES:
            out['Risk (MSE)'][kl][fn] = time_agg_risk(fn, K)

    return out


def normalise(d):
    """Normalise a dict of floats to sum to 1."""
    total = sum(abs(v) for v in d.values())
    if total < 1e-14:
        return {k: 0.0 for k in d}
    return {k: v/total for k, v in d.items()}


def ranking(d):
    """Sort features by absolute importance, descending."""
    return sorted(d.keys(), key=lambda k: -abs(d[k]))

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def savefig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')

def _spine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def make_ranking_games_figure():
    importances = get_importances()

    # -----------------------------------------------------------------------
    # Print ranking table
    # -----------------------------------------------------------------------
    print('\nTime-aggregated importance rankings')
    print('='*62)
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

    # -----------------------------------------------------------------------
    # Figure: 2 rows
    #   Row 0: time-resolved pure effects for each game (3 panels)
    #   Row 1: time-aggregated bar chart for each game (3 panels)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        'Ranking preservation across kernels — all three games\n'
        r'ICU model: $F(\mathbf{x})(t)=X_1 e^{-0.2t}+'
        r'X_2 e^{-(t-10)^2/2}+X_3 e^{-(t-18)^2/2}$,'
        r'  $\mathbf{x}^*=(0.8,\,0.9,\,0.7)$',
        fontsize=10, fontweight='bold',
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.50, wspace=0.32,
        left=0.07, right=0.96,
        top=0.88, bottom=0.08,
    )

    pred_effects = pure_effects_pred()
    sens_effects = {fn: sensitivity_pure_effect(fn) for fn in FEAT_NAMES}
    risk_effects = {fn: risk_pure_effect(fn) for fn in FEAT_NAMES}

    game_effects = {
        'Prediction':  pred_effects,
        'Sensitivity': sens_effects,
        'Risk (MSE)':  risk_effects,
    }
    game_ylabels = {
        'Prediction':  r'$f^{\mathrm{pred}}_j(t) = (x^*_j - \mu)\,\phi_j(t)$',
        'Sensitivity': r'$m_j(t) = \mathrm{Var}(X_j)\,\phi_j(t)^2$',
        'Risk (MSE)':  r'$m^{\mathrm{risk}}_j(t) = -\mathrm{Var}(X_j)\,\phi_j(t)^2$',
    }

    # ------------------------------------------------------------------
    # Row 0: time-resolved effect curves, one panel per game
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
        ax.axvspan(8,  12, alpha=0.07, color=C_X2, zorder=0)
        ax.axvspan(16, 20, alpha=0.07, color=C_X3, zorder=0)
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
        ax.set_xticklabels([str(v) for v in range(0, 25, 4)], fontsize=7)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=8); _spine(ax)
        ax.set_xlabel('Time (h)', fontsize=8)
        ax.set_ylabel(game_ylabels[game], fontsize=7.5)
        ax.set_title(f'{game} game — pure effects',
                     fontsize=9, fontweight='bold')
        if col == 0:
            ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)

    # ------------------------------------------------------------------
    # Row 1: time-aggregated bar chart, one panel per game
    # ------------------------------------------------------------------
    n_k   = len(KERNELS)
    n_f   = len(FEAT_NAMES)
    bar_w = 0.65 / n_k
    offsets = np.linspace(-(n_k-1)/2, (n_k-1)/2, n_k) * bar_w
    x_pos   = np.arange(n_f)

    for col, game in enumerate(GAMES):
        ax = fig.add_subplot(gs[1, col])

        for k_idx, (kl, K, kc) in enumerate(KERNELS):
            imp  = importances[game][kl]
            nimp = normalise(imp)
            vals = [nimp[fn] for fn in FEAT_NAMES]
            ax.bar(x_pos + offsets[k_idx], vals,
                   width=bar_w*0.88, color=kc,
                   alpha=0.85, label=kl)

        # Ranking labels
        ref_rank = ranking(normalise(importances[game][KERNELS[0][0]]))
        all_same = all(
            ranking(normalise(importances[game][kl])) == ref_rank
            for kl, _, _ in KERNELS
        )
        status_color = '#009E73' if all_same else '#D55E00'
        status_text  = '✓ Ranking preserved' if all_same else '✗ Ranking changed'
        ax.text(0.97, 0.97, status_text,
                transform=ax.transAxes, fontsize=8,
                va='top', ha='right', color=status_color,
                fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(FEAT_NAMES, fontsize=9)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        # y-limits: risk game values are negative, others positive
        all_bar_vals = [normalise(importances[game][kl])[fn]
                        for kl, _, _ in KERNELS for fn in FEAT_NAMES]
        ylo = min(min(all_bar_vals)*1.3, -0.05)
        yhi = max(max(all_bar_vals)*1.3,  0.05)
        ax.set_ylim(ylo, yhi)
        ax.yaxis.grid(True, linestyle=':', alpha=0.4, color='gray')
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8); _spine(ax)
        ax.set_ylabel('Normalised $\\Phi_S$ (sums to 1)', fontsize=8)
        ax.set_title(f'{game} game — time-aggregated importance',
                     fontsize=9, fontweight='bold')
        if col == 0:
            ax.legend(fontsize=7.5, loc='center right', framealpha=0.85)

    # Row labels — use the already-created col-2 axes, don't create new ones
    for ax_last, label in [(fig.axes[2], 'Time-resolved'),
                           (fig.axes[5], 'Time-aggregated')]:
        ax_last.text(1.02, 0.5, label,
                     transform=ax_last.transAxes, fontsize=9,
                     va='center', rotation=270, color='gray')

    savefig(fig, 'fig6_ranking_preservation_games.pdf')


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '='*60)
    print('  Part 3 — Ranking Preservation Across Games')
    print('='*60)
    make_ranking_games_figure()
    print(f'\nFigure saved to {PLOT_DIR}/')