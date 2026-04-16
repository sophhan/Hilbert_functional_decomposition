"""
Kernel Narrative — Three Illustrative Examples
================================================
One coherent narrative showing when and why different kernels produce
meaningful results for different explanation goals.

Model (ICU early warning — biomarker trajectory):
  F(x)(t) = x1 * exp(-0.2*t)              baseline recovery (global)
           + x2 * exp(-(t-10)^2 / 2)      early shock event  (localised t~10)
           + x3 * exp(-(t-18)^2 / 2)      late deterioration (localised t~18)

  X = (X1, X2, X3) ~ Uniform[0,1]^3,  T = [0, 24] hours

Three examples, one file:
  Example 1 — OU kernel,          prediction behavior
              "smooths noise, reveals sustained events"
  Example 2 — Correlation kernel, prediction behavior
              "discovers phase structure from population"
  Example 3 — Covariance kernel,  sensitivity/variance behavior
              "emphasises variance in clinically important phases"

Figures:
  fig1_model.pdf              Model, baseline, pure effects
  fig2_ou_prediction.pdf      Example 1: OU vs identity, prediction
  fig3_corr_prediction.pdf    Example 2: Correlation vs identity, prediction
  fig4_cov_sensitivity.pdf    Example 3: Covariance vs constant, sensitivity
  fig5_kernel_comparison.pdf  All kernels side-by-side on X2 effect
  fig6_problems.pdf           Problems: leakage, scale, ℓ-sensitivity, additivity
  fig7_kernel_suitability.pdf When each kernel is appropriate or problematic

Saved to: plots/icu_illustration/
"""

import os
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.linalg import eigh

# ---------------------------------------------------------------------------
# Directories and grid
# ---------------------------------------------------------------------------
PLOT_DIR = os.path.join('plots', 'icu_illustration')
os.makedirs(PLOT_DIR, exist_ok=True)

T_MAX    = 24.0
T_POINTS = 480
t        = np.linspace(0, T_MAX, T_POINTS)
dt       = t[1] - t[0]

MU  = 0.5
VAR = 1 / 12

# Palette
C_X1   = '#c1121f'   # baseline recovery — deep red
C_X2   = '#2a9d8f'   # early shock       — teal
C_X3   = '#e9c46a'   # late deterioration — amber
C_BASE = '#adb5bd'   # baseline           — grey
C_FULL = '#1b2631'   # full model         — near-black
C_ID   = '#888888'   # identity kernel    — mid-grey
C_OU   = '#457b9d'   # OU kernel          — steel blue
C_CORR = '#8338ec'   # correlation kernel — purple
C_COV  = '#f4a261'   # covariance kernel  — orange
C_SENS = '#ee9b00'   # sensitivity game   — gold
C_PROB = '#e63946'   # problem highlight  — red

FEAT_COLORS = {'X1': C_X1, 'X2': C_X2, 'X3': C_X3}
FEAT_LABELS = {
    'X1': r'$X_1$ (baseline recovery)',
    'X2': r'$X_2$ (early shock, $t\approx10$h)',
    'X3': r'$X_3$ (late deterioration, $t\approx18$h)',
}

X_STAR = np.array([0.8, 0.9, 0.7])   # high-risk patient

# ---------------------------------------------------------------------------
# 1.  Model, effects, games  (all analytical)
# ---------------------------------------------------------------------------

def phi1(t): return np.exp(-0.2 * t)
def phi2(t): return np.exp(-0.5 * (t - 10.0)**2)
def phi3(t): return np.exp(-0.5 * (t - 18.0)**2)

def model(x1, x2, x3, t):
    return x1*phi1(t) + x2*phi2(t) + x3*phi3(t)

def baseline_fn(t):
    return MU*(phi1(t) + phi2(t) + phi3(t))

def eff_X1(x1, t): return (x1 - MU) * phi1(t)
def eff_X2(x2, t): return (x2 - MU) * phi2(t)
def eff_X3(x3, t): return (x3 - MU) * phi3(t)

x1s, x2s, x3s = X_STAR
e1 = eff_X1(x1s, t)
e2 = eff_X2(x2s, t)
e3 = eff_X3(x3s, t)
F_full = model(x1s, x2s, x3s, t)
F_base = baseline_fn(t)

# Sensitivity game Möbius values (analytical)
def sens_mobius(t):
    """
    For additive model, Möbius pure variance effects are:
      m_{Xi}(t,s) = VAR * phi_i(t) * phi_i(s)
    After applying kernel K to integrate over s:
      (K m_{Xi})(t) = VAR * phi_i(t) * int K(t,s) phi_i(s) ds
    Returns dict: feature -> (T,) integrated variance effect
    """
    return {
        'X1': VAR * phi1(t)**2,
        'X2': VAR * phi2(t)**2,
        'X3': VAR * phi3(t)**2,
    }

def sobol_analytical(t):
    v1 = VAR * phi1(t)**2
    v2 = VAR * phi2(t)**2
    v3 = VAR * phi3(t)**2
    vF = v1 + v2 + v3
    vF = np.where(vF < 1e-14, 1.0, vF)
    return {'X1': v1/vF, 'X2': v2/vF, 'X3': v3/vF}

# ---------------------------------------------------------------------------
# 2.  Kernel constructors
# ---------------------------------------------------------------------------

def make_identity(t):
    return np.eye(len(t))

def make_constant(t):
    return np.ones((len(t), len(t)))

def make_ou(t, ell=4.0):
    return np.exp(-np.abs(t[:, None] - t[None, :]) / ell)

def make_gaussian(t, sigma=2.0):
    d = t[:, None] - t[None, :]
    return np.exp(-0.5 * (d / sigma)**2)

def make_correlation(t):
    """
    Analytical correlation kernel Corr(F(t), F(s)) under X ~ Uniform[0,1]^3.
    Cov(F(t),F(s)) = VAR * [phi1(t)phi1(s) + phi2(t)phi2(s) + phi3(t)phi3(s)]
    Because the Gaussians are nearly orthogonal (separated by 8h, sigma=1),
    this is nearly block-diagonal: three peaks at (0,0), (10,10), (18,18).
    We normalize to correlation (divide by outer std) — NO further row-norm.
    """
    p1, p2, p3 = phi1(t), phi2(t), phi3(t)
    C = VAR * (np.outer(p1, p1) + np.outer(p2, p2) + np.outer(p3, p3))
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    return C / np.outer(std, std)

def make_covariance(t):
    """
    Raw analytical covariance kernel Cov(F(t), F(s)).
    NOT normalized — preserves magnitude for sensitivity game.
    """
    p1, p2, p3 = phi1(t), phi2(t), phi3(t)
    return VAR * (np.outer(p1, p1) + np.outer(p2, p2) + np.outer(p3, p3))

def kernel_sqrt(K):
    evals, evecs = eigh(K + 1e-8 * np.eye(len(K)))
    evals = np.maximum(evals, 0)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.T

# ---------------------------------------------------------------------------
# 3.  Kernel operators
# ---------------------------------------------------------------------------

def apply_kernel(effect, K, dt):
    """(Kg)(t) = int K(t,s) g(s) ds  — plain integration, no row-norm."""
    return K @ effect * dt

def apply_kernel_to_variance(var_effect, K, dt):
    """
    For sensitivity game: m_S(t,s) = VAR * phi_i(t) * phi_i(s).
    Apply K as: (K m_S)(t) = int K(t,s) * m_S_diag(s) ds
    where m_S_diag(s) = VAR * phi_i(s)^2 is the diagonal of the cov surface.
    This gives: VAR * phi_i(t) * int K(t,s) phi_i(s)^2 ds  [for prediction]
    But for sensitivity we want: int K(t,s) * VAR*phi_i(s)^2 ds
    = K @ (VAR * phi_i^2) * dt   which is just apply_kernel on the diagonal.
    """
    return K @ var_effect * dt

def integrated(effect, K, dt):
    return float(np.sum(apply_kernel(effect, K, dt)) * dt)

# ---------------------------------------------------------------------------
# 4.  Style helpers
# ---------------------------------------------------------------------------

def style_ax(ax, ylabel=None, xlabel='Time (h)', ylim=None, shade_phases=True):
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylim:   ax.set_ylim(*ylim)
    if shade_phases:
        ax.axvspan(8, 12, alpha=0.06, color=C_X2, zorder=0)
        ax.axvspan(16, 20, alpha=0.06, color=C_X3, zorder=0)

def phase_labels(ax, y_frac=0.92):
    ylim = ax.get_ylim()
    y = ylim[0] + y_frac * (ylim[1] - ylim[0])
    ax.text(10, y, 'Phase 1\n[8–12h]', ha='center', fontsize=6.5,
            color=C_X2, style='italic')
    ax.text(18, y, 'Phase 2\n[16–20h]', ha='center', fontsize=6.5,
            color=C_X3, style='italic')

def savefig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')

def kernel_heatmap(ax, K, title, step=None, cmap='YlOrRd'):
    step = step or max(1, T_POINTS // 60)
    im = ax.imshow(K[::step, ::step], aspect='auto', cmap=cmap,
                   extent=[0, 24, 24, 0], vmin=0, vmax=K.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax.set_xlabel('s (h)', fontsize=8)
    ax.set_ylabel('t (h)', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    # Mark phase blocks
    for lo, hi, c in [(8, 12, C_X2), (16, 20, C_X3)]:
        ax.add_patch(mpatches.Rectangle(
            (lo, lo), hi-lo, hi-lo,
            fill=False, edgecolor=c, lw=1.5, ls='--'))

# Pre-compute kernels used across figures
K_id   = make_identity(t)
K_ou   = make_ou(t, ell=4.0)
K_corr = make_correlation(t)
K_cov  = make_covariance(t)
K_const = make_constant(t)

# ===========================================================================
# FIG 1 — Model and pure effects  (narrative setup)
# ===========================================================================
print('\n[Fig 1] Model and pure effects ...')

fig = plt.figure(figsize=(15, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.32,
                        left=0.07, right=0.97, top=0.90, bottom=0.08)
fig.suptitle(
    r'ICU Early Warning: $F(\mathbf{x})(t)='
    r'X_1 e^{-0.2t}+X_2 e^{-(t-10)^2/2}+X_3 e^{-(t-18)^2/2}$'
    r'$\quad\mathbf{x}^*=(0.8,\,0.9,\,0.7)$',
    fontsize=11, fontweight='bold',
)

# Panel A — prediction vs baseline
ax = fig.add_subplot(gs[0, 0])
ax.plot(t, F_full, color=C_FULL, lw=2.5, label=r'$F(\mathbf{x}^*)(t)$')
ax.plot(t, F_base, color=C_BASE, lw=1.8, ls='--',
        label=r'Baseline $\mathbb{E}[F(\mathbf{X})(t)]$')
ax.fill_between(t, F_base, F_full, alpha=0.13, color=C_FULL)
style_ax(ax, ylabel='Biomarker (a.u.)')
ax.legend(fontsize=8)
ax.set_title('(a) Prediction vs baseline', fontsize=10)
phase_labels(ax)

# Panel B — pure effects
ax = fig.add_subplot(gs[0, 1])
ax.plot(t, e1, color=C_X1, lw=2.2, label=FEAT_LABELS['X1'])
ax.plot(t, e2, color=C_X2, lw=2.2, label=FEAT_LABELS['X2'])
ax.plot(t, e3, color=C_X3, lw=2.2, label=FEAT_LABELS['X3'])
style_ax(ax, ylabel='Effect (a.u.)')
ax.legend(fontsize=7.5)
ax.set_title(r'(b) Pure effects $f^H_S(\mathbf{x}^*)(t)$ — Identity kernel', fontsize=10)
phase_labels(ax)

# Panel C — stacked decomposition
ax = fig.add_subplot(gs[0, 2])
s0 = F_base
s1 = F_base + e1
s2 = s1 + e2
s3 = s2 + e3
ax.fill_between(t, 0, s0,  alpha=0.35, color=C_BASE, label='Baseline')
ax.fill_between(t, s0, s1, alpha=0.65, color=C_X1,  label=r'$+f_{X_1}$')
ax.fill_between(t, s1, s2, alpha=0.65, color=C_X2,  label=r'$+f_{X_2}$')
ax.fill_between(t, s2, s3, alpha=0.65, color=C_X3,  label=r'$+f_{X_3}$')
ax.plot(t, F_full, color=C_FULL, lw=2, ls='--',
        label=r'$F(\mathbf{x}^*)$ (exact)')
style_ax(ax, ylabel='Biomarker (a.u.)')
ax.legend(fontsize=7, ncol=2)
ax.set_title('(c) Additive decomposition (exact)', fontsize=10)

# Panel D — Sobol indices
ax = fig.add_subplot(gs[1, 0])
sobol = sobol_analytical(t)
for feat, color in FEAT_COLORS.items():
    ax.plot(t, sobol[feat], color=color, lw=2.2, label=FEAT_LABELS[feat])
ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
style_ax(ax, ylabel=r'Sobol index $S_i(t)$', ylim=(-0.02, 1.12))
ax.legend(fontsize=7.5)
ax.set_title('(d) Time-resolved Sobol indices', fontsize=10)
phase_labels(ax)

# Panel E — covariance kernel heatmap (motivation for Example 2)
ax = fig.add_subplot(gs[1, 1])
kernel_heatmap(ax, K_corr,
    r'(e) Correlation kernel $\mathrm{Corr}(F(t),F(s))$'
    '\nNearly block-diagonal: two physiological phases')

# Panel F — variance contributions
ax = fig.add_subplot(gs[1, 2])
sv = sens_mobius(t)
for feat, color in FEAT_COLORS.items():
    ax.plot(t, sv[feat], color=color, lw=2.2, label=FEAT_LABELS[feat])
ax.fill_between(t, sv['X1'], alpha=0.20, color=C_X1)
ax.fill_between(t, sv['X2'], alpha=0.20, color=C_X2)
ax.fill_between(t, sv['X3'], alpha=0.20, color=C_X3)
style_ax(ax, ylabel=r'$\mathrm{Var}(X_i)\,\phi_i(t)^2$')
ax.legend(fontsize=7.5)
ax.set_title('(f) Variance contributions per feature', fontsize=10)
phase_labels(ax)

savefig(fig, 'fig1_model.pdf')

# ===========================================================================
# FIG 2 — Example 1: OU kernel, prediction behavior
# ===========================================================================
print('[Fig 2] Example 1: OU kernel, prediction ...')

ou_ells   = [1.0, 2.0, 4.0, 8.0]
ou_cmap   = plt.get_cmap('Blues')
ou_colors = [ou_cmap(v) for v in np.linspace(0.35, 0.90, len(ou_ells))]

fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.32,
                        left=0.07, right=0.97, top=0.90, bottom=0.06)
fig.suptitle(
    'Example 1 — OU kernel on prediction behavior\n'
    r'$K(t,s)=e^{-|t-s|/\ell}$: smooths locally, reveals sustained events, '
    r'$\ell$ controls neighbourhood width',
    fontsize=11, fontweight='bold',
)

effects = {'X1': e1, 'X2': e2, 'X3': e3}

# Row 0: effect curves for each feature across OU length-scales
for col, (feat, eff, color) in enumerate([
    ('X1', e1, C_X1), ('X2', e2, C_X2), ('X3', e3, C_X3)
]):
    ax = fig.add_subplot(gs[0, col])
    ax.plot(t, eff, color=C_ID, lw=1.5, ls='--', alpha=0.8,
            label='Identity', zorder=5)
    for ell, lc in zip(ou_ells, ou_colors):
        K = make_ou(t, ell=ell)
        kw = apply_kernel(eff, K, dt)
        ax.plot(t, kw, color=lc, lw=2.0, label=fr'OU $\ell={ell:.0f}$h')
    style_ax(ax, ylabel='Effect (a.u.)' if col == 0 else None)
    ax.set_title(FEAT_LABELS[feat], fontsize=9, color=color, fontweight='bold')
    if col == 0:
        ax.legend(fontsize=7, loc='upper right')

# Row 0, col 3: OU heatmap (ell=4h)
ax = fig.add_subplot(gs[0, 3])
kernel_heatmap(ax, K_ou,
    r'OU kernel $K(t,s)=e^{-|t-s|/4}$' '\nSmooth off-diagonal bands')

# Row 1: identity vs OU side-by-side for all three features at ell=4h
for col, (feat, eff, color) in enumerate([
    ('X1', e1, C_X1), ('X2', e2, C_X2), ('X3', e3, C_X3)
]):
    ax = fig.add_subplot(gs[1, col])
    kw = apply_kernel(eff, K_ou, dt)
    ax.plot(t, eff, color=C_ID, lw=1.5, ls='--', alpha=0.7, label='Identity')
    ax.plot(t, kw,  color=C_OU, lw=2.5, label=r'OU $\ell=4$h')
    ax.fill_between(t, eff, kw, alpha=0.15, color=C_OU)
    style_ax(ax, ylabel='Effect (a.u.)' if col == 0 else None)
    if col == 0:
        ax.legend(fontsize=8)
    ax.set_title(f'Identity vs OU ($\\ell=4$h): {FEAT_LABELS[feat]}',
                 fontsize=8.5)

# Row 1, col 3: key insight annotation
ax = fig.add_subplot(gs[1, 3])
ax.axis('off')
txt = (
    r'$\bf{Key\ insight}$' '\n\n'
    'Identity kernel:\n'
    r'  $X_2$ matters only at $t=10$' '\n\n'
    r'OU kernel ($\ell=4$h):' '\n'
    r'  $X_2$ spreads over $t\in[6,14]$' '\n\n'
    'Interpretation:\n'
    '"This feature drives a\n'
    r' $\it{sustained\ event}$' ' around\n'
    r' $t=10$, not isolated noise"' '\n\n'
    r'$\bf{Problem}$' ':\n'
    r'$\ell$ must be specified' '\n'
    'externally — no automatic\n'
    'choice from data'
)
ax.text(0.05, 0.95, txt, transform=ax.transAxes,
        fontsize=8.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', fc='#f0f4ff', ec=C_OU, lw=1.5))

# Row 2: integrated importance bars for identity vs OU
hours_eval = [0, 6, 10, 14, 18, 24]
for col, (feat, eff, color) in enumerate([
    ('X1', e1, C_X1), ('X2', e2, C_X2), ('X3', e3, C_X3)
]):
    ax = fig.add_subplot(gs[2, col])
    kw_id = eff.copy()
    kw_ou = apply_kernel(eff, K_ou, dt)
    vals_id = np.interp(hours_eval, t, kw_id)
    vals_ou = np.interp(hours_eval, t, kw_ou)
    x_pos = np.arange(len(hours_eval))
    w = 0.35
    ax.bar(x_pos - w/2, vals_id, width=w, color=C_ID, alpha=0.75, label='Identity')
    ax.bar(x_pos + w/2, vals_ou, width=w, color=C_OU, alpha=0.85, label=r'OU $\ell=4$h')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f't={h}' for h in hours_eval], fontsize=7)
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if col == 0:
        ax.set_ylabel('Local explanation $(Kf_S)(t_0)$', fontsize=8)
        ax.legend(fontsize=7)
    ax.set_title(f'Local explanations: {FEAT_LABELS[feat]}', fontsize=8.5)

# Row 2, col 3: integrated scalar comparison
ax = fig.add_subplot(gs[2, 3])
feats = ['X1', 'X2', 'X3']
effs_list = [e1, e2, e3]
int_id = [integrated(e, K_id, dt) for e in effs_list]
int_ou = [integrated(e, K_ou, dt) for e in effs_list]
x_pos = np.arange(3)
w = 0.35
ax.bar(x_pos - w/2, int_id, width=w, color=C_ID, alpha=0.75, label='Identity')
ax.bar(x_pos + w/2, int_ou, width=w, color=C_OU, alpha=0.85, label=r'OU $\ell=4$h')
ax.set_xticks(x_pos)
ax.set_xticklabels([r'$X_1$', r'$X_2$', r'$X_3$'], fontsize=9)
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_ylabel(r'Integrated $\Phi_S$', fontsize=9)
ax.set_title('Integrated importance', fontsize=9)
ax.legend(fontsize=7)

savefig(fig, 'fig2_ou_prediction.pdf')

# ===========================================================================
# FIG 3 — Example 2: Correlation kernel, prediction behavior
# ===========================================================================
print('[Fig 3] Example 2: Correlation kernel, prediction ...')

fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.32,
                        left=0.07, right=0.97, top=0.90, bottom=0.06)
fig.suptitle(
    'Example 2 — Correlation kernel on prediction behavior\n'
    r'$K(t,s)=\mathrm{Corr}(F(t),F(s))$: discovers phase structure from population,'
    ' no $\\ell$ needed',
    fontsize=11, fontweight='bold',
)

# Row 0: correlation kernel heatmap + explanation of what it encodes
ax = fig.add_subplot(gs[0, 0])
kernel_heatmap(ax, K_corr,
    'Correlation kernel\n'
    r'$K(t,s)=\mathrm{Corr}(F(t),F(s))$')

ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
txt = (
    r'$\bf{What\ the\ kernel\ encodes}$' '\n\n'
    'Block structure from population:\n'
    '  • High corr within [8–12h]\n'
    '    (early shock patients)\n'
    '  • High corr within [16–20h]\n'
    '    (late deterioration)\n'
    '  • Low corr between phases\n\n'
    r'$\bf{Key\ formula\ (local\ at\ }t_0\bf{):}$' '\n\n'
    r'$\tilde{f}_{X_2}(t_0)=\int\mathrm{Corr}(F(t_0),F(s))\cdot f_{X_2}(s)\,ds$'
    '\n\n'
    r'"Aggregate $f_{X_2}(s)$ over all $s$' '\n'
    r' that behave like $t_0$ across patients"'
)
ax2.text(0.05, 0.95, txt, transform=ax2.transAxes,
         fontsize=8.5, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.5', fc='#f5f0ff', ec=C_CORR, lw=1.5))

# Row 0, cols 2-3: identity vs correlation for X2 and X3
for col_offset, (feat, eff, color) in enumerate([
    ('X2', e2, C_X2), ('X3', e3, C_X3)
]):
    ax = fig.add_subplot(gs[0, 2 + col_offset])
    kw_id   = eff.copy()
    kw_corr = apply_kernel(eff, K_corr, dt)
    ax.plot(t, kw_id,   color=C_ID,   lw=1.5, ls='--', alpha=0.7, label='Identity')
    ax.plot(t, kw_corr, color=C_CORR, lw=2.5, label='Corr. kernel')
    ax.fill_between(t, kw_id, kw_corr, alpha=0.15, color=C_CORR)
    style_ax(ax)
    ax.set_title(f'Identity vs Corr.: {FEAT_LABELS[feat]}', fontsize=8.5)
    if col_offset == 0:
        ax.legend(fontsize=8)

# Row 1: all three features, identity vs correlation
for col, (feat, eff, color) in enumerate([
    ('X1', e1, C_X1), ('X2', e2, C_X2), ('X3', e3, C_X3)
]):
    ax = fig.add_subplot(gs[1, col])
    kw_id   = eff.copy()
    kw_corr = apply_kernel(eff, K_corr, dt)
    ax.plot(t, kw_id,   color=C_ID,   lw=1.5, ls='--', alpha=0.7, label='Identity')
    ax.plot(t, kw_corr, color=C_CORR, lw=2.5, label='Corr. kernel')
    ax.fill_between(t, kw_id, kw_corr, alpha=0.15, color=C_CORR)
    style_ax(ax, ylabel='Effect (a.u.)' if col == 0 else None)
    ax.set_title(FEAT_LABELS[feat], fontsize=9, color=color, fontweight='bold')
    if col == 0:
        ax.legend(fontsize=8)
    phase_labels(ax)

# Row 1, col 3: OU vs Correlation comparison narrative
ax = fig.add_subplot(gs[1, 3])
ax.axis('off')
txt = (
    r'$\bf{Corr\ vs\ OU\ kernel}$' '\n\n'
    'OU:\n'
    '  Smooths locally (±ℓ)\n'
    '  Treats phases symmetrically\n'
    '  Requires ℓ specification\n\n'
    'Correlation:\n'
    '  Discovers phase boundaries\n'
    '  Respects population structure\n'
    '  Data-driven, no free param\n\n'
    r'$\bf{At\ }t_0=10$:' '\n'
    '  Corr kernel aggregates\n'
    '  over all s∈[8–12h]\n'
    '  (high correlation with t=10)\n'
    '  → "effect on entire phase"'
)
ax.text(0.05, 0.95, txt, transform=ax.transAxes,
        fontsize=8.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', fc='#f5f0ff', ec=C_CORR, lw=1.5))

# Row 2: redistribution (corr - identity) for each feature
annots = [
    r'$X_1$ global: slight redistribution' '\ntoward high-variance phases',
    r'$X_2$ shock: now reflects' '\nentire early phase [8–12h],\nnot just $t=10$',
    r'$X_3$ late: aggregated over' '\nlate phase [16–20h],\nnot just $t=18$',
]
for col, (feat, eff, color, note) in enumerate(zip(
    ['X1','X2','X3'], [e1,e2,e3],
    [C_X1, C_X2, C_X3], annots
)):
    ax = fig.add_subplot(gs[2, col])
    diff = apply_kernel(eff, K_corr, dt) - eff
    ax.fill_between(t, 0, diff, where=diff >= 0,
                    alpha=0.75, color=color,   label='Gained weight')
    ax.fill_between(t, 0, diff, where=diff <  0,
                    alpha=0.75, color=C_PROB, label='Lost weight')
    ax.axhline(0, color='gray', lw=0.9)
    style_ax(ax, ylabel='Redistribution (Corr−Id)' if col==0 else None)
    if col == 0: ax.legend(fontsize=7)
    ax.text(0.97, 0.97, note, transform=ax.transAxes,
            fontsize=7.5, va='top', ha='right', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f8f8f8',
                      ec=color, lw=1.2))

# Row 2, col 3: integrated comparison
ax = fig.add_subplot(gs[2, 3])
int_id   = [integrated(e, K_id,   dt) for e in [e1, e2, e3]]
int_corr = [integrated(e, K_corr, dt) for e in [e1, e2, e3]]
x_pos = np.arange(3)
w = 0.35
ax.bar(x_pos - w/2, int_id,   width=w, color=C_ID,   alpha=0.75, label='Identity')
ax.bar(x_pos + w/2, int_corr, width=w, color=C_CORR, alpha=0.85, label='Corr. kernel')
ax.set_xticks(x_pos)
ax.set_xticklabels([r'$X_1$', r'$X_2$', r'$X_3$'], fontsize=9)
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_ylabel(r'Integrated $\Phi_S$', fontsize=9)
ax.set_title('Integrated importance', fontsize=9)
ax.legend(fontsize=7)

savefig(fig, 'fig3_corr_prediction.pdf')

# ===========================================================================
# FIG 4 — Example 3: Covariance kernel, sensitivity game
# ===========================================================================
print('[Fig 4] Example 3: Covariance kernel, sensitivity ...')

sv = sens_mobius(t)

# Apply kernels to diagonal variance effects
def apply_sens_kernel(feat, K, dt):
    """Apply kernel to the diagonal variance effect VAR * phi_i^2."""
    return apply_kernel(sv[feat], K, dt)

fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.32,
                        left=0.07, right=0.97, top=0.90, bottom=0.06)
fig.suptitle(
    'Example 3 — Covariance kernel on sensitivity/variance behavior\n'
    r'$K(t,s)=\mathrm{Cov}(F(t),F(s))$: weights variance by clinically important phases',
    fontsize=11, fontweight='bold',
)

# Row 0: raw variance effects + covariance kernel heatmap
ax = fig.add_subplot(gs[0, 0])
for feat, color in FEAT_COLORS.items():
    ax.plot(t, sv[feat], color=color, lw=2.2, label=FEAT_LABELS[feat])
    ax.fill_between(t, sv[feat], alpha=0.15, color=color)
style_ax(ax, ylabel=r'$\mathrm{Var}(X_i)\,\phi_i(t)^2$')
ax.legend(fontsize=7.5)
ax.set_title(r'(a) Variance effects $m_S(t)$ — constant kernel (Sobol)', fontsize=9)
phase_labels(ax)

ax2 = fig.add_subplot(gs[0, 1])
kernel_heatmap(ax2, K_cov,
    r'(b) Covariance kernel $\mathrm{Cov}(F(t),F(s))$'
    '\nAmplifies high-variance phases')

ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
txt = (
    r'$\bf{Why\ covariance\ kernel\ here?}$' '\n\n'
    'Sensitivity game output:\n'
    r'  $\nu_{\rm var}(S)(t,s)=\mathrm{Cov}(F^H_S(t),F^H_S(s))$' '\n\n'
    'Applying covariance kernel:\n'
    r'  $\int\mathrm{Cov}(F(t),F(s))\cdot\mathrm{Var}(F^H_S(s))\,ds$'
    '\n\n'
    'Units consistent:\n'
    '  Cov × Var × time → [var²·h]\n\n'
    'Meaning:\n'
    '"Which features drive variance\n'
    ' in the phases that matter\n'
    ' most to the trajectory?"'
)
ax3.text(0.05, 0.95, txt, transform=ax3.transAxes,
         fontsize=8.5, va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.5', fc='#fff8f0', ec=C_COV, lw=1.5))

# Row 0, col 3: Sobol indices (constant kernel on sensitivity)
ax4 = fig.add_subplot(gs[0, 3])
sobol = sobol_analytical(t)
for feat, color in FEAT_COLORS.items():
    ax4.plot(t, sobol[feat], color=color, lw=2, label=FEAT_LABELS[feat])
ax4.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
style_ax(ax4, ylabel=r'Sobol $S_i(t)$', ylim=(-0.02, 1.12))
ax4.set_title('(d) Classical Sobol (constant kernel)', fontsize=9)
ax4.legend(fontsize=7)
phase_labels(ax4)

# Row 1: constant kernel (classical Sobol) vs covariance kernel
for col, (feat, color) in enumerate(FEAT_COLORS.items()):
    ax = fig.add_subplot(gs[1, col])
    sv_const = apply_sens_kernel(feat, K_const, dt)
    sv_cov   = apply_sens_kernel(feat, K_cov,   dt)
    ax.plot(t, sv_const, color=C_ID,  lw=1.8, ls='--', alpha=0.8,
            label='Constant (Sobol)')
    ax.plot(t, sv_cov,   color=C_COV, lw=2.5, label='Cov. kernel')
    ax.fill_between(t, sv_const, sv_cov, alpha=0.18, color=C_COV)
    style_ax(ax, ylabel='Variance effect' if col == 0 else None)
    ax.set_title(FEAT_LABELS[feat], fontsize=9, color=color, fontweight='bold')
    if col == 0: ax.legend(fontsize=8)
    phase_labels(ax)

# Row 1, col 3: narrative
ax = fig.add_subplot(gs[1, 3])
ax.axis('off')
txt = (
    r'$\bf{Constant\ vs\ Cov\ kernel}$' '\n\n'
    'Constant kernel (Sobol):\n'
    '  Equal weight to all times\n'
    r'  $X_1$ wins — active everywhere' '\n\n'
    'Covariance kernel:\n'
    '  Amplifies high-cov phases\n'
    r'  $X_2$, $X_3$ win — drive var' '\n'
    '  in the structured phases\n\n'
    'Clinical interpretation:\n'
    r'  $X_1$ has global but low-' '\n'
    '  information effect\n'
    r'  $X_2$/$X_3$ drive variance' '\n'
    '  exactly where it is most\n'
    '  prognostically relevant'
)
ax.text(0.05, 0.95, txt, transform=ax.transAxes,
        fontsize=8.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', fc='#fff8f0', ec=C_COV, lw=1.5))

# Row 2: integrated importance — constant vs covariance kernel
feats = ['X1', 'X2', 'X3']
for col, (feat, color) in enumerate(FEAT_COLORS.items()):
    ax = fig.add_subplot(gs[2, col])
    diff = apply_sens_kernel(feat, K_cov, dt) - apply_sens_kernel(feat, K_const, dt)
    ax.fill_between(t, 0, diff, where=diff >= 0,
                    alpha=0.75, color=color,  label='Gained weight')
    ax.fill_between(t, 0, diff, where=diff <  0,
                    alpha=0.75, color=C_PROB, label='Lost weight')
    ax.axhline(0, color='gray', lw=0.9)
    style_ax(ax, ylabel='Redistribution (Cov−Const)' if col==0 else None)
    if col == 0: ax.legend(fontsize=7)
    phase_labels(ax)

ax = fig.add_subplot(gs[2, 3])
int_const = [float(np.sum(apply_sens_kernel(f, K_const, dt))*dt) for f in feats]
int_cov   = [float(np.sum(apply_sens_kernel(f, K_cov,   dt))*dt) for f in feats]
# Normalise for fair comparison
def norm(v): mx = max(abs(x) for x in v); return [x/mx for x in v]
int_const_n = norm(int_const)
int_cov_n   = norm(int_cov)
x_pos = np.arange(3)
w = 0.35
ax.bar(x_pos - w/2, int_const_n, width=w, color=C_ID,  alpha=0.75, label='Constant (Sobol)')
ax.bar(x_pos + w/2, int_cov_n,   width=w, color=C_COV, alpha=0.85, label='Cov. kernel')
ax.set_xticks(x_pos)
ax.set_xticklabels([r'$X_1$', r'$X_2$', r'$X_3$'], fontsize=9)
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_ylabel('Integrated (normalised)', fontsize=8)
ax.set_title('Integrated importance\n(normalised for comparison)', fontsize=8.5)
ax.legend(fontsize=7)

savefig(fig, 'fig4_cov_sensitivity.pdf')

# ===========================================================================
# FIG 5 — All kernels side-by-side on X2 effect (comprehensive comparison)
# ===========================================================================
print('[Fig 5] All kernels comparison ...')

kernels_compare = [
    ('Identity',          K_id,    C_ID,   'Pointwise, no aggregation'),
    (r'OU $\ell=2$h',     make_ou(t, ell=2), '#9dc4e0', 'Local ±2h smoothing'),
    (r'OU $\ell=4$h',     K_ou,    C_OU,   'Local ±4h smoothing'),
    (r'OU $\ell=8$h',     make_ou(t, ell=8), '#1a5276', 'Local ±8h smoothing'),
    ('Correlation',       K_corr,  C_CORR, 'Phase-aware (data-driven)'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    r'Comprehensive kernel comparison on $X_2$ (early shock) and $X_3$ (late deterioration)'
    '\nPrediction behavior — each kernel answers a different question',
    fontsize=11, fontweight='bold',
)

# Row 0: X2 effect under all kernels
ax = axes[0, 0]
for k_name, K, kc, desc in kernels_compare:
    kw = apply_kernel(e2, K, dt)
    ax.plot(t, kw, color=kc, lw=2.0, label=k_name)
style_ax(ax, ylabel='Effect (a.u.)')
ax.set_title(r'$X_2$ (early shock): all kernels', fontsize=10)
ax.legend(fontsize=7.5)
phase_labels(ax)

# Row 0: X3 effect under all kernels
ax = axes[0, 1]
for k_name, K, kc, desc in kernels_compare:
    kw = apply_kernel(e3, K, dt)
    ax.plot(t, kw, color=kc, lw=2.0, label=k_name)
style_ax(ax)
ax.set_title(r'$X_3$ (late deterioration): all kernels', fontsize=10)
ax.legend(fontsize=7.5)
phase_labels(ax)

# Row 0: X1 effect under all kernels
ax = axes[0, 2]
for k_name, K, kc, desc in kernels_compare:
    kw = apply_kernel(e1, K, dt)
    ax.plot(t, kw, color=kc, lw=2.0, label=k_name)
style_ax(ax)
ax.set_title(r'$X_1$ (baseline recovery): all kernels', fontsize=10)
ax.legend(fontsize=7.5)
phase_labels(ax)

# Row 1: Integrated importance bars for all kernels
ax = axes[1, 0]
x_pos = np.arange(3)
bar_width = 0.15
for k_idx, (k_name, K, kc, desc) in enumerate(kernels_compare):
    vals = [integrated(e, K, dt) for e in [e1, e2, e3]]
    ax.bar(x_pos + (k_idx - 2)*bar_width, vals, width=bar_width,
           color=kc, alpha=0.85, label=k_name)
ax.set_xticks(x_pos)
ax.set_xticklabels([r'$X_1$', r'$X_2$', r'$X_3$'], fontsize=9)
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_ylabel(r'Integrated $\Phi_S$', fontsize=9)
ax.set_title('Integrated importance by kernel', fontsize=10)
ax.legend(fontsize=7)

# Row 1: Sensitivity game — constant vs covariance
ax = axes[1, 1]
for k_name, K, kc, desc in [
    ('Constant (Sobol)', K_const, C_ID,  ''),
    ('Cov. kernel',      K_cov,   C_COV, ''),
]:
    for feat, color, ls in [('X1',C_X1,'-'),('X2',C_X2,'--'),('X3',C_X3,':')]:
        sv_k = apply_sens_kernel(feat, K, dt)
        ax.plot(t, sv_k, color=color, lw=2, ls=ls,
                label=f'{k_name}: {feat}' if feat=='X2' else None)
ax.set_title('Sensitivity game:\nConstant vs Covariance kernel', fontsize=9)
style_ax(ax, ylabel='Variance effect')
phase_labels(ax)

# Row 1: Summary table as text panel
ax = axes[1, 2]
ax.axis('off')
table_txt = (
    r'$\bf{Summary:\ which\ kernel\ for\ which\ goal?}$' '\n\n'
    'Identity:\n'
    '  ✓ Pointwise attribution\n'
    '  ✗ Cannot detect sustained events\n\n'
    r'OU ($\ell$ chosen):' '\n'
    '  ✓ Smooth, noise-robust\n'
    '  ✗ ℓ must be externally set\n\n'
    'Correlation (prediction):\n'
    '  ✓ Phase-aware, data-driven\n'
    '  ✗ Leakage if non-orthogonal basis\n'
    '  ✗ Requires population estimate\n\n'
    'Covariance (sensitivity):\n'
    '  ✓ Dimensionally consistent\n'
    '  ✓ Links to Shi et al. (2018)\n'
    '  ✗ No pointwise additivity\n'
    '  ✗ Normalisation instability'
)
ax.text(0.03, 0.97, table_txt, transform=ax.transAxes,
        fontsize=8.5, va='top', ha='left', family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', fc='#f9f9f9', ec='#cccccc', lw=1.2))

plt.tight_layout()
savefig(fig, 'fig5_kernel_comparison.pdf')

# ===========================================================================
# FIG 6 — Problems panel
# ===========================================================================
print('[Fig 6] Problems panel ...')

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.35,
                        left=0.06, right=0.97, top=0.90, bottom=0.07)
fig.suptitle(
    'Known problems with each kernel choice\n'
    'Every kernel answers a different question — and each has a cost',
    fontsize=11, fontweight='bold',
)

# Problem 1: ℓ sensitivity (OU)
ax = fig.add_subplot(gs[0, 0])
for ell, color in zip([0.5, 2, 4, 8, 16],
                      plt.get_cmap('Blues')(np.linspace(0.2, 0.9, 5))):
    K = make_ou(t, ell=ell)
    kw = apply_kernel(e2, K, dt)
    ax.plot(t, kw, color=color, lw=2, label=fr'$\ell={ell}$h')
ax.plot(t, e2, color=C_PROB, lw=1.5, ls='--', label='Identity')
style_ax(ax, ylabel='Effect (a.u.)')
ax.set_title(r'P1: OU — $\ell$ sensitivity on $f_{X_2}$'
             '\nSmall ℓ→identity, large ℓ→over-smooth', fontsize=9)
ax.legend(fontsize=7, ncol=2)
phase_labels(ax)

# Problem 2: leakage — what happens with non-orthogonal basis (modified model)
ax = fig.add_subplot(gs[0, 1])
# Add a sine component to create leakage
t_sin = np.sin(np.pi * t / 24)
C_leak = VAR * (np.outer(phi1(t), phi1(t)) +
                np.outer(phi2(t), phi2(t)) +
                np.outer(phi3(t), phi3(t)) +
                np.outer(t_sin, t_sin))   # sine overlaps with Gaussians
std_l = np.sqrt(np.diag(C_leak))
std_l = np.where(std_l < 1e-12, 1.0, std_l)
K_leak = C_leak / np.outer(std_l, std_l)
kw_clean = apply_kernel(e2, K_corr, dt)
kw_leak  = apply_kernel(e2, K_leak,  dt)
ax.plot(t, e2,       color=C_ID,   lw=1.5, ls='--', alpha=0.7, label='Identity')
ax.plot(t, kw_clean, color=C_CORR, lw=2.2, label='Corr. (orthogonal)')
ax.plot(t, kw_leak,  color=C_PROB, lw=2.2, ls='-.', label='Corr. (+ sine, leakage)')
style_ax(ax)
ax.set_title(r'P2: Correlation — leakage with non-orthogonal basis'
             '\nSine component leaks into $f_{X_2}$ attribution', fontsize=9)
ax.legend(fontsize=7.5)

# Problem 3: dimensional incoherence — covariance kernel on prediction behavior
# The covariance kernel has units of variance (units^2), the effect has units
# of prediction (units). The product is dimensionally incoherent and the
# numerical scale is model-dependent — could be larger or smaller than identity.
# The correlation kernel is dimensionless and avoids this problem.
ax = fig.add_subplot(gs[0, 2])
kw_pred_cov  = apply_kernel(e2, K_cov,  dt)
kw_pred_corr = apply_kernel(e2, K_corr, dt)

ax.plot(t, e2,            color=C_ID,   lw=1.5, ls='--', alpha=0.8,
        label=r'Identity $\|f_{X_2}\|_\infty$='f'{np.abs(e2).max():.3f}')
ax.plot(t, kw_pred_corr,  color=C_CORR, lw=2.2,
        label=r'Corr. kernel $\|\cdot\|_\infty$='
              f'{np.abs(kw_pred_corr).max():.3f}')
ax.plot(t, kw_pred_cov,   color=C_PROB, lw=2.2, ls='-.',
        label=r'Cov. kernel $\|\cdot\|_\infty$='
              f'{np.abs(kw_pred_cov).max():.3f}')
style_ax(ax, ylabel='Effect (a.u.)')
ax.legend(fontsize=7)
ax.set_title('P3: Covariance kernel on prediction behavior\n'
             r'Units: Cov$\times$effect$\times$time — dimensionally incoherent',
             fontsize=9)

# Problem 4: no pointwise additivity — Remark 1 from the paper
# The correct demonstration: E_t(F) != sum_S E_{S,t}
# where E_{S,t}(x) = f_S(t) * (K f_S)(t)  (time-resolved quadratic effect)
# Expanding: F(t)*(KF)(t) = sum_S f_S(t)*(Kf_S)(t)
#                         + sum_{S!=L} f_S(t)*(Kf_L)(t)
# The cross terms f_S(t)*(Kf_L)(t) are the additivity gap.
ax = fig.add_subplot(gs[0, 3])

# Time-resolved total quadratic effect
E_total = F_full * apply_kernel(F_full - F_base, K_cov, dt)

# Sum of individual time-resolved effects
E_x1 = e1 * apply_kernel(e1, K_cov, dt)
E_x2 = e2 * apply_kernel(e2, K_cov, dt)
E_x3 = e3 * apply_kernel(e3, K_cov, dt)
E_sum = E_x1 + E_x2 + E_x3

# Cross terms = the gap (Remark 1 in the paper)
cross = E_total - E_sum

ax.plot(t, E_total, color=C_FULL, lw=2.2,
        label=r'$F(t)\cdot(KF)(t)$ — total')
ax.plot(t, E_sum,   color=C_COV,  lw=2.2, ls='--',
        label=r'$\sum_i f_{X_i}(t)\cdot(Kf_{X_i})(t)$')
ax.fill_between(t, E_sum, E_total, alpha=0.35, color=C_PROB,
                label='Cross terms (gap)')
ax.axhline(0, color='gray', lw=0.6, ls=':')
style_ax(ax, ylabel='Quadratic effect (a.u.)', shade_phases=True)
ax.set_title('P4: No pointwise additivity (Remark 1)\n'
             r'Cross terms $f_S(t)(Kf_L)(t)\neq 0$ fill the gap',
             fontsize=9)
ax.legend(fontsize=7)
phase_labels(ax)

# Row 1: remedies and guidance
remedies = [
    ('P1: OU — choosing length-scale',
     r'$\ell$ is a modelling choice encoding' '\n'
     r'prior knowledge about the output' '\n'
     r'domain — not learnable from the' '\n'
     r'explanation task itself.' '\n\n'
     r'Principled options:' '\n'
     r'  • Domain knowledge: $\ell$ = typical' '\n'
     r'    physiological decorrelation lag' '\n'
     r'    (e.g. 4h for ICU vital signs)' '\n\n'
     r'  • Sensitivity analysis: show' '\n'
     r'    explanations across a range of' '\n'
     r'    $\ell$ values (as in row 0 above)' '\n'
     r'    and report the stable conclusions' '\n\n'
     r'  • Fit $\ell$ to empirical autocorr.' '\n'
     r'    of observed outcome trajectories',
     C_OU),
    ('P2: Correlation — leakage in practice',
     r'Leakage occurs whenever basis' '\n'
     r'functions $\phi_i$, $\phi_j$ overlap' '\n'
     r'in $L^2(\mathcal{T})$. For black-box' '\n'
     r'ML models the basis functions are' '\n'
     r'unknown and leakage cannot be' '\n'
     r'verified or controlled.' '\n\n'
     r'Remedy:' '\n'
     r'  Use an externally specified kernel' '\n'
     r'  (OU, Matern, periodic) whose' '\n'
     r'  eigenvectors are independent of' '\n'
     r'  the model structure — leakage' '\n'
     r'  cannot occur by construction.' '\n\n'
     r'  Reserve model-derived correlation' '\n'
     r'  for sensitivity game where it is' '\n'
     r'  theoretically grounded (Thm. 2).',
     C_CORR),
    ('P3 Remedy: use correlation not covariance',
     r'The covariance kernel has units of' '\n'
     r'variance (units$^2$); the effect has' '\n'
     r'units of prediction (units). Their' '\n'
     r'product is dimensionally incoherent' '\n'
     r'for prediction behavior — the' '\n'
     r'numerical scale is model-dependent' '\n'
     r'and uninterpretable.' '\n\n'
     r'Remedy: normalise to' '\n'
     r'$K(t,s)=\mathrm{Corr}(F(t),F(s))$' '\n'
     r'(dimensionless, no row-norm).' '\n\n'
     r'Reserve raw covariance kernel for' '\n'
     r'sensitivity/variance behavior where' '\n'
     r'cov $\times$ cov is consistent.',
     C_COV),
    ('P4 Remedy: integrate before reporting',
     r'Cross terms vanish only after' '\n'
     r'integrating over the full domain:' '\n\n'
     r'$\int_T F(t)(KF)(t)\,dt$' '\n'
     r'$= \sum_S \int_T f_S(t)(Kf_S)(t)\,dt$' '\n\n'
     r'(Lemma 1: additivity holds for' '\n'
     r'integrated effects, not pointwise).' '\n\n'
     r'Report integrated $\Phi_S$, not' '\n'
     r'pointwise $f_S(t)(Kf_S)(t)$ when' '\n'
     r'using a non-identity kernel for' '\n'
     r'the sensitivity game.',
     C_SENS),
]
for col, (title, body, color) in enumerate(remedies):
    ax = fig.add_subplot(gs[1, col])
    ax.axis('off')
    ax.text(0.5, 0.98, title, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='center', color=color)
    ax.text(0.05, 0.82, body, transform=ax.transAxes,
            fontsize=8.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', fc='#fafafa',
                      ec=color, lw=1.5))

savefig(fig, 'fig6_problems.pdf')

# ===========================================================================
# FIG 7 — Kernel suitability: when each kernel is meaningful or problematic
# ===========================================================================
print('[Fig 7] Kernel suitability ...')

# ---------------------------------------------------------------------------
# Additional kernel constructors needed for this figure
# ---------------------------------------------------------------------------

def make_matern32(t, ell=4.0):
    """Matern-3/2: once-differentiable, intermediate between OU and Gaussian."""
    adiff = np.abs(t[:, None] - t[None, :])
    r = np.sqrt(3) * adiff / ell
    return (1 + r) * np.exp(-r)

def make_gaussian_k(t, sigma=2.0):
    """Gaussian kernel: infinitely differentiable, very smooth."""
    d = t[:, None] - t[None, :]
    return np.exp(-0.5 * (d / sigma)**2)

def make_periodic(t, period=24.0, length_scale=4.0):
    """Periodic kernel: encodes exact periodicity with given period."""
    adiff = np.abs(t[:, None] - t[None, :])
    return np.exp(-2 * np.sin(np.pi * adiff / period)**2 / length_scale**2)

def make_constant(t):
    return np.ones((len(t), len(t)))

def make_ar(t, rho=0.85):
    """
    AR kernel: K(t,s) = rho^|t-s|.
    WARNING: rho is per time-unit (hours), not per time-step.
    rho=0.85/h means correlation at 4h lag = 0.85^4 = 0.52.
    Do NOT confuse with rho-per-time-step which gives near-identity behaviour.
    """
    adiff = np.abs(t[:, None] - t[None, :])
    return rho ** adiff

def make_ar_naive(t, rho_per_step=0.85):
    """
    Naive AR: rho applied per time-step (not per hour).
    With dt=0.05h, rho^(1/dt) = 0.85^20 ≈ 0.04/h — near-identity in practice.
    This illustrates the time-step reparametrisation pitfall.
    """
    adiff = np.abs(t[:, None] - t[None, :])
    # steps = adiff / dt, so K(t,s) = rho^(|t-s|/dt)
    return rho_per_step ** (adiff / dt)

# Pre-compute all kernels
K_ou4      = make_ou(t, ell=4.0)
K_matern   = make_matern32(t, ell=4.0)
K_gauss2   = make_gaussian_k(t, sigma=2.0)
K_periodic = make_periodic(t, period=24.0, length_scale=4.0)
K_constant = make_constant(t)
K_ar_good  = make_ar(t, rho=0.85)          # rho per hour — meaningful
K_ar_naive = make_ar_naive(t, rho_per_step=0.85)  # rho per step — pitfall
K_id_      = make_identity(t)

# ---------------------------------------------------------------------------
# Figure layout: 3 rows x 4 cols
#   Row 0: "Good choices" — OU, Matern, Gaussian, Correlation
#           applied to f_{X2}, showing localisation differences
#   Row 1: "Poor choices" — Constant, Periodic, AR-naive, AR-good
#           showing why each is inappropriate or needs care
#   Row 2: Integrated importance bars — all kernels, all features
#           showing how kernel choice changes the importance ranking
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(18, 15))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.72, wspace=0.32,
                        left=0.07, right=0.97, top=0.93, bottom=0.06)
fig.suptitle(
    'Kernel suitability for the ICU early warning example\n'
    r'Prediction behavior — $f_{X_2}(\mathbf{x}^*)(t)$ = early shock effect'
    '  (and integrated importance for all features)',
    fontsize=11, fontweight='bold',
)

# ---- Row 0: Good choices ---------------------------------------------------
good_kernels = [
    (K_ou4,    C_OU,    r'OU ($\ell=4$h)',
     'Continuous but non-differentiable.\n'
     'Good for sharp clinical events:\n'
     'attribution can change abruptly.\n'
     r'Correlates $t=10$ with $[6,14]$h.'),
    (K_matern, '#6a4c93', r'Matern-3/2 ($\ell=4$h)',
     'Once differentiable — smooth\n'
     'transitions between time points.\n'
     'Good when attribution curves\n'
     'should be physiologically smooth.'),
    (K_gauss2, '#219ebc', r'Gaussian ($\sigma=2$h)',
     'Infinitely differentiable.\n'
     'Narrower window than OU at same\n'
     r'$\ell$ — concentrates attribution\n'
     'more tightly around the event.'),
    (K_corr,   C_CORR,  'Correlation',
     'Data-driven phase structure.\n'
     'No length-scale to choose.\n'
     'Couples t=10 only with times\n'
     'that co-vary across patients.\n'
     'Best when population data\n'
     'available and basis orthogonal.'),
]

for col, (K, color, name, note) in enumerate(good_kernels):
    ax = fig.add_subplot(gs[0, col])
    ax.plot(t, e2, color=C_ID, lw=1.4, ls='--', alpha=0.7,
            label='Identity', zorder=5)
    kw = apply_kernel(e2, K, dt)
    ax.plot(t, kw, color=color, lw=2.4, label=name)
    ax.fill_between(t, e2, kw, alpha=0.13, color=color)
    style_ax(ax, ylabel=r'$(Kf_{X_2})(t)$' if col == 0 else None)
    ax.set_title(f'✓  {name}', fontsize=9, fontweight='bold', color=color)
    # Expand y-axis top by 55% to carve out annotation space above curves
    ylo, yhi = ax.get_ylim()
    yrange = yhi - ylo
    ax.set_ylim(ylo, yhi + 0.55 * yrange)
    # Legend top-left (curves start from zero there for all good kernels)
    ax.legend(fontsize=7, loc='upper left')
    phase_labels(ax, y_frac=0.96)
    # Annotation upper-right — now in the expanded empty space above curves
    ax.text(0.97, 0.97, note, transform=ax.transAxes,
            fontsize=7, va='top', ha='right', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f0fff4',
                      ec=color, lw=1.2, alpha=0.95))

# ---- Row 1: Poor choices ---------------------------------------------------
poor_kernels = [
    (K_constant, '#888888', 'Constant',
     '✗  Wrong for localised events\n\n'
     'Averages $f_{X_2}$ over entire day.\n'
     'Shock at $t=10$ contributes only\n'
     '~1/24 of its peak to each $t$\n'
     '→ nearly zero everywhere.\n'
     'Destroys the localisation that\n'
     'makes the event meaningful.'),
    (K_periodic, '#e07b39', r'Periodic ($p=24$h)',
     '✗  Wrong conceptual assumption\n\n'
     'Implies trajectory repeats every\n'
     '24h — couples $t=10$h with\n'
     '$t=10$h on the "next day",\n'
     'which does not exist in a\n'
     'single ICU stay.\n'
     'Use only for multi-day data\n'
     'with genuine periodicity.'),
    (K_ar_naive, '#c77dff', r'AR $\rho{=}0.85$/step',
     '✗  Time-step reparametrisation\n\n'
     r'$\rho{=}0.85$ per step, $dt{=}0.05$h:'
     '\ncorr. at 1h lag $= 0.85^{20}$'
     r'$\approx 0.04$'
     '\n→ effectively identity kernel.\n'
     r'Must use $\rho$ per hour, not'
     '\nper step: $K(t,s)=\\rho^{|t-s|}$\n'
     'with hours as the unit.'),
    (K_ar_good,  '#5e60ce', r'AR $\rho{=}0.85$/hour',
     '✓  Correctly parametrised AR\n\n'
     'Corr. at 4h lag: $0.85^4{\\approx}0.52$.\n'
     'Equivalent to OU with\n'
     r'$\ell{=}{-}1/\log(0.85){\approx}6.1$h.'
     '\nProduces broader smoothing\n'
     'than OU $\\ell{=}4$h.\n'
     'Natural for discrete-time\n'
     'AR(1) process outputs.'),
]

for col, (K, color, name, note) in enumerate(poor_kernels):
    ax = fig.add_subplot(gs[1, col])
    ax.plot(t, e2, color=C_ID, lw=1.4, ls='--', alpha=0.7,
            label='Identity', zorder=5)
    kw = apply_kernel(e2, K, dt)
    ax.plot(t, kw, color=color, lw=2.4, label=name)
    ax.fill_between(t, e2, kw, alpha=0.13, color=color)
    style_ax(ax, ylabel=r'$(Kf_{X_2})(t)$' if col == 0 else None)
    tick = '✓' if 'hour' in name else '✗'
    fc   = '#f0fff4' if tick == '✓' else '#fff0f0'
    ax.set_title(f'{tick}  {name}', fontsize=9, fontweight='bold', color=color)
    # Expand y-axis top by 55% to carve out annotation space above curves
    ylo, yhi = ax.get_ylim()
    yrange = yhi - ylo
    ax.set_ylim(ylo, yhi + 0.55 * yrange)
    phase_labels(ax, y_frac=0.96)

    if col in (0, 1):
        legend_loc   = 'upper left'
        annot_x, annot_y, annot_va = 0.97, 0.97, 'top'
    else:
        legend_loc   = 'lower left'
        annot_x, annot_y, annot_va = 0.97, 0.97, 'top'

    ax.legend(fontsize=7, loc=legend_loc)
    ax.text(annot_x, annot_y, note, transform=ax.transAxes,
            fontsize=6.8, va=annot_va, ha='right', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', fc=fc, ec=color,
                      lw=1.2, alpha=0.95))

# ---- Row 2: Integrated importance — all kernels, all features -------------
all_kernels_ranked = [
    ('Identity',             K_id_,     C_ID),
    (r'OU $\ell=4$h',        K_ou4,     C_OU),
    (r'Matern-3/2 $\ell=4$h',K_matern, '#6a4c93'),
    (r'Gaussian $\sigma=2$h',K_gauss2, '#219ebc'),
    ('Correlation',          K_corr,    C_CORR),
    ('Constant',             K_constant,'#888888'),
    (r'Periodic $p=24$h',   K_periodic,'#e07b39'),
    (r'AR $\rho=0.85$/h',   K_ar_good, '#5e60ce'),
]

# Panel: integrated importance bars for each kernel
ax = fig.add_subplot(gs[2, :2])   # span first two columns
n_kernels = len(all_kernels_ranked)
n_feats   = 3
feat_effs = [e1, e2, e3]
feat_cols = [C_X1, C_X2, C_X3]
feat_labs = [r'$X_1$ recovery', r'$X_2$ shock', r'$X_3$ deterioration']

x_pos   = np.arange(n_kernels)
width   = 0.22
offsets = [-width, 0, width]

for f_idx, (eff, fc, fl) in enumerate(zip(feat_effs, feat_cols, feat_labs)):
    vals = [integrated(eff, K, dt) for _, K, _ in all_kernels_ranked]
    ax.bar(x_pos + offsets[f_idx], vals, width=width,
           color=fc, alpha=0.85, label=fl)

ax.set_xticks(x_pos)
ax.set_xticklabels([n for n, _, _ in all_kernels_ranked],
                   fontsize=7.5, rotation=20, ha='right')
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_ylabel(r'Integrated $\Phi_S = \int(Kf_S)(t)\,dt$', fontsize=9)
ax.set_title('Integrated importance by kernel — '
             r'does kernel choice change the ranking of $X_1,X_2,X_3$?',
             fontsize=9)
ax.legend(fontsize=8)

# Vertical separator between good and poor kernels
ax.axvline(4.5, color='gray', lw=1.0, ls='--', alpha=0.5)
ax.text(2.0, ax.get_ylim()[1]*0.92, '← Good choices',
        ha='center', fontsize=7.5, color='#2d6a2d', style='italic')
ax.text(6.0, ax.get_ylim()[1]*0.92, 'Poor / needs care →',
        ha='center', fontsize=7.5, color='#8b0000', style='italic')

# Panel: heatmap comparison of OU vs Periodic (structural difference)
ax2 = fig.add_subplot(gs[2, 2])
step = max(1, T_POINTS // 60)
im = ax2.imshow(K_ou4[::step, ::step], aspect='auto', cmap='YlOrRd',
                extent=[0, 24, 24, 0], vmin=0, vmax=1)
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
ax2.set_xlabel('s (h)', fontsize=8); ax2.set_ylabel('t (h)', fontsize=8)
ax2.set_title(r'OU $\ell=4$h kernel structure' '\n(diagonal bands)', fontsize=9)
ax2.tick_params(labelsize=7)
for lo, hi, c in [(8,12,C_X2),(16,20,C_X3)]:
    ax2.add_patch(mpatches.Rectangle((lo,lo),hi-lo,hi-lo,
                  fill=False,edgecolor=c,lw=1.5,ls='--'))

ax3 = fig.add_subplot(gs[2, 3])
im2 = ax3.imshow(K_periodic[::step, ::step], aspect='auto', cmap='YlOrRd',
                 extent=[0, 24, 24, 0], vmin=0, vmax=1)
plt.colorbar(im2, ax=ax3, fraction=0.046, pad=0.03)
ax3.set_xlabel('s (h)', fontsize=8); ax3.set_ylabel('t (h)', fontsize=8)
ax3.set_title(r'Periodic $p=24$h kernel structure' '\n(corner coupling — wrong for ICU)',
              fontsize=9)
ax3.tick_params(labelsize=7)
# Highlight the corner coupling that is clinically wrong
ax3.add_patch(mpatches.Rectangle((0, 20), 4, 4,
              fill=False, edgecolor=C_PROB, lw=2.0, ls='--'))
ax3.add_patch(mpatches.Rectangle((20, 0), 4, 4,
              fill=False, edgecolor=C_PROB, lw=2.0, ls='--'))
ax3.text(2, 18.5, 'Spurious\ncoupling', ha='center', fontsize=6,
         color=C_PROB)

savefig(fig, 'fig7_kernel_suitability.pdf')

# ===========================================================================
# Summary
# ===========================================================================
print(f'\nAll figures saved to {PLOT_DIR}/')
for f in sorted(os.listdir(PLOT_DIR)):
    sz = os.path.getsize(os.path.join(PLOT_DIR, f)) // 1024
    print(f'  {f}  ({sz} KB)')