"""
Part 3 — Kernel Guidance (complete script)
==========================================================================
Produces all Part 3 figures:

  fig1_icu_kernel_guidance.pdf          — ICU example (full detail)
  fig2_pricepulse_kernel_guidance.pdf   — Price Pulse example (full detail)
  fig3_periodic_kernel_guidance.pdf     — Periodic example (no annotation boxes)
  fig4_ranking_summary.pdf              — Time-aggregated ranking summary
  fig5_condensed_kernel_guidance.pdf    — Condensed 2×3 main-paper figure

All computations are analytical (oracle / true model).
Row-normalised kernel application throughout.

Examples:
  1 — ICU early warning     Kernels: Identity | OU (ell=4h) | Correlation
  2 — Price Pulse            Kernels: Identity | Causal | Gaussian (wrong)
  3 — Periodic (3-day)       Kernels: Identity | OU (ell=4h) | Periodic (p=24h)

Output: Hilbert_functional_decomposition/plots/synthetic_experiments/part_3/

Usage:
  python part3_synthetic_experiments_full.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

PLOT_DIR = os.path.join(
    'plots', 'synthetic_experiments', 'part_3',
)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared color constants
# ---------------------------------------------------------------------------

C_ID   = '#888888'   # Identity   — unchanged
C_OU   = '#E69F00'   # OU         — was #457b9d, now amber (KC1[1])
C_CORR = '#009E73'   # Correlation— was #8338ec, now dark-teal (KC1[2])
C_CAUS = '#2a9d8f'   # Causal     — unchanged (KC2[1])
C_GAUS = '#E35B1A'   # Gaussian   — was #e63946, now vermillion (KC2[2])
C_PER  = '#f4a261'   # Periodic   — unchanged (KC3[2])
C_X1   = '#0072B2'
C_X2   = '#D55E00'
C_X3   = '#CC79A7'

# ===========================================================================
# Kernel constructors
# ===========================================================================

def kernel_identity(t):
    return np.eye(len(t))

def kernel_ou(t, ell):
    return np.exp(-np.abs(t[:,None]-t[None,:])/ell)

def kernel_causal(t, ell):
    d = t[:,None]-t[None,:]
    return np.where(d >= 0, np.exp(-d/ell), 0.0)

def kernel_gaussian(t, sigma):
    return np.exp(-0.5*((t[:,None]-t[None,:])/sigma)**2)

def kernel_periodic(t, period, ell):
    d = np.abs(t[:,None]-t[None,:])
    return np.exp(-2.0*np.sin(np.pi*d/period)**2/ell**2)

def kernel_correlation_icu(t):
    VAR = 1.0/12.0
    p1  = np.exp(-0.2*t)
    p2  = np.exp(-0.5*(t-10.0)**2)
    p3  = np.exp(-0.5*(t-18.0)**2)
    C   = VAR*(np.outer(p1,p1)+np.outer(p2,p2)+np.outer(p3,p3))
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)
    return C/np.outer(std,std)

# ===========================================================================
# Kernel application — row-normalised
# ===========================================================================

def apply_kernel(effect, K, dt):
    """
    Row-normalised: (Kf)(t) = [int K(t,s) ds]^{-1} * int K(t,s) f(s) ds
    Identity kernel returns f(t) exactly.
    """
    if np.allclose(K, np.eye(K.shape[0]), atol=1e-10):
        return effect.copy()
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return (K / rs) @ effect * dt


def time_aggregated(effect, K, dt):
    """Phi_S = int (Kf_S)(t) dt."""
    ke = apply_kernel(effect, K, dt)
    if np.allclose(K, np.eye(K.shape[0]), atol=1e-10):
        return float(np.sum(ke) * dt)
    return float(np.trapezoid(ke, dx=dt))


def relative_importance(effects_dict, K, dt):
    raw   = {fn: time_aggregated(eff, K, dt) for fn, eff in effects_dict.items()}
    total = sum(abs(v) for v in raw.values())
    if total < 1e-14:
        return {fn: 0.0 for fn in raw}
    return {fn: v/total for fn, v in raw.items()}

# ===========================================================================
# Plotting helpers
# ===========================================================================

def savefig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


def _spine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _shade(ax, lo, hi, color, alpha=0.07):
    ax.axvspan(lo, hi, alpha=alpha, color=color, zorder=0)

# ===========================================================================
# Shared build_figure for figs 1–3 (full detail, 3-row layout)
# ===========================================================================

def build_figure(
    title, t, dt, effects, feat_colors, kernels, landmarks,
    wrong_markers=None, phase_bands=None, causal_vline=None,
    day_vlines=None, t_xlim=None, t_xticks=None, t_xticklabels=None,
    annotations=None,
):
    """
    Row 0 — Time-resolved  : one panel per kernel.
    Row 1 — Time-specific  : grouped bars at landmark times.
    Row 2 — Time-aggregated: grouped bars, all kernels side by side.
    """
    n_k        = len(kernels)
    feat_names = list(effects.keys())
    n_feats    = len(feat_names)
    n_lm       = len(landmarks)
    n_cols_top = max(n_k, n_lm)

    fig = plt.figure(figsize=(4.5*n_cols_top, 14))
    gs  = gridspec.GridSpec(3, n_cols_top, figure=fig,
                            hspace=0.58, wspace=0.28,
                            left=0.07, right=0.97,
                            top=0.92, bottom=0.05)
    fig.suptitle(title, fontsize=10, fontweight='bold')

    xl    = t_xlim    or (float(t.min()), float(t.max()))
    xtk   = t_xticks  or list(np.arange(0, t.max()+1e-6,
                                         max(1, int(t.max()/6))))
    xlabs = t_xticklabels or [str(int(v)) for v in xtk]

    # ------------------------------------------------------------------
    # Row 0: time-resolved
    # ------------------------------------------------------------------
    for col, (k_label, K, k_color) in enumerate(kernels):
        ax     = fig.add_subplot(gs[0, col])
        curves = {fn: apply_kernel(eff, K, dt) for fn, eff in effects.items()}
        ymax   = max(np.abs(c).max() for c in curves.values()) * 1.30
        ymin   = min(min(c.min() for c in curves.values()) * 1.10,
                     -0.02*ymax)

        for fname, ke in curves.items():
            ax.plot(t, ke, color=feat_colors[fname], lw=2.2, label=fname)

        ax.axhline(0, color='gray', lw=0.6, ls=':')
        if causal_vline is not None:
            ax.axvline(causal_vline, color='#888', lw=0.9, ls='--', alpha=0.6)
        if phase_bands:
            for lo, hi, pc in phase_bands:
                _shade(ax, lo, hi, pc)
        if day_vlines:
            for dv in day_vlines:
                ax.axvline(dv, color='gray', lw=0.8, ls=':', alpha=0.5)

        ax.set_xlim(*xl)
        ax.set_xticks(xtk); ax.set_xticklabels(xlabs, fontsize=7)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=8); _spine(ax)
        ax.set_title(k_label, fontsize=10, fontweight='bold', color=k_color)
        ax.set_xlabel('Time', fontsize=8)
        if col == 0:
            ax.set_ylabel(r'$f_S(t)$  /  $(Kf_S)(t)$', fontsize=9)
            ax.legend(fontsize=7.5, loc='upper right')
        else:
            ax.set_ylabel(r'$(Kf_S)(t)$', fontsize=9)
        if annotations and k_label in annotations:
            ax.text(0.97, 0.97, annotations[k_label],
                    transform=ax.transAxes, fontsize=7,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='#fafafa', ec=k_color, lw=1.2, alpha=0.95))
        if col == n_k-1:
            ax.text(1.03, 0.5, 'Time-resolved',
                    transform=ax.transAxes, fontsize=9,
                    va='center', rotation=270, color='gray')

    # ------------------------------------------------------------------
    # Row 1: time-specific
    # ------------------------------------------------------------------
    bar_w   = 0.75/n_k
    offsets = np.linspace(-(n_k-1)/2, (n_k-1)/2, n_k)*bar_w

    for lm_col, (lm_t, lm_label) in enumerate(landmarks):
        ax  = fig.add_subplot(gs[1, lm_col])
        idx = np.argmin(np.abs(t-lm_t))

        all_lm  = [apply_kernel(eff, K, dt)[idx]
                   for _, K, _ in kernels for eff in effects.values()]
        lm_ymax = max(abs(v) for v in all_lm) * 1.30

        for k_idx, (k_label, K, k_color) in enumerate(kernels):
            vals  = [apply_kernel(effects[fn], K, dt)[idx]
                     for fn in feat_names]
            x_pos = np.arange(n_feats)
            ax.bar(x_pos+offsets[k_idx], vals,
                   width=bar_w*0.88, color=k_color,
                   alpha=0.85, label=k_label)
            if wrong_markers:
                for wk, wf, wt in wrong_markers:
                    if wk == k_label and abs(lm_t-wt) < 1e-6:
                        fi = feat_names.index(wf)
                        ax.plot(fi+offsets[k_idx],
                                vals[fi]+0.02*lm_ymax,
                                marker='x', color='#e63946',
                                ms=9, mew=2.2, zorder=10, clip_on=False)

        ax.set_xticks(np.arange(n_feats))
        ax.set_xticklabels(feat_names, fontsize=7.5)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.set_ylim(-0.05*lm_ymax, lm_ymax)
        ax.set_title(f't = {lm_label}', fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=8); _spine(ax)
        if lm_col == 0:
            ax.set_ylabel(r'$(Kf_S)(t_0)$', fontsize=9)
            ax.legend(fontsize=6, loc='upper right')
        if lm_col == n_lm-1:
            ax.text(1.03, 0.5, 'Time-specific',
                    transform=ax.transAxes, fontsize=9,
                    va='center', rotation=270, color='gray')

    # ------------------------------------------------------------------
    # Row 2: time-aggregated
    # ------------------------------------------------------------------
    ax_agg = fig.add_subplot(gs[2, :])
    x_pos  = np.arange(n_feats)
    bar_w2 = 0.75/n_k
    off2   = np.linspace(-(n_k-1)/2, (n_k-1)/2, n_k)*bar_w2

    for k_idx, (k_label, K, k_color) in enumerate(kernels):
        vals = [time_aggregated(effects[fn], K, dt) for fn in feat_names]
        ax_agg.bar(x_pos+off2[k_idx], vals,
                   width=bar_w2*0.88, color=k_color,
                   alpha=0.85, label=k_label)

    ax_agg.set_xticks(x_pos)
    ax_agg.set_xticklabels(feat_names, fontsize=9)
    ax_agg.axhline(0, color='gray', lw=0.6, ls=':')
    ax_agg.tick_params(labelsize=8); _spine(ax_agg)
    ax_agg.set_ylabel(r'$\Phi_S = \int(Kf_S)(t)\,dt$', fontsize=9)
    ax_agg.set_title(
        'Time-aggregated importance  '
        r'(ranking preserved across kernels)',
        fontsize=9)
    ax_agg.legend(fontsize=8, ncol=n_k, loc='upper right')
    ax_agg.text(1.01, 0.5, 'Time-aggregated',
                transform=ax_agg.transAxes, fontsize=9,
                va='center', rotation=270, color='gray')
    return fig

# ===========================================================================
# Fig 1 — ICU Early Warning
# ===========================================================================

def make_icu_figure():
    T, TP = 24.0, 240
    t  = np.linspace(0, T, TP); dt = t[1]-t[0]; MU = 0.5
    effects = {
        'X1': (0.8-MU)*np.exp(-0.2*t),
        'X2': (0.9-MU)*np.exp(-0.5*(t-10.0)**2),
        'X3': (0.7-MU)*np.exp(-0.5*(t-18.0)**2),
    }
    kernels = [
        ('Identity',         kernel_identity(t),         C_ID),
        ('OU  ($\\ell=4$h)', kernel_ou(t, ell=4.0),      C_OU),
        ('Correlation',      kernel_correlation_icu(t),   C_CORR),
    ]
    landmarks = [
        (0.0,'0 h'), (5.0,'5 h'),
        (10.0,'10 h\n(shock peak)'),
        (18.0,'18 h\n(detn. peak)'), (22.0,'22 h'),
    ]
    annotations = {
        'OU  ($\\ell=4$h)':
            'Spreads $X_2$ over $[6,14]$h\n'
            'Answers: "effect of $X_2$\n'
            r'over the shock neighbourhood"',
        'Correlation':
            'Aggregates $t_0$ with times\n'
            'that co-vary across patients\n'
            '→ phase-aware, no $\\ell$ needed',
    }
    title = (
        'Example 1 — ICU Early Warning  (row-normalised kernel)\n'
        r'$F(\mathbf{x})(t)=X_1 e^{-0.2t}+X_2 e^{-(t-10)^2/2}+'
        r'X_3 e^{-(t-18)^2/2}$'
        r',  $\mathbf{x}^*=(0.8,\,0.9,\,0.7)$'
        '\nIdentity (raw effects) | OU smooths neighbourhood | '
        'Correlation discovers phase structure'
    )
    fig = build_figure(
        title=title, t=t, dt=dt,
        effects=effects, feat_colors={'X1':C_X1,'X2':C_X2,'X3':C_X3},
        kernels=kernels, landmarks=landmarks,
        phase_bands=[(8,12,C_X2),(16,20,C_X3)],
        annotations=annotations,
        t_xlim=(0,24), t_xticks=list(range(0,25,4)),
        t_xticklabels=[str(v) for v in range(0,25,4)],
    )
    savefig(fig, 'fig1_icu_kernel_guidance.pdf')

# ===========================================================================
# Fig 2 — Price Pulse Demand Response
# ===========================================================================

def make_pricepulse_figure():
    T, TP = 4.0, 480
    t  = np.linspace(0, T, TP); dt = t[1]-t[0]; MU = 0.5
    effects = {
        'X1 (pulse)': (0.9-MU)*((t>=0.5)&(t<1.0)).astype(float),
        'X2 (temp.)': (0.7-MU)*np.exp(-0.5*((t-2.0)/1.5)**2),
        'X3 (base)':  (0.6-MU)*np.ones_like(t),
    }
    feat_colors = {'X1 (pulse)':C_X1,'X2 (temp.)':C_X2,'X3 (base)':C_X3}
    kernels = [
        ('Identity',                     kernel_identity(t),           C_ID),
        ('Causal  ($\\ell=0.33$h)',      kernel_causal(t, ell=0.33),   C_CAUS),
        ('Gaussian  ($\\sigma=0.3$h) ✗', kernel_gaussian(t, sigma=0.3),C_GAUS),
    ]
    landmarks = [
        (0.3,'0.3 h\n(pre-pulse)'), (0.75,'0.75 h\n(in pulse)'),
        (1.5,'1.5 h\n(post-pulse)'),(2.5,'2.5 h\n(later)'),
    ]
    wrong_markers = [('Gaussian  ($\\sigma=0.3$h) ✗','X1 (pulse)',0.3)]
    annotations = {
        'Causal  ($\\ell=0.33$h)':
            'Exactly 0 before pulse\n'
            'Lingering post-pulse\n'
            '(AR(1) dynamics)',
        'Gaussian  ($\\sigma=0.3$h) ✗':
            '✗ Nonzero BEFORE pulse\n'
            '  (anticipates future event)\n'
            '  Physically impossible',
    }
    title = (
        'Example 2 — Price Pulse Demand Response  (row-normalised kernel)\n'
        r'$F(\mathbf{x})(t)=X_1\,\mathbf{1}_{[0.5,1.0)}(t)+'
        r'X_2\,e^{-(t-2)^2/4.5}+X_3$'
        r',  $\mathbf{x}^*=(0.9,\,0.7,\,0.6)$'
        '\nIdentity (raw effects) | Causal (correct) | '
        'Gaussian (wrong: symmetric)'
    )
    fig = build_figure(
        title=title, t=t, dt=dt,
        effects=effects, feat_colors=feat_colors,
        kernels=kernels, landmarks=landmarks,
        wrong_markers=wrong_markers, causal_vline=0.5,
        annotations=annotations,
        t_xlim=(0,4),
        t_xticks=[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0],
        t_xticklabels=['0','0.5','1','1.5','2','2.5','3','3.5','4'],
    )
    for ax in fig.axes[:3]:
        ax.axvspan(0.5, 1.0, alpha=0.10, color=C_X1, zorder=0)
    savefig(fig, 'fig2_pricepulse_kernel_guidance.pdf')

# ===========================================================================
# Fig 3 — Periodic (no annotation boxes)
# ===========================================================================

def make_periodic_figure():
    T, TP = 72.0, 720
    t  = np.linspace(0, T, TP); dt = t[1]-t[0]; MU = 0.5

    effects = {
        'X1 (8am med.)':    (0.8-MU)*np.exp(-0.5*((t%24-8)**2)/4.0),
        'X2 (day-1 8pm)':   (0.9-MU)*np.exp(-0.5*(t-20.0)**2/0.5),
    }
    feat_colors = {'X1 (8am med.)':C_X1,'X2 (day-1 8pm)':C_X2}
    kernels = [
        ('Identity',              kernel_identity(t),              C_ID),
        ('OU  ($\\ell=4$h)',      kernel_ou(t, ell=4.0),           C_OU),
        ('Periodic ($p=24$h)',    kernel_periodic(t, 24.0, 1.0),   C_PER),
    ]
    landmarks = [
        (8.0,  'Day 1\n8 am'),
        (20.0, 'Day 1\n8 pm\n(acute)'),
        (32.0, 'Day 2\n8 am'),
        (44.0, 'Day 2\n8 pm'),
        (56.0, 'Day 3\n8 am'),
    ]
    wrong_markers = [('Periodic ($p=24$h)', 'X2 (day-1 8pm)', 44.0)]
    phase_bands = [
        (6,10,C_X1),(18,22,C_X2),
        (30,34,C_X1),(42,46,C_X2),
        (54,58,C_X1),(66,70,C_X2),
    ]

    feat_names = list(effects.keys())
    n_k    = len(kernels)
    n_lm   = len(landmarks)
    n_cols = max(n_k, n_lm)

    fig = plt.figure(figsize=(4.5*n_cols, 14))
    gs  = gridspec.GridSpec(3, n_cols, figure=fig,
                            hspace=0.58, wspace=0.28,
                            left=0.07, right=0.97,
                            top=0.92, bottom=0.05)
    fig.suptitle(
        'Example 3 — Sleep Quality: 3-Day Recurring Medication  '
        '(row-normalised kernel)\n'
        r'$X_1$: morning medication (8am, every day)  |  '
        r'$X_2$: acute evening event (8pm, day 1 only)'
        '\n'
        r'$\mathbf{x}^*=(0.8,\,0.9)$,  $T=[0,72]$h  |  '
        'Identity | OU | Periodic ($p=24$h, $\\ell=1$)',
        fontsize=10, fontweight='bold',
    )

    xtk   = list(range(0, 73, 12))
    xlabs = [f'{v}h' for v in xtk]

    # Row 0: time-resolved — no annotation boxes
    for col, (k_label, K, k_color) in enumerate(kernels):
        ax     = fig.add_subplot(gs[0, col])
        curves = {fn: apply_kernel(eff, K, dt) for fn, eff in effects.items()}
        ymax   = max(np.abs(c).max() for c in curves.values()) * 1.30
        ymin   = min(min(c.min() for c in curves.values()) * 1.10,
                     -0.02*ymax)
        for fname, ke in curves.items():
            ax.plot(t, ke, color=feat_colors[fname], lw=2.2, label=fname)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        for pb in phase_bands:
            ax.axvspan(pb[0], pb[1], alpha=0.07, color=pb[2], zorder=0)
        for dv in [24.0, 48.0]:
            ax.axvline(dv, color='gray', lw=0.8, ls=':', alpha=0.5)
        ax.set_xlim(0, 72); ax.set_xticks(xtk)
        ax.set_xticklabels(xlabs, fontsize=7)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=8); _spine(ax)
        ax.set_title(k_label, fontsize=10, fontweight='bold', color=k_color)
        ax.set_xlabel('Time', fontsize=8)
        if col == 0:
            ax.set_ylabel(r'$(Kf_S)(t)$', fontsize=9)
            ax.legend(fontsize=7.5, loc='upper right')
        if col == n_k-1:
            ax.text(1.03, 0.5, 'Time-resolved',
                    transform=ax.transAxes, fontsize=9,
                    va='center', rotation=270, color='gray')
        for d, dlabel in [(12,'Day 1'),(36,'Day 2'),(60,'Day 3')]:
            ax.text(d, ymax*0.95, dlabel,
                    ha='center', fontsize=7, color='gray', style='italic')

    # Row 1: time-specific
    bar_w   = 0.75/n_k
    offsets = np.linspace(-(n_k-1)/2, (n_k-1)/2, n_k)*bar_w
    for lm_col, (lm_t, lm_label) in enumerate(landmarks):
        ax  = fig.add_subplot(gs[1, lm_col])
        idx = np.argmin(np.abs(t-lm_t))
        all_vals = [apply_kernel(eff, K, dt)[idx]
                    for _, K, _ in kernels for eff in effects.values()]
        lm_ymax  = max(abs(v) for v in all_vals) * 1.30
        for k_idx, (k_label, K, k_color) in enumerate(kernels):
            vals  = [apply_kernel(effects[fn], K, dt)[idx]
                     for fn in feat_names]
            x_pos = np.arange(len(feat_names))
            ax.bar(x_pos+offsets[k_idx], vals,
                   width=bar_w*0.88, color=k_color,
                   alpha=0.85, label=k_label)
            for wk, wf, wt in wrong_markers:
                if wk == k_label and abs(lm_t-wt) < 1e-6:
                    fi = feat_names.index(wf)
                    ax.plot(fi+offsets[k_idx],
                            vals[fi]+0.02*lm_ymax,
                            marker='x', color='#e63946',
                            ms=9, mew=2.2, zorder=10, clip_on=False)
        ax.set_xticks(np.arange(len(feat_names)))
        ax.set_xticklabels(feat_names, fontsize=7.5)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.set_ylim(-0.05*lm_ymax, lm_ymax)
        ax.set_title(f't = {lm_label}', fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=8); _spine(ax)
        if lm_col == 0:
            ax.set_ylabel(r'$(Kf_S)(t_0)$', fontsize=9)
            ax.legend(fontsize=6, loc='upper right')
        if lm_col == n_lm-1:
            ax.text(1.03, 0.5, 'Time-specific',
                    transform=ax.transAxes, fontsize=9,
                    va='center', rotation=270, color='gray')

    # Row 2: time-aggregated
    ax_agg = fig.add_subplot(gs[2, :])
    x_pos  = np.arange(len(feat_names))
    bar_w2 = 0.75/n_k
    off2   = np.linspace(-(n_k-1)/2, (n_k-1)/2, n_k)*bar_w2
    for k_idx, (k_label, K, k_color) in enumerate(kernels):
        vals = [time_aggregated(effects[fn], K, dt) for fn in feat_names]
        ax_agg.bar(x_pos+off2[k_idx], vals,
                   width=bar_w2*0.88, color=k_color,
                   alpha=0.85, label=k_label)
    ax_agg.set_xticks(x_pos)
    ax_agg.set_xticklabels(feat_names, fontsize=9)
    ax_agg.axhline(0, color='gray', lw=0.6, ls=':')
    ax_agg.tick_params(labelsize=8); _spine(ax_agg)
    ax_agg.set_ylabel(r'$\Phi_S = \int(Kf_S)(t)\,dt$', fontsize=9)
    ax_agg.set_title('Time-aggregated importance', fontsize=9)
    ax_agg.legend(fontsize=8, ncol=n_k, loc='upper right')
    ax_agg.text(1.01, 0.5, 'Time-aggregated',
                transform=ax_agg.transAxes, fontsize=9,
                va='center', rotation=270, color='gray')

    savefig(fig, 'fig3_periodic_kernel_guidance.pdf')

# ===========================================================================
# Fig 4 — Ranking summary
# ===========================================================================

def make_ranking_summary():
    MU = 0.5

    T1,TP1=24.0,240; t1=np.linspace(0,T1,TP1); dt1=t1[1]-t1[0]
    eff_icu = {'X1':(0.8-MU)*np.exp(-0.2*t1),
               'X2':(0.9-MU)*np.exp(-0.5*(t1-10)**2),
               'X3':(0.7-MU)*np.exp(-0.5*(t1-18)**2)}
    icu_k   = [('Identity',    kernel_identity(t1),         C_ID),
               ('OU ℓ=4h',     kernel_ou(t1,4.0),           C_OU),
               ('Correlation', kernel_correlation_icu(t1),  C_CORR)]

    T2,TP2=4.0,480; t2=np.linspace(0,T2,TP2); dt2=t2[1]-t2[0]
    eff_pp  = {'X1\n(pulse)':(0.9-MU)*((t2>=0.5)&(t2<1.0)).astype(float),
               'X2\n(temp.)':(0.7-MU)*np.exp(-0.5*((t2-2)/1.5)**2),
               'X3\n(base)': (0.6-MU)*np.ones_like(t2)}
    pp_k    = [('Identity',      kernel_identity(t2),        C_ID),
               ('Causal',        kernel_causal(t2,0.33),     C_CAUS),
               ('Gaussian\n(✗)', kernel_gaussian(t2,0.3),    C_GAUS)]

    T3,TP3=72.0,720; t3=np.linspace(0,T3,TP3); dt3=t3[1]-t3[0]
    eff_per = {'X1\n(8am)': (0.8-MU)*np.exp(-0.5*((t3%24-8)**2)/4),
               'X2\n(8pm)': (0.9-MU)*np.exp(-0.5*(t3-20)**2/0.5)}
    per_k   = [('Identity',       kernel_identity(t3),            C_ID),
               ('OU ℓ=4h',        kernel_ou(t3,4.0),              C_OU),
               ('Periodic p=24h', kernel_periodic(t3,24.0,1.0),   C_PER)]

    examples = [
        ('ICU Early Warning',  eff_icu, icu_k, dt1),
        ('Price Pulse',        eff_pp,  pp_k,  dt2),
        ('Periodic (3-day)',   eff_per, per_k, dt3),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        r'Time-aggregated importance $\Phi_S = \int(Kf_S)(t)\,dt$'
        ' — all examples and kernels\n'
        'Ranking is preserved across kernels in all three examples '
        '(redistribution visible in relative bar heights).',
        fontsize=10, fontweight='bold',
    )

    for col, (ex_title, effects, ex_kernels, ex_dt) in enumerate(examples):
        feat_names = list(effects.keys())
        n_feats    = len(feat_names)
        n_k        = len(ex_kernels)
        bar_w      = 0.75/n_k
        offsets    = np.linspace(-(n_k-1)/2,(n_k-1)/2,n_k)*bar_w
        x_pos      = np.arange(n_feats)
        ax         = axes[col]

        for k_idx, (k_label, K, k_color) in enumerate(ex_kernels):
            vals = [time_aggregated(effects[fn], K, ex_dt)
                    for fn in feat_names]
            ax.bar(x_pos+offsets[k_idx], vals,
                   width=bar_w*0.88, color=k_color,
                   alpha=0.85, label=k_label)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(feat_names, fontsize=8)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.tick_params(labelsize=8); _spine(ax)
        ax.set_ylabel(r'$\Phi_S$', fontsize=9)
        ax.set_title(ex_title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')

        print(f'\n  {ex_title}:')
        for k_label, K, _ in ex_kernels:
            rel    = relative_importance(effects, K, ex_dt)
            ranked = sorted(feat_names, key=lambda f: -rel[f])
            s = '  '.join(f'{f}({rel[f]:.3f})' for f in ranked)
            print(f'    {k_label:20s}: {s}')

    plt.tight_layout()
    savefig(fig, 'fig4_ranking_summary.pdf')

# ===========================================================================
# Fig 5 — Condensed 2×3 figure for main paper body
# ===========================================================================

def make_condensed_figure():
    """
    Layout: 2 rows × 3 columns
      Row 0: time-resolved — color=kernel (per-example palette), ls=feature
      Row 1: time-specific grouped bars — color=kernel (same palette), hatch=feature

    Color schemes (colorblind-friendly, one per example, Wong 2011 palette):
      ICU        : grey / amber #E69F00 / dark-teal #009E73
      Price Pulse: grey / teal  #2a9d8f / vermillion #E35B1A
      Periodic   : grey / steel-blue #3A86C8 / orange #f4a261

    Linestyle / hatch conventions:
      feature 1  : solid '-'   / no hatch
      feature 2  : dashed '--' / '///'
      feature 3  : dotted ':'  / 'xxx'  (ICU only)
    """
    MU = 0.5

    # Per-example kernel color palettes
    KC1 = ['#888888', '#E69F00', '#009E73']   # ICU: grey / amber / dark-teal
    KC2 = ['#888888', '#2a9d8f', '#E35B1A']   # Price Pulse: grey / teal / vermillion
    KC3 = ['#888888', '#3A86C8', '#f4a261']   # Periodic: grey / steel-blue / orange

    LS_FEATS    = ['-', '--', ':']
    HATCH_FEATS = ['', '///', 'xxx']

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    T1, TP1 = 24.0, 240
    t1 = np.linspace(0, T1, TP1)
    eff1 = {
        'X1': (0.8-MU)*np.exp(-0.2*t1),
        'X2': (0.9-MU)*np.exp(-0.5*(t1-10.0)**2),
        'X3': (0.7-MU)*np.exp(-0.5*(t1-18.0)**2),
    }
    K1_id, K1_ou, K1_corr = (kernel_identity(t1),
                              kernel_ou(t1, ell=4.0),
                              kernel_correlation_icu(t1))
    lm1 = [(5.0,'$t{=}5$h'), (10.0,'$t{=}10$h'), (22.0,'$t{=}22$h')]

    T2, TP2 = 4.0, 480
    t2 = np.linspace(0, T2, TP2)
    eff2 = {
        'X1': (0.9-MU)*((t2>=0.5)&(t2<1.0)).astype(float),
        'X2': (0.7-MU)*np.exp(-0.5*((t2-2.0)/1.5)**2),
        'X3': (0.6-MU)*np.ones_like(t2),
    }
    K2_id, K2_caus, K2_gaus = (kernel_identity(t2),
                                kernel_causal(t2, ell=0.33),
                                kernel_gaussian(t2, sigma=0.3))
    lm2 = [(0.3,'$t{=}0.3$h'), (0.75,'$t{=}0.75$h'), (1.5,'$t{=}1.5$h')]

    T3, TP3 = 72.0, 720
    t3 = np.linspace(0, T3, TP3)
    eff3 = {
        'X1': (0.8-MU)*np.exp(-0.5*((t3%24-8)**2)/4.0),
        'X2': (0.9-MU)*np.exp(-0.5*(t3-20.0)**2/0.5),
    }
    K3_id, K3_ou, K3_per = (kernel_identity(t3),
                             kernel_ou(t3, ell=4.0),
                             kernel_periodic(t3, 24.0, 1.0))
    lm3 = [(20.0,'Day 1\n8pm'), (32.0,'Day 2\n8am'), (44.0,'Day 2\n8pm')]

    # kernels_info: (label, K, color)
    kernels1 = [('Identity',    K1_id,   KC1[0]),
                ('OU',          K1_ou,   KC1[1]),
                ('Correlation', K1_corr, KC1[2])]
    kernels2 = [('Identity',   K2_id,   KC2[0]),
                ('Causal',     K2_caus, KC2[1]),
                ('Gaussian ✗', K2_gaus, KC2[2])]
    kernels3 = [('Identity', K3_id,  KC3[0]),
                ('OU',       K3_ou,  KC3[1]),
                ('Periodic', K3_per, KC3[2])]

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        'Kernel guidance: time-resolved effects and time-specific attribution',
        fontsize=11, fontweight='bold',
    )
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.38, wspace=0.32,
        left=0.06, right=0.97,
        top=0.88, bottom=0.08,
    )

    # ------------------------------------------------------------------
    # draw_resolved: color=kernel, ls=feature
    # ------------------------------------------------------------------
    def draw_resolved(ax, t, effects, kernels_info,
                      xlim, xticks, xticklabels,
                      phase_bands=None, day_vlines=None, causal_vline=None):
        dt         = t[1]-t[0]
        feat_names = list(effects.keys())
        curves     = {kl: {fn: apply_kernel(eff, K, dt)
                           for fn, eff in effects.items()}
                      for kl, K, _ in kernels_info}
        ymax = max(np.abs(c).max()
                   for kd in curves.values() for c in kd.values()) * 1.28
        ymin = -0.05 * ymax
        if phase_bands:
            for lo, hi, pc in phase_bands:
                ax.axvspan(lo, hi, alpha=0.07, color=pc, zorder=0)
        if day_vlines:
            for dv in day_vlines:
                ax.axvline(dv, color='gray', lw=0.7, ls=':', alpha=0.5)
        if causal_vline is not None:
            ax.axvline(causal_vline, color='#aaa', lw=0.8, ls='--', alpha=0.6)
        for fi, fn in enumerate(feat_names):
            ls = LS_FEATS[fi]
            for kl, K, kc in kernels_info:
                ax.plot(t, curves[kl][fn], color=kc,
                        ls=ls, lw=1.9, alpha=0.92)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_xlim(*xlim); ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=7)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=8); _spine(ax)
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel(r'$(Kf_S)(t)$', fontsize=9)
        return ymax

    # ------------------------------------------------------------------
    # draw_specific: color=kernel, hatch=feature
    # x-ticks = feature names; landmark labels as text above groups
    # ------------------------------------------------------------------
    def draw_specific(ax, t, effects, kernels_info, landmarks,
                      wrong_markers=None):
        feat_names = list(effects.keys())
        n_feats    = len(feat_names)
        n_k        = len(kernels_info)
        n_lm       = len(landmarks)
        dt         = t[1]-t[0]
        bar_w      = 0.55 / n_k
        gap        = n_feats + 0.8
        all_vals   = [
            apply_kernel(effects[fn], K, dt)[np.argmin(np.abs(t-lm_t))]
            for lm_t, _ in landmarks
            for _, K, _ in kernels_info
            for fn in feat_names
        ]
        ymax          = max(abs(v) for v in all_vals) * 1.35
        group_offsets = np.linspace(-(n_k-1)/2, (n_k-1)/2, n_k) * bar_w
        xtick_pos, xtick_labels = [], []

        for lm_idx, (lm_t, lm_label) in enumerate(landmarks):
            idx      = np.argmin(np.abs(t - lm_t))
            x_origin = lm_idx * (n_feats + gap)
            feat_x   = np.arange(n_feats, dtype=float) + x_origin

            for k_idx, (kl, K, kc) in enumerate(kernels_info):
                vals = [apply_kernel(effects[fn], K, dt)[idx]
                        for fn in feat_names]
                for fi, (fn, v) in enumerate(zip(feat_names, vals)):
                    ax.bar(feat_x[fi] + group_offsets[k_idx], v,
                           width=bar_w*0.88, color=kc,
                           hatch=HATCH_FEATS[fi],
                           edgecolor='white', linewidth=0.4,
                           alpha=0.85,
                           label=(kl if (lm_idx == 0 and fi == 0) else '_'))

            ax.text(x_origin + (n_feats-1)/2, ymax*0.98, lm_label,
                    ha='center', va='top', fontsize=7.5,
                    fontweight='bold', color='#333333')

            if lm_idx < n_lm-1:
                ax.axvline(x_origin + n_feats - 1 + gap/2,
                           color='#cccccc', lw=1.0, ls='-')

            for fi, fn in enumerate(feat_names):
                xtick_pos.append(feat_x[fi])
                xtick_labels.append(fn)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, fontsize=7.5)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_ylim(-0.08*ymax, ymax)
        ax.tick_params(labelsize=8); _spine(ax)
        ax.set_ylabel(r'$(Kf_S)(t_0)$', fontsize=9)

    # ------------------------------------------------------------------
    # Legend builders
    # ------------------------------------------------------------------
    def make_resolved_handles(feat_names, feat_labels, kernels_info):
        handles = []
        for kl, _, kc in kernels_info:
            handles.append(mlines.Line2D([], [], color=kc, lw=2, ls='-',
                                         label=kl))
        handles.append(mlines.Line2D([], [], color='none', label=''))
        for fi, fl in enumerate(feat_labels):
            handles.append(mlines.Line2D([], [], color='#333333',
                                         ls=LS_FEATS[fi], lw=1.8, label=fl))
        return handles

    def make_specific_handles(feat_names, feat_labels, kernels_info):
        handles = []
        for kl, _, kc in kernels_info:
            handles.append(mpatches.Patch(facecolor=kc, alpha=0.85, label=kl))
        handles.append(mlines.Line2D([], [], color='none', label=''))
        for fi, fl in enumerate(feat_labels):
            handles.append(mpatches.Patch(facecolor='#cccccc',
                                          hatch=HATCH_FEATS[fi],
                                          edgecolor='#333333', label=fl))
        return handles

    # ------------------------------------------------------------------
    # ---- ICU (Example 1) ----
    # ------------------------------------------------------------------
    ax_r1 = fig.add_subplot(gs[0, 0])
    draw_resolved(ax_r1, t1, eff1, kernels1,
                  xlim=(0,24), xticks=list(range(0,25,4)),
                  xticklabels=[str(v) for v in range(0,25,4)],
                  phase_bands=[(8,12,KC1[1]),(16,20,KC1[2])])
    ax_r1.set_title(
        'Example 1 — ICU Early Warning\n'
        r'$F(\mathbf{x})(t)=X_1 e^{-0.2t}+X_2 e^{-(t-10)^2/2}+X_3 e^{-(t-18)^2/2}$',
        fontsize=8.5, fontweight='bold')
    ax_r1.legend(
        handles=make_resolved_handles(
            list(eff1.keys()),
            ['X1 (baseline)', 'X2 (shock)', 'X3 (detn.)'],
            kernels1),
        fontsize=6.5, loc='upper right', framealpha=0.85)

    ax_s1 = fig.add_subplot(gs[1, 0])
    draw_specific(ax_s1, t1, eff1, kernels1, lm1)
    ax_s1.legend(
        handles=make_specific_handles(
            list(eff1.keys()), ['X1', 'X2', 'X3'], kernels1),
        fontsize=6.5, loc='center right', framealpha=0.85)

    # ------------------------------------------------------------------
    # ---- Price Pulse (Example 2) ----
    # ------------------------------------------------------------------
    ax_r2 = fig.add_subplot(gs[0, 1])
    draw_resolved(ax_r2, t2, eff2, kernels2,
                  xlim=(0,4),
                  xticks=[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0],
                  xticklabels=['0','0.5','1','1.5','2','2.5','3','3.5','4'],
                  causal_vline=0.5)
    ax_r2.axvspan(0.5, 1.0, alpha=0.10, color=KC2[0], zorder=0)
    ax_r2.set_title(
        'Example 2 — Price Pulse Demand Response\n'
        r'$F(\mathbf{x})(t)=X_1\mathbf{1}_{[0.5,1.0)}(t)+X_2 e^{-(t-2)^2/4.5}+X_3$',
        fontsize=8.5, fontweight='bold')
    ax_r2.legend(
        handles=make_resolved_handles(
            list(eff2.keys()),
            ['X1 (pulse)', 'X2 (temp.)', 'X3 (base)'],
            kernels2),
        fontsize=6.5, loc='upper right', framealpha=0.85)

    ax_s2 = fig.add_subplot(gs[1, 1])
    draw_specific(ax_s2, t2, eff2, kernels2, lm2)
    ax_s2.legend(
        handles=make_specific_handles(
            list(eff2.keys()), ['X1', 'X2', 'X3'], kernels2),
        fontsize=6.5, loc='center left', framealpha=0.85)

    # ------------------------------------------------------------------
    # ---- Periodic (Example 3) ----
    # ------------------------------------------------------------------
    ax_r3 = fig.add_subplot(gs[0, 2])
    draw_resolved(ax_r3, t3, eff3, kernels3,
                  xlim=(0,72), xticks=list(range(0,73,12)),
                  xticklabels=[f'{v}h' for v in range(0,73,12)],
                  phase_bands=[(6,10,KC3[0]),(18,22,KC3[0]),
                               (30,34,KC3[0]),(42,46,KC3[0]),
                               (54,58,KC3[0]),(66,70,KC3[0])],
                  day_vlines=[24.0, 48.0])
    ax_r3.set_title(
        'Example 3 — Periodic (3-day Medication)\n'
        r'$X_1$: daily 8am (recurring),  $X_2$: acute 8pm day 1 only',
        fontsize=8.5, fontweight='bold')
    ymax3 = ax_r3.get_ylim()[1]
    for d, dl in [(12,'Day 1'),(36,'Day 2'),(60,'Day 3')]:
        ax_r3.text(d, ymax3*0.97, dl,
                   ha='center', va='top', fontsize=6.5,
                   color='gray', style='italic')
    ax_r3.legend(
        handles=make_resolved_handles(
            list(eff3.keys()),
            ['X1 (8am daily)', 'X2 (8pm day-1)'],
            kernels3),
        fontsize=6.5, loc='upper right', bbox_to_anchor=(1.0, 0.88),
        framealpha=0.85)

    ax_s3 = fig.add_subplot(gs[1, 2])
    draw_specific(ax_s3, t3, eff3, kernels3, lm3,
                  wrong_markers=[('Periodic', 'X2', 44.0)])
    ax_s3.legend(
        handles=make_specific_handles(
            list(eff3.keys()), ['X1', 'X2'], kernels3),
        fontsize=6.5, loc='center right', framealpha=0.85)

    # Row labels on right edge
    for ax, label in [(ax_r3, 'Time-resolved'), (ax_s3, 'Time-specific')]:
        ax.text(1.02, 0.5, label, transform=ax.transAxes, fontsize=9,
                va='center', rotation=270, color='gray')

    savefig(fig, 'fig5_condensed_kernel_guidance.pdf')

# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '='*60)
    print('  Part 3 — Kernel Guidance (full script)')
    print('='*60)
    print('\n[1] ICU ...');           make_icu_figure()
    print('\n[2] Price Pulse ...');   make_pricepulse_figure()
    print('\n[3] Periodic ...');      make_periodic_figure()
    print('\n[4] Ranking summary ...');make_ranking_summary()
    print('\n[5] Condensed (main paper) ...'); make_condensed_figure()
    print(f'\nAll figures saved to {PLOT_DIR}/')