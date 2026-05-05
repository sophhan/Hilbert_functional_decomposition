"""
Kernel Guidance: Illustrative Examples
==========================================================================

This script demonstrates how the choice of output kernel K affects the
interpretation of functional feature attributions. All computations are
analytical (oracle / true model) — no learning is involved.

Scientific context
------------------
In the H-FD framework, a scalar output kernel K(t, s) encodes which temporal
relationships the analyst considers relevant when aggregating or comparing
attribution curves f_S(t). Different kernels answer different questions:

  - Identity kernel:     pointwise attribution at each t independently
  - OU kernel:           attribution smoothed over a local neighbourhood of width ell
  - Causal kernel:       attribution that respects temporal causality (no future leakage)
  - Correlation kernel:  attribution aggregated according to the model's own output
                         covariance structure (data-adaptive, no bandwidth to set)
  - Gaussian kernel:     symmetric smoothing (useful but non-causal)
  - Periodic kernel:     attribution that ties together recurring phases across periods

Row-normalised kernel application is used throughout:
    (Kf)(t) = [K(t,:) @ f] / [sum_s K(t,s) * dt]

This ensures that attribution magnitudes remain comparable across kernels.

Three examples
--------------
Example 1 — ICU Early Warning (24h horizon)
    F(x)(t) = X1*exp(-0.2t) + X2*exp(-(t-10)^2/2) + X3*exp(-(t-18)^2/2)
    Kernels compared: Identity, OU (ell=4h), Correlation

Example 2 — Price Pulse Demand Response (4h horizon)
    F(x)(t) = X1*1_{[0.5,1.0)}(t) + X2*exp(-(t-2)^2/4.5) + X3
    Kernels compared: Identity, Causal (ell=0.33h), Gaussian (sigma=0.3h, marked wrong)
    Key message: Gaussian leaks attribution backwards in time before the pulse event.

Example 3 — Recurring Medication / Sleep Quality (72h, 3-day)
    F(x)(t) = X1*exp(-0.5*(t%24-8)^2/4) + X2*exp(-0.5*(t-20)^2/0.5)
    Kernels compared: Identity, OU (ell=4h), Periodic (p=24h)
    Key message: Periodic kernel recognises X1 as a recurring daily effect;
    OU conflates X2's acute day-1 event with subsequent X1 occurrences.

Each example produces a three-row figure:
    Row 0: Time-resolved attribution curves (Kf_S)(t) per kernel
    Row 1: Time-specific attribution at selected landmark time points
    Row 2: Time-aggregated scalar attribution integral(Kf_S)(t) dt

Figure 5 is a condensed 3x3 summary combining all three examples in a
single publication figure.

Outputs
-------
All figures saved to:
    plots/synthetic_experiments/kernel_guidance/

    fig1_icu_kernel_guidance.pdf
    fig2_pricepulse_kernel_guidance.pdf
    fig3_periodic_kernel_guidance.pdf
    fig5_condensed_kernel_guidance.pdf   — 3x3: resolved / specific / aggregated

Usage
-----
    python kernel_guidance.py
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

PLOT_DIR = os.path.join('plots', 'synthetic_experiments', 'kernel_guidance')
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global font sizes — large throughout, unified across figs 1/2/3
# Fig 5 (condensed) overrides these locally via plt.rc_context.
# ---------------------------------------------------------------------------

FS_SUPTITLE  = 26
FS_TITLE     = 23
FS_LABEL     = 20
FS_TICK      = 18
FS_LEGEND    = 17
FS_ANNOT     = 17
FS_ROW_LABEL = 19
FS_AGG_TITLE = 24

matplotlib.rcParams.update({
    'font.size':        FS_TICK,
    'axes.titlesize':   FS_TITLE,
    'axes.labelsize':   FS_LABEL,
    'xtick.labelsize':  FS_TICK,
    'ytick.labelsize':  FS_TICK,
    'legend.fontsize':  FS_LEGEND,
})

# ---------------------------------------------------------------------------
# Shared color constants
# C_ID / C_OU / ... are used in the full-page figures (figs 1-3).
# KC1 / KC2 / KC3 are the per-example kernel color triples used in fig 5.
# C_X1/X2/X3 are feature colors used across all figures.
# ---------------------------------------------------------------------------

C_ID   = '#888888'   # Identity kernel
C_OU   = '#E69F00'   # OU kernel
C_CORR = '#009E73'   # Correlation kernel
C_CAUS = '#2a9d8f'   # Causal kernel
C_GAUS = '#E35B1A'   # Gaussian kernel (marked "wrong" in Ex. 2)
C_PER  = '#f4a261'   # Periodic kernel
C_X1   = '#0072B2'   # Feature X1
C_X2   = '#D55E00'   # Feature X2
C_X3   = '#CC79A7'   # Feature X3

# Kernel color triples for the condensed fig 5
KC1 = ['#888888', '#E69F00', '#009E73']   # Ex. 1: Identity, OU, Correlation
KC2 = ['#888888', '#2a9d8f', '#E35B1A']   # Ex. 2: Identity, Causal, Gaussian
KC3 = ['#888888', '#3A86C8', '#f4a261']   # Ex. 3: Identity, OU, Periodic

# Line styles and hatch patterns for features within a single kernel panel
LS_FEATS    = ['-', '--', ':']
HATCH_FEATS = ['', '', '']

# ===========================================================================
# Kernel constructors
# Each returns a (T, T) matrix K where K[i, j] = K(t_i, t_j).
# ===========================================================================

def kernel_identity(t):
    """Identity kernel: K(t,s) = delta(t-s). No temporal coupling."""
    return np.eye(len(t))


def kernel_ou(t, ell):
    """
    Ornstein–Uhlenbeck (exponential) kernel: K(t,s) = exp(-|t-s| / ell).

    Induces local temporal smoothing with correlation length ell.
    Symmetric: couples t equally with past and future.
    Appropriate when the analyst wants a neighbourhood-averaged effect.
    """
    return np.exp(-np.abs(t[:, None] - t[None, :]) / ell)


def kernel_causal(t, ell):
    """
    Causal (one-sided exponential) kernel: K(t,s) = exp(-(t-s)/ell) if t>=s, else 0.

    Only couples t with its past — no future leakage.
    Corresponds to an AR(1) / exponential decay impulse response.
    Appropriate for systems where effects accumulate over past time.
    """
    d = t[:, None] - t[None, :]
    return np.where(d >= 0, np.exp(-d / ell), 0.0)


def kernel_gaussian(t, sigma):
    """
    Gaussian (RBF) kernel: K(t,s) = exp(-0.5 * ((t-s)/sigma)^2).

    Symmetric local smoothing with bandwidth sigma.
    NOTE: Being symmetric, this leaks attribution from future times backwards,
    which is inappropriate when the output process has causal structure.
    Marked as the 'wrong' choice in Example 2.
    """
    return np.exp(-0.5 * ((t[:, None] - t[None, :]) / sigma) ** 2)


def kernel_periodic(t, period, ell):
    """
    Periodic kernel: K(t,s) = exp(-2 * sin^2(pi*|t-s|/period) / ell^2).

    Couples time points that are a multiple of `period` apart.
    Appropriate when the output process has known periodicity (e.g. daily cycles).
    A recurring daily effect (X1) accumulates attribution across all periods;
    a one-off event (X2) does not benefit from cross-period coupling.
    """
    d = np.abs(t[:, None] - t[None, :])
    return np.exp(-2.0 * np.sin(np.pi * d / period) ** 2 / ell ** 2)


def kernel_correlation_icu(t):
    """
    Data-adaptive correlation kernel derived from the ICU model's output covariance.

    Constructs the model covariance C(t,s) = Var(X) * sum_j phi_j(t)*phi_j(s),
    then normalises to a correlation matrix: K(t,s) = C(t,s) / sqrt(C(t,t)*C(s,s)).

    This ties together times that co-vary across patients according to the model's
    own structure, without requiring the analyst to specify a bandwidth parameter.
    Appropriate when domain knowledge about temporal co-variation is encoded in the
    model rather than in a hand-tuned kernel.
    """
    VAR = 1.0 / 12.0   # Var(Xi) for Uniform[0,1]
    p1  = np.exp(-0.2 * t)
    p2  = np.exp(-0.5 * (t - 10.0) ** 2)
    p3  = np.exp(-0.5 * (t - 18.0) ** 2)
    C   = VAR * (np.outer(p1, p1) + np.outer(p2, p2) + np.outer(p3, p3))
    std = np.sqrt(np.diag(C))
    std = np.where(std < 1e-12, 1.0, std)   # avoid division by zero at tails
    return C / np.outer(std, std)

# ===========================================================================
# Kernel application and aggregation
# ===========================================================================

def apply_kernel(effect, K, dt):
    """
    Apply kernel K to a functional effect curve via row-normalised convolution.

        (Kf)(t) = [sum_s K(t,s) * f(s) * dt] / [sum_s K(t,s) * dt]

    Row normalisation ensures attribution magnitudes are comparable across
    kernels with different total mass. For the identity kernel, this reduces
    to the original effect curve unchanged.

    Parameters
    ----------
    effect : ndarray of shape (T,)  — pure effect f_S(t)
    K      : ndarray of shape (T, T)
    dt     : float — time step

    Returns
    -------
    ke : ndarray of shape (T,)  — kernel-smoothed effect (Kf_S)(t)
    """
    if np.allclose(K, np.eye(K.shape[0]), atol=1e-10):
        return effect.copy()
    rs = K.sum(axis=1, keepdims=True) * dt
    rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
    return (K / rs) @ effect * dt


def time_aggregated(effect, K, dt):
    """
    Compute the scalar time-aggregated attribution:

        Phi_S = integral (Kf_S)(t) dt

    Uses the trapezoidal rule (or a simple sum for the identity kernel).

    Parameters
    ----------
    effect : ndarray of shape (T,)
    K      : ndarray of shape (T, T)
    dt     : float

    Returns
    -------
    float
    """
    ke = apply_kernel(effect, K, dt)
    if np.allclose(K, np.eye(K.shape[0]), atol=1e-10):
        return float(np.sum(ke) * dt)
    return float(np.trapezoid(ke, dx=dt))


def relative_importance(effects_dict, K, dt):
    """
    Compute relative feature importance under kernel K:
        importance_S = |Phi_S| / sum_j |Phi_j|

    Returns dict mapping feature name -> float in [0, 1].
    Not used in the main figures but useful for diagnostic logging.
    """
    raw   = {fn: time_aggregated(eff, K, dt)
             for fn, eff in effects_dict.items()}
    total = sum(abs(v) for v in raw.values())
    if total < 1e-14:
        return {fn: 0.0 for fn in raw}
    return {fn: v / total for fn, v in raw.items()}

# ===========================================================================
# Shared plotting helpers
# ===========================================================================

def savefig(fig, name):
    """Save figure as PDF to PLOT_DIR and close it."""
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


def _spine(ax):
    """Remove top and right spines for a cleaner look."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _shade(ax, lo, hi, color, alpha=0.07):
    """Add a light background shading band between lo and hi."""
    ax.axvspan(lo, hi, alpha=alpha, color=color, zorder=0)

# ===========================================================================
# Shared three-row figure builder (used for figs 1–3)
# ===========================================================================

def build_figure(
    title, t, dt, effects, feat_colors, kernels, landmarks,
    wrong_markers=None, phase_bands=None, causal_vline=None,
    day_vlines=None, t_xlim=None, t_xticks=None, t_xticklabels=None,
    annotations=None,
    legend_loc_col0='upper right',
    row1_legend_col=0,
):
    """
    Build the standard three-row kernel guidance figure.

    Layout:
        Row 0 (cols = n_kernels): Time-resolved (Kf_S)(t) for each kernel
        Row 1 (cols = n_landmarks): Time-specific bar charts at landmark t values
        Row 2 (full width): Time-aggregated bar chart across all kernels

    Parameters
    ----------
    title           : str — figure suptitle
    t               : ndarray of shape (T,) — time grid
    dt              : float — time step
    effects         : dict str -> ndarray (T,) — pure effect per feature
    feat_colors     : dict str -> color — color per feature name
    kernels         : list of (label, K, color) tuples
    landmarks       : list of (t_value, label_str) for time-specific panels
    wrong_markers   : list of (kernel_label, feat_name, t_value) to mark with 'x'
                      Indicates a physically inappropriate kernel choice.
    phase_bands     : list of (t_lo, t_hi, color) for background shading
    causal_vline    : float or None — vertical line marking a causal boundary
    day_vlines      : list of floats — vertical dotted lines for day boundaries
    t_xlim          : (lo, hi) tuple for time axis limits
    t_xticks        : list of tick positions
    t_xticklabels   : list of tick labels
    annotations     : dict kernel_label -> str — text box annotations in row 0
    legend_loc_col0 : legend location for the leftmost time-resolved panel
    row1_legend_col : which time-specific column gets the kernel legend

    Returns
    -------
    fig : matplotlib Figure
    """
    n_k        = len(kernels)
    feat_names = list(effects.keys())
    n_feats    = len(feat_names)
    n_lm       = len(landmarks)
    n_cols_top = max(n_k, n_lm)

    fig = plt.figure(figsize=(5.5 * n_cols_top, 18))

    gs = gridspec.GridSpec(
        3, n_cols_top, figure=fig,
        hspace=0.55, wspace=0.32,
        left=0.07, right=0.97,
        top=0.88, bottom=0.05,
    )
    fig.suptitle(title, fontsize=FS_SUPTITLE, fontweight='bold', y=0.975)

    xl    = t_xlim   or (float(t.min()), float(t.max()))
    xtk   = t_xticks or list(np.arange(
        0, t.max() + 1e-6, max(1, int(t.max() / 6))
    ))
    xlabs = t_xticklabels or [str(int(v)) for v in xtk]

    # ------------------------------------------------------------------
    # Row 0: Time-resolved panels — one per kernel
    # ------------------------------------------------------------------
    for col, (k_label, K, k_color) in enumerate(kernels):
        ax     = fig.add_subplot(gs[0, col])
        curves = {fn: apply_kernel(eff, K, dt)
                  for fn, eff in effects.items()}
        ymax   = max(np.abs(c).max() for c in curves.values()) * 1.55
        ymin   = min(min(c.min() for c in curves.values()) * 1.10,
                     -0.02 * ymax)

        for fname, ke in curves.items():
            ax.plot(t, ke, color=feat_colors[fname], lw=2.4, label=fname)

        ax.axhline(0, color='gray', lw=0.6, ls=':')
        if causal_vline is not None:
            # Mark the causal boundary (e.g. onset of a pulse event)
            ax.axvline(causal_vline, color='#888', lw=0.9,
                       ls='--', alpha=0.6)
        if phase_bands:
            for lo, hi, pc in phase_bands:
                _shade(ax, lo, hi, pc)
        if day_vlines:
            for dv in day_vlines:
                ax.axvline(dv, color='gray', lw=0.8, ls=':', alpha=0.5)

        ax.set_xlim(*xl)
        ax.set_xticks(xtk)
        ax.set_xticklabels(xlabs, fontsize=FS_TICK)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=FS_TICK)
        _spine(ax)
        ax.set_title(k_label, fontsize=FS_TITLE, fontweight='bold',
                     color=k_color, pad=12)
        ax.set_xlabel('Time', fontsize=FS_LABEL)
        if col == 0:
            ax.set_ylabel(r'$f_S(t)$  /  $(Kf_S)(t)$', fontsize=FS_LABEL)
            ax.legend(fontsize=FS_LEGEND, loc=legend_loc_col0)
        else:
            ax.set_ylabel(r'$(Kf_S)(t)$', fontsize=FS_LABEL)
        # Optional annotation boxes explaining each kernel's interpretation
        if annotations and k_label in annotations:
            ax.text(0.97, 0.98, annotations[k_label],
                    transform=ax.transAxes, fontsize=FS_ANNOT,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.35',
                              fc='#fafafa', ec=k_color,
                              lw=1.5, alpha=0.95))
        if col == n_k - 1:
            ax.text(1.03, 0.5, 'Time-resolved',
                    transform=ax.transAxes, fontsize=FS_ROW_LABEL,
                    va='center', rotation=270, color='gray')

    # ------------------------------------------------------------------
    # Row 1: Time-specific panels — one per landmark
    # Bars show the kernel-smoothed effect at a single time point t0,
    # grouped by kernel. 'x' markers flag physically inappropriate kernels.
    # ------------------------------------------------------------------
    bar_w   = 0.75 / n_k
    offsets = np.linspace(-(n_k - 1) / 2, (n_k - 1) / 2, n_k) * bar_w

    for lm_col, (lm_t, lm_label) in enumerate(landmarks):
        ax  = fig.add_subplot(gs[1, lm_col])
        idx = np.argmin(np.abs(t - lm_t))

        # Scale y-axis consistently across all landmark panels
        all_lm  = [apply_kernel(eff, K, dt)[idx]
                   for _, K, _ in kernels for eff in effects.values()]
        lm_ymax = max(abs(v) for v in all_lm) * 1.30

        for k_idx, (k_label, K, k_color) in enumerate(kernels):
            vals  = [apply_kernel(effects[fn], K, dt)[idx]
                     for fn in feat_names]
            x_pos = np.arange(n_feats)
            ax.bar(x_pos + offsets[k_idx], vals,
                   width=bar_w * 0.88, color=k_color,
                   alpha=0.85, label=k_label)
            # Mark physically impossible kernel choices with a red 'x'
            if wrong_markers:
                for wk, wf, wt in wrong_markers:
                    if wk == k_label and abs(lm_t - wt) < 1e-6:
                        fi = feat_names.index(wf)
                        ax.plot(fi + offsets[k_idx],
                                vals[fi] + 0.02 * lm_ymax,
                                marker='x', color='#e63946',
                                ms=10, mew=2.5, zorder=10,
                                clip_on=False)

        ax.set_xticks(np.arange(n_feats))
        ax.set_xticklabels(feat_names, fontsize=FS_TICK)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.set_ylim(-0.05 * lm_ymax, lm_ymax)
        ax.set_title(f't = {lm_label}', fontsize=FS_TITLE,
                     fontweight='bold', pad=12)
        ax.tick_params(labelsize=FS_TICK)
        _spine(ax)
        if lm_col == 0:
            ax.set_ylabel(r'$(Kf_S)(t_0)$', fontsize=FS_LABEL)
        if lm_col == row1_legend_col:
            ax.legend(fontsize=FS_LEGEND, loc='upper right')
        if lm_col == n_lm - 1:
            ax.text(1.03, 0.5, 'Time-specific',
                    transform=ax.transAxes, fontsize=FS_ROW_LABEL,
                    va='center', rotation=270, color='gray')

    # ------------------------------------------------------------------
    # Row 2: Time-aggregated bar chart — spans all columns
    # Shows integral(Kf_S)(t) dt for each (feature, kernel) combination.
    # ------------------------------------------------------------------
    ax_agg = fig.add_subplot(gs[2, :])
    x_pos  = np.arange(n_feats)
    bar_w2 = 0.75 / n_k
    off2   = np.linspace(-(n_k - 1) / 2, (n_k - 1) / 2, n_k) * bar_w2

    for k_idx, (k_label, K, k_color) in enumerate(kernels):
        vals = [time_aggregated(effects[fn], K, dt) for fn in feat_names]
        ax_agg.bar(x_pos + off2[k_idx], vals,
                   width=bar_w2 * 0.88, color=k_color,
                   alpha=0.85, label=k_label)

    ax_agg.set_xticks(x_pos)
    ax_agg.set_xticklabels(feat_names, fontsize=FS_LABEL)
    ax_agg.axhline(0, color='gray', lw=0.6, ls=':')
    ax_agg.tick_params(labelsize=FS_TICK)
    _spine(ax_agg)
    ax_agg.set_ylabel(r'$\int(Kf_S)(t)\,dt$', fontsize=FS_LABEL)
    ax_agg.set_title(
        'Time-aggregated importance  '
        r'(ranking preserved across kernels)',
        fontsize=FS_AGG_TITLE, fontweight='bold', pad=12)
    ax_agg.legend(fontsize=FS_LEGEND, ncol=n_k, loc='upper right')
    ax_agg.text(1.01, 0.5, 'Time-aggregated',
                transform=ax_agg.transAxes, fontsize=FS_ROW_LABEL,
                va='center', rotation=270, color='gray')
    return fig

# ===========================================================================
# Fig 1 — ICU Early Warning (24h horizon)
# ===========================================================================

def make_icu_figure():
    """
    Generate fig1_icu_kernel_guidance.pdf.

    Demonstrates three kernels on the ICU additive model at x*=(0.8,0.9,0.7):
      - Identity: raw pointwise attribution
      - OU (ell=4h): neighbourhood-smoothed attribution
      - Correlation: data-adaptive, phase-aware attribution without bandwidth

    Key message: the correlation kernel automatically identifies that
    the shock peak (X2) and late deterioration (X3) occupy non-overlapping
    temporal regions, giving more interpretable aggregated importance scores.
    """
    T, TP = 24.0, 240
    t  = np.linspace(0, T, TP); dt = t[1] - t[0]; MU = 0.5

    # Analytical pure effects at x* = (0.8, 0.9, 0.7)
    effects = {
        'X1': (0.8 - MU) * np.exp(-0.2 * t),
        'X2': (0.9 - MU) * np.exp(-0.5 * (t - 10.0) ** 2),
        'X3': (0.7 - MU) * np.exp(-0.5 * (t - 18.0) ** 2),
    }
    kernels = [
        ('Identity',         kernel_identity(t),        C_ID),
        ('OU  ($\\ell=4$h)', kernel_ou(t, ell=4.0),     C_OU),
        ('Correlation',      kernel_correlation_icu(t), C_CORR),
    ]
    landmarks = [
        (0.0,  '0 h'),
        (5.0,  '5 h'),
        (10.0, '10 h\n(shock peak)'),
        (18.0, '18 h\n(detn. peak)'),
        (22.0, '22 h'),
    ]
    # Annotation boxes embedded in the time-resolved panels
    annotations = {
        'OU  ($\\ell=4$h)':
            'Spreads $X_2$ over $[6,14]$h\n'
            'Answers: "effect of $X_2$\n'
            r'over the shock neighbourhood"',
        'Correlation':
            'Aggregates $t_0$ with times\n'
            'that co-vary across patients\n'
            r'$\rightarrow$ phase-aware, no $\ell$ needed',
    }
    title = (
        'Example 1 — ICU Early Warning  (row-normalised kernel)\n'
        r'$F(\mathbf{x})(t)=X_1 e^{-0.2t}+X_2 e^{-(t-10)^2/2}+'
        r'X_3 e^{-(t-18)^2/2}$'
        r',  $\mathbf{x}^*=(0.8,\,0.9,\,0.7)$'
    )
    fig = build_figure(
        title=title, t=t, dt=dt,
        effects=effects,
        feat_colors={'X1': C_X1, 'X2': C_X2, 'X3': C_X3},
        kernels=kernels, landmarks=landmarks,
        phase_bands=[(8, 12, C_X2), (16, 20, C_X3)],
        annotations=annotations,
        t_xlim=(0, 24), t_xticks=list(range(0, 25, 4)),
        t_xticklabels=[str(v) for v in range(0, 25, 4)],
    )
    savefig(fig, 'fig1_icu_kernel_guidance.pdf')

# ===========================================================================
# Fig 2 — Price Pulse Demand Response (4h horizon)
# ===========================================================================

def make_pricepulse_figure():
    """
    Generate fig2_pricepulse_kernel_guidance.pdf.

    Demonstrates correct (causal) vs. incorrect (Gaussian) kernel choice
    for a model with a sharp causal event (price pulse in [0.5, 1.0)h).

    Key message: the Gaussian kernel is symmetric in time, so it produces
    nonzero attribution for X1 BEFORE the pulse starts (t < 0.5). This
    is physically impossible and would lead to incorrect interpretations.
    The causal kernel correctly assigns zero attribution before the pulse.
    """
    T, TP = 4.0, 480
    t  = np.linspace(0, T, TP); dt = t[1] - t[0]; MU = 0.5

    # X1 is a rectangular pulse; X2 is a broad temperature effect; X3 is constant
    effects = {
        'X1 (pulse)': (0.9 - MU) * ((t >= 0.5) & (t < 1.0)).astype(float),
        'X2 (temp.)': (0.7 - MU) * np.exp(-0.5 * ((t - 2.0) / 1.5) ** 2),
        'X3 (base)':  (0.6 - MU) * np.ones_like(t),
    }
    feat_colors = {
        'X1 (pulse)': C_X1, 'X2 (temp.)': C_X2, 'X3 (base)': C_X3
    }
    kernels = [
        ('Identity',                     kernel_identity(t),             C_ID),
        ('Causal  ($\\ell=0.33$h)',      kernel_causal(t, ell=0.33),    C_CAUS),
        ('Gaussian  ($\\sigma=0.3$h) ✗', kernel_gaussian(t, sigma=0.3), C_GAUS),
    ]
    landmarks = [
        (0.3,  '0.3 h\n(pre-pulse)'),
        (0.75, '0.75 h\n(in pulse)'),
        (1.5,  '1.5 h\n(post-pulse)'),
        (2.5,  '2.5 h\n(later)'),
    ]
    # Mark the Gaussian kernel at t=0.3 (pre-pulse) as physically impossible
    wrong_markers = [
        ('Gaussian  ($\\sigma=0.3$h) ✗', 'X1 (pulse)', 0.3)
    ]
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
    )
    fig = build_figure(
        title=title, t=t, dt=dt,
        effects=effects, feat_colors=feat_colors,
        kernels=kernels, landmarks=landmarks,
        wrong_markers=wrong_markers, causal_vline=0.5,
        annotations=annotations,
        t_xlim=(0, 4),
        t_xticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        t_xticklabels=['0','0.5','1','1.5','2','2.5','3','3.5','4'],
        row1_legend_col=1,
    )
    # Add pulse shading to all time-resolved panels
    for ax in fig.axes[:3]:
        ax.axvspan(0.5, 1.0, alpha=0.10, color=C_X1, zorder=0)

    # Shrink the row-1 legend (panel index 4) to avoid overlapping bars
    ax_r1_legend = fig.axes[3 + 1]
    leg = ax_r1_legend.get_legend()
    if leg is not None:
        leg.remove()
        ax_r1_legend.legend(fontsize=FS_LEGEND - 5, loc='upper right',
                            framealpha=0.9)

    # Aggregated panel (last axis): use single-column legend in upper-left
    ax_agg = fig.axes[-1]
    leg = ax_agg.get_legend()
    if leg is not None:
        leg.remove()
        ax_agg.legend(fontsize=FS_LEGEND, ncol=1, loc='upper left',
                      framealpha=0.9)

    savefig(fig, 'fig2_pricepulse_kernel_guidance.pdf')

# ===========================================================================
# Fig 3 — Periodic / Recurring Medication (72h, 3-day)
# ===========================================================================

def make_periodic_figure():
    """
    Generate fig3_periodic_kernel_guidance.pdf.

    Demonstrates three kernels for a model with a daily recurring effect (X1)
    and a one-off acute event (X2) over a 3-day window.

    Key message: the periodic kernel (p=24h) correctly recognises X1 as a
    recurring daily effect and accumulates its attribution across all three
    days. The OU kernel smears the acute X2 event into adjacent days,
    incorrectly suggesting it has multi-day influence. The identity kernel
    shows the raw per-time-point attribution without any coupling.
    """
    T, TP = 72.0, 720
    t  = np.linspace(0, T, TP); dt = t[1] - t[0]; MU = 0.5

    # X1 recurs every 24h (daily 8am medication); X2 is a one-off 8pm day-1 event
    effects = {
        'X1 (8am med.)':  (0.8 - MU) * np.exp(
            -0.5 * ((t % 24 - 8) ** 2) / 4.0),
        'X2 (day-1 8pm)': (0.9 - MU) * np.exp(
            -0.5 * (t - 20.0) ** 2 / 0.5),
    }
    feat_colors = {'X1 (8am med.)': C_X1, 'X2 (day-1 8pm)': C_X2}
    kernels = [
        ('Identity',           kernel_identity(t),            C_ID),
        ('OU  ($\\ell=4$h)',   kernel_ou(t, ell=4.0),         C_OU),
        ('Periodic ($p=24$h)', kernel_periodic(t, 24.0, 1.0), C_PER),
    ]
    landmarks = [
        (8.0,  'Day 1\n8 am'),
        (20.0, 'Day 1\n8 pm\n(acute)'),
        (32.0, 'Day 2\n8 am'),
        (44.0, 'Day 2\n8 pm'),
        (56.0, 'Day 3\n8 am'),
    ]
    # Mark the periodic kernel at Day 2 8pm as wrong: it incorrectly
    # couples X2's acute day-1 event with the Day 2 8pm slot
    wrong_markers = [('Periodic ($p=24$h)', 'X2 (day-1 8pm)', 44.0)]
    phase_bands = [
        (6, 10, C_X1), (18, 22, C_X2),
        (30, 34, C_X1), (42, 46, C_X2),
        (54, 58, C_X1), (66, 70, C_X2),
    ]

    feat_names = list(effects.keys())
    n_k        = len(kernels)
    n_lm       = len(landmarks)
    n_cols     = max(n_k, n_lm)

    fig = plt.figure(figsize=(5.5 * n_cols, 18))
    gs  = gridspec.GridSpec(
        3, n_cols, figure=fig,
        hspace=0.55, wspace=0.32,
        left=0.07, right=0.97, top=0.85, bottom=0.05,
    )
    fig.suptitle(
        'Example 3 — Sleep Quality: 3-Day Recurring Medication  '
        '(row-normalised kernel)\n'
        r'$X_1$: morning medication (8am, every day)  |  '
        r'$X_2$: acute evening event (8pm, day 1 only)'
        '\n'
        r'$\mathbf{x}^*=(0.8,\,0.9)$,  $T=[0,72]\,\mathrm{h}$  |  '
        'Identity | OU | Periodic ($p=24\\,\\mathrm{h}$, $\\ell=1$)',
        fontsize=FS_SUPTITLE, fontweight='bold', y=0.98,
    )

    xtk   = list(range(0, 73, 12))
    xlabs = [f'{v}h' for v in xtk]

    # --- Row 0: time-resolved ---
    for col, (k_label, K, k_color) in enumerate(kernels):
        ax     = fig.add_subplot(gs[0, col])
        curves = {fn: apply_kernel(eff, K, dt)
                  for fn, eff in effects.items()}
        ymax   = max(np.abs(c).max() for c in curves.values()) * 1.55
        ymin   = min(min(c.min() for c in curves.values()) * 1.10,
                     -0.02 * ymax)
        for fname, ke in curves.items():
            ax.plot(t, ke, color=feat_colors[fname], lw=2.4, label=fname)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        for pb in phase_bands:
            ax.axvspan(pb[0], pb[1], alpha=0.07, color=pb[2], zorder=0)
        for dv in [24.0, 48.0]:
            ax.axvline(dv, color='gray', lw=0.8, ls=':', alpha=0.5)
        ax.set_xlim(0, 72); ax.set_xticks(xtk)
        ax.set_xticklabels(xlabs, fontsize=FS_TICK)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=FS_TICK); _spine(ax)
        ax.set_title(k_label, fontsize=FS_TITLE, fontweight='bold',
                     color=k_color, pad=12)
        ax.set_xlabel('Time', fontsize=FS_LABEL)
        if col == 0:
            ax.set_ylabel(r'$(Kf_S)(t)$', fontsize=FS_LABEL)
            ax.legend(fontsize=FS_LEGEND, loc='upper right',
                      bbox_to_anchor=(1.0, 0.82))
        if col == n_k - 1:
            ax.text(1.03, 0.5, 'Time-resolved',
                    transform=ax.transAxes, fontsize=FS_ROW_LABEL,
                    va='center', rotation=270, color='gray')
        # Day labels centered within each 24h block
        for d, dlabel in [(12, 'Day 1'), (36, 'Day 2'), (60, 'Day 3')]:
            ax.text(d, ymax * 0.95, dlabel, ha='center',
                    fontsize=FS_TICK, color='gray', style='italic')

    # --- Row 1: time-specific ---
    bar_w   = 0.75 / n_k
    offsets = np.linspace(-(n_k - 1) / 2, (n_k - 1) / 2, n_k) * bar_w
    for lm_col, (lm_t, lm_label) in enumerate(landmarks):
        ax  = fig.add_subplot(gs[1, lm_col])
        idx = np.argmin(np.abs(t - lm_t))
        all_vals = [apply_kernel(eff, K, dt)[idx]
                    for _, K, _ in kernels
                    for eff in effects.values()]
        lm_ymax  = max(abs(v) for v in all_vals) * 1.30
        for k_idx, (k_label, K, k_color) in enumerate(kernels):
            vals  = [apply_kernel(effects[fn], K, dt)[idx]
                     for fn in feat_names]
            x_pos = np.arange(len(feat_names))
            ax.bar(x_pos + offsets[k_idx], vals,
                   width=bar_w * 0.88, color=k_color,
                   alpha=0.85, label=k_label)
            for wk, wf, wt in wrong_markers:
                if wk == k_label and abs(lm_t - wt) < 1e-6:
                    fi = feat_names.index(wf)
                    ax.plot(fi + offsets[k_idx],
                            vals[fi] + 0.02 * lm_ymax,
                            marker='x', color='#e63946',
                            ms=10, mew=2.5, zorder=10, clip_on=False)
        ax.set_xticks(np.arange(len(feat_names)))
        ax.set_xticklabels(feat_names, fontsize=FS_TICK)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.set_ylim(-0.05 * lm_ymax, lm_ymax)
        ax.set_title(f't = {lm_label}', fontsize=FS_TITLE,
                     fontweight='bold', pad=12)
        ax.tick_params(labelsize=FS_TICK); _spine(ax)
        if lm_col == 0:
            ax.set_ylabel(r'$(Kf_S)(t_0)$', fontsize=FS_LABEL)
            ax.legend(fontsize=FS_LEGEND, loc='upper right')
        if lm_col == n_lm - 1:
            ax.text(1.03, 0.5, 'Time-specific',
                    transform=ax.transAxes, fontsize=FS_ROW_LABEL,
                    va='center', rotation=270, color='gray')

    # --- Row 2: time-aggregated ---
    ax_agg = fig.add_subplot(gs[2, :])
    x_pos  = np.arange(len(feat_names))
    bar_w2 = 0.75 / n_k
    off2   = np.linspace(-(n_k - 1) / 2, (n_k - 1) / 2, n_k) * bar_w2
    for k_idx, (k_label, K, k_color) in enumerate(kernels):
        vals = [time_aggregated(effects[fn], K, dt) for fn in feat_names]
        ax_agg.bar(x_pos + off2[k_idx], vals,
                   width=bar_w2 * 0.88, color=k_color,
                   alpha=0.85, label=k_label)
    ax_agg.set_xticks(x_pos)
    ax_agg.set_xticklabels(feat_names, fontsize=FS_LABEL)
    ax_agg.axhline(0, color='gray', lw=0.6, ls=':')
    ax_agg.tick_params(labelsize=FS_TICK); _spine(ax_agg)
    ax_agg.set_ylabel(r'$\int(Kf_S)(t)\,dt$', fontsize=FS_LABEL)
    ax_agg.set_title('Time-aggregated importance',
                     fontsize=FS_AGG_TITLE, fontweight='bold', pad=12)
    ax_agg.legend(fontsize=FS_LEGEND, ncol=n_k, loc='upper right')
    ax_agg.text(1.01, 0.5, 'Time-aggregated',
                transform=ax_agg.transAxes, fontsize=FS_ROW_LABEL,
                va='center', rotation=270, color='gray')

    savefig(fig, 'fig3_periodic_kernel_guidance.pdf')

# ===========================================================================
# Fig 5 — Condensed 3×3 summary figure
# ===========================================================================

def make_condensed_figure():
    """
    Generate fig5_condensed_kernel_guidance.pdf.

    A compact 3x3 publication figure summarising all three kernel guidance
    examples in a single page:
        Rows:    Time-resolved | Time-specific | Time-aggregated
        Columns: Example 1 (ICU) | Example 2 (Price Pulse) | Example 3 (Periodic)

    Uses smaller font sizes than figs 1–3 (pinned via plt.rc_context so
    the global rcParams used by the full-page figures are not overwritten).

    Within the time-resolved panels, line style encodes the feature and
    color encodes the kernel (inverting the color/style convention of row 0
    in the full-page figures, where color = feature and line = shown once).
    This allows all kernels × features to be visible simultaneously without
    a combinatorially large legend.
    """
    # Pin fig 5 font sizes independently of the global FS_* constants
    FS_SUPTITLE  = 22
    FS_TITLE     = 19
    FS_LABEL     = 17
    FS_TICK      = 16
    FS_LEGEND    = 15
    FS_ANNOT     = 15
    FS_ROW_LABEL = 17
    FS_AGG_TITLE = 21

    rc_overrides = {
        'font.size':       FS_TICK,
        'axes.titlesize':  FS_TITLE,
        'axes.labelsize':  FS_LABEL,
        'xtick.labelsize': FS_TICK,
        'ytick.labelsize': FS_TICK,
        'legend.fontsize': FS_LEGEND,
    }

    with plt.rc_context(rc_overrides):
        _make_condensed_figure_body(
            FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK,
            FS_LEGEND, FS_ANNOT, FS_ROW_LABEL, FS_AGG_TITLE,
        )


def _make_condensed_figure_body(
    FS_SUPTITLE, FS_TITLE, FS_LABEL, FS_TICK,
    FS_LEGEND, FS_ANNOT, FS_ROW_LABEL, FS_AGG_TITLE,
):
    """
    Internal implementation of the condensed figure.
    Receives font size constants from make_condensed_figure to ensure
    they are applied consistently throughout without relying on globals.
    """
    MU = 0.5

    # ------------------------------------------------------------------
    # Data: replicate the three examples with their respective grids
    # ------------------------------------------------------------------

    # Example 1 — ICU (24h)
    T1, TP1 = 24.0, 240
    t1  = np.linspace(0, T1, TP1); dt1 = t1[1] - t1[0]
    eff1 = {
        'X1': (0.8 - MU) * np.exp(-0.2 * t1),
        'X2': (0.9 - MU) * np.exp(-0.5 * (t1 - 10.0) ** 2),
        'X3': (0.7 - MU) * np.exp(-0.5 * (t1 - 18.0) ** 2),
    }
    K1_id, K1_ou, K1_corr = (kernel_identity(t1),
                              kernel_ou(t1, ell=4.0),
                              kernel_correlation_icu(t1))
    # Landmark time points for time-specific panels
    lm1 = [(5.0,  r'$\mathbf{t{=}5\,h}$'),
            (10.0, r'$\mathbf{t{=}10\,h}$'),
            (22.0, r'$\mathbf{t{=}22\,h}$')]

    # Example 2 — Price Pulse (4h)
    T2, TP2 = 4.0, 480
    t2  = np.linspace(0, T2, TP2); dt2 = t2[1] - t2[0]
    eff2 = {
        'X1': (0.9 - MU) * ((t2 >= 0.5) & (t2 < 1.0)).astype(float),
        'X2': (0.7 - MU) * np.exp(-0.5 * ((t2 - 2.0) / 1.5) ** 2),
        'X3': (0.6 - MU) * np.ones_like(t2),
    }
    K2_id, K2_caus, K2_gaus = (kernel_identity(t2),
                                kernel_causal(t2, ell=0.33),
                                kernel_gaussian(t2, sigma=0.3))
    lm2 = [(0.3,  r'$\mathbf{t{=}0.3\,h}$'),
            (0.75, r'$\mathbf{t{=}0.75\,h}$'),
            (1.5,  r'$\mathbf{t{=}1.5\,h}$')]

    # Example 3 — Periodic (72h, 3-day)
    T3, TP3 = 72.0, 720
    t3  = np.linspace(0, T3, TP3); dt3 = t3[1] - t3[0]
    eff3 = {
        'X1': (0.8 - MU) * np.exp(-0.5 * ((t3 % 24 - 8) ** 2) / 4.0),
        'X2': (0.9 - MU) * np.exp(-0.5 * (t3 - 20.0) ** 2 / 0.5),
    }
    K3_id, K3_ou, K3_per = (kernel_identity(t3),
                             kernel_ou(t3, ell=4.0),
                             kernel_periodic(t3, 24.0, 1.0))
    lm3 = [(20.0, 'Day 1\n8pm'),
            (32.0, 'Day 2\n8am'),
            (44.0, 'Day 2\n8pm')]

    # Kernel triples: (label, matrix, color)
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
    # Figure layout: 3 rows × 3 columns
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(24, 13))
    fig.suptitle(
        'Kernel guidance: time-resolved, time-specific, and '
        'time-aggregated attribution',
        fontsize=FS_SUPTITLE + 2, fontweight='bold', y=1.01,
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.38, wspace=0.32,
        left=0.06, right=0.95,
        top=0.92, bottom=0.06,
        height_ratios=[0.6, 0.4, 0.4],
    )

    # ------------------------------------------------------------------
    # Inner helper: draw time-resolved panel (Row 0)
    # Color encodes kernel; line style encodes feature.
    # ------------------------------------------------------------------
    def draw_resolved(ax, t, dt, effects, kernels_info,
                      xlim, xticks, xticklabels,
                      phase_bands=None, day_vlines=None,
                      causal_vline=None):
        feat_names = list(effects.keys())
        # Pre-compute all (kernel, feature) smoothed curves
        curves     = {kl: {fn: apply_kernel(eff, K, dt)
                           for fn, eff in effects.items()}
                      for kl, K, _ in kernels_info}
        ymax = max(np.abs(c).max()
                   for kd in curves.values()
                   for c in kd.values()) * 1.28
        ymin = -0.05 * ymax
        if phase_bands:
            for lo, hi, pc in phase_bands:
                ax.axvspan(lo, hi, alpha=0.07, color=pc, zorder=0)
        if day_vlines:
            for dv in day_vlines:
                ax.axvline(dv, color='gray', lw=0.7, ls=':', alpha=0.5)
        if causal_vline is not None:
            ax.axvline(causal_vline, color='#aaa', lw=0.8,
                       ls='--', alpha=0.6)
        # Color = kernel, line style = feature
        for fi, fn in enumerate(feat_names):
            for kl, K, kc in kernels_info:
                ax.plot(t, curves[kl][fn], color=kc,
                        ls=LS_FEATS[fi], lw=2.4, alpha=0.92)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=FS_TICK)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=FS_TICK)
        _spine(ax)
        ax.set_xlabel('Time', fontsize=FS_LABEL)
        ax.set_ylabel(r'$(Kf_S)(t)$', fontsize=FS_LABEL)

    # ------------------------------------------------------------------
    # Inner helper: draw time-specific panel (Row 1)
    # Compact grouped bar chart: all landmarks × kernels × features
    # in a single axis, separated by vertical dividers.
    # ------------------------------------------------------------------
    def draw_specific(ax, t, dt, effects, kernels_info, landmarks,
                      wrong_markers=None):
        feat_names    = list(effects.keys())
        n_feats       = len(feat_names)
        n_k           = len(kernels_info)
        n_lm          = len(landmarks)

        bar_w         = 0.28
        group_width   = n_k * bar_w
        feat_spacing  = group_width + 0.15   # gap between adjacent features
        lm_gap        = 0.6                  # gap between landmark groups

        group_offsets = np.linspace(
            -(n_k - 1) / 2, (n_k - 1) / 2, n_k
        ) * bar_w

        # Determine consistent y scale across all landmarks and kernels
        all_vals = [
            apply_kernel(effects[fn], K, dt)[
                np.argmin(np.abs(t - lm_t))]
            for lm_t, _ in landmarks
            for _, K, _ in kernels_info
            for fn in feat_names
        ]
        ymax = max(abs(v) for v in all_vals) * 1.38

        xtick_pos, xtick_labels = [], []

        # Compute x positions for each landmark × feature group
        feat_xs = []
        for lm_idx in range(n_lm):
            x_origin = lm_idx * (n_feats * feat_spacing + lm_gap)
            feat_xs.append(np.array([x_origin + fi * feat_spacing
                                      for fi in range(n_feats)]))

        for lm_idx, (lm_t, lm_label) in enumerate(landmarks):
            idx    = np.argmin(np.abs(t - lm_t))
            feat_x = feat_xs[lm_idx]

            for k_idx, (kl, K, kc) in enumerate(kernels_info):
                vals = [apply_kernel(effects[fn], K, dt)[idx]
                        for fn in feat_names]
                for fi, (fn, v) in enumerate(zip(feat_names, vals)):
                    # Only add legend label for the first landmark/feature
                    ax.bar(feat_x[fi] + group_offsets[k_idx], v,
                           width=bar_w * 0.92, color=kc, alpha=0.88,
                           label=(kl if (lm_idx == 0 and fi == 0)
                                  else '_'))
                if wrong_markers:
                    for wk, wf, wt in wrong_markers:
                        if wk == kl and abs(lm_t - wt) < 1e-6:
                            fi = feat_names.index(wf)
                            ax.plot(
                                feat_x[fi] + group_offsets[k_idx],
                                vals[fi] + 0.02 * ymax,
                                marker='x', color='#e63946',
                                ms=10, mew=2.5, zorder=10,
                                clip_on=False)

            # Landmark label centered above the group
            block_centre = feat_x[0] + (n_feats - 1) / 2 * feat_spacing
            ax.text(block_centre, ymax * 0.98,
                    lm_label, ha='center', va='top',
                    fontsize=FS_TICK, fontweight='bold', color='#333333')

            # Vertical divider between landmark groups
            if lm_idx < n_lm - 1:
                right_edge = (feat_xs[lm_idx][-1]
                              + group_offsets[-1]
                              + bar_w * 0.92 / 2)
                left_edge  = (feat_xs[lm_idx + 1][0]
                              + group_offsets[0]
                              - bar_w * 0.92 / 2)
                divider_x  = (right_edge + left_edge) / 2
                ax.axvline(divider_x, color='#cccccc', lw=1.0, ls='-')

            for fi, fn in enumerate(feat_names):
                xtick_pos.append(feat_x[fi])
                xtick_labels.append(fn)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, fontsize=FS_TICK)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_ylim(-0.08 * ymax, ymax)
        ax.tick_params(labelsize=FS_TICK)
        _spine(ax)
        ax.set_ylabel(r'$(Kf_S)(t_0)$', fontsize=FS_LABEL)
        ax.set_title('Time-specific',
                     fontsize=FS_AGG_TITLE, fontweight='bold', pad=12)

    # ------------------------------------------------------------------
    # Inner helper: draw time-aggregated bar chart (Row 2)
    # ------------------------------------------------------------------
    def draw_aggregated(ax, effects, kernels_info, dt):
        feat_names = list(effects.keys())
        n_k        = len(kernels_info)
        bar_w      = 0.90 / n_k
        offsets    = np.linspace(
            -(n_k - 1) / 2, (n_k - 1) / 2, n_k
        ) * bar_w
        x_pos      = np.arange(len(feat_names))

        for k_idx, (kl, K, kc) in enumerate(kernels_info):
            vals = [time_aggregated(effects[fn], K, dt)
                    for fn in feat_names]
            ax.bar(x_pos + offsets[k_idx], vals,
                   width=bar_w * 0.94, color=kc, alpha=0.88, label=kl)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(feat_names, fontsize=FS_LABEL)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.tick_params(labelsize=FS_TICK)
        _spine(ax)
        ax.set_ylabel(r'$\int(Kf_S)(t)\,dt$', fontsize=FS_LABEL)
        ax.legend(fontsize=FS_LEGEND, loc='upper right')

    # ------------------------------------------------------------------
    # Legend builders for the condensed figure
    # ------------------------------------------------------------------
    def make_bar_handles(kernels_info):
        """Patch handles for kernel bars (used in specific and aggregated rows)."""
        return [mpatches.Patch(facecolor=kc, alpha=0.88, label=kl)
                for kl, _, kc in kernels_info]

    # ==================================================================
    # Column 0 — ICU Early Warning
    # ==================================================================
    ax_r1 = fig.add_subplot(gs[0, 0])
    draw_resolved(ax_r1, t1, dt1, eff1, kernels1,
                  xlim=(0, 24), xticks=list(range(0, 25, 4)),
                  xticklabels=[str(v) for v in range(0, 25, 4)],
                  phase_bands=[(8, 12, KC1[1]), (16, 20, KC1[2])])
    ax_r1.set_title(
        'Example 1 — ICU Early Warning\n'
        r'$X_1 e^{-0.2t}+X_2 e^{-(t-10)^2/2}+X_3 e^{-(t-18)^2/2}$',
        fontsize=FS_TITLE, fontweight='bold', pad=12)
    # Two separate legends: kernel colors (upper right) and feature line styles (upper left)
    kernel_handles1 = [mlines.Line2D([], [], color=kc, lw=2.6, ls='-', label=kl)
                       for kl, _, kc in kernels1]
    feat_handles1   = [mlines.Line2D([], [], color='#333333', ls=LS_FEATS[i],
                                     lw=2.2, label=fl)
                       for i, fl in enumerate(['X1 (baseline)', 'X2 (shock)', 'X3 (detn.)'])]
    leg_kernels1 = ax_r1.legend(handles=kernel_handles1,
                                fontsize=FS_LEGEND, loc='upper right',
                                bbox_to_anchor=(1.0, 0.88), framealpha=0.88)
    ax_r1.add_artist(leg_kernels1)
    ax_r1.legend(handles=feat_handles1,
                 fontsize=FS_LEGEND, loc='upper left',
                 bbox_to_anchor=(0.0, 0.88), framealpha=0.88)

    ax_s1 = fig.add_subplot(gs[1, 0])
    draw_specific(ax_s1, t1, dt1, eff1, kernels1, lm1)
    ax_s1.legend(handles=make_bar_handles(kernels1),
                 fontsize=FS_LEGEND, loc='center left', framealpha=0.88)

    ax_a1 = fig.add_subplot(gs[2, 0])
    draw_aggregated(ax_a1, eff1, kernels1, dt1)
    ax_a1.set_title('Time-aggregated',
                    fontsize=FS_AGG_TITLE, fontweight='bold', pad=12)
    ax_a1.legend(handles=make_bar_handles(kernels1),
                 fontsize=FS_LEGEND, loc='upper right', framealpha=0.88)

    # ==================================================================
    # Column 1 — Price Pulse Demand Response
    # ==================================================================
    ax_r2 = fig.add_subplot(gs[0, 1])
    draw_resolved(ax_r2, t2, dt2, eff2, kernels2,
                  xlim=(0, 4),
                  xticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                  xticklabels=['0','0.5','1','1.5','2','2.5','3','3.5','4'],
                  causal_vline=0.5)
    # Add pulse shading (X1 active region)
    ax_r2.axvspan(0.5, 1.0, alpha=0.10, color=KC2[0], zorder=0)
    ax_r2.set_title(
        'Example 2 — Price Pulse\n'
        r'$X_1\mathbf{1}_{[0.5,1.0)}(t)+X_2 e^{-(t-2)^2/4.5}+X_3$',
        fontsize=FS_TITLE, fontweight='bold', pad=12)
    kernel_handles2 = [mlines.Line2D([], [], color=kc, lw=2.6, ls='-', label=kl)
                       for kl, _, kc in kernels2]
    feat_handles2   = [mlines.Line2D([], [], color='#333333', ls=LS_FEATS[i],
                                     lw=2.2, label=fl)
                       for i, fl in enumerate(['X1 (pulse)', 'X2 (temp.)', 'X3 (base)'])]
    leg_kernels2 = ax_r2.legend(handles=kernel_handles2,
                                fontsize=FS_LEGEND, loc='upper right',
                                bbox_to_anchor=(1.0, 0.88), framealpha=0.88)
    ax_r2.add_artist(leg_kernels2)
    ax_r2.legend(handles=feat_handles2,
                 fontsize=FS_LEGEND, loc='upper center',
                 bbox_to_anchor=(0.5, 0.88), framealpha=0.88)

    ax_s2 = fig.add_subplot(gs[1, 1])
    draw_specific(ax_s2, t2, dt2, eff2, kernels2, lm2)
    ax_s2.legend(handles=make_bar_handles(kernels2),
                 fontsize=FS_LEGEND, loc='center left', framealpha=0.88)

    ax_a2 = fig.add_subplot(gs[2, 1])
    draw_aggregated(ax_a2, eff2, kernels2, dt2)
    ax_a2.set_title('Time-aggregated',
                    fontsize=FS_AGG_TITLE, fontweight='bold', pad=12)
    ax_a2.legend(handles=make_bar_handles(kernels2),
                 fontsize=FS_LEGEND, loc='upper left', framealpha=0.88)

    # ==================================================================
    # Column 2 — Periodic / Recurring Medication
    # ==================================================================
    ax_r3 = fig.add_subplot(gs[0, 2])
    draw_resolved(ax_r3, t3, dt3, eff3, kernels3,
                  xlim=(0, 72), xticks=list(range(0, 73, 12)),
                  xticklabels=[f'{v}h' for v in range(0, 73, 12)],
                  phase_bands=[
                      (6, 10, KC3[0]), (18, 22, KC3[0]),
                      (30, 34, KC3[0]), (42, 46, KC3[0]),
                      (54, 58, KC3[0]), (66, 70, KC3[0])],
                  day_vlines=[24.0, 48.0])
    ax_r3.set_title(
        'Example 3 — Periodic (3-day Medication)\n'
        r'$X_1$: daily 8am (recurring),  $X_2$: acute 8pm day 1',
        fontsize=FS_TITLE, fontweight='bold', pad=12)
    # Day labels
    ymax3 = ax_r3.get_ylim()[1]
    for d, dl in [(12, 'Day 1'), (36, 'Day 2'), (60, 'Day 3')]:
        ax_r3.text(d, ymax3 * 0.97, dl, ha='center', va='top',
                   fontsize=FS_TICK, color='gray', style='italic')
    kernel_handles3 = [mlines.Line2D([], [], color=kc, lw=2.6, ls='-', label=kl)
                       for kl, _, kc in kernels3]
    feat_handles3   = [mlines.Line2D([], [], color='#333333', ls=LS_FEATS[i],
                                     lw=2.2, label=fl)
                       for i, fl in enumerate(['X1 (8am daily)', 'X2 (8pm day-1)'])]
    leg_kernels3 = ax_r3.legend(handles=kernel_handles3,
                                fontsize=FS_LEGEND, loc='upper right',
                                bbox_to_anchor=(1.0, 0.88), framealpha=0.88)
    ax_r3.add_artist(leg_kernels3)
    ax_r3.legend(handles=feat_handles3,
                 fontsize=FS_LEGEND, loc='upper center',
                 bbox_to_anchor=(0.5, 0.88), framealpha=0.88)

    ax_s3 = fig.add_subplot(gs[1, 2])
    draw_specific(ax_s3, t3, dt3, eff3, kernels3, lm3,
                  wrong_markers=[('Periodic', 'X2', 44.0)])
    ax_s3.legend(handles=make_bar_handles(kernels3),
                 fontsize=FS_LEGEND, loc='center right', framealpha=0.88)

    ax_a3 = fig.add_subplot(gs[2, 2])
    draw_aggregated(ax_a3, eff3, kernels3, dt3)
    ax_a3.set_title('Time-aggregated',
                    fontsize=FS_AGG_TITLE, fontweight='bold', pad=12)
    ax_a3.legend(handles=make_bar_handles(kernels3),
                 fontsize=FS_LEGEND, loc='upper right', framealpha=0.88)

    # Row labels on right edge of the rightmost column
    for ax, label in [
        (ax_r3, 'Time-resolved'),
        (ax_s3, 'Time-specific'),
        (ax_a3, 'Time-aggregated'),
    ]:
        ax.text(1.03, 0.5, label,
                transform=ax.transAxes, fontsize=FS_ROW_LABEL,
                va='center', rotation=270, color='gray')

    savefig(fig, 'fig5_condensed_kernel_guidance.pdf')


# ===========================================================================
# Main entry point
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('  Kernel Guidance')
    print('=' * 60)
    print('\n[1] ICU Early Warning ...');             make_icu_figure()
    print('\n[2] Price Pulse Demand Response ...');   make_pricepulse_figure()
    print('\n[3] Periodic Recurring Medication ...'); make_periodic_figure()
    print('\n[4] Condensed 3x3 summary ...');         make_condensed_figure()
    print(f'\nAll figures saved to {PLOT_DIR}/')