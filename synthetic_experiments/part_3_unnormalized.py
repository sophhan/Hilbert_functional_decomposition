"""
Part 3 — Kernel Guidance: Effect of Kernel Choice on Temporal Aggregation
==========================================================================
All computations are analytical (oracle / true model).

Kernel application — UNNORMALISED throughout:
  (Kf)(t) = int K(t,s) f(s) ds

Identity kernel special case:
  In continuous theory: int delta(t-s) f(s) ds = f(t)  exactly.
  The discrete approximation K_id = I gives I @ f * dt = f * dt,
  introducing a spurious dt factor.  We handle identity analytically:
    apply_kernel(f, K_id, dt)  ->  f          (raw pure effect)
    time_aggregated(f, K_id, dt) ->  sum(f)*dt  (standard integral)
  All other kernels use the standard discrete formula K @ f * dt.

Time-resolved plots use per-panel y-limits (each kernel fills its panel).

Time-aggregated row:
  Left:  amplification ratio  Phi_S(K) / Phi_S(K_id)  (identity = 1)
  Right: relative importance  Phi_S / sum_j |Phi_j|

Examples:
  1 — ICU early warning     Kernels: Identity | OU (ell=4h) | Correlation
  2 — Price Pulse            Kernels: Identity | Causal | Gaussian (wrong)
  3 — Periodic (3-day)       Kernels: Identity | OU (ell=4h) | Periodic

Output: Hilbert_functional_decomposition/plots/synthetic_experiments/part_3/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PLOT_DIR = os.path.join(
    'Hilbert_functional_decomposition',
    'plots', 'synthetic_experiments', 'part_3_unnormalized',
)
os.makedirs(PLOT_DIR, exist_ok=True)

C_ID   = '#888888'
C_OU   = '#457b9d'
C_CORR = '#8338ec'
C_CAUS = '#2a9d8f'
C_GAUS = '#e63946'
C_PER  = '#f4a261'
C_X1   = '#c1121f'
C_X2   = '#2a9d8f'
C_X3   = '#e9c46a'


# ===========================================================================
# 1.  Kernel constructors
# ===========================================================================

def kernel_identity(t):    return np.eye(len(t))
def kernel_ou(t, ell):     return np.exp(-np.abs(t[:,None]-t[None,:])/ell)
def kernel_causal(t, ell):
    d = t[:,None]-t[None,:]
    return np.where(d>=0, np.exp(-d/ell), 0.0)
def kernel_gaussian(t, sigma):
    return np.exp(-0.5*((t[:,None]-t[None,:])/sigma)**2)
def kernel_periodic(t, period, ell):
    d = np.abs(t[:,None]-t[None,:])
    return np.exp(-2.0*np.sin(np.pi*d/period)**2/ell**2)
def kernel_correlation_icu(t):
    VAR=1.0/12.0
    p1=np.exp(-0.2*t); p2=np.exp(-0.5*(t-10.0)**2); p3=np.exp(-0.5*(t-18.0)**2)
    C=VAR*(np.outer(p1,p1)+np.outer(p2,p2)+np.outer(p3,p3))
    std=np.sqrt(np.diag(C)); std=np.where(std<1e-12,1.0,std)
    return C/np.outer(std,std)


# ===========================================================================
# 2.  Kernel application
# ===========================================================================

def _is_identity(K):
    """True if K is numerically the identity matrix."""
    return np.allclose(K, np.eye(K.shape[0]), atol=1e-10)


def apply_kernel(effect, K, dt):
    """
    (Kf)(t) = int K(t,s) f(s) ds  — unnormalised.

    Identity kernel special case: returns f(t) directly, matching the
    continuous-theory result int delta(t-s) f(s) ds = f(t) and avoiding
    the spurious dt factor from the discrete approximation I @ f * dt.
    """
    if _is_identity(K):
        return effect.copy()
    return K @ effect * dt


def time_aggregated(effect, K, dt):
    """
    Phi_S = int (Kf_S)(t) dt.

    Identity: sum(f) * dt  =  trapezoidal integral of the raw effect.
    Other K:  sum(K @ f) * dt  (single dt — outer integral only).
    """
    if _is_identity(K):
        return float(np.sum(effect) * dt)
    return float(np.sum(K @ effect) * dt)


def relative_importance(effects_dict, K, dt):
    """Phi_S / sum_j |Phi_j| — redistribution independent of scale."""
    raw   = {fn: time_aggregated(eff, K, dt) for fn, eff in effects_dict.items()}
    total = sum(abs(v) for v in raw.values())
    if total < 1e-14:
        return {fn: 0.0 for fn in raw}
    return {fn: v/total for fn, v in raw.items()}


def amplification_ratio(effects_dict, K, K_id, dt):
    """
    Phi_S(K) / Phi_S(K_id).  Always 1.0 for the identity kernel.
    Differences across features reveal redistribution of importance.
    """
    id_vals = {fn: time_aggregated(eff, K_id, dt) for fn, eff in effects_dict.items()}
    k_vals  = {fn: time_aggregated(eff, K,    dt) for fn, eff in effects_dict.items()}
    return {
        fn: k_vals[fn]/id_vals[fn] if abs(id_vals[fn]) > 1e-14 else 0.0
        for fn in effects_dict
    }


# ===========================================================================
# 3.  Plotting helpers
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
# 4.  Figure builder
# ===========================================================================

def build_figure(
    title, t, dt, effects, feat_colors, kernels, landmarks,
    wrong_markers=None, phase_bands=None, causal_vline=None,
    day_vlines=None, t_xlim=None, t_xticks=None, t_xticklabels=None,
    annotations=None,
):
    """
    Row 0 — Time-resolved  : one panel per kernel, per-panel y-limits.
                             Identity panel shows raw pure effects f_S(t).
    Row 1 — Time-specific  : grouped bars at each landmark.
    Row 2 — Time-aggregated: left  = amplification ratio vs identity;
                             right = relative importance (normalised).
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
    xtk   = t_xticks  or list(np.arange(0, t.max()+1e-6, max(1, int(t.max()/6))))
    xlabs = t_xticklabels or [str(int(v)) for v in xtk]
    K_id  = kernels[0][1]   # identity is always first

    # ------------------------------------------------------------------
    # Row 0: time-resolved, per-panel y-limits
    # ------------------------------------------------------------------
    for col, (k_label, K, k_color) in enumerate(kernels):
        ax     = fig.add_subplot(gs[0, col])
        curves = {fn: apply_kernel(eff, K, dt) for fn, eff in effects.items()}
        ymax   = max(np.abs(c).max() for c in curves.values()) * 1.30
        ymin   = min(min(c.min() for c in curves.values()) * 1.10, -0.02*ymax)

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
            ax.set_ylabel(r'$f_S(t)$  (pure effect)', fontsize=9)
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
            vals  = [apply_kernel(effects[fn], K, dt)[idx] for fn in feat_names]
            x_pos = np.arange(n_feats)
            ax.bar(x_pos+offsets[k_idx], vals,
                   width=bar_w*0.88, color=k_color, alpha=0.85, label=k_label)
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
    # Row 2: time-aggregated — amplification ratio + relative importance
    # ------------------------------------------------------------------
    n_half = n_cols_top // 2
    ax_amp = fig.add_subplot(gs[2, :n_half])
    ax_rel = fig.add_subplot(gs[2, n_half:])

    x_pos  = np.arange(n_feats)
    bar_w2 = 0.75/n_k
    off2   = np.linspace(-(n_k-1)/2, (n_k-1)/2, n_k)*bar_w2

    for k_idx, (k_label, K, k_color) in enumerate(kernels):
        amp_dict = amplification_ratio(effects, K, K_id, dt)
        rel_dict = relative_importance(effects, K, dt)
        ax_amp.bar(x_pos+off2[k_idx], [amp_dict[fn] for fn in feat_names],
                   width=bar_w2*0.88, color=k_color, alpha=0.85, label=k_label)
        ax_rel.bar(x_pos+off2[k_idx], [rel_dict[fn] for fn in feat_names],
                   width=bar_w2*0.88, color=k_color, alpha=0.85, label=k_label)

    ax_amp.axhline(1.0, color='#e63946', lw=1.0, ls='--', alpha=0.7,
                   label='Identity (=1)')

    for ax, ylabel, subtitle in [
        (ax_amp,
         r'$\Phi_S(K)\,/\,\Phi_S(K_\mathrm{id})$  (amplification)',
         'Amplification ratio vs identity\n'
         r'(identity $\equiv$ 1.0 — differences across features = redistribution)'),
        (ax_rel,
         r'Relative importance  $\Phi_S\,/\,\sum_j|\Phi_j|$',
         'Relative importance (normalised within kernel)\n'
         '(redistribution between features, independent of scale)'),
    ]:
        ax.set_xticks(x_pos); ax.set_xticklabels(feat_names, fontsize=9)
        ax.axhline(0, color='gray', lw=0.6, ls=':')
        ax.tick_params(labelsize=8); _spine(ax)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(subtitle, fontsize=8.5)
        ax.legend(fontsize=7.5, loc='upper right')

    ax_rel.text(1.01, 0.5, 'Time-aggregated',
                transform=ax_rel.transAxes, fontsize=9,
                va='center', rotation=270, color='gray')
    return fig


# ===========================================================================
# 5.  Example 1 — ICU
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
        ('Identity',         kernel_identity(t),          C_ID),
        ('OU  ($\\ell=4$h)', kernel_ou(t, ell=4.0),       C_OU),
        ('Correlation',      kernel_correlation_icu(t),    C_CORR),
    ]
    landmarks = [
        (0.0,'0 h'), (5.0,'5 h'),
        (10.0,'10 h\n(shock peak)'),
        (18.0,'18 h\n(detn. peak)'), (22.0,'22 h'),
    ]
    annotations = {
        'OU  ($\\ell=4$h)':
            'Spreads $X_2$ over $[6,14]$h\n'
            'Boundary features amplified\n'
            'less than interior features\n'
            r'(column sums vary with $s$)',
        'Correlation':
            'Strongly amplifies $X_1$\n'
            '(baseline recovery coupled\n'
            'globally via off-diagonal\n'
            'correlation blocks)\n'
            '$X_2$/$X_3$: within-phase only',
    }
    title = (
        'Example 1 — ICU Early Warning  (unnormalised kernel)\n'
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
# 6.  Example 2 — Price Pulse
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
            '(AR(1) dynamics)\n'
            'Amplif. $X_1$ > 1 (lingering)',
        'Gaussian  ($\\sigma=0.3$h) ✗':
            '✗ Nonzero BEFORE pulse\n'
            '  (anticipates future event)\n'
            'Amplif. $X_1$ > 1\n'
            'but for the WRONG reason\n'
            '  (spurious anticipation)',
    }
    title = (
        'Example 2 — Price Pulse Demand Response  (unnormalised kernel)\n'
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
# 7.  Example 3 — Periodic
# ===========================================================================

def make_periodic_figure():
    T, TP = 72.0, 720
    t  = np.linspace(0, T, TP); dt = t[1]-t[0]; MU = 0.5
    effects = {
        'X1 (daily med.)':   (0.8-MU)*np.exp(-0.5*((t%24-8)**2)/4.0),
        'X2 (day-1 stress)': (0.9-MU)*np.exp(-0.5*(t-10.0)**2),
    }
    feat_colors = {'X1 (daily med.)':C_X1,'X2 (day-1 stress)':C_X2}
    kernels = [
        ('Identity',           kernel_identity(t),             C_ID),
        ('OU  ($\\ell=4$h)',   kernel_ou(t, ell=4.0),          C_OU),
        ('Periodic ($p=24$h)', kernel_periodic(t, 24.0, 4.0),  C_PER),
    ]
    landmarks = [
        (8.0,'Day 1\n8 am'), (10.0,'Day 1\n10 am\n(acute)'),
        (32.0,'Day 2\n8 am'),(56.0,'Day 3\n8 am'),
    ]
    wrong_markers = [
        ('Periodic ($p=24$h)','X2 (day-1 stress)',32.0),
        ('Periodic ($p=24$h)','X2 (day-1 stress)',56.0),
    ]
    annotations = {
        'OU  ($\\ell=4$h)':
            'Smooths locally ($\\ell=4$h)\n'
            'Cannot bridge 24h gap\n'
            '$X_2\\approx 0$ on days 2 & 3\n'
            '(correct)',
        'Periodic ($p=24$h)':
            '✓ $X_1$: same curve every day\n'
            '  (phase-stable — correct)\n\n'
            '✗ $X_2$: nonzero on days 2 & 3\n'
            '  (imports day-1 event)\n'
            '  Amplif. $X_2$ inflated $\\sim$3×\n'
            '  (use only if feature is\n'
            '   known to be periodic)',
    }
    title = (
        'Example 3 — Sleep Quality: 3-Day Recurring Medication  '
        '(unnormalised kernel)\n'
        r'$F(\mathbf{x})(t)=X_1\,\phi_\text{daily}(t)+'
        r'X_2\,e^{-(t-10)^2/2}$'
        r',  $\mathbf{x}^*=(0.8,\,0.9)$,  $T=[0,72]$h'
        '\nIdentity (raw effects) | OU (local smoothing) | '
        'Periodic (phase-stable for $X_1$, leaks $X_2$)'
    )
    fig = build_figure(
        title=title, t=t, dt=dt,
        effects=effects, feat_colors=feat_colors,
        kernels=kernels, landmarks=landmarks,
        wrong_markers=wrong_markers,
        phase_bands=[(6,10,C_X1),(30,34,C_X1),(54,58,C_X1)],
        day_vlines=[24.0,48.0], annotations=annotations,
        t_xlim=(0,72), t_xticks=list(range(0,73,12)),
        t_xticklabels=[f'{v}h' for v in range(0,73,12)],
    )
    for ax in fig.axes[:3]:
        for d, label in [(12,'Day 1'),(36,'Day 2'),(60,'Day 3')]:
            ax.text(d, ax.get_ylim()[1]*0.95, label,
                    ha='center', fontsize=7, color='gray', style='italic')
    savefig(fig, 'fig3_periodic_kernel_guidance.pdf')


# ===========================================================================
# 8.  Fig 4 — Ranking summary
# ===========================================================================

def make_ranking_summary():
    MU = 0.5

    T, TP = 24.0,240; t1=np.linspace(0,T,TP);  dt1=t1[1]-t1[0]
    eff_icu = {'X1':(0.8-MU)*np.exp(-0.2*t1),
               'X2':(0.9-MU)*np.exp(-0.5*(t1-10)**2),
               'X3':(0.7-MU)*np.exp(-0.5*(t1-18)**2)}
    icu_k   = [('Identity',   kernel_identity(t1),          C_ID),
               ('OU ℓ=4h',    kernel_ou(t1,4.0),            C_OU),
               ('Correlation',kernel_correlation_icu(t1),   C_CORR)]

    T2,TP2=4.0,480; t2=np.linspace(0,T2,TP2); dt2=t2[1]-t2[0]
    eff_pp  = {'X1\n(pulse)':(0.9-MU)*((t2>=0.5)&(t2<1.0)).astype(float),
               'X2\n(temp.)':(0.7-MU)*np.exp(-0.5*((t2-2)/1.5)**2),
               'X3\n(base)': (0.6-MU)*np.ones_like(t2)}
    pp_k    = [('Identity',      kernel_identity(t2),         C_ID),
               ('Causal',        kernel_causal(t2,0.33),      C_CAUS),
               ('Gaussian\n(✗)', kernel_gaussian(t2,0.3),     C_GAUS)]

    T3,TP3=72.0,720; t3=np.linspace(0,T3,TP3); dt3=t3[1]-t3[0]
    eff_per = {'X1\n(daily)':(0.8-MU)*np.exp(-0.5*((t3%24-8)**2)/4),
               'X2\n(day-1)':(0.9-MU)*np.exp(-0.5*(t3-10)**2)}
    per_k   = [('Identity',       kernel_identity(t3),            C_ID),
               ('OU ℓ=4h',        kernel_ou(t3,4.0),              C_OU),
               ('Periodic p=24h', kernel_periodic(t3,24.0,4.0),   C_PER)]

    examples = [
        ('ICU Early Warning',  eff_icu, icu_k, dt1),
        ('Price Pulse',        eff_pp,  pp_k,  dt2),
        ('Periodic (3-day)',   eff_per, per_k,  dt3),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        r'Time-aggregated importance — all examples and kernels''\n'
        'Top: amplification ratio vs identity  '
        r'($\Phi_S(K)/\Phi_S(K_\mathrm{id})$, identity $\equiv$ 1.0).  '
        'Bottom: relative importance (normalised within kernel).',
        fontsize=10, fontweight='bold',
    )

    for col, (ex_title, effects, ex_kernels, ex_dt) in enumerate(examples):
        feat_names = list(effects.keys())
        n_feats    = len(feat_names)
        n_k        = len(ex_kernels)
        bar_w      = 0.75/n_k
        offsets    = np.linspace(-(n_k-1)/2,(n_k-1)/2,n_k)*bar_w
        x_pos      = np.arange(n_feats)
        K_id       = ex_kernels[0][1]

        for k_idx, (k_label, K, k_color) in enumerate(ex_kernels):
            amp_dict = amplification_ratio(effects, K, K_id, ex_dt)
            rel_dict = relative_importance(effects, K, ex_dt)
            axes[0,col].bar(x_pos+offsets[k_idx],
                            [amp_dict[fn] for fn in feat_names],
                            width=bar_w*0.88, color=k_color,
                            alpha=0.85, label=k_label)
            axes[1,col].bar(x_pos+offsets[k_idx],
                            [rel_dict[fn] for fn in feat_names],
                            width=bar_w*0.88, color=k_color,
                            alpha=0.85, label=k_label)

        axes[0,col].axhline(1.0, color='#e63946', lw=1.0,
                            ls='--', alpha=0.7, label='Identity (=1)')

        for row, (ax, ylabel) in enumerate([
            (axes[0,col], r'$\Phi_S(K)/\Phi_S(K_\mathrm{id})$'),
            (axes[1,col], 'Relative importance'),
        ]):
            ax.set_xticks(x_pos); ax.set_xticklabels(feat_names, fontsize=8)
            ax.axhline(0, color='gray', lw=0.6, ls=':')
            ax.tick_params(labelsize=8); _spine(ax)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=7, loc='upper right')

        axes[0,col].set_title(ex_title, fontsize=10, fontweight='bold')

        print(f'\n  {ex_title}:')
        for k_label, K, _ in ex_kernels:
            rel = relative_importance(effects, K, ex_dt)
            amp = amplification_ratio(effects, K, K_id, ex_dt)
            ranked = sorted(feat_names, key=lambda f: -rel[f])
            s = '  '.join(
                f'{f}(rel={rel[f]:.3f}, amp={amp[f]:.1f}x)' for f in ranked)
            print(f'    {k_label:20s}: {s}')

    plt.tight_layout()
    savefig(fig, 'fig4_ranking_summary.pdf')


# ===========================================================================
# 9.  Main
# ===========================================================================

if __name__ == '__main__':
    print('\n' + '='*60)
    print('  Part 3 — Kernel Guidance (identity special-cased)')
    print('='*60)
    print('\n[1] ICU ...');        make_icu_figure()
    print('\n[2] Price Pulse ...'); make_pricepulse_figure()
    print('\n[3] Periodic ...');   make_periodic_figure()
    print('\n[4] Summary ...');    make_ranking_summary()
    print(f'\nAll figures saved to {PLOT_DIR}/')