"""
Test script for fig0_main_body_v2 layout changes (v3).

Changes tested:
  1. Row-slice panels same size and square as correlation heatmaps
  2. Light shaded background boxes for IHEPC (teal) and NESO (orange)
  3. AM/PM peak shading in partial panels AND row-slice panels (fixed)
  4. Larger font sizes
  5. Row 1 height further reduced
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Import shared infrastructure from main script
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from energy_demand_all_games import (
    load_ihepc, load_neso,
    RFModel, kernel_identity, kernel_correlation, apply_kernel,
    FunctionalGame, moebius_transform, shapley_values,
    _pure_effects_e, _full_effects_e,
    GAME_TYPES, FEAT_COLORS, DS_COLOR, DS_LABEL, FEAT_ABBR,
    RNG_SEED,
)

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
FS_SUP  = 16
FS_T    = 13
FS_AX   = 12
FS_TK   = 10.5
FS_LEG  = 10.5
FS_RLAB = 13

ID_ALPHA = 0.45
ID_LW    = 1.8
MX_LW    = 2.4

# AM/PM shading — stronger alpha so it's clearly visible
SHADE_AM_COLOR = '#4a90e2'
SHADE_PM_COLOR = '#e24a4a'
SHADE_ALPHA    = 0.28   # stronger shading

_NODE_POS = '#2a9d8f'
_NODE_NEG = '#e63946'
_EDGE_SYN = '#2a9d8f'
_EDGE_RED = '#e63946'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xticks(ax, ds, sparse=False):
    T, tl = ds['T'], ds['tlabels']
    step  = max(1, T // 6) * (2 if sparse else 1)
    idxs  = list(range(0, T, step))
    ax.set_xticks(idxs)
    ax.set_xticklabels([tl[i] for i in idxs], rotation=45,
                       ha='right', fontsize=FS_TK)
    ax.set_xlim(-0.5, T - 0.5)

def _shade(ax, ds):
    """
    AM/PM shading. Called AFTER all plot elements so it overlays them with
    low alpha — this guarantees visibility regardless of draw order.
    """
    ax.axvspan(*ds['morning'], alpha=SHADE_ALPHA, color=SHADE_AM_COLOR,
               zorder=10, lw=0)
    ax.axvspan(*ds['evening'], alpha=SHADE_ALPHA, color=SHADE_PM_COLOR,
               zorder=10, lw=0)

def _network_importances(mob, shap, p, T, K):
    t_grid   = np.arange(T, dtype=float)
    node_imp = np.array([float(np.sum(np.abs(apply_kernel(shap[i], K))))
                         for i in range(p)])
    node_sgn = np.array([np.sign(float(np.trapz(apply_kernel(shap[i], K), t_grid)))
                         for i in range(p)])
    edge_imp = {}
    for i in range(p):
        for j in range(i+1, p):
            raw = np.zeros(T)
            for S, m in mob.items():
                if i in S and j in S:
                    raw += m / len(S)
            val = float(np.trapz(apply_kernel(raw, K), t_grid))
            if abs(val) > 0:
                edge_imp[(i, j)] = val
    return node_imp, edge_imp, node_sgn

def _draw_network(ax, features, node_imp, edge_imp, node_sign, title):
    import math
    p     = len(features)
    angle = [math.pi/2 - 2*math.pi*i/p for i in range(p)]
    pos   = {i: (math.cos(a), math.sin(a)) for i, a in enumerate(angle)}
    ax.set_aspect('equal'); ax.axis('off')
    if title:
        ax.set_title(title, fontsize=FS_T, fontweight='bold', pad=4)
    max_imp  = max(float(node_imp.max()), 1e-9)
    max_edge = max((abs(v) for v in edge_imp.values()), default=1e-9)
    for (i, j), val in edge_imp.items():
        xi, yi = pos[i]; xj, yj = pos[j]
        lw  = 0.4 + 6.5 * abs(val) / max_edge
        col = _EDGE_SYN if val > 0 else _EDGE_RED
        ax.plot([xi, xj], [yi, yj], color=col, lw=lw,
                alpha=0.3 + 0.6*abs(val)/max_edge,
                solid_capstyle='round', zorder=1)
    for i in range(p):
        x, y = pos[i]
        r  = 0.07 + 0.19 * (node_imp[i] / max_imp)
        fc = _NODE_POS if node_sign[i] >= 0 else _NODE_NEG
        ax.add_patch(plt.Circle((x, y), r, color=fc, ec='white',
                                linewidth=1.2, zorder=2, alpha=0.88))
        ax.add_patch(plt.Circle((x, y), r*0.52, color='white',
                                ec='none', zorder=3, alpha=0.95))
        abbr = FEAT_ABBR.get(features[i], features[i][:3])
        ax.text(x, y, abbr, ha='center', va='center',
                fontsize=max(5.0, r*22), fontweight='bold',
                color='#222', zorder=4)
    pad = 0.32
    ax.set_xlim(-1-pad, 1+pad); ax.set_ylim(-1-pad, 1+pad)

def _add_bg_box(fig, axes_list, color, pad=0.014):
    """Draw a lightly tinted rounded rectangle behind a group of axes."""
    renderer = fig.canvas.get_renderer()
    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    for ax in axes_list:
        bb     = ax.get_window_extent(renderer=renderer)
        bb_fig = bb.transformed(fig.transFigure.inverted())
        xmins.append(bb_fig.x0); ymins.append(bb_fig.y0)
        xmaxs.append(bb_fig.x1); ymaxs.append(bb_fig.y1)
    x0 = min(xmins) - pad
    y0 = min(ymins) - pad
    w  = max(xmaxs) - x0 + pad
    h  = max(ymaxs) - y0 + pad
    rect = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle='round,pad=0.006',
        linewidth=1.4, edgecolor=color,
        facecolor=color, alpha=0.10,
        transform=fig.transFigure,
        zorder=0, clip_on=False)
    fig.add_artist(rect)

# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def fig0_test(ds_ih, ds_ne, mob_ih, shap_ih, mob_ne, shap_ne, K_ih, K_ne):

    K_id_ih = kernel_identity(ds_ih['T'])
    K_id_ne = kernel_identity(ds_ne['T'])

    # Reduced figure height (was 9.0)
    fig = plt.figure(figsize=(26, 8.2))

    # Row 1 ratio reduced: was 1.05, now 0.80
    gs_outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.0, 0.80],
        hspace=0.48,
        top=0.88, bottom=0.14,
        left=0.04, right=0.98,
    )

    # Row 0: 6 columns — 2 network panels | IHEPC pair | NESO pair | 2 network panels
    # The heatmap+rowslice pairs use nested gridspecs so wspace controls the gap
    # between heatmap and row-slice independently of network spacing.
    gs0 = GridSpecFromSubplotSpec(
        1, 6, subplot_spec=gs_outer[0],
        wspace=0.28,
        width_ratios=[0.85, 0.85, 2.2, 2.2, 0.85, 0.85],
    )
    ax_net_ih_sens = fig.add_subplot(gs0[0])
    ax_net_ih_pred = fig.add_subplot(gs0[1])
    ax_net_ne_sens = fig.add_subplot(gs0[4])
    ax_net_ne_pred = fig.add_subplot(gs0[5])

    # IHEPC heatmap + rowslice: nested 1x2 — wspace controls gap between them
    gs_ih_pair = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs0[2], wspace=0.90)
    ax_ih_heat = fig.add_subplot(gs_ih_pair[0])
    ax_ih_row  = fig.add_subplot(gs_ih_pair[1])

    # NESO heatmap + rowslice: same structure
    gs_ne_pair = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs0[3], wspace=0.90)
    ax_ne_heat = fig.add_subplot(gs_ne_pair[0])
    ax_ne_row  = fig.add_subplot(gs_ne_pair[1])

    # Row 1
    gs1 = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[1], wspace=0.32)
    ax_ih_part = fig.add_subplot(gs1[0])
    ax_ih_gap  = fig.add_subplot(gs1[1])
    ax_ne_gap  = fig.add_subplot(gs1[2])
    ax_ne_part = fig.add_subplot(gs1[3])

    fig.suptitle(
        'Energy demand: correlation structure drives explanation shape  '
        '[layout test — v3]\n'
        'UCI IHEPC (single household, kW)  vs  NESO GB Demand (national grid, MW)',
        fontsize=FS_SUP, fontweight='bold', y=0.975)

    # ── Heatmaps ──────────────────────────────────────────────────────────
    def _heatmap(ax, ds, K, tag):
        T, tl = ds['T'], ds['tlabels']
        step  = max(1, T // 6)
        ticks = list(range(0, T, step))
        im = ax.imshow(K, aspect='equal', origin='upper',
                       cmap='RdBu_r', vmin=-0.2, vmax=1.0)
        ax.set_xticks(ticks)
        ax.set_xticklabels([tl[i] for i in ticks], rotation=45,
                           ha='right', fontsize=7.0)
        ax.set_yticks(ticks)
        ax.set_yticklabels([tl[i] for i in ticks], fontsize=6.5)
        ax.set_title(DS_LABEL[tag].replace('\n', ' ') +
                     '\ncorrelation kernel $K$',
                     fontsize=FS_AX, fontweight='bold', color=DS_COLOR[tag])
        am = (ds['morning'][0] + ds['morning'][1]) // 2
        ax.axhline(am, color='white', lw=0.8, ls='--', alpha=0.6)
        ax.axvline(am, color='white', lw=0.8, ls='--', alpha=0.6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(
            labelsize=6.5)

    # ── Row-slice ─────────────────────────────────────────────────────────
    def _rowslice(ax, ds, K, tag):
        T, tl            = ds['T'], ds['tlabels']
        morning, evening = ds['morning'], ds['evening']
        am    = (morning[0] + morning[1]) // 2
        step  = max(1, T // 6)
        ticks = list(range(0, T, step))

        ax.plot(np.arange(T), K[am, :], color=DS_COLOR[tag], lw=2.2, zorder=2)
        ax.axhline(0, color='gray', lw=0.5, ls=':', zorder=1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([tl[i] for i in ticks], rotation=45,
                           ha='right', fontsize=7.0)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylabel('$K(t_{\\mathrm{AM}}, s)$', fontsize=FS_AX)
        ax.set_xlabel('Time $s$', fontsize=FS_AX)
        ax.tick_params(labelsize=FS_TK)

        # Shorter than heatmap — 0.50 makes it half as tall as wide
        ax.set_box_aspect(0.50)

        title = ('Structured (AM$\\leftrightarrow$PM)'
                 if tag == 'ihepc' else 'Uniform (regime-dominated)')
        ax.set_title(title, fontsize=FS_AX, color=DS_COLOR[tag],
                     fontweight='bold')

        # AM/PM shading — called LAST so it sits on top of plot lines
        _shade(ax, ds)

    _heatmap( ax_ih_heat, ds_ih, K_ih, 'ihepc')
    _rowslice(ax_ih_row,  ds_ih, K_ih, 'ihepc')
    _heatmap( ax_ne_heat, ds_ne, K_ne, 'neso')
    _rowslice(ax_ne_row,  ds_ne, K_ne, 'neso')

    # ── Network panels ─────────────────────────────────────────────────────
    net_handles = [
        Patch(facecolor=_NODE_POS, edgecolor='none', label='Positive'),
        Patch(facecolor=_NODE_NEG, edgecolor='none', label='Negative'),
    ]
    for ax, mob, shap, ds, title in [
        (ax_net_ih_sens, mob_ih['sensitivity'], shap_ih['sensitivity'],
         ds_ih, 'IHEPC sens.\npartial (corr)'),
        (ax_net_ih_pred, mob_ih['prediction'],  shap_ih['prediction'],
         ds_ih, 'IHEPC pred.\npartial (corr)'),
        (ax_net_ne_sens, mob_ne['sensitivity'], shap_ne['sensitivity'],
         ds_ne, 'NESO sens.\npartial (corr)'),
        (ax_net_ne_pred, mob_ne['prediction'],  shap_ne['prediction'],
         ds_ne, 'NESO pred.\npartial (corr)'),
    ]:
        features = ds['features']
        p, T     = len(features), ds['T']
        K        = K_ih if ds['tag'] == 'ihepc' else K_ne
        ni, ei, ns = _network_importances(mob, shap, p, T, K)
        _draw_network(ax, features, ni, ei, ns, title)
        ax.legend(handles=net_handles, loc='lower center', ncol=2,
                  fontsize=FS_LEG - 1.5, framealpha=0.88,
                  bbox_to_anchor=(0.5, -0.18), bbox_transform=ax.transAxes,
                  borderpad=0.4, handlelength=1.2)

    # ── Partial panels ────────────────────────────────────────────────────
    def _partial_panel(ax, ds, shap, K_id, K_corr, tag, force_features):
        features = ds['features']
        p, T     = len(features), ds['T']
        partial  = shap['prediction']
        top2     = [features.index(f) for f in force_features]

        for fi in top2:
            col = FEAT_COLORS[features[fi]]
            ls  = '-' if fi == top2[0] else '--'
            ax.plot(ds['t_grid'], apply_kernel(partial[fi], K_id),
                    color=col, lw=ID_LW, ls=ls, alpha=ID_ALPHA, zorder=2)
            ax.plot(ds['t_grid'], apply_kernel(partial[fi], K_corr),
                    color=col, lw=MX_LW, ls=ls,
                    label=features[fi] + ' (corr.)', zorder=3)

        ax.axhline(0, color='gray', lw=0.5, ls=':', zorder=1)
        _xticks(ax, ds, sparse=True)
        ax.tick_params(labelsize=FS_TK)
        ax.set_xlabel('Time', fontsize=FS_AX)
        ax.set_ylabel(ds['ylabel']['prediction'], fontsize=FS_AX)
        ax.set_title('Partial  $\\phi_i \\equiv$ SHAP\n' +
                     DS_LABEL[tag].split('\n')[0],
                     fontsize=FS_T, fontweight='bold', color=DS_COLOR[tag])
        extra = Line2D([0], [0], color='gray', lw=ID_LW, ls='-',
                       alpha=ID_ALPHA, label='identity (faded)')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [extra], labels + ['identity (faded)'],
                  fontsize=FS_LEG, loc='upper center',
                  bbox_to_anchor=(0.5, -0.22), ncol=3, framealpha=0.85)

        # AM/PM shading — called LAST so it overlays plot lines
        _shade(ax, ds)

    _partial_panel(ax_ih_part, ds_ih, shap_ih, K_id_ih, K_ih,
                   'ihepc', ['month', 'lag_morning'])
    _partial_panel(ax_ne_part, ds_ne, shap_ne, K_id_ne, K_ne,
                   'neso', ['season', 'month'])

    # ── Sensitivity gap panels ────────────────────────────────────────────
    def _gap_panel(ax, ds, mob_sens, K_corr, tag):
        features     = ds['features']
        p, T, t_grid = len(features), ds['T'], ds['t_grid']
        pure_eff = _pure_effects_e(mob_sens, p, T)
        full_eff = _full_effects_e(mob_sens, p, T)
        gap      = {i: full_eff[i] - pure_eff[i] for i in range(p)}
        gap_imp  = {i: float(np.sum(np.abs(apply_kernel(gap[i], K_corr))))
                    for i in range(p)}
        fi  = max(gap_imp, key=gap_imp.get)
        col = FEAT_COLORS[features[fi]]

        pure_c = apply_kernel(pure_eff[fi], K_corr)
        full_c = apply_kernel(full_eff[fi], K_corr)
        gap_c  = apply_kernel(gap[fi],      K_corr)

        ax.fill_between(t_grid, pure_c, full_c, color=col, alpha=0.20,
                        label='gap region', zorder=1)
        ax.plot(t_grid, full_c, color=col, lw=2.0, ls='-',
                label=r'Full $\bar{\tau}_i$ (Total Sobol)', zorder=3)
        ax.plot(t_grid, pure_c, color=col, lw=2.0, ls='--',
                label=r'Pure $\tau^{\mathrm{cl}}_i$ (Closed Sobol)', zorder=3)
        ax.plot(t_grid, gap_c, color='black', lw=1.4, ls=':', alpha=0.7,
                label=r'Gap $\Delta\tau_i$', zorder=3)
        ax.axhline(0, color='gray', lw=0.5, ls=':', zorder=1)
        _xticks(ax, ds, sparse=True)
        ax.tick_params(labelsize=FS_TK)
        ax.set_xlabel('Time', fontsize=FS_AX)
        ax.set_ylabel(ds['ylabel']['sensitivity'], fontsize=FS_AX)
        integ = float(np.trapz(np.abs(apply_kernel(gap[fi], K_corr)), t_grid))
        ax.set_title(
            'Sensitivity gap  —  corr. kernel\n'
            '{} — {}  $\\int|\\Delta\\tau_i|\\,dt = {:.3g}$'.format(
                DS_LABEL[tag].split('\n')[0], features[fi], integ),
            fontsize=FS_T - 1, fontweight='bold', color=DS_COLOR[tag])
        ax.legend(fontsize=FS_LEG, loc='upper center',
                  bbox_to_anchor=(0.5, -0.22), ncol=2, framealpha=0.85)

        # AM/PM shading — called LAST
        _shade(ax, ds)

    _gap_panel(ax_ih_gap, ds_ih, mob_ih['sensitivity'], K_ih, 'ihepc')
    _gap_panel(ax_ne_gap, ds_ne, mob_ne['sensitivity'], K_ne, 'neso')

    # ── Margin labels ─────────────────────────────────────────────────────
    for ax, label, tag, x, ha in [
        (ax_net_ih_sens, 'IHEPC', 'ihepc', -0.14, 'right'),
        (ax_net_ne_pred, 'NESO',  'neso',   1.14, 'left'),
        (ax_ih_part,     'IHEPC', 'ihepc', -0.20, 'right'),
        (ax_ne_part,     'NESO',  'neso',   1.20, 'left'),
    ]:
        ax.text(x, 0.5, label, transform=ax.transAxes,
                fontsize=FS_RLAB, va='center', ha=ha, rotation=90,
                color=DS_COLOR[tag], fontweight='bold')

    # Background boxes — draw canvas, then add overlays
    fig.canvas.draw()
    _add_bg_box(fig,
                [ax_net_ih_sens, ax_net_ih_pred,
                 ax_ih_heat, ax_ih_row,
                 ax_ih_part, ax_ih_gap],
                DS_COLOR['ihepc'], pad=0.012)
    _add_bg_box(fig,
                [ax_ne_heat, ax_ne_row,
                 ax_net_ne_sens, ax_net_ne_pred,
                 ax_ne_gap, ax_ne_part],
                DS_COLOR['neso'], pad=0.012)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Loading data ...')
    ds_ih = load_ihepc()
    ds_ne = load_neso()

    print('Fitting models ...')
    for ds, name in [(ds_ih, 'IHEPC'), (ds_ne, 'NESO')]:
        X_tr, X_te, Y_tr, Y_te = train_test_split(
            ds['X_np'], ds['Y_adj'], test_size=0.2, random_state=RNG_SEED)
        m = RFModel(); m.fit(X_tr, Y_tr)
        r2 = m.evaluate(X_te, Y_te)
        print(f'  [{name}] Test R²: {r2:.4f}')
        ds['model'] = m

    print('Building kernels ...')
    K_ih = kernel_correlation(ds_ih['Y_raw'])
    K_ne = kernel_correlation(ds_ne['Y_raw'])

    print('Selecting profiles ...')
    X_ih = ds_ih['X_np']; fn_ih = ds_ih['features']
    X_ne = ds_ne['X_np']; fn_ne = ds_ne['features']

    def _find(X, features, conds):
        mask = np.ones(len(X), dtype=bool)
        for f, (lo, hi) in conds.items():
            ci = features.index(f)
            mask &= (X[:, ci] >= lo) & (X[:, ci] <= hi)
        hits = X[mask]
        if not len(hits): raise RuntimeError(f'No match: {conds}')
        return hits[len(hits) // 2]

    def _y(ds, xp):
        diffs = np.abs(ds['X_np'] - xp[None, :]).sum(axis=1)
        return ds['Y_adj'][int(np.argmin(diffs))]

    x_ih = _find(X_ih, fn_ih,
                 {'is_weekend': (-0.1, 0.1), 'day_of_week': (0.9, 4.1)})
    x_ne = _find(X_ne, fn_ne,
                 {'is_weekend': (-0.1, 0.1), 'season': (0.9, 1.1)})

    print('Running games ...')
    def _run(ds, x_exp, y_obs):
        results = {}
        for gt in GAME_TYPES:
            game = FunctionalGame(
                predict_fn=ds['model'].predict, X_bg=ds['X_np'],
                x_exp=x_exp, T=ds['T'], features=ds['features'],
                game_type=gt, Y_obs=y_obs,
                sample_size=ds['sample'][gt], random_seed=RNG_SEED)
            game.precompute()
            mob  = moebius_transform(game)
            shap = shapley_values(mob, game.p, game.T)
            results[gt] = (mob, shap)
        return results

    ih_res = _run(ds_ih, x_ih, _y(ds_ih, x_ih))
    ne_res = _run(ds_ne, x_ne, _y(ds_ne, x_ne))

    mob_ih  = {gt: ih_res[gt][0] for gt in GAME_TYPES}
    shap_ih = {gt: ih_res[gt][1] for gt in GAME_TYPES}
    mob_ne  = {gt: ne_res[gt][0] for gt in GAME_TYPES}
    shap_ne = {gt: ne_res[gt][1] for gt in GAME_TYPES}

    print('Generating figure ...')
    fig = fig0_test(ds_ih, ds_ne, mob_ih, shap_ih, mob_ne, shap_ne, K_ih, K_ne)
    fig.savefig('fig0_test_v3.pdf', bbox_inches='tight', dpi=150)
    print('Saved: fig0_test_v3.pdf')
    plt.close(fig)