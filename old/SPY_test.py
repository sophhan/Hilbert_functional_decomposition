"""
Minimal height-tuning test for fig0_main_body_summary_v2.
Imports real data + real computed effects from the main script.
Adjust ROW_HEIGHT to find the sweet spot.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# ── Point to your main script ─────────────────────────────────────────────
MAIN_SCRIPT_DIR = '/home/slangbei/Hilbert_functional_decomposition/SPY'
sys.path.insert(0, MAIN_SCRIPT_DIR)

from SPY_all_games import (
    load_and_aggregate,
    RFModel,
    FunctionalGame,
    functional_moebius_transform,
    shapley_from_moebius,
    DAY_FEATURE_NAMES,
    SAMPLE_SIZE,
    RNG_SEED,
    T_BARS,
    t_grid,
    _pure_effects,
    _full_effects,
    apply_kernel,
    kernel_identity,
    kernel_ou,
    kernel_causal,
    _scale,
    _set_time_axis,
    _period_shade,
    _ann_vline,
    GAME_YLABEL,
    FEAT_COLORS,
    FS_SUPTITLE,
    FS_TITLE,
    FS_AXIS,
    FS_TICK,
    FS_LEGEND,
    FS_ANNOT,
)

# ── Tunable ───────────────────────────────────────────────────────────────
ROW_HEIGHT = 2.8
OUT_PATH   = os.path.join(MAIN_SCRIPT_DIR, 'test_fig0_height.pdf')
# ─────────────────────────────────────────────────────────────────────────


def plot_fig0_real(moebius_hv, shapley_hv, pnames, row_height=ROW_HEIGHT):
    n_players = len(pnames)
    fi_vix    = pnames.index('vix_prev')
    fi_ann    = pnames.index('ann_indicator')

    K_id     = kernel_identity(t_grid)
    K_ou     = kernel_ou(t_grid,     length_scale=8.0)
    K_causal = kernel_causal(t_grid, length_scale=8.0)

    def K_mixed(fi):
        return K_causal if pnames[fi] == 'ann_indicator' else K_ou

    pure_pred    = _pure_effects(moebius_hv['prediction'], n_players)
    partial_pred = shapley_hv['prediction']
    pure_risk    = _pure_effects(moebius_hv['risk'],       n_players)
    partial_risk = shapley_hv['risk']

    def _int_curve(game_type, kern):
        raw = moebius_hv[game_type].get(
            (fi_vix, fi_ann), np.zeros(T_BARS))
        return apply_kernel(raw, kern)

    row_specs = [
        ('prediction', GAME_YLABEL['prediction'],
         r'Pure  $m_i$  $\equiv$  local PDP',
         r'Partial  $\phi_i$  $\equiv$  SHAP',
         pure_pred, partial_pred, lambda fi: K_id),
        ('prediction', GAME_YLABEL['prediction'],
         r'Pure  $m_i$  $\equiv$  local PDP  (mixed kernel)',
         r'Partial  $\phi_i$  $\equiv$  SHAP  (mixed kernel)',
         pure_pred, partial_pred, K_mixed),
        ('risk', GAME_YLABEL['risk'],
         r'Pure  $m_i$  $\equiv$  local pure risk  (mixed kernel)',
         r'Partial  $\phi_i$  $\equiv$  local SAGE-style  (mixed kernel)',
         pure_risk, partial_risk, K_mixed),
    ]
    row_labels = ['Prediction\n(Identity)', 'Prediction\n(Mixed)', 'Risk\n(Mixed)']

    c_vix = FEAT_COLORS['vix_prev']
    c_ann = FEAT_COLORS['ann_indicator']

    # Slightly reduced title font size
    title_fs = FS_TITLE - 1

    # Per-row annotation settings for col 1:
    #   r=0 → upper left   (y=0.97, va='top')
    #   r=1 → center left  (y=0.50, va='center')
    #   r=2 → upper left   (y=0.97, va='top')
    annot_col1 = {
        0: (0.97, 'top'),
        1: (0.50, 'center'),
        2: (0.97, 'top'),
    }

    n_rows = len(row_specs)
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(19, row_height * n_rows),
        gridspec_kw={'width_ratios': [3, 3, 3, 1.8]},
    )
    fig.suptitle(
        'High-VIX Announcement Profile (local): '
        'Kernel Choice, Game Type and Effect Decomposition',
        fontsize=FS_SUPTITLE, fontweight='bold',
    )

    for r, (gtype, y_label, lbl_pure, lbl_partial,
            pure_eff, partial_eff, kern_fn) in enumerate(row_specs):
        sc = _scale(gtype)

        # ── col 0: pure effects ───────────────────────────────────────────
        ax = axes[r, 0]
        ax.plot(t_grid, apply_kernel(pure_eff[fi_vix], kern_fn(fi_vix)) * sc,
                color=c_vix, lw=2.2, ls='-',  label='vix_prev')
        ax.plot(t_grid, apply_kernel(pure_eff[fi_ann], kern_fn(fi_ann)) * sc,
                color=c_ann, lw=2.2, ls='--', label='ann_indicator')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_ylabel(y_label, fontsize=FS_AXIS)
        ax.set_title(lbl_pure, fontsize=title_fs, fontweight='bold')
        ax.text(-0.32, 0.5, row_labels[r], transform=ax.transAxes,
                fontsize=FS_AXIS, va='center', ha='right',
                rotation=90, color='#333', fontweight='bold')

        # Legend only in row 0, col 0 → upper left
        if r == 0:
            ax.legend(fontsize=FS_LEGEND, loc='upper left', framealpha=0.9)

        # ── col 1: partial effects ────────────────────────────────────────
        ax = axes[r, 1]
        ax.plot(t_grid, apply_kernel(partial_eff[fi_vix], kern_fn(fi_vix)) * sc,
                color=c_vix, lw=2.2, ls='-')
        ax.plot(t_grid, apply_kernel(partial_eff[fi_ann], kern_fn(fi_ann)) * sc,
                color=c_ann, lw=2.2, ls='--')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        ax.set_title(lbl_partial, fontsize=title_fs, fontweight='bold')

        pure_int = float(np.sum(np.abs(
            apply_kernel(pure_eff[fi_ann], kern_fn(fi_ann))))) * sc
        part_int = float(np.sum(np.abs(
            apply_kernel(partial_eff[fi_ann], kern_fn(fi_ann))))) * sc
        ratio = part_int / pure_int if pure_int > 1e-12 else 1.0

        annot_y, annot_va = annot_col1[r]
        ax.text(0.03, annot_y,
                'ann_indicator: partial/pure\n= {:.2f}$\\times$'.format(ratio),
                transform=ax.transAxes, fontsize=FS_ANNOT - 1,
                va=annot_va, ha='left',
                color=c_ann,
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#ddd', alpha=0.9))

        # ── col 2: interaction ────────────────────────────────────────────
        ax  = axes[r, 2]
        kern_for_pair = K_id if r == 0 else K_ou
        int_mx = _int_curve(gtype, kern_for_pair) * sc
        pos = np.where(int_mx >= 0, int_mx, 0.0)
        neg = np.where(int_mx <  0, int_mx, 0.0)
        ax.fill_between(t_grid, 0, pos, color='#2a9d8f', alpha=0.30)
        ax.fill_between(t_grid, 0, neg, color='#e63946', alpha=0.30)
        ax.plot(t_grid, int_mx, color='#333', lw=1.8)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        _period_shade(ax); _ann_vline(ax); _set_time_axis(ax)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_xlabel('Time', fontsize=FS_AXIS)
        # No backslash before underscore
        ax.set_title(
            r'Interaction  $m_{ij}(t)$  — vix_prev $\times$ ann_indicator',
            fontsize=title_fs, fontweight='bold')
        integ = float(np.trapezoid(
            moebius_hv[gtype].get((fi_vix, fi_ann), np.zeros(T_BARS)),
            t_grid)) * sc
        ax.text(0.03, 0.97,
                r'$\int m_{{ij}}\,dt$ = {:.3f}'.format(integ),
                transform=ax.transAxes, fontsize=FS_ANNOT,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.25',
                          fc='white', ec='#aaa', alpha=0.85))

        # ── col 3: bar chart ──────────────────────────────────────────────
        ax_bar = axes[r, 3]
        imps_part = {i: float(np.sum(np.abs(
            apply_kernel(partial_eff[i], kern_fn(i))))) * sc
            for i in range(n_players)}
        imps_pure = {i: float(np.sum(np.abs(
            apply_kernel(pure_eff[i], kern_fn(i))))) * sc
            for i in range(n_players)}
        order = sorted(range(n_players),
                       key=lambda i: imps_part[i], reverse=True)
        y_pos = np.arange(len(order))
        bar_h = 0.35
        ax_bar.barh(y_pos - bar_h / 2,
                    [imps_pure[i] for i in order], height=bar_h,
                    color=[FEAT_COLORS[pnames[i]] for i in order],
                    alpha=0.45, hatch='//', label='pure')
        ax_bar.barh(y_pos + bar_h / 2,
                    [imps_part[i] for i in order], height=bar_h,
                    color=[FEAT_COLORS[pnames[i]] for i in order],
                    alpha=0.90, label='partial')
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([pnames[i] for i in order], fontsize=FS_TICK)
        ax_bar.axvline(0, color='gray', lw=0.8, ls=':')
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=FS_TICK)
        ax_bar.set_title('Integrated\nimportance', fontsize=title_fs,
                         fontweight='bold')
        ax_bar.legend(fontsize=FS_LEGEND, loc='upper right')

    # ── shared y-limits across time-series columns ────────────────────────
    for r in range(n_rows):
        ymin = min(axes[r, c].get_ylim()[0] for c in range(3))
        ymax = max(axes[r, c].get_ylim()[1] for c in range(3))
        for c in range(3):
            axes[r, c].set_ylim(ymin, ymax)

    plt.tight_layout(rect=[0.04, 0, 1, 0.95])
    fig.subplots_adjust(top=0.92, hspace=0.70)   # <── slightly more row whitespace
    return fig


if __name__ == '__main__':
    print('Loading data ...')
    X_day, Y_day, Y_adj, diurnal_mean = load_and_aggregate()
    X_day_np = X_day.to_numpy().astype(float)

    print('Fitting model ...')
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_day_np, Y_adj, test_size=0.2, random_state=RNG_SEED)
    model = RFModel()
    model.fit(X_tr, Y_tr)

    print('Selecting High-VIX Announcement profile ...')
    vix_col = DAY_FEATURE_NAMES.index('vix_prev')
    vix_p75 = float(np.percentile(X_day_np[:, vix_col], 75))
    ann_col = DAY_FEATURE_NAMES.index('ann_indicator')
    mask    = (X_day_np[:, ann_col] > 0.9) & (X_day_np[:, vix_col] >= vix_p75)
    hits    = X_day_np[mask]
    x_hv    = hits[len(hits) // 2]
    diffs   = np.abs(X_day_np - x_hv[None, :]).sum(axis=1)
    y_hv    = Y_adj[int(np.argmin(diffs))]

    print('Computing local games ...')
    moebius_hv = {}
    shapley_hv = {}
    for gtype in ('prediction', 'sensitivity', 'risk'):
        print('  game: {} ...'.format(gtype))
        game = FunctionalGame(
            predict_fn   = model.predict,
            X_background = X_day_np,
            x_explain    = x_hv,
            game_type    = gtype,
            Y_obs        = y_hv,
            sample_size  = SAMPLE_SIZE[gtype],
            random_seed  = RNG_SEED,
        )
        game.precompute()
        moebius_hv[gtype] = functional_moebius_transform(game)
        shapley_hv[gtype] = shapley_from_moebius(
            moebius_hv[gtype], game.n_players)

    print('Plotting (ROW_HEIGHT={}) ...'.format(ROW_HEIGHT))
    fig = plot_fig0_real(moebius_hv, shapley_hv,
                         list(DAY_FEATURE_NAMES), row_height=ROW_HEIGHT)
    fig.savefig(OUT_PATH, bbox_inches='tight', dpi=150)
    print('Saved:', OUT_PATH)
    plt.close(fig)