"""
Functional Explanation Framework -- Combined Energy Example  (v4)
=================================================================
Changes vs v3:
  - fig0_main_body_v2: layout from test script v3 integrated
      (nested heatmap+rowslice gridspec, wspace=0.90, box_aspect=0.50,
       AM/PM shading called last with SHADE_ALPHA=0.28, background colour boxes,
       legends below partial/gap panels)
  - _fig_ppf refactored to accept legend_rules dict and bar_legend_inside flag
  - fig2_local  (ihepc): row2 col0 legend smaller; row0 col0 normal
  - fig2_global (ihepc): all functional legends moved to col2, bottom-left
  - fig3_local  (neso):  remove col0 row0 legend; smaller fontsize col0 row1
  - fig3_global (neso):  remove col0 row0 legend; smaller fontsize col0 row1
  - fig5_local  (ihepc): move legend col0 row1 → col1 row1; remove all others
  - fig5_global (ihepc): move legend col0 row0 → col2 row0 bottom-left; remove all others
  - fig6_local  (neso):  same as fig5_local + remove col2 row2
  - fig6_global (neso):  move legend col0 row1 → upper-left (smaller); remove all others
  - All figs 2,3,5,6: bar-chart legends inside plot (top-right)
"""

import itertools
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0.  Global settings
# ---------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 42
RF_N_EST = 300
RF_JOBS  = -1

BASE_PLOT_DIR = os.path.join('plots', 'energy_comparison_all_games')

GLOBAL_N_INSTANCES = 30
GLOBAL_SAMPLE_SIZE = 100

FS_SUPTITLE = 13
FS_TITLE    = 11
FS_AXIS     = 10
FS_TICK     = 9
FS_LEGEND   = 8.5
FS_ANNOT    = 8.5

IHEPC_DATA_DIR  = os.path.join(_HERE, 'data')
IHEPC_DATA_FILE = os.path.join(IHEPC_DATA_DIR, 'household_power_consumption.parquet')
IHEPC_T      = 24
IHEPC_LABELS = ['{:02d}:00'.format(h) for h in range(IHEPC_T)]
IHEPC_TGRID  = np.arange(IHEPC_T, dtype=float)
IHEPC_FEATURES = ['day_of_week','is_weekend','month','season','lag_daily_mean','lag_morning']
IHEPC_MORNING = (6,  10)
IHEPC_EVENING = (17, 22)
IHEPC_SAMPLE  = {'prediction': 150, 'sensitivity': 200, 'risk': 200}
IHEPC_YLABEL  = {
    'prediction' : 'Effect on power (kW)',
    'sensitivity': r'Var$[F(t)]$ (kW$^2$)',
    'risk'       : r'Effect on MSE (kW$^2$)',
}

NESO_DATA_DIR = os.path.join(_HERE, 'data')
NESO_YEARS    = [2018, 2019, 2020, 2021, 2022]
NESO_T      = 48
NESO_LABELS = ['{:02d}:{:02d}'.format((i*30)//60,(i*30)%60) for i in range(NESO_T)]
NESO_TGRID  = np.arange(NESO_T, dtype=float)
NESO_FEATURES = ['day_of_week','is_weekend','month','season','lag_daily_mean','lag_morning','lag_evening']
NESO_MORNING = (12, 19)
NESO_EVENING = (34, 42)
NESO_SAMPLE  = {'prediction': 150, 'sensitivity': 200, 'risk': 200}
NESO_YLABEL  = {
    'prediction' : 'Effect on demand (MW)',
    'sensitivity': r'Var$[F(t)]$ (MW$^2$)',
    'risk'       : r'Effect on MSE (MW$^2$)',
}

GAME_TYPES = ['prediction', 'sensitivity', 'risk']

FEAT_COLORS = {
    'day_of_week'   : '#1f77b4',
    'is_weekend'    : '#ff7f0e',
    'month'         : '#2ca02c',
    'season'        : '#d62728',
    'lag_daily_mean': '#9467bd',
    'lag_morning'   : '#8c564b',
    'lag_evening'   : '#e377c2',
}

DS_LABEL = {
    'ihepc': 'UCI IHEPC\n(Single household, kW)',
    'neso' : 'NESO GB Demand\n(National grid, MW)',
}
DS_COLOR = {'ihepc': '#2a9d8f', 'neso': '#e76f51'}

_XAI_LABELS_LOCAL = {
    ('prediction',  'pure')   : 'Pure  $m_i$  $\\equiv$  local PDP',
    ('prediction',  'partial'): 'Partial  $\\phi_i$  $\\equiv$  SHAP',
    ('prediction',  'full')   : 'Full  $\\Phi_i$  $\\equiv$  local ICE-agg.',
    ('sensitivity', 'pure')   : r'Pure  $\equiv$  local closed Sobol',
    ('sensitivity', 'partial'): r'Partial  $\equiv$  local Shapley-sens.',
    ('sensitivity', 'full')   : r'Full  $\equiv$  local total Sobol',
    ('risk',        'pure')   : 'Pure  $\\equiv$  local pure risk',
    ('risk',        'partial'): 'Partial  $\\equiv$  local SAGE-style',
    ('risk',        'full')   : 'Full  $\\equiv$  local PFI-style',
}
_XAI_LABELS_GLOBAL = {
    ('prediction',  'pure')   : 'Pure  $m_i$  $\\equiv$  PDP',
    ('prediction',  'partial'): 'Partial  $\\phi_i$  $\\equiv$  global SHAP',
    ('prediction',  'full')   : 'Full  $\\Phi_i$  $\\equiv$  ICE-agg.',
    ('sensitivity', 'pure')   : r'Pure  $\equiv$  closed Sobol',
    ('sensitivity', 'partial'): r'Partial  $\equiv$  Shapley-sens.',
    ('sensitivity', 'full')   : r'Full  $\equiv$  total Sobol',
    ('risk',        'pure')   : 'Pure  $\\equiv$  pure risk',
    ('risk',        'partial'): 'Partial  $\\equiv$  SAGE',
    ('risk',        'full')   : 'Full  $\\equiv$  PFI',
}

_EFFECT_TYPES_E = ['pure', 'partial', 'full']
_LEG_LOC_E      = {0: 'upper left', 1: 'lower left', 2: 'lower left'}

_NODE_POS = '#2a9d8f'; _NODE_NEG = '#e63946'
_EDGE_SYN = '#2a9d8f'; _EDGE_RED = '#e63946'

FEAT_ABBR = {
    'day_of_week'   : 'DoW', 'is_weekend'    : 'WeD',
    'month'         : 'Mon', 'season'        : 'Sea',
    'lag_daily_mean': 'LDM', 'lag_morning'   : 'LMo', 'lag_evening': 'LEv',
}

# fig0 shading
SHADE_AM_COLOR = '#4a90e2'
SHADE_PM_COLOR = '#e24a4a'
SHADE_ALPHA    = 0.28


# ===========================================================================
# 1. Infrastructure
# ===========================================================================

def _require_dir(path): os.makedirs(path, exist_ok=True)

def _month_to_season(m):
    if m in (12,1,2): return 1
    elif m in (3,4,5): return 2
    elif m in (6,7,8): return 3
    else: return 4

class RFModel:
    def __init__(self, random_state=RNG_SEED):
        self.model = RandomForestRegressor(n_estimators=RF_N_EST,n_jobs=RF_JOBS,random_state=random_state)
    def fit(self,X,Y): self.model.fit(X,Y); return self
    def predict(self,X): return self.model.predict(X)
    def evaluate(self,X_te,Y_te):
        Yp=self.predict(X_te)
        return 1.0-np.sum((Y_te-Yp)**2)/np.sum((Y_te-Y_te.mean())**2)

class FunctionalGame:
    def __init__(self,predict_fn,X_bg,x_exp,T,features,game_type='prediction',
                 Y_obs=None,sample_size=150,random_seed=RNG_SEED):
        if game_type=='risk' and Y_obs is None: raise ValueError('Y_obs required for risk.')
        self.predict_fn=predict_fn; self.X_bg=X_bg; self.x_exp=x_exp; self.T=T
        self.game_type=game_type; self.Y_obs=Y_obs; self.n=sample_size; self.seed=random_seed
        self.p=len(features); self.player_names=list(features)
        self.coalitions=np.array(list(itertools.product([False,True],repeat=self.p)),dtype=bool)
        self.nc=len(self.coalitions)
        self._idx={tuple(c):i for i,c in enumerate(self.coalitions)}
        self.values=None
    def _impute(self,coal):
        rng=np.random.default_rng(self.seed)
        idx=rng.integers(0,len(self.X_bg),size=self.n)
        X=self.X_bg[idx].copy()
        for j in range(self.p):
            if coal[j]: X[:,j]=self.x_exp[j]
        return X
    def value_function(self,coal):
        X=self._impute(coal); Yp=self.predict_fn(X)
        if self.game_type=='prediction': return Yp.mean(axis=0)
        elif self.game_type=='sensitivity': return Yp.var(axis=0)
        else: return ((self.Y_obs[None,:]-Yp)**2).mean(axis=0)
    def precompute(self):
        print('    [{}] {} coalitions x {} samples x T={} ...'.format(
            self.game_type,self.nc,self.n,self.T))
        self.values=np.zeros((self.nc,self.T))
        for i,c in enumerate(self.coalitions):
            self.values[i]=self.value_function(tuple(c))
            if (i+1)%32==0 or i+1==self.nc: print('      {}/{}'.format(i+1,self.nc))
    def __getitem__(self,c): return self.values[self._idx[c]]
    @property
    def empty_value(self): return self[tuple([False]*self.p)]

def moebius_transform(game):
    p=game.p
    all_S=list(itertools.chain.from_iterable(
        itertools.combinations(range(p),r) for r in range(p+1)))
    mob={}
    for S in all_S:
        m=np.zeros(game.T)
        for L in itertools.chain.from_iterable(
                itertools.combinations(S,r) for r in range(len(S)+1)):
            c=tuple(i in L for i in range(p))
            m+=(-1)**(len(S)-len(L))*game[c]
        mob[S]=m
    return mob

def shapley_values(mob,p,T):
    shap={i:np.zeros(T) for i in range(p)}
    for S,m in mob.items():
        if len(S)==0: continue
        for i in S: shap[i]+=m/len(S)
    return shap


# ===========================================================================
# 2. Kernels
# ===========================================================================

def kernel_identity(T): return np.eye(T)

def kernel_correlation(Y_raw):
    C=np.cov(Y_raw.T); std=np.sqrt(np.diag(C))
    std=np.where(std<1e-12,1.0,std)
    return np.clip(C/np.outer(std,std),-1.0,1.0)

def apply_kernel(effect,K,dt=1.0):
    rs=K.sum(axis=1,keepdims=True)*dt
    rs=np.where(np.abs(rs)<1e-12,1.0,rs)
    return (K/rs)@effect*dt


# ===========================================================================
# 3. Pure / partial / full helpers
# ===========================================================================

def _pure_effects_e(mob,p,T):
    return {i:mob.get((i,),np.zeros(T)).copy() for i in range(p)}

def _full_effects_e(mob,p,T):
    full={i:np.zeros(T) for i in range(p)}
    for S,m in mob.items():
        if len(S)==0: continue
        for i in S: full[i]+=m
    return full


# ===========================================================================
# 4. IHEPC data loading
# ===========================================================================

def load_ihepc():
    if os.path.isfile(IHEPC_DATA_FILE):
        print('  [IHEPC] Loading parquet cache ...')
        df=pd.read_parquet(IHEPC_DATA_FILE)
        if 'date' not in df.columns:
            df['date']=pd.to_datetime(df['datetime']).dt.date.astype(str)
        if 'hour' not in df.columns:
            df['hour']=pd.to_datetime(df['datetime']).dt.hour
    else:
        import importlib
        if importlib.util.find_spec('ucimlrepo') is None: raise RuntimeError('pip install ucimlrepo')
        from ucimlrepo import fetch_ucirepo
        print('  [IHEPC] Downloading from UCI ML Repo ...')
        ds=fetch_ucirepo(id=235); df=ds.data.features.copy()
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime']=pd.to_datetime(df['Date']+' '+df['Time'],dayfirst=True,errors='coerce')
            df=df.drop(columns=['Date','Time'])
        else:
            df=df.reset_index(); df.columns=['datetime']+list(df.columns[1:])
            df['datetime']=pd.to_datetime(df['datetime'],errors='coerce')
        df=df.dropna(subset=['datetime'])
        for col in [c for c in df.columns if c not in {'datetime','date','hour'}]:
            df[col]=pd.to_numeric(df[col],errors='coerce')
        df=df.dropna(subset=['Global_active_power'])
        df['date']=df['datetime'].dt.date.astype(str); df['hour']=df['datetime'].dt.hour
        _require_dir(IHEPC_DATA_DIR); df.to_parquet(IHEPC_DATA_FILE,index=False)
    T=IHEPC_T
    hourly=(df.groupby(['date','hour'])['Global_active_power'].mean()
            .unstack('hour').reindex(columns=range(T)))
    hourly=hourly[hourly.notna().sum(axis=1)==T]
    Y_raw=hourly.values.astype(float); dates=hourly.index.tolist()
    diurnal=Y_raw.mean(axis=0); Y_adj=Y_raw-diurnal[None,:]
    records=[]
    for i,date_str in enumerate(dates):
        dt_obj=pd.Timestamp(date_str); m,dow=dt_obj.month,dt_obj.dayofweek
        lmean=float(Y_raw.mean()) if i==0 else float(Y_raw[i-1].mean())
        lmorn=(float(Y_raw[:,IHEPC_MORNING[0]:IHEPC_MORNING[1]].mean()) if i==0
               else float(Y_raw[i-1,IHEPC_MORNING[0]:IHEPC_MORNING[1]].mean()))
        records.append({'day_of_week':float(dow),'is_weekend':float(dow>=5),
                        'month':float(m),'season':float(_month_to_season(m)),
                        'lag_daily_mean':lmean,'lag_morning':lmorn})
    X_day=pd.DataFrame(records,index=dates)
    print('  [IHEPC] {} days, mean={:.3f} kW'.format(len(dates),Y_raw.mean()))
    return {'tag':'ihepc','X_np':X_day.to_numpy().astype(float),
            'Y_raw':Y_raw,'Y_adj':Y_adj,'diurnal':diurnal,'dates':dates,
            'features':IHEPC_FEATURES,'T':T,'t_grid':IHEPC_TGRID,
            'tlabels':IHEPC_LABELS,'sample':IHEPC_SAMPLE,'ylabel':IHEPC_YLABEL,
            'morning':IHEPC_MORNING,'evening':IHEPC_EVENING}


# ===========================================================================
# 5. NESO data loading
# ===========================================================================

def load_neso():
    dfs=[]
    for yr in NESO_YEARS:
        path=os.path.join(NESO_DATA_DIR,'demanddata_{}.csv'.format(yr))
        if not os.path.isfile(path): raise RuntimeError('Missing NESO file: {}'.format(path))
        dfs.append(pd.read_csv(path,low_memory=False))
    raw=pd.concat(dfs,ignore_index=True)
    raw.columns=[c.strip().upper() for c in raw.columns]
    date_col=next(c for c in raw.columns if 'DATE' in c)
    period_col=next(c for c in raw.columns if 'PERIOD' in c)
    demand_col='ND' if 'ND' in raw.columns else 'TSD'
    raw[date_col]=raw[date_col].astype(str).str.strip()
    raw[period_col]=pd.to_numeric(raw[period_col],errors='coerce')
    raw[demand_col]=pd.to_numeric(raw[demand_col],errors='coerce')
    raw=raw.dropna(subset=[date_col,period_col,demand_col])
    raw=raw[(raw[period_col]>=1)&(raw[period_col]<=NESO_T)].copy()
    raw['period_idx']=(raw[period_col]-1).astype(int)
    pivot=raw.pivot_table(index=date_col,columns='period_idx',values=demand_col,aggfunc='mean')
    pivot=pivot.reindex(columns=range(NESO_T))
    pivot=pivot[pivot.notna().sum(axis=1)==NESO_T]
    Y_raw=pivot.values.astype(float); dates=pivot.index.tolist()
    diurnal=Y_raw.mean(axis=0); Y_adj=Y_raw-diurnal[None,:]
    T=NESO_T; records=[]
    for i,date_str in enumerate(dates):
        dt_obj=pd.Timestamp(date_str); m,dow=dt_obj.month,dt_obj.dayofweek
        if i==0:
            lmean=float(Y_raw.mean())
            lmorn=float(Y_raw[:,NESO_MORNING[0]:NESO_MORNING[1]].mean())
            leve=float(Y_raw[:,NESO_EVENING[0]:NESO_EVENING[1]].mean())
        else:
            lmean=float(Y_raw[i-1].mean())
            lmorn=float(Y_raw[i-1,NESO_MORNING[0]:NESO_MORNING[1]].mean())
            leve=float(Y_raw[i-1,NESO_EVENING[0]:NESO_EVENING[1]].mean())
        records.append({'day_of_week':float(dow),'is_weekend':float(dow>=5),
                        'month':float(m),'season':float(_month_to_season(m)),
                        'lag_daily_mean':lmean,'lag_morning':lmorn,'lag_evening':leve})
    X_day=pd.DataFrame(records,index=dates)
    print('  [NESO] {} days, mean={:.0f} MW'.format(len(dates),Y_raw.mean()))
    return {'tag':'neso','X_np':X_day.to_numpy().astype(float),
            'Y_raw':Y_raw,'Y_adj':Y_adj,'diurnal':diurnal,'dates':dates,
            'features':NESO_FEATURES,'T':T,'t_grid':NESO_TGRID,
            'tlabels':NESO_LABELS,'sample':NESO_SAMPLE,'ylabel':NESO_YLABEL,
            'morning':NESO_MORNING,'evening':NESO_EVENING}


# ===========================================================================
# 6. Plotting helpers
# ===========================================================================

def _xticks(ax,ds,sparse=False):
    T,tlabels=ds['T'],ds['tlabels']
    step=max(1,T//8)*(2 if sparse else 1)
    idxs=list(range(0,T,step))
    ax.set_xticks(idxs)
    ax.set_xticklabels([tlabels[i] for i in idxs],rotation=45,ha='right',fontsize=FS_TICK)
    ax.set_xlim(-0.5,T-0.5)

def _shade(ax,ds):
    """AM/PM shading — call LAST so it overlays all plot elements."""
    ax.axvspan(*ds['morning'],alpha=SHADE_ALPHA,color=SHADE_AM_COLOR,zorder=10,lw=0)
    ax.axvspan(*ds['evening'],alpha=SHADE_ALPHA,color=SHADE_PM_COLOR,zorder=10,lw=0)

def savefig(fig,name):
    path=os.path.join(BASE_PLOT_DIR,name)
    fig.savefig(path,bbox_inches='tight',dpi=150)
    print('  Saved: {}'.format(path))
    plt.close(fig)


# ===========================================================================
# 7. PPF figure builder (shared core)
#
# legend_rules: dict mapping (row, col) -> {'show', 'loc', 'fs', 'anchor'}
#   Missing entries -> show=False (no legend drawn).
#   Legends are only placed where explicitly requested.
# bar_legend_inside: True  -> ax.legend(loc='upper right') inside the bar plot
#                    False -> bbox_to_anchor outside (old behaviour)
# ===========================================================================

def _fig_ppf(ds, effect_dicts_fn, top_k, xai_labels, suptitle, K, bar_title,
             legend_rules=None, bar_legend_inside=True):
    features = ds['features']
    p, T     = len(features), ds['T']
    t_grid   = ds['t_grid']
    ylabel   = ds['ylabel']

    fig, axes = plt.subplots(
        3, 4, figsize=(19, 4.0*3),
        gridspec_kw={'width_ratios': [3, 3, 3, 2.4]})
    fig.suptitle(suptitle, fontsize=FS_SUPTITLE, fontweight='bold')

    for r, gtype in enumerate(GAME_TYPES):
        effect_dicts = effect_dicts_fn(gtype)
        imps_partial = {i: float(np.sum(np.abs(apply_kernel(effect_dicts['partial'][i], K))))
                        for i in range(p)}
        top = sorted(imps_partial, key=imps_partial.get, reverse=True)[:top_k]

        for c, etype in enumerate(_EFFECT_TYPES_E):
            ax  = axes[r, c]
            eff = effect_dicts[etype]
            for fi in top:
                ax.plot(t_grid, apply_kernel(eff[fi], K),
                        color=FEAT_COLORS[features[fi]], lw=2.0,
                        label=features[fi])
            ax.axhline(0, color='gray', lw=0.5, ls=':')
            _xticks(ax, ds)
            ax.tick_params(labelsize=FS_TICK)
            ax.set_xlabel('Time', fontsize=FS_AXIS)
            ax.set_title(xai_labels[(gtype, etype)], fontsize=FS_TITLE, fontweight='bold')
            if c == 0:
                ax.set_ylabel(ylabel[gtype], fontsize=FS_AXIS)

            # Legend placement driven entirely by legend_rules
            rule = (legend_rules or {}).get((r, c))
            if rule and rule.get('show', False):
                fs     = rule.get('fs') or FS_LEGEND
                loc    = rule.get('loc') or 'upper left'
                anchor = rule.get('anchor')
                kwargs = dict(fontsize=fs, loc=loc, framealpha=0.85)
                if anchor is not None:
                    kwargs['bbox_to_anchor'] = anchor
                ax.legend(**kwargs)

            _shade(ax, ds)

        # Bar chart (col 3)
        ax_bar = axes[r, 3]
        imps_all = {
            etype: {i: float(np.sum(np.abs(apply_kernel(effect_dicts[etype][i], K))))
                    for i in range(p)}
            for etype in _EFFECT_TYPES_E}
        order   = sorted(range(p), key=lambda i: imps_all['partial'][i], reverse=True)
        y_pos   = np.arange(len(order))
        bar_h   = 0.25
        offsets = {'pure': -bar_h, 'partial': 0.0, 'full': bar_h}
        alphas  = {'pure': 0.55,   'partial': 0.90, 'full': 0.55}
        hatches = {'pure': '//',   'partial': '',   'full': '\\\\'}
        for etype in _EFFECT_TYPES_E:
            ax_bar.barh(y_pos+offsets[etype],
                        [imps_all[etype][i] for i in order], height=bar_h,
                        color=[FEAT_COLORS[features[i]] for i in order],
                        alpha=alphas[etype], hatch=hatches[etype], label=etype)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([features[i] for i in order], fontsize=FS_TICK)
        ax_bar.axvline(0, color='gray', lw=0.8, ls=':')
        ax_bar.set_xlabel(r'$\int|\cdot|\,dt$', fontsize=FS_AXIS)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.tick_params(labelsize=FS_TICK)
        ax_bar.set_title(bar_title, fontsize=FS_TITLE, fontweight='bold')
        if bar_legend_inside:
            ax_bar.legend(fontsize=FS_LEGEND, loc='upper right', framealpha=0.9)
        else:
            ax_bar.legend(fontsize=FS_LEGEND, loc='upper right',
                          bbox_to_anchor=(1.22, 1.0), borderaxespad=0.)

    plt.tight_layout()
    return fig


# ===========================================================================
# 8. fig2 — Local, identity kernel
#
# IHEPC (fig2_local):
#   row0 col0 — normal legend (upper left)
#   row2 col0 — smaller legend (lower left)
#   all other functional panels — no legend
#
# NESO (fig3_local):
#   row0 col0 — no legend
#   row1 col0 — smaller legend (lower left)
#   all other functional panels — no legend
# ===========================================================================

def fig_main_effects_ppf_identity(ds, mob_dict, shap_dict, top_k=5):
    tag=ds['tag']; T=ds['T']; K=kernel_identity(T)
    def _ed(gtype):
        mob=mob_dict[gtype]; p=len(ds['features'])
        return {'pure':_pure_effects_e(mob,p,T),'partial':shap_dict[gtype],'full':_full_effects_e(mob,p,T)}

    if tag == 'ihepc':
        rules = {
            (0,0): {'show':True,  'loc':'upper left', 'fs':FS_LEGEND,     'anchor':None},
            (2,0): {'show':True,  'loc':'lower left', 'fs':FS_LEGEND-1.5, 'anchor':None},
        }
    else:  # neso / fig3_local
        rules = {
            (1,0): {'show':True, 'loc':'lower left', 'fs':FS_LEGEND-1.5, 'anchor':None},
        }

    return _fig_ppf(ds, _ed, top_k, _XAI_LABELS_LOCAL,
        'Local main effects — Identity kernel — pure / partial / full\n'
        '{}'.format(DS_LABEL[tag].replace('\n','  ')),
        K, 'Integrated\nimportance\n(identity kernel)',
        legend_rules=rules, bar_legend_inside=True)


# ===========================================================================
# 9. Global computation helper
# ===========================================================================

def compute_global_shapley_e(ds,game_type,n_instances,sample_size,seed):
    X_bg=ds['X_np']; Y_adj=ds['Y_adj']; T=ds['T']; p=len(ds['features'])
    rng=np.random.default_rng(seed); idxs=rng.choice(len(X_bg),size=n_instances,replace=False)
    sum_shap={i:np.zeros(T) for i in range(p)}
    sum_pure={i:np.zeros(T) for i in range(p)}
    sum_full={i:np.zeros(T) for i in range(p)}
    for k,idx in enumerate(idxs):
        x_inst=X_bg[idx]; y_inst=Y_adj[idx] if game_type=='risk' else None
        game=FunctionalGame(predict_fn=ds['model'].predict,X_bg=X_bg,x_exp=x_inst,
                            T=T,features=ds['features'],game_type=game_type,
                            Y_obs=y_inst,sample_size=sample_size,random_seed=seed+k)
        game.precompute(); mob=moebius_transform(game)
        shap=shapley_values(mob,p,T); pure=_pure_effects_e(mob,p,T); full=_full_effects_e(mob,p,T)
        for i in range(p):
            sum_shap[i]+=shap[i]; sum_pure[i]+=pure[i]; sum_full[i]+=full[i]
        print('    [{} global] instance {}/{} done.'.format(game_type,k+1,n_instances))
    return ({i:sum_shap[i]/n_instances for i in range(p)},
            {i:sum_pure[i]/n_instances for i in range(p)},
            {i:sum_full[i]/n_instances for i in range(p)})


# ===========================================================================
# 10. fig2_global & fig3_global — Global, identity kernel
#
# IHEPC (fig2_global): move all functional legends to col2, bottom-left
# NESO  (fig3_global): remove col0 row0; smaller fontsize col0 row1
# ===========================================================================

def fig_global_main_effects_identity(ds, global_effects, top_k=5):
    tag=ds['tag']; T=ds['T']; K=kernel_identity(T)
    def _ed(gtype):
        avg_shap,avg_pure,avg_full=global_effects[gtype]
        return {'pure':avg_pure,'partial':avg_shap,'full':avg_full}

    if tag == 'ihepc':
        rules = {
            (0,2): {'show':True, 'loc':'lower left', 'fs':FS_LEGEND, 'anchor':None},
            (1,2): {'show':True, 'loc':'lower left', 'fs':FS_LEGEND, 'anchor':None},
            (2,2): {'show':True, 'loc':'lower left', 'fs':FS_LEGEND, 'anchor':None},
        }
    else:  # neso / fig3_global
        rules = {
            (1,0): {'show':True, 'loc':'lower left', 'fs':FS_LEGEND-1.5, 'anchor':None},
        }

    return _fig_ppf(ds, _ed, top_k, _XAI_LABELS_GLOBAL,
        'Global main effects — Identity kernel — pure / partial / full\n'
        '(averaged over {} instances)  —  {}'.format(
            GLOBAL_N_INSTANCES, DS_LABEL[tag].replace('\n','  ')),
        K, 'Integrated\nimportance\n(identity kernel)',
        legend_rules=rules, bar_legend_inside=True)


# ===========================================================================
# 11. fig5 & fig6 — Local, correlation kernel
#
# IHEPC (fig5_local): legend col0 row1 → col1 row1; remove all others
# NESO  (fig6_local): same + remove col2 row2
# ===========================================================================

def fig_main_effects_ppf(ds, mob_dict, shap_dict, K, top_k=5, legend_on_full=False):
    tag=ds['tag']; T=ds['T']
    def _ed(gtype):
        mob=mob_dict[gtype]; p=len(ds['features'])
        return {'pure':_pure_effects_e(mob,p,T),'partial':shap_dict[gtype],'full':_full_effects_e(mob,p,T)}

    # Both ihepc and neso: move legend from col0 row1 to col1 row1
    rules = {
        (1,1): {'show':True, 'loc':'lower left', 'fs':FS_LEGEND, 'anchor':None},
    }
    # neso only: also suppress col2 row2 (legend_on_full path removed)
    # (col2 row2 has no rule → no legend drawn, so nothing extra needed)

    return _fig_ppf(ds, _ed, top_k, _XAI_LABELS_LOCAL,
        'Local main effects — Empirical correlation kernel — pure / partial / full\n'
        '{}'.format(DS_LABEL[tag].replace('\n','  ')),
        K, 'Integrated\nimportance\n(corr. kernel)',
        legend_rules=rules, bar_legend_inside=True)


# ===========================================================================
# 12. fig5_global & fig6_global — Global, correlation kernel
#
# IHEPC (fig5_global): move legend col0 row0 → col2 row0 bottom-left; remove all others
# NESO  (fig6_global): move legend col0 row1 → upper-left (smaller); remove all others
# ===========================================================================

def fig_global_main_effects_corr(ds, global_effects, K, top_k=5):
    tag=ds['tag']
    def _ed(gtype):
        avg_shap,avg_pure,avg_full=global_effects[gtype]
        return {'pure':avg_pure,'partial':avg_shap,'full':avg_full}

    if tag == 'ihepc':
        rules = {
            (0,2): {'show':True, 'loc':'lower left', 'fs':FS_LEGEND, 'anchor':None},
        }
    else:  # neso
        rules = {
            (1,0): {'show':True, 'loc':'upper left', 'fs':FS_LEGEND-1.5, 'anchor':None},
        }

    return _fig_ppf(ds, _ed, top_k, _XAI_LABELS_GLOBAL,
        'Global main effects — Empirical correlation kernel — pure / partial / full\n'
        '(averaged over {} instances)  —  {}'.format(
            GLOBAL_N_INSTANCES, DS_LABEL[tag].replace('\n','  ')),
        K, 'Integrated\nimportance\n(corr. kernel)',
        legend_rules=rules, bar_legend_inside=True)


# ===========================================================================
# 13. Figure 7 — Combined sensitivity gap
# ===========================================================================

def fig_sensitivity_gap_combined(ds_ih,mob_sens_ih,K_ih,ds_ne,mob_sens_ne,K_ne,top_k=3):
    fig,axes=plt.subplots(2,top_k,figsize=(4.5*top_k,4.5*2),sharey='row')
    fig.suptitle(
        r'Sensitivity gap  $\Delta\tau_i(t) = \bar{\tau}_i(t) - \tau^{\mathrm{cl}}_i(t)$'
        '  —  Empirical correlation kernel'
        '\nTotal Sobol minus Closed Sobol: interaction contribution over time',
        fontsize=FS_SUPTITLE, fontweight='bold')
    for row_idx,(ds,mob_sens,K) in enumerate([(ds_ih,mob_sens_ih,K_ih),(ds_ne,mob_sens_ne,K_ne)]):
        tag=ds['tag']; features=ds['features']; p,T,t_grid=len(features),ds['T'],ds['t_grid']
        pure_eff=_pure_effects_e(mob_sens,p,T); full_eff=_full_effects_e(mob_sens,p,T)
        gap={i:full_eff[i]-pure_eff[i] for i in range(p)}
        gap_imp={i:float(np.sum(np.abs(apply_kernel(gap[i],K)))) for i in range(p)}
        top=sorted(gap_imp,key=gap_imp.get,reverse=True)[:top_k]
        for col_idx,fi in enumerate(top):
            ax=axes[row_idx,col_idx]; col=FEAT_COLORS[features[fi]]
            pure_c=apply_kernel(pure_eff[fi],K); full_c=apply_kernel(full_eff[fi],K)
            gap_c=apply_kernel(gap[fi],K)
            ax.fill_between(t_grid,pure_c,full_c,color=col,alpha=0.18,label='gap region')
            ax.plot(t_grid,full_c,color=col,lw=2.0,ls='-',label=r'Full  $\bar{\tau}_i$  (Total Sobol)')
            ax.plot(t_grid,pure_c,color=col,lw=2.0,ls='--',label=r'Pure  $\tau^{\mathrm{cl}}_i$  (Closed Sobol)')
            ax.plot(t_grid,gap_c,color='black',lw=1.4,ls=':',alpha=0.7,label=r'Gap  $\Delta\tau_i$')
            ax.axhline(0,color='gray',lw=0.5,ls=':')
            _xticks(ax,ds); ax.tick_params(labelsize=FS_TICK); ax.set_xlabel('Time',fontsize=FS_AXIS)
            if col_idx==0:
                ax.set_ylabel(ds['ylabel']['sensitivity'],fontsize=FS_AXIS)
                ax.text(-0.28,0.5,DS_LABEL[tag].replace('\n',' '),transform=ax.transAxes,
                        fontsize=FS_AXIS,va='center',ha='right',rotation=90,
                        color=DS_COLOR[tag],fontweight='bold')
            integ=float(np.trapz(np.abs(apply_kernel(gap[fi],K)),t_grid))
            ax.set_title('{}\n'.format(features[fi])+
                         r'$\int|\Delta\tau_i|\,dt$ = {:.4f}'.format(integ),
                         fontsize=FS_TITLE,fontweight='bold',color=col)
            ax.legend(fontsize=FS_LEGEND,loc='upper center',
                      bbox_to_anchor=(0.5,-0.22),ncol=2,framealpha=0.85)
            _shade(ax,ds)
    plt.tight_layout(); fig.subplots_adjust(bottom=0.18)
    return fig


# ===========================================================================
# 14. Network helpers
# ===========================================================================

def _network_importances(mob,shap,p,T,K,effect_type='partial'):
    pure_eff=_pure_effects_e(mob,p,T); full_eff=_full_effects_e(mob,p,T)
    eff=pure_eff if effect_type=='pure' else (shap if effect_type=='partial' else full_eff)
    t_grid=np.arange(T,dtype=float)
    node_imp=np.array([float(np.sum(np.abs(apply_kernel(eff[i],K)))) for i in range(p)])
    node_sign=np.array([np.sign(float(np.trapz(apply_kernel(eff[i],K),t_grid))) for i in range(p)])
    edge_imp={}
    for i in range(p):
        for j in range(i+1,p):
            if effect_type=='pure': raw=mob.get((i,j),np.zeros(T))
            elif effect_type=='partial':
                raw=np.zeros(T)
                for S,m in mob.items():
                    if i in S and j in S: raw=raw+m/len(S)
            else:
                raw=np.zeros(T)
                for S,m in mob.items():
                    if i in S and j in S: raw=raw+m
            val=float(np.trapz(apply_kernel(raw,K),t_grid))
            if abs(val)>0: edge_imp[(i,j)]=val
    return node_imp,edge_imp,node_sign

def _draw_network(ax,features,node_imp,edge_imp,node_sign,title,fs_title=None):
    import math
    fs_t=fs_title if fs_title is not None else FS_TITLE
    p=len(features); angle=[math.pi/2-2*math.pi*i/p for i in range(p)]
    pos={i:(math.cos(a),math.sin(a)) for i,a in enumerate(angle)}
    ax.set_aspect('equal'); ax.axis('off')
    if title: ax.set_title(title,fontsize=fs_t,fontweight='bold',pad=4)
    max_imp=float(node_imp.max()) if node_imp.max()>0 else 1.0
    node_r={i:0.07+0.19*(node_imp[i]/max_imp) for i in range(p)}
    max_edge=max((abs(v) for v in edge_imp.values()),default=1.0); max_edge=max(max_edge,1e-12)
    for (i,j),val in edge_imp.items():
        xi,yi=pos[i]; xj,yj=pos[j]
        lw=0.4+6.5*abs(val)/max_edge; col=_EDGE_SYN if val>0 else _EDGE_RED
        alph=0.30+0.60*abs(val)/max_edge
        ax.plot([xi,xj],[yi,yj],color=col,lw=lw,alpha=alph,solid_capstyle='round',zorder=1)
    for i in range(p):
        x,y=pos[i]; r=node_r[i]; fc=_NODE_POS if node_sign[i]>=0 else _NODE_NEG
        ax.add_patch(plt.Circle((x,y),r,color=fc,ec='white',linewidth=1.2,zorder=2,alpha=0.88))
        ax.add_patch(plt.Circle((x,y),r*0.52,color='white',ec='none',zorder=3,alpha=0.95))
        abbr=FEAT_ABBR.get(features[i],features[i][:3])
        ax.text(x,y,abbr,ha='center',va='center',fontsize=max(4.5,r*22),
                fontweight='bold',color='#222',zorder=4)
    pad=0.32; ax.set_xlim(-1.0-pad,1.0+pad); ax.set_ylim(-1.0-pad,1.0+pad)

def fig_network_appendix(ds,mob_dict,shap_dict,K_corr):
    tag=ds['tag']; features=ds['features']; p=len(features); T=ds['T']
    game_specs=[
        ('prediction','Prediction','local PDP','SHAP','local ICE-agg.'),
        ('sensitivity','Sensitivity','local closed Sobol','local Shapley-sens.','local total Sobol'),
        ('risk','Risk (MSE)','local pure risk','local SAGE-style','local PFI-style'),
    ]
    fig=plt.figure(figsize=(10,10))
    fig.suptitle('Local network plots — correlation kernel — all games\n'
                 '{}'.format(DS_LABEL[tag].replace('\n','  ')),
                 fontsize=FS_SUPTITLE,fontweight='bold',y=1.01)
    gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.08,wspace=0.08,
                         left=0.09,right=0.98,top=0.91,bottom=0.07)
    for r,(gtype,glabel,lp,lpa,lf) in enumerate(game_specs):
        col_labels=['Pure  $m_i \\equiv$ {}'.format(lp),
                    'Partial  $\\phi_i \\equiv$ {}'.format(lpa),
                    'Full  $\\Phi_i \\equiv$ {}'.format(lf)]
        for c,etype in enumerate(['pure','partial','full']):
            ax=fig.add_subplot(gs[r,c])
            node_imp,edge_imp,node_sign=_network_importances(
                mob_dict[gtype],shap_dict[gtype],p,T,K_corr,etype)
            _draw_network(ax,features,node_imp,edge_imp,node_sign,
                          col_labels[c] if r==0 else '')
            if c==0:
                ax.text(-0.03,0.5,glabel,transform=ax.transAxes,fontsize=FS_AXIS,
                        va='center',ha='right',rotation=90,color='#333',fontweight='bold')
    leg_handles=[Patch(facecolor=_NODE_POS,edgecolor='none',label='Positive effect'),
                 Patch(facecolor=_NODE_NEG,edgecolor='none',label='Negative effect')]
    fig.legend(handles=leg_handles,loc='lower center',ncol=2,fontsize=FS_LEGEND,
               framealpha=0.9,bbox_to_anchor=(0.5,0.01))
    return fig


# ===========================================================================
# 15. Figure 0 — Main body v2  (layout from test script v3)
# ===========================================================================

def _add_bg_box(fig, axes_list, color, pad=0.014):
    renderer = fig.canvas.get_renderer()
    xmins,ymins,xmaxs,ymaxs=[],[],[],[]
    for ax in axes_list:
        bb=ax.get_window_extent(renderer=renderer)
        bb_fig=bb.transformed(fig.transFigure.inverted())
        xmins.append(bb_fig.x0); ymins.append(bb_fig.y0)
        xmaxs.append(bb_fig.x1); ymaxs.append(bb_fig.y1)
    x0=min(xmins)-pad; y0=min(ymins)-pad
    w=max(xmaxs)-x0+pad; h=max(ymaxs)-y0+pad
    rect=FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=0.006',
                        linewidth=1.4,edgecolor=color,facecolor=color,alpha=0.10,
                        transform=fig.transFigure,zorder=0,clip_on=False)
    fig.add_artist(rect)


def fig0_main_body_v2(ds_ih, ds_ne, mob_ih, shap_ih, mob_ne, shap_ne, K_ih, K_ne):
    FS_SUP=16; FS_T=13; FS_AX=12; FS_TK=10.5; FS_LEG=10.5; FS_RLAB=13
    ID_ALPHA=0.45; ID_LW=1.8; MX_LW=2.4

    K_id_ih=kernel_identity(ds_ih['T']); K_id_ne=kernel_identity(ds_ne['T'])

    fig=plt.figure(figsize=(26,8.2))
    gs_outer=GridSpec(2,1,figure=fig,height_ratios=[1.0,0.80],hspace=0.48,
                      top=0.88,bottom=0.14,left=0.04,right=0.98)

    # Row 0: 6 columns; heatmap+rowslice pairs in nested 1x2 gridspecs
    gs0=GridSpecFromSubplotSpec(1,6,subplot_spec=gs_outer[0],wspace=0.28,
                                width_ratios=[0.85,0.85,2.2,2.2,0.85,0.85])
    ax_net_ih_sens=fig.add_subplot(gs0[0]); ax_net_ih_pred=fig.add_subplot(gs0[1])
    ax_net_ne_sens=fig.add_subplot(gs0[4]); ax_net_ne_pred=fig.add_subplot(gs0[5])

    gs_ih_pair=GridSpecFromSubplotSpec(1,2,subplot_spec=gs0[2],wspace=0.90)
    ax_ih_heat=fig.add_subplot(gs_ih_pair[0]); ax_ih_row=fig.add_subplot(gs_ih_pair[1])

    gs_ne_pair=GridSpecFromSubplotSpec(1,2,subplot_spec=gs0[3],wspace=0.90)
    ax_ne_heat=fig.add_subplot(gs_ne_pair[0]); ax_ne_row=fig.add_subplot(gs_ne_pair[1])

    # Row 1: 4 equal columns
    gs1=GridSpecFromSubplotSpec(1,4,subplot_spec=gs_outer[1],wspace=0.32)
    ax_ih_part=fig.add_subplot(gs1[0]); ax_ih_gap=fig.add_subplot(gs1[1])
    ax_ne_gap=fig.add_subplot(gs1[2]); ax_ne_part=fig.add_subplot(gs1[3])

    fig.suptitle('Energy demand: correlation structure drives explanation shape\n'
                 'UCI IHEPC (single household, kW)  vs  NESO GB Demand (national grid, MW)',
                 fontsize=FS_SUP,fontweight='bold',y=0.975)

    def _heatmap(ax,ds,K,tag):
        T,tl=ds['T'],ds['tlabels']; step=max(1,T//6); ticks=list(range(0,T,step))
        im=ax.imshow(K,aspect='equal',origin='upper',cmap='RdBu_r',vmin=-0.2,vmax=1.0)
        ax.set_xticks(ticks); ax.set_xticklabels([tl[i] for i in ticks],rotation=45,ha='right',fontsize=7.0)
        ax.set_yticks(ticks); ax.set_yticklabels([tl[i] for i in ticks],fontsize=6.5)
        ax.set_title(DS_LABEL[tag].replace('\n',' ')+'\ncorrelation kernel $K$',
                     fontsize=FS_AX,fontweight='bold',color=DS_COLOR[tag])
        am=(ds['morning'][0]+ds['morning'][1])//2
        ax.axhline(am,color='white',lw=0.8,ls='--',alpha=0.6)
        ax.axvline(am,color='white',lw=0.8,ls='--',alpha=0.6)
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.03).ax.tick_params(labelsize=6.5)

    def _rowslice(ax,ds,K,tag):
        T,tl=ds['T'],ds['tlabels']; am=(ds['morning'][0]+ds['morning'][1])//2
        step=max(1,T//6); ticks=list(range(0,T,step))
        ax.plot(np.arange(T),K[am,:],color=DS_COLOR[tag],lw=2.2,zorder=2)
        ax.axhline(0,color='gray',lw=0.5,ls=':',zorder=1)
        ax.set_xticks(ticks); ax.set_xticklabels([tl[i] for i in ticks],rotation=45,ha='right',fontsize=7.0)
        ax.set_xlim(-0.5,T-0.5); ax.set_ylabel('$K(t_{\\mathrm{AM}}, s)$',fontsize=FS_AX)
        ax.set_xlabel('Time $s$',fontsize=FS_AX); ax.tick_params(labelsize=FS_TK)
        ax.set_box_aspect(0.50)
        title=('Structured (AM$\\leftrightarrow$PM)' if tag=='ihepc' else 'Uniform (regime-dominated)')
        ax.set_title(title,fontsize=FS_AX,color=DS_COLOR[tag],fontweight='bold')
        _shade(ax,ds)  # called last

    _heatmap(ax_ih_heat,ds_ih,K_ih,'ihepc'); _rowslice(ax_ih_row,ds_ih,K_ih,'ihepc')
    _heatmap(ax_ne_heat,ds_ne,K_ne,'neso');  _rowslice(ax_ne_row,ds_ne,K_ne,'neso')

    net_handles=[Patch(facecolor=_NODE_POS,edgecolor='none',label='Positive'),
                 Patch(facecolor=_NODE_NEG,edgecolor='none',label='Negative')]
    for ax,mob,shap,ds,title in [
        (ax_net_ih_sens,mob_ih['sensitivity'],shap_ih['sensitivity'],ds_ih,'IHEPC sens.\npartial (corr)'),
        (ax_net_ih_pred,mob_ih['prediction'], shap_ih['prediction'], ds_ih,'IHEPC pred.\npartial (corr)'),
        (ax_net_ne_sens,mob_ne['sensitivity'],shap_ne['sensitivity'],ds_ne,'NESO sens.\npartial (corr)'),
        (ax_net_ne_pred,mob_ne['prediction'], shap_ne['prediction'], ds_ne,'NESO pred.\npartial (corr)'),
    ]:
        features=ds['features']; p,T=len(features),ds['T']
        K=K_ih if ds['tag']=='ihepc' else K_ne
        ni,ei,ns=_network_importances(mob,shap,p,T,K,'partial')
        _draw_network(ax,features,ni,ei,ns,title,fs_title=FS_T)
        ax.legend(handles=net_handles,loc='lower center',ncol=2,fontsize=FS_LEG-1.5,
                  framealpha=0.88,bbox_to_anchor=(0.5,-0.18),bbox_transform=ax.transAxes,
                  borderpad=0.4,handlelength=1.2)

    def _partial_panel(ax,ds,shap,K_id,K_corr,tag,force_features):
        features=ds['features']; p,T=len(features),ds['T']
        partial=shap['prediction']; top2=[features.index(f) for f in force_features]
        for fi in top2:
            col=FEAT_COLORS[features[fi]]; ls='-' if fi==top2[0] else '--'
            ax.plot(ds['t_grid'],apply_kernel(partial[fi],K_id),color=col,lw=ID_LW,ls=ls,alpha=ID_ALPHA,zorder=2)
            ax.plot(ds['t_grid'],apply_kernel(partial[fi],K_corr),color=col,lw=MX_LW,ls=ls,
                    label=features[fi]+' (corr.)',zorder=3)
        ax.axhline(0,color='gray',lw=0.5,ls=':',zorder=1)
        _xticks(ax,ds,sparse=True); ax.tick_params(labelsize=FS_TK)
        ax.set_xlabel('Time',fontsize=FS_AX); ax.set_ylabel(ds['ylabel']['prediction'],fontsize=FS_AX)
        ax.set_title('Partial  $\\phi_i \\equiv$ SHAP\n'+DS_LABEL[tag].split('\n')[0],
                     fontsize=FS_T,fontweight='bold',color=DS_COLOR[tag])
        extra=Line2D([0],[0],color='gray',lw=ID_LW,ls='-',alpha=ID_ALPHA,label='identity (faded)')
        handles,labels=ax.get_legend_handles_labels()
        ax.legend(handles+[extra],labels+['identity (faded)'],fontsize=FS_LEG,loc='upper center',
                  bbox_to_anchor=(0.5,-0.22),ncol=3,framealpha=0.85)
        _shade(ax,ds)  # called last

    def _gap_panel(ax,ds,mob_sens,K_corr,tag):
        features=ds['features']; p,T,t_grid=len(features),ds['T'],ds['t_grid']
        pure_eff=_pure_effects_e(mob_sens,p,T); full_eff=_full_effects_e(mob_sens,p,T)
        gap={i:full_eff[i]-pure_eff[i] for i in range(p)}
        gap_imp={i:float(np.sum(np.abs(apply_kernel(gap[i],K_corr)))) for i in range(p)}
        fi=max(gap_imp,key=gap_imp.get); col=FEAT_COLORS[features[fi]]
        pure_c=apply_kernel(pure_eff[fi],K_corr); full_c=apply_kernel(full_eff[fi],K_corr)
        gap_c=apply_kernel(gap[fi],K_corr)
        ax.fill_between(t_grid,pure_c,full_c,color=col,alpha=0.20,label='gap region',zorder=1)
        ax.plot(t_grid,full_c,color=col,lw=2.0,ls='-',label=r'Full $\bar{\tau}_i$ (Total Sobol)',zorder=3)
        ax.plot(t_grid,pure_c,color=col,lw=2.0,ls='--',label=r'Pure $\tau^{\mathrm{cl}}_i$ (Closed Sobol)',zorder=3)
        ax.plot(t_grid,gap_c,color='black',lw=1.4,ls=':',alpha=0.7,label=r'Gap $\Delta\tau_i$',zorder=3)
        ax.axhline(0,color='gray',lw=0.5,ls=':',zorder=1)
        _xticks(ax,ds,sparse=True); ax.tick_params(labelsize=FS_TK)
        ax.set_xlabel('Time',fontsize=FS_AX); ax.set_ylabel(ds['ylabel']['sensitivity'],fontsize=FS_AX)
        integ=float(np.trapz(np.abs(apply_kernel(gap[fi],K_corr)),t_grid))
        ax.set_title('Sensitivity gap  —  corr. kernel\n'
                     '{} — {}  $\\int|\\Delta\\tau_i|\\,dt = {:.3g}$'.format(
                         DS_LABEL[tag].split('\n')[0],features[fi],integ),
                     fontsize=FS_T-1,fontweight='bold',color=DS_COLOR[tag])
        ax.legend(fontsize=FS_LEG,loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=2,framealpha=0.85)
        _shade(ax,ds)  # called last

    _partial_panel(ax_ih_part,ds_ih,shap_ih,K_id_ih,K_ih,'ihepc',['month','lag_morning'])
    _partial_panel(ax_ne_part,ds_ne,shap_ne,K_id_ne,K_ne,'neso', ['season','month'])
    _gap_panel(ax_ih_gap,ds_ih,mob_ih['sensitivity'],K_ih,'ihepc')
    _gap_panel(ax_ne_gap,ds_ne,mob_ne['sensitivity'],K_ne,'neso')

    for ax,label,tag,x,ha in [
        (ax_net_ih_sens,'IHEPC','ihepc',-0.14,'right'),
        (ax_net_ne_pred,'NESO', 'neso',  1.14,'left'),
        (ax_ih_part,    'IHEPC','ihepc',-0.20,'right'),
        (ax_ne_part,    'NESO', 'neso',  1.20,'left'),
    ]:
        ax.text(x,0.5,label,transform=ax.transAxes,fontsize=FS_RLAB,va='center',ha=ha,
                rotation=90,color=DS_COLOR[tag],fontweight='bold')

    fig.canvas.draw()
    _add_bg_box(fig,[ax_net_ih_sens,ax_net_ih_pred,ax_ih_heat,ax_ih_row,ax_ih_part,ax_ih_gap],
                DS_COLOR['ihepc'],pad=0.012)
    _add_bg_box(fig,[ax_ne_heat,ax_ne_row,ax_net_ne_sens,ax_net_ne_pred,ax_ne_gap,ax_ne_part],
                DS_COLOR['neso'],pad=0.012)
    return fig


# ===========================================================================
# 16. Run helpers
# ===========================================================================

def run_games(ds, x_primary, y_primary):
    results={}
    for gtype in GAME_TYPES:
        game=FunctionalGame(predict_fn=ds['model'].predict,X_bg=ds['X_np'],
                            x_exp=x_primary,T=ds['T'],features=ds['features'],
                            game_type=gtype,Y_obs=y_primary,
                            sample_size=ds['sample'][gtype],random_seed=RNG_SEED)
        game.precompute(); mob=moebius_transform(game)
        shap=shapley_values(mob,game.p,game.T); results[gtype]=(mob,shap)
    return results


# ===========================================================================
# 17. Main
# ===========================================================================

if __name__ == '__main__':
    print('\n'+'='*60); print('  Energy Combined Example  (IHEPC + NESO)  v4'); print('='*60)
    _require_dir(BASE_PLOT_DIR)

    print('\n[1] Loading data ...'); ds_ih=load_ihepc(); ds_ne=load_neso()

    print('\n[2] Fitting models ...')
    for ds,name in [(ds_ih,'IHEPC'),(ds_ne,'NESO')]:
        X_tr,X_te,Y_tr,Y_te=train_test_split(ds['X_np'],ds['Y_adj'],test_size=0.2,random_state=RNG_SEED)
        m=RFModel(); m.fit(X_tr,Y_tr)
        print('  [{}] Test R²: {:.4f}'.format(name,m.evaluate(X_te,Y_te))); ds['model']=m

    print('\n[3] Building correlation kernels ...')
    K_ih=kernel_correlation(ds_ih['Y_raw']); K_ne=kernel_correlation(ds_ne['Y_raw'])

    print('\n[4] Selecting profiles ...')
    X_ih=ds_ih['X_np']; fn_ih=ds_ih['features']
    lag_p75_ih=float(np.percentile(X_ih[:,fn_ih.index('lag_daily_mean')],75))
    def find_ih(conds,lbl):
        mask=np.ones(len(X_ih),dtype=bool)
        for f,(lo,hi) in conds.items():
            ci=fn_ih.index(f); mask&=(X_ih[:,ci]>=lo)&(X_ih[:,ci]<=hi)
        hits=X_ih[mask]
        if not len(hits): raise RuntimeError('No match: {}'.format(lbl))
        print('  IHEPC "{}": {} days'.format(lbl,len(hits))); return hits[len(hits)//2]
    x_ih1=find_ih({'is_weekend':(-0.1,0.1),'day_of_week':(0.9,4.1)},'Typical weekday')
    x_ih2=find_ih({'is_weekend':(0.9,1.1)},'Weekend')
    x_ih3=find_ih({'season':(0.9,1.1),'is_weekend':(-0.1,0.1),'lag_daily_mean':(lag_p75_ih,9e9)},'Cold winter day')
    def _y_ih(xp):
        diffs=np.abs(X_ih-xp[None,:]).sum(axis=1); return ds_ih['Y_adj'][int(np.argmin(diffs))]

    X_ne=ds_ne['X_np']; fn_ne=ds_ne['features']
    lag_p75_ne=float(np.percentile(X_ne[:,fn_ne.index('lag_daily_mean')],75))
    def find_ne(conds,lbl):
        mask=np.ones(len(X_ne),dtype=bool)
        for f,(lo,hi) in conds.items():
            ci=fn_ne.index(f); mask&=(X_ne[:,ci]>=lo)&(X_ne[:,ci]<=hi)
        hits=X_ne[mask]
        if not len(hits): raise RuntimeError('No match: {}'.format(lbl))
        print('  NESO "{}": {} days'.format(lbl,len(hits))); return hits[len(hits)//2]
    x_ne1=find_ne({'is_weekend':(-0.1,0.1),'season':(0.9,1.1)},'Winter weekday')
    x_ne2=find_ne({'is_weekend':(0.9,1.1),'season':(2.9,3.1)},'Summer weekend')
    x_ne3=find_ne({'is_weekend':(-0.1,0.1),'season':(0.9,1.1),'lag_daily_mean':(lag_p75_ne,9e9)},'Cold snap weekday')
    def _y_ne(xp):
        diffs=np.abs(X_ne-xp[None,:]).sum(axis=1); return ds_ne['Y_adj'][int(np.argmin(diffs))]

    print('\n[5] Computing local games ...')
    print('\n  IHEPC — Typical weekday:')
    ih_games=run_games(ds_ih,x_ih1,_y_ih(x_ih1))
    mob_ih={gt:ih_games[gt][0] for gt in GAME_TYPES}; shap_ih={gt:ih_games[gt][1] for gt in GAME_TYPES}
    print('\n  NESO — Winter weekday:')
    ne_games=run_games(ds_ne,x_ne1,_y_ne(x_ne1))
    mob_ne={gt:ne_games[gt][0] for gt in GAME_TYPES}; shap_ne={gt:ne_games[gt][1] for gt in GAME_TYPES}

    print('\n[6] Computing global effects ({} instances per game) ...'.format(GLOBAL_N_INSTANCES))
    global_ih={}; global_ne={}
    for gtype in GAME_TYPES:
        print('\n  IHEPC global {} ...'.format(gtype))
        global_ih[gtype]=compute_global_shapley_e(ds_ih,gtype,GLOBAL_N_INSTANCES,GLOBAL_SAMPLE_SIZE,RNG_SEED)
        print('\n  NESO global {} ...'.format(gtype))
        global_ne[gtype]=compute_global_shapley_e(ds_ne,gtype,GLOBAL_N_INSTANCES,GLOBAL_SAMPLE_SIZE,RNG_SEED)

    print('\n[7] Generating figures ...')
    savefig(fig0_main_body_v2(ds_ih,ds_ne,mob_ih,shap_ih,mob_ne,shap_ne,K_ih,K_ne),'fig0_main_body_v2.pdf')
    savefig(fig_main_effects_ppf_identity(ds_ih,mob_ih,shap_ih),'fig2_local_main_effects_ppf_identity_ihepc.pdf')
    savefig(fig_global_main_effects_identity(ds_ih,global_ih),'fig2_global_main_effects_identity_ihepc.pdf')
    savefig(fig_main_effects_ppf_identity(ds_ne,mob_ne,shap_ne),'fig3_local_main_effects_ppf_identity_neso.pdf')
    savefig(fig_global_main_effects_identity(ds_ne,global_ne),'fig3_global_main_effects_identity_neso.pdf')
    savefig(fig_main_effects_ppf(ds_ih,mob_ih,shap_ih,K_ih),'fig5_local_main_effects_ppf_ihepc.pdf')
    savefig(fig_global_main_effects_corr(ds_ih,global_ih,K_ih),'fig5_global_main_effects_ppf_ihepc.pdf')
    savefig(fig_main_effects_ppf(ds_ne,mob_ne,shap_ne,K_ne),'fig6_local_main_effects_ppf_neso.pdf')
    savefig(fig_global_main_effects_corr(ds_ne,global_ne,K_ne),'fig6_global_main_effects_ppf_neso.pdf')
    savefig(fig_sensitivity_gap_combined(ds_ih,mob_ih['sensitivity'],K_ih,ds_ne,mob_ne['sensitivity'],K_ne),'fig7_sensitivity_gap_combined.pdf')
    savefig(fig_network_appendix(ds_ih,mob_ih,shap_ih,K_ih),'fig_network_appendix_ihepc.pdf')
    savefig(fig_network_appendix(ds_ne,mob_ne,shap_ne,K_ne),'fig_network_appendix_neso.pdf')

    print('\n'+'='*60); print('  Done.  Figures in {}/'.format(BASE_PLOT_DIR)); print('='*60)