# A Hilbert-Valued Functional Decomposition Framework for Explaining Time-Dependent Outputs

Code repository for the paper

> **A Hilbert-Valued Functional Decomposition Framework for Explaining Time-Dependent Outputs**  
> *NeurIPS 2026*

---

## Overview

The *H-FD framework* extends cooperative game-theoretic explanation methods to models whose outputs are trajectories or functional curves rather than scalar values. Given a model $F : \mathcal{X} \to \mathcal{H}$ mapping a feature vector to a Hilbert-space-valued output (e.g. an intraday volatility curve or a 24-hour demand profile), H-FD decomposes the output into *pure*, *partial*, and *full* feature effects via the functional Möbius transform, parameterised by a user-chosen output kernel $K$ that encodes which temporal relationships matter for the explanation.

---

## Repository structure

```
.
├── synthetic_experiments/              # Synthetic experiment scripts
│   ├── gt_validation.py                # Part 1: ground truth effect recovery
│   ├── thm2_validation.py              # Part 2: Sobol theorem validation
│   ├── kernel_guidance.py              # Part 3: kernel guidance figures
│   ├── ranking_preservation.py         # Part 3: ranking preservation across games
│   └── working_example_ICU.py          # ICU working example / intro figure data
│
├── spy/                                # SPY intraday volatility experiment
│   ├── spy_hfd.py                      # Main script
│   ├── data/                           # Raw data (see Data section)
│   └── game_results/                   # Cached game results (.npz)
│
├── ihepc_ngeso/                        # Energy demand experiment
│   ├── energy_hfd.py                   # Main script
│   ├── data/                           # Raw data (see Data section)
│   └── game_results_energy/            # Cached game results (.npz)
│
└── plots/                              # All generated figures
    ├── synthetic_experiments/
    │   ├── gt_validation/
    │   ├── thm2_validation/
    │   └── kernel_guidance/
    ├── spy/
    ├── energy/
    └── icu_illustration/
```

---

## Scripts

| Script | Paper section | What it produces |
|--------|--------------|-----------------|
| `synthetic_experiments/gt_validation.py` | Sec. 5 / App. D.1 | Effect recovery vs. $n$ for four model classes + oracle (Figs. 5, 6) |
| `synthetic_experiments/thm2_validation.py` | Sec. 5 / App. D.1 | Sobol index recovery under the constant kernel (Fig. 7) |
| `synthetic_experiments/kernel_guidance.py` | Sec. 5.1 / App. D.2 | Kernel guidance figures — ICU, price pulse, periodic (Fig. 2) |
| `synthetic_experiments/ranking_preservation.py` | App. D.2 | Ranking preservation across all three game types (Fig. 8) |
| `synthetic_experiments/working_example_ICU.py` | Fig. 1 | CSV and LaTeX macros for the intro figure |
| `spy/spy_hfd.py` | Sec. 5.2 / App. D.3 | Random Forest on SPY 5-min bars (Figs. 3, 9–12) |
| `ihepc_ngeso/energy_hfd.py` | Sec. 5.2 / App. D.4 | UCI IHEPC and NESO GB national demand (Figs. 4, 13–21) |

---

## Data

### Synthetic experiments
Fully self-contained — no external data required.

### SPY intraday volatility
The 5-minute bar data was purchased from a commercial provider (Polygon.io, Stocks Starter tier) and **cannot be redistributed**. The precomputed game result caches in `spy/game_results/` are provided so that all figures can be reproduced without the raw data — the script detects and loads `.npz` cache files automatically:

```bash
python spy/spy_hfd.py
```

If you have the raw bar data, place it at `spy/data/spy_5min_cache.csv` and the script will run the full pipeline including model fitting and game computation.

VIX data is fetched automatically from Yahoo Finance via `yfinance` and cached at `spy/data/vix_daily_cache.csv`.

**Note on cache contents.** Each cached game-result file stores an explicand feature vector `x_inst` alongside the attribution trajectories. To stay clearly within the Polygon.io license terms, two fields derived from the proprietary bar data — `overnight_ret` and `trailing_rv` — are replaced with NaN in the public caches. The remaining four fields (`vix_prev` from Yahoo Finance, `ann_indicator`, `day_of_week`, `month` from public schedules) are preserved. This affects only the x-axis bins of those two features in the PDP figure (fig3); all other figures are reproduced exactly. The stripping procedure is implemented in `anonymize_caches.py`.

### Energy demand — UCI IHEPC
The Individual Household Electric Power Consumption dataset is downloaded automatically from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/235) via the `ucimlrepo` package on first run, and cached locally as a parquet file. No manual download needed.

### Energy demand — NESO GB national grid
Half-hourly national demand files must be downloaded from the [NESO data portal](https://www.neso.energy/data-portal/historic-demand-data) for years 2018–2022 and placed in the `ihepc_ngeso/data/` directory:

```
ihepc_ngeso/data/demanddata_2018.csv
ihepc_ngeso/data/demanddata_2019.csv
ihepc_ngeso/data/demanddata_2020.csv
ihepc_ngeso/data/demanddata_2021.csv
ihepc_ngeso/data/demanddata_2022.csv
```

As with SPY, precomputed caches in `ihepc_ngeso/game_results_energy/` allow figure reproduction without the raw data files.

---

## Reproducing the paper figures

### Synthetic experiments (no data required)

```bash
# Ground truth validation — effect recovery vs. n
# Full run (≈ 30 min on 32 cores):
python synthetic_experiments/gt_validation.py --n_runs 30 --n_jobs 32

# Quick smoke test (1 seed, reduced n grid):
python synthetic_experiments/gt_validation.py --quick

# Regenerate figures only from existing cache:
python synthetic_experiments/gt_validation.py --plots_only

# Sobol theorem validation
python synthetic_experiments/thm2_validation.py

# Kernel guidance and ranking preservation
python synthetic_experiments/kernel_guidance.py
python synthetic_experiments/ranking_preservation.py
```

### Real-data experiments

```bash
# SPY intraday volatility (figures from cache, no bar data needed)
python spy/spy_hfd.py

# Energy demand (figures from cache, no raw demand files needed)
python ihepc_ngeso/energy_hfd.py
```

---

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn ngboost torch joblib yfinance ucimlrepo
```

No GPU is required; all scripts default to CPU. Tested with Python 3.10.

---

## Cached game results

Computing game values from scratch requires evaluating $2^p$ coalition values per game and is computationally expensive. All precomputed `.npz` cache files are provided in `spy/game_results/` and `ihepc_ngeso/game_results_energy/`. Scripts load these automatically when present.

Each script defines `CACHE_VERSION_*` string constants that tag the cache filenames. **Do not change these constants** unless intentionally recomputing with new settings, as doing so will bypass the provided caches.

---

## Game formulations

The framework supports three cooperative game types:

**Prediction game** *(local, instance-specific)*  
$v(S)(t) = \mathbb{E}_{X_{-S}}[F(x^*_S, X_{-S})(t)]$ — how much does each feature shift the predicted trajectory for a specific input $x^*$?

**Sensitivity game** *(global)*  
$v(S)(t) = \mathrm{Var}_{X_S}[\mathbb{E}_{X_{-S}}[F(X_S, X_{-S})(t)]]$ — how much of total trajectory variance is explained by $S$?  
Pure = closed Sobol; Partial = functional Shapley sensitivity; Full = total Sobol.

**Risk game** *(global, SAGE/PFI sign convention)*  
$v(S)(t) = \mathbb{E}[(Y(t){-}\mu(t))^2] - \mathbb{E}[(Y(t){-}\mathbb{E}_{X_{-S}}[F(X_S, X_{-S})(t)])^2]$ — how much does knowing $S$ reduce prediction error?  
Pure = pure risk reduction; Partial = SAGE; Full = PFI.

---

## Output kernels

The kernel $K$ controls how temporal context is weighted when aggregating attribution curves. All kernels are applied in row-normalised form so that attribution magnitudes remain comparable across kernel choices.

| Kernel | When to use |
|--------|-------------|
| Identity $K(t,s) = \delta(t{-}s)$ | Pointwise attribution; no temporal context |
| OU $K(t,s) = e^{-\|t-s\|/\ell}$ | Local symmetric smoothing over a neighbourhood |
| Causal $K(t,s) = e^{-(t-s)/\ell} \cdot \mathbf{1}_{t \geq s}$ | Forward-only; no anticipation of future events |
| Correlation $K(t,s) = \mathrm{Corr}(F(X)(t), F(X)(s))$ | Data-adaptive; no bandwidth parameter needed |
| Periodic $K(t,s) = e^{-2\sin^2(\pi\|t-s\|/p)/\ell^2}$ | Recurring daily or seasonal patterns |

---

