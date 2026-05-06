"""
Anonymize SPY game-result caches for public release.
=====================================================

The cached game results in `spy/game_results/` contain stored explicand
feature vectors (`x_inst`) for ~150 specific trading dates. Two of these
six features are derived from Polygon.io's intraday bar data:

    Index 1: overnight_ret  (close-to-open log return — derived from bars)
    Index 4: trailing_rv    (5-day mean of |log returns| — derived from bars)

Polygon's Stocks Starter license permits sharing of derived data, but to
stay clearly within the spirit of the agreement we strip these two fields
from every cached `x_inst` before publishing the repository.

The remaining four fields are publicly available regardless of Polygon:

    Index 0: vix_prev       (VIX from Yahoo Finance / CBOE — public)
    Index 2: ann_indicator  (FOMC / CPI / NFP schedule — public)
    Index 3: day_of_week    (calendar fact)
    Index 5: month          (calendar fact)

Effect on figures
-----------------
- fig0, fig1, fig2, fig4 are fully reproducible (do not use stripped fields).
- fig3 (PDP plots) loses x-axis values for the two stripped features. The
  cache-only branch in `load_per_instance_effects_pdp` will still load
  the cached attribution trajectories; the PDP binning along these two
  features will be incorrect, but plotting the remaining four is unaffected.

Usage
-----
Run once before pushing the repository:

    python anonymize_caches.py

Idempotent: stripped fields are written as NaN, and re-running the script
on already-stripped caches is a no-op (NaN replaced with NaN).

The script operates in-place. Make a backup of `game_results/` first if
you want to preserve the unaltered caches locally.
"""

import glob
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join('spy', 'game_results')

# Feature indices in DAY_FEATURE_NAMES that are derived from Polygon bars
# and must be stripped from x_inst before public release.
POLYGON_DERIVED_IDX = [1, 4]   # overnight_ret, trailing_rv

# Names corresponding to those indices (for the report).
POLYGON_DERIVED_NAMES = ['overnight_ret', 'trailing_rv']


# ---------------------------------------------------------------------------
# Anonymization logic
# ---------------------------------------------------------------------------

def anonymize_cache_file(path):
    """
    Load a single .npz cache, replace Polygon-derived fields in x_inst with
    NaN, and save back in place. Returns True if any modification was made,
    False if the cache had nothing to scrub.
    """
    d = dict(np.load(path, allow_pickle=True))

    if 'x_inst' not in d:
        return False

    x = d['x_inst'].astype(float).copy()

    # Single-vector caches store a flat (p,) array; avoid touching anything
    # with unexpected shape.
    if x.ndim != 1:
        return False

    modified = False
    for i in POLYGON_DERIVED_IDX:
        if i < len(x) and not np.isnan(x[i]):
            x[i]     = np.nan
            modified = True

    if not modified:
        return False

    d['x_inst'] = x
    np.savez_compressed(path, **d)
    return True


def main():
    if not os.path.isdir(CACHE_DIR):
        print('Cache directory not found: {}'.format(CACHE_DIR))
        print('(Run this script from the repository root.)')
        sys.exit(1)

    paths = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npz')))
    if not paths:
        print('No .npz files found in {}'.format(CACHE_DIR))
        sys.exit(0)

    print('Anonymizing {} cache files in {} ...'.format(len(paths), CACHE_DIR))
    print('Stripping fields: {}'.format(', '.join(POLYGON_DERIVED_NAMES)))
    print()

    n_modified = 0
    n_skipped  = 0
    for path in paths:
        rel = os.path.relpath(path, CACHE_DIR)
        if anonymize_cache_file(path):
            n_modified += 1
            print('  modified  {}'.format(rel))
        else:
            n_skipped += 1
            print('  unchanged {}'.format(rel))

    print()
    print('Done. {} modified, {} unchanged.'.format(n_modified, n_skipped))
    print('Caches in {} are now safe for public release.'.format(CACHE_DIR))


if __name__ == '__main__':
    main()