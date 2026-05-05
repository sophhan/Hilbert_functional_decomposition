"""
Working Example ICU
=======================

Self-contained script that generates the data files consumed by the TikZ
intro figure (intro_figure.tex):

    curves_id.csv   -- t, x1, x2, x3   (identity-kernel time-resolved curves)
    curves_co.csv   -- t, x1, x2, x3   (correlation-kernel time-resolved curves)
    values.tex      -- LaTeX macros holding bar heights and y-axis maxima

ICU early-warning toy model
----------------------------
Three features X1, X2, X3 affect a function-valued output F(x)(t) over
t in [0, 24] h:

    F(x)(t) = X1 * exp(-0.2*t)             # baseline recovery trend
            + X2 * exp(-(t-10)^2 / 2)      # early shock at t=10h
            + X3 * exp(-(t-18)^2 / 2)      # late deterioration at t=18h

Feature distribution: X_i ~ Uniform[0, 1], independent.
    E[X_i] = 0.5,  Var[X_i] = 1/12.

Centred (zero-mean-background) effect of feature i at value x_i*:
    e_i(t) = (x_i* - mu) * phi_i(t)

This is the instantaneous (identity-kernel) attribution for feature i.

Two output kernels are demonstrated
-------------------------------------
1. Identity kernel  K(t, s) = delta(t - s)
   Effect: (K e_i)(t) = e_i(t)  — pointwise attribution, no temporal coupling.

2. Correlation-aware kernel
       K(t, s) = Cov(F(X)(t), F(X)(s)) / (std(F(X)(t)) * std(F(X)(s)))
   where:
       Cov(F(X)(t), F(X)(s)) = sum_i Var(X_i) * phi_i(t) * phi_i(s)

   This correlation matrix ties together time points that co-vary across the
   feature distribution — i.e. phases that are driven by the same underlying
   basis function. It is applied in row-normalised form so that each time
   point t receives a weighted average of the instantaneous effect over the
   temporal neighbourhood defined by the model's own covariance structure.

   Effect: (K e_i)(t) = [K(t,:) @ e_i] / [sum_s K(t,s) * dt]
   This is the "phase-aware redistributed" attribution.

Three aggregation levels are computed
--------------------------------------
- Time-resolved:    curve (K e_i)(t) for all t          -> written to CSV
- Time-specific:    (K e_i)(t0) at a single focus point  -> written to values.tex
- Time-aggregated:  integral_T (K e_i)(t) dt             -> written to values.tex

The values.tex file defines LaTeX \\newcommand macros that are directly
\\input'd by the TikZ figure source, avoiding any manual copy-paste of
numerical values into the LaTeX source.

Output files (written to plots/synthetic_experiments/working_example_ICU/)
--------------------------------------------------------------------------
    curves_id.csv   — 240-row CSV: t, e1, e2, e3 under identity kernel
    curves_co.csv   — 240-row CSV: t, e1, e2, e3 under correlation kernel
    values.tex      — LaTeX macros for bar heights and y-axis limits

Usage
-----
    python generate_figure_data.py
"""

import os

import numpy as np


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUT_DIR = os.path.join("plots", "synthetic_experiments", "working_example_ICU")


# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

T_MAX    = 24.0    # time horizon in hours
T_POINTS = 240     # number of grid points (resolution for curves and integrals)
T0_FOCUS = 6.0     # focus time point t0 for the time-specific bar panels

X_STAR = (0.8, 0.9, 0.7)   # specific input x* being explained
MU     = 0.5                # E[X_i] for X_i ~ Uniform[0, 1]
VAR    = 1.0 / 12.0         # Var[X_i] for X_i ~ Uniform[0, 1]


# ---------------------------------------------------------------------------
# Basis functions defining the ICU model
# ---------------------------------------------------------------------------

def phi1(tt: np.ndarray) -> np.ndarray:
    """
    Basis function for X1: slow exponential decay.
    Models a baseline recovery trend that decays over the full 24h window.
    """
    return np.exp(-0.2 * tt)


def phi2(tt: np.ndarray) -> np.ndarray:
    """
    Basis function for X2: sharp Gaussian peak at t = 10h.
    Models an early shock event with localised temporal influence.
    """
    return np.exp(-0.5 * (tt - 10.0) ** 2)


def phi3(tt: np.ndarray) -> np.ndarray:
    """
    Basis function for X3: Gaussian peak at t = 18h.
    Models a late deterioration event in the final hours of the window.
    """
    return np.exp(-0.5 * (tt - 18.0) ** 2)


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

def correlation_kernel(t_grid: np.ndarray) -> np.ndarray:
    """
    Construct the output-correlation kernel induced by the ICU toy model.

    The kernel is derived from the model's output covariance:
        Cov(F(X)(t), F(X)(s)) = Var(X) * sum_i phi_i(t) * phi_i(s)

    Normalised to a correlation matrix (unit diagonal):
        K(t, s) = Cov(F(X)(t), F(X)(s)) / sqrt(Var(F(X)(t)) * Var(F(X)(s)))

    This kernel ties together time points that co-vary in the model's output
    distribution — i.e. times driven by the same underlying basis functions.
    It is data-adaptive (no bandwidth parameter) and respects the model's
    own temporal structure.

    Parameters
    ----------
    t_grid : ndarray of shape (T,)

    Returns
    -------
    K : ndarray of shape (T, T), symmetric, unit diagonal
    """
    p1  = phi1(t_grid)
    p2  = phi2(t_grid)
    p3  = phi3(t_grid)
    cov = VAR * (np.outer(p1, p1) + np.outer(p2, p2) + np.outer(p3, p3))
    std = np.sqrt(np.diag(cov))
    std = np.where(std < 1e-12, 1.0, std)   # avoid division by zero at tails
    return cov / np.outer(std, std)


def apply_kernel(effect: np.ndarray, K: np.ndarray, dt: float) -> np.ndarray:
    """
    Apply kernel K to an instantaneous effect curve in row-normalised form.

    Computes the "phase-aware redistributed" attribution:
        (Ke)(t) = [sum_s K(t,s) * e(s) * dt] / [sum_s K(t,s) * dt]

    Row normalisation ensures that:
      - Attribution magnitudes remain comparable across kernels.
      - A constant input e(t) = c is mapped to (Ke)(t) = c for all kernels.
      - The result is a weighted temporal average of e, where the weights
        at each t are given by the kernel row K(t,:).

    For the identity kernel this is a no-op, so it is not called for that case.

    Parameters
    ----------
    effect : ndarray of shape (T,)   — instantaneous attribution e_i(t)
    K      : ndarray of shape (T, T) — kernel matrix
    dt     : float                   — time step

    Returns
    -------
    ke : ndarray of shape (T,)   — kernel-redistributed attribution (Ke)(t)
    """
    row_sum = K.sum(axis=1, keepdims=True) * dt
    row_sum = np.where(np.abs(row_sum) < 1e-12, 1.0, row_sum)
    return (K / row_sum) @ effect * dt


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Execute the full data generation pipeline:
        1. Compute instantaneous (identity-kernel) effects e_i(t)
        2. Apply correlation kernel to get redistributed effects
        3. Write time-resolved curves to CSV files
        4. Compute time-specific (at t0) and time-aggregated scalar values
        5. Write all scalar values as LaTeX \\newcommand macros to values.tex

    All output files are written to OUT_DIR:
        plots/synthetic_experiments/working_example_ICU/
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    t  = np.linspace(0.0, T_MAX, T_POINTS)
    dt = float(t[1] - t[0])

    # ------------------------------------------------------------------
    # Centred instantaneous effects: e_i(t) = (x*_i - mu) * phi_i(t)
    # These are the identity-kernel attributions (no temporal coupling).
    # ------------------------------------------------------------------
    e1 = (X_STAR[0] - MU) * phi1(t)
    e2 = (X_STAR[1] - MU) * phi2(t)
    e3 = (X_STAR[2] - MU) * phi3(t)

    # ------------------------------------------------------------------
    # Correlation-kernel redistributed attribution
    # Each e_i is convolved with the correlation kernel (row-normalised),
    # spreading effect mass into the temporal neighbourhood of each t.
    # ------------------------------------------------------------------
    K     = correlation_kernel(t)
    e1_co = apply_kernel(e1, K, dt)
    e2_co = apply_kernel(e2, K, dt)
    e3_co = apply_kernel(e3, K, dt)

    # ------------------------------------------------------------------
    # Write time-resolved curves to CSV
    # Each row: t, e_X1(t), e_X2(t), e_X3(t)
    # ------------------------------------------------------------------
    np.savetxt(
        os.path.join(OUT_DIR, "curves_id.csv"),
        np.column_stack([t, e1, e2, e3]),
        header="t,x1,x2,x3", comments="", delimiter=",", fmt="%.6f",
    )
    np.savetxt(
        os.path.join(OUT_DIR, "curves_co.csv"),
        np.column_stack([t, e1_co, e2_co, e3_co]),
        header="t,x1,x2,x3", comments="", delimiter=",", fmt="%.6f",
    )

    # ------------------------------------------------------------------
    # Time-specific attribution at the focus point t0
    # ------------------------------------------------------------------
    idx0    = int(np.argmin(np.abs(t - T0_FOCUS)))
    spec_id = [float(e1[idx0]),    float(e2[idx0]),    float(e3[idx0])]
    spec_co = [float(e1_co[idx0]), float(e2_co[idx0]), float(e3_co[idx0])]

    # ------------------------------------------------------------------
    # Time-aggregated attribution: integral_T e_i(t) dt
    # Computed via the trapezoidal rule.
    # ------------------------------------------------------------------
    agg_id = [
        float(np.trapezoid(e1,    dx=dt)),
        float(np.trapezoid(e2,    dx=dt)),
        float(np.trapezoid(e3,    dx=dt)),
    ]
    agg_co = [
        float(np.trapezoid(e1_co, dx=dt)),
        float(np.trapezoid(e2_co, dx=dt)),
        float(np.trapezoid(e3_co, dx=dt)),
    ]

    # ------------------------------------------------------------------
    # Y-axis maxima for the TikZ plots (with 30% headroom)
    # ------------------------------------------------------------------
    ymax_curve_id = float(np.max(np.abs([e1,    e2,    e3])))
    ymax_curve_co = float(np.max(np.abs([e1_co, e2_co, e3_co])))

    # ------------------------------------------------------------------
    # Console summary for verification
    # ------------------------------------------------------------------
    print(f"t0 = {T0_FOCUS:g} h")
    print(f"  spec_id (Identity    @ t0) = {spec_id}")
    print(f"  spec_co (Correlation @ t0) = {spec_co}")
    print(f"  agg_id  (Identity,    integrated over T) = {agg_id}")
    print(f"  agg_co  (Correlation, integrated over T) = {agg_co}")

    # ------------------------------------------------------------------
    # Write LaTeX macro file
    # Defines \newcommand macros for all bar heights and y-axis limits.
    # These are \input'd directly by intro_figure.tex to avoid manual
    # copy-paste of numerical values into the LaTeX source.
    #
    # Naming convention:
    #   spec{ID|CO}{a|b|c}   — time-specific values for X1/X2/X3
    #   agg{ID|CO}{a|b|c}    — time-aggregated values for X1/X2/X3
    #   ymax{Curve|Spec|Agg}{ID|CO} — y-axis upper limits with headroom
    # ------------------------------------------------------------------
    with open(os.path.join(OUT_DIR, "values.tex"), "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by working_example_ICU.py — do not edit by hand.\n")
        # Time-specific bar heights
        for name, value in zip(["specIDa", "specIDb", "specIDc"], spec_id):
            fh.write(f"\\newcommand{{\\{name}}}{{{value:.4f}}}\n")
        for name, value in zip(["specCOa", "specCOb", "specCOc"], spec_co):
            fh.write(f"\\newcommand{{\\{name}}}{{{value:.4f}}}\n")
        # Time-aggregated bar heights
        for name, value in zip(["aggIDa",  "aggIDb",  "aggIDc"],  agg_id):
            fh.write(f"\\newcommand{{\\{name}}}{{{value:.4f}}}\n")
        for name, value in zip(["aggCOa",  "aggCOb",  "aggCOc"],  agg_co):
            fh.write(f"\\newcommand{{\\{name}}}{{{value:.4f}}}\n")
        # Y-axis limits with headroom factors
        fh.write(f"\\newcommand{{\\ymaxCurveID}}{{{ymax_curve_id * 1.30:.4f}}}\n")
        fh.write(f"\\newcommand{{\\ymaxCurveCO}}{{{ymax_curve_co * 1.30:.4f}}}\n")
        fh.write(f"\\newcommand{{\\ymaxSpecID}}{{{max(spec_id) * 1.40:.4f}}}\n")
        fh.write(f"\\newcommand{{\\ymaxSpecCO}}{{{max(spec_co) * 1.40:.4f}}}\n")
        fh.write(f"\\newcommand{{\\ymaxAggID}}{{{max(agg_id) * 1.30:.4f}}}\n")
        fh.write(f"\\newcommand{{\\ymaxAggCO}}{{{max(agg_co) * 1.30:.4f}}}\n")

    print(f"\nAll outputs written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()