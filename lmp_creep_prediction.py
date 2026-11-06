#!/usr/bin/env python3
"""
Larson-Miller Parameter (LMP) creep rupture life prediction.

LMP = T * (C + log10(t_r))  [T in Kelvin, t_r in hours]

Creep rupture is thermally activated, so T and log10(t_r) combine through a
single material constant C into one number that collapses stress-rupture
tests from many (stress, temperature) combinations onto one curve. Fit that
curve once from short tests, then read off rupture life at any stress/
temperature pair.

Data: 31 public creep-rupture tests on a superalloy (grade not stated in the
source), 650-1095 degC, 10-480 MPa (Hertzberg, "Deformation and Fracture
Mechanics of Engineering Materials", Problem 5.3). Swap creep_data.csv for a
larger set (e.g. NIMS Creep Data Sheets) without touching the code below.
"""

import argparse
import csv

import numpy as np
from scipy.optimize import minimize_scalar


def load_data(path):
    with open(path, newline="") as f:
        rows = [tuple(map(float, r)) for r in csv.reader(f) if r and r[0] != "temperature_C"]
    T_C, stress, t_r = map(np.array, zip(*rows))
    return T_C + 273.15, stress, t_r  # Kelvin, MPa, hours


def lmp(T_K, t_r, C):
    return T_K * (C + np.log10(t_r))


def fit_curve(T_K, stress, t_r, C, degree):
    """LMP as a polynomial in log10(stress); inverted later to get t_r."""
    return np.poly1d(np.polyfit(np.log10(stress), lmp(T_K, t_r, C), degree))


def predict_log10_tr(stress, T_K, C, poly):
    return poly(np.log10(stress)) / T_K - C


def loocv_rmse(T_K, stress, t_r, C, degree):
    """Leave-one-out cross-validated error, in log10(hours). This -- not
    training error -- is the honest accuracy number: each point is predicted
    from a model that never saw it."""
    errors = []
    for i in range(len(t_r)):
        keep = np.arange(len(t_r)) != i
        poly = fit_curve(T_K[keep], stress[keep], t_r[keep], C, degree)
        pred = predict_log10_tr(stress[i], T_K[i], C, poly)
        errors.append(pred - np.log10(t_r[i]))
    return float(np.sqrt(np.mean(np.square(errors))))


def fit_model(T_K, stress, t_r, degrees=(1, 2, 3)):
    """Pick (C, degree) that minimizes LOOCV error. C is optimized per
    degree rather than fixed at the textbook default of 20, since the
    best-fit value is material-specific (literature reports ~20 for general
    heat-resistant steels up to ~30 for high creep-resistant alloys)."""
    best = None
    for d in degrees:
        res = minimize_scalar(lambda C: loocv_rmse(T_K, stress, t_r, C, d), bounds=(10, 30), method="bounded")
        rmse = loocv_rmse(T_K, stress, t_r, res.x, d)
        if best is None or rmse < best[2]:
            best = (res.x, d, rmse)
    return best  # C, degree, loocv_rmse


def nested_loocv_rmse(T_K, stress, t_r):
    """Unbiased accuracy: C and degree are re-selected inside every fold, so the
    held-out point never influences the choice of C or degree. The plain LOOCV
    returned by fit_model is slightly optimistic because it is also the number
    that C and degree were tuned to minimize."""
    errors = []
    for i in range(len(t_r)):
        keep = np.arange(len(t_r)) != i
        C, d, _ = fit_model(T_K[keep], stress[keep], t_r[keep])
        poly = fit_curve(T_K[keep], stress[keep], t_r[keep], C, d)
        errors.append(predict_log10_tr(stress[i], T_K[i], C, poly) - np.log10(t_r[i]))
    return float(np.sqrt(np.mean(np.square(errors))))


def replicate_scatter(T_K, stress, t_r):
    """Pooled SD of log10(t_r) between repeat tests at identical (T, stress).
    This is test-to-test scatter in the material itself, so no curve fit can
    predict better than this. None if the data has no repeat tests."""
    groups = {}
    for T, s, y in zip(T_K, stress, np.log10(t_r)):
        groups.setdefault((T, s), []).append(y)
    reps = [np.array(v) for v in groups.values() if len(v) > 1]
    df = sum(len(v) - 1 for v in reps)
    return float(np.sqrt(sum(((v - v.mean()) ** 2).sum() for v in reps) / df)) if df else None


def predict(stress, T_C, C, poly, lmp_range):
    T_K = T_C + 273.15
    t_r = 10 ** predict_log10_tr(stress, T_K, C, poly)
    q = poly(np.log10(stress))
    extrapolating = not (lmp_range[0] <= q <= lmp_range[1])
    return t_r, extrapolating


R = 8.314  # J/mol/K


def creep_mechanism(stress, T_C, C, poly):
    """Read the material's apparent creep mechanism off the fitted curve.

    n = -d(log10 t_r)/d(log10 stress) at fixed T. Minimum creep rate is
    roughly proportional to 1/t_r (Monkman-Grant), so n approximates the
    stress exponent in Norton's law (rate ~ stress^n) and places the
    material on the standard deformation-mechanism map. Q = 2.303*R*LMP
    follows directly from the Arrhenius assumption LMP is built on, and is
    the apparent activation energy for rupture at that condition. Both are
    inferred from the rupture-life trend, not measured independently, so
    treat them as indicative, not as substitutes for strain-rate or
    diffusion data.
    """
    T_K = T_C + 273.15
    x, h = np.log10(stress), 0.01
    n = -(predict_log10_tr(10 ** (x + h), T_K, C, poly) - predict_log10_tr(10 ** (x - h), T_K, C, poly)) / (2 * h)
    Q = 2.303 * R * poly(x) / 1000  # kJ/mol

    if n < 1.5:
        regime = "diffusional creep (Nabarro-Herring/Coble) -- rate nearly linear in stress"
    elif n < 3:
        regime = "grain-boundary sliding / mixed diffusional-dislocation regime"
    elif n <= 8:
        regime = "dislocation climb-controlled power-law creep (classic secondary-creep regime)"
    else:
        regime = "power-law breakdown -- approaching yield-dominated plastic flow"
    return float(n), float(Q), regime


def plot(T_K, stress, t_r, C, degree, poly, rmse, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(lmp(T_K, t_r, C), stress, color="#2A6F97", zorder=3, label="Test data")
    x = np.linspace(np.log10(stress.min()) - 0.1, np.log10(stress.max()) + 0.1, 300)
    ax.plot(poly(x), 10**x, color="#C1440E", linewidth=2, label=f"Master curve (deg {degree})")
    ax.set_yscale("log")
    ax.set_xlabel(f"LMP = T(C + log10 t_r), C = {C:.2f}")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title("Larson-Miller Master Curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{prefix}_master_curve.png", dpi=160)
    plt.close(fig)
    print(f"Saved {prefix}_master_curve.png (scatter factor {10**rmse:.2f}x)")


def main():
    ap = argparse.ArgumentParser(description="Larson-Miller creep rupture prediction")
    ap.add_argument("--data", default="creep_data.csv")
    ap.add_argument("--stress", type=float)
    ap.add_argument("--temp", type=float, help="degC")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    T_K, stress, t_r = load_data(args.data)
    C, degree, _ = fit_model(T_K, stress, t_r)
    rmse = nested_loocv_rmse(T_K, stress, t_r)
    floor = replicate_scatter(T_K, stress, t_r)
    poly = fit_curve(T_K, stress, t_r, C, degree)
    lmp_range = (lmp(T_K, t_r, C).min(), lmp(T_K, t_r, C).max())

    print(f"{len(t_r)} points | C = {C:.2f} | degree {degree} | "
          f"nested LOOCV RMSE = {rmse:.3f} log10(h) -> scatter factor {10**rmse:.2f}x")
    if floor:
        print(f"Repeat tests at identical conditions scatter by {10**floor:.2f}x on their own, "
              f"so that is the best any model can do on this data.")
    print(f"Training LMP range: {lmp_range[0]:.0f}-{lmp_range[1]:.0f} (predictions outside this are extrapolations)")
    print("Note: n (stress exponent) is the reliable mechanism indicator below; Q tracks C 1:1 and C was "
          "fit for predictive accuracy, not pinned to known diffusion data, so treat Q as approximate.")

    queries = [(args.stress, args.temp)] if args.stress and args.temp else [(300, 700), (150, 760), (42, 870)]
    for s, t in queries:
        life, extrap = predict(s, t, C, poly, lmp_range)
        n, Q, regime = creep_mechanism(s, t, C, poly)
        flag = "  [EXTRAPOLATION]" if extrap else ""
        print(f"{s:>6.1f} MPa @ {t:>6.1f} C -> {life:,.1f} h ({life/8760:.2f} yr){flag}")
        print(f"    apparent stress exponent n = {n:.1f}, activation energy Q = {Q:.0f} kJ/mol -> {regime}")

    if args.plot:
        plot(T_K, stress, t_r, C, degree, poly, rmse, "lmp")


if __name__ == "__main__":
    main()
