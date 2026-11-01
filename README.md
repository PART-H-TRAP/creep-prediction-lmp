# Creep Prediction — Larson-Miller Parameter (LMP)

Predicts creep rupture life from stress and temperature using the
Larson-Miller time-temperature parameter:

```
LMP = T * (C + log10(t_r))      [T in Kelvin, t_r in hours]
```

Creep rupture is thermally activated, so T and log10(t_r) combine through a
single material constant C into one number that collapses stress-rupture
tests from many (stress, temperature) combinations onto a single master
curve. Fit that curve once from short tests, then read off rupture life at
any stress/temperature pair.

## Data

`creep_data.csv` — 31 public creep-rupture tests on a Ni-base superalloy,
650–1095 °C, 10–480 MPa (Hertzberg, *Deformation and Fracture Mechanics of
Engineering Materials*, Problem 5.3). Swap this file for a larger dataset
(e.g. the [NIMS Creep Data Sheets](https://cds.nims.go.jp/)) without
touching the code.

## Method

- `C` and the master-curve polynomial degree (1–3) are fit by minimizing
  **leave-one-out cross-validated (LOOCV) error**, not training error —
  each point is predicted from a model that never saw it, which is the
  honest accuracy number for an extrapolation method like this.
- Result on this dataset: **C = 27.53** (within the literature range of
  ~20 for general heat-resistant steels to ~30 for high creep-resistant
  alloys), cubic master curve, **scatter factor ≈ 1.9×** — a prediction is
  typically within about a factor of 2 of the true rupture life, consistent
  with published LMP accuracy benchmarks.
- Predictions outside the training LMP range are flagged as extrapolations.

## Mechanism diagnostics

Each prediction also reports:

- **Apparent stress exponent n** — the local slope of log10(rupture time) vs
  log10(stress) at fixed T. Since minimum creep rate is roughly
  proportional to 1/t_r (Monkman-Grant), n approximates the exponent in
  Norton's law (rate ~ stress^n), placing the material on the standard
  deformation-mechanism map: n < 1.5 diffusional creep, 1.5–3 grain-boundary
  sliding / mixed, 3–8 dislocation climb-controlled power-law creep, >8
  power-law breakdown.
- **Apparent activation energy Q** = 2.303·R·LMP, following directly from
  the Arrhenius assumption the LMP rests on.

Both are read off the fitted curve, not measured independently. Q scales
directly with the fitted C (which is optimized for predictive accuracy,
not pinned to known diffusion data), so treat Q as approximate — n is the
more robust diagnostic.

## Usage

```bash
pip install numpy scipy matplotlib
python3 lmp_creep_prediction.py --plot
python3 lmp_creep_prediction.py --stress 300 --temp 700
```

`lmp_master_curve.png` shows the fitted master curve against the 31 data
points.

## Using a different material

The code is dataset-agnostic — swap `--data` for any CSV with the same
three columns (`temperature_C`, `stress_MPa`, `rupture_time_h`), no code
changes needed:

```bash
python3 lmp_creep_prediction.py --data creep_data_304SS.csv --stress 100 --temp 650
```

`creep_data_304SS.csv` is a second public dataset included as a worked
example: 9 points for 304 stainless steel at 593–760 °C (NASA TM, Fritz &
Koster, *Tensile, Creep-Rupture, and Poisson's Ratio Properties of
Uncoated and Coated Engineering Alloys*, NASA CR-135314, 1977). Run as
above and the model refits cleanly — C = 15.91, scatter factor ≈1.15×
(tighter than the superalloy fit, since these are already-smoothed
design-curve values rather than raw scattered test points), apparent
Q ≈ 337 kJ/mol, close to the known self-diffusion energy for austenitic
Fe-Cr-Ni alloys (~280–350 kJ/mol) — a useful cross-check that the method
is behaving physically on a dataset where the fitted C lands near the
textbook default (~20) rather than at the high end.
