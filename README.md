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

`creep_data.csv` — 31 public creep-rupture tests on a superalloy (grade
not stated in the source), 650–1095 °C, 10–480 MPa (Hertzberg, *Deformation and Fracture Mechanics of
Engineering Materials*, Problem 5.3). Swap this file for a larger dataset
(e.g. the [NIMS Creep Data Sheets](https://cds.nims.go.jp/)) without
touching the code.

## Method

- `C` and the master-curve polynomial degree (1–3) are fit by minimizing
  **leave-one-out cross-validated (LOOCV) error**, not training error —
  each point is predicted from a model that never saw it.
- The reported accuracy is **nested LOOCV**: C and degree are re-selected
  inside every fold, so the held-out point cannot influence the tuning.
  Plain LOOCV is slightly optimistic because it is also the number C and
  degree were tuned to minimize.
- Result on this dataset: **C = 27.53** (the literature reports ~20 for
  general heat-resistant steels up to ~30 for high creep-resistant alloys),
  cubic master curve, **scatter factor ≈ 1.93×** (plain LOOCV gives 1.89×).
  A prediction is typically within a factor of about 2 of the true
  rupture life.
- Repeat tests at identical stress and temperature in this dataset differ
  by a factor of 1.63× on their own (pooled over 7 repeat groups), so the
  model is already close to the best any curve fit can do on this data.
  Alternatives tried (weighted fit, Orr-Sherby-Dorn, Manson-Haferd, larger
  stress-temperature surface fits) gave no improvement that 31 points can
  resolve: the best was 1.87× against 1.93×, and Orr-Sherby-Dorn (2.4× or
  worse) and Manson-Haferd with a physically sensible convergence
  temperature (2.45× nested) were worse.
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
example: 9 points for 304 stainless steel at 593–760 °C, the stress for
rupture at 100, 300 and 1000 h (NASA Technical Reports Server, NTRS ID
19770016292). Run as above and the model refits cleanly: C = 15.91 and
scatter factor ≈ 1.22× (nested LOOCV). That is tighter than the superalloy
fit because these are tabulated design values rather than raw scattered
test points, and with only 9 points it is a demonstration, not a
validation. The apparent Q (about 337 kJ/mol) has not been checked against
independent diffusion data.
