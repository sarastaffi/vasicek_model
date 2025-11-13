# vasicek_model

Code for calibrating a single-factor Vasicek short-rate model to the ECB zero-coupon yield curve on a date-by-date basis. The repository contains a script that reads ECB zero-coupon yields from a CSV, fits Vasicek parameters per date, writes the calibrated parameter time series to CSV, and optionally plots fits and a 3D yield surface.

Contents
- `ecb_vasicek.py` — main script: reads data, calibrates per date, produces outputs and plots.
- `ecb_data.csv` — expected input CSV with ECB zero-coupon yields (see "Input data" below).
- `vasicek_params.csv` — output file written by the script after calibration.

Features
- Parses maturities from column names (`ecb_3m`, `ecb_1y`, etc.).
- Calibrates Vasicek parameters (kappa, theta, sigma) for each date, holding the short rate r0 fixed to the observed `ecb_0`.
- Saves calibrated parameters as a time series CSV.
- Plots example fits and a 3D yield surface (optional).

Dependencies
- Python 3.8+
- numpy
- pandas
- scipy
- matplotlib

Install dependencies
You can install dependencies with pip:

python -m pip install -r requirements.txt

If you don't have a requirements.txt, install directly:

python -m pip install numpy pandas scipy matplotlib

Input data: expected CSV format
- The script expects a CSV named `ecb_data.csv` in the working directory.
- The CSV must have a date column named `TIME_PERIOD` (ISO dates) which is used as the index.
- Other columns should follow the naming pattern:
  - `ecb_0` — the observed short-rate (in percent, e.g. 0.05 for 0.05%? — see note below)
  - `ecb_3m`, `ecb_6m`, `ecb_1y`, `ecb_2y`, ... — zero-coupon yields for each maturity, in percent.
- Example header (dates as strings):
  TIME_PERIOD,ecb_0,ecb_3m,ecb_6m,ecb_1y,ecb_2y
  2019-10-17,0.08,0.09,0.10,0.15,0.30


Usage
Run the calibration script from the repository root:

python ecb_vasicek.py --start 2019-10-17 --end 2024-12-31 --sample monthly

Options
- --start YYYY-MM-DD : first date to consider (default: 2019-10-17 in the script).
- --end YYYY-MM-DD : last date to consider (default: today's date).
- --sample {all,monthly,quarterly} : subsample time series to speed up calibration (default: monthly).
  - all — every observation
  - monthly — first observation of each month
  - quarterly — first observation of each quarter
- --no-plots : run without creating any plots (useful for CI or headless environments)

Outputs
- `vasicek_params.csv` — contains calibrated parameters per date. Columns: `kappa, theta, sigma, r0, success` (indexed by date).
- If plotting is enabled, example per-date fit plots will be shown for a few dates and a 3D surface plot of the yield surface will be displayed.

How the calibration works (short summary)
- r0 (the short rate at time 0) is fixed to the observed `ecb_0` value for each date.
- The script minimizes squared errors between observed continuous-compounded yields and model-implied yields from the Vasicek zero-coupon price formula.
- The optimizer uses L-BFGS-B with bounds and a strong penalty that enforces r0 ≈ observed `ecb_0`.
- Initial guesses and bounds are moderate; you may need to adjust them if calibration frequently fails or if you extend the model.
