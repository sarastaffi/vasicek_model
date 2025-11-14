"""
Fetch ECB yield curve data, calibrate Vasicek per date, and plot results.

Usage:
  python ecb_vasicek.py --start 2019-10-17 --end 2024-12-31 --sample monthly
"""

import argparse
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# -------------------------------
# Vasicek helpers
# -------------------------------

def parse_maturities(columns: List[str]) -> np.ndarray:
    """Extract maturities (in years) from column names e.g. 'ecb_3m', 'ecb_1y'."""
    taus = []
    for c in columns:
        if c == "ecb_0":
            continue
        stem = c.replace("ecb_", "")
        if stem.endswith("m"):
            num = int(stem.replace("m", ""))
            taus.append(num / 12.0)
        elif stem.endswith("y"):
            num = int(stem.replace("y", ""))
            taus.append(float(num))
    return np.array(taus, dtype=float)

def vasicek_zero_price(tau: np.ndarray, r0: float, kappa: float, theta: float, sigma: float) -> np.ndarray:
    """
    Zero-coupon bond price under Vasicek.
    tau: maturities in years (array)
    r0 : short rate at time 0 (level)
    """
    # Avoid division by zero for very small kappa
    k = np.maximum(kappa, 1e-8)
    B = (1.0 - np.exp(-k * tau)) / k
    A = (B - tau) * (theta - (sigma ** 2) / (2.0 * (k ** 2))) - (sigma ** 2) * (B ** 2) / (4.0 * k)
    return np.exp(A - B * r0)

def yields_from_prices(tau: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Continuously-compounded yields Y = -ln(P)/tau (in decimals)."""
    tau_safe = np.maximum(tau, 1e-8)
    return -np.log(prices) / tau_safe

def obj_func(params, tau, ecb_yields_pct, r0_fixed):
    # r0 fixed through constraint -> pass in for clarity
    r0, kappa, theta, sigma = params
    # Enforce r0 = r0_fixed softly in objective (robust in some solvers)
    penalty = 1e6 * (r0 - r0_fixed) ** 2
    p = vasicek_zero_price(tau, r0, kappa, theta, sigma)
    y_model = yields_from_prices(tau, p)          # decimals
    y_target = ecb_yields_pct / 100.0             # convert percent -> decimals
    return float(np.sum((y_model - y_target) ** 2) + penalty)

def calibrate_row(row: pd.Series, taus: np.ndarray):
    """
    Calibrate kappa, theta, sigma for one date using the observed yield curve.
    Returns dict with params.
    """
    values = row.dropna().to_numpy()
    if len(values) < 3:
        return None  # not enough points

    r0_fixed = float(row["ecb_0"]) / 100.0
    # The remaining columns must be aligned with taus ordering:
    # Build ecb yields array in the same order as taus
    ecb_cols = [c for c in row.index if c != "ecb_0"]
    # Ensure sorting matches the taus order extracted from columns
    # A simple way: rebuild names from taus and map
    def name_from_tau(t):
        if t < 1:
            # months
            m = int(round(t * 12))
            return f"ecb_{m}m"
        else:
            return f"ecb_{int(round(t))}y"

    ordered_cols = [name_from_tau(t) for t in taus]
    ecb = row[ordered_cols].astype(float).to_numpy()

    # Initial guess
    x0 = np.array([r0_fixed, 0.5, 0.02, 0.1])
    bounds = [
        (r0_fixed, r0_fixed),   # r0 fixed
        (1e-6, 5.0),            # kappa
        (-0.05, 0.20),          # theta (wide)
        (1e-6, 1.0)             # sigma
    ]

    res = minimize(
        obj_func, x0, args=(taus, ecb, r0_fixed),
        method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000}
    )
    r0, kappa, theta, sigma = res.x
    return {"kappa": kappa, "theta": theta, "sigma": sigma, "r0": r0, "success": bool(res.success)}

# -------------------------------
# Plotting
# -------------------------------

def plot_surface(ecb_df: pd.DataFrame):
    """3D surface of the yield curve (percent)."""
    # Build maturity vector in years for all columns except ecb_0
    taus = parse_maturities([c for c in ecb_df.columns if c != "ecb_0"])

    # Z must be shape (n_maturities, n_times)
    Z = ecb_df[[c for c in ecb_df.columns if c != "ecb_0"]].T.values  # (M, T)
    T_idx = np.arange(Z.shape[1])               # time index
    Tau_idx = taus                              # maturity in years

    # Mesh
    T_mesh, Tau_mesh = np.meshgrid(T_idx, Tau_idx)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T_mesh, Tau_mesh, Z, edgecolor="k", alpha=0.8)
    ax.set_title("ECB Zero-Coupon Yield Surface (%)")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Yield (%)")
    fig.colorbar(surf, shrink=0.6, aspect=12)
    plt.tight_layout()
    plt.show()

def plot_fit_example(date: pd.Timestamp, row: pd.Series, taus: np.ndarray, params: dict):
    """Plot one-date fit: ECB vs Vasicek."""
    p_model = vasicek_zero_price(taus, params["r0"], params["kappa"], params["theta"], params["sigma"])
    y_model_pct = yields_from_prices(taus, p_model) * 100.0
    # Build target curve in the same order
    def name_from_tau(t):
        if t < 1:
            m = int(round(t * 12))
            return f"ecb_{m}m"
        else:
            return f"ecb_{int(round(t))}y"
    target_cols = [name_from_tau(t) for t in taus]
    y_target_pct = row[target_cols].astype(float).to_numpy()

    plt.figure(figsize=(7, 5))
    plt.plot(taus * 12, y_target_pct, label="ECB yield")
    plt.plot(taus * 12, y_model_pct, label="Vasicek (fit)")
    plt.xlabel("Maturity (months)")
    plt.ylabel("Yield (%)")
    plt.title(f"Vasicek fit — {date.date()}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------
# Sampling helper (speed)
# -------------------------------

def sample_dates(index: pd.DatetimeIndex, mode: str) -> pd.DatetimeIndex:
    if mode == "all":
        return index
    if mode == "monthly":
        return index.to_series().groupby([index.year, index.month]).head(1).index
    if mode == "quarterly":
        return index.to_series().groupby([index.year, (index.quarter)]).head(1).index
    raise ValueError("sample mode must be one of: all | monthly | quarterly")

# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2019-10-17")
    parser.add_argument("--end", type=str, default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sample", type=str, default="monthly", choices=["all", "monthly", "quarterly"])
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting (for CI/headless)")
    args = parser.parse_args()

    # Load data from local CSV (ensure TIME_PERIOD is used as the index)
    ecb_df = pd.read_csv("ecb_data.csv", index_col="TIME_PERIOD", parse_dates=True)
    # keep the index name consistent
    ecb_df.index.name = "TIME_PERIOD"

    # Calibrate
    taus = parse_maturities([c for c in ecb_df.columns if c != "ecb_0"])
    dates = sample_dates(ecb_df.index, args.sample)

    results = []
    for d in dates:
        row = ecb_df.loc[d]
        params = calibrate_row(row, taus)
        if params is None: 
            continue
        params["date"] = d
        results.append(params)
        # Optionally show a quick fit for a few dates
        if not args.no_plots and len(results) in (1, len(dates)//2, len(dates)):
            plot_fit_example(d, row, taus, params)

    params_df = pd.DataFrame(results).set_index("date").sort_index()
    params_df.to_csv("vasicek_params.csv")
    print(f"Saved vasicek_params.csv with shape {params_df.shape}")

    # 3D surface
    if not args.no_plots:
        plot_surface(ecb_df)

if __name__ == "__main__":
    main()
