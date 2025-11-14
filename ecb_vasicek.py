"""
Fetch ECB yield curve from local CSV, calibrate Vasicek model per date, save parameters, and plot.
Assumes ecb_data.csv exists with TIME_PERIOD index and columns: ecb_0, ecb_3m, ecb_6m, ecb_1y, ...

Usage:
  python ecb_vasicek.py --start 2019-10-17 --end 2024-12-31 --sample monthly
"""
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# -------------------------------
# Vasicek Model Helpers
# -------------------------------
def parse_maturities(columns: List[str]) -> np.ndarray:
    """Extract maturities in years from column names like 'ecb_3m', 'ecb_1y'."""
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
    return np.array(sorted(taus), dtype=float)  # Sort for consistency


def vasicek_zero_price(tau: np.ndarray, r0: float, kappa: float, theta: float, sigma: float) -> np.ndarray:
    """Zero-coupon bond price under Vasicek model."""
    k = np.maximum(kappa, 1e-8)
    B = (1.0 - np.exp(-k * tau)) / k
    A = (B - tau) * (theta - (sigma**2) / (2.0 * k**2)) - (sigma**2) * (B**2) / (4.0 * k)
    return np.exp(A - B * r0)


def yields_from_prices(tau: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Continuously compounded yield: Y = -ln(P)/tau."""
    tau_safe = np.maximum(tau, 1e-8)
    return -np.log(np.clip(prices, 1e-8, None)) / tau_safe


def obj_func(params, tau, ecb_yields_pct, r0_fixed):
    r0, kappa, theta, sigma = params
    penalty = 1e6 * (r0 - r0_fixed) ** 2  # Enforce r0 = ecb_0
    P = vasicek_zero_price(tau, r0, kappa, theta, sigma)
    y_model = yields_from_prices(tau, P)
    y_target = ecb_yields_pct / 100.0
    return float(np.sum((y_model - y_target) ** 2) + penalty)


def calibrate_row(row: pd.Series, taus: np.ndarray) -> Optional[Dict]:
    """
    Calibrate Vasicek (kappa, theta, sigma) for one date.
    r0 is fixed to ecb_0 value.
    """
    # Build expected column names
    def name_from_tau(t):
        if t < 1:
            m = int(round(t * 12))
            return f"ecb_{m}m"
        else:
            y = int(round(t))
            return f"ecb_{y}y"

    expected_cols = [name_from_tau(t) for t in taus]
    available_cols = [c for c in expected_cols if c in row.index and pd.notna(row[c])]
    if len(available_cols) < 3:
        return None  # Need at least 3 points + r0

    # Extract r0 and yields
    if "ecb_0" not in row.index or pd.isna(row["ecb_0"]):
        return None
    r0_fixed = float(row["ecb_0"]) / 100.0
    ecb_yields_pct = row[available_cols].astype(float).to_numpy()

    # Subset taus to match available yields
    tau_used = np.array([t for t, col in zip(taus, expected_cols) if col in available_cols])

    # Initial guess
    mean_yield = np.mean(ecb_yields_pct[ecb_yields_pct > 0]) / 100.0 if np.any(ecb_yields_pct > 0) else 0.02
    x0 = np.array([r0_fixed, 0.3, mean_yield, 0.015])

    bounds = [
        (r0_fixed, r0_fixed),   # r0 fixed
        (1e-6, 5.0),            # kappa
        (-0.1, 0.3),            # theta
        (1e-6, 0.5),            # sigma
    ]

    res = minimize(
        obj_func, x0,
        args=(tau_used, ecb_yields_pct, r0_fixed),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 3000, "ftol": 1e-9}
    )

    if not res.success:
        return None

    r0, kappa, theta, sigma = res.x

    # Compute RMSE for quality check
    P = vasicek_zero_price(tau_used, r0, kappa, theta, sigma)
    y_model = yields_from_prices(tau_used, P)
    y_target = ecb_yields_pct / 100.0
    rmse = np.sqrt(np.mean((y_model - y_target) ** 2))

    if rmse > 0.01:  # >100 bps → likely bad fit
        return None

    return {
        "r0": r0,
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rmse": rmse,
        "success": True
    }


# -------------------------------
# Plotting
# -------------------------------
def plot_fit_example(date: pd.Timestamp, row: pd.Series, taus: np.ndarray, params: dict):
    """Plot ECB vs Vasicek fit for one date."""
    def name_from_tau(t):
        if t < 1:
            m = int(round(t * 12))
            return f"ecb_{m}m"
        else:
            return f"ecb_{int(round(t))}y"

    cols = [name_from_tau(t) for t in taus]
    available = [c for c in cols if c in row.index and pd.notna(row[c])]
    if len(available) < 2:
        return
    tau_used = np.array([t for t, c in zip(taus, cols) if c in available])
    y_target_pct = row[available].astype(float).to_numpy()

    P = vasicek_zero_price(tau_used, params["r0"], params["kappa"], params["theta"], params["sigma"])
    y_model_pct = yields_from_prices(tau_used, P) * 100.0

    plt.figure(figsize=(8, 5))
    plt.plot(tau_used, y_target_pct, 'o-', label="ECB Yield", markersize=6)
    plt.plot(tau_used, y_model_pct, '--s', label="Vasicek Fit", markersize=5)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Yield (%)")
    plt.title(f"Vasicek Calibration — {date.date()}\nRMSE: {params['rmse']:.1f} bps")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_surface(ecb_df: pd.DataFrame):
    """3D surface plot of ECB yield curve."""
    cols = [c for c in ecb_df.columns if c != "ecb_0"]
    taus = parse_maturities(cols)
    if len(taus) == 0:
        return

    Z = ecb_df[cols].T.values  # (maturities, times)
    T_idx = np.arange(Z.shape[1])
    Tau_mesh, T_mesh = np.meshgrid(taus, T_idx)

    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T_mesh, Tau_mesh, Z, cmap='viridis', edgecolor='k', alpha=0.8, linewidth=0.2)
    ax.set_title("ECB Spot Rate Yield Surface (%)")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Yield (%)")
    fig.colorbar(surf, shrink=0.6, aspect=12)
    plt.tight_layout()
    plt.show()


# -------------------------------
# Sampling
# -------------------------------
def sample_dates(index: pd.DatetimeIndex, mode: str) -> pd.DatetimeIndex:
    if mode == "all":
        return index
    elif mode == "monthly":
        return index.to_series().groupby([index.year, index.month]).apply(lambda x: x.iloc[0]).index
    elif mode == "quarterly":
        return index.to_series().groupby([index.year, index.quarter]).apply(lambda x: x.iloc[0]).index
    raise ValueError("sample must be 'all', 'monthly', or 'quarterly'")


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Calibrate Vasicek model on ECB yield curve (local CSV)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.today().strftime("%Y-%m-%d"), help="End date")
    parser.add_argument("--sample", type=str, default="monthly", choices=["all", "monthly", "quarterly"])
    parser.add_argument("--no-plots", action="store_true", help="Skip plots (for CI/headless)")
    args = parser.parse_args()

    # Load local CSV
    csv_path = "ecb_data.csv"
    if not pd.io.common.file_exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Please provide the ECB yield curve CSV.")

    print(f"Loading {csv_path}...")
    ecb_df = pd.read_csv(csv_path, index_col="TIME_PERIOD", parse_dates=True)
    ecb_df.index.name = "TIME_PERIOD"

    # Filter date range
    mask = (ecb_df.index >= args.start) & (ecb_df.index <= args.end)
    ecb_df = ecb_df.loc[mask]
    if ecb_df.empty:
        raise ValueError("No data in selected date range.")

    # Parse maturities
    taus = parse_maturities(ecb_df.columns)
    if len(taus) == 0:
        raise ValueError("No maturity columns found (expected ecb_3m, ecb_1y, etc.)")

    # Sample dates
    dates = sample_dates(ecb_df.index, args.sample)
    print(f"Calibrating Vasicek on {len(dates)} dates ({args.sample} sampling)...")

    results = []
    example_dates = [dates[0], dates[len(dates)//2], dates[-1]] if len(dates) > 2 else [dates[0]]

    for d in tqdm(dates, desc="Calibrating", unit="date"):
        row = ecb_df.loc[d]
        params = calibrate_row(row, taus)
        if params is None:
            continue
        params["date"] = d
        results.append(params)

        # Plot example fits
        if not args.no_plots and d in example_dates:
            plot_fit_example(d, row, taus, params)

    if not results:
        print("No successful calibrations.")
        return

    params_df = pd.DataFrame(results).set_index("date").sort_index()
    params_df.to_csv("vasicek_params.csv")
    print(f"\nSaved vasicek_params.csv → {params_df.shape[0]} dates, {params_df.shape[1]} params")
    print(f"   Mean kappa: {params_df['kappa'].mean():.3f}, theta: {params_df['theta'].mean()*100:.2f}%, sigma: {params_df['sigma'].mean():.4f}")

    if not args.no_plots:
        plot_surface(ecb_df)


if __name__ == "__main__":
    main()
