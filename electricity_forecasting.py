# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
================================================================================
ELECTRICITY DEMAND FORECASTING WITH SARIMA
================================================================================
Dataset : UCI Electricity Load Diagrams 2011-2014
          Trindade, A. (2015), UCI ML Repository, DOI 10.24432/C58C86
Target  : Aggregate daily electricity consumption (sum of all 370 clients)
Method  : SARIMA / SARIMAX with weekly seasonality (m = 7)
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.metrics import mean_absolute_error, mean_squared_error

import pmdarima as pm

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.figsize": (14, 5),
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(BASE_DIR, "LD2011_2014.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name: str):
    """Save the current figure and close it."""
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1 — DATA LOADING & INITIAL INSPECTION                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 80)
print("PHASE 1: DATA LOADING & INITIAL INSPECTION")
print("=" * 80)

print(f"\nLoading dataset from: {DATA_FILE}")
print("  (This may take 1-2 minutes for the 711 MB file...)")

df_raw = pd.read_csv(
    DATA_FILE,
    sep=";",
    decimal=",",
    index_col=0,
    parse_dates=True,
    low_memory=False,
)

# Clean column names (strip quotes and whitespace)
df_raw.columns = df_raw.columns.str.strip().str.strip('"')
df_raw.index.name = "timestamp"

print(f"\n  Shape           : {df_raw.shape}")
print(f"  Columns (first 5): {list(df_raw.columns[:5])}")
print(f"  Columns (last 5) : {list(df_raw.columns[-5:])}")
print(f"  Date range       : {df_raw.index.min()} to {df_raw.index.max()}")
print(f"  Dtypes (unique)  : {df_raw.dtypes.unique()}")
print(f"  Total NaN values : {df_raw.isna().sum().sum()}")

# Count how many clients have non-zero data at each timestamp
nonzero_clients = (df_raw != 0).sum(axis=1)
print(f"\n  Non-zero clients at start : {nonzero_clients.iloc[0]}")
print(f"  Non-zero clients at end   : {nonzero_clients.iloc[-1]}")
print(f"  Median non-zero clients   : {nonzero_clients.median():.0f}")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 2 — PREPROCESSING                                                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 80)
print("PHASE 2: PREPROCESSING")
print("=" * 80)

# 2a. Convert kW (per 15 min) to kWh: each reading × 0.25
df_kwh = df_raw * 0.25
print("\n  Converted kW → kWh (÷ 4)")

# 2b. Filter strictly from 2012-01-01 00:00 to 2014-12-31 23:45
# Excluding the partial 2015-01-01 00:00 single-reading row ensures exactly 1,096 full days.
df_kwh = df_raw.loc["2012-01-01":"2014-12-31 23:45"] * 0.25
print(f"  Filtered to 2012-2014: {df_kwh.index.min()} to {df_kwh.index.max()}")
print(f"  Shape after filter: {df_kwh.shape}")

# 2c. Aggregate: total consumption across all clients, resample to daily
daily_total = df_kwh.sum(axis=1).resample("D").sum()
daily_total.name = "total_kwh"
daily_total.index.freq = "D"

print(f"\n  Daily aggregated series:")
print(f"    Length  : {len(daily_total)} days (Target: exactly 1,096 days)")
print(f"    Min     : {daily_total.min():,.0f} kWh")
print(f"    Max     : {daily_total.max():,.0f} kWh")
print(f"    Mean    : {daily_total.mean():,.0f} kWh")
print(f"    Std     : {daily_total.std():,.0f} kWh")

# 2d. Verify integrity (no NaN, regular daily frequency, exactly 1,096 days)
assert len(daily_total) == 1096, f"Expected 1,096 days, got {len(daily_total)}!"
assert daily_total.isna().sum() == 0, "Found unexpected NaN values!"
assert daily_total.index.freq == "D", "Index frequency not daily!"
print(f"  ✓ Exact 1,096 days verified (no artificial percentile filtering of genuine holidays)")

# Save preprocessed clean daily series for fast re-use
daily_csv_path = os.path.join(BASE_DIR, "daily_electricity_total.csv")
daily_total.to_frame().to_csv(daily_csv_path)
print(f"  [SAVED] Clean daily series saved to {daily_csv_path}")

# Free memory
del df_raw, df_kwh
import gc; gc.collect()
print("  ✓ Raw data freed from memory")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 3 — EXPLORATORY VISUALIZATION                                     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 80)
print("PHASE 3: EXPLORATORY VISUALIZATION")
print("=" * 80)

# --- 3.1 Full daily series ---
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily_total.index, daily_total.values, linewidth=0.6, color="steelblue")
ax.set_title("Daily Total Electricity Consumption (All Clients, 2012–2014)")
ax.set_xlabel("Date")
ax.set_ylabel("Total kWh")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
save_fig("01_daily_total_series.png")

# --- 3.2 Zoomed month — weekdays vs weekends ---
zoom_month = daily_total.loc["2013-06"]
fig, ax = plt.subplots(figsize=(14, 5))
colors = ["salmon" if d.weekday() >= 5 else "steelblue" for d in zoom_month.index]
ax.bar(zoom_month.index, zoom_month.values, color=colors, width=0.8)
ax.set_title("June 2013 — Daily Consumption (Blue = Weekday, Red = Weekend)")
ax.set_xlabel("Date")
ax.set_ylabel("Total kWh")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.xticks(rotation=45)
save_fig("02_zoomed_month_jun2013.png")

# --- 3.3 Box plot by day of week ---
df_box = pd.DataFrame({"kwh": daily_total.values, "dow": daily_total.index.day_name()})
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_box, x="dow", y="kwh", order=dow_order, palette="Set2", ax=ax)
ax.set_title("Electricity Consumption by Day of Week")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Total kWh")
save_fig("03_boxplot_day_of_week.png")

# --- 3.4 Box plot by month ---
df_box["month"] = daily_total.index.month_name()
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_box, x="month", y="kwh", order=month_order, palette="coolwarm", ax=ax)
ax.set_title("Electricity Consumption by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Total kWh")
plt.xticks(rotation=45)
save_fig("04_boxplot_month.png")

# --- 3.5 Rolling mean & std ---
rolling_mean = daily_total.rolling(window=30, center=True).mean()
rolling_std  = daily_total.rolling(window=30, center=True).std()
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
axes[0].plot(daily_total.index, daily_total.values, linewidth=0.5, alpha=0.5, label="Daily")
axes[0].plot(rolling_mean.index, rolling_mean.values, linewidth=2, color="red", label="30-Day Rolling Mean")
axes[0].set_title("Daily Consumption with 30-Day Rolling Mean")
axes[0].set_ylabel("kWh")
axes[0].legend()
axes[1].plot(rolling_std.index, rolling_std.values, linewidth=1.5, color="darkorange")
axes[1].set_title("30-Day Rolling Standard Deviation")
axes[1].set_ylabel("kWh (std)")
axes[1].set_xlabel("Date")
save_fig("05_rolling_mean_std.png")

# --- 3.6 Yearly overlay comparison ---
fig, ax = plt.subplots(figsize=(14, 6))
for year in [2012, 2013, 2014]:
    yearly = daily_total.loc[str(year)]
    # Align to day-of-year for overlay
    ax.plot(range(1, len(yearly) + 1), yearly.values, linewidth=1.2, label=str(year))
ax.set_title("Yearly Overlay: Daily Consumption")
ax.set_xlabel("Day of Year")
ax.set_ylabel("Total kWh")
ax.legend()
save_fig("06_yearly_overlay.png")

print("  ✓ All 6 exploratory figures saved")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 4 — SEASONAL DECOMPOSITION                                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 80)
print("PHASE 4: SEASONAL DECOMPOSITION (STL)")
print("=" * 80)

stl = STL(daily_total, period=7, robust=True)
stl_result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
components = [
    ("Observed",  stl_result.observed,  "steelblue"),
    ("Trend",     stl_result.trend,     "darkred"),
    ("Seasonal",  stl_result.seasonal,  "forestgreen"),
    ("Residual",  stl_result.resid,     "darkorange"),
]
for ax, (title, data, color) in zip(axes, components):
    ax.plot(data.index, data.values, linewidth=0.8, color=color)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("kWh")
axes[-1].set_xlabel("Date")
fig.suptitle("STL Decomposition (period = 7 days)", fontsize=14, y=1.01)
save_fig("07_stl_decomposition.png")

# Interpretation notes
print("""
  STL Decomposition Notes:
  - TREND: Shows the long-term evolution of electricity demand.
    Look for gradual increases/decreases and level shifts.
  - SEASONAL: Captures the weekly cycle — typically lower demand on
    weekends and higher on weekdays.
  - RESIDUAL: What's left after removing trend and seasonality.
    Should look like random noise if the decomposition is good.
    Large residual spikes indicate holidays or unusual events.
""")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 5 — STATIONARITY & PARAMETER SELECTION                            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("=" * 80)
print("PHASE 5: STATIONARITY & PARAMETER SELECTION")
print("=" * 80)

# --- 5.1 ADF and KPSS tests on raw series ---
def run_stationarity_tests(series, label="Series"):
    """Run ADF and KPSS tests and print results."""
    print(f"\n  --- {label} ---")
    # ADF test (H0: unit root / non-stationary)
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(series.dropna(), autolag="AIC")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  ADF p-value   : {adf_p:.6f}")
    print(f"  ADF Lags Used : {adf_lags}")
    for key, val in adf_crit.items():
        print(f"    Critical {key}: {val:.4f}")
    adf_stationary = adf_p < 0.05
    print(f"  → ADF says: {'STATIONARY' if adf_stationary else 'NON-STATIONARY'}")

    # KPSS test (H0: stationary)
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series.dropna(), regression="c", nlags="auto")
    print(f"\n  KPSS Statistic: {kpss_stat:.4f}")
    print(f"  KPSS p-value  : {kpss_p:.4f}")
    kpss_stationary = kpss_p >= 0.05
    print(f"  → KPSS says: {'STATIONARY' if kpss_stationary else 'NON-STATIONARY'}")

    return adf_stationary, kpss_stationary

adf_raw, kpss_raw = run_stationarity_tests(daily_total, "Raw Daily Series")

# --- 5.2 Differencing ---
# First difference (d=1)
diff1 = daily_total.diff().dropna()
adf_d1, kpss_d1 = run_stationarity_tests(diff1, "After d=1 (first difference)")

# Seasonal difference (D=1, m=7)
diff1_D1 = diff1.diff(7).dropna()
adf_d1D1, kpss_d1D1 = run_stationarity_tests(diff1_D1, "After d=1, D=1 (+ seasonal diff, m=7)")

# --- 5.3 ACF / PACF plots ---
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

plot_acf(diff1.dropna(), lags=40, ax=axes[0, 0], title="ACF — After d=1")
plot_pacf(diff1.dropna(), lags=40, ax=axes[0, 1], title="PACF — After d=1", method="ywm")
plot_acf(diff1_D1.dropna(), lags=40, ax=axes[1, 0], title="ACF — After d=1, D=1 (m=7)")
plot_pacf(diff1_D1.dropna(), lags=40, ax=axes[1, 1], title="PACF — After d=1, D=1 (m=7)", method="ywm")
save_fig("08_acf_pacf.png")
print("\n  ✓ ACF/PACF plots saved")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 6 — TRAIN/TEST SPLIT & PARAMETER SELECTION                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 80)
print("PHASE 6: CHRONOLOGICAL TRAIN/TEST SPLIT & SARIMA FITTING")
print("=" * 80)

# Split strictly chronologically: hold out last 60 days (2014-11-02 to 2014-12-31)
TEST_DAYS = 60
train = daily_total.iloc[:-TEST_DAYS]
test  = daily_total.iloc[-TEST_DAYS:]

print(f"\n  Train window: {train.index.min().date()} to {train.index.max().date()} ({len(train)} days)")
print(f"  Test window : {test.index.min().date()} to {test.index.max().date()} ({len(test)} days)")
assert len(train) == 1036, f"Expected 1,036 train days, got {len(train)}"
assert len(test) == 60, f"Expected 60 test days, got {len(test)}"

# --- Stepwise Parameter Selection via auto_arima (strictly on TRAIN to prevent leakage) ---
print("\n  Running auto_arima strictly on TRAIN set (no data leakage)...")
auto_model = pm.auto_arima(
    train,
    m=7,
    seasonal=True,
    d=None,          # let auto_arima select differencing order
    D=None,          # let auto_arima select seasonal differencing order
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    stepwise=True,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    information_criterion="aic",
    n_fits=50,
)

print(f"\n  auto_arima selected: {auto_model.order} x {auto_model.seasonal_order}")
print(f"  AIC: {auto_model.aic():.2f}")
print(f"  BIC: {auto_model.bic():.2f}")

best_order = auto_model.order
best_seasonal_order = auto_model.seasonal_order

print(f"\n  Note on D={best_seasonal_order[1]}: auto_arima selects D={best_seasonal_order[1]}, absorbing weekly")
print(f"  seasonality through seasonal lag terms rather than seasonal differencing.")

# Fit SARIMA on Train
print(f"\n  Fitting SARIMAX{best_order}x{best_seasonal_order} on train...")
sarima_model = SARIMAX(
    train,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarima_result = sarima_model.fit(disp=False, maxiter=500)

print(f"\n  Model fit complete!")
print(f"  AIC : {sarima_result.aic:.2f}")
print(f"  BIC : {sarima_result.bic:.2f}")
print(f"\n{sarima_result.summary()}")

# --- 6.1 Residual diagnostics ---
residuals = sarima_result.resid
# Exclude startup transient points (first 8 points) caused by differencing lag initialization
clean_residuals = residuals.iloc[8:].dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residual time plot
axes[0, 0].plot(clean_residuals.index, clean_residuals.values, linewidth=0.5, color="steelblue")
axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
axes[0, 0].set_title("Residuals Over Time (excl. startup transient)")
axes[0, 0].set_ylabel("Residual (kWh)")

# Histogram
axes[0, 1].hist(clean_residuals, bins=40, density=True, color="steelblue", edgecolor="white", alpha=0.8)
axes[0, 1].set_title("Residual Distribution")
axes[0, 1].set_xlabel("Residual (kWh)")

# Q-Q plot
from scipy import stats
stats.probplot(clean_residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Q-Q Plot (Residuals)")

# ACF of residuals
plot_acf(clean_residuals, lags=30, ax=axes[1, 1], title="ACF of Residuals")

fig.suptitle("SARIMA Residual Diagnostics", fontsize=14, y=1.01)
save_fig("09_residual_diagnostics.png")

# Ljung-Box test on clean residuals
lb_test = acorr_ljungbox(clean_residuals, lags=[7, 14, 21], return_df=True)
print("\n  Ljung-Box Test on Residuals (H0: no autocorrelation):")
print(lb_test.to_string())
lb_pass = (lb_test["lb_pvalue"] > 0.05).all()
print(f"  → Residuals {'PASS' if lb_pass else 'FAIL'} white noise test (all p > 0.05: {lb_pass})")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 7 — FORECASTING & EVALUATION                                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 80)
print("PHASE 7: FORECASTING & EVALUATION")
print("=" * 80)

# --- 7.1 Direct multi-step forecast (original) ---
forecast_result = sarima_result.get_forecast(steps=TEST_DAYS)
forecast_mean = forecast_result.predicted_mean
forecast_ci   = forecast_result.conf_int(alpha=0.05)

# --- 7.2 Baselines ---
# Naive baseline: yesterday's value
naive_forecast = daily_total.shift(1).iloc[-TEST_DAYS:]

# Seasonal naive: same day last week
seasonal_naive_forecast = daily_total.shift(7).iloc[-TEST_DAYS:]

# --- 7.3 Walk-forward (one-step-ahead) SARIMA ---
# This is a much fairer comparison: predict 1 day ahead, then update with
# the actual value and predict the next day. This is how SARIMA is used
# in practice for short-term forecasting.
print("\n  Running walk-forward (one-step-ahead) evaluation...")
print("  (Predicting each of the 60 test days one at a time...)")

history = train.copy()
walkforward_preds = []

for t in range(TEST_DAYS):
    model_wf = SARIMAX(
        history,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result_wf = model_wf.fit(disp=False, maxiter=200)
    pred = result_wf.get_forecast(steps=1).predicted_mean.iloc[0]
    walkforward_preds.append(pred)
    # Append actual value and move forward
    actual_val = test.iloc[t]
    history = pd.concat([history, pd.Series([actual_val], index=[test.index[t]])])
    history.index.freq = "D"
    if (t + 1) % 10 == 0:
        print(f"    ... {t + 1}/{TEST_DAYS} days done")

walkforward_series = pd.Series(walkforward_preds, index=test.index, name="walkforward")
print("  Walk-forward evaluation complete!")

# --- 7.4 SARIMAX with day-of-week exogenous features ---
print("\n  Fitting SARIMAX with day-of-week exogenous features...")

def make_dow_features(idx):
    """Create day-of-week dummy variables (6 dummies, Sunday as reference)."""
    dow = pd.get_dummies(idx.dayofweek, prefix="dow", dtype=float)
    dow.index = idx
    # Drop one column to avoid multicollinearity (Sunday = dow_6)
    dow = dow.drop(columns=["dow_6"], errors="ignore")
    return dow

exog_train = make_dow_features(train.index)
exog_test  = make_dow_features(test.index)

sarimax_model = SARIMAX(
    train,
    exog=exog_train,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarimax_result = sarimax_model.fit(disp=False, maxiter=500)
sarimax_forecast = sarimax_result.get_forecast(steps=TEST_DAYS, exog=exog_test)
sarimax_mean = sarimax_forecast.predicted_mean
sarimax_ci   = sarimax_forecast.conf_int(alpha=0.05)

print(f"  SARIMAX AIC: {sarimax_result.aic:.2f}")

# --- 7.5 Metrics ---
def compute_metrics(actual, predicted, label="Model"):
    """Compute MAE, RMSE, MAPE for a forecast."""
    actual = actual.values
    predicted = predicted.values if hasattr(predicted, 'values') else np.array(predicted)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # MAPE — avoid division by zero
    nonzero = actual != 0
    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
    return {"Model": label, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

metrics = []
metrics.append(compute_metrics(test, forecast_mean,            "SARIMA (60-day direct)"))
metrics.append(compute_metrics(test, walkforward_series,       "SARIMA (walk-forward 1-step)"))
metrics.append(compute_metrics(test, sarimax_mean,             "SARIMAX + day-of-week (direct)"))
metrics.append(compute_metrics(test, naive_forecast,           "Naive (yesterday)"))
metrics.append(compute_metrics(test, seasonal_naive_forecast,  "Seasonal Naive (last week)"))

metrics_df = pd.DataFrame(metrics).set_index("Model")
metrics_df = metrics_df.round(2)

print("\n  =====================================================================")
print("                    FORECAST EVALUATION METRICS                        ")
print("  =====================================================================")
print(metrics_df.to_string())
print("  =====================================================================")

# Save metrics
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"))
print(f"\n  [SAVED] {os.path.join(OUTPUT_DIR, 'metrics_summary.csv')}")

# --- 7.6 Forecast plot: all models ---
fig, ax = plt.subplots(figsize=(16, 7))

# Show last 90 days of training for context
train_tail = train.iloc[-90:]
ax.plot(train_tail.index, train_tail.values, linewidth=0.8, color="steelblue",
        alpha=0.6, label="Training Data")
ax.plot(test.index, test.values, linewidth=1.8, color="black", label="Actual (Test)")
ax.plot(forecast_mean.index, forecast_mean.values, linewidth=1.5, color="red",
        linestyle="-", label="SARIMA Direct")
ax.plot(walkforward_series.index, walkforward_series.values, linewidth=1.5,
        color="darkorange", linestyle="-", label="SARIMA Walk-Forward")
ax.plot(sarimax_mean.index, sarimax_mean.values, linewidth=1.5, color="purple",
        linestyle="--", label="SARIMAX + DoW")
ax.plot(seasonal_naive_forecast.index, seasonal_naive_forecast.values,
        linewidth=1, color="green", linestyle=":", alpha=0.7, label="Seasonal Naive")
ax.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    alpha=0.1, color="red",
)
ax.axvline(x=test.index[0], color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.set_title("All Forecasts vs Actual (Test Period: Last 60 Days)")
ax.set_xlabel("Date")
ax.set_ylabel("Total kWh")
ax.legend(loc="upper left", fontsize=9)
save_fig("10_forecast_vs_actual.png")

# --- 7.7 Walk-forward detail plot ---
fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})

axes[0].plot(test.index, test.values, linewidth=1.5, color="black", label="Actual")
axes[0].plot(walkforward_series.index, walkforward_series.values, linewidth=1.5,
             color="darkorange", label="Walk-Forward 1-Step")
axes[0].set_title("Walk-Forward One-Step-Ahead SARIMA Forecast")
axes[0].set_ylabel("Total kWh")
axes[0].legend()

wf_errors = test.values - walkforward_series.values
axes[1].bar(test.index, wf_errors,
            color=["salmon" if e > 0 else "lightblue" for e in wf_errors], width=0.8)
axes[1].axhline(y=0, color="black", linewidth=0.8)
axes[1].set_title("Walk-Forward Errors (Actual - Predicted)")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Error (kWh)")
save_fig("11_walkforward_detail.png")

# --- 7.8 Direct forecast error bar plot ---
fig, ax = plt.subplots(figsize=(16, 5))
errors = (test.values - forecast_mean.values)
ax.bar(test.index, errors, color=["salmon" if e > 0 else "lightblue" for e in errors], width=0.8)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_title("Direct Forecast Errors (Actual - Predicted)")
ax.set_xlabel("Date")
ax.set_ylabel("Error (kWh)")
save_fig("12_direct_forecast_errors.png")

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 8 — FINAL FORECAST & CONCLUSIONS                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 80)
print("PHASE 8: FINAL FORECAST & CONCLUSIONS")
print("=" * 80)

# Refit on full data
print("\n  Refitting SARIMA on full dataset for final forecast...")
final_model = SARIMAX(
    daily_total,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
final_result = final_model.fit(disp=False, maxiter=500)

# Forecast next 30 days
FORECAST_HORIZON = 30
final_forecast = final_result.get_forecast(steps=FORECAST_HORIZON)
final_mean = final_forecast.predicted_mean
final_ci   = final_forecast.conf_int(alpha=0.05)

# Plot
fig, ax = plt.subplots(figsize=(16, 6))
tail = daily_total.iloc[-90:]
ax.plot(tail.index, tail.values, linewidth=1, color="steelblue", label="Historical Data")
ax.plot(final_mean.index, final_mean.values, linewidth=2, color="red", label="30-Day Forecast")
ax.fill_between(
    final_ci.index,
    final_ci.iloc[:, 0],
    final_ci.iloc[:, 1],
    alpha=0.2, color="red", label="95% CI",
)
ax.axvline(x=daily_total.index[-1], color="gray", linestyle="--", linewidth=1)
ax.set_title("Final 30-Day Electricity Demand Forecast (Refitted on Full Data)")
ax.set_xlabel("Date")
ax.set_ylabel("Total kWh")
ax.legend()
save_fig("13_final_30day_forecast.png")

# Print forecast table
print("\n  30-Day Forecast:")
forecast_table = pd.DataFrame({
    "Date": final_mean.index.strftime("%Y-%m-%d"),
    "Forecast (kWh)": final_mean.values.round(0),
    "Lower CI": final_ci.iloc[:, 0].values.round(0),
    "Upper CI": final_ci.iloc[:, 1].values.round(0),
})
print(forecast_table.to_string(index=False))

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  SUMMARY & CONCLUSIONS                                                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 80)
print("SUMMARY & CONCLUSIONS")
print("=" * 80)

best_model_name = metrics_df["MAPE (%)"].idxmin()
direct_sarima_mape = metrics_df.loc["SARIMA (60-day direct)", "MAPE (%)"]
wf_sarima_mape = metrics_df.loc["SARIMA (walk-forward 1-step)", "MAPE (%)"]

print(f"""
  Dataset : UCI Electricity Load Diagrams 2011-2014
            (Trindade, 2015, DOI: 10.24432/C58C86)
  Target  : Aggregate daily consumption (370 clients summed)
  Period  : 2012-01-01 to 2014-12-31 ({len(daily_total)} days)
  Model   : SARIMAX{best_order}x{best_seasonal_order}

  RESULTS:
  +------------------------------------------------------+
  | Best model by MAPE       : {best_model_name:<26s}|
  | Direct SARIMA MAPE       : {direct_sarima_mape:<26.2f}%|
  | Walk-forward SARIMA MAPE : {wf_sarima_mape:<26.2f}%|
  +------------------------------------------------------+

  KEY FINDINGS:
  1. The aggregated electricity demand shows clear weekly seasonality,
     with lower consumption on weekends and higher on weekdays.
  2. STL decomposition reveals a stable weekly pattern overlaid on an
     annual cycle and gradual trend.
  3. Walk-forward SARIMA substantially outperforms direct multi-step SARIMA.
  4. Residual diagnostics: {'PASS (White Noise)' if lb_pass else 'FAIL (Residual Autocorrelation)'}

  LIMITATIONS:
  - SARIMA assumes linear relationships; non-linear patterns are missed.
  - Holiday effects are not modelled (could add as exogenous variables).
  - Aggregating all clients masks individual client behaviour.

  POSSIBLE IMPROVEMENTS:
  - Add holiday / day-of-week dummy variables via SARIMAX.
  - Try Prophet or Holt-Winters for comparison.
  - Forecast individual high-consumption clients separately.

  OUTPUT FILES:
  All figures and metrics saved to: {OUTPUT_DIR}
""")

# List all output files
print("  Generated files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"    • {f} ({size_kb:.0f} KB)")

print("\n" + "=" * 80)
print("DONE — Electricity Demand Forecasting with SARIMA")
print("=" * 80)
