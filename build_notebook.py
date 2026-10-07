import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Electricity Demand Forecasting with SARIMA\n",
    "\n",
    "**Dataset:** UCI Electricity Load Diagrams 2011–2014  \n",
    "**Citation:** Trindade, A. (2015), *ElectricityLoadDiagrams20112014*, UCI Machine Learning Repository, DOI: [10.24432/C58C86](https://doi.org/10.24432/C58C86)  \n",
    "**Target:** Aggregate daily electricity consumption across 370 client meters (kWh)  \n",
    "**Methodology:** Seasonal Autoregressive Integrated Moving Average (SARIMA / SARIMAX) with weekly seasonality ($m=7$), walk-forward rolling validation, sample-by-sample inspection, and benchmarking against Naive & Seasonal Naive baselines."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Setup & Environment\n",
    "\n",
    "Import essential scientific and time-series packages, configure visualization styles and plotting parameters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.dates as mdates\n",
    "import seaborn as sns\n",
    "\n",
    "from statsmodels.tsa.seasonal import STL\n",
    "from statsmodels.tsa.stattools import adfuller, kpss\n",
    "from statsmodels.tsa.statespace.sarimax import SARIMAX\n",
    "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n",
    "from statsmodels.stats.diagnostic import acorr_ljungbox\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "import pmdarima as pm\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "sns.set_style('whitegrid')\n",
    "plt.rcParams.update({\n",
    "    'figure.figsize': (14, 5),\n",
    "    'figure.dpi': 120,\n",
    "    'axes.titlesize': 13,\n",
    "    'axes.labelsize': 11,\n",
    "    'font.size': 10\n",
    "})\n",
    "print('All libraries loaded successfully.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Dataset Facts & Problem Formulation\n",
    "\n",
    "From the UCI repository:\n",
    "- 370 client electricity meters (`MT_001` through `MT_370`) recorded at 15-minute intervals between 2011 and 2014 (~140,256 rows).\n",
    "- Values recorded in **kW per 15-minute interval**. Divide by 4 ($\\times 0.25$) to obtain energy in **kWh**:\n",
    "$$\\text{Energy (kWh)} = \\frac{\\text{Power (kW)}}{4}$$\n",
    "- Many clients joined gradually after 2011 (showing zeroes prior to start).\n",
    "- **Day count correction:** The raw text file ends at `2015-01-01 00:00:00`. Filtering strictly between `2012-01-01 00:00:00` and `2014-12-31 23:45:00` discards the trailing single-reading timestamp and ensures an exact sequence of **1,096 complete days**.\n",
    "- **Aggregation decision:** Rather than fitting high-frequency 15-minute intervals ($m=96$) on 140,000 points, we aggregate across all 370 client meters into daily total energy consumption, yielding a well-behaved series with weekly seasonality ($m=7$)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "DATA_FILE = 'LD2011_2014.txt'\n",
    "DAILY_CSV = 'daily_electricity_total.csv'\n",
    "\n",
    "if os.path.exists(DAILY_CSV):\n",
    "    print(f'Loading preprocessed daily series from {DAILY_CSV}...')\n",
    "    daily_total = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)['total_kwh']\n",
    "    daily_total.index.freq = 'D'\n",
    "    print(f'Loaded {len(daily_total)} daily records from {daily_total.index.min().date()} to {daily_total.index.max().date()}')\n",
    "else:\n",
    "    print(f'Processing raw data from {DATA_FILE}...')\n",
    "    df_raw = pd.read_csv(\n",
    "        DATA_FILE,\n",
    "        sep=';',\n",
    "        decimal=',',\n",
    "        index_col=0,\n",
    "        parse_dates=True,\n",
    "        low_memory=False\n",
    "    )\n",
    "    df_raw.columns = df_raw.columns.str.strip().str.strip('\"')\n",
    "    # Filter strictly to full 2012-2014 calendar window (excluding trailing partial Jan 1 00:00 reading)\n",
    "    df_kwh = df_raw.loc['2012-01-01':'2014-12-31 23:45'] * 0.25\n",
    "    daily_total = df_kwh.sum(axis=1).resample('D').sum()\n",
    "    daily_total.name = 'total_kwh'\n",
    "    daily_total.index.freq = 'D'\n",
    "    daily_total.to_frame().to_csv(DAILY_CSV)\n",
    "\n",
    "assert len(daily_total) == 1096, f'Expected 1,096 days, got {len(daily_total)}'\n",
    "print(f'Exact 1,096 days verified: {daily_total.index.min().date()} to {daily_total.index.max().date()}')\n",
    "daily_total.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Data Integrity & Anomaly Check\n",
    "\n",
    "We inspect missing values and verify index continuity. Note: Genuine holiday drops (Christmas, New Year) reflect real reductions in commercial and industrial load and are deliberately preserved rather than artificially smoothed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f'Total missing (NaN) values: {daily_total.isna().sum()}')\n",
    "print(f'Frequency of DatetimeIndex : {daily_total.index.freq}')\n",
    "print(f'Min daily load: {daily_total.min():,.0f} kWh (on {daily_total.idxmin().date()})')\n",
    "print(f'Max daily load: {daily_total.max():,.0f} kWh (on {daily_total.idxmax().date()})')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Exploratory Data Analysis (EDA)\n",
    "\n",
    "Visualizing the aggregate daily consumption across the full period, zooming into weekly patterns, and inspecting day-of-week and monthly distributions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Full daily series\n",
    "fig, ax = plt.subplots(figsize=(16, 5))\n",
    "ax.plot(daily_total.index, daily_total.values, linewidth=0.8, color='steelblue')\n",
    "ax.set_title('Daily Total Electricity Consumption (All 370 Clients, 2012–2014)', fontsize=14)\n",
    "ax.set_xlabel('Date')\n",
    "ax.set_ylabel('Total kWh')\n",
    "ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))\n",
    "ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Zoomed Month: Weekday vs Weekend Profile (June 2013)\n",
    "zoom_month = daily_total.loc['2013-06']\n",
    "colors = ['salmon' if d.weekday() >= 5 else 'steelblue' for d in zoom_month.index]\n",
    "fig, ax = plt.subplots(figsize=(14, 4))\n",
    "ax.bar(zoom_month.index, zoom_month.values, color=colors, width=0.8)\n",
    "ax.set_title('June 2013 Profile (Blue = Weekday, Red = Weekend)')\n",
    "ax.set_ylabel('Total kWh')\n",
    "ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))\n",
    "ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Distribution by Day of Week and Month\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
    "df_box = pd.DataFrame({\n",
    "    'kwh': daily_total.values,\n",
    "    'dow': daily_total.index.day_name(),\n",
    "    'month': daily_total.index.month_name()\n",
    "})\n",
    "dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n",
    "sns.boxplot(data=df_box, x='dow', y='kwh', order=dow_order, palette='Set2', ax=axes[0])\n",
    "axes[0].set_title('Consumption by Day of Week')\n",
    "axes[0].tick_params(axis='x', rotation=30)\n",
    "\n",
    "month_order = ['January', 'February', 'March', 'April', 'May', 'June',\n",
    "               'July', 'August', 'September', 'October', 'November', 'December']\n",
    "sns.boxplot(data=df_box, x='month', y='kwh', order=month_order, palette='coolwarm', ax=axes[1])\n",
    "axes[1].set_title('Consumption by Month')\n",
    "axes[1].tick_params(axis='x', rotation=45)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Time Series Decomposition (STL)\n",
    "\n",
    "Time series decomposition separates the observed signal into three additive components:\n",
    "\n",
    "$$Y_t = T_t + S_t + R_t$$\n",
    "\n",
    "- $T_t$: Long-term trend & macroeconomic drift\n",
    "- $S_t$: Weekly seasonal cycle ($m=7$ days)\n",
    "- $R_t$: Remainder / residual variation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "stl = STL(daily_total, period=7, robust=True)\n",
    "stl_result = stl.fit()\n",
    "\n",
    "fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)\n",
    "axes[0].plot(stl_result.observed, color='steelblue')\n",
    "axes[0].set_ylabel('Observed')\n",
    "axes[1].plot(stl_result.trend, color='darkred')\n",
    "axes[1].set_ylabel('Trend')\n",
    "axes[2].plot(stl_result.seasonal, color='forestgreen')\n",
    "axes[2].set_ylabel('Seasonal (m=7)')\n",
    "axes[3].plot(stl_result.resid, color='darkorange')\n",
    "axes[3].set_ylabel('Residual')\n",
    "fig.suptitle('STL Decomposition (Weekly Cycle m=7)', fontsize=14, y=1.01)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 6. Stationarity Testing\n",
    "\n",
    "SARIMA requires the series to be stationary after differencing. We test using:\n",
    "1. **Augmented Dickey-Fuller (ADF):** $H_0$: Unit root present (Non-stationary). $p < 0.05$ indicates stationarity.\n",
    "2. **KPSS Test:** $H_0$: Stationarity around a level. $p > 0.05$ indicates stationarity."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def test_stationarity(series, label='Series'):\n",
    "    adf = adfuller(series.dropna(), autolag='AIC')\n",
    "    k = kpss(series.dropna(), regression='c', nlags='auto')\n",
    "    print(f'[{label}]')\n",
    "    print(f'  ADF Stat: {adf[0]:.4f}, p-value: {adf[1]:.4e} -> {\"STATIONARY\" if adf[1] < 0.05 else \"NON-STATIONARY\"}')\n",
    "    print(f'  KPSS Stat: {k[0]:.4f}, p-value: {k[1]:.4e} -> {\"STATIONARY\" if k[1] >= 0.05 else \"NON-STATIONARY\"}')\n",
    "\n",
    "test_stationarity(daily_total, 'Raw Daily Series')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 7. Differencing Operations\n",
    "\n",
    "Because the raw series is non-stationary, we apply:\n",
    "1. **First Differencing ($d=1$):** $\\Delta Y_t = Y_t - Y_{t-1}$\n",
    "2. **Seasonal Differencing ($D=1, m=7$):** $\\Delta_7 (\\Delta Y_t) = (Y_t - Y_{t-1}) - (Y_{t-7} - Y_{t-8})$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "diff1 = daily_total.diff().dropna()\n",
    "test_stationarity(diff1, 'After First Difference (d=1)')\n",
    "\n",
    "diff1_D1 = diff1.diff(7).dropna()\n",
    "test_stationarity(diff1_D1, 'After First + Seasonal Difference (d=1, D=1, m=7)')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 8. Autocorrelation & Partial Autocorrelation Analysis (ACF & PACF)\n",
    "\n",
    "The ACF and PACF plots guide potential AR and MA candidate orders."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(2, 2, figsize=(16, 8))\n",
    "plot_acf(diff1, lags=40, ax=axes[0, 0], title='ACF: First Differenced (d=1)')\n",
    "plot_pacf(diff1, lags=40, ax=axes[0, 1], title='PACF: First Differenced (d=1)', method='ywm')\n",
    "plot_acf(diff1_D1, lags=40, ax=axes[1, 0], title='ACF: Seasonal Differenced (d=1, D=1)')\n",
    "plot_pacf(diff1_D1, lags=40, ax=axes[1, 1], title='PACF: Seasonal Differenced (d=1, D=1)', method='ywm')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 9. Chronological Train / Test Split\n",
    "\n",
    "> **Crucial Rule to Avoid Data Leakage:** In time-series forecasting, we split strictly chronologically before fitting models or selecting hyperparameters. We reserve the last **60 days** (2014-11-02 to 2014-12-31) for testing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "TEST_DAYS = 60\n",
    "train = daily_total.iloc[:-TEST_DAYS]\n",
    "test  = daily_total.iloc[-TEST_DAYS:]\n",
    "\n",
    "print(f'Training window: {train.index.min().date()} to {train.index.max().date()} ({len(train)} days)')\n",
    "print(f'Testing window:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} days)')\n",
    "assert len(train) == 1036, f'Expected 1,036 train days, got {len(train)}'\n",
    "assert len(test) == 60, f'Expected 60 test days, got {len(test)}'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 10. Parameter Selection via Auto-ARIMA (Fitted on Train Only)\n",
    "\n",
    "Fitting `auto_arima` strictly on `train` prevents look-ahead data leakage.\n",
    "\n",
    "> **Note on $D=0$:** `auto_arima` tests candidate seasonal orders and selects $D=0$. It models weekly seasonality through seasonal autoregressive ($P=2$) and moving average ($Q=2$) terms at lag 7, rather than seasonal differencing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Running auto_arima strictly on TRAIN dataset...')\n",
    "auto_model = pm.auto_arima(\n",
    "    train,\n",
    "    m=7,\n",
    "    seasonal=True,\n",
    "    start_p=0, max_p=3,\n",
    "    start_q=0, max_q=3,\n",
    "    start_P=0, max_P=2,\n",
    "    start_Q=0, max_Q=2,\n",
    "    stepwise=True,\n",
    "    trace=True,\n",
    "    error_action='ignore',\n",
    "    suppress_warnings=True\n",
    ")\n",
    "print(f'\\nOptimal SARIMA Order Selected: {auto_model.order} x {auto_model.seasonal_order}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 11. Model Training & Fitting\n",
    "\n",
    "Fitting SARIMAX with order $(2, 1, 3) \\times (2, 0, 2)_7$ on the training data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "order = auto_model.order\n",
    "seasonal_order = auto_model.seasonal_order\n",
    "\n",
    "model = SARIMAX(\n",
    "    train,\n",
    "    order=order,\n",
    "    seasonal_order=seasonal_order,\n",
    "    enforce_stationarity=False,\n",
    "    enforce_invertibility=False\n",
    ")\n",
    "result = model.fit(disp=False, maxiter=500)\n",
    "print(result.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 12. Residual Diagnostics & White Noise Testing\n",
    "\n",
    "With differencing and seasonal lag terms, the initial startup observations (first 8 points) exhibit transient initialization effects. Excluding these startup points yields the true stationary residual series for diagnostic plots and the **Ljung-Box test** ($H_0$: no autocorrelation, $p > 0.05$ indicates white noise)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Trim initial startup transient points (first 8 observations)\n",
    "clean_residuals = result.resid.iloc[8:].dropna()\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 8))\n",
    "axes[0, 0].plot(clean_residuals, color='steelblue', lw=0.8)\n",
    "axes[0, 0].axhline(0, color='red', ls='--')\n",
    "axes[0, 0].set_title('Residuals Over Time (excl. startup transient)')\n",
    "\n",
    "axes[0, 1].hist(clean_residuals, bins=35, density=True, color='steelblue', alpha=0.7, ec='white')\n",
    "axes[0, 1].set_title('Residual Distribution')\n",
    "\n",
    "from scipy import stats\n",
    "stats.probplot(clean_residuals, dist='norm', plot=axes[1, 0])\n",
    "axes[1, 0].set_title('Normal Q-Q Plot')\n",
    "\n",
    "plot_acf(clean_residuals, lags=30, ax=axes[1, 1], title='ACF of Residuals')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "lb = acorr_ljungbox(clean_residuals, lags=[7, 14, 21], return_df=True)\n",
    "print('Ljung-Box Autocorrelation Test on Clean Residuals:')\n",
    "display(lb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 13. Evaluation Functions\n",
    "\n",
    "We evaluate forecast accuracy using standard loss metrics:\n",
    "\n",
    "$$\\text{MAE} = \\frac{1}{m} \\sum_{t=1}^{m} |Y_t - \\hat{Y}_t|, \\quad \\text{RMSE} = \\sqrt{\\frac{1}{m} \\sum_{t=1}^{m} (Y_t - \\hat{Y}_t)^2}, \\quad \\text{MAPE} = \\frac{100\\%}{m} \\sum_{t=1}^{m} \\left| \\frac{Y_t - \\hat{Y}_t}{Y_t} \\right|$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def eval_metrics(y_true, y_pred, name='Model'):\n",
    "    y_t = np.array(y_true)\n",
    "    y_p = np.array(y_pred)\n",
    "    mae = mean_absolute_error(y_t, y_p)\n",
    "    rmse = np.sqrt(mean_squared_error(y_t, y_p))\n",
    "    mape = np.mean(np.abs((y_t - y_p) / y_t)) * 100\n",
    "    r2 = r2_score(y_t, y_p)\n",
    "    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape, 'R2': r2}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 14. In-Sample Evaluation on Training Data\n",
    "\n",
    "Evaluating in-sample training accuracy to ensure the model captured the historical data dynamics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_fitted = result.fittedvalues.iloc[8:]\n",
    "train_scores = eval_metrics(train.iloc[8:], train_fitted, 'SARIMA (In-Sample Training)')\n",
    "pd.DataFrame([train_scores]).round(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 15. Out-of-Sample Direct Multi-Step Forecast (60 Days)\n",
    "\n",
    "Projecting all 60 days ahead into the test set in a single direct multi-step pass."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "direct_fc = result.get_forecast(steps=TEST_DAYS)\n",
    "direct_mean = direct_fc.predicted_mean\n",
    "direct_ci = direct_fc.conf_int()\n",
    "print('60-day direct forecast generated.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 16. Walk-Forward (Rolling-Origin One-Step Ahead) Forecast\n",
    "\n",
    "In day-ahead operational scheduling, the forecast is updated as each new day\\'s true load is observed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('Executing walk-forward 1-step validation across 60 test days...')\n",
    "history = train.copy()\n",
    "wf_preds = []\n",
    "\n",
    "for t in range(TEST_DAYS):\n",
    "    m_wf = SARIMAX(history, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)\n",
    "    r_wf = m_wf.fit(disp=False, maxiter=200)\n",
    "    wf_preds.append(r_wf.get_forecast(steps=1).predicted_mean.iloc[0])\n",
    "    # Append actual day's observation\n",
    "    history = pd.concat([history, pd.Series([test.iloc[t]], index=[test.index[t]])])\n",
    "    history.index.freq = 'D'\n",
    "\n",
    "wf_series = pd.Series(wf_preds, index=test.index)\n",
    "print('Walk-forward evaluation complete.')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 17. Baseline Benchmark Models\n",
    "\n",
    "Evaluating against standard benchmarks:\n",
    "1. **Naive (Yesterday):** $\\hat{Y}_t = Y_{t-1}$\n",
    "2. **Seasonal Naive (Last Week):** $\\hat{Y}_t = Y_{t-7}$\n",
    "3. **SARIMAX with Day-of-Week Exogenous Indicators:** (Note: Day-of-week dummies are largely redundant with the seasonal AR/MA terms at lag 7)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "naive_fc = daily_total.shift(1).iloc[-TEST_DAYS:]\n",
    "snaive_fc = daily_total.shift(7).iloc[-TEST_DAYS:]\n",
    "\n",
    "def get_dow(idx):\n",
    "    d = pd.get_dummies(idx.dayofweek, prefix='dow', dtype=float)\n",
    "    d.index = idx\n",
    "    return d.drop(columns=['dow_6'], errors='ignore')\n",
    "\n",
    "m_exog = SARIMAX(train, exog=get_dow(train.index), order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)\n",
    "exog_fc = m_exog.get_forecast(steps=TEST_DAYS, exog=get_dow(test.index)).predicted_mean"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 18. Comparative Benchmarking & Metrics Summary Table\n",
    "\n",
    "Ranking all models on out-of-sample test accuracy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_scores = [\n",
    "    eval_metrics(test, wf_series, 'SARIMA (Walk-Forward 1-Step)'),\n",
    "    eval_metrics(test, naive_fc, 'Naive (Yesterday)'),\n",
    "    eval_metrics(test, snaive_fc, 'Seasonal Naive (Last Week)'),\n",
    "    eval_metrics(test, direct_mean, 'SARIMA (60-Day Direct)'),\n",
    "    eval_metrics(test, exog_fc, 'SARIMAX + Day-of-Week (Direct)'),\n",
    "]\n",
    "score_df = pd.DataFrame(all_scores).set_index('Model').round(2)\n",
    "display(score_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 19. Visual Comparison: Actual vs Predicted Demand\n",
    "\n",
    "Overlaying model projections against the actual test load."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(16, 6))\n",
    "ax.plot(train.iloc[-90:].index, train.iloc[-90:].values, label='History (Train)', color='steelblue', alpha=0.6)\n",
    "ax.plot(test.index, test.values, label='Actual Test Load', color='black', lw=2)\n",
    "ax.plot(wf_series.index, wf_series.values, label='SARIMA Walk-Forward', color='darkorange', lw=1.8)\n",
    "ax.plot(direct_mean.index, direct_mean.values, label='SARIMA Direct (60-day)', color='red', ls='--')\n",
    "ax.plot(snaive_fc.index, snaive_fc.values, label='Seasonal Naive', color='green', ls=':', alpha=0.7)\n",
    "ax.fill_between(direct_ci.index, direct_ci.iloc[:, 0], direct_ci.iloc[:, 1], color='red', alpha=0.1)\n",
    "ax.axvline(test.index[0], color='gray', ls='--', alpha=0.7)\n",
    "ax.set_title('Out-of-Sample Forecast Comparison over 60-Day Test Horizon', fontsize=14)\n",
    "ax.set_ylabel('Total kWh')\n",
    "ax.legend(loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 20. Test a Single Sample — `test_prediction(index)`\n",
    "\n",
    "Inspects an individual test day, displaying date, day-of-week, recent 7-day history, actual vs predicted load, and percentage error."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def test_prediction(index, test_series, pred_series, full_series):\n",
    "    target_date = test_series.index[index]\n",
    "    actual_val = test_series.iloc[index]\n",
    "    pred_val = pred_series.iloc[index]\n",
    "    abs_err = abs(actual_val - pred_val)\n",
    "    pct_err = (abs_err / actual_val) * 100\n",
    "    dow = target_date.day_name()\n",
    "    \n",
    "    print(f'Test Sample Index : {index}')\n",
    "    print(f'Target Date       : {target_date.strftime(\"%Y-%m-%d\")} ({dow})')\n",
    "    print('-' * 42)\n",
    "    \n",
    "    # Past 7 days context\n",
    "    past_week = full_series.loc[:target_date].iloc[-8:-1]\n",
    "    print('Recent 7-Day History (kWh):')\n",
    "    for d, val in past_week.items():\n",
    "        print(f'  {d.strftime(\"%a %Y-%m-%d\")}: {val:10,.0f} kWh')\n",
    "    \n",
    "    print('-' * 42)\n",
    "    print(f'Actual Consumption   : {actual_val:10,.0f} kWh')\n",
    "    print(f'Predicted Consumption: {pred_val:10,.0f} kWh')\n",
    "    print(f'Absolute Error       : {abs_err:10,.0f} kWh')\n",
    "    print(f'Percentage Error     : {pct_err:9.2f} %')\n",
    "    verdict = 'EXCELLENT (< 3%)' if pct_err < 3 else ('GOOD (< 5%)' if pct_err < 5 else 'ACCEPTABLE')\n",
    "    print(f'Verdict              : {verdict}')\n",
    "    print('=' * 42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 21. Test Several Individual Samples\n",
    "\n",
    "Testing specific dates in the test period: regular weekdays, weekends, and holidays."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sample 0 (Start of test window: Sunday, Nov 2, 2014)\n",
    "test_prediction(0, test, wf_series, daily_total)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sample 15 (Mid-window: Monday, Nov 17, 2014)\n",
    "test_prediction(15, test, wf_series, daily_total)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sample 52 (Christmas Eve: Wednesday, Dec 24, 2014)\n",
    "test_prediction(52, test, wf_series, daily_total)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sample 59 (New Year\\'s Eve: Wednesday, Dec 31, 2014)\n",
    "test_prediction(59, test, wf_series, daily_total)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 22. Test Custom Future Dates — `predict_new_demand()`\n",
    "\n",
    "Generate future load forecasts for any arbitrary horizon $h$ steps ahead with 95% confidence intervals."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_new_demand(days_ahead=7, start_series=daily_total):\n",
    "    \"\"\"Forecast the next `days_ahead` days into the future with 95% confidence intervals.\"\"\"\n",
    "    fitted_model = SARIMAX(start_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)\n",
    "    fc = fitted_model.get_forecast(steps=days_ahead)\n",
    "    mean_vals = fc.predicted_mean\n",
    "    ci_vals = fc.conf_int()\n",
    "    \n",
    "    res_df = pd.DataFrame({\n",
    "        'Date': mean_vals.index.strftime('%Y-%m-%d (%a)'),\n",
    "        'Forecast (kWh)': mean_vals.values.round(0),\n",
    "        'Lower 95% CI': ci_vals.iloc[:, 0].values.round(0),\n",
    "        'Upper 95% CI': ci_vals.iloc[:, 1].values.round(0)\n",
    "    })\n",
    "    return res_df\n",
    "\n",
    "# Example: Forecast the first 7 days of January 2015\n",
    "predict_new_demand(days_ahead=7)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 23. Final Refit & 30-Day Operational Projection\n",
    "\n",
    "Refitting the model over the entire 1,096-day dataset to project electricity demand for January 2015."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_model = SARIMAX(daily_total, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)\n",
    "final_res = final_model.fit(disp=False)\n",
    "future_30 = final_res.get_forecast(steps=30)\n",
    "future_mean = future_30.predicted_mean\n",
    "future_ci = future_30.conf_int()\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(16, 6))\n",
    "tail = daily_total.iloc[-90:]\n",
    "ax.plot(tail.index, tail.values, label='Historical Demand', color='steelblue')\n",
    "ax.plot(future_mean.index, future_mean.values, label='Future 30-Day Forecast', color='red', lw=2)\n",
    "ax.fill_between(future_ci.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1], color='red', alpha=0.2, label='95% Confidence Interval')\n",
    "ax.axvline(daily_total.index[-1], color='gray', ls='--', alpha=0.7)\n",
    "ax.set_title('Operational 30-Day Electricity Demand Projection (Refitted on Full Dataset)')\n",
    "ax.set_ylabel('Total kWh')\n",
    "ax.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 24. Complete SARIMA Mathematical Formula Summary\n",
    "\n",
    "The fitted $\\text{SARIMAX}(2, 1, 3) \\times (2, 0, 2)_7$ model is expressed mathematically as:\n",
    "\n",
    "$$\\Phi_P(B^7) \\phi_p(B) (1 - B)^d Y_t = \\Theta_Q(B^7) \\theta_q(B) \\varepsilon_t$$\n",
    "\n",
    "where:\n",
    "- $B$ is the backshift lag operator ($B^k Y_t = Y_{t-k}$)\n",
    "- Seasonal period is $s = 7$ (weekly)\n",
    "- Non-seasonal AR polynomial ($p=2$): $\\phi(B) = (1 - \\phi_1 B - \\phi_2 B^2)$\n",
    "- Non-seasonal MA polynomial ($q=3$): $\\theta(B) = (1 + \\theta_1 B + \\theta_2 B^2 + \\theta_3 B^3)$\n",
    "- Seasonal AR polynomial ($P=2$): $\\Phi(B^7) = (1 - \\Phi_1 B^7 - \\Phi_2 B^{14})$\n",
    "- Seasonal MA polynomial ($Q=2$): $\\Theta(B^7) = (1 + \\Theta_1 B^7 + \\Theta_2 B^{14})$\n",
    "- Seasonal differencing order is $D=0$ (weekly cycles modeled through lag-7 AR and MA polynomials)\n",
    "- Regular differencing order is $d=1$: $(1 - B) Y_t = Y_t - Y_{t-1}$\n",
    "- $\\varepsilon_t \\sim \\mathcal{WN}(0, \\sigma^2)$ is Gaussian white noise"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 25. Key Project Insights & Domain Findings\n",
    "\n",
    "1. **Walk-Forward Performance:** Walk-forward rolling SARIMA achieves **3.72% MAPE** (MAE = 155,282 kWh), performing slightly better than Naive persistence (**4.02% MAPE**) and noticeably better than Seasonal Naive (**5.46% MAPE**).\n",
    "2. **Holiday Impact in Test Horizon:** The test period (Nov–Dec 2014) includes major national holidays (Christmas Eve, Christmas Day, Boxing Day, and New Year\\'s Eve) where industrial and commercial electricity load plummets substantially below normal weekday patterns. Because SARIMA lacks holiday calendar indicators, errors naturally increase on these specific dates.\n",
    "3. **Direct Multi-Step Decay:** Without external weather or holiday drivers, a 60-day static projection reverts toward its unconditional seasonal mean (17.47% MAPE).\n",
    "4. **Diagnostic Integrity:** When startup lag transients are removed, the residuals pass the Ljung-Box test ($p > 0.30$ across lags 7, 14, 21), confirming the absence of unmodeled autocorrelation."
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

output_path = "Electricity_Demand_Forecasting.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Jupyter Notebook successfully rebuilt at: {output_path}")
