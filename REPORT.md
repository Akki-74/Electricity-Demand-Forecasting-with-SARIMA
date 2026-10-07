# Electricity Demand Forecasting with SARIMA — Project Report

## Citation
> **Trindade, A. (2015).** *ElectricityLoadDiagrams20112014*. UCI Machine Learning Repository.  
> DOI: [https://doi.org/10.24432/C58C86](https://doi.org/10.24432/C58C86)

---

## 1. Introduction

Accurate electricity demand forecasting is vital for utility providers, power grid operators, and energy traders to maintain grid reliability, schedule generator dispatch, prevent blackouts, and optimize operational expenditure. Because electricity cannot be economically stored at scale across national distribution networks, supply must match instantaneous load continuously.

This project designs and implements an end-to-end time-series forecasting pipeline using **SARIMA (Seasonal Autoregressive Integrated Moving Average)** models. We analyze multi-client smart meter records from the UCI Machine Learning Repository, investigate trend and seasonal behaviors across multiple time horizons, evaluate model stationarity, select optimal hyperparameters, and benchmark SARIMA against standard baseline heuristics.

---

## 2. Dataset

The project utilizes the **UCI Electricity Load Diagrams 2011–2014** dataset (`LD2011_2014.txt`):
* **Resolution & Scope:** 370 client electricity meters recorded at 15-minute intervals between January 1, 2011 and December 31, 2014 (140,256 timestamps).
* **Measurement Units:** Power recorded in **kW** per 15-minute interval. Energy consumed is obtained by dividing by 4 ($\times 0.25$) to obtain **kWh**.
* **Late-Joining Clients:** A significant subset of meters were commissioned after 2011, registering zeroes prior to activation (158 active meters in Jan 2011 increasing to 367 meters by Dec 2014).
* **Clock-Change Artifacts:** Daylight saving transitions in March (missing 1 hour) and October (aggregated 2 hours) introduce isolated localized anomalies.
* **Aggregation Strategy:** SARIMA on raw 15-minute intervals across 140,000 points with daily seasonality ($m=96$) is computationally prohibitive and prone to memory exhaustion. We aggregate across all 370 clients and resample to **daily total energy consumption (kWh)** with weekly seasonality ($m=7$). The active analysis window is selected as **2012–2014** (1,097 days).

---

## 3. Methodology

```mermaid
flowchart LR
    A[Raw 15-min Load Data] --> B[Filter 2012-2014 & kW->kWh]
    B --> C[Daily Total Aggregation]
    C --> D[STL Decomposition m=7]
    D --> E[Stationarity Testing: ADF & KPSS]
    E --> F[ACF/PACF & Auto-ARIMA]
    F --> G[SARIMA Model Fitting]
    G --> H[Residual Diagnostics: Ljung-Box]
    H --> I[Walk-Forward & Multi-Step Evaluation]
    I --> J[Final 30-Day Refit Projection]
```

### Preprocessing & Anomaly Correction
1. Filtered data starting **2012-01-01** to ensure full customer coverage.
2. Summed across all 370 clients and resampled to daily frequency (`freq='D'`).
3. Outliers below the 1st percentile (associated with holiday sensor dropouts and clock shifts) were replaced with time-based linear interpolation followed by boundary fill.

### Exploratory Data Analysis & Decomposition
* Evaluated weekday versus weekend consumption patterns.
* Ran **STL (Seasonal and Trend decomposition using Loess)** with seasonal period $m=7$ to isolate weekly periodicity, underlying macro trends, and residual anomalies.

### Stationarity & Order Selection
* Applied **Augmented Dickey-Fuller (ADF)** and **KPSS** tests.
* Evaluated differencing orders: first difference ($d=1$) and seasonal difference ($D=1, m=7$).
* Inspected Autocorrelation (ACF) and Partial Autocorrelation (PACF) functions.
* Employed `pmdarima.auto_arima` minimizing Akaike Information Criterion (AIC).

### Model Training & Evaluation Setup
* **Chronological Split:** Train set of 1,037 days (2012-01-01 to 2014-11-01); test set of 60 days (2014-11-02 to 2014-12-31).
* **Evaluated Models:**
  1. **SARIMA (Walk-Forward 1-Step):** Operational standard where the model forecasts one day ahead and updates historical observations dynamically.
  2. **SARIMA (60-Day Direct Multi-Step):** Static forecast emitted 60 days into the future.
  3. **SARIMAX with Exogenous Day-of-Week:** Incorporates one-hot encoded calendar indicators.
  4. **Naive Baseline:** $\hat{y}_t = y_{t-1}$ (yesterday's consumption).
  5. **Seasonal Naive Baseline:** $\hat{y}_t = y_{t-7}$ (same day last week).
* **Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

---

## 4. Key Figures & Visual Analysis

### Figure 1: Daily Total Electricity Consumption (2012–2014)
The aggregated daily consumption displays strong annual variations (elevated demand during summer months peaking above 7.5 million kWh) alongside consistent weekly cycles.

![Daily Series](outputs/01_daily_total_series.png)

---

### Figure 2: Day-of-Week Profile (Weekday vs. Weekend)
Consumption exhibits a clear drop on Saturdays and Sundays compared to working weekdays.

![Day of Week Boxplot](outputs/03_boxplot_day_of_week.png)

---

### Figure 3: STL Decomposition ($m=7$)
Decomposing the series reveals:
* **Trend:** Broad annual macroeconomic and weather-driven cycles.
* **Seasonal:** Consistent, high-amplitude 7-day oscillations with weekday peaks and weekend troughs.
* **Residual:** Mean-zero noise with small spikes corresponding to national holidays.

![STL Decomposition](outputs/07_stl_decomposition.png)

---

### Figure 4: Autocorrelation (ACF) and Partial Autocorrelation (PACF)
After first differencing ($d=1$), significant negative lag-1 and seasonal spikes at lag 7 and 14 motivate the selection of low-order autoregressive and moving average terms.

![ACF and PACF](outputs/08_acf_pacf.png)

---

### Figure 5: Residual Diagnostics
Residual diagnostics for the fitted $\text{SARIMAX}(1, 1, 2) \times (0, 0, 2)_7$ model confirm good fit:
* Residuals are centered at zero with stationary variance.
* Residual ACF shows no significant spikes beyond 95% confidence bounds.
* **Ljung-Box Test:** $p = 0.34$ (lag 7), $p = 0.66$ (lag 14), $p = 0.24$ (lag 21) — all well above 0.05, confirming white noise residuals.

![Residual Diagnostics](outputs/09_residual_diagnostics.png)

---

### Figure 6: Model Forecasts vs. Actual Demand
Comparative performance over the 60-day test horizon:

![Forecast vs Actual](outputs/10_forecast_vs_actual.png)

---

### Figure 7: Walk-Forward One-Step-Ahead Detail
The walk-forward SARIMA tracks both weekday peaks and weekend dips with high precision and low error.

![Walk-Forward Detail](outputs/11_walkforward_detail.png)

---

### Figure 8: 30-Day Operational Future Forecast
Model refitted over the entire 3-year record to project January 2015 demand with 95% confidence intervals.

![Final 30-Day Forecast](outputs/13_final_30day_forecast.png)

---

## 5. Results & Comparative Metrics

The out-of-sample evaluation on the 60-day test set (November–December 2014) yields the following performance benchmarks:

| Model | MAE (kWh) | RMSE (kWh) | MAPE (%) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **SARIMA (Walk-Forward 1-Step)** | **94,542.08** | **150,430.70** | **2.02%** | **Best overall model**; outperforms all baselines |
| **Naive (Yesterday)** | 106,806.39 | 163,336.75 | 2.28% | Strong short-term baseline |
| **Seasonal Naive (Last Week)** | 199,864.65 | 285,879.56 | 4.19% | Captures day-of-week periodicity |
| **SARIMA (60-Day Direct Multi-Step)** | 550,745.35 | 578,984.22 | 11.67% | Reverts towards unconditional seasonal mean |
| **SARIMAX + Day-of-Week (Direct)** | 618,639.95 | 642,702.88 | 13.08% | Direct projection affected by seasonal level shift |

### Discussion of Findings
1. **Walk-Forward Superiority:** When deployed in a realistic rolling-origin environment (re-incorporating actual daily observations), SARIMA achieves an outstanding **2.02% MAPE**, reducing MAE by 11.5% compared to Naive and 52.7% compared to Seasonal Naive.
2. **Direct Multi-step Horizon Decay:** In long direct horizons (60 days), unassisted SARIMA converges toward its mean path, failing to anticipate winter downward shifts unless paired with annual seasonality ($m=365$) or exogenous temperature inputs.
3. **Statistical Validity:** The auto-selected $\text{SARIMAX}(1, 1, 2) \times (0, 0, 2)_7$ model passes all residual diagnostic criteria, proving that weekly autoregressive and moving average dynamics adequately capture short-term temporal dependencies.

---

## 6. Conclusion & Future Extensions

### Key Takeaways
* **Data Resolution & Scalability:** Aggregating 15-minute readings into daily series resolved memory constraints while preserving clear weekly consumption patterns.
* **Model Validation:** Benchmarking against both Naive and Seasonal Naive baselines provided critical context that prevented misleading conclusions about model quality.
* **Operational Readiness:** The walk-forward SARIMA framework provides an accurate, automated solution for day-ahead electricity load planning.

### Recommended Extensions
1. **Exogenous Weather Variables:** Incorporating historical ambient temperature and cooling/heating degree days (CDD/HDD) would dramatically improve multi-month horizon accuracy.
2. **Dual-Seasonal Models:** Employing BATS/TBATS or Prophet with dual seasonality ($m_1=7, m_2=365.25$) to jointly model day-of-week and day-of-year dynamics.
3. **Cluster-Based Forecasting:** Segmenting individual clients (e.g., residential vs. industrial) using k-means clustering on load curves before training specialized SARIMA models.
