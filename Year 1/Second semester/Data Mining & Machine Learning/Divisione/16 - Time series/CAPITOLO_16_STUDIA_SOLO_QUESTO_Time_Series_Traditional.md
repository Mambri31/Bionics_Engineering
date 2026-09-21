# Chapter 16 — Time Series: Traditional (Statistical) Approaches

> **Course:** Data Mining and Machine Learning / Bioinspired Computational Methods — Biological Data Mining
> **Instructor:** Prof. Francesco Marcelloni (University of Pisa)
> **Sources merged into this file:** professor's slides (`16-TimeSeries_TraditionalApproaches.pdf`, 33 pages) + lecture notes (`DM_11-17 (1)-115-147.pdf`, 33 pages — lesson **L17, 05/05/2026**)
>
> **⚠️ Note on the material**: the slides are **extracted from the book Marco Peixeiro, *Time Series Forecasting in Python*, Manning Publications Co., 2022** — not from Han–Kamber–Pei.
>
> **⚠️ Note on the notes**: the lecture-notes PDF contains **TWO passes over the same material** (a first set covering pages 1–9 and a second, more discursive set from page 10 on). Both have been merged here. Its **final page opens the Machine Learning chapter**, which is **Chapter 17** — see that file.
>
> **Instructions for Claude:** this is the single reference file for Chapter 16. When the user asks study questions about this chapter, use **only** this file as context (no need to re-read the original PDFs unless a detail is missing). The file merges the official slide content with the lecture's discursive explanations and worked examples, keeping both levels of detail.

---

## Chapter outline
1. Time series, decomposition, and forecasting
2. Baseline models and MAPE
3. The random walk process
4. Stationarity, differencing, and the augmented Dickey-Fuller test
5. Autocorrelation (ACF) and the diagnostic procedure
6. Forecasting a random walk
7. The Moving Average model MA(q) and rolling forecasts
8. The Autoregressive process AR(p) and the PACF
9. ARMA(p,q), AIC, and residual analysis
10. ARIMA(p,d,q) for non-stationary series
11. SARIMA for seasonality
12. SARIMAX with exogenous variables

---

## 1. Time series, decomposition, forecasting

> **A TIME SERIES is a set of data points ORDERED IN TIME.** **The data is EQUALLY SPACED in time** — for instance recorded every hour, minute, month or quarter.

> **The lecture's framing**: in this type of data **the TEMPORAL ORDER IS ESSENTIAL**, because the aim is to study **how a phenomenon changes and evolves over time.** Even when the original phenomenon is **continuous**, it can be **represented through DISCRETE measurements taken at fixed time steps.**
>
> **⚠️ THE FUNDAMENTAL POINT: the ORDER of the data MATTERS.** *"We cannot treat the samples as independent observations that can be randomly mixed, because the value observed today may depend on the value observed yesterday."*
> **⇒ CONSEQUENCE: we CANNOT split the training and test set RANDOMLY, as in classical machine learning. We must respect the temporal order: past data to TRAIN, future data to TEST.**

**The running example**: **quarterly earnings per share (EPS) of Johnson & Johnson in USD, 1960–1980**, which shows **a positive trend and a cyclical behaviour.**

### 1.1 Decomposition

> **DECOMPOSITION: the statistical task that separates a time series into its different components — TREND, SEASONALITY and RESIDUALS.**
> - **TREND**: **slow-moving changes** in a time series — the long-term tendency (increase, decrease, or a generally non-constant direction).
> - **SEASONALITY**: the **seasonal/periodic pattern** — behaviours that **repeat at regular intervals** (daily, weekly, monthly, quarterly, yearly).
> - **RESIDUALS**: **behaviour that CANNOT be explained by the trend and seasonality** — the random, irregular part.

> **⚠️ NB — the difference between SEASONALITY and NOISE** (lecture): if a monthly sales series shows that **every summer sales increase**, that is a **SEASONAL component**. If the series **slowly increases year after year**, that is a **TREND**. If there are **small random fluctuations from one month to another**, that is the **NOISE** component.

### 1.2 Forecasting

> **FORECASTING is predicting the future using HISTORICAL DATA and KNOWLEDGE OF FUTURE EVENTS that might affect our forecasts.**

**⚠️ Why is it different from other regression tasks? TWO reasons:**
> 1. **TIME SERIES HAVE AN ORDER.**
> 2. **IT IS POSSIBLE TO FORECAST A TIME SERIES WITHOUT THE USE OF FEATURES.**

> **The contrast spelled out** (lecture): in a normal regression problem the **order often does not matter** — predicting house prices, each house is an **independent sample** with features (size, location, rooms). In a time series **the order matters because each value comes after another in time**, so **the PAST CONTAINS INFORMATION ABOUT THE FUTURE**, and **we can predict future values using the previous values of the same series, even without external features.**

**Application domains** mentioned in class: **marketing** (past sales → future demand), **finance** (past stock fluctuations → possible future evolution, for investment decisions), and **server workload prediction** — *if the future workload is expected to be low in some interval, part of the computation can be moved to another server, letting the less active one be switched off* ⇒ **forecasting supports ENERGY SAVING.**

**The forecasting life cycle** (lecture): **define the GOAL** (what has to be predicted) → **establish the forecasting HORIZON** (one minute, hour, day, several days…) → **collect the relevant data** → **develop the model** → **deploy to production** → **MONITOR its behaviour over time**, since **new data are progressively collected and the model may need to be corrected or updated if its performance changes.**

> **And a forward pointer**: in many real cases **the time series alone may not be sufficient** — especially in financial or economic contexts, **EXOGENOUS VARIABLES** (external variables not belonging to the series but influencing it) matter. That is §12.

## 2. Baseline models and MAPE

> **A BASELINE MODEL is a TRIVIAL solution to the forecasting problem under consideration.**

> **Its purpose** (lecture): **not to provide the best prediction, but to establish a MINIMUM LEVEL OF PERFORMANCE. ⚠️ If a more complex model performs WORSE than the baseline, it cannot be considered an effective solution.**

**The metric — Mean Absolute Percentage Error (MAPE):**
$$\text{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{A_i - F_i}{A_i}\right|$$
where **$A_i$ is the actual value at point $i$ in time, $F_i$ the forecast value at point $i$, and $n$ the number of forecasts.**

> **⚠️ Why MAPE and not a raw error** (lecture): **it NORMALIZES the error with respect to the MAGNITUDE of the actual values.** *"The same absolute error can have a very different meaning depending on the scale of the data: an error of 0.01 may be relevant if the values range between 0 and 1, but negligible if they range between 0 and 1000."*

### The three baselines on the Johnson & Johnson EPS

| Baseline | Method | **MAPE** |
|---|---|---|
| **Historical mean** | **compute the mean of the values over a certain period and assume future values equal that mean** | **70%** |
| **Last known value** (naïve forecast) | **predict that all future values equal the most recent observed value** | **30.45%** |
| **Naïve seasonal forecast** | **take the LAST OBSERVED CYCLE and REPEAT it into the future** | **11.56%** |

> **Reading the results:**
> - The **mean baseline** *"makes sense only if the series is relatively stable around a constant average. If the series has trend or seasonality, the mean baseline is too naive."*
> - The **last-value baseline** at least **follows the increasing trend** — *"if the series is growing, the last value is usually closer to the next value than the global mean"* — it **captures LOCAL CONTINUITY rather than seasonality**, treating the series as **locally a random walk**. **⚠️ But it does NOT take the seasonality into account**, and EPS is **high during the first three quarters and then falls in the last quarter.**
> - The **naïve seasonal forecast** predicts each quarter using **the same quarter of the previous cycle**. **⇒ SEASONALITY HAS A SIGNIFICANT IMPACT: repeating the last season into the future yields fairly accurate forecasts.**

## 3. The random walk process

> **A RANDOM WALK is a process in which there is an EQUAL CHANCE of going UP or DOWN by a RANDOM NUMBER.**
> $$y_t = C + \alpha_1 y_{t-1} + \varepsilon_t$$
> **the current value $y_t$ is a function of the value at the previous timestep $y_{t-1}$, a constant $C$, and a random number $\varepsilon_t$, also termed WHITE NOISE.**
>
> **FORMALLY: a random walk is a series whose FIRST DIFFERENCE IS STATIONARY AND UNCORRELATED — the process moves COMPLETELY AT RANDOM.**

**In the pure random walk case $\alpha_1 = 1$:** $\;y_t = C + y_{t-1} + \varepsilon_t$, and with $C = 0$: $\;y_t = y_{t-1} + \varepsilon_t$, hence
$$y_t - y_{t-1} = \varepsilon_t \qquad \text{— the first difference IS white noise.}$$

> **Why a random walk drifts** (lecture): assuming $y_0 = 0$,
> $$y_1 = y_0 + \varepsilon_1 = \varepsilon_1, \qquad y_2 = y_1 + \varepsilon_2 = \varepsilon_1 + \varepsilon_2$$
> **each value ACCUMULATES the random shocks that occurred before.** **This explains why a random walk may appear to move upward or downward over time even though the individual changes are random** — and why it is **non-stationary**.

**The motivating example**: buying shares of a company. **The daily closing price shows a long-term increasing trend but abrupt changes**, and **can be modelled using the random walk model.**

## 4. Stationarity, differencing, and the ADF test

> **A STATIONARY time series is one whose STATISTICAL PROPERTIES DO NOT CHANGE OVER TIME.** It has a **CONSTANT MEAN, VARIANCE, and AUTOCORRELATION**, and **these properties are INDEPENDENT OF TIME.**
>
> **⚠️ MANY FORECASTING MODELS ASSUME STATIONARITY** — so we need ways to **transform the series to make it stationary.**

> **The intuitive reading** (lecture): a stationary series **"oscillates" around a stable average level and does not show a persistent increasing or decreasing trend.** This does **not** mean all values are equal — it means **the statistical behaviour does not change drastically over time.**
> **⚠️ WHY it matters: "if the series continuously changes its structure, THE PAST BECOMES LESS INFORMATIVE FOR PREDICTING THE FUTURE."**

### 4.1 The transformations

> - **The simplest transformation is DIFFERENCING**, which calculates the change from one timestep to another, **thus STABILIZING THE MEAN**:
> $$y'_t = y_t - y_{t-1}$$
> **It is possible to difference a time series MANY TIMES to obtain a stationary series.**
> - **To obtain a CONSTANT VARIANCE we can use LOGARITHMS**, which help stabilize the variance.

> **⚠️ THE UNTRANSFORM WARNING**: *"when we model a time series which has been transformed, we have to UNTRANSFORM IT to return the results of the model to the original units of measurement."* **If differencing was applied, the forecasted differences must be ADDED BACK to the previous values**; if a **log transformation** was applied, **raise the forecast values to the power of 10** to bring them back to their original magnitude.

### 4.2 The stationarity condition

Recall the series $y_t = C + \alpha_1 y_{t-1} + \varepsilon_t$ — *the simplest autoregressive model, AR(1).*

> **$\alpha_1$ is the ROOT of the time series. The series is stationary ONLY IF THE ROOT LIES WITHIN THE UNIT CIRCLE — its value must be BETWEEN −1 AND 1 (extremes excluded). Otherwise the series is NON-STATIONARY.**

| Process | Verdict |
|---|---|
| $y_t = 0.5\,y_{t-1} + \varepsilon_t$ | **STATIONARY** (coefficient inside $(-1,1)$) |
| $y_t = y_{t-1} + \varepsilon_t$ | **NON-STATIONARY** (coefficient equal to 1 ⇒ a random walk) |

> **⚠️ The mean and variance of the STATIONARY process become CONSTANT after the first few timesteps.** In the non-stationary case they **continue to change over time.**
>
> **If $\alpha_1$ is very close to 1, the series has a STRONG MEMORY of the past**; at $\alpha_1 = 1$ **shocks ACCUMULATE**, so the series **can move further and further from its initial value without necessarily returning to a stable mean.**
> **⚠️ NB: WHITE NOISE CAN BE STATIONARY, WHILE A RANDOM WALK IS NOT.**

### 4.3 The augmented Dickey-Fuller (ADF) test

> **The ADF test helps determine if a time series is stationary by TESTING FOR THE PRESENCE OF A UNIT ROOT. If a unit root is present, the series is NOT stationary.**
>
> - **$H_0$ (NULL HYPOTHESIS): a UNIT ROOT IS PRESENT, meaning the time series is NOT STATIONARY.**
> - **$H_1$: no unit root ⇒ the series IS stationary.**
>
> **If the test returns a p-value LESS than a significance level (typically 0.05 or 0.01), we REJECT the null hypothesis ⇒ there are no unit roots ⇒ THE SERIES IS STATIONARY.**
> **If the p-value is LARGE, we CANNOT reject $H_0$ ⇒ the series is likely non-stationary.**

> **⚠️ Note the logic of the test** (lecture): *"the test STARTS BY ASSUMING that there is a unit root. Only if the data provide enough evidence against this hypothesis can we reject it."* — a large p-value means **we do not have enough evidence to say the series is stationary**, not that it is proven non-stationary.

> **Once we have a stationary series, we must determine whether there is AUTOCORRELATION or not. Remember that a RANDOM WALK is a series whose first difference is stationary AND UNCORRELATED.**

## 5. Autocorrelation (ACF)

> **The AUTOCORRELATION FUNCTION (ACF) measures the LINEAR RELATIONSHIP BETWEEN LAGGED VALUES of a time series** — **the LAG is the number of timesteps separating two values.** In other words **it measures the correlation of the time series WITH ITSELF.**
>
> **The coefficients $r_1, r_2, r_3, \dots$ are computed between, respectively, $y_t$ and $y_{t-1}$, $y_t$ and $y_{t-2}$, $y_t$ and $y_{t-3}$, …**

> **⚠️ The two diagnostic signatures:**
> - **In the presence of a TREND, the ACF coefficients are HIGH FOR SHORT LAGS and DECREASE LINEARLY as the lag increases.**
> - **If the data is SEASONAL, the ACF plot will also display CYCLICAL PATTERNS.**

> **What the ACF really tells you** (lecture): **how much TEMPORAL MEMORY is present in the series.** *"If autocorrelation is high for small lags, values close in time are connected. If autocorrelation quickly goes to zero, after a few time steps the series no longer has significant memory of the past."*

**The confidence band:**
> **The SHADED AREA represents the 95% CONFIDENCE INTERVAL: if a point is WITHIN the shaded area, it is NOT significantly different from 0; otherwise the coefficient is SIGNIFICANT.** **Generally the evaluation of the shaded area is based on BARTLETT'S STANDARD ERRORS.**

### 5.1 The worked random-walk diagnosis

Generate a series from $y_t = y_{t-1} + \varepsilon_t$ and apply the tests:

| Step | Result | Conclusion |
|---|---|---|
| **ADF on the raw series** | **statistic $=-0.97$, p-value $= 0.77$** | **p $\gg 0.05$ ⇒ we CANNOT reject $H_0$ ⇒ NON-STATIONARY** |
| **ACF on the raw series** | **coefficients decrease SLOWLY as the lag increases** | **a clear indicator that the random walk is NOT stationary** |
| **Apply FIRST-ORDER DIFFERENCING**, then re-test | **ADF $=-31.79$, p-value $= 0$** | **$H_0$ REJECTED ⇒ STATIONARY** |
| **ACF on the differenced series** | **apart from lag 0 (always 1), all values close to zero** | **STATIONARY PROCESS, COMPLETELY RANDOM ⇒ WHITE NOISE** |

> **⚠️ WHY slow decay signals non-stationarity** (lecture): *"in a STATIONARY process, when the actual value goes far from the mean, THE SYSTEM FORCES THE VALUE TO COME BACK to the mean. As a consequence, the value at time $t$ isn't so useful to predict the value at time $t+100$."* A non-stationary series has no such restoring force, so dependence persists across long lags.

## 6. Forecasting a random walk

> **⚠️ In case of a random walk, since the values change randomly, NO STATISTICAL LEARNING MODEL CAN BE APPLIED. We can ONLY use naïve forecasting methods or baselines: the HISTORICAL MEAN, the LAST KNOWN VALUE, and the DRIFT METHOD.**

**The DRIFT METHOD**: **calculate the slope between the FIRST and LAST value of the training set and simply EXTRAPOLATE this straight line into the future:**
$$\text{forecast} = \frac{y_f - y_i}{\#\text{timesteps} - 1}\cdot \text{timestep}$$
where **$y_f$ and $y_i$ are the LAST and FIRST values in the training set.**

> **⚠️ THE VERDICT: FORECASTING A RANDOM WALK ON A LONG HORIZON DOES NOT MAKE SENSE.** *(Lecture: the drift method assumes the average trend observed in training will continue, but "there is no guarantee that this slope will persist in the test set — the process can change direction unpredictably.")*

**Forecasting the NEXT STEP of a random walk**: **predict the last known value** — $\hat{y}_t = y_{t-1}$.
> **This is the NATURAL forecast for a random walk**: since $y_t = y_{t-1} + \varepsilon_t$ and **the future shock is random with mean zero, the best point forecast is the last value.**
> **⚠️ But it is a VERY LIMITED approach — it only works for the ONE-STEP-AHEAD forecast; predicting many steps ahead makes uncertainty increase significantly.**

### 6.1 ⚠️ The diagnostic procedure so far (memorise this flow)

> 1. **Check whether the series is STATIONARY** (plot + ADF test).
> 2. **If NOT stationary, TRANSFORM it** (differencing) and test again.
> 3. **Once stationary, analyse the ACF.**
> 4. **If there is NO significant autocorrelation** ⇒ the differenced series is **WHITE NOISE** ⇒ the original process is a **RANDOM WALK** ⇒ **advanced forecasting models are USELESS; rely on baselines, specifically the last known value for one-step-ahead forecasts.**
> 5. **If autocorrelation IS present**, the series contains **predictable structure** ⇒ proceed to the models below.

## 7. The Moving Average model — MA(q)

> **MOVING AVERAGE MODEL: the current value depends LINEARLY on the MEAN OF THE SERIES, the CURRENT ERROR TERM, and PAST ERROR TERMS.** **MA(q), where $q$ is the ORDER (the number of past error terms).**

$$y_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1} + \theta_2\varepsilon_{t-2} + \dots + \theta_q\varepsilon_{t-q}$$

where **$\mu$ is the mean of the series**, **$\varepsilon_t$ the current error term** — *normally distributed white noise with mean 0 and constant variance* — the **$\varepsilon_{t-i}$ are the past error terms**, and the **$\theta_i$ are the coefficients which determine the impact.**

> **⚠️ THE KEY IDEA: each observation is influenced NOT ONLY by the current random noise, but also by SHOCKS THAT OCCURRED IN PREVIOUS TIME STEPS. THE SYSTEM MEMORY IS CONTAINED IN THE ERRORS.**

> **What a "shock" is** (lecture): *"we do not necessarily mean huge events. A shock is simply the UNPREDICTABLE PART of the series at a certain time instant."* For daily product demand, the model has an average prediction, but **every day there may be a random deviation — that deviation is the shock.**
> **⚠️ And a shock does NOT have an infinite effect: after $q$ steps its effect DISAPPEARS.**

**Example, MA(2):** $\;X_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1} + \theta_2\varepsilon_{t-2}$

### 7.1 How to choose $q$

> **The order $q$ determines the number of past error terms affecting the present value — the larger $q$, the more past error terms.**
> **⚠️ OBSERVE THE AUTOCORRELATION PLOT: the plot presents SIGNIFICANT correlation coefficients UP UNTIL LAG $q$, AFTER WHICH ALL COEFFICIENTS WILL BE NON-SIGNIFICANT. If that is the case, we have a moving average process of order $q$.**

> **The reasoning** (lecture): *"if after a certain lag there is no longer significant autocorrelation, then it does not make sense to include shocks that are further in the past."* Choosing $q$ larger than needed **would mainly contribute NOISE rather than meaningful predictive structure.**

### 7.2 The worked example

- Start from the original series, **apply differencing** to obtain stationarity.
- Split **90% training / 10% test** ⇒ **we must forecast 50 timesteps into the future.**
- The **ACF coefficients become non-significant after lag 2** ⇒ **MA(2).**

### 7.3 ⚠️ Rolling forecasts — and why they are necessary

> **When using an MA(q) model, FORECASTING BEYOND $q$ STEPS INTO THE FUTURE WILL SIMPLY RETURN THE MEAN, because there are no error terms to estimate beyond $q$ steps.**
> **⇒ We use ROLLING FORECASTS to predict UP TO $q$ STEPS AT A TIME, in order to avoid predicting only the mean of the series.**

**The procedure, with MA(2):**
> **We train on the first 449 timesteps and predict timesteps 450 and 451. Then, on the second pass, we train on the first 451 timesteps and predict timesteps 452 and 453…** and so on until all predictions in the test set are made.

**Step by step** (lecture):
> 1. Train the model on the training set.
> 2. Predict the first value(s) of the test set.
> 3. **UPDATE the observation window by including the REAL value just observed.**
> 4. Use this new information to predict the next point.
> 5. Repeat until the end of the test set.

> **Two practical notes:**
> - **You do NOT need to forecast exactly $q$ steps ahead — you can forecast $q-1$, $q-2$, etc.** **The SHORTER the forecast horizon, the MORE RELIABLE the prediction.**
> - **Implementation: the `SARIMAX` function from the `statsmodels` library.**
> - **⚠️ If we estimated the first-order difference, we must ADD THE INITIAL VALUE $y_0$ of the test set to the first differenced value to determine the first predicted value** (undoing the transformation).

**Result**: the MA(2) model **follows the general behaviour of the test series better than the simple baselines** — the **mean baseline gives a very high MSE, the last-value method performs better, and the MA model further reduces the error.**

## 8. The Autoregressive process — AR(p)

> **An AUTOREGRESSIVE PROCESS is a REGRESSION OF A VARIABLE AGAINST ITSELF.** In a time series, **the present value is LINEARLY DEPENDENT ON ITS PAST VALUES.** Denoted **AR(p)**, where $p$ is the order:
> $$y_t = C + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \varepsilon_t$$
>
> **⚠️ THE RANDOM WALK IS A SPECIAL CASE OF AN AUTOREGRESSIVE PROCESS ($p = 1$).**

> **⚠️ THE AR/MA DISTINCTION — stated twice in the lecture as "very important":**
> **In the AR model the present depends on PAST VALUES OF THE SERIES. In the MA model the present depends on PAST SHOCKS (errors).** In **ARMA/ARIMA** both ideas are combined.

**The example**: **forecast the AVERAGE WEEKLY FOOT TRAFFIC in a retail store**, so the manager can better manage the staff's schedule.

| Step | Result |
|---|---|
| **ADF on the raw series** | **$-1.18$, p-value $= 0.68$ ⇒ $H_0$ cannot be rejected ⇒ non-stationary** |
| **ADF on the differenced series** | **$-5.27$, p-value $= 6.36\times10^{-6}$ ⇒ $H_0$ rejected ⇒ STATIONARY** |
| **ACF** | **there is NO lag at which the coefficients ABRUPTLY become non-significant** — they decrease slowly |

> **⇒ CONCLUSION: we do NOT have a moving average process, and we are LIKELY studying an AUTOREGRESSIVE process.**

### 8.1 The Partial Autocorrelation Function (PACF) — how to find $p$

> **The PACF measures the correlation between LAGGED VALUES in a time series WHEN WE REMOVE THE INFLUENCE OF CORRELATED LAGGED VALUES IN BETWEEN.**
>
> **⚠️ WHY the plain ACF is not enough**: *"we wish to measure how $y_t$ relates to $y_{t-2}$. When we measure the AUTOcorrelation between $y_t$ and $y_{t-2}$ we are NOT taking into account that $y_{t-1}$ has an influence on BOTH $y_t$ and $y_{t-2}$. This means we are NOT measuring the REAL impact of $y_{t-2}$ on $y_t$."*
>
> **⇒ Plot the PACF to determine the order of a stationary AR(p) process: THE COEFFICIENTS WILL BE NON-SIGNIFICANT AFTER LAG $p$.**

**The worked result**: **947 weeks in the training set, 52 weeks in the test set.** Computing the PACF on the differenced series, **there are no significant coefficients after lag 3** ⇒ **the differenced average weekly foot traffic is an AR(3) process.** Forecasting then uses a **rolling forecast with window 1.**

> **⚠️ THE TWO CUT-OFF RULES — the single most examinable pair in this chapter:**
> | Model | Diagnostic | Signature |
> |---|---|---|
> | **MA(q)** | **ACF** | **cuts off after lag $q$** |
> | **AR(p)** | **PACF** | **cuts off after lag $p$** |

## 9. ARMA(p,q)

**The example**: **predict BANDWIDTH USAGE for a large data center** — *bandwidth is the maximum rate of data that can be transferred, base unit bits per second (bps).* Predicting it helps **manage network resources, avoid congestion, and plan capacity.**

**The four diagnostic steps:**
> 1. **Check stationarity — it is NOT stationary.**
> 2. **Transform using differences — the differenced series IS stationary.**
> 3. **Plot the ACF: there are significant coefficients after lag 0 and these coefficients SLOWLY DECAY.**
> 4. **Plot the PACF: it has a SINUSOIDAL pattern.**
>
> **⇒ Thus it is NOT a purely moving average process AND NOT a purely autoregressive process.**

> **AUTOREGRESSIVE MOVING AVERAGE: a combination of the autoregressive process and the moving average process.** Denoted **ARMA(p,q)**:
> $$y_t = C + \phi_1 y_{t-1} + \dots + \phi_p y_{t-p} + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q} + \varepsilon_t$$
>
> - **An ARMA(0,q) process is equivalent to an MA(q) process**, since $p=0$ cancels the AR portion.
> - **An ARMA(p,0) process is equivalent to an AR(p) process**, since $q=0$ cancels the MA portion.

### 9.1 ⚠️ Determining $p$ and $q$ — a NEW procedure is needed

> **For MA we used the ACF cut-off; for AR the PACF cut-off. But for ARMA THERE IS NO SIMPLE CUT-OFF RULE.**
>
> **⇒ With a list of possible values for $p$ and $q$, FIT EVERY UNIQUE COMBINATION of ARMA(p,q) and evaluate it using the AKAIKE INFORMATION CRITERION (AIC). THE MODEL WITH THE LOWEST AIC IS SELECTED.**
> **Then ANALYSE THE MODEL'S RESIDUALS. If the residuals look like WHITE NOISE, they are uncorrelated and independently distributed and the model is ready for forecasting; OTHERWISE a different set of values for $p$ and $q$ has to be evaluated.**

### 9.2 The Akaike Information Criterion

> **The AIC is a function of the NUMBER OF PARAMETERS $k$ in a model and the MAXIMUM VALUE OF THE LIKELIHOOD FUNCTION $\hat{L}$:**
> $$\text{AIC} = 2k - 2\ln(\hat{L})$$
> **THE LOWER THE AIC, THE BETTER THE MODEL.**
> **⚠️ AIC allows us to keep a BALANCE BETWEEN THE COMPLEXITY of a model and its GOODNESS OF FIT to the data.** **In our case $k = p + q$.**
>
> **The LIKELIHOOD FUNCTION measures the goodness of fit** — it answers: *"How likely is it that my observed data is coming from an ARMA(1,1) model?"*

> **Why complexity must be penalised** (lecture): *"a model with many parameters can fit the training data very well, but it may OVERFIT."*

**The example**: **$p$ and $q$ vary from 0 to 3 ⇒ 16 possible combinations.** We iterate over each $(p,q)$, compute the ARMA model, evaluate the AIC, and select the lowest.

### 9.3 Residual analysis

> **⚠️ THE PRINCIPLE: A GOOD MODEL MUST ACCOUNT FOR THE ENTIRE PREDICTABLE STRUCTURE. ONLY RANDOM NOISE SHOULD REMAIN IN THE RESIDUALS.**

**(a) QUALITATIVE analysis — the Q-Q plot**
> **Constructed by plotting the QUANTILES OF THE RESIDUALS on the y-axis against the quantiles of a THEORETICAL (normal) DISTRIBUTION on the x-axis.**
> **If the model is a good fit, the residuals are similar to white noise** — they **lie approximately on the straight diagonal line.** **If they DEVIATE from the line, there may be remaining structure in the data and the model is not adequate.**

**(b) QUANTITATIVE analysis — the Ljung-Box test**
> **A statistical test that determines whether the AUTOCORRELATION of a group of data is SIGNIFICANTLY DIFFERENT FROM 0.**
> **NULL HYPOTHESIS: the data is INDEPENDENTLY DISTRIBUTED, meaning there is NO autocorrelation.**
> - **p-value LARGER than 0.05 ⇒ we CANNOT reject $H_0$ ⇒ residuals are independently distributed ⇒ NO autocorrelation ⇒ residuals are similar to white noise ⇒ THE MODEL CAN BE USED FOR FORECASTING.**
> - **p-value LESS than 0.05 ⇒ we REJECT $H_0$ ⇒ residuals are correlated ⇒ THE MODEL CANNOT BE USED FOR FORECASTING.**

> **⚠️ Note the direction of the test — it is the OPPOSITE of the ADF convention.** For **ADF** you WANT a **small** p-value (to reject non-stationarity); for **Ljung-Box** you WANT a **large** p-value (to keep the "no autocorrelation" hypothesis). Confusing the two is the classic error.

**The four residual diagnostics plotted together**: **time series of residuals** (check for obvious patterns) · **histogram + estimated density** (should be roughly normal) · **Q-Q plot** (normality) · **ACF of the residuals** (should be near zero at all lags).

## 10. ARIMA(p,d,q) — non-stationary series

> **We observed that the ARMA model is suitable ONLY for STATIONARY time series. For NON-STATIONARY series we need to add a component: the INTEGRATION ORDER.**
>
> **AUTOREGRESSIVE INTEGRATED MOVING AVERAGE (ARIMA)**:
> - **the combination of the AR(p) and MA(q) processes, BUT IN TERMS OF THE DIFFERENCED SERIES;**
> - **denoted ARIMA(p,d,q)**, where **$p$ is the order of the AR process, $d$ the ORDER OF INTEGRATION, $q$ the order of the MA process;**
> - **⚠️ INTEGRATION IS THE REVERSE OF DIFFERENCING, and the order of integration $d$ EQUALS THE NUMBER OF TIMES THE SERIES HAS BEEN DIFFERENCED to be rendered stationary.**

$$y'_t = C + \phi_1 y'_{t-1} + \dots + \phi_p y'_{t-p} + \theta_1\varepsilon_{t-1} + \dots + \theta_q\varepsilon_{t-q} + \varepsilon_t$$

where **$y'_t$ represents the DIFFERENCED series, and it may have been differenced MORE THAN ONCE.**

> **We have to FIND the order of integration, which corresponds to the MINIMUM NUMBER OF TIMES a series must be differenced to become stationary.** *(Lecture: if the series has a quadratic trend, $d=2$ might be necessary.)*

> **⚠️ What happens if you apply ARMA to a non-stationary series** (lecture): **the residuals are NOT white noise; the model may FAIL to capture the underlying structure; forecasting becomes UNRELIABLE.**

### The full ARIMA procedure

> 1. **Check stationarity of the original series** (ADF or other tests).
> 2. **Apply differencing to achieve stationarity** — determine $d$ (minimum number of differences).
> 3. **Fit ARMA(p,q) to the differenced series** — try multiple combinations of $p$ and $q$, use **AIC** to select the best.
> 4. **Check residuals** — must behave like white noise (**Q-Q plot** qualitative, **Ljung-Box** quantitative).
> 5. **Forecast future values of the differenced series.**
> 6. **RECONSTRUCT the original series by REVERSING the differencing.**

**The worked result on the EPS series** (which shows a strong upward trend): **the best model is ARIMA(3,2,3)** — indeed **the Ljung-Box test on the first 10 lags of the residuals returns p-values ALL LARGER THAN 0.05.**

> **KEY IDEA: ARIMA captures BOTH the TREND (through integration/differencing) AND the SHORT-TERM DEPENDENCIES (AR and MA components) in the stationary transformed series.**

## 11. SARIMA — non-stationary series with seasonality

> **The problem** (lecture): **non-stationarity is caused by TRENDS *and* SEASONAL patterns.** After first-order differencing the series **may become trend-stationary, BUT THE SEASONAL PATTERN REMAINS.** To make it completely stationary we often need **SEASONAL DIFFERENCING** — subtracting the value from the same season in the previous period:
> $$y'_t = y_t - y_{t-m}$$
> where **$m$ is the length of the seasonality** (12 months, 4 quarters…).

**The benchmark** is the **naïve seasonal method**: *the first quarter of 1979 is used to forecast the EPS of the first quarter of 1980; the second quarter of 1979 forecasts the second quarter of 1980, and so on.* **It captures seasonality but IGNORES TREND.**

**The example**: the **monthly total number of air passengers for an airline** — *an increasing trend AND seasonality.*

> **SEASONAL AUTOREGRESSIVE INTEGRATED MOVING AVERAGE (SARIMA): adds seasonal parameters to ARIMA(p,d,q).**
> **Denoted SARIMA(p,d,q)(P,D,Q)$_m$**, where:
> - **$P$ = order of the seasonal AR(P) process;**
> - **$D$ = seasonal order of integration;**
> - **$Q$ = order of the seasonal MA(Q) process;**
> - **$m$ = the FREQUENCY, i.e. the NUMBER OF OBSERVATIONS PER SEASONAL CYCLE.**
>
> **⚠️ A SARIMA(p,d,q)(0,0,0)$_m$ model is EQUIVALENT to an ARIMA(p,d,q) model.**
> **We can think of SARIMA as TWO SUPERIMPOSED MODELS: one LOCAL and one SEASONAL.** *(Lecture: "it is like merging the ARMA model with another ARMA model that takes seasonality into consideration.")*

### 11.1 How to choose $P$, $D$, $Q$ — with $m = 12$

> - **$P = 2$** ⇒ we include **two past values of the series at a lag that is a MULTIPLE OF $m$** ⇒ the values at **$y_{t-12}$ and $y_{t-24}$.**
> - **$D = 1$** ⇒ **a seasonal difference makes the series stationary**; with $m=12$ this corresponds to $y'_t = y_t - y_{t-12}$.
> - **$Q = 2$** ⇒ we include **past error terms at lags that are multiples of $m$** ⇒ the errors **$\varepsilon_{t-12}$ and $\varepsilon_{t-24}$.**

### 11.2 How to identify seasonal patterns

> - **VISUALLY, by plotting the series;**
> - **through TIME SERIES DECOMPOSITION (the `STL` function from the `statsmodels` library)** — separating the series into **trend, seasonal component, and residuals.**

**Two contrasting decomposition results:**
| Case | Seasonal component |
|---|---|
| **Air passengers, $m=12$** | **the seasonal component IS PRESENT** *(note: the scale of the residuals differs, so they are not as large as they appear)* |
| **A simulated LINEAR series, $m=12$** | **the seasonal component is a FLAT HORIZONTAL LINE AT 0 ⇒ NO seasonal pattern** |

> **⚠️ A complication noted in the lecture**: **seasonal autocorrelation can show MULTIPLE PEAKS at multiples of the seasonal period, making it harder to choose $P$ and $Q$.**

### 11.3 The procedure and the result

> **We have to check ALL possible parameters (a GRID SEARCH), running TWO experiments:**
> 1. **SARIMA(p,d,q)(0,0,0)$_m$** — equivalent to ARIMA(p,d,q), i.e. **no seasonal component** ⇒ tests whether seasonal effects are needed;
> 2. **SARIMA(p,d,q)(P,D,Q)$_m$** — the **full seasonal model.**
> **Then compute the AIC across all combinations and select the lowest; validate residuals with Q-Q plot, histogram, ACF and Ljung-Box.**

**Experiment 1 — SARIMA(p,d,q)(0,0,0)$_m$**: **the Ljung-Box p-values are all greater than 0.05 EXCEPT FOR THE FIRST TWO** ⇒ **the residuals are uncorrelated STARTING AT LAG 3** ⇒ **the test suggests the model is NOT PERFECT, although it is not performing so badly.**

**Experiment 2 — the full model**: **$d$ and $D$ equal to 1; values in $[0,1,2,3]$ tried for $p, q, P, Q$** ⇒ **SARIMA(2,1,1)(1,1,2)$_{12}$ has the LOWEST AIC**, and **the Ljung-Box p-values are ALL greater than 0.05.**

**The final comparison (MAPE):**

| Model | MAPE |
|---|---|
| **Naïve seasonal** | **9.99%** |
| **ARIMA(1,1,2)** | **3.85%** |
| **SARIMA(2,1,1)(1,1,2)$_{12}$** | **2.85%** |

> **⇒ The ARIMA model WITHOUT seasonality presents a HIGHER error than SARIMA — the seasonal model approximates the real test set better.**

## 12. SARIMAX — adding exogenous variables

> **Often, EXTERNAL VARIABLES are also predictive of a time series.** *Example: **Gross Domestic Product (GDP)** — the total market value of all finished goods and services produced within a country — is defined as the sum of **consumption C, government spending G, investments I, and net exports NX**. **Each element is probably affected by external variables.***

> **The SARIMAX model, where X denotes EXOGENOUS VARIABLES, simply adds a LINEAR COMBINATION OF EXOGENOUS VARIABLES to the SARIMA model.** This allows **modelling the impact of external variables on the future value of a time series.**

**⚠️ The hierarchy of models — know how SARIMAX degenerates:**
> **The SARIMAX model is THE MOST GENERAL model for forecasting time series:**
> - **no seasonal patterns ⇒ it becomes ARIMAX;**
> - **no exogenous variables ⇒ it becomes SARIMA;**
> - **no seasonality AND no exogenous variables ⇒ it becomes ARIMA.**

### 12.1 ⚠️ The warning about multi-step forecasting

> **What if you wish to predict TWO timesteps into the future?** **While this is possible with a SARIMA model, THE SARIMAX MODEL REQUIRES US TO FORECAST THE EXOGENOUS VARIABLES TOO.**
>
> *(Lecture: at instant $t$, $y_t$ depends on the SARIMA part **but also on the linear combination of the values of the exogenous variables at instant $t$**. To predict the next value, we would have to assume the exogenous variables are available at $t+1$ or $t+2$.)*
>
> **⇒ The only way to avoid that situation is to PREDICT ONLY ONE TIMESTEP INTO THE FUTURE and WAIT TO OBSERVE the exogenous variable before predicting the target for another timestep.**
>
> **⇒ On the other hand, if your exogenous variable is EASY TO PREDICT — it follows a known function that can be accurately predicted — THERE IS NO HARM in forecasting it and using those forecasts to predict the target.**

**The general procedure**: **no real changes with respect to the SARIMA procedure, except for the use of SARIMAX as the fitting model.** In the worked example the **optimal model is ARIMAX(3,1,3) — no seasonality.**

---

## Key points / potential exam pitfalls

### Fundamentals
- **⚠️ A time series is ORDERED and EQUALLY SPACED, and the ORDER MATTERS — so the train/test split must be CHRONOLOGICAL, never random.** This is the single most important practical difference from classical ML.
- **Two reasons time series differ from regression: (1) they have an ORDER; (2) they can be forecast WITHOUT FEATURES, using only past values.**
- **Decomposition = TREND + SEASONALITY + RESIDUALS.** Be able to distinguish **seasonality** (repeats at a fixed period) from **trend** (long-term direction) from **noise** (small random fluctuations).
- **MAPE normalizes the error by the magnitude of the actual value** — that is why it is used instead of a raw absolute error.
- **⚠️ Memorise the three J&J baselines and their MAPEs: MEAN 70% → LAST VALUE 30.45% → NAÏVE SEASONAL 11.56%**, and *why* each improves on the previous (trend, then seasonality).

### Random walk and stationarity
- **Random walk: $y_t = C + \alpha_1 y_{t-1} + \varepsilon_t$ with $\alpha_1 = 1$; its FIRST DIFFERENCE is STATIONARY AND UNCORRELATED (white noise).** Shocks **ACCUMULATE**, which is why it drifts.
- **Stationary = constant MEAN, VARIANCE and AUTOCORRELATION, independent of time.** The condition on the AR(1) root is **$-1 < \alpha_1 < 1$** (extremes excluded).
- **⚠️ WHITE NOISE IS STATIONARY; A RANDOM WALK IS NOT.**
- **Differencing stabilizes the MEAN; LOGARITHMS stabilize the VARIANCE.** **And always UNTRANSFORM the forecasts back to the original units.**
- **⚠️ ADF test: $H_0$ = A UNIT ROOT IS PRESENT = NON-STATIONARY. SMALL p-value ⇒ REJECT ⇒ STATIONARY.** Know the worked numbers: raw random walk **$-0.97$, p $= 0.77$** (non-stationary); differenced **$-31.79$, p $= 0$** (stationary).
- **⚠️ Ljung-Box goes the OTHER WAY: $H_0$ = NO autocorrelation, and you WANT a LARGE p-value ($> 0.05$) to accept the model.** Mixing up the two conventions is the classic exam error.

### ACF, PACF, and model identification
- **ACF measures correlation of the series WITH ITSELF at lag $k$.** **TREND ⇒ high coefficients at short lags decaying LINEARLY. SEASONALITY ⇒ CYCLICAL patterns in the ACF.**
- **⚠️ SLOW DECAY of the ACF signals NON-STATIONARITY** — because a stationary process is pulled back toward its mean, so distant values stop being informative.
- **The shaded band is the 95% confidence interval (Bartlett's standard errors); inside ⇒ not significantly different from 0.**
- **⚠️ THE PACF EXISTS BECAUSE THE ACF IS CONFOUNDED**: measuring $y_t$ vs. $y_{t-2}$ ignores that $y_{t-1}$ influences BOTH. The PACF **removes the influence of the intermediate lags.**
- **⚠️ THE IDENTIFICATION TABLE:**
  | | cut-off in | order |
  |---|---|---|
  | **MA(q)** | **ACF** | $q$ |
  | **AR(p)** | **PACF** | $p$ |
  | **ARMA(p,q)** | **NEITHER cuts off** (ACF decays slowly, PACF sinusoidal) ⇒ **grid search + AIC** | $p, q$ |
- **AR = present depends on past VALUES. MA = present depends on past SHOCKS (errors).** *"The system memory is contained in the errors"* is the MA slogan.
- **The RANDOM WALK is a special case of AR with $p=1$.**

### Models and procedure
- **⚠️ MA(q) can only forecast meaningfully $q$ STEPS AHEAD — beyond that it RETURNS THE MEAN.** Hence **ROLLING FORECASTS**: predict $\le q$ steps, append the newly observed real value, retrain, repeat. (Worked numbers: train on 449 → predict 450–451; train on 451 → predict 452–453.)
- **AIC $= 2k - 2\ln\hat{L}$ with $k = p+q$; LOWEST AIC WINS.** It **balances GOODNESS OF FIT against COMPLEXITY** — the penalty exists because **more parameters can overfit.**
- **Residual analysis is MANDATORY after fitting**: **Q-Q plot (qualitative) + Ljung-Box (quantitative)**, plus the residual time plot, histogram and residual ACF. **A good model leaves ONLY WHITE NOISE.**
- **⚠️ ARIMA(p,d,q): $d$ = the MINIMUM number of differencings needed for stationarity, and INTEGRATION IS THE REVERSE OF DIFFERENCING.** Remember the final step: **RECONSTRUCT the original series by reversing the differencing.** Worked result: **ARIMA(3,2,3)** for the EPS series.
- **SARIMA(p,d,q)(P,D,Q)$_m$: $m$ = observations per seasonal cycle; $P$ and $Q$ act at lags that are MULTIPLES OF $m$; $D$ is SEASONAL differencing $y_t - y_{t-m}$.** **SARIMA(p,d,q)(0,0,0)$_m$ = ARIMA(p,d,q).**
- **Worked SARIMA result: $d = D = 1$, grid over $[0,1,2,3]$ ⇒ SARIMA(2,1,1)(1,1,2)$_{12}$**, and the **MAPE ladder: naïve seasonal 9.99% → ARIMA(1,1,2) 3.85% → SARIMA 2.85%.**
- **⚠️ THE MODEL HIERARCHY: SARIMAX is the most general. Drop seasonality ⇒ ARIMAX. Drop exogenous ⇒ SARIMA. Drop both ⇒ ARIMA.**
- **⚠️ THE SARIMAX TRAP: to forecast more than ONE step ahead you must ALSO FORECAST THE EXOGENOUS VARIABLES.** The safe strategy is **one step at a time, waiting to observe the exogenous variable** — unless it is easy to predict from a known function.
- **⚠️ Forecasting a RANDOM WALK on a LONG HORIZON does not make sense**; only baselines apply (**historical mean, last known value, DRIFT method** = slope between first and last training value, extrapolated).

### The master procedure (be able to recite it)
> **1. Plot + ADF ⇒ stationary? → 2. If not, DIFFERENCE (and log if the variance drifts), re-test → 3. ACF: no significant autocorrelation ⇒ RANDOM WALK, use baselines only → 4. ACF cuts off at $q$ ⇒ MA(q) → 5. PACF cuts off at $p$ ⇒ AR(p) → 6. Neither cuts off ⇒ ARMA/ARIMA: grid-search $(p,q)$ [and $(P,D,Q)_m$ if seasonal], select by AIC → 7. Validate residuals (Q-Q + Ljung-Box) → 8. Forecast → 9. UNTRANSFORM back to the original scale.**

### Cross-chapter connections
- **The Q-Q plot** was introduced in **Chapter 2** for checking distributional assumptions; here it is reused to test whether residuals are normal.
- **Overfitting and the complexity/goodness-of-fit trade-off** behind **AIC** is the same tension as in **Chapter 4** (model complexity vs. generalization); AIC plays the role that pruning and validation play there.
- **The train/test split discipline** comes from **Chapter 4** — but with the crucial modification that **randomness is forbidden here.**
- **Autocorrelation and lag structure** connect to the **temporal ordering** that made **Chapter 12 (Sequential Patterns)** different from plain frequent patterns: both chapters are about what changes when order carries information.
- **Concept drift** (**Chapter 15**) is the streaming analogue of **non-stationarity** here: in both cases the statistical properties move, and in both cases the response is to transform, re-estimate, or adapt.
- **Exogenous variables** in SARIMAX play the role that **features** play in ordinary regression (**Chapter 4**) — this is where time series forecasting reconnects with standard supervised learning, which is exactly the bridge to **Chapter 17**.

---

*File auto-generated by merging `16-TimeSeries_TraditionalApproaches.pdf` (professor's slides, 33 pages, extracted from Peixeiro, "Time Series Forecasting in Python", Manning 2022) and `DM_11-17 (1)-115-147.pdf` (lecture notes, 33 pages — lesson L17 of 05/05/2026, containing two passes over the same material, both merged here). Formulas that the deck stored as images (MAPE, the random walk and AR/MA/ARMA/ARIMA equations, the drift formula, AIC, seasonal differencing) have been reconstructed in standard form. The final page of the notes opens the Machine Learning chapter, which is covered in Chapter 17. For questions about this chapter, refer only to this file.*
