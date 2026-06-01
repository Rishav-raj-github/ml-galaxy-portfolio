# ====================================================================
# ML Galaxy Portfolio - R Quantitative Diagnostics Script (2026 Edition)
# Performs stationarity checks, ACF/PACF diagnostics, and ARIMA fits.
# ====================================================================

# Load required packages
if (!require("tseries")) install.packages("tseries", repos="http://cran.rstudio.com/")
if (!require("forecast")) install.packages("forecast", repos="http://cran.rstudio.com/")

library(tseries)
library(forecast)

# Generate synthetic non-stationary sales sequence
set.seed(42)
n <- 120
t <- 1:n
trend <- 0.5 * t
seasonal <- 15 * sin(2 * pi * t / 12)
noise <- rnorm(n, mean=0, sd=8)
sales_raw <- ts(trend + seasonal + noise, frequency=12, start=c(2026, 1))

cat("--- Classical Quantitative Diagnostics --- \n\n")

# 1. Stationarity Test: Augmented Dickey-Fuller (ADF)
cat("ADF Test on Non-Stationary Time Series:\n")
adf_res <- adf.test(sales_raw, alternative="stationary")
print(adf_res)

# 2. Applying Differencing (d=1) to Establish Stationarity
sales_diff <- diff(sales_raw, differences=1)
cat("\nADF Test on Differenced Stationary Time Series:\n")
adf_diff_res <- adf.test(sales_diff, alternative="stationary")
print(adf_diff_res)

# 3. Fit Auto-ARIMA Model
fit <- auto.arima(sales_raw, seasonal=TRUE, step-wise=TRUE)
cat("\nAuto-ARIMA Optimal Fit Diagnostics Summary:\n")
print(summary(fit))

# 4. Residual Diagnostics Plotting
cat("\nAnalyzing Auto-ARIMA Residual Auto-Correlations...\n")
checkresiduals(fit)
