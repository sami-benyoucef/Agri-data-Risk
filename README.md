# Agricultural Commodities: Quantitative Risk & Structural Modeling

This repository contains a quantitative framework for analyzing structural risk regimes, price volatility, and supply-demand statistics within agricultural (soft) commodity markets. 

## 📌 Project Overview
The objective of this modeling suite is to ingest fundamental agricultural datasets, engineer statistical features, and evaluate market risk exposures. By applying robust econometric and statistical techniques, this toolkit provides actionable insights into tail-risk and volatility clustering.

## ⚙️ Core Architecture & Data Operations
* **Data Pipelines:** Automated ingestion and cleaning of agricultural and climatic datasets (only light agri version present in the repo).
* **Feature Engineering:** Transformation of fundamental metrics (e.g., crop yields, weather anomalies) into stationary time-series for predictive modeling.
* **Risk Analytics:** Calculation of empirical and parametric risk metrics.

## 🧮 Mathematical Framework

### Value at Risk (VaR) & Expected Shortfall (CVaR)
The framework evaluates downside tail risk by computing the Conditional Value at Risk (CVaR), focusing on the expected loss given that the loss exceeds the VaR threshold:

$$\text{CVaR}_\alpha = \frac{1}{1-\alpha} \int_{-1}^{\text{VaR}_\alpha} x \cdot f(x) \, dx$$

Where $f(x)$ represents the probability density function of the commodity returns under stressed fundamental scenarios.

### Volatility Modeling
Implementation of rolling-window variance analysis to capture volatility regimes and heteroskedasticity inherent in agricultural supply shocks:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

## 🚀 Quick Start
```bash
# Clone the repository
git clone [https://github.com/sami-benyoucef/Agri-data-Risk.git](https://github.com/sami-benyoucef/Agri-data-Risk.git)

# Install dependencies
pip install -r requirements.txt# EdC-BCG-Agri-data
