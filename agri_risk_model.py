import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class AgriRiskModel:
    """
    Quantitative framework for evaluating structural risk regimes, 
    price volatility, and tail-risk statistics in agricultural commodities.
    """
    
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.data = None

    def generate_synthetic_data(self, days=1000, initial_price=100.0):
        """
        Generates synthetic commodity price data with volatility clustering 
        (GARCH-like behavior) to simulate agricultural supply shocks.
        """
        np.random.seed(42)
        returns = np.zeros(days)
        volatility = np.zeros(days)
        
        # Baseline volatility and shock parameters
        volatility[0] = 0.015 
        omega, alpha, beta = 0.00002, 0.1, 0.85 

        for t in range(1, days):
            # Simulate heteroskedasticity (volatility clustering)
            volatility[t] = np.sqrt(omega + alpha * (returns[t-1]**2) + beta * (volatility[t-1]**2))
            returns[t] = np.random.normal(0, volatility[t])

        prices = initial_price * np.exp(np.cumsum(returns))
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='B')
        
        self.data = pd.DataFrame({'Price': prices}, index=dates)
        print(f"[*] Ingested {days} days of simulated commodity price data.")
        return self.data

    def ingest_csv(self, filepath, date_col, price_col):
        """
        Production method to ingest real historical market data.
        """
        try:
            df = pd.read_csv(filepath, parse_dates=[date_col], index_col=date_col)
            self.data = pd.DataFrame({'Price': df[price_col]})
            print(f"[*] Successfully ingested data from {filepath}")
        except Exception as e:
            print(f"[!] Data ingestion failed: {e}")

    def calculate_features(self):
        """
        Calculates log returns and exponentially weighted moving average (EWMA) volatility.
        """
        if self.data is None:
            raise ValueError("No data ingested. Run ingestion first.")
            
        # Log Returns
        self.data['Log_Returns'] = np.log(self.data['Price'] / self.data['Price'].shift(1))
        
        # EWMA Volatility (RiskMetrics approach, lambda = 0.94)
        self.data['EWMA_Vol'] = self.data['Log_Returns'].ewm(alpha=0.06, adjust=False).std()
        
        self.data.dropna(inplace=True)
        print("[*] Feature engineering complete (Log Returns, EWMA Volatility).")

    def calculate_historical_var(self):
        """Calculates non-parametric Historical Value at Risk."""
        var = np.percentile(self.data['Log_Returns'], self.alpha * 100)
        return var

    def calculate_parametric_var(self):
        """Calculates Parametric (Gaussian) Value at Risk."""
        mu = np.mean(self.data['Log_Returns'])
        sigma = np.std(self.data['Log_Returns'])
        var = stats.norm.ppf(self.alpha, mu, sigma)
        return var

    def calculate_cvar(self):
        """Calculates Conditional Value at Risk (Expected Shortfall)."""
        historical_var = self.calculate_historical_var()
        tail_losses = self.data['Log_Returns'][self.data['Log_Returns'] <= historical_var]
        cvar = tail_losses.mean()
        return cvar

    def generate_risk_report(self):
        """Compiles structural risk metrics into a standardized report."""
        hist_var = self.calculate_historical_var()
        param_var = self.calculate_parametric_var()
        cvar = self.calculate_cvar()
        ann_vol = self.data['Log_Returns'].std() * np.sqrt(252)

        print("\n" + "="*45)
        print(f"   AGRICULTURAL COMMODITIES RISK REPORT")
        print("="*45)
        print(f"Confidence Level:      {self.confidence_level * 100:.1f}%")
        print(f"Annualized Volatility: {ann_vol * 100:.2f}%")
        print("-" * 45)
        print(f"Historical VaR (Daily): {hist_var * 100:.2f}%")
        print(f"Parametric VaR (Daily): {param_var * 100:.2f}%")
        print(f"Conditional VaR (CVaR): {cvar * 100:.2f}%")
        print("="*45 + "\n")

    def plot_volatility_regimes(self):
        """Visualizes price action against volatility regimes and risk thresholds."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Price Subplot
        ax1.plot(self.data.index, self.data['Price'], color='#1f77b4', label='Commodity Price')
        ax1.set_title('Commodity Price Action', fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Returns and VaR Subplot
        hist_var = self.calculate_historical_var()
        cvar = self.calculate_cvar()
        
        ax2.plot(self.data.index, self.data['Log_Returns'], color='gray', alpha=0.5, label='Daily Returns')
        ax2.axhline(hist_var, color='orange', linestyle='--', linewidth=2, label=f'VaR ({self.confidence_level*100:.0f}%)')
        ax2.axhline(cvar, color='red', linestyle='-', linewidth=2, label='CVaR (Expected Shortfall)')
        
        ax2.set_title('Returns & Tail Risk Metrics', fontweight='bold')
        ax2.set_ylabel('Log Returns')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 1. Initialize the Quantitative Model (99% Confidence Interval)
    model = AgriRiskModel(confidence_level=0.99)
    
    # 2. Ingest Data (Using synthetic GARCH-like data for demonstration)
    # To use real data, swap this with: model.ingest_csv('your_data.csv', 'Date', 'Settle_Price')
    model.generate_synthetic_data(days=1260) # Approx 5 years of trading days
    
    # 3. Engineer Structural Features
    model.calculate_features()
    
    # 4. Run Risk Analytics & Output
    model.generate_risk_report()
    
    # 5. Visualize Regimes
    model.plot_volatility_regimes()
