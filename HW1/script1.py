import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df = pd.read_csv('data/investment.csv')
df.rename(columns={'annual_invest_musd': 'annual_invest', 'cum_invest_musd': 'cum_invest'}, inplace=True)

def N_bass(t, p, q, M):
    expo = np.exp(-(p + q) * t)
    return M * (1 - expo) / (1 + (q / p) * expo) # bass.F() for F(t)

t = (df['year'] - df['year'].min()).values
y = df['cum_invest'].values

p0 = 0.02 #some initial guess
q0 = 0.4 #some initial guess
M0 = 0

popt, pcov = curve_fit(N_bass, t, y, p0=[p0, q0, M0])
p_est, q_est, M_est = popt

future_years = np.arange(2016, 2046)  
t_future = (future_years - 2016)
N_hat = N_bass(t_future, *popt)
A_hat = np.diff(np.r_[0, N_hat])  

forecast_df = pd.DataFrame({
    "year": future_years,
    "cum_invest_hat": N_hat,
    "annual_invest": A_hat
})


df.to_csv('data/investment.csv', index=False)
forecast_df.to_csv('data/investment_forecast.csv', index=False)
params_df = pd.DataFrame({
    "param": ["p", "q", "M_usd_millions"],
    "estimate": [p_est, q_est, M_est],
})
params_df.to_csv('data/bass_params_investment_proxy.csv', index=False)


cf = 7000  # I am assuming we need about 7000$ to get one adopter, since there are very few adopters

forecast_df[f'cum_adopters_units_cf_{cf}'] = forecast_df['cum_invest_hat'] * 1e6 / cf
forecast_df[f'annual_adopters_units_cf_{cf}'] = forecast_df['annual_invest'] * 1e6 / cf

forecast_df.to_csv('data/investment_forecast_with_conversions.csv', index=False)

#cumulative
plt.figure(figsize=(10,6))
plt.plot(df['year'], df['cum_invest'], 'o', label='Observed cumulative invested ($M)')
plt.plot(forecast_df['year'], forecast_df['cum_invest_hat'], '-', label='Bass fit & forecast (cum $M)')
plt.xlabel('Year'); plt.ylabel('Cumulative invested capital (USD millions)')
plt.title('Bass model fit to cumulative invested capital (GFI proxy)')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig('img/bass_model_fit_cumulative.png', dpi=300, bbox_inches='tight')
plt.show()

#standard
plt.figure(figsize=(10,6))
plt.plot(df['year'], df['annual_invest'], 'o', label='Observed annual invested ($M)')
plt.plot(forecast_df['year'], forecast_df['annual_invest'], '-', label='Bass forecast (annual $M)')
plt.xlabel('Year'); plt.ylabel('Annual invested capital (USD millions)')
plt.title('Bass model forecast for annual invested capital (GFI proxy)')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig('img/bass_model_annual_fit_standard.png', dpi=300, bbox_inches='tight')
plt.show()
