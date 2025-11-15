import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter, ExponentialFitter
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('telco.csv')
print("Churn rate:", (df['churn'] == 'Yes').sum() / len(df) * 100, "%")

df['churn_event'] = (df['churn'] == 'Yes').astype(int)
df['region_Zone1'] = (df['region'] == 'Zone 1').astype(int)
df['region_Zone2'] = (df['region'] == 'Zone 2').astype(int)
df['region_Zone3'] = (df['region'] == 'Zone 3').astype(int)
df['marital_Married'] = (df['marital'] == 'Married').astype(int)
ed_order = {
    'Did not complete high school': 1,
    'High school degree': 2,
    'College degree': 3,
    'Post-undergraduate degree': 4
}
df['ed_level'] = df['ed'].map(ed_order)
df['retire_Yes'] = (df['retire'] == 'Yes').astype(int)
df['gender_Male'] = (df['gender'] == 'Male').astype(int)
df['voice_Yes'] = (df['voice'] == 'Yes').astype(int)
df['internet_Yes'] = (df['internet'] == 'Yes').astype(int)
df['forward_Yes'] = (df['forward'] == 'Yes').astype(int)

custcat_dummies = pd.get_dummies(df['custcat'], prefix='custcat')
df = pd.concat([df, custcat_dummies], axis=1)

feature_cols = ['age', 'marital_Married', 'address', 'income', 'ed_level', 
                'retire_Yes', 'gender_Male', 'voice_Yes', 'internet_Yes', 
                'forward_Yes', 'region_Zone2', 'region_Zone3',
                'custcat_E-service', 'custcat_Plus service', 'custcat_Total service']

survival_df = df[feature_cols + ['tenure', 'churn_event']].copy()
survival_df = survival_df.dropna()

print("\nFitting AFT models...")
models = {}
model_metrics = {}

weibull_aft = WeibullAFTFitter()
weibull_aft.fit(survival_df, duration_col='tenure', event_col='churn_event')
models['Weibull'] = weibull_aft
model_metrics['Weibull'] = {
    'AIC': weibull_aft.AIC_,
    'BIC': weibull_aft.BIC_,
    'log_likelihood': weibull_aft.log_likelihood_
}

lognormal_aft = LogNormalAFTFitter()
lognormal_aft.fit(survival_df, duration_col='tenure', event_col='churn_event')
models['LogNormal'] = lognormal_aft
model_metrics['LogNormal'] = {
    'AIC': lognormal_aft.AIC_,
    'BIC': lognormal_aft.BIC_,
    'log_likelihood': lognormal_aft.log_likelihood_
}

loglogistic_aft = LogLogisticAFTFitter()
loglogistic_aft.fit(survival_df, duration_col='tenure', event_col='churn_event')
models['LogLogistic'] = loglogistic_aft
model_metrics['LogLogistic'] = {
    'AIC': loglogistic_aft.AIC_,
    'BIC': loglogistic_aft.BIC_,
    'log_likelihood': loglogistic_aft.log_likelihood_
}

print("\nModel Comparison:")
comparison_df = pd.DataFrame(model_metrics).T
comparison_df = comparison_df.sort_values('AIC')
print(comparison_df)
print(f"Best model: {comparison_df.index[0]}")

best_model_name = comparison_df.index[0]
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name} (AIC: {best_model.AIC_:.2f})")

summary_df = best_model.summary
significant_features = summary_df[summary_df['p'] < 0.05]
print(f"Significant features: {len(significant_features)}")

significant_feature_names = [col for col in significant_features.index if col in feature_cols]
if len(significant_feature_names) > 0:
    print(f"Refitting with {len(significant_feature_names)} significant features...")
    survival_df_sig = survival_df[significant_feature_names + ['tenure', 'churn_event']].copy()
    
    if best_model_name == 'Weibull':
        final_model = WeibullAFTFitter()
    elif best_model_name == 'LogNormal':
        final_model = LogNormalAFTFitter()
    else:
        final_model = LogLogisticAFTFitter()
    
    final_model.fit(survival_df_sig, duration_col='tenure', event_col='churn_event')
    print(f"Final Model AIC: {final_model.AIC_:.2f}")
else:
    final_model = best_model
    survival_df_sig = survival_df

print("\nGenerating visualizations...")

fig, ax = plt.subplots(figsize=(12, 7))
time_points = np.linspace(0, survival_df['tenure'].max(), 100)

median_profile = survival_df[feature_cols].median().to_frame().T

for model_name, model in models.items():
    surv_func = model.predict_survival_function(median_profile, times=time_points)
    ax.plot(time_points, surv_func.values.flatten(), label=f'{model_name} AFT', linewidth=2.5)

ax.set_xlabel('Time (Tenure in Months)', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('Survival Curves: Comparison of AFT Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('aft_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Calculating CLV...")

monthly_revenue = 50  # Average monthly revenue per customer in $
discount_rate = 0.01  # Monthly discount rate (12% annual)
time_horizon = 60  # 5 years

df_clv = df.copy()

time_points_clv = np.arange(1, time_horizon + 1)

clv_values = []

for idx in range(len(survival_df_sig)):
    customer_features = survival_df_sig.iloc[[idx]][significant_feature_names if len(significant_feature_names) > 0 else feature_cols]
    
    surv_probs = final_model.predict_survival_function(customer_features, times=time_points_clv)
    
    clv = 0
    for t in range(len(time_points_clv)):
        discount_factor = (1 + discount_rate) ** (-time_points_clv[t])
        clv += monthly_revenue * surv_probs.iloc[t, 0] * discount_factor
    
    clv_values.append(clv)

df_clv = df_clv.iloc[survival_df_sig.index].copy()
df_clv['CLV'] = clv_values

print(f"Mean CLV: ${df_clv['CLV'].mean():.2f}, Total: ${df_clv['CLV'].sum():,.2f}")

df_clv['income_quartile'] = pd.qcut(df_clv['income'], 4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
df_clv['age_group'] = pd.cut(df_clv['age'], bins=[0, 30, 40, 50, 100], 
                              labels=['<30', '30-40', '40-50', '50+'])
df_clv['high_value'] = (df_clv['CLV'] > df_clv['CLV'].quantile(0.75)).astype(int)
df_clv['high_income'] = (df_clv['income'] > df_clv['income'].median()).astype(int)

print("Calculating retention metrics...")

one_year_surv_probs = []
for idx in range(len(survival_df_sig)):
    customer_features = survival_df_sig.iloc[[idx]][significant_feature_names if len(significant_feature_names) > 0 else feature_cols]
    surv_prob = final_model.predict_survival_function(customer_features, times=[12])
    one_year_surv_probs.append(surv_prob.iloc[0, 0])

df_clv['surv_prob_1year'] = one_year_surv_probs
df_clv['churn_risk_1year'] = 1 - df_clv['surv_prob_1year']

expected_loss = (df_clv['churn_risk_1year'] * df_clv['CLV']).sum()
retention_budget_low = expected_loss * 0.10
retention_budget_high = expected_loss * 0.20
print(f"Expected annual loss: ${expected_loss:,.2f}")
print(f"Recommended retention budget: ${retention_budget_low:,.2f} - ${retention_budget_high:,.2f}")

df_clv.to_csv('customer_clv_analysis.csv', index=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df_clv['CLV'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Customer Lifetime Value ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of CLV')
axes[0, 0].axvline(df_clv['CLV'].mean(), color='red', linestyle='--', label=f'Mean: ${df_clv["CLV"].mean():.0f}')
axes[0, 0].legend()

clv_custcat = df_clv.groupby('custcat')['CLV'].mean().sort_values(ascending=True)
axes[0, 1].barh(range(len(clv_custcat)), clv_custcat.values)
axes[0, 1].set_yticks(range(len(clv_custcat)))
axes[0, 1].set_yticklabels(clv_custcat.index)
axes[0, 1].set_xlabel('Average CLV ($)')
axes[0, 1].set_title('Average CLV by Customer Category')

axes[1, 0].hist(df_clv['surv_prob_1year'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_xlabel('1-Year Survival Probability')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of 1-Year Survival Probability')
axes[1, 0].axvline(0.5, color='red', linestyle='--', label='50% threshold')
axes[1, 0].legend()

scatter = axes[1, 1].scatter(df_clv['churn_risk_1year'], df_clv['CLV'], 
                             alpha=0.5, c=df_clv['CLV'], cmap='viridis')
axes[1, 1].set_xlabel('1-Year Churn Risk')
axes[1, 1].set_ylabel('Customer Lifetime Value ($)')
axes[1, 1].set_title('CLV vs Churn Risk')
axes[1, 1].axvline(0.5, color='red', linestyle='--', alpha=0.5)
plt.colorbar(scatter, ax=axes[1, 1], label='CLV ($)')

plt.tight_layout()
plt.savefig('clv_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated files: aft_model_comparison.png, clv_analysis.png, customer_clv_analysis.csv")

