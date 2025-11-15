# Customer Churn Survival Analysis

## Overview
Survival analysis on telecommunications customer data using AFT models to identify churn risk factors, calculate Customer Lifetime Value (CLV), and recommend retention strategies.

## Setup & Run

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install lifelines pandas numpy matplotlib scipy

# Run analysis
python survival_analysis.py
```

## Output Files

**Visualizations:**
- `aft_model_comparison.png` - Survival curves for Weibull, Log-Normal, and Log-Logistic models
- `clv_analysis.png` - CLV distribution and segment analysis

**Data:**
- `customer_clv_analysis.csv` - Customer-level CLV, survival probabilities, and churn risk scores

**Report:**
- `report.md` - Key findings and recommendations

## What It Does

1. Fits 3 AFT models (Weibull, Log-Normal, Log-Logistic) and selects the best one
2. Identifies significant features affecting customer churn
3. Calculates CLV for each customer using survival probabilities
4. Analyzes CLV across different customer segments
5. Estimates retention budget based on expected value loss
6. Generates visualizations and detailed customer data
