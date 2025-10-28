# Multi-Armed Bandit: Epsilon-Greedy vs Thompson Sampling

## Overview

1. **Epsilon-Greedy**: A simple yet effective algorithm that balances exploration and exploitation using a decaying epsilon parameter (ε = 1/t)
2. **Thompson Sampling**: A Bayesian approach using Beta-Bernoulli conjugate priors with known precision

## Project Structure

```
HW2/
├── data/
│   ├── epsilon_greedy_results.csv
│   └── thompson_sampling_results.csv
│
├── graphs/
│   ├── epsilon_greedy_learning.png
│   ├── thompson_sampling_learning.png
│   └── algorithm_comparison.png
│
├── Bandit.py
├── EpsilonGreedy.py
├── ThompsonSampling.py
├── Performance.py
├── main.py
├── requirements.txt
└── README.md
```

## Running the Code

```bash
# 1. Activate virtual environment (if you have one, if not create one or skip this)
source ../venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the experiment
python main.py
```

