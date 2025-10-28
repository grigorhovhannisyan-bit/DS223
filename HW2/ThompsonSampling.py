import numpy as np
import pandas as pd
from Bandit import Bandit


class ThompsonSampling(Bandit):
    
    def __init__(self, p: list[float], precision: float = 1.0):
        self.p = np.array(p)
        # initialize with uniform distribution B(1,1).
        self.alpha: np.array = np.ones(len(p)) * precision
        self.beta: np.array = np.ones(len(p)) * precision
        self.precision: float = precision
        self.rewards: list[float] = []
        self.choices: list[int] = []
        self.alpha_history: list = []  # track alpha over time
        self.beta_history: list = []   # track beta over time

    def __repr__(self):
        # called when we print the object
        return f"ThompsonSampling(precision={self.precision}, arms={len(self.p)})"

    def pull(self, arm: int) -> float:
        # pull an arm and receive a reward 
        return np.random.random() < self.p[arm]

    def update(self, arm, reward):
        # update the beta distribution parameters after pulling an arm (for Beta-Bernoulli conjugate prior)
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def experiment(self, num_trials: int = 1000):
        for trial in range(num_trials):
            # sample from the beta distribution for each arm
            sampled_values = np.random.beta(self.alpha, self.beta)
            
            # choose the arm with the highest sampled value
            arm = np.argmax(sampled_values)
            
            # pull the arm and get the reward
            reward = self.pull(arm)
            
            # update the posterior distribution
            self.update(arm, reward)
            
            # Store history
            self.rewards.append(reward)
            self.choices.append(arm)
            self.alpha_history.append(self.alpha.copy())
            self.beta_history.append(self.beta.copy())

    def report(self, filename: str = 'results_thompson.csv'):
        # report the performance of the thompson sampling algorithm
        cumulative_reward = np.sum(self.rewards)
        
        # calculate expected optimal reward (expected value if always chose best arm)
        # regret measures the difference from optimal expected performance
        expected_optimal_reward = np.max(self.p) * len(self.rewards)
        cumulative_regret = expected_optimal_reward - cumulative_reward
        
        avg_reward = cumulative_reward / len(self.rewards) if self.rewards else 0
        
        # calculate the posterior mean estimates: E[theta] = alpha / (alpha + beta)
        posterior_means = self.alpha / (self.alpha + self.beta)
        
        # calculate how many times each arm was pulled
        arm_pulls = np.bincount(self.choices, minlength=len(self.p))
        arm_pull_percentages = (arm_pulls / len(self.choices)) * 100
        
        # save to CSV with arm selection counts included
        df = pd.DataFrame({
            'Bandit': [f'Arm_{i}' for i in self.choices],
            'Reward': self.rewards,
            'Algorithm': ['ThompsonSampling'] * len(self.rewards)
        })
        
        # add summary rows for arm selection counts
        for i in range(len(self.p)):
            summary_row = pd.DataFrame({
                'Bandit': [f'Arm_{i}_Total_Pulls'],
                'Reward': [arm_pulls[i]],
                'Algorithm': ['Summary']
            })
            df = pd.concat([df, summary_row], ignore_index=True)
        
        df.to_csv(filename, index=False)
        
        # print the report
        print(f"\n{'='*60}")
        print(f"THOMPSON SAMPLING ALGORITHM REPORT")
        print(f"{'='*60}")
        print(f"Total Trials: {len(self.rewards)}")
        print(f"Cumulative Reward: {cumulative_reward:.2f}")
        print(f"Cumulative Regret: {cumulative_regret:.2f}")
        print(f"Average Reward per Trial: {avg_reward:.4f}")
        print(f"Optimal Average Reward: {np.max(self.p):.4f}")
        print(f"Precision: {self.precision}")
        print(f"Posterior Mean Estimates: {posterior_means}")
        print(f"True Probabilities: {self.p}")
        print(f"Alpha parameters: {self.alpha}")
        print(f"Beta parameters: {self.beta}")
        print(f"\nArm Selection Count:")
        for i, (count, pct) in enumerate(zip(arm_pulls, arm_pull_percentages)):
            print(f"  Arm {i}: {count:>6} pulls ({pct:>6.2f}%)")
        print(f"Results saved to: {filename}")
        print(f"{'='*60}\n")
        
        return {
            'cumulative_reward': cumulative_reward,
            'cumulative_regret': cumulative_regret,
            'avg_reward': avg_reward,
            'rewards': self.rewards,
            'choices': self.choices,
            'posterior_means': posterior_means,
            'arm_pulls': arm_pulls,
            'alpha_history': np.array(self.alpha_history),
            'beta_history': np.array(self.beta_history),
            'true_probabilities': self.p
        }