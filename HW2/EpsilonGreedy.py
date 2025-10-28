import numpy as np
import pandas as pd
from Bandit import Bandit


class EpsilonGreedy(Bandit):

    def __init__(self, p: list[float], epsilon: float = 1.0):
        self.p: np.array = np.array(p)
        self.p_estimate: np.array = np.zeros(len(p))
        self.N: np.array = np.zeros(len(p))
        self.epsilon: float = epsilon
        self.t: int = 0
        self.rewards: list[float] = []
        self.choices: list[int] = []
        self.p_estimate_history: list = []  # track probability estimates over time

    def __repr__(self):
        # called when we print the object
        return f"EpsilonGreedy(epsilon={self.epsilon:.4f}, arms={len(self.p)})"

    def pull(self, arm: int) -> float:
        # pull an arm and receive a reward 
        return np.random.random() < self.p[arm]

    def update(self, arm: int, reward: float):
        # update estimates after pulling an arm
        self.N[arm] += 1
        self.p_estimate[arm] += (reward - self.p_estimate[arm]) / self.N[arm]

    def experiment(self, num_trials: int = 1000):
        # run the epsilon-greedy experiment for a given number of trials
        for trial in range(num_trials):
            self.t += 1
            
            # epsilon decays logarithmically: 1/log(t+1) for slower decay
            current_epsilon = self.epsilon / np.log(self.t + 1)
            
            if np.random.random() < current_epsilon:
                # explore
                arm = np.random.randint(len(self.p))
            else:
                # exploit
                arm = np.argmax(self.p_estimate)
            
            # pull arm and get reward
            reward = self.pull(arm)
            
            # update estimates
            self.update(arm, reward)
            
            # store history
            self.rewards.append(reward)
            self.choices.append(arm)
            self.p_estimate_history.append(self.p_estimate.copy())

    def report(self, filename: str = 'results.csv'):
        # report the performance of the epsilon-greedy algorithm

        cumulative_reward = np.sum(self.rewards)
        
        # calculate expected optimal reward (expected value if always chose best arm)
        # regret measures the difference from optimal expected performance
        expected_optimal_reward = np.max(self.p) * len(self.rewards)
        cumulative_regret = expected_optimal_reward - cumulative_reward 
        
        avg_reward = cumulative_reward / len(self.rewards) if self.rewards else 0
        
        # calculate how many times each arm was pulled
        arm_pulls = np.bincount(self.choices, minlength=len(self.p))
        arm_pull_percentages = (arm_pulls / len(self.choices)) * 100
        
        # save to CSV with arm selection counts included
        df = pd.DataFrame({
            'Bandit': [f'Arm_{i}' for i in self.choices],
            'Reward': self.rewards,
            'Algorithm': ['EpsilonGreedy'] * len(self.rewards)
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
        
        # print report
        print(f"\n{'='*60}")
        print(f"EPSILON-GREEDY ALGORITHM REPORT")
        print(f"{'='*60}")
        print(f"Total Trials: {len(self.rewards)}")
        print(f"Cumulative Reward: {cumulative_reward:.2f}")
        print(f"Cumulative Regret: {cumulative_regret:.2f}")
        print(f"Average Reward per Trial: {avg_reward:.4f}")
        print(f"Optimal Average Reward: {np.max(self.p):.4f}")
        print(f"Final Epsilon: {self.epsilon / np.log(self.t + 1):.6f}")
        print(f"Estimated Probabilities: {self.p_estimate}")
        print(f"True Probabilities: {self.p}")
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
            'arm_pulls': arm_pulls,
            'p_estimate_history': np.array(self.p_estimate_history),
            'true_probabilities': self.p
        }