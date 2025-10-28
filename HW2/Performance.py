import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


class Performance:
    
    def __init__(self):
        # initialize the performance visualization class
        plt.style.use('default')
        
    def plot1(self, bandit_results, algorithm_name, save_path=None):
        # visualize probability estimates over time for epsilon-greedy
        # or beta distributions at specific trials for thompson sampling
        
        if 'p_estimate_history' in bandit_results:
            # epsilon-greedy: plot probability estimates over time
            self._plot_epsilon_greedy_learning(bandit_results, algorithm_name, save_path)
        elif 'alpha_history' in bandit_results:
            # thompson sampling: plot beta distributions at specific trials
            self._plot_thompson_distributions(bandit_results, algorithm_name, save_path)
    
    def _plot_epsilon_greedy_learning(self, results, algorithm_name, save_path):
        # plot how probability estimates change over time for each arm (one subplot per arm)
        p_history = results['p_estimate_history']
        true_p = results['true_probabilities']
        trials = np.arange(1, len(p_history) + 1)
        
        num_arms = len(true_p)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{algorithm_name} - Probability Estimate Evolution', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
        
        # plot each arm in its own subplot
        for arm in range(num_arms):
            row, col = arm // 2, arm % 2
            ax = axes[row, col]
            
            # plot estimated probability for this arm
            ax.plot(trials, p_history[:, arm], linewidth=2, 
                   label=f'Estimated', color=colors[arm % len(colors)])
            # plot true probability as horizontal line
            ax.axhline(y=true_p[arm], color='red', 
                      linestyle='--', linewidth=2, label=f'True p={true_p[arm]:.1f}')
            
            ax.set_xlabel('Trial', fontsize=10)
            ax.set_ylabel('Probability Estimate', fontsize=10)
            ax.set_title(f'Arm {arm} - Probability Learning', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, max(1.05, true_p[arm] + 0.2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def _plot_thompson_distributions(self, results, algorithm_name, save_path):
        # plot beta distributions at trials 100, 500, 1000, 2000
        alpha_history = results['alpha_history']
        beta_history = results['beta_history']
        true_p = results['true_probabilities']
        
        trials_to_plot = [100, 500, 1000, 2000]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{algorithm_name} - Beta Distribution Evolution', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
        x = np.linspace(0, 1, 1000)
        
        for idx, trial in enumerate(trials_to_plot):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            if trial <= len(alpha_history):
                alpha = alpha_history[trial - 1]
                beta = beta_history[trial - 1]
                
                # plot beta distribution for each arm
                for arm in range(len(true_p)):
                    y = beta_dist.pdf(x, alpha[arm], beta[arm])
                    ax.plot(x, y, linewidth=2, label=f'Arm {arm}', color=colors[arm % len(colors)])
                    # mark true probability with vertical line
                    ax.axvline(x=true_p[arm], color=colors[arm % len(colors)], 
                              linestyle='--', linewidth=1.5, alpha=0.5)
                
                ax.set_xlabel('Probability', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.set_title(f'After {trial} Trials', fontsize=12)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Trial {trial}\nnot reached', 
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot2(self, eg_results, ts_results, save_path=None):
        eg_rewards = eg_results['rewards']
        ts_rewards = ts_results['rewards']
        
        eg_cumulative = np.cumsum(eg_rewards)
        ts_cumulative = np.cumsum(ts_rewards)
        
        trials = np.arange(1, len(eg_rewards) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Algorithm Comparison: Epsilon-Greedy vs Thompson Sampling', 
                     fontsize=16, fontweight='bold')
        
        plot_configs = [
            (0, eg_cumulative, ts_cumulative, 'Cumulative Reward', 'Cumulative Rewards', False),
            (1, eg_cumulative, ts_cumulative, 'Cumulative Reward', 'Cumulative Rewards', True),
        ]
        
        for col, eg_data, ts_data, ylabel, title, use_log in plot_configs:
            axes[col].plot(trials, eg_data, linewidth=2, label='Epsilon-Greedy', color='blue')
            axes[col].plot(trials, ts_data, linewidth=2, label='Thompson Sampling', color='green')
            axes[col].set_xlabel('Trial', fontsize=12)
            axes[col].set_ylabel(ylabel, fontsize=12)
            scale = 'Log Scale' if use_log else 'Linear Scale'
            axes[col].set_title(f'{title} ({scale})', fontsize=14)
            if use_log:
                axes[col].set_xscale('log')
            axes[col].legend(fontsize=11)
            axes[col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()
    
    def comparison(self, eg_results, ts_results):
        print(f"\n{'=' * 60}")
        print(f"algorithm comparison summary")
        print(f"{'=' * 60}")
        
        print(f"\n{'Metric':<30} {'E-Greedy':>12} {'Thompson':>12} {'Winner':>10}")
        print(f"{'-' * 60}")
        
        metrics = [
            ('Cumulative Reward', 'cumulative_reward', '.2f', False),
            ('Cumulative Regret', 'cumulative_regret', '.2f', True),
            ('Average Reward', 'avg_reward', '.4f', False),
        ]
        
        for metric_name, key, fmt, lower_is_better in metrics:
            eg_val = eg_results[key]
            ts_val = ts_results[key]
            
            if lower_is_better:
                winner = 'E-Greedy' if eg_val < ts_val else 'Thompson'
            else:
                winner = 'E-Greedy' if eg_val > ts_val else 'Thompson'
            
            print(f"{metric_name:<30} {eg_val:>12{fmt}} {ts_val:>12{fmt}} {winner:>10}")
        
        print(f"{'='*60}\n")
