"""
Main execution script for Multi-Armed Bandit experiments.

This script runs experiments comparing Epsilon-Greedy and Thompson Sampling
algorithms on a multi-armed bandit problem.
"""

import numpy as np
from loguru import logger
from EpsilonGreedy import EpsilonGreedy
from ThompsonSampling import ThompsonSampling
from Performance import Performance


def main():
    logger.info("Starting Multi-Armed Bandit Experiments")
    
    # bandit arm probabilities
    bandit_probabilities = [0.1, 0.2, 0.3, 0.4] 
    
    logger.info(f"Bandit arm probabilities: {bandit_probabilities}")
    logger.info(f"Optimal arm: Arm {np.argmax(bandit_probabilities)} "
                f"with probability {np.max(bandit_probabilities)}")
    
    # for reproducibility
    np.random.seed(42)
    
    # params
    num_trials = 20000
    epsilon = 1.0  # initial epsilon 
    precision = 1.0  # precision param
    
    print("\n" + "=" * 70)
    print("MULTI-ARMED BANDIT EXPERIMENT")
    print("=" * 70)
    print(f"Number of arms: {len(bandit_probabilities)}")
    print(f"True probabilities: {bandit_probabilities}")
    print(f"Number of trials: {num_trials}")
    print(f"Epsilon-Greedy initial epsilon: {epsilon} (decays by 1/log(t+1))")
    print(f"Thompson Sampling precision: {precision}")
    print("=" * 70 + "\n")
    
    # ========== EPSILON-GREEDY EXPERIMENT ==========
    logger.info("Running Epsilon-Greedy experiment")
    print("\n" + ">" * 70)
    print("RUNNING EPSILON-GREEDY ALGORITHM")
    print(">" * 70)
    
    eg_bandit = EpsilonGreedy(p=bandit_probabilities, epsilon=epsilon)
    logger.debug(f"Initialized: {eg_bandit}")
    
    eg_bandit.experiment(num_trials=num_trials)
    logger.info("Epsilon-Greedy experiment completed")
    
    eg_results = eg_bandit.report(filename='data/epsilon_greedy_results.csv')
    
    # ========== THOMPSON SAMPLING EXPERIMENT ==========
    logger.info("Running Thompson Sampling experiment")
    print("\n" + ">" * 70)
    print("RUNNING THOMPSON SAMPLING ALGORITHM")
    print(">" * 70)
    
    ts_bandit = ThompsonSampling(p=bandit_probabilities, precision=precision)
    logger.debug(f"Initialized: {ts_bandit}")
    
    ts_bandit.experiment(num_trials=num_trials)
    logger.info("Thompson Sampling experiment completed")
    
    ts_results = ts_bandit.report(filename='data/thompson_sampling_results.csv')
    
    # ========== PERFORMANCE VISUALIZATION ==========
    logger.info("Generating performance visualizations")
    print("\n" + ">" * 70)
    print("GENERATING VISUALIZATIONS")
    print(">" * 70 + "\n")
    
    performance = Performance()
    
    print("Generating Epsilon-Greedy learning process plot...")
    performance.plot1(eg_results, "Epsilon-Greedy", 
                     save_path='graphs/epsilon_greedy_learning.png')
    
    print("Generating Thompson Sampling learning process plot...")
    performance.plot1(ts_results, "Thompson Sampling", 
                     save_path='graphs/thompson_sampling_learning.png')
    
    print("Generating algorithm comparison plot...")
    performance.plot2(eg_results, ts_results, 
                     save_path='graphs/algorithm_comparison.png')
    
    performance.comparison(eg_results, ts_results)
    
    # ========== FINAL SUMMARY ==========
    logger.success("All experiments completed successfully!")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  Data:")
    print("    - data/epsilon_greedy_results.csv")
    print("    - data/thompson_sampling_results.csv")
    print("  Graphs:")
    print("    - graphs/epsilon_greedy_learning.png")
    print("    - graphs/thompson_sampling_learning.png")
    print("    - graphs/algorithm_comparison.png")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred (it worked on my machine :( ): {e}")
        raise
