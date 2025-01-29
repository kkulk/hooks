import numpy as np
import matplotlib.pyplot as plt
import time
from mdptoolbox.mdp import ValueIteration
from amm_mdp import AMM_MDP

def run_volatility_simulation(sigma, mdp_params):
    """
    Run the MDP simulation for a given volatility (sigma).
    Args:
        sigma: Volatility value
        mdp_params: MDP parameter dictionary
    Returns:
        tuple: (sigma, mean value function)
    """
    mdp_params['sigma'] = sigma  # inject sigma into mdp params
    mdp = AMM_MDP(**mdp_params) 
    P, R = mdp.build_mdp_matrices()
    vi = ValueIteration(P, R, 0.95)
    vi.run()

    simulation_list = mdp.benchmark_against_twamm(vi.policy)
    mean_excess_return = np.mean(simulation_list)
    reward = mdp.reward_path(vi.policy)
    print(f"Liquidation reward: {reward:.6f}")
    return sigma, mean_excess_return


def sequential_volatility_analysis(volatilities, mdp_params):
    """
    Perform MDP simulations for a range of volatilities sequentially.
    Args:
        volatilities: List or array of volatilities to test
        mdp_params: Dictionary of MDP parameters
    Returns:
        List of tuples: [(sigma1, avg_value1), (sigma2, avg_value2), ...]
    """
    results = []
    for sigma in volatilities:
        print(f"Running simulation for sigma={sigma:.6f}...")
        result = run_volatility_simulation(sigma, mdp_params)
        results.append(result)
        print(f"Completed simulation for sigma={sigma:.6f}")
    return results


if __name__ == "__main__":
    # Parameters
    T = 1000
    Delta_bar = 1000
    R0 = 1e5
    R1 = 1e5 * 5000 
    gamma = 30
    g = 2
    mu = 0
    xi = 0
    INV_SPACE = 50
    Z_SPACE = 25
    SWAP_SPACE = 25

    mdp_params = {
        'T': T, 'Delta_bar': Delta_bar, 'R0': R0, 'R1': R1, 'gamma': gamma, 'g': g,
        'mu': mu, 'xi': xi, 'INV_SPACE': INV_SPACE, 'Z_SPACE': Z_SPACE, 'SWAP_SPACE': SWAP_SPACE
    }

    volatilities = np.logspace(0, 1.8, 5)  # Logarithmically spaced volatilities (10^0 to 10^1.5), remember we're in bps space

    start_time = time.time()
    results = sequential_volatility_analysis(volatilities, mdp_params)
    end_time = time.time()

    # print(f"Total computation time: {end_time - start_time:.2f} seconds")

    sigmas, avg_values = zip(*results)

    plt.figure(figsize=(12, 8))
    plt.plot(sigmas, avg_values, marker='o')
    plt.xscale('log')
    plt.xlabel('Volatility (σ)')
    plt.ylabel('Average Value')
    plt.title('Average Value vs Volatility')
    plt.grid(True)
    plt.show()

    for sigma, avg_value in results:
        print(f"Volatility: {sigma:.6f}, Average Value: {avg_value:.6f}")
        continue
