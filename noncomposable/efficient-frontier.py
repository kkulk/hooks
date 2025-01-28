import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from latexify import latexify

# Define the G2 function (Hook invariant)
def G2(x, alpha):
    R1 = 10  # Hook reserve
    R0 = 10  # Hook reserve
    return 2 * x - (R1 / R0) * (x ** (1 + alpha))

# Define the variance functions
def variance_linear(delta, beta):
    return beta * delta

def variance_quadratic(delta, beta):
    return beta * delta**2

def variance_superlinear(delta, beta):
    return beta * delta**1.5  # Superlinear variance

# Simulation parameters
beta = 1  # Scaling factor for variance
R = 5  # Uniswap reserve R
R_prime = 5  # Uniswap reserve R'
T_values = np.linspace(1, 10, 10)  # Range for target return T
alpha_values = [0, 0.1, 0.2, 0.3, 0.4, 0.7]  # List of α values to simulate
input_reserves = 100  # Assume input reserves are 100 for percentage calculation

# Function to run the optimization for a given α
def run_simulation(alpha):
    efficient_frontier_linear = []
    efficient_frontier_quadratic = []
    efficient_frontier_superlinear = []

    # Perform simulation for linear variance
    for T in T_values:
        delta = cp.Variable()
        y = cp.Variable()
        new_reserves = cp.vstack([R + (100 - delta), R_prime - y])
        reserves = cp.vstack([R, R_prime])
        objective = cp.Minimize(variance_linear(delta, beta))
        constraints = [
            delta >= 0, delta <= 100,
            cp.geo_mean(new_reserves) >= cp.geo_mean(reserves),
            y + G2(delta, alpha) >= T
        ]

        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(qcp=True)
            efficient_frontier_linear.append(result if result is not None else np.nan)
        except:
            efficient_frontier_linear.append(np.nan)

    # Perform simulation for quadratic variance
    for T in T_values:
        delta = cp.Variable()
        y = cp.Variable()
        new_reserves = cp.vstack([R + (100 - delta), R_prime - y])
        reserves = cp.vstack([R, R_prime])
        objective = cp.Minimize(variance_quadratic(delta, beta))
        constraints = [
            delta >= 0, delta <= 100,
            cp.geo_mean(new_reserves) >= cp.geo_mean(reserves),
            y + G2(delta, alpha) >= T
        ]

        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(qcp=True)
            efficient_frontier_quadratic.append(result if result is not None else np.nan)
        except:
            efficient_frontier_quadratic.append(np.nan)

    # Perform simulation for superlinear variance
    for T in T_values:
        delta = cp.Variable()
        y = cp.Variable()
        new_reserves = cp.vstack([R + (100 - delta), R_prime - y])
        reserves = cp.vstack([R, R_prime])
        objective = cp.Minimize(variance_superlinear(delta, beta))
        constraints = [
            delta >= 0, delta <= 100,
            cp.geo_mean(new_reserves) >= cp.geo_mean(reserves),
            y + G2(delta, alpha) >= T
        ]

        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(qcp=True)
            efficient_frontier_superlinear.append(result if result is not None else np.nan)
        except:
            efficient_frontier_superlinear.append(np.nan)

    return (
        np.array(efficient_frontier_linear),
        np.array(efficient_frontier_quadratic),
        np.array(efficient_frontier_superlinear),
    )

# Run simulations for each α and plot the results
for alpha in alpha_values:
    efficient_frontier_linear, efficient_frontier_quadratic, efficient_frontier_superlinear = run_simulation(alpha)

    # Convert T values to percentage return
    percentage_returns = (T_values / input_reserves) * 100

    # Plot results
    latexify(fig_width=6, fig_height=3.5)
    plt.figure(figsize=(10, 6))
    plt.plot(efficient_frontier_linear, percentage_returns, marker='o', color='blue', linestyle='-', label='Linear Variance', linewidth=3)
    plt.plot(efficient_frontier_quadratic, percentage_returns, marker='x', color='green', linestyle='--', label='Quadratic Variance', linewidth=3)
    plt.plot(efficient_frontier_superlinear, percentage_returns, marker='s', color='red', linestyle='-.', label='Superlinear Variance', linewidth=3)

    plt.xlabel('Optimal Variance', fontsize=16)  # Increased font size
    plt.ylabel('$\%$ Return', fontsize=16)  # Updated to reflect percentage return
    plt.title(f'Efficient frontier, $\\alpha = {alpha}$', fontsize=18)  # Increased font size
    plt.legend(fontsize=14, loc='lower right')  # Legend in bottom right
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.savefig(f'../figures/efficient_frontier_alpha_{alpha}.png', dpi=500)  # Save plot with alpha in filename
