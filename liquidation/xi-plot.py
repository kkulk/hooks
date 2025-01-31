from amm_mdp import AMM_MDP
from mdptoolbox.mdp import ValueIteration
import numpy as np
import matplotlib.pyplot as plt
import mdptoolbox
import time
import pickle
from latexify import latexify

def run_simulation(xi_value, mdp, T, discount_factor=0.99):
    mdp.xi = xi_value
    P, R = mdp.build_mdp_matrices()
    vi = ValueIteration(P, R, discount_factor)
    vi.run()
    policy = np.array(vi.policy)
    return mdp.simulate(policy,num_simulations=100 )

xi_values = [0, 3, 10]
T = 200
Delta_bar = 1000
R0 = 1e5
R1 = 1e5 * 5000
gamma = 30
sigma = 2
g = 2
xi = 0
mu = 0
INV_SPACE = 50
Z_SPACE = 30
SWAP_SPACE = 25

mdp = AMM_MDP(T, Delta_bar, R0, R1, gamma, sigma, mu, xi, g, INV_SPACE, Z_SPACE, SWAP_SPACE)

# Run simulations and store results
results = {}
start_time = time.time()
for xi in xi_values:
    print(f"Running simulation for xi = {xi}")
    results[xi] = run_simulation(xi, mdp, T)
end_time = time.time()
print(f"Total computation time: {end_time - start_time:.2f} seconds")

# Store results
with open('amm_mdp_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Plot results
#latexify(fig_width=8, fig_height=5)
plt.figure()
for xi, inventory in results.items():
    plt.plot(inventory, label=f'$\\xi$ = {xi}', linewidth=3)

plt.xlabel('Time, $t$', fontsize=16)
plt.ylabel('Inventory, $I$', fontsize=16)
plt.title('Inventory over time for different $\\xi$ values', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
plt.savefig("../figures/xi-plot.png", bbox_inches="tight", dpi=500)
