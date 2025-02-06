from amm_mdp import AMM_MDP
import numpy as np
import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration


# MDP params
T = 1000
Delta_bar = 100
R0 = 1e5
R1 = 1e5 * 5000
gamma = 30
g = 2
sigma = 20
mu = 0
phi = 1


mdp = AMM_MDP(T, Delta_bar, R0, R1, gamma, sigma, mu, phi, g)

#  transition + reward matrices
P, R = mdp.build_mdp_matrices()

# Solve w value iteration
vi = ValueIteration(P, R, .99, epsilon=1e-6)
vi.run()

# Reshape the results
value_function = np.array(vi.V).reshape((len(mdp.Delta_space), len(mdp.z_space)))
optimal_policy = np.array(vi.policy).reshape((len(mdp.Delta_space), len(mdp.z_space)))

# Plot Value Function
plt.figure(figsize=(10, 8))
plt.imshow(value_function, origin='lower', aspect='auto',
           extent=[min(mdp.z_space), max(mdp.z_space), min(mdp.Delta_space), max(mdp.Delta_space)],
           cmap='viridis')
plt.colorbar(label='Value Function')
plt.xlabel('Bps mispricing (z)')
plt.ylabel('Inventory (Δ)')
plt.title('Value Function Heatmap')
plt.show()

# Plot Optimal Policy
plt.figure(figsize=(10, 8))
plt.imshow(optimal_policy, origin='lower', aspect='auto',
           extent=[min(mdp.z_space), max(mdp.z_space), min(mdp.Delta_space), max(mdp.Delta_space)],
           cmap='plasma')
plt.colorbar(label='Optimal Action')
plt.xlabel('Bps mispricing (z)')
plt.ylabel('Inventory (Δ)')
plt.title('Optimal Policy Heatmap')
plt.show()
