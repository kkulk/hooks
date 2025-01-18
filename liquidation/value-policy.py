from optimal_liquidation import AMM_MDP
import numpy as np
import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration
from latexify import latexify

# MDP params
T = 1000
Delta_bar = 100
R0 = 1e5
R1 = 1e5 * 5000
gamma = 30
g = 2
sigma = 20
mu = 0
phi = 0.0

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
# latexify(fig_width=6, fig_height=3.5)
# plt.imshow(value_function, origin='lower', aspect='auto',
#            extent=[min(mdp.z_space), max(mdp.z_space), min(mdp.Delta_space), max(mdp.Delta_space)],
#            cmap='viridis')
# plt.colorbar(label='Value Function')
# plt.xlabel('Bps mispricing (z)')
# plt.ylabel('Inventory (Δ)')
# plt.title('Value Function Heatmap')
# plt.show()

# Plot Optimal Policy
# latexify(fig_width=8, fig_height=5)
# plt.imshow(optimal_policy, origin='lower', aspect='auto',
#            extent=[min(mdp.z_space), max(mdp.z_space), min(mdp.Delta_space), max(mdp.Delta_space)],
#            cmap='plasma')
# plt.colorbar(label='Optimal Action')
# plt.xlabel('Bps mispricing (z)')
# plt.ylabel('Inventory (Δ)')
# plt.title('Optimal Policy Heatmap')
# plt.show()


latexify(fig_width=6, fig_height=3.5)
plt.figure()
plt.imshow(value_function, origin='lower', aspect='auto',
           extent=[min(mdp.z_space), max(mdp.z_space), min(mdp.Delta_space), max(mdp.Delta_space)],
           cmap='plasma')
cb = plt.colorbar(label='Optimal Action')
plt.xlabel('Bps mispricing ($z$)', fontsize=16)
plt.ylabel('Inventory ($\Delta$)', fontsize=16)
plt.title('Value Function Heatmap', fontsize=18)
plt.grid(True)
plt.legend(fontsize=14)
plt.savefig('../figures/value-function.png', bbox_inches='tight', dpi=500)


# Plot Optimal Policy
latexify(fig_width=6, fig_height=3.5)
plt.figure()
plt.imshow(optimal_policy, origin='lower', aspect='auto',
           extent=[min(mdp.z_space), max(mdp.z_space), min(mdp.Delta_space), max(mdp.Delta_space)],
           cmap='plasma')
cb = plt.colorbar(label='Optimal Action')
plt.xlabel('bps mispricing ($z$)', fontsize=16)
plt.ylabel('Inventory ($\Delta$)', fontsize=16)
plt.title('Optimal policy', fontsize=18)
plt.grid(True)
plt.legend(fontsize=14)
plt.savefig('../figures/optimal-policy.png', bbox_inches='tight', dpi=500)