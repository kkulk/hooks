from amm_mdp import AMM_MDP
import numpy as np
import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration
from latexify import latexify

# MDP params
T = 1000
Delta_bar = 1000
R0 = 1e4
R1 = 1e4 * 5000
gamma = 30
g = 2
sigma = 8
mu = 0
xi = 0.1

mdp = AMM_MDP(T, Delta_bar, R0, R1, gamma, sigma, mu, xi, g, INV_SPACE=40, Z_SPACE=40, SWAP_SPACE=40)

#  transition + reward matrices
P, R = mdp.build_mdp_matrices()

# Solve w value iteration
vi = ValueIteration(P, R, .99, epsilon=1e-6)
vi.run()

print("REWARD PATHS")
print(mdp.reward_path(vi.policy))
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
cb = plt.colorbar(label='Optimal value')
plt.xlabel('Bps mispricing, $z$', fontsize=16)
plt.ylabel('Inventory, $I$', fontsize=16)
plt.title('Value function', fontsize=18)
plt.grid(True)
plt.savefig('../figures/value-function-xi-pt3.png', bbox_inches='tight', dpi=500)


# Plot Optimal Policy
latexify(fig_width=6, fig_height=3.5)
plt.figure()
plt.imshow(optimal_policy, origin='lower', aspect='auto',
           extent=[min(mdp.z_space), max(mdp.z_space), min(mdp.Delta_space), max(mdp.Delta_space)],
           cmap='plasma')
cb = plt.colorbar(label='Optimal trade')
plt.xlabel('Bps mispricing, $z$', fontsize=16)
plt.ylabel('Inventory, $I$', fontsize=16)
plt.title('Optimal policy', fontsize=18)
plt.grid(True)
plt.savefig('../figures/optimal-policy-xi-pt3.png', bbox_inches='tight', dpi=500)
