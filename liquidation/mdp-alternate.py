from amm import AMM
import numpy as np
from mdptoolbox.mdp import ValueIteration, PolicyIteration
import matplotlib.pyplot as plt

class MDP:
    def __init__(self, amm, T, D, mu, sigma, phi, g, INV_SPACE=30, SWAP_SPACE=30, Z_SPACE= 40, dt=12):
        self.amm = amm
        self.T = T
        self.D = D
        self.mu = mu
        self.sigma = sigma
        self.phi = phi
        self.g = g
        self.INV_SPACE = INV_SPACE
        self.SWAP_SPACE = SWAP_SPACE
        self.Z_SPACE = Z_SPACE
        self.dt = dt
        self.external_price = self.amm.instantaneous_x_price()

        # discretize state and action spaces
        self.Delta_space = np.linspace(0, self.D, self.INV_SPACE)  # inventory space
        self.actions = np.linspace(0, self.D / 10, self.SWAP_SPACE)  # swap sizes space
        self.z_space = np.linspace(-self.amm.gamma, self.amm.gamma, self.Z_SPACE)
        
        self.n_states = self.Z_SPACE * self.INV_SPACE  # Total number of states
        self.n_actions = len(self.actions)  # Total number of actions

    def state_index(self, I):
        """Map continuous inventory to discrete state index"""
        idx = np.searchsorted(self.Delta_space, I) - 1
        return np.clip(idx, 0, len(self.Delta_space) - 1)

    def transition_reward(self, z, I, Delta):
        # I is the inventory level
        # z is the mispricing level
        # Delta is the swap size
        Delta = min(Delta, I)
        I_next = I - Delta
        #print(Delta)
        #z = np.log(self.external_price/self.amm.instantaneous_x_price())
        #external_price = self.amm.instantaneous_x_price() *np.exp(z)
        self.external_price = self.amm.instantaneous_x_price() *np.exp(z)
        print("External price:")
        print(self.external_price)

        #z_clipped = np.clip(z, -self.amm.gamma, self.amm.gamma)
          #arbitrage the pool
        self.amm.arb(self.external_price)
        print("Mispricing:")
        print(z)
        post_arb_price = self.amm.instantaneous_x_price()
        print("Post arb price:")
        print(post_arb_price)
        trade_result, average_price = self.amm.trade(Delta, 0)
        post_trade_price = self.amm.instantaneous_x_price()
        #z_intermediate = np.log(self.external_price/post_trade_price)
        print("Post trade price:")
        print(post_trade_price)
        reward = trade_result.x_out *(average_price - self.external_price* (1-self.amm.gamma)) - self.phi * I - self.g*(Delta>0) 
        print("Reward:")
        print(reward)
        #z_next = z_intermediate + self.mu*self.dt + self.sigma/10000*np.random.normal()*np.sqrt(self.dt)
        last_price = self.external_price
        #self.external_price = last_price + last_price*self.mu*self.dt + last_price*self.sigma/10000*np.random.normal()*np.sqrt(self.dt)
        z_intermediate = np.log(self.external_price/self.amm.instantaneous_x_price())
        z_next = z_intermediate + self.mu*self.dt + self.sigma*np.random.normal()*np.sqrt(self.dt)
        #z_next_bps = z_next*10000

        return (z_next, I_next, reward)

    def state_to_idx(self, z_idx, i_idx):
        """Map (z_idx, i_idx) to single state index."""
        return z_idx * self.INV_SPACE + i_idx

    def idx_to_state(self, s):
        """Map single state index to (z_idx, i_idx)."""
        z_idx = s // self.INV_SPACE
        i_idx = s % self.INV_SPACE
        return z_idx, i_idx

    def discretize_z(self, z):
        """Map continuous z to nearest z index."""
        idx = np.argmin(np.abs(self.z_space - z))
        return idx

    def discretize_inventory(self, I):
        """Map continuous inventory I to nearest inventory index."""
        idx = np.argmin(np.abs(self.Delta_space - I))
        return idx

    def build_mdp_matrices(self):
        """Build transition probability matrix P and reward matrix R."""
        n_states = self.n_states
        n_actions = self.n_actions
        P = np.zeros((n_actions, n_states, n_states))
        R = np.zeros((n_states, n_actions))

        for s in range(n_states):
            # Get (z_idx, i_idx) from flattened state index
            z_idx, i_idx = self.idx_to_state(s)
            z = self.z_space[z_idx]
            I = self.Delta_space[i_idx]

            for a_idx, Delta in enumerate(self.actions):
                # Use transition_reward to get the next state and reward
                z_next, I_next, reward = self.transition_reward(z, I, Delta)

                # Discretize the next state
                z_next_idx = self.discretize_z(z_next)
                i_next_idx = self.discretize_inventory(I_next)

                # Flatten the next state index
                s_next = self.state_to_idx(z_next_idx, i_next_idx)

                # Update the transition matrix and reward matrix
                P[a_idx, s, s_next] = 1.0  # Deterministic transitions
                R[s, a_idx] = reward

        # Normalize transition probabilities (optional here, since they sum to 1)
        return P, R

    def solve_value_iteration(self, discount=0.99, epsilon=0.01):
        """Solve the MDP using Value Iteration."""
        P, R = self.build_mdp_matrices()
        vi = ValueIteration(P, R, discount=discount, epsilon=epsilon)
        vi.run()
        self.V = np.array(vi.V)  # Ensure V is a NumPy array
        self.policy = np.array(vi.policy)  # Ensure policy is a NumPy array
        return self.policy, self.V

    def solve_policy_iteration(self, discount=0.99):
        """Solve the MDP using Policy Iteration."""
        P, R = self.build_mdp_matrices()
        pi = PolicyIteration(P, R, discount=discount)
        pi.run()
        self.V = np.array(pi.V)  # Ensure V is a NumPy array
        self.policy = np.array(pi.policy)  # Ensure policy is a NumPy array
        return self.policy, self.V


    def plot_results(self, method="Value Iteration"):
        """Plot the value function and policy as heatmaps."""
        if not hasattr(self, "V") or not hasattr(self, "policy"):
            raise ValueError("Run value or policy iteration before plotting!")

        # Reshape value function and policy to (INV_SPACE, Z_SPACE)
        V_2d = self.V.reshape(self.Z_SPACE, self.INV_SPACE).T  # Transpose to match axes
        policy_2d = np.array(self.policy).reshape(self.Z_SPACE, self.INV_SPACE).T  # Transpose to match axes

        # Plot Value Function Heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(V_2d, origin='lower', aspect='auto',
                extent=[-self.amm.gamma * 10000, self.amm.gamma * 10000, 0, self.D],
                cmap='viridis')
        plt.colorbar(label='Value Function')
        plt.xlabel('Bps mispricing (z)')
        plt.ylabel('Inventory (Δ)')
        plt.title(f'Value Function Heatmap ({method})')
        plt.show()

        # Plot Policy Heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(policy_2d, origin='lower', aspect='auto',
                extent=[-self.amm.gamma * 10000, self.amm.gamma * 10000, 0, self.D],
                cmap='coolwarm')
        plt.colorbar(label='Optimal Action (Index)')
        plt.xlabel('Bps mispricing (z)')
        plt.ylabel('Inventory (Δ)')
        plt.title(f'Policy Heatmap ({method})')
        plt.show()



        # fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # im1 = axs[0].imshow(V_2d, origin="lower", aspect="auto",
        #                     extent=[0, self.INV_SPACE, 0, self.Z_SPACE])
        # axs[0].set_title(f"Value Function ({method})")
        # axs[0].set_xlabel("Inventory Index")
        # axs[0].set_ylabel("Mispricing Index")
        # fig.colorbar(im1, ax=axs[0])

        # im2 = axs[1].imshow(policy_2d, origin="lower", aspect="auto",
        #                     extent=[0, self.INV_SPACE, 0, self.Z_SPACE])
        # axs[1].set_title(f"Policy ({method})")
        # axs[1].set_xlabel("Inventory Index")
        # axs[1].set_ylabel("Mispricing Index")
        # fig.colorbar(im2, ax=axs[1])

        # plt.tight_layout()
        # plt.show()


# Example Usage
def run_mdp_example():
    # Create AMM
    amm = AMM(1e5, 1e5 * 5000, 30)  # x_reserves, y_reserves, gamma (bps)
    # Initialize MDP
    mdp = MDP(amm, T=1000, D=100, mu=0, sigma=20, phi=0.01, g=2, dt=1)

    # Solve using Value Iteration
    policy_vi, value_vi = mdp.solve_value_iteration(discount=0.99, epsilon=0.01)
    mdp.plot_results(method="Value Iteration")

    # Solve using Policy Iteration
    policy_pi, value_pi = mdp.solve_policy_iteration(discount=0.99)
    mdp.plot_results(method="Policy Iteration")


if __name__ == "__main__":
    run_mdp_example()
