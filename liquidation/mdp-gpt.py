import numpy as np
from mdptoolbox.mdp import ValueIteration
import matplotlib.pyplot as plt

from amm import AMM

class MDP:
    def __init__(self, amm, T, D, mu, sigma, phi, g, INV_SPACE=5, SWAP_SPACE=3, Z_SPACE=10, dt=1):
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

        # Discretize state and action spaces
        self.Delta_space = np.linspace(0, self.D, self.INV_SPACE)  # inventory space
        self.z_space = np.linspace(-self.amm.gamma, self.amm.gamma, self.Z_SPACE)  # mispricing space
        self.actions = np.linspace(0, self.D / 10, self.SWAP_SPACE)  # swap sizes space

        # Precompute number of states, etc.
        self.n_states = self.Z_SPACE * self.INV_SPACE
        self.n_actions = len(self.actions)

        # For storing results of Value Iteration
        self.V = None
        self.policy = None

    def state_to_idx(self, z_idx, i_idx):
        """Flatten (z_idx, i_idx) -> single state index."""
        return z_idx * self.INV_SPACE + i_idx

    def idx_to_state(self, s):
        """Unflatten single index -> (z_idx, i_idx)."""
        z_idx = s // self.INV_SPACE
        i_idx = s % self.INV_SPACE
        return z_idx, i_idx

    def discretize_z(self, z_cont):
        """
        Find the closest index in z_space to the continuous z_cont.
        (Or do a proper bin search if you prefer.)
        """
        # Basic approach: pick nearest bin
        idx = np.argmin(np.abs(self.z_space - z_cont))
        return idx

    def discretize_inventory(self, I_cont):
        """
        Find the closest index in Delta_space to I_cont.
        """
        idx = np.argmin(np.abs(self.Delta_space - I_cont))
        return idx

    def transition_reward(self, z, I, Delta):
        """
        Given continuous z, I, and Delta, return (z_next, I_next, reward).

        NOTE: If you keep random draws here, you're effectively building
              a single-sample realization for your MDP. This is not standard
              for a tabular MDP. It's shown here as an example.
        """
        # We only allow the agent to trade up to their inventory
        Delta = min(Delta, I)

        # Next inventory after trading
        I_next = I - Delta

        # External price from mispricing
        external_price = self.amm.instantaneous_x_price() * np.exp(z)

        # We'll "simulate" the pool for this state transition:
        # 1) Save the pool reserves
        old_x = self.amm.x_reserves
        old_y = self.amm.y_reserves

        # 2) Do the actions
        self.amm.arb(external_price)
        trade_result, average_price = self.amm.trade(Delta, 0)

        # 3) Compute the reward
        #    Reward = # of x * ( (avg price) - external_price*(1-fee) ) - phi * I - g*(Delta>0)
        #    (You may want a more stable formula depending on your convention)
        reward = (trade_result.x_out *
                  (average_price - external_price * (1 - self.amm.gamma))
                  ) - self.phi * I - self.g * (Delta > 0)

        # 4) Revert the AMM to old reserves so the next transition is unaffected
        self.amm.x_reserves = old_x
        self.amm.y_reserves = old_y

        # 5) Compute a next z (deterministically or stochastically):
        #    For demonstration, let's do a simple *deterministic* shift:
        #    z_next = (z + mu*dt) clipped into [-gamma, gamma], etc.
        #    or if you keep the random draw, you get different transitions each time...
        z_next_cont = z + self.mu * self.dt  # no random for demonstration
        # Clip or re-center as needed
        z_next_cont = np.clip(z_next_cont, -self.amm.gamma, self.amm.gamma)

        return z_next_cont, I_next, reward

    def build_mdp_matrices(self):
        """
        Build transition and reward matrices P, R for a tabular MDP with states = (z_idx, i_idx).
        P is shape (n_actions, n_states, n_states)
        R is shape (n_states, n_actions).
        """
        nS = self.n_states
        nA = self.n_actions
        P = np.zeros((nA, nS, nS))
        R = np.zeros((nS, nA))

        # Loop over every discrete state
        for s in range(nS):
            # Convert to (z_idx, i_idx)
            z_idx, i_idx = self.idx_to_state(s)
            z_val = self.z_space[z_idx]
            I_val = self.Delta_space[i_idx]

            # For each action
            for a_idx, Delta_val in enumerate(self.actions):
                # Get next (z_next, I_next, reward)
                z_next, I_next, reward = self.transition_reward(z_val, I_val, Delta_val)

                # Discretize next z, I
                z_next_idx = self.discretize_z(z_next)
                i_next_idx = self.discretize_inventory(I_next)
                s_next = self.state_to_idx(z_next_idx, i_next_idx)

                # Fill P and R
                P[a_idx, s, s_next] = 1.0
                R[s, a_idx] = reward

        # Normalize transitions if needed (here they are deterministic: sum=1).
        # But if you add randomness, you'd build each row with probabilities < 1,
        # and then we might force row sums = 1. 
        return P, R

    def solve_mdp(self, discount=0.99, epsilon=0.01, max_iter=1000):
        """
        Solve MDP via ValueIteration, store self.V and self.policy
        """
        P, R = self.build_mdp_matrices()
        vi = ValueIteration(P, R, discount, epsilon=epsilon, max_iter=max_iter)
        vi.run()
        self.V = np.array(vi.V)
        self.policy = np.array(vi.policy)
        return self.policy, self.V

    def plot_results(self):
        """
        Plot value function and policy as 2D heatmaps:
            x-axis = inventory index
            y-axis = z index
        """
        if self.V is None or self.policy is None:
            raise ValueError("Must run solve_mdp before plotting.")

        # Reshape V and policy from shape (nS,) -> (Z_SPACE, INV_SPACE)
        V_2d = self.V.reshape(self.Z_SPACE, self.INV_SPACE)
        policy_2d = self.policy.reshape(self.Z_SPACE, self.INV_SPACE)

        # Plot the value function
        fig, axs = plt.subplots(1, 2, figsize=(12,5))

        im1 = axs[0].imshow(V_2d, origin='lower', aspect='auto', 
                            extent=(0, self.INV_SPACE, 0, self.Z_SPACE))
        axs[0].set_title("Value Function (V)")
        axs[0].set_xlabel("Inventory index (i_idx)")
        axs[0].set_ylabel("Mispricing index (z_idx)")
        fig.colorbar(im1, ax=axs[0])

        # Plot the policy (action indices) 
        # If you'd rather show the actual Delta (swap size) instead of action index:
        # you can map policy_2d -> self.actions[policy_2d.astype(int)]
        im2 = axs[1].imshow(policy_2d, origin='lower', aspect='auto',
                            extent=(0, self.INV_SPACE, 0, self.Z_SPACE))
        axs[1].set_title("Policy (action index)")
        axs[1].set_xlabel("Inventory index (i_idx)")
        axs[1].set_ylabel("Mispricing index (z_idx)")
        fig.colorbar(im2, ax=axs[1])

        plt.tight_layout()
        plt.show()


# Example usage
def run_example():
    # Suppose amm is a constant-product market maker with some reserves
    amm = AMM(x_reserves=1e5, y_reserves=5e8, gamma=30)  # gamma=30 -> 30 bps fee
    mdp = MDP(amm, T=1000, D=100, mu=0, sigma=20, phi=0.0, g=2,
              INV_SPACE=5, SWAP_SPACE=3, Z_SPACE=10, dt=1)

    policy, value = mdp.solve_mdp(discount=0.99, epsilon=0.01)
    mdp.plot_results()
    return mdp

if __name__ == "__main__":
    mdp = run_example()
