from amm import AMM
import numpy as np
from mdptoolbox.mdp import ValueIteration
import matplotlib.pyplot as plt


class MDP:
    def __init__(self, amm, T, D, mu, sigma, phi, g, INV_SPACE=75, Z_SPACE=40, SWAP_SPACE=50, dt = 12):
        self.amm = amm
        self.T =  T
        self.D = D
        self.mu = mu
        self.sigma = sigma
        self.phi = phi
        self.g = g
        self.INV_SPACE = INV_SPACE
        self.Z_SPACE = Z_SPACE
        self.SWAP_SPACE = SWAP_SPACE
        self.dt = dt

        # discretize state and action spaces
        self.Delta_space = np.linspace(0, self.D, self.INV_SPACE)  # inventory space
        self.z_space = np.linspace(-1.25 * self.amm.gamma, 1.25 * self.amm. gamma, self.Z_SPACE)  # mispricing space
        self.actions = np.linspace(0, self.D / 10, self.SWAP_SPACE)  # swap sizes spaceE


    def transition_reward(self, I, z, Delta):
        # I is the inventory level
        # z is the mispricing level
        # Delta is the swap size
        Delta = min(Delta, I)
        I_next = I - Delta
        print(Delta)
        external_price = self.amm.instantaneous_x_price() *np.exp(z)
        #print(external_price)

        z_intermediate = np.clip(z, -self.amm.gamma, self.amm.gamma)
              #arbitrage the pool
        self.amm.arb(z_intermediate)
        post_arb_price = self.amm.instantaneous_x_price()
        trade_result, average_price = self.amm.trade(Delta, 0)

        reward = trade_result.x_out *(average_price - external_price* (1-self.amm.gamma)) - self.phi * I - self.g*(Delta>0)
        z_next = z_intermediate + self.mu*self.dt + self.sigma/10000*np.random.normal()*np.sqrt(self.dt)
        z_next_bps = z_next*10000

        return (I_next, z_next_bps, reward)
    
    def state_index(self, Delta, z):
        """
        Map a continuous state (Delta, z) to a discrete index.
        """
        Delta_idx = np.searchsorted(self.Delta_space, Delta) - 1
        z_idx = np.searchsorted(self.z_space, z) - 1
        return Delta_idx * len(self.z_space) + z_idx

    def build_mdp_matrices(self):
        n_states = len(self.Delta_space) * len(self.z_space)
        n_actions = len(self.actions)
        
        P = np.zeros((n_actions, n_states, n_states))
        R = np.zeros((n_states, n_actions))

        for i, Delta in enumerate(self.Delta_space):
            for j, z in enumerate(self.z_space):
                state_idx = i * len(self.z_space) + j
                for k, a in enumerate(self.actions):
                    # Simulate transition
                    Delta_next, z_next, reward = self.transition_reward(Delta, z, a)
                    reward = np.clip(reward, -1e5, 1e5)


                    # Find next state index
                    next_state_idx = self.state_index(Delta_next, z_next)

                    # Update matrices
                    P[k, state_idx, next_state_idx] += 1
                    R[state_idx, k] = reward

        # normalize probabilities
        for k in range(n_actions):
            row_sums = P[k].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            P[k] /= row_sums

        return P, R

    def solve_mdp(self, discount_factor=0.99):
        """
        Solve the MDP using value iteration.

        discount_factor: Discount factor for future rewards
        Returns: Optimal policy
        """
        P, R = self.build_mdp_matrices()
        
        
        vi = ValueIteration(P, R, discount_factor)
        vi.run()
        self.value_function = vi.V
        self.policy = np.array(vi.policy)

        return self.policy
    
    def simulate(self, policy, num_simulations=100):
        """
        Simulate the MDP policy over multiple runs and calculate the mean inventory path.

        Args:
            policy (array): Optimal policy mapping states to actions.
            num_simulations (int): Number of independent simulations to run.

        Returns:
            np.array: Mean inventory path over all simulations.
        """
        inventory_paths = []

        for _ in range(num_simulations):
            Delta = self.D
            z = 0 
            inventory = [Delta]

            for t in range(self.T):
                # current state index
                Delta_idx = np.searchsorted(self.Delta_space, Delta) - 1
                z_idx = np.searchsorted(self.z_space, z) - 1
                state_index = Delta_idx * len(self.z_space) + z_idx

                # get action from policy and constrain it
                action_index = policy[state_index]
                action = self.actions[action_index]
                action = min(action, Delta) 

                # transition
                Delta, z, _, _, _ = self.transition(Delta, z, action)
                inventory.append(Delta)

            inventory_paths.append(inventory)
        
        return np.mean(inventory_paths, axis=0)
    