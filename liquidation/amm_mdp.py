import numpy as np
import matplotlib.pyplot as plt
from mdptoolbox.mdp import ValueIteration

def clamp_index(idx, max_size):
    """
    Helper function that clamps idx to [0, max_size-1].
    """
    if idx < 0:
        return 0
    if idx >= max_size:
        return max_size - 1
    return idx

class AMM_MDP:
    def __init__(
        self,
        T,
        Delta_bar,
        R0,
        R1,
        gamma,
        sigma,
        mu,
        xi,
        g,
        INV_SPACE=80,
        Z_SPACE=40,
        SWAP_SPACE=40,
        dt=12
    ):
        """
        T: Total number of blocks
        Delta_bar: Maximum inventory
        R0, R1: Initial reserves of both assets in AMM
        gamma: AMM fee, in basis points (e.g. 30 = 30bps)
        sigma: Volatility (basis points) of log mispricing
        mu: Drift of log mispricing
        xi: Running inventory cost coefficient
        g: Gas cost per transaction
        INV_SPACE, Z_SPACE, SWAP_SPACE: Discretization sizes
        dt: Time step in 'block' units
        """
        self.T = T
        self.Delta_bar = Delta_bar
        self.R0 = R0
        self.R1 = R1
        self.L = self.R0 * self.R1
        self.gamma = gamma       # (basis points)
        self.sigma = sigma       # (basis points)
        self.mu = mu
        self.xi = xi
        self.g = g
        self.INV_SPACE = INV_SPACE
        self.Z_SPACE = Z_SPACE
        self.SWAP_SPACE = SWAP_SPACE
        self.dt = dt

        # Discretize state and action spaces
        self.Delta_space = np.linspace(0, Delta_bar, self.INV_SPACE) 
        # Important: scale gamma by 1/10000 so z_space is in decimal, not in bps
        self.z_space = np.linspace(
            -1.25 * gamma / 10000,
             1.25 * gamma / 10000,
            self.Z_SPACE
        )
        self.actions = np.linspace(0, Delta_bar / 10, self.SWAP_SPACE)

    def transition_adjusted_deterministic(self, Delta, z_liq, z_twamm, a):
        """
        Transition that does NOT sample from a normal distribution.
        Used when building the MDP (we do not want randomness in P, R).
        """
        # make sure we can only swap up to our remaining inventory
        Delta_next = max(Delta - a, 0)

        # clip the mispricing to fee bounds
        z_star = np.clip(z_liq, -self.gamma/10000, self.gamma/10000)
        z_star_twamm = np.clip(z_twamm, -self.gamma/10000, self.gamma/10000)

        # approx external price
        p_ext = self.R1 / self.R0

        # convert from mispricing to pool price
        p_init = p_ext * np.exp(-z_star)
        p_twamm = p_ext * np.exp(-z_star_twamm)

        # AMM math for the swap
        R0_init =  np.sqrt(self.L / p_init)
        R1_init =  self.L / R0_init
        R0_next = R0_init + a * (1 - self.gamma/10000)
        R1_next = self.L / R0_next

        amount_out = R1_init - R1_next
        p_next = R1_next / R0_next

        # new mispricing after the user trades
        z_next = np.log(p_ext / p_next)

        # TWAMM math (for reference)
        R0_init_twamm = np.sqrt(self.L / p_twamm)
        R1_init_twamm = self.L / R0_init_twamm
        R0_next_twamm = R0_init_twamm + (self.Delta_bar/self.T)*(1 - self.gamma/10000)
        R1_next_twamm = self.L / R0_next_twamm
        amount_out_twamm = R1_init_twamm - R1_next_twamm

        # execution price improvement for the user's trade
        execution_amount_improvement = amount_out - (p_ext * (1 - self.gamma/10000)) * a
        inventory_cost = -self.xi * abs(Delta)
        gas_cost = -self.g * (a > 0)

        reward = execution_amount_improvement + inventory_cost + gas_cost

        # now the user-coded "deterministic" jump:
        jump = (self.mu - (self.sigma/10000)**2 / 2) * self.dt + (self.sigma/10000)*np.random.normal()*np.sqrt(self.dt)

        z_next_adjusted = z_next + jump
        z_next_twamm = z_star_twamm + jump

        return Delta_next, z_next_adjusted, z_next_twamm, reward, amount_out, amount_out_twamm

    def transition_adjusted_stochastic(self, Delta, z_liq, z_twamm, a):
        """
        Transition that DOES sample a normal random draw.
        Used during simulation / benchmarking.
        """
        # Make sure we can only swap up to our remaining inventory
        Delta_next = max(Delta - a, 0)

        # Clip the mispricing to fee bounds
        z_star = np.clip(z_liq, -self.gamma/10000, self.gamma/10000)
        z_star_twamm = np.clip(z_twamm, -self.gamma/10000, self.gamma/10000)

        p_ext = self.R1 / self.R0
        p_init = p_ext * np.exp(-z_star)
        p_twamm = p_ext * np.exp(-z_star_twamm)

        # AMM math for the swap
        R0_init = np.sqrt(self.L / p_init)
        R1_init = self.L / R0_init
        R0_next = R0_init + a*(1 - self.gamma/10000)
        R1_next = self.L / R0_next

        amount_out = R1_init - R1_next
        p_next = R1_next / R0_next
        z_next = np.log(p_ext / p_next)

        # TWAMM math
        R0_init_twamm = np.sqrt(self.L / p_twamm)
        R1_init_twamm = self.L / R0_init_twamm
        R0_next_twamm = R0_init_twamm + (self.Delta_bar/self.T)*(1 - self.gamma/10000)
        R1_next_twamm = self.L / R0_next_twamm
        amount_out_twamm = R1_init_twamm - R1_next_twamm

        # Execution price improvement
        execution_amount_improvement = amount_out - (p_ext*(1 - self.gamma/10000))*a
        inventory_cost = -self.xi * abs(Delta)
        gas_cost = -self.g * (a > 0)
        reward = execution_amount_improvement + inventory_cost + gas_cost

        # Stochastic jump
        jump = (
            (self.mu - (self.sigma/10000)**2 / 2) * self.dt
            + (self.sigma/10000)*np.random.normal()*np.sqrt(self.dt)
        )
        z_next_adjusted = z_next + jump
        z_next_twamm = z_star_twamm + jump

        return Delta_next, z_next_adjusted, z_next_twamm, reward, amount_out, amount_out_twamm

    def build_mdp_matrices(self):
        """
        Build the transition (P) and reward (R) matrices for the MDP 
        using the deterministic approximation for the 'jump.'
        """
        n_states = len(self.Delta_space) * len(self.z_space)
        n_actions = len(self.actions)

        P = np.zeros((n_actions, n_states, n_states))
        R = np.zeros((n_states, n_actions))

        for i, Delta in enumerate(self.Delta_space):
            for j, z in enumerate(self.z_space):
                state_idx = i * len(self.z_space) + j

                for k, a in enumerate(self.actions):
                    # use the determinstic transition
                    (
                        Delta_next,
                        z_next,
                        _,
                        reward,
                        _,
                        _
                    ) = self.transition_adjusted_deterministic(Delta, z, 0, a)

                    # bound reward to avoid numerical issues 
                    reward = np.clip(reward, -1e6, 1e6)

                    next_state_idx = self.state_index(Delta_next, z_next)

                    # update P, R
                    P[k, state_idx, next_state_idx] += 1
                    R[state_idx, k] = reward

        # normalize probabilities row-wise
        for k in range(n_actions):
            row_sums = P[k].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            P[k] /= row_sums

        return P, R

    def state_index(self, Delta, z):
        Delta_idx = np.searchsorted(self.Delta_space, Delta) - 1
        z_idx     = np.searchsorted(self.z_space, z) - 1

        Delta_idx = clamp_index(Delta_idx, len(self.Delta_space))
        z_idx     = clamp_index(z_idx, len(self.z_space))

        return Delta_idx * len(self.z_space) + z_idx

    def solve_mdp(self, discount_factor=0.99):
        """
        Solve the MDP using ValueIteration on the built P, R.
        """
        P, R = self.build_mdp_matrices()
        vi = ValueIteration(P, R, discount_factor)
        vi.run()
        self.value_function = vi.V
        self.policy = np.array(vi.policy)
        return self.policy

    def benchmark_against_twamm(self, policy, num_simulations=5000):
        """
        Compare the MDP policy vs. the TWAMM baseline 
        using STOCHASTIC transitions in the simulation.
        """
        comparison_paths = []

        for _ in range(num_simulations):
            Delta = self.Delta_bar
            z = 0
            z_twamm = 0
            liquidation_usd = 0
            twamm_usd = 0

            for t in range(self.T):
                # Use the *clamped* state index from the policy
                state_index = self.state_index(Delta, z)

                action_index = policy[state_index]
                action = self.actions[action_index]
                action = min(action, Delta)

                # Stochastic transition
                Delta, z, z_twamm, reward_next, amount_out, amount_out_twamm = (
                    self.transition_adjusted_stochastic(Delta, z, z_twamm, action)
                )

                # Accumulate outputs from the policy
                liquidation_usd += amount_out - self.g * (action > 0)
                twamm_usd       += amount_out_twamm

            # TWAMM only pays gas once
            twamm_usd -= self.g

            # Discount leftover inventory by 20% at final external price
            liquidation_usd += Delta * (self.R1 / self.R0) * 0.8
            comparison_paths.append(liquidation_usd - twamm_usd)

        return comparison_paths

    def reward_path(self, policy, num_simulations=500):
        """
        Simulate the total reward path (stochastic transitions).
        """
        reward_paths = []

        for _ in range(num_simulations):
            Delta = self.Delta_bar
            z = 0
            z_twamm = 0
            total_reward = 0

            for t in range(self.T):
                # Again, clamp the state index properly
                state_index = self.state_index(Delta, z)

                action_index = policy[state_index]
                action = self.actions[action_index]
                action = min(action, Delta)

                Delta, z, z_twamm, reward_next, _, _ = (
                    self.transition_adjusted_stochastic(Delta, z, z_twamm, action)
                )
                total_reward += reward_next

            reward_paths.append(total_reward)

        return np.mean(reward_paths, axis=0)