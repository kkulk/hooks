import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from latexify import latexify

# Define global and local indices
global_indices = list(range(3))
local_indices = [
    [0, 1, 2],
    [0, 1],
    [1, 2],
    [0, 2],
    [0, 2]
]

# Define reserves for pools
reserves = list(map(np.array, [
    [3, .2, 1],  # Balancer pool
    [10, 1],     # Uniswap pool 1
    [1, 10],     # Uniswap pool 2
    [20, 50],    # Uniswap pool 3 
    [10, 10]     # Constant sum pool 
]))

# Define fees for the pools
fees = np.array([
    .98,
    .99,
    .96,
    .97,
    .99
])

# Define range of input, t, values
amounts = np.linspace(1, 500, 100)

# Limit order
limit_price = 0.5  
limit_volume = 40  

u_t_with_limit = []
u_t_without_limit = []

all_values_with_limit = [np.zeros((len(l), len(amounts))) for l in local_indices]
all_values_without_limit = [np.zeros((len(l), len(amounts))) for l in local_indices]

for j, t in enumerate(amounts):
    current_assets = np.array([t, 0, 0])  

    n = len(global_indices)
    m = len(local_indices)

    A = []
    for l in local_indices:
        n_i = len(l)
        A_i = np.zeros((n, n_i))
        for i, idx in enumerate(l):
            A_i[idx, i] = 1
        A.append(A_i)

    # Separate variables (with and without limit order)
    deltas_with_limit = [cp.Variable(len(l), nonneg=True) for l in local_indices]
    lambdas_with_limit = [cp.Variable(len(l), nonneg=True) for l in local_indices]

    deltas_without_limit = [cp.Variable(len(l), nonneg=True) for l in local_indices]
    lambdas_without_limit = [cp.Variable(len(l), nonneg=True) for l in local_indices]

    # Limit order variables
    z1 = cp.Variable(nonneg=True)  # Input amount of asset 0 to limit order
    z2 = cp.Variable(nonneg=True)  # Output amount of asset 2 from limit order

    psi_with_limit = (
        cp.sum([A_i @ (L - D) for A_i, D, L in zip(A, deltas_with_limit, lambdas_with_limit)])
        - z1 * np.array([1, 0, 0])  
        + z2 * np.array([0, 0, 1])  
    )

    psi_without_limit = cp.sum([A_i @ (L - D) for A_i, D, L in zip(A, deltas_without_limit, lambdas_without_limit)])

    obj_with_limit = cp.Maximize(psi_with_limit[2])
    obj_without_limit = cp.Maximize(psi_without_limit[2])

    new_reserves_with_limit = [R + gamma_i * D - L for R, gamma_i, D, L in zip(reserves, fees, deltas_with_limit, lambdas_with_limit)]
    new_reserves_without_limit = [R + gamma_i * D - L for R, gamma_i, D, L in zip(reserves, fees, deltas_without_limit, lambdas_without_limit)]

    # Constraints
    common_cons_with_limit = [
        cp.geo_mean(new_reserves_with_limit[0], p=np.array([3, 2, 1])) >= cp.geo_mean(reserves[0], p=np.array([3, 2, 1])),
        cp.geo_mean(new_reserves_with_limit[1]) >= cp.geo_mean(reserves[1]),
        cp.geo_mean(new_reserves_with_limit[2]) >= cp.geo_mean(reserves[2]),
        cp.geo_mean(new_reserves_with_limit[3]) >= cp.geo_mean(reserves[3]),
        cp.sum(new_reserves_with_limit[4]) >= cp.sum(reserves[4]),
        new_reserves_with_limit[4] >= 0,
        psi_with_limit + current_assets >= 0
    ]

    common_cons_without_limit = [
        cp.geo_mean(new_reserves_without_limit[0], p=np.array([3, 2, 1])) >= cp.geo_mean(reserves[0], p=np.array([3, 2, 1])),
        cp.geo_mean(new_reserves_without_limit[1]) >= cp.geo_mean(reserves[1]),
        cp.geo_mean(new_reserves_without_limit[2]) >= cp.geo_mean(reserves[2]),
        cp.geo_mean(new_reserves_without_limit[3]) >= cp.geo_mean(reserves[3]),
        cp.sum(new_reserves_without_limit[4]) >= cp.sum(reserves[4]),
        new_reserves_without_limit[4] >= 0,
        psi_without_limit + current_assets >= 0
    ]

    # Constraints for the case with limit orders
    cons_with_limit = common_cons_with_limit + [
        limit_price * z1 >= z2, 
        z2 <= limit_volume,       
        z1 <= t              
    ]

    prob_with_limit = cp.Problem(obj_with_limit, cons_with_limit)
    prob_with_limit.solve()
    u_t_with_limit.append(obj_with_limit.value)

    prob_without_limit = cp.Problem(obj_without_limit, common_cons_without_limit)
    prob_without_limit.solve()
    u_t_without_limit.append(obj_without_limit.value)

    for k in range(len(local_indices)):
        all_values_with_limit[k][:, j] = lambdas_with_limit[k].value - deltas_with_limit[k].value
        all_values_without_limit[k][:, j] = lambdas_without_limit[k].value - deltas_without_limit[k].value

# ============================= PLOTTING =============================

# Plot u(t) with and without limit orders
latexify(fig_width=6, fig_height=3.5)
plt.figure()
plt.plot(amounts, u_t_with_limit, "b", label="with limit order")
plt.plot(amounts, u_t_without_limit, "r--", label="without limit order")
plt.xlabel("$t$")
plt.ylabel("$u(t)$")
plt.title("$u(t)$, with and without limit orders")
plt.grid(True)
plt.legend()
plt.savefig("../figures/u-multi-asset-limit.png", bbox_inches="tight", dpi=500)

# Plot trades in each market with and without the limit order
latexify(fig_width=8, fig_height=5)
plt.figure()
for k in range(len(local_indices)):
    curr_value_with_limit = all_values_with_limit[k]
    curr_value_without_limit = all_values_without_limit[k]

    for i in range(curr_value_with_limit.shape[0]):
        plt.plot(amounts, curr_value_with_limit[i, :], label=f"Market {k+1}, Asset {i+1} (with limit order)")
        plt.plot(amounts, curr_value_without_limit[i, :], "--", label=f"Market {k+1}, Asset {i+1} (without limit order)")

plt.legend(loc='center left', bbox_to_anchor=(1, .5))
plt.xlabel("$t$")
plt.ylabel("Trade volume")
plt.title("Trade volume in each market with and without limit order")
plt.grid(True)
plt.savefig("../figures/trades-multi-asset-limit.png", bbox_inches="tight", dpi=500)
