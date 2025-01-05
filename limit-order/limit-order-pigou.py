import cvxpy as cp  
import numpy as np
import matplotlib.pyplot as plt 
from latexify import latexify

R1 = 4  # Initial reserves for token 1 in CFMM
R2 = 10  # Initial reserves for token 2 in CFMM
p = 0.5 # Price of the limit order
v = 3 # Volume of the limit order

### D is the amount of input token the user wants to trade. 
### We sweep over various values of D
D_values = np.linspace(1, 20, 100) 
total_outputs = []
cfmm_only_outputs = []

# Iterate over different values of D
for D in D_values:
    y = cp.Variable(1, nonneg=True)  # Output from CFMM
    x = cp.Variable(1, nonneg=True)  # Input to CFMM
    z1 = cp.Variable(1, nonneg=True)  # Input to limit order
    z2 = cp.Variable(1, nonneg=True)  # Output from limit order

    # Objective: Maximize total output
    obj = cp.Maximize(y + z2)

    # Constraints for CFMM + limit order
    cons = [
        # CFMM constraint
        cp.geo_mean(cp.hstack([R1 + x, R2 - y])) >= cp.geo_mean([R1, R2]),

        # Limit order constraints
        p * z1 >= z2,  # Price constraint
        z2 <= v,       # Output cap
        
        z1 + x <= D,   # Total input constraint
    ]

    prob = cp.Problem(obj, cons)
    prob.solve()
    total_outputs.append(y.value[0] + z2.value[0])

    # Problem with CFMM only, without limit order
    obj_cfmm = cp.Maximize(y)
    cons_cfmm = [
        cp.geo_mean(cp.hstack([R1 + x, R2 - y])) >= cp.geo_mean([R1, R2]),
        x <= D  # Total input constraint
    ]
    prob_cfmm = cp.Problem(obj_cfmm, cons_cfmm)
    prob_cfmm.solve()
    cfmm_only_outputs.append(y.value[0])

# Plot results
latexify(fig_width=6, fig_height=3.5)
plt.plot(D_values, total_outputs, label='CFMM + limit order', color='blue')
plt.plot(D_values, cfmm_only_outputs, label='CFMM only', color='orange', linestyle='--')
plt.xlabel('$D$, total input size', fontsize = 16)
plt.ylabel('Optimal output', fontsize = 16)
plt.title('Optimal output vs. input trade size, Pigou network', fontsize= 18)
plt.legend()
plt.grid(True)
plt.savefig('../figures/pigou-limit.png', dpi=500, bbox_inches="tight")