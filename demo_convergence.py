import numpy as np
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
from spring import SpringCoupledMasses

# define system
fine_solution = SpringCoupledMasses(
    m=[1, 1, 1],
    pairs=[(0, 1), (0, 2), (1, 2)],
    stiffness=[1, 1, 1],
    damping=[0, 0, 0],
)
fine_solution.set_initial_condition(d=3)
coarse_solution = deepcopy(fine_solution)
# compute coarse and fine solution
T = 20
n_coarse = 1001
n_fine = 100001
start = time()
coarse_solution.rk4(T=T, timesteps=n_coarse, frames=101)
print(f"Solved coarse solution in {time() - start:.2f} s")
start = time()
fine_solution.rk4(T=T, timesteps=n_fine, frames=101)
print(f"Solved fine solution in {time() - start:.2f} s")

# plot
m1error = np.abs(coarse_solution.X[:, 0, 0] - fine_solution.X[:, 0, 0])
m2error = np.abs(coarse_solution.X[:, 1, 0] - fine_solution.X[:, 1, 0])
m3error = np.abs(coarse_solution.X[:, 2, 0] - fine_solution.X[:, 2, 0])
plt.semilogy(coarse_solution.t, m1error, label='m1')
plt.semilogy(coarse_solution.t, m2error, label='m2')
plt.semilogy(coarse_solution.t, m3error, label='m3')
plt.legend()
plt.xlabel('t')
plt.ylabel('abs error x')
plt.title(f"Timestep size of {T / (n_fine - 1):.0e} vs {T / (n_coarse - 1):.0e}")
plt.show()