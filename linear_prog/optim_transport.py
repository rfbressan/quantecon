from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from quantecon.optimize.linprog_simplex import linprog_simplex
from scipy.optimize import OptimizeResult, linprog

# import ot
from scipy.stats import betabinom

# import networkx as nx


# Define the class for the Optimal Transport problem
@dataclass
class OptimalTransport:
    """Implements the Optimal Transport problem in the standard form."""

    C_mat: NDArray  # Cost matrix
    p_vec: NDArray  # Capacity of the supply nodes
    q_vec: NDArray  # Demand of the demand nodes
    result: OptimizeResult = field(init=False)  # Result of the optimization

    def solve(self, verbose=True):
        """Solves the Optimal Transport problem using the simplex method from SciPy."""
        # Check the dimensions of the inputs
        p_len = len(self.p_vec)
        q_len = len(self.q_vec)
        assert self.C_mat.shape[0] == p_len
        assert self.C_mat.shape[1] == q_len
        # Vectorize the cost matrix
        C_vec = self.C_mat.flatten(order="F")
        # Construct the constraint matrix using Kronecker product
        A1 = np.kron(np.ones(q_len), np.identity(p_len))
        A2 = np.kron(np.identity(q_len), np.ones(p_len))
        A_mat = np.vstack([A1, A2])
        # Construct vector b
        b_vec = np.hstack([self.p_vec, self.q_vec])
        # Solve the linear programming problem
        self.result = linprog(C_vec, A_eq=A_mat, b_eq=b_vec)
        if verbose:
            print(self.result.message)


def main():
    p = np.array([50, 100, 150])
    q = np.array([25, 115, 60, 30, 70])

    C = np.array([[10, 15, 20, 20, 40], [20, 40, 15, 30, 30], [30, 35, 40, 55, 25]])

    optim_transport = OptimalTransport(C, p, q)
    optim_transport.solve()
    print(optim_transport.result.fun)
    print(optim_transport.result.x.reshape((len(p), len(q)), order="F"))


if __name__ == "__main__":
    main()
