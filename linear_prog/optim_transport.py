from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, linprog
from scipy.stats import betabinom

# from quantecon.optimize.linprog_simplex import linprog_simplex
# import ot


@dataclass
class Node:
    """Holds the information of a node in the network."""

    x: float
    y: float
    mass: float
    group: str
    name: str


# Define the class for the Optimal Transport problem
@dataclass
class OptimalTransport:
    """Implements the Optimal Transport problem in the standard form."""

    p_nodes: list[Node] = field(repr=False)
    q_nodes: list[Node] = field(repr=False)
    p_vec: NDArray = field(init=False)  # Capacity of the supply nodes
    q_vec: NDArray = field(init=False)  # Demand of the demand nodes
    C_mat: NDArray = field(init=False)  # Cost matrix
    result: OptimizeResult = field(init=False, repr=False)  # Result of the optimization

    def __post_init__(self):
        self.C_mat = self.distance_matrix(self.p_nodes, self.q_nodes)
        self.p_vec = np.array([node.mass for node in self.p_nodes])
        self.q_vec = np.array([node.mass for node in self.q_nodes])

    def distance_matrix(self, nodes1: list, nodes2: list):
        """Computes the distance matrix between two lists of nodes."""
        n1 = len(nodes1)
        n2 = len(nodes2)
        D = np.empty((n1, n2))
        for i in range(n1):
            for j in range(n2):
                D[i, j] = np.sqrt(
                    (nodes1[i].x - nodes2[j].x) ** 2 + (nodes1[i].y - nodes2[j].y) ** 2
                )
        return D

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


# Auxiliary Functions
def build_random_nodes(group: str, n: int = 4, seed: int = 98765):
    nodes = []
    np.random.seed(seed)
    for i in range(n):
        if group == "p":
            m = 1 / n
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
        else:
            m = float(betabinom.pmf(i, n - 1, 2, 2))
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)

        name = group + str(i)
        nodes.append(Node(x, y, m, group, name))

    return nodes


def plot_network(opt_trans: OptimalTransport):
    g = nx.DiGraph()
    g.add_nodes_from([p.name for p in opt_trans.p_nodes])
    g.add_nodes_from([q.name for q in opt_trans.q_nodes])
    # Add the edges from result of opt_trans
    n_p = len(opt_trans.p_nodes)
    n_q = len(opt_trans.q_nodes)
    result_mat = opt_trans.result.x.reshape((n_p, n_q), order="F")
    for i in range(n_p):
        for j in range(n_q):
            if result_mat[i, j] > 0:
                g.add_edge(
                    opt_trans.p_nodes[i].name,
                    opt_trans.q_nodes[j].name,
                    weight=result_mat[i, j],
                )

    # Positioning the nodes
    node_positions = {}
    node_colors = []
    node_sizes = []
    scale = 1_000

    for p in opt_trans.p_nodes:
        node_positions[p.name] = (p.x, p.y)
        node_colors.append("blue")
        node_sizes.append(p.mass * scale)

    for q in opt_trans.q_nodes:
        node_positions[q.name] = (q.x, q.y)
        node_colors.append("red")
        node_sizes.append(q.mass * scale)

    # Plot the network
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis("off")
    nx.draw_networkx_nodes(
        g,
        node_positions,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="grey",
        linewidths=1,
        alpha=0.5,
        ax=ax,
    )
    nx.draw_networkx_edges(
        g,
        node_positions,
        arrows=True,
        connectionstyle="arc3, rad = 0.1",
        alpha=0.6,
    )
    plt.show()


def main():
    # Testing the nodes
    nodes_p = build_random_nodes("p")
    nodes_q = build_random_nodes("q")

    optim_transport = OptimalTransport(nodes_p, nodes_q)
    optim_transport.solve()
    print(optim_transport.result.x.reshape((len(nodes_p), len(nodes_q)), order="F"))
    plot_network(optim_transport)


if __name__ == "__main__":
    main()
