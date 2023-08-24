"""
Lake Model of Unemployment and Employment Dynamics
"""
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


@dataclass
class LakeModel:
    """
    Solves the lake model and computes dynamics of the unemployment stocks and
    rates.

    Parameters:
    ------------
    λ : scalar
        The job finding rate for currently unemployed workers
    α : scalar
        The dismissal rate for currently employed workers
    b : scalar
        Entry rate into the labor force
    d : scalar
        Exit rate from the labor force

    """

    λ: float = 0.1
    α: float = 0.013
    b: float = 0.0124
    d: float = 0.00822
    g: float = field(init=False)
    A: NDArray = field(init=False)
    ū: float = field(init=False)
    ē: float = field(init=False)

    def __post_init__(self):
        self.g = self.b - self.d
        self.A = np.array(
            [
                [(1 - self.d) * (1 - self.λ) + self.b, self.α * (1 - self.d) + self.b],
                [(1 - self.d) * self.λ, (1 - self.α) * (1 - self.d)],
            ]
        )
        self.ū = (1 + self.g - (1 - self.d) * (1 - self.α)) / (
            1 + self.g - (1 - self.d) * (1 - self.α) + (1 - self.d) * self.λ
        )
        self.ē = 1 - self.ū

    def simulate_path(self, x0, T=1000):
        """
        Simulates the sequence of employment and unemployment

        Parameters
        ----------
        x0 : array
            Contains initial values (u0,e0)
        T : int
            Number of periods to simulate

        Returns
        ----------
        x : iterator
            Contains sequence of employment and unemployment rates

        """
        x0 = np.atleast_1d(x0)  # Recast as array just in case
        x_ts = np.zeros((2, T))
        x_ts[:, 0] = x0
        for t in range(1, T):
            x_ts[:, t] = self.A @ x_ts[:, t - 1]
        return x_ts


# Plot Functions


def plot_dynamics(x_ts: NDArray):
    """Dynamics of employment and unemployment rates

    Parameters
    ----------
    x_ts : ndarray
        Path of unemployed and employed populations
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    population = x_ts.sum(axis=0)
    # Unemployment panel
    axes[0].plot(x_ts[0, :] / population, lw=2)
    axes[0].set_title("Unemployment Rate")

    axes[1].plot(x_ts[1, :] / population, lw=2)
    axes[1].set_title("Employment Rate")

    for ax in axes:
        ax.grid()

    plt.tight_layout()
    plt.show()


def plot_path(lake_model: LakeModel, x_ts: NDArray):
    """Path diagram of unemployed and employed populations

    Parameters
    ----------
    x_ts : ndarray
        Path of unemployed and employed populations
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_ts[0, :], x_ts[1, :], "o", color="green", alpha=0.5)
    # Plot the dashed line along the long term growth path
    ax.axline((0, 0), (lake_model.ū, lake_model.ē), linestyle="--", color="black")
    ax.set_xlabel("Unemployed")
    ax.set_ylabel("Employed")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.grid()
    plt.show()


def main():
    """Example of usage of class LakeModel"""
    e_0 = 0.92  # Initial employment
    u_0 = 1 - e_0  # Initial unemployment, given initial n_0 = 1
    T = 100  # Simulation length
    x_0 = np.array([u_0, e_0])

    lm = LakeModel()
    x_path = lm.simulate_path(x_0, T)
    # plot_dynamics(x_path)
    plot_path(lm, x_path)


if __name__ == "__main__":
    main()
