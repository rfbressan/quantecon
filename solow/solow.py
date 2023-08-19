import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


class Solow:
    def __init__(self, s, delta, k0, production="Cobb-Douglas", **kwargs):
        self.s, self.delta, self.k0, self.production = s, delta, k0, production

        # Cobb-Douglas attributes
        if production == "Cobb-Douglas":
            self.A, self.alpha = kwargs.get('A'), kwargs.get('alpha')
        # CES attributes
        elif production == "CES":
            self.A, self.rho, self.alpha = kwargs.get('A'), kwargs.get('rho'), kwargs.get('alpha')
        else:
            raise ValueError("Production function must be either 'Cobb-Douglas' or 'CES'")

    def __repr__(self) -> str:
        return f"Solow model with parameters:\n s:{self.s}\n delta:{self.delta}\n k0:{self.k0} \n Production:{self.production}\n"

    def f(self, k):
        """Production function as a function of capital per capita

        :param k: effective capital per capita
        :type k: float numpy array
        :return: production level
        :rtype: float numpy array
        """
        k = np.asarray(k)
        if self.production == "Cobb-Douglas":
            return self.A * k**self.alpha
        elif self.production == "CES":
            return self.A * (self.alpha * k**self.rho + (1 - self.alpha))**(1 / self.rho)

    def k_1(self, k):
        """Capital accumulation

        :param k: effective capital per capita
        :type k: float numpy array
        :return: capital per capita in the next period
        :rtype: float numpy array
        """

        k = np.asarray(k)
        return (1 - self.delta) * k + self.s * self.f(k)

    def steady_state(self):
        """Steady state of the model

        :return: steady state capital per capita
        :rtype: float
        """
        # Objective function to find the root
        def obj(k):
            return k - self.k_1(k)
        
        try:
            # Find the root of the objective function
            return optimize.root_scalar(obj, bracket=[0.01, 100]).root
        except:
            print("No steady state found")
            return None
        # return ((self.s * self.A) / (self.delta)) ** (1 / (1 - self.alpha))

    def consumption_ss(self):
        """Consumption per capita in the steady state

        :return: consumption value
        :rtype: float
        """
        return (1 - self.s) * self.f(self.steady_state())

    def plot45(self, kgrid, kstar):
        """Plot the 45-degree line and the capital accumulation curve

        :param kgrid: grid of capital per capita values
        :type kgrid: float numpy array
        :param kstar: steady state capital per capita
        :type kstar: float
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(kgrid, self.k_1(kgrid), "b-", label="$k_{t+1}$")
        ax.plot(kgrid, kgrid, "k--", label="$45$")
        # plot a star at the steady state
        ax.plot(kstar, kstar, "*", markersize=10.0, color="green")
        # annotate the value of kstar
        ax.annotate(
            f"$k^*=${kstar:.2f}",
            xy=(kstar, kstar),
            xytext=(kstar + 0.1, kstar - 0.1),
            fontsize=12,
            color="green",
        )
        ax.set_title("Capital accumulation")
        ax.set_xlabel("$k_t$")
        ax.set_ylabel("$k_{t+1}$")
        ax.legend()
        plt.show()

    def simulate_ts(self, k_init, ts_length, plot=True):

        ts = np.zeros(ts_length)
        ts[0] = k_init
        for t in range(1, ts_length):
            ts[t] = self.k_1(ts[t - 1])

        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(ts, "b-", label="$k_{t+1}$")
            ax.set_xlabel("$t$")
            ax.set_ylabel("$k_{t+1}$")
            ax.set_title("Time series for capital per capita")
            ax.legend()
            plt.show()

        return ts

    def plot_consumption_savings(self, s, plot=True):
        """Consumption per capita in the steady state

        :param s: Exogenous saving rates
        :type s: numpy array
        :param plot: Show the plot, defaults to True
        :type plot: bool, optional
        :return: Consumption per capita values
        :rtype: numpy array
        """
        s = np.asarray(s)
        # Compute steady state capital per capita for the given saving rate
        kstar = ((s * self.A) / (self.delta)) ** (1 / (1 - self.alpha))
        consumption = (1 - s) * self.f(kstar)
        # Golden rule saving rate
        golden_idx = np.argmax(consumption)
        golden_savings = s[golden_idx]
        golden_consumption = consumption[golden_idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(s, consumption, "b-", label="$k^*$")
        ax.plot(
            golden_savings,
            golden_consumption,
            "*",
            markersize=10.0,
            color="green",
        )
        # Annotate the golden rule
        ax.annotate(
            f"Golden rule: $s_{{gr}}=${golden_savings:.2f}",
            xy=(golden_savings, golden_consumption),
            xytext=(golden_savings - 0.1, golden_consumption - 0.4),
            fontsize=12,
            color="green",
        )
        ax.set_xlabel("Saving rate")
        ax.set_ylabel("Steady state consumption")
        ax.legend()

        if plot:
            plt.show()

        return consumption
