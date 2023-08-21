"""Implementation of the Solow growth model"""

from abc import ABC, abstractmethod

import numpy as np
from scipy import optimize

# from dataclasses import dataclass


class Solow:
    """Solow growth model"""

    def __init__(self, savings, delta, k_0, production):
        self.savings = savings
        self.delta = delta
        self.k_0 = k_0
        self.production = production

    def __repr__(self) -> str:
        return f"Solow model with parameters:\n savings:{self.savings}\n delta:{self.delta}\n k_0:{self.k_0} \n production:{self.production}\n"

    def k_update(self, k):
        """Capital accumulation rule

        :param k: effective capital per capita
        :type k: float numpy array
        :return: capital per capita in the next period
        :rtype: float numpy array
        """

        k = np.asarray(k)
        return (1 - self.delta) * k + self.savings * self.production.function(k)

    def steady_state(self):
        """Steady state of the model

        :return: steady state capital per capita
        :rtype: float
        """

        # Objective function to find the root
        def obj(k):
            return k - self.k_update(k)

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
        return (1 - self.savings) * self.production.function(
            self.steady_state()
        )

    def simulate_ts(self, k_init, ts_length):
        ts = np.zeros(ts_length)
        ts[0] = k_init
        for t in range(1, ts_length):
            ts[t] = self.k_update(ts[t - 1])

        return ts

    def capital_savings(self, sgrid):
        """Steady state capital per capita for different saving rates

        :param sgrid: Grid of saving rates
        :type sgrid: numpy array
        :return: Steady state capital per capita values
        :rtype: numpy array
        """
        sgrid = np.asarray(sgrid)
        kgrid = np.empty(len(sgrid))

        # Objective function to find the root
        def obj(x, s):
            return x - ((1 - self.delta) * x + s * self.production.function(x))

        for i, s in enumerate(sgrid):
            kgrid[i] = optimize.root_scalar(
                obj, args=(s), bracket=[0.001, 100]
            ).root

        return kgrid
        # try:
        #     # Find the root of the objective function
        #     return optimize.root_scalar(obj, bracket=[0.01, 100]).root
        # except:
        #     print("No steady state found")
        #     return None

    def consumption_savings(self, sgrid):
        """Consumption per capita in the steady state

        :param s: Exogenous saving rates
        :type s: numpy array
        :return: Consumption per capita values
        :rtype: numpy array
        """
        sgrid = np.asarray(sgrid)
        # Compute steady state capital per capita for the given saving rate
        kstar = self.capital_savings(sgrid)
        consumption = (1 - sgrid) * self.production.function(kstar)
        # Golden rule saving rate
        golden_idx = np.argmax(consumption)
        golden_savings = sgrid[golden_idx]
        golden_consumption = consumption[golden_idx]

        return {
            "consumption": consumption,
            "consumption_gr": golden_consumption,
            "savings_gr": golden_savings,
        }


class Production(ABC):
    """Base class for production functions."""

    @abstractmethod
    def function(self, k) -> float:
        """Abstract representation of the production function"""


class CobbDouglas(Production):
    """Cobb-Douglas production function"""

    def __init__(self, A, alpha):
        self.A = A
        self.alpha = alpha

    def __repr__(self) -> str:
        return "Cobb-Douglas"

    def function(self, k):
        """Production function as a function of capital per capita
        :param k: effective capital per capita
        :type k: float numpy array
        :return: production level
        :rtype: float numpy array
        """
        k = np.asarray(k)
        return self.A * k**self.alpha


class CES(Production):
    """CES production function"""

    def __init__(self, A, alpha, rho):
        self.A = A
        self.alpha = alpha
        self.rho = rho

    def __repr__(self) -> str:
        return "CES"

    def function(self, k):
        k = np.asarray(k)
        return self.A * (self.alpha * k**self.rho + (1 - self.alpha)) ** (
            1 / self.rho
        )
