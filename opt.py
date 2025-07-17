"""
This is a very first implementation of a Markov Chain based
Dynamic Programming problem for inventory management.

It is currently plagued by the curse of dimensionality, but the current
script serves as a starting point for further development, in particular
for the dimensionality reduction techniques that will be needed in
the future.
"""


from util import np, isincreasing
from _types import Immutable, Float, Int
from dist import LogNormal, Geometric, Poisson, Bernoulli

from typing import Tuple, Optional, Callable
from collections import namedtuple

from numpy.typing import NDArray
from matplotlib import pyplot as plt


# by default we choose the grid's window size to be 2 standard deviations
STD_WINDOW = 2


InventoryParameters = namedtuple('InventoryProblemInput',
                                 ['inventory_capacity',
                                  'production_capacity',
                                  'storage_cost',
                                  'selling_price',
                                  'tariff_rate'])


def create_inventory(inventory_capacity: Int,
                     production_capacity: Int,
                     storage_cost: Float,
                     selling_price: Float,
                     tariff_rate: Float) -> InventoryParameters:

  ret = InventoryParameters(int(inventory_capacity),
                            int(production_capacity),
                            float(storage_cost),
                            float(selling_price),
                            float(tariff_rate))

  assert all( x >= 0 for x in ret[:-1] )
  assert 0 <= ret.tariff_rate <= 1

  return ret


class InventoryProblem(Immutable):

  @classmethod
  def from_default_inputs(cls,
                          inventory_capacity: Int,
                          production_capacity: Int,
                          storage_cost: Float,
                          selling_price: Float,
                          tariff_rate: Float,
                          R_data: Tuple[Float, Float],  # resource [mean, std]
                          E_data: Tuple[Float, Float],  # energy cost [mean, std]
                          p_delay: Float,               # resource delay probability
                          lambda_demand: Float,         # demand rate
                          p_tariff: Float               # tariff [mean, std]
                          ) -> 'InventoryProblem':

    resource_cost = LogNormal.from_mean_std(*R_data)  # mean, std^2
    energy_cost = LogNormal.from_mean_std(*E_data)  # mean, std^2
    resource_delay = Geometric(p_delay)
    demand = Poisson(lambda_demand)
    tariff = Bernoulli(p_tariff)

    return cls(create_inventory(inventory_capacity,
                                production_capacity,
                                storage_cost,
                                selling_price,
                                tariff_rate),
               resource_cost,
               energy_cost,
               resource_delay,
               demand,
               tariff)

  def __init__(self,
               inventory_parameters: InventoryParameters,
               energy_cost: LogNormal,
               resource_cost: LogNormal,
               resource_delay: Geometric,
               demand: Poisson,
               tariff: Bernoulli) -> None:

    self.inventory_parameters = inventory_parameters
    assert isinstance(self.inventory_parameters, InventoryParameters)

    self.energy_cost = LogNormal(energy_cost)
    self.resource_cost = LogNormal(resource_cost)
    self.resource_delay = Geometric(resource_delay)
    self.demand = Poisson(demand)
    self.tariff = Bernoulli(tariff)

  def plot_distributions(self, T=20, show=True):
    xi = np.arange(T)

    fig, ax = plt.subplots()

    for name in 'energy_cost', 'resource_cost', 'resource_delay', 'demand':
      ax.plot(xi, getattr(self, name).sample(T), '-o', label=name)

    ax.plot(xi, self.tariff.sample(T).cumsum() % 2, '-o', label='tariff')

    ax.legend()
    ax.set_xticks(xi)
    ax.grid(True)

    if show:
      plt.show()

    return ax

  def create_grid(self,
                  nstorage: Optional[Int] = None,
                  nproduction: Optional[Int] = None,
                  nenergy: Optional[Tuple[Float, Float, Int]] = None,  # [minval, maxval, nsteps]
                  nresource: Optional[Tuple[Int, Int, Int]] = None,    # [min, max, nsteps]
                  ndemand: Optional[Tuple[Int, Int]] = None,           # [maxval, nsteps]
                  ntariff: Optional[Tuple[Float, Int]] = None          # [maxtariff between 0 and 1, nsteps]
                  ) -> Tuple[NDArray, ...]:

    # TODO: default values could become attributes of the Distribution classes

    """
    Create a grid for the dynamic programming problem.
    """

    inventory_capacity, production_capacity, \
    storage_cost, selling_price, tariff_rate = self.inventory_parameters

    if nstorage is None:
      nstorage = 1

    assert inventory_capacity % nstorage == 0
    xstorage = np.arange(0, inventory_capacity + 1, nstorage, dtype=int)

    if nproduction is None:
      nproduction = 1

    assert production_capacity % nproduction == 0
    xproduction = np.arange(0, production_capacity + 1, nproduction, dtype=int)

    if nenergy is None:
      nenergy = *self.energy_cost.credible_interval(.95), 5

    xenergy = np.linspace(*nenergy)
    assert isincreasing(xenergy)

    if nresource is None:
      nresource = *self.resource_cost.credible_interval(.95), 5

    xresource = np.linspace(*nresource)
    xrdelay = np.arange(4, dtype=int)

    if ndemand is None:
      var = self.demand.moment(2)
      maxdemand = int(self.demand.expectation + STD_WINDOW * var)
      ndemand = (maxdemand, maxdemand // 3)

    assert ndemand[0] % ndemand[1] == 0

    xdemand = np.arange(0, ndemand[0] + 1, ndemand[1], dtype=int)
    xtariff = np.arange(2)

    return xstorage, xproduction, xenergy, xresource, xrdelay, xdemand, xtariff


class MarkovChainCost(Immutable):
    """
    Represents the cost associated with a Markov chain transition.
    """

    def __init__(self, cost: Float, probability: Float):
        self.cost = float(cost)
        self.probability = float(probability)

    def __repr__(self):
        return f"MarkovChainCost(cost={self.cost}, probability={self.probability})"


def discrete_markov_step(costfunction: Callable,
                         sdp: InventoryProblem,
                         grid: Tuple[NDArray, ...],
                         n_MC: Int = 100,
                         theta: Float = 0.01):
  pass


if __name__ == '__main__':

  # Example usage
  dp_problem = InventoryProblem.from_default_inputs(
      inventory_capacity=100,
      production_capacity=10,
      storage_cost=1.0,
      selling_price=20.0,
      tariff_rate=.3,
      R_data=(8, 1),  # mean, std
      E_data=(2, .2),
      p_delay=2/3,
      lambda_demand=10,
      p_tariff=.1
  )

  grid = dp_problem.create_grid(nstorage=10,
                                nproduction=2)
  print(grid)

  import ipdb
  ipdb.set_trace()
