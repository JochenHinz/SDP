from util import np
from _types import Immutable, Float, Int
from dist import ContinuousDistribution, DiscreteDistribution, as_distribution, \
                 Dirac, Uniform, LogNormal, Normal, Geometric, Poisson

from typing import Tuple, Optional

from numpy.typing import NDArray


class DynamicProgrammingProblem(Immutable):

  @classmethod
  def from_default_inputs(cls,
                          inventory_capacity: Int,
                          production_capacity: Int,
                          storage_cost: Float,
                          selling_price: Float,
                          resource_base_cost: Float,
                          sigma_R: Float,                   # resource cost uncertainty
                          E_data: Tuple[Float, Float],      # energy cost mean and uncertainty
                          p_delay: Float,                   # resource delay probability
                          lambda_demand: Float,             # demand rate
                          tariff_data: Tuple[Float, Float]  # tariff mean and uncertainty
                          ) -> 'DynamicProgrammingProblem':

    resource_cost = LogNormal(0, sigma_R**2)
    energy_cost = LogNormal(*E_data)
    resource_delay = Geometric(p_delay)
    demand = Poisson(lambda_demand)
    tariff = LogNormal(*tariff_data)

  def __init__(self,
               inventory_capacity: Int,
               production_capacity: Int,
               storage_cost: Float,
               selling_price: Float,
               resource_base_cost: Float,
               energy_cost: ContinuousDistribution,
               resource_cost: ContinuousDistribution,
               resource_delay: DiscreteDistribution,
               demand: DiscreteDistribution,
               tariff: ContinuousDistribution):

    self.inventory_capacity = int(inventory_capacity)
    self.production_capacity = int(production_capacity)
    self.storage_cost = float(storage_cost)
    self.selling_price = float(selling_price)
    self.resource_base_cost = float(resource_base_cost)

    assert all( x >= 0 for x in (self.inventory_capacity,
                                 self.production_capacity,
                                 self.storage_cost,
                                 self.selling_price,
                                 self.resource_base_cost) )

    self.energy_cost = as_distribution(energy_cost)
    self.resource_cost = as_distribution(resource_cost)
    self.resource_delay = as_distribution(resource_delay)
    self.demand = as_distribution(demand)
    self.tariff = as_distribution(tariff)

  def create_grid(self,
                  nstorage: Optional[Int] = None,
                  nproduction: Optional[Int] = None,
                  nenergy: Optional[Tuple[Float, Int]] = None,       # [maxval, nsteps]
                  nresource: Optional[Tuple[Int, Int, Int]] = None,  # [min, max, nsteps]
                  nrdelay: Optional[Int] = 3,                        # delay steps from 0 to 3
                  ndemand: Optional[Tuple[Int, Int]] = None,         # [maxval, nsteps]
                  ntarrif: Optional[Tuple[Float, Int]] = None        # [maxtariff between 0 and 1, nsteps]
                  ) -> Tuple[NDArray, ...]:

    """
    Create a grid for the dynamic programming problem.
    """
    if nstorage is None:
      nstorage = self.inventory_capacity + 1

    assert (self.inventory_capacity + 1) % nstorage == 0
    xstorage = np.arange(0,
                         self.inventory_capacity + 1,
                         (self.inventory_capacity + 1) // nstorage)

    if nproduction is None:
      nproduction = self.production_capacity + 1

    assert (self.production_capacity + 1) % nproduction == 0
    xproduction = np.arange(0,
                            self.production_capacity + 1,
                            (self.production_capacity + 1) // nproduction)
