from util import np
from _types import Float, Int
from opt import RewardTransitionFunction, DiscreteSpace, \
                StochasticDynamicProgrammingProblem
from dist import VectorialDistribution, LogNormal, Geometric, Poisson, \
                 Bernoulli
from log import logger as log

from typing import Tuple
from collections import namedtuple

from numpy.typing import NDArray
from numba import njit


"""
Xk = [stored_resources, stored_products]
Uk = [sell_products, buy_resources, produce_products]
Wk = [resource_cost, energy_cost, resource_delay, demand, tariff]
"""


InventoryParameters = namedtuple('InventoryProblemInput',
                                 ['resource_inventory_capacity',
                                  'product_inventory_capacity',
                                  'production_capacity',
                                  'resource_storage_cost',
                                  'product_storage_cost',
                                  'production_cost',
                                  'selling_price',
                                  'tariff_rate'])


def create_inventory(resource_inventory_capacity: Int,
                     product_inventory_capacity: Int,
                     production_capacity: Int,
                     resource_storage_cost: Float,
                     product_storage_cost: Float,
                     production_cost: Float,
                     selling_price: Float,
                     tariff_rate: Float) -> InventoryParameters:

  ret = InventoryParameters(int(resource_inventory_capacity),
                            int(product_inventory_capacity),
                            int(production_capacity),
                            float(resource_storage_cost),
                            float(product_storage_cost),
                            float(production_cost),
                            float(selling_price),
                            float(tariff_rate))

  assert all( x >= 0 for x in ret[:-1] )
  assert 0 <= ret.tariff_rate <= 1

  return ret


def create_stochastic_variables(R_data: Tuple[Float, Float],  # resource [mean, std]
                                E_data: Tuple[Float, Float],  # energy cost [mean, std]
                                p_delay: Float,               # resource delay probability
                                lambda_demand: Float,         # demand rate
                                p_tariff: Float               # tariff introduction / abolition probability
                                ) -> VectorialDistribution:

  resource_cost = LogNormal.from_mean_std(*R_data)  # mean, std^2
  energy_cost = LogNormal.from_mean_std(*E_data)  # mean, std^2
  resource_delay = Geometric(p_delay)
  demand = Poisson(lambda_demand)
  tariff = Bernoulli(p_tariff)

  return VectorialDistribution([resource_cost,
                                energy_cost,
                                resource_delay,
                                demand,
                                tariff])


def reward_transition_function(inventory_parameters: InventoryParameters) -> RewardTransitionFunction:

  # reward = g(Xk, Uk, Wk), Xk+1 = F(Xk, Uk, Wk)

  resource_inventory_capacity, product_inventory_capacity, \
  production_capacity, resource_storage_cost, product_storage_cost, \
  production_cost, selling_price, tariff_rate = inventory_parameters

  @njit(cache=True, fastmath=True)
  def reward_transition(xk, uk, wk) -> Tuple[Float | NDArray, Float | NDArray]:
    # TODO: vectorize or use numba for performance

    """
    At the beginning of each cycle, we can decide to sell as many products
    as we have in the product inventory, capped by the current day's demand.
    We cannot sell what we produce that day but the products we sell
    immediately clear up the product inventory for potential storage.

    We may buy as many resources as we can. By assumption the resources become
    available immediately, so we can produce products on the same day.
    That means that after we sell products, we can buy resources and then
    produce as much as as our storage after selling allows us to.

    Upon buying resources, we can choose to produce as many products as we
    can after buying the resources. The product is produced over night and
    becomes available the next day. We assume that the storage costs are
    immediately incurred that night.

    Nomenclature:

    Xk = [stored_resources, stored_products]
    Uk = [sell_products, buy_resources, produce_products]
    Wk = [resource_cost, energy_cost, resource_delay, demand, tariff]

    new_products: the number of products available after selling that day
    new_resources: the number of resources available after buying that day

    next_products: the number of products available after producing that day
    next_resources: the number of resources available after producing that day
    """

    resources, products = xk
    buy, produce, sell = uk

    # we ignore the delay for now
    resource_cost, energy_cost, resource_delay, demand, tariff = wk

    # We cannot sell more than the demand or the number of available products.
    amount_sold = min(sell, products, demand)

    # The number of products is immediately reduced by the number of products
    # sold.
    new_products = products - amount_sold  # current day's products

    # We can buy as many resources as we please, capped by the resource
    # inventory capacity.
    resources_bought = min(buy, resource_inventory_capacity - resources)

    # the resources available to us for overnight production
    new_resources = resources + buy

    # we can produce as much as can be produced overnight (production capacity)
    # and as much as we have resources for.
    amount_produced = min(produce, production_capacity, new_resources)

    # The next state is given by [new_resources - amount_produced,
    #                             new_products + amount_produced]
    next_resources = new_resources - amount_produced
    next_products = new_products + amount_produced

    Xk1 = np.array([next_resources, next_products], dtype=np.int64)

    # The immediate reward is given by
    reward = selling_price * amount_sold * (1 - tariff * tariff_rate) - \
             resource_cost * resources_bought - \
             (production_cost + energy_cost) * amount_produced - \
             resource_storage_cost * next_resources - \
             product_storage_cost * next_products

    return reward, Xk1

  return RewardTransitionFunction(reward_transition)


def create_state_space(inventory_parameters: InventoryParameters,
                       nrstorage: Int,  # nsteps number of resource storage units
                       npstorage: Int   # nsteps number of product storage units
                       ) -> DiscreteSpace:

  resource_inventory_capacity, product_inventory_capacity, *ignore = inventory_parameters

  assert resource_inventory_capacity % nrstorage == 0
  xrstorage = np.arange(0, resource_inventory_capacity + 1, nrstorage, dtype=int)

  assert product_inventory_capacity % npstorage == 0
  xpstorage = np.arange(0, product_inventory_capacity + 1, npstorage, dtype=int)

  return DiscreteSpace([xrstorage, xpstorage])


def create_control_space(inventory_parameters: InventoryParameters,
                         nsell: Int,        # nsteps number of product units to sell
                         nbuy: Int,         # nsteps number of resource units to buy
                         nproduction: Int,  # nsteps number of production units
                         ) -> DiscreteSpace:

  resource_inventory_capacity, \
  product_inventory_capacity, \
  production_capacity, *ignore = inventory_parameters

  assert product_inventory_capacity % nsell == 0
  xsell = np.arange(0, product_inventory_capacity + 1, nsell, dtype=int)

  assert resource_inventory_capacity % nbuy == 0
  xbuy = np.arange(0, resource_inventory_capacity + 1, nbuy, dtype=int)

  assert production_capacity % nproduction == 0
  xproduce = np.arange(0, production_capacity + 1, nproduction, dtype=int)

  return DiscreteSpace([xsell, xbuy, xproduce])


if __name__ == '__main__':

  horizon = 6

  inventory = create_inventory(
      resource_inventory_capacity=50,
      product_inventory_capacity=50,
      production_capacity=10,
      resource_storage_cost=2.0,
      product_storage_cost=4.0,
      production_cost=5.0,
      selling_price=30.0,
      tariff_rate=.3
  )

  stochastic_variables = \
    create_stochastic_variables(R_data=(7, 1),     # Resource cost mean, std
                                E_data=(2, .2),    # Energy cost mean std
                                p_delay=2/3,       # Resource delay probability
                                lambda_demand=10,  # Demand rate
                                p_tariff=.1)       # Tariff on / off rate

  state_space = create_state_space(inventory, 5, 10)
  print("The state space has {} elemens.".format(state_space.nelems))

  control_space = create_control_space(inventory, 5, 5, 2)
  print("The control space has {} elements".format(control_space.nelems))

  # Example usage
  dp_problem = \
    StochasticDynamicProgrammingProblem(stochastic_variables,
                                        reward_transition_function(inventory),
                                        state_space,
                                        control_space,
                                        stochastic_variable_names=('resource_cost',
                                                                   'energy_cost',
                                                                   'resource_delay',
                                                                   'demand_rate',
                                                                   'tariff'))

  rewards = [dp_problem.create_salvage_function(lambda xk: 2 * xk.sum())]  # Salvage function, can be 0 or a function
  policy_functions = []

  for i in range(horizon):
    policy, reward = dp_problem.markov_step(rewards[-1])
    policy_functions.append(policy)
    rewards.append(reward)

  # turn both around
  policy_functions.reverse()
  rewards.reverse()

  xk_names = 'resource_inventory', 'product_inventory'
  uk_names = 'sell_products', 'buy_resources', 'produce_products'

  nrealizations = 10
  for i in range(nrealizations):
    log.warning(f"plotting realization {i + 1} of {nrealizations}")

    # we assume that we start with empty resource and product inventory
    dp_problem.realize([0, 0], policy_functions, xk_names, uk_names)
