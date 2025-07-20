"""
NDArray version of the `dist` classes.
Also supports multivariate distributions.
Forthcoming.
"""

from dist import Distribution, ScalarDistribution, ContinuousDistribution, \
                 DiscreteDistribution, as_distribution
from _types import Int, Float
from util import np

from functools import cached_property
from typing import Sequence, Callable, Optional


class ProductDistribution(Distribution):
  """
  A distribution representing the product of two independent distributions.
  """

  # TODO: In the long run we obviously need to implement correlated distributions.
  #       The independent distribution case should be a special case of that.

  def __init__(self, distributions: ScalarDistribution | Sequence[ScalarDistribution]) -> None:

    if isinstance(distributions, ScalarDistribution):
      distributions = distributions,

    self.distributions = tuple(map(as_distribution, distributions))
    assert self.distributions, "At least one distribution must be provided."

    base = {True: ContinuousDistribution,
            False: DiscreteDistribution}[self.continuous]

    assert all( isinstance(dist, base) for dist in self.distributions ), \
      NotImplementedError("Mixed distributions are not supported yet.")

  @property
  def shape(self):
    return ()

  @cached_property
  def continuous(self):
    return isinstance(self.distributions[0], ContinuousDistribution)

  @cached_property
  def pdf(self) -> Callable:
    return \
      lambda x: np.multiply.reduce([dist.pdf(x[..., i])
                                    for i, dist in enumerate(self.distributions)])

  def _sample(self, size: Int) -> np.ndarray:
    return np.prod([dist.sample(size) for dist in self.distributions], axis=0)

  def _raw_moment(self, order: Int) -> Float:
    return np.prod([dist.raw_moment(order) for dist in self.distributions])

  def __len__(self):
    return len(self.distributions)


class VectorialDistribution(Distribution):

  # TODO: for now only vectorial. Tensorial follows later.

  def __init__(self, distributions: Distribution,
                     dependencies: Optional[Sequence[Sequence[Int]]] = None) -> None:

    # We handle only ProductDistribution in this class.
    # If another distribution is passed, it will simply be coerced.
    self.distributions = tuple(map(ProductDistribution, distributions))
    assert all( dist.continuous == self.distributions[0].continuous for dist in self.distributions ), \
      NotImplementedError("Mixed distributions are not supported yet.")

    if dependencies is None:
      dependencies = tuple( tuple(range(len(dist))) for dist in self.distributions )

    self.dependencies = tuple( tuple(int(i) for i in dep) for dep in dependencies )
    assert len(self.dependencies) == len(self.distributions) and \
           all( len(dep) == len(set(dep)) for dep in self.dependencies )
    assert all( len(dep) == len(dist) for dep, dist in zip(self.dependencies,
                                                           self.distributions) )
    depunion = set.union(*map(set, self.dependencies))
    assert depunion == set(range(len(depunion)))

  def __len__(self):
    return len(self.distributions)

  @property
  def shape(self):
    return len(self),

  @cached_property
  def pdf(self):
    def _pdf(x):
      return np.stack([ dist.pdf(x[..., dep])
                        for dist, dep in zip(self.distributions,
                                             self.dependencies) ], axis=-1)
    return _pdf

  def _sample(self, size: Int) -> np.ndarray:
    samples = [ dist.sample(size) for dist in self.distributions ]
    return np.stack(samples, axis=-1)

  def _raw_moment(self, order: Int) -> Float:
    return np.array([ dist.raw_moment(order) for dist in self.distributions ])
