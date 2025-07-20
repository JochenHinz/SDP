"""
This is a very first implementation of a Markov Chain based
Dynamic Programming problem for inventory management.

It is currently plagued by the curse of dimensionality, but the current
script serves as a starting point for further development, in particular
for the dimensionality reduction techniques that will be needed in
the future.
"""


from util import np, frozen, isincreasing
from _types import Immutable
from dist import VectorialDistribution

from typing import Tuple, Callable, Sequence
from functools import cached_property

from numpy.typing import NDArray
from matplotlib import pyplot as plt
from scipy import interpolate


# by default we choose the grid's window size to be 2 standard deviations
STD_WINDOW = 2


class RewardTransitionFunction(Immutable):

  def __init__(self, function: Callable):

    self.function = function  # g(xk, uk, wk)
    assert isinstance(function, Callable)

  def __call__(self, xk, uk, wk) -> Tuple[NDArray, NDArray]:
    """
    Evaluate the cost function at the given state xk, action uk, and noise wk.
    """
    return self.function(xk, uk, wk)


class DiscreteSpace(Immutable):
  """
  For state and control.
  """

  def __init__(self, grid: Sequence[NDArray]) -> None:
    """
    Initialize the state space with a grid of NDArray.
    """
    self.grid = tuple(frozen(g) for g in grid)
    assert all(isincreasing(g) for g in self.grid), \
      "All grids must be increasing sequences."
    assert all( grid.ndim == 1 for grid in self.grid ), \
      "All grids must be one-dimensional arrays."

  def __iter__(self):
    yield from self.grid

  def __len__(self):
    """
    Return the number of dimensions in the state space.
    """
    return len(self.grid)

  @cached_property
  def nelems(self) -> int:
    """
    Return the number of elements in the state space.
    """
    return np.prod([g.size for g in self.grid], dtype=int)

  @property
  def shape(self) -> Tuple[int, ...]:
    """
    Return the shape of the state space.
    """
    return tuple(g.size for g in self.grid)

  @property
  def intervals(self) -> Tuple[NDArray, ...]:
    """
    Return the intervals of the state space.
    """
    return tuple((g[0], g[-1]) for g in self.grid)


class StochasticDynamicProgrammingProblem(Immutable):

  def __init__(self,
               stochastic_variables: VectorialDistribution,           # stochastic variables
               reward_transition_function: RewardTransitionFunction,  # immediate reward function
               state_space: DiscreteSpace,                            # state space grid
               control_space: DiscreteSpace,                          # control space grid
               ) -> None:

    self.stochastic_variables = \
      VectorialDistribution(stochastic_variables)
    self.reward_transition_function = \
      RewardTransitionFunction(reward_transition_function)

    self.state_space = DiscreteSpace(state_space)
    self.control_space = DiscreteSpace(control_space)

  def plot_distributions(self, T=20, show=True):
    xi = np.arange(T)

    fig, ax = plt.subplots()

    realizations = self.stochastic_variables.sample(T)

    for real in realizations:
      ax.plot(xi, real, '-o')

    ax.legend()
    ax.set_xticks(xi)
    ax.grid(True)

    if show:
      plt.show()

    return ax


class MultilinearInterpolation(Immutable):
  """
  For linearly interpolating continuous variables from data over
  discrete grid.
  """

  def __init__(self, grid: DiscreteSpace, values: NDArray) -> None:
    """
    Initialize the multilinear interpolation with a grid and corresponding values.
    """
    self.grid = DiscreteSpace(grid)
    self.values = frozen(values)

    assert self.values.shape[:1] == (self.grid.nelems,), \
      "Values shape must match the number of elements in the grid."

    self.interpolator = interpolate.RegularGridInterpolator(
      self.grid.grid, self.values.reshape(self.grid.shape), bounds_error=False
    )

  def __call__(self, points: NDArray) -> NDArray:
    raise NotImplementedError("Multilinear interpolation is not implemented yet.")
