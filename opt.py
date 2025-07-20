"""
This is a very first implementation of a Markov Chain based
Dynamic Programming problem for inventory management.

It is currently plagued by the curse of dimensionality, but the current
script serves as a starting point for further development, in particular
for the dimensionality reduction techniques that will be needed in
the future.
"""


from util import np, frozen, isincreasing, _
from _types import Immutable, Int
from dist import VectorialDistribution

from typing import Tuple, Callable, Sequence, Optional
from functools import cached_property
from itertools import product

from numpy.typing import NDArray
from matplotlib import pyplot as plt
from scipy import interpolate

from numba import njit, prange


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
    yield from product(*self.grid)

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
      self.grid.grid, self.values.reshape(self.grid.shape + self.values.shape[1:]), bounds_error=False
    )

  def __call__(self, points: NDArray) -> NDArray:
    points = np.asarray(points, dtype=np.float64)
    assert points.shape[1:] == (len(self.grid),)

    points = np.stack([ np.clip(p, *interval)
                        for p, interval in zip(points.T, self.grid.intervals) ], axis=-1)

    return self.interpolator(points)


@njit(cache=True, fastmath=True)
def _MC_sample(Xk, Uk, noise, reward_transition_function):
  N = noise.shape[0]
  m = Xk.shape[0]

  immediate_rewards = np.empty((N,), dtype=np.float64)
  next_states = np.empty((N, m), dtype=np.float64)

  for i in prange(N):
    wk = noise[i]
    immediate_reward, next_state = reward_transition_function(Xk, Uk, wk)
    immediate_rewards[i] = immediate_reward
    next_states[i] = next_state

  return immediate_rewards, next_states


class StochasticDynamicProgrammingProblem(Immutable):

  def __init__(self,
               stochastic_variables: VectorialDistribution,               # stochastic variables
               reward_transition_function: RewardTransitionFunction,      # immediate reward function
               state_space: DiscreteSpace,                                # state space grid
               control_space: DiscreteSpace,                              # control space grid
               stochastic_variable_names: Optional[Sequence[str]] = None  # names of stochastic variables
               ) -> None:

    self.stochastic_variables = \
      VectorialDistribution(stochastic_variables)
    self.reward_transition_function = \
      RewardTransitionFunction(reward_transition_function)

    self.state_space = DiscreteSpace(state_space)
    self.control_space = DiscreteSpace(control_space)

    if stochastic_variable_names is None:
      stochastic_variable_names = map('X{}'.format, range(self.stochastic_variables.ndim))

    self.stochastic_variable_names = tuple(map(str, stochastic_variable_names))
    assert len(self.stochastic_variable_names) == self.stochastic_variables.shape[0]

  def create_salvage_function(self,
                              salvage_function: Int | Callable = 0) -> RewardTransitionFunction:

    if isinstance((salvage := salvage_function), Int):
      salvage_function = lambda xk: salvage

    data = []
    for elem in self.state_space:
      data.append(salvage_function(np.array(elem, dtype=float)[_]))

    data = np.array(data, dtype=float)

    return MultilinearInterpolation(self.state_space, data)

  def markov_step(self, VK1: MultilinearInterpolation, N_MC=101) -> Tuple[MultilinearInterpolation, MultilinearInterpolation]:

    control_states = list(self.control_space)

    best_policy = []  # the best policy for each state
    reward = []
    for i, Xk in enumerate(map(np.asarray, self.state_space)):
      print(i)

      control_reward = []

      for j, Uk in enumerate(map(np.asarray, control_states)):
        immediate_rewards, next_states = _MC_sample(Xk,
                                                    Uk,
                                                    self.stochastic_variables.sample(N_MC),
                                                    self.reward_transition_function.function)

        control_reward.append(np.mean(immediate_rewards + VK1(next_states)))

      # Find the best control for the current state
      ibest_reward = np.argmax(control_reward)

      reward.append(control_reward[ibest_reward])
      best_policy.append( control_states[ibest_reward] )

    return MultilinearInterpolation(self.state_space, np.array(best_policy, dtype=float)), \
           MultilinearInterpolation(self.state_space, np.array(reward, dtype=float))

  def plot_distributions(self, T=20, show=True):
    xi = np.arange(T)

    fig, ax = plt.subplots()

    realizations = self.stochastic_variables.sample(T)

    for name, real in zip(self.stochastic_variable_names, realizations.T):
      ax.plot(xi, real, '-o', label=name)

    ax.legend()
    ax.set_xticks(xi)
    ax.grid(True)

    if show:
      plt.show()

    return ax

  def realize(self, X0,
                    policy_functions: Sequence[MultilinearInterpolation],
                    state_space_names: Optional[Sequence[str]] = None,
                    control_space_names: Optional[Sequence[str]] = None) -> None:

    X0 = np.asarray(X0, dtype=float)
    assert (N := len(policy_functions))
    stochastic_realizations = self.stochastic_variables.sample(N)

    if state_space_names is None:
      state_space_names = list(map('state{}'.format, range(N)))

    if control_space_names is None:
      control_space_names = list(map('control{}'.format, range(N)))

    Xs = [X0]
    Us, Ws = [], []
    compound_rewards = [0.0]

    for wk, policy in zip(stochastic_realizations, policy_functions):
      mypolicy = policy(Xs[-1][_]).ravel()
      immediate_reward, next_state = \
        self.reward_transition_function(Xs[-1], mypolicy, wk)
      compound_rewards.append(immediate_reward)
      Xs.append(next_state)
      Us.append(mypolicy)
      Ws.append(wk)

    Xs = np.array(Xs, dtype=float)
    Us = np.array(Us, dtype=float)
    Ws = np.array(Ws, dtype=float)
    compound_rewards = np.array(compound_rewards, dtype=float).cumsum()

    fig, axes = plt.subplots(1, 4)

    for ax, data, names in zip(axes,
                               (Xs, Us, Ws),
                               (state_space_names,
                                control_space_names,
                                self.stochastic_variable_names)):
      for mydata, myname in zip(data.T, names):
        ax.plot(mydata, '-o', label=myname)
      ax.legend()
      ax.set_xticks(np.arange(len(mydata)))
      ax.grid(True)

    axes[-1].plot(compound_rewards, '-o', label='reward')
    axes[-1].legend()
    axes[-1].set_xticks(np.arange(len(compound_rewards)))
    axes[-1].grid(True)

    plt.show()
