from _types import Immutable, Int, Float
from util import np, frozen

from abc import abstractmethod
from typing import Callable, Sequence
from functools import cached_property

from scipy.special import binom, stirling2, factorial
from scipy import stats


class Distribution(Immutable):

  """
  Base class for distrbutions that can be expressed in closed form.
  Must have known moments and probability density function (PDF).
  """

  @abstractmethod
  def _sample(self, size: Int) -> np.ndarray:
    ...

  def sample(self, *shape: Int) -> np.ndarray:
    """
    Sample from the distribution.
    """
    shape = tuple(map(int, shape))
    ret = self._sample(np.prod(shape, dtype=int))
    if np.isscalar(ret):
      return ret
    return ret.reshape(shape + self.shape)

  @abstractmethod
  @cached_property
  def shape(self):
    ...


class ClosedFormDistribution(Distribution):

  """
  Base class for distrbutions that can be expressed in closed form.
  Must have known moments and probability density function (PDF).
  """

  @abstractmethod
  @cached_property
  def pdf(self, x: Int | Float) -> Float:
    ...

  @abstractmethod
  def _raw_moment(self, order: Int) -> Float:
    """
    Return the raw moments of the distribution.
    """
    ...

  def raw_moment(self, order: Int = 1) -> Float:
    # TODO: change to cache to one that does not allow overwrites

    assert order >= 0
    if order == 0:
      return 1.0
    if not hasattr(self, '__raw_moment_cache'):
      self.__raw_moment_cache = {}
    try:
      return self.__raw_moment_cache[order]
    except KeyError:
      return self.__raw_moment_cache.setdefault(order, self._raw_moment(order))

  def moment(self, order: Int = 1) -> Float:
    assert order >= 0
    if order == 0:
      return 1.0
    if not hasattr(self, '__moment_cache'):
      self.__moment_cache = {}
    try:
      return self.__moment_cache[order]
    except KeyError:
      mean = self.expectation
      # compute from raw moments
      return self.__moment_cache.setdefault(
                order,
                sum(binom(order, k) * (-mean) ** (order - k) * self.raw_moment(k)
                    for k in range(order + 1))
              )

  @cached_property
  def expectation(self) -> Float:
    """
    Return the expectation (mean) of the distribution.
    """
    return self.raw_moment(1)

  @cached_property
  def variance(self) -> Float:
    """
    Return the variance of the distribution.
    """
    return self.raw_moment(2) - self.expectation ** 2

  @cached_property
  def standard_deviation(self) -> Float:
    """
    Return the standard deviation of the distribution.
    """
    return np.sqrt(self.variance)

  def __add__(self, other):
    raise NotImplementedError("Addition is not implemented for distributions.")

  def __neg__(self):
    raise NotImplementedError("Negation is not implemented for distributions.")

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    raise NotImplementedError("Addition is not implemented for distributions.")


class ScalarDistribution(ClosedFormDistribution):

  """
  Non-vectorial distribution.
  """

  def __matmul__(self, other):
    """
    This one should return the product of two distributions, uncorrelated.
    F(x, y) = F0(x) * F1(y)
    """
    raise NotImplementedError

  @property
  def shape(self):
    return ()


class ContinuousDistribution(ScalarDistribution):

  @cached_property
  def pdf(self) -> Callable:
    """
    Return the probability density function of the continuous distribution.
    """
    raise NotImplementedError("PDF must be implemented in subclasses.")

  def _raw_moment(self, order: Int) -> Float:
    raise NotImplementedError("Raw moments must be implemented in subclasses.")

  @abstractmethod
  def ppf(self, q: Float) -> Float:
    ...

  def credible_interval(self, alpha: Float = 0.95) -> tuple[Float, Float]:
    q = (1 - alpha) / 2.0
    return self.ppf(q), self.ppf(1 - q)


class Normal(ContinuousDistribution):

  def __init__(self, mean: Float = 0, std: Float = 1) -> None:
    self.mean = float(mean)
    self.std = float(std)
    assert self.std > 0, "Standard deviation must be positive."

  @cached_property
  def pdf(self) -> Callable:
    """
    Return the probability density function of the normal distribution.
    """
    return lambda x: np.exp(-(x - self.mean) ** 2 / (2 * self.std ** 2)) / (self.std * np.sqrt(2 * np.pi))

  def _sample(self, size: Int) -> np.ndarray:
    """
    Sample from the normal distribution.
    """
    return np.random.normal(self.mean, self.std, size=size)

  def _raw_moment(self, order: Int) -> Float:
    """
    Return the moments of the normal distribution.
    The first moment is the mean, the second is the variance,
    and so on.
    """
    if order == 1:
      return self.mean
    elif order == 2:
      return self.std ** 2 + self.mean ** 2
    else:
      raise NotImplementedError("Higher moments are not implemented for Normal distribution.")

  def ppf(self, q: Float) -> Float:
    return stats.norm(self.mean, self.std).ppf(q)


class LogNormal(ContinuousDistribution):

  @classmethod
  def from_mean_std(cls, mean: Float, std: Float) -> 'LogNormal':
    sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return cls(mu, sigma)

  def __init__(self, mu: Float = 0, sigma: Float = 1) -> None:
    self.mu = float(mu)
    self.sigma = float(sigma)
    assert self.sigma > 0, "Standard deviation must be positive."

  @cached_property
  def pdf(self) -> Callable:
    """
    Return the probability density function of the log-normal distribution.
    """
    return lambda x: np.where(x > 0.0,
                              (np.exp(-(np.log(x) - self.mu) ** 2
                               / (2 * self.sigma ** 2))
                               / (x * self.sigma * np.sqrt(2 * np.pi))),
                              0)

  def _sample(self, size: Int) -> np.ndarray:
    """
    Sample from the log-normal distribution.
    """
    return np.random.lognormal(self.mu, self.sigma, size=size)

  def _raw_moment(self, order: Int) -> Float:
    """
    Return the moments of the log-normal distribution.
    The nth moment is given by exp(n * mean + 0.5 * n^2 * std^2).
    """
    return np.exp(order * self.mu + 0.5 * order ** 2 * self.sigma ** 2)

  def ppf(self, q: Float) -> Float:
    return stats.lognorm(self.mu, self.sigma).ppf(q)


class Uniform(ContinuousDistribution):

  def __init__(self, low: Float = 0, high: Float = 1) -> None:
    self.low = float(low)
    self.high = float(high)
    assert self.high > self.low

  @cached_property
  def pdf(self) -> Callable:
    """
    Return the probability density function of the uniform distribution.
    """
    return lambda x: np.piecewise(
        x, [x < self.low, (x >= self.low) & (x <= self.high), x > self.high],
        [0, 1 / (self.high - self.low), 0]
    )

  def _sample(self, size: Int) -> np.ndarray:
    return np.random.uniform(self.low, self.high, size=size)

  def _raw_moment(self, order: Int) -> Float:
    """
    Return the raw moments of the uniform distribution.
    """
    return ((self.high ** (order + 1) - self.low ** (order + 1)) /
            (self.high - self.low)) / (order + 1)

  def ppf(self, q: Float) -> Float:
    return stats.uniform(self.low, self.high - self.low).ppf(q)


class Beta(ContinuousDistribution):

  def __init__(self, a: Float = 1, b: Float = 1) -> None:
    self.a = float(a)
    self.b = float(b)
    assert self.a > 0 and self.b > 0, "Parameters a and b must be positive."

  @cached_property
  def pdf(self) -> Callable:
    """
    Return the probability density function of the beta distribution.
    """
    return lambda x: np.where((x >= 0) & (x <= 1),
                              (x ** (self.a - 1) * (1 - x) ** (self.b - 1)) /
                              stats.beta(self.a, self.b).pdf(1),
                              0)

  def _sample(self, size: Int) -> np.ndarray:
    """
    Sample from the beta distribution.
    """
    return np.random.beta(self.a, self.b, size=size)

  def _raw_moment(self, order: Int) -> Float:
    if order == 1:
      return self.a / (self.a + self.b)
    return (self.a + order - 1) / (self.a + self.b + order - 1) * self._raw_moment(order - 1)

  def ppf(self, q: Float) -> Float:
    """
    Return the percent point function (PPF) of the beta distribution.
    """
    return stats.beta(self.a, self.b).ppf(q)


class Dirac(ContinuousDistribution):

  def __init__(self, value: Float = 0) -> None:
    self.value = float(value)

  @cached_property
  def pdf(self) -> Callable:
    """
    Return the probability density function of the Dirac delta distribution.
    """
    return lambda x: np.where(x == self.value, np.inf, 0)

  def _sample(self, size: Int) -> np.ndarray:
    return np.full(size, self.value)

  def _raw_moment(self, order: Int) -> Float:
    return self.value ** order

  def ppf(self, q: Float) -> Float:
    """
    The PPF of a Dirac delta distribution is always the value itself.
    """
    return self.value


def as_distribution(dist: ClosedFormDistribution) -> ClosedFormDistribution:
  """
  Ensure that the object is a Distribution.
  If it is not, raise an error.
  """
  if not isinstance(dist, ClosedFormDistribution):
    return Dirac(dist)
  return dist


class DiscreteDistribution(ScalarDistribution):

  @cached_property
  def pdf(self) -> Callable:
    """
    Return the probability density function of the continuous distribution.
    """
    raise NotImplementedError("PDF must be implemented in subclasses.")

  def _raw_moment(self, order: Int) -> Float:
    raise NotImplementedError("Raw moments must be implemented in subclasses.")


class RandInt(DiscreteDistribution):

  def __init__(self, low: Int = 0, high: Int = 1) -> None:
    self.low = int(low)
    self.high = int(high)
    assert self.high > self.low

  @cached_property
  def pdf(self) -> Callable:
    """
    Return the probability mass function of the discrete uniform distribution.
    """
    return lambda x: np.piecewise(
        x,
        [x < self.low, (x >= self.low) & (x < self.high), x >= self.high],
        [0, 1 / (self.high - self.low), 0]
    )

  def _sample(self, size: Int) -> np.ndarray:
    return np.random.randint(self.low, self.high, size=size)

  def _raw_moment(self, order: Int) -> Float:
    return np.mean(np.arange(self.low, self.high) ** order)


class Geometric(DiscreteDistribution):

  def __init__(self, p: Float = 0.5) -> None:
    self.p = float(p)
    assert 0 < self.p < 1, "Probability p must be in (0, 1)."

  @cached_property
  def pdf(self) -> Callable:
    """
    Return the probability mass function of the geometric distribution.
    """
    return lambda x: np.where(x >= 0, (1 - self.p) ** x * self.p, 0)

  def _sample(self, size: Int) -> np.ndarray:
    """
    Sample from the geometric distribution.
    """
    return np.random.geometric(self.p, size=size) - 1

  def _raw_moment(self, order: Int) -> Float:
    """
    Return the raw moments of the geometric distribution.
    The nth moment is given by (1 - p)^n / p * (1 + n).
    """
    return sum( stirling2(order, k)
                * (1 - self.p) ** k * factorial(k) / self.p**k
                for k in range(order + 1) )


class Poisson(DiscreteDistribution):

  def __init__(self, lam: Float = 1) -> None:
    self.lam = float(lam)
    assert self.lam > 0, "Lambda must be positive."

  @cached_property
  def pdf(self) -> Callable:
    """
    Return the probability mass function of the Poisson distribution.
    """
    return lambda x: np.where(x >= 0, (np.exp(-self.lam) * self.lam ** x) / np.math.factorial(x), 0)

  def _sample(self, size: Int) -> np.ndarray:
    return np.random.poisson(self.lam, size=size)

  def _raw_moment(self, order: Int) -> Float:
    return sum( self.lam**i * stirling2(order, i) for i in range(order+1) )


class Bernoulli(DiscreteDistribution):

  def __init__(self, p: Float) -> None:
    self.p = float(p)
    assert 0 <= self.p <= 1, "Probability p must be in [0, 1]."

  @cached_property
  def pdf(self) -> Callable:
    """
    Return the probability mass function of the Bernoulli distribution.
    """
    return lambda x: np.where(x == 1, self.p, np.where(x == 0, 1 - self.p, 0))

  def _sample(self, size: Int) -> np.ndarray:
    return np.random.binomial(1, self.p, size=size)

  def _raw_moment(self, order: Int) -> Float:
    """
    Return the raw moments of the Bernoulli distribution.
    The nth moment is p if n is odd, and 1 - p if n is even.
    """
    return self.p


class VectorialDistribution(Distribution):

  def __init__(self, distributions: Sequence[ClosedFormDistribution]) -> None:
    self.distributions = tuple(map(as_distribution, distributions))
    assert self.shape, "At least one distribution must be provided."
    self.ndim = len(self.distributions)

  @property
  def shape(self):
    return len(self.distributions),

  @cached_property
  def pdf(self):
    raise NotImplementedError("PDF is not implemented for vectorial distributions.")

  def __getitem__(self, index):
    ret = self.distributions[index]
    if isinstance(ret, tuple):
      return VectorialDistribution(ret)
    return ret

  def _sample(self, size: Int) -> np.ndarray:
    """
    Sample from the tensorial distribution.
    """
    return np.stack([dist.sample(size) for dist in self.distributions], axis=-1)

  def _raw_moment(self, order: Int) -> Float:
    """
    Return the raw moments of the vectorial distribution.
    """
    return np.prod([dist.raw_moment(order) for dist in self.distributions])
