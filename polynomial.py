# -*- coding: UTF-8 -*-
from __future__ import division # to avoid the mess with integer divisions

__all__ = ['Polynomial', 'TrigPolynomial', 'Zero', 'One', 'X']

__version__ = "$Id: polynomial.py 17 2007-09-02 17:50:03Z olivier $"


"""
Classes to model polynomial and trigonometric polynomials.
It also defines a Zero and One polynomials
"""

import numpy
import pylab
from numpy import array


def cast_scalars(method):
  """Decorator used to cast a scalar to a polynomial"""
  def newMethod(self, other):
    if numpy.isscalar(other):
      other = Polynomial(other)
    return method(self, other)
  return newMethod

class Polynomial (object):
  """
  Model class for a polynomial.
  The usual operations (+, -, *, **) are provided
  Comparison between polynomials is defined
  Scalars are automatically cast to polynomials
  Trailing zeros are allowed in the coefficients
  Examples of the variations on the syntax:
    Polynomial(3)
    Polynomial([3,4,1])
    Polynomial([3,4,1,0,0])
    P = 3 + 4*X + X**2
    P(3)  # value at 3
    P[10] # 10th coefficient (zero)

  Remarks:
  1. One may use any ring for the coefficient of this polynomial class
  2. the evaluation method works automatically on arrays
  3. the polynomial is itself a ring, and may be used as coefficients in an array:
    AP = numpy.array([p1,p2])
    numpy.dot(AP, AP)
  (An application could be to compute the characteristic polynomial of a matrix.)
  """
  def __init__(self, coeffs):
    """
    Create a polynomial from a list or array of coefficients
    There may be additional trailing zeros.
    """
    # we allow the creation of polynomials from scalars:
    if numpy.isscalar(coeffs):
      coeffs = [coeffs]
    elif not list(coeffs): # empty coeff list
      coeffs = [0]
    self.coeffs = array(coeffs)

  def __str__(self):
    """Pretty presentation"""
    return ' + '.join("%sX^%d" % (str(coeff), index) for (index, coeff) in enumerate(self.coeffs[:len(self)]))

  def __repr__(self):
    """Make it easy to create a new polynomial from of this output"""
    return "Polynomial(%s)" % str(list(self.coeffs[:len(self)]))

  def __getitem__(self, index):
    """Simulate the [] access and return zero for indices out of range"""
    # note: this method is used in the addition and multiplication operations
    try:
      return self.coeffs[index]
    except IndexError:
      return 0

  def __setitem__(self, index, value):
    """Allow to change an arbitrary coefficient (even out of range)"""
    raise NotImplementedError

  def __len__(self):
    """"Length" of the polynomial (degree + 1)"""
    for index, coeff in enumerate(reversed(list(self.coeffs))):
      if coeff != 0:
        break
    return len(self.coeffs)-index

  def degree(self):
    """Degree of the polynomial (biggest non zero coefficient)"""
    return len(self) - 1

  @cast_scalars
  def __add__(self, other):
    """P1 + P2"""
    maxLength = max(len(self), len(other))
    return Polynomial([self[index] + other[index] for index in range(maxLength)])

  __radd__ = __add__

  def __neg__(self):
    """-P"""
    return Polynomial(-self.coeffs)

  def __sub__(self, other):
    """P1 - P2"""
    return self + (-other)

  def __rsub__(self, other):
    return -(self - other)

  @cast_scalars
  def __mul__(self, other):
    """P1 * P2"""
    # length of the resulting polynomial:
    length = len(self) + len(other)
    newCoeffs = [numpy.sum(self[j]*other[i-j] for j in range(i+1)) for i in range(length)]
    return Polynomial(newCoeffs)

  __rmul__ = __mul__

  def __pow__(self, n):
    """P**n"""
    def mul(a,b): return a*b
    return reduce(mul, [self]*n)

  class ConstantPolynomialError(Exception):
    """Exception for constant polynomials"""

  def companion(self):
    """Companion matrix"""
    from numpy import eye
    degree = self.degree()
    if degree == 0:
      raise self.ConstantPolynomialError("Constant polynomials have no companion matrix")
    companion = eye(degree, k=-1, dtype=complex)
    companion[:,-1] = -self.coeffs[:degree]/self.coeffs[degree]
    return companion

  def zeros(self):
    """Compute the zeros via the companion matrix"""
    try:
      companion = self.companion()
    except self.ConstantPolynomialError:
      return []
    else:
      return list(numpy.linalg.eigvals(companion))

  resolution = 200
  def plot(self, a, b):
    """Plot the polynomial between a and b"""
    xx = pylab.linspace(a, b, self.resolution)
    # here we use the fact that evaluation works on arrays:
    pylab.plot(xx, self(xx))

  def __call__(self, x):
    """
    Numerical value of the polynomial at x
      x may be a scalar or an array
    """
    # note: the following technique certainly obfuscates the code...
    # just take it as an example of dynamic functions (called "closures")

    # Notice how the following "sub-function" depends on x:
    def simpleMult(a, b):
      return a*x + b
    # the third argument is to take care of constant polynomials!
    return reduce(simpleMult, reversed(self.coeffs), 0)

  epsilon = 1e-10
  def __nonzero__(self):
    """Test for difference from zero (up to epsilon)"""
    return any(abs(coeff) > self.epsilon for coeff in self.coeffs)

  def __eq__(self, other):
    """P1 == P2"""
    return not (self - other)

  def __ne__(self, other):
    """P1 != P2"""
    return not (self == other)

  def differentiate(self):
    """Symbolic differentiation"""
    return Polynomial((numpy.arange(len(self.coeffs))*self.coeffs)[1:])

  # this one is for fun only
  enlarge_coeff = .2
  def plot_zeros(self):
    """Plot the zeros in the complex plane."""
    zeros = self.zeros()
    from numpy import real, imag, diff
    from pylab import axis
    plot(real(zeros), imag(zeros), 'o')

    # now we enlarge the graph a bit
    zone = array(axis()).reshape(2,2)
    padding = self.enlarge_coeff * diff(zone)
    zone += numpy.hstack((-padding, padding))
    axis(zone.reshape(-1))    

  # this is for playing around
  @classmethod
  def random(cls, N=10, comp=False):
    """
    Create a random polynomial of degree <= N
    Coefficients are in [-.5, .5] (+1j[-.5, .5])
      comp – whether the polynomial may have complex coefficients
    """
    def random_coeffs(size):
      return numpy.random.rand(size) - .5
    coeffs = random_coeffs(numpy.random.randint(N))
    if comp:
      coeffs = coeffs + 1j*random_coeffs(len(coeffs))
    return cls(coeffs)

  random_real = random

  @classmethod
  def random_complex(cls, *args):
    """Create a random complex polynomial"""
    return cls.random(comp=True, *args)

  def test_zeros(self):
    """Check that we are really zero on the zeros (up to epsilon)"""
    return all(self(z) < self.epsilon for z in self.zeros())

class TrigPolynomial (Polynomial):
  """Model for a trigonometric polynomial"""

  def __call__(self, theta):
    from numpy import exp
    return Polynomial.eval(self, exp(1j*theta))

del cast_scalars

for cls in (Polynomial, TrigPolynomial):
  cls.eval = cls.__call__ # aliases eval = __call__
del cls

Zero = Polynomial([]) # the zero polynomial (extreme case with an empty coeff list)

One = Polynomial([1]) # the unit polynomial

X = Polynomial([0,1])

if __name__ != "__main__": # when importing:
  # a bit of cleaning up
  del array, numpy, pylab, division
else: # here we do some tests that will not be run when importing this module
  assert not Zero
  assert not Polynomial([0,0])
  assert Zero.degree() == 0
  assert One.degree() == 0
  assert not One.zeros(), "Constants have no zero"
  assert One == Polynomial([1,0,0,0])
  assert One == Polynomial(1), "Creation from scalars"

  assert X.degree() == 1
  assert (X+X).degree()==1
  assert len(One) == 1
  assert len(X) == 2

  assert Polynomial((0,)) != Polynomial([0,2])

  r = numpy.random.random()
  assert Zero(r) == 0
  assert One(r) == 1
  assert X(r) == r

  assert Zero == 0
  assert One == 1
  assert not 2*Zero != 3*Zero

  assert X.differentiate() == 1
  assert (X**2).differentiate() == 2*X
  assert (X-1)*(X+1) == X**2 - 1
  
  myLength = 10
  myArray = numpy.random.rand(myLength)

  for p in (Zero, One, X, Polynomial.random_real(), Polynomial.random_complex()):
    assert p * One == p
    assert p * Zero == Zero
    assert p + Zero == p
    assert p == p
    # scalar muliplications:
    assert 1 * p == p
    assert p * 1 == p
    assert p * 0 == Zero
    assert 0 * p == Zero
    assert 2 * p == p * 2
    assert p**2 == p * p
    assert (p * p) * p == p * (p * p)
    assert (p + p) + p == p + (p + p)
    assert p-p == Zero
    assert p+p == 2*p
    if p != Zero:
      assert 2*p != 3*p
      assert p
    else:
      assert 2*p == 3*p
      assert not p
    assert p + One == p + 1
    assert p + 0 == p
    assert p.test_zeros()
    assert isinstance(p.zeros(), list) # a list, not an array
    assert isinstance(p(myArray), numpy.ndarray), "Evaluation should work on arrays"
  
  # tests with some specific polynomials
  p1 = Polynomial([2.,0,3.,0]) # 2 + 3x^2
  p2 = Polynomial([3.,2.]) #3 + 2x

  assert p1.degree() == 2
  assert p1[4] == 0, "Index out of range should return zero"
  assert p1[0] == 2
  assert p1+p2 == Polynomial([5,2,3])
  assert p1.differentiate() == Polynomial([0,6])
  assert p1(2) == 14
  assert p1*p2 == Polynomial([6., 4., 9., 6.])
  tp = TrigPolynomial([-1,1])
  assert abs(tp(numpy.pi/2) - (-1+1j)) < 1e-10
