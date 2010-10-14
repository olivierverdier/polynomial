# -*- coding: UTF-8 -*-
from __future__ import division # to avoid the mess with integer divisions

# determine what is imported during a `from polynomial import *`
__all__ = ['Polynomial', 'TrigPolynomial', 'Zero', 'One', 'X']

"""
Classes to model polynomial and trigonometric polynomials.
It also defines a Zero and One polynomials
"""

from numpy import array,isscalar


def cast_scalars(method):
	"""
	Decorator used to cast a scalar to a polynomial
	"""
	def newMethod(self, other):
		if isscalar(other):
			other = Polynomial(other)
		return method(self, other)
	return newMethod

class Polynomial (object):
	"""
	Model class for a polynomial.
	
	Features
	========
	
	* The usual operations (``+``, ``-``, ``*``, ``**``) are provided
	* Comparison between polynomials is defined
	* Scalars are automatically cast to polynomials
	* Trailing zeros are allowed in the coefficients
	
	Examples
	========
	::
	
		Polynomial(3)
		Polynomial([3,4,1])
		Polynomial([3,4,1,0,0])
		P = 3 + 4*X + X**2
		P(3)	# value at 3
		P[10] # 10th coefficient (zero)
		P[10] = 1 # setting the tenth coefficient
	"""
	def __init__(self, coeffs):
		"""
		Create a polynomial from a list or array of coefficients
		There may be additional trailing zeros.
		"""
		# we allow the creation of polynomials from scalars:
		if isscalar(coeffs):
			coeffs = [coeffs]
		elif not list(coeffs): # empty coeff list
			coeffs = [0]
		self.coeffs = array(coeffs)

	def __str__(self):
		"""
		Pretty presentation.
		"""
		return ' + '.join("%sX^%d" % (str(coeff), index) for (index, coeff) in enumerate(self.coeffs[:self.length()]) if coeff != 0)

	def __repr__(self):
		"""
		Make it easy to create a new polynomial from of this output.
		"""
		return "%s(%s)" % (type(self).__name__, str(list(self.coeffs[:self.length()])))

	def __getitem__(self, index):
		"""
		Simulate the [] access and return zero for indices out of range.
		"""
		# note: this method is used in the addition and multiplication operations
		try:
			return self.coeffs[index]
		except IndexError:
			return 0.

	def __setitem__(self, index, value):
		"""
		Change an arbitrary coefficient (even out of range)
		"""
		try:
			self.coeffs[index] = value
		except IndexError:
			from numpy import append,zeros
			newcoeffs = append(self.coeffs, zeros(index-len(self.coeffs)+1))
			newcoeffs[index] = value
			self.coeffs = newcoeffs

	def length(self):
		"""
		"Length" of the polynomial (degree + 1)
		"""
		for index, coeff in enumerate(reversed(list(self.coeffs))):
			if coeff != 0:
				break
		return len(self.coeffs)-index

	def degree(self):
		"""
		Degree of the polynomial (biggest non zero coefficient).
		"""
		return self.length() - 1

	@cast_scalars
	def __add__(self, other):
		"""
		P1 + P2
		"""
		maxLength = max(self.length(), other.length())
		return Polynomial([self[index] + other[index] for index in range(maxLength)])

	__radd__ = __add__

	def __neg__(self):
		"""
		-P
		"""
		return Polynomial(-self.coeffs)

	def __pos__(self):
		return Polynomial(self.coeffs)

	def __sub__(self, other):
		"""
		P1 - P2
		"""
		return self + (-other)

	def __rsub__(self, other):
		return -(self - other)

	@cast_scalars
	def __mul__(self, other):
		"""
		P1 * P2
		"""
		# length of the resulting polynomial:
		length = self.length() + other.length()
		newCoeffs = [sum(self[j]*other[i-j] for j in range(i+1)) for i in range(length)]
		return Polynomial(newCoeffs)

	__rmul__ = __mul__

	def __div__(self, scalar):
		return self * (1/scalar)

	__truediv__ = __div__

	def __pow__(self, n):
		"""
		P**n
		"""
		def mul(a,b): return a*b
		return reduce(mul, [self]*n, 1.)

	class ConstantPolynomialError(Exception):
		"""
		Exception for constant polynomials
		"""

	def companion(self):
		"""
		Companion matrix.
		"""
		from numpy import eye
		degree = self.degree()
		if degree == 0:
			raise self.ConstantPolynomialError("Constant polynomials have no companion matrix")
		companion = eye(degree, k=-1, dtype=complex)
		companion[:,-1] = -self.coeffs[:degree]/self.coeffs[degree]
		return companion

	def zeros(self):
		"""
		Compute the zeros via the companion matrix.
		"""
		try:
			companion = self.companion()
		except self.ConstantPolynomialError:
			if self: # non zero
				return []
			else:
				raise self.ConstantPolynomialError("The zero polynomial has infinitely many zeroes")
		else:
			from numpy.linalg import eigvals
			return eigvals(companion).tolist()

	# this is a default class parameter
	resolution = 200
	def plot(self, a, b):
		"""
		Plot the polynomial between a and b.
		"""
		from numpy import linspace
		from pylab import plot
		xx = linspace(a, b, self.resolution)
		# here we use the fact that evaluation works on arrays:
		plot(xx, self(xx))

	def __call__(self, x):
		"""
		Numerical value of the polynomial at x
			x may be a scalar or an array
		"""
		# note: the following technique certainly obfuscates the code...
		#
		# Notice how the following "sub-function" depends on x:
		def simpleMult(a, b): return a*x + b
		# the third argument is to take care of constant polynomials!
		return reduce(simpleMult, reversed(self.coeffs), 0)

	epsilon = 1e-10
	def __nonzero__(self):
		"""
		Test for difference from zero (up to epsilon)
		"""
		# notice the use of a generator inside the parenthesis
		# the any function will return True for the first True element encountered in the generator
		return any(abs(coeff) > self.epsilon for coeff in self.coeffs)

	def __eq__(self, other):
		"""
		P1 == P2
		"""
		return not (self - other)

	def __ne__(self, other):
		"""
		P1 != P2
		"""
		return not (self == other)

	def differentiate(self):
		"""
		Symbolic differentiation
		"""
		return Polynomial((numpy.arange(len(self.coeffs))*self.coeffs)[1:])

	# this one is for fun only
	enlarge_coeff = .2
	def plot_zeros(self, **kwargs):
		# note the **kwargs which are passed on to the plot function
		"""
		Plot the zeros in the complex plane.
		"""
		zeros = self.zeros()
		from numpy import real, imag, diff, hstack
		from pylab import axis, plot
		plot(real(zeros), imag(zeros), '+', markersize=10, **kwargs)

		# now we enlarge the graph a bit
		zone = array(axis()).reshape(2,2)
		padding = self.enlarge_coeff * diff(zone)
		zone += hstack((-padding, padding))
		axis(zone.reshape(-1))		


# note: The following class is a (bad) exampmle of inheritance.
# it is only here for illustration purpose
class TrigPolynomial (Polynomial):
	"""
	Model for a trigonometric polynomial.
	"""

	def __call__(self, theta):
		from numpy import exp
		return Polynomial.eval(self, exp(1j*theta))

# just for a cleaner import we delete this decorator
del cast_scalars

for cls in (Polynomial, TrigPolynomial):
	cls.eval = cls.__call__ # aliases eval = __call__
del cls

Zero = Polynomial([]) # the zero polynomial (extreme case with an empty coeff list)

One = Polynomial([1]) # the unit polynomial

X = Polynomial([0,1])

# now come the tests
# should always be put *in a different file*!

# class method for testing
@classmethod
def random(cls, N=10, real=False):
	"""
	Create a random polynomial of (random) degree <= N
	Coefficients are in [-.5, .5] (+1j[-.5, .5])
		real – whether the polynomial should have only real coefficients
	"""
	def random_coeffs(size):
		return numpy.random.random([size]) - .5
	degree = numpy.random.randint(N)
	coeffs = random_coeffs(degree) + 1j*random_coeffs(degree)
	if real:
		coeffs = coeffs.real
	return cls(coeffs)

# add this class method dynamically
Polynomial.random = random

import numpy

from nose.tools import raises

def test_Zero_One_X():
	assert not Zero
	assert not Polynomial([0,0])
	assert Zero.degree() == 0
	assert One.degree() == 0
	assert not One.zeros(), "Constants have no zero"
	assert One == Polynomial([1,0,0,0])
	assert One == Polynomial(1), "Creation from scalars"

	assert X.degree() == 1
	assert (X+X).degree()==1
	assert One.length() == 1
	assert X.length() == 2

	r = numpy.random.random()
	assert Zero(r) == 0
	assert One(r) == 1
	assert X(r) == r

	assert Zero == 0
	assert One == 1

	assert not Zero
	assert One

	assert not 2*Zero != 3*Zero

	assert X.differentiate() == 1
	assert (X**2).differentiate() == 2*X
	assert (X-1)*(X+1) == X**2 - 1

	assert Polynomial((0,)) != Polynomial([0,2])




class Harness(object):
	
	def test_zero_one(self):
		p = self.p
		assert p * One == p
		assert p * Zero == Zero
		assert p + Zero == p
		
	
	def test_power(self):
		p = self.p
		assert p**0 == 1
		assert p**1 == p
		assert p**2 == p*p
		
	def test_equality(self):
		p = self.p
		assert p == p
		assert p + 1 != p
	
	def test_scalar_mul(self):
		"""scalar muliplications"""
		p = self.p
		assert 1 * p == p
		assert p * 1 == p
		assert p * 0 == Zero
		assert 0 * p == Zero
		assert 2 * p == p * 2
		assert 2 * (p/2) == p
		assert (2*p == 3*p) == (p == Zero) # 2p == 3p <==> p == 0
		assert (2*p != 3*p) == (p != Zero) # 2p != 3p <==> p != 0
	
	def test_operations(self):
		p = self.p
		assert (p * p) * p == p * (p * p)
		assert (p + p) + p == p + (p + p)
		assert p-p == Zero
	
	def test_unitary(self):
		"""unitary operations"""
		p = self.p
		assert p + (-p) == 0
		assert p == +p
		assert p == -(-p)
	
	def test_scalar_add(self):
		p = self.p
		assert p+p == 2*p
		assert p + One == p + 1
		assert p + 0 == p
		assert (p+2) -2 == p
		
	
	def test_array_evaluation(self):
		"Evaluation should work on arrays"
		self.myLength = 10
		self.myArray = numpy.random.rand(self.myLength)
		assert isinstance(self.p(self.myArray), numpy.ndarray)

	def test_zeros(self):
		def test_zeros(self):
			"""
			Check that we are really zero on the zeros (up to epsilon)
			"""
			if self: # if p != 0
				return all(self(z) < self.epsilon for z in self.zeros())
			else:
				return True
		assert test_zeros(self.p)

	def test_zeros_is_list(self):
		if self.p:
			assert isinstance(self.p.zeros(), list) # a list, not an array

class Test_Zero(Harness):
	def setUp(self):
		self.p = Zero

class Test_One(Harness):
	def setUp(self):
		self.p = One

class Test_X(Harness):
	def setUp(self):
		self.p = X

class Test_Random(Harness):
	def setUp(self):
		self.p = Polynomial.random(real=True)

class Test_RandomComplex(Harness):
	def setUp(self):
		self.p = Polynomial.random()

@raises(Polynomial.ConstantPolynomialError)
def test_Zero_zeros():
	"""Asking for zeros of Zero raises an exception"""
	Zero.zeros()

class Test_Simple(object):
	"""tests with some specific polynomials"""
	def setUp(self):	
		self.p1 = Polynomial([2.,0,3.,0]) # 2 + 3x^2
		self.p2 = Polynomial([3.,2.]) #3 + 2x

	def test_operations(self):
		p1 = self.p1
		p2 = self.p2
		assert p1.degree() == 2
		assert p1[4] == 0, "Index out of range should return zero"
		assert p1[0] == 2
		assert p1+p2 == Polynomial([5,2,3])
		assert p1.differentiate() == Polynomial([0,6])
		assert p1(2) == 14
		assert p1*p2 == Polynomial([6., 4., 9., 6.])


def test_trigpolynomial():
	raise Exception
	tp = TrigPolynomial([-1,1])
	assert abs(tp(numpy.pi/2) - (-1+1j)) < 1e-10
test_trigpolynomial.__test__ = False # don't test this...
	
	
