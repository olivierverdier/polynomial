# determine what is imported during a `from polynomial import *`
__all__ = ['Polynomial', 'Zero', 'One', 'X']

"""
Classes to model polynomials.
It also defines a Zero and One polynomials
"""

import numpy as np
import numpy.linalg as nl
import functools


def cast_scalars(method):
	"""
	Decorator used to cast a scalar to a polynomial
	"""
	def newMethod(self, other):
		if np.isscalar(other):
			other = Polynomial(other)
		return method(self, other)
	return newMethod

class Polynomial:
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
		if np.isscalar(coeffs):
			coeffs = [coeffs]
		elif not list(coeffs): # empty coeff list
			coeffs = [0]
		self.coeffs = np.array(coeffs)

	def str_power(self, d, X='X'):
		if d == 0:
			return ''
		if d == 1:
			return X
		return X+'^{}'.format(d)

	def __str__(self):
		"""
		Pretty presentation.
		"""
		return ' + '.join(str(coeff)+self.str_power(index) for (index, coeff) in enumerate(self.coeffs[:self.length()]) if coeff != 0)

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
			newcoeffs = np.append(self.coeffs, np.zeros(index-len(self.coeffs)+1))
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
		return type(self)([self[index] + other[index] for index in range(maxLength)])

	__radd__ = __add__

	def __neg__(self):
		"""
		-P
		"""
		return type(self)(-self.coeffs)

	def __pos__(self):
		return type(self)(self.coeffs)

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
		return type(self)(newCoeffs)

	__rmul__ = __mul__

	def __div__(self, scalar):
		return self * (1/scalar)

	__truediv__ = __div__

	def __pow__(self, n):
		"""
		P**n
		"""
		def mul(a,b): return a*b
		return functools.reduce(mul, [self]*n, 1.)

	class ConstantPolynomialError(Exception):
		"""
		Exception for constant polynomials
		"""

	def companion(self):
		"""
		Companion matrix.
		"""
		degree = self.degree()
		if degree == 0:
			raise self.ConstantPolynomialError("Constant polynomials have no companion matrix")
		companion = np.eye(degree, k=-1, dtype=complex)
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
			return nl.eigvals(companion).tolist()

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
		return functools.reduce(simpleMult, reversed(self.coeffs), 0)

	epsilon = 1e-10
	def __bool__(self):
		"""
		Test for difference from zero (up to epsilon)
		"""
		# notice the use of a generator inside the parenthesis
		# the any function will return True for the first True element encountered in the generator
		return any(abs(coeff) > self.epsilon for coeff in self.coeffs)

	__nonzero__ = __bool__ # compatibility Python 2

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
		return type(self)((numpy.arange(len(self.coeffs))*self.coeffs)[1:])

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


# note: The following class is a (bad) example of inheritance.
# it is only here for illustration purpose
class TrigPolynomial (Polynomial):
	"""
	Model for a trigonometric polynomial.
	"""

	def __call__(self, theta):
		from numpy import exp
		return type(self).eval(self, exp(1j*theta))

# just for a cleaner import we delete this decorator
del cast_scalars

Zero = Polynomial([]) # the zero polynomial (extreme case with an empty coeff list)

One = Polynomial([1]) # the unit polynomial

X = Polynomial([0,1])

