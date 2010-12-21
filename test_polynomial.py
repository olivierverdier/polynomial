# -*- coding: UTF-8 -*-
from __future__ import division

import numpy

from polynomial import Polynomial, Zero, X, One

# now come the tests
# should always be put *in a different file*!

# class method for testing
@classmethod
def random(cls, N=10, real=False):
	"""
	Create a random polynomial of (random) degree <= N
	Coefficients are in [-.5, .5] (+1j[-.5, .5])
		real - whether the polynomial should have only real coefficients
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
	assert One.degree() == 0
	assert X.degree() == 1

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

def test_simple_operations():
	"""tests with some specific polynomials"""
	p1 = Polynomial([2.,0,3.,0]) # 2 + 3x^2
	p2 = Polynomial([3.,2.]) #3 + 2x
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


