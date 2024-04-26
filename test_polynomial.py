import numpy as np
import numpy.testing as npt

import unittest

from polynomial import Polynomial, Zero, X, One

# now come the tests
# should always be put *in a different file*!

rng = np.random.default_rng()

def get_random_polynomial(rng, N=10, real=False):
	"""
	Create a random polynomial of (random) degree <= N
	Coefficients are in [-.5, .5] (+1j[-.5, .5])
		real - whether the polynomial should have only real coefficients
	"""
	degree = rng.integers(N)
	coeff_mat = [rng.random(degree) - .5 for i in range(2)]
	if real:
		coeffs = coeff_mat[0]
	else:
		 coeffs = coeff_mat[0] + 1j*coeff_mat[1]
	return Polynomial(coeffs)

def test_init():
	for p in [Polynomial(), Polynomial([]), Polynomial(np.array([]))]:
		assert p == Zero
		assert p.length() == 1


class Test_ZeroOne(unittest.TestCase):
	def test(self):
		self.assertFalse(Zero)
		self.assertFalse(Polynomial([0,0]))
		self.assertEqual(Zero.degree(), 0)
		self.assertEqual(One.degree(), 0)
		self.assertFalse(One.zeros(), "Constants have no zero")
		self.assertEqual(One, Polynomial([1,0,0,0]))
		self.assertEqual(One, Polynomial(1), "Creation from scalars")

		self.assertEqual(X.degree(), 1)
		self.assertEqual((X+X).degree(), 1)
		self.assertEqual(One.degree(), 0)
		self.assertEqual(X.degree(), 1)

		r = rng.random()
		self.assertEqual(Zero(r), 0)
		self.assertEqual(One(r), 1)
		self.assertEqual(X(r), r)

		self.assertEqual(Zero, 0)
		self.assertEqual(One, 1)

		self.assertFalse(Zero)
		self.assertTrue(One)

		self.assertFalse(2*Zero != 3*Zero)

		self.assertEqual(X.differentiate(), 1)
		self.assertEqual((X**2).differentiate(), 2*X)
		self.assertEqual((X-1)*(X+1), X**2 - 1)

		self.assertNotEqual(Polynomial((0,)), Polynomial([0,2]))




class Test_One(unittest.TestCase):
	def setUp(self):
		self.p = One

	def test_zero_one(self):
		p = self.p
		self.assertEqual(p * One, p)
		self.assertEqual(p * Zero, Zero)
		self.assertEqual(p + Zero, p)


	def test_power(self):
		p = self.p
		self.assertEqual(p**0, 1)
		self.assertEqual(p**1, p)
		self.assertEqual(p**2, p*p)

	def test_equality(self):
		p = self.p
		self.assertEqual(p, p)
		self.assertNotEqual(p + 1, p)

	def test_scalar_mul(self):
		"""scalar muliplications"""
		p = self.p
		self.assertEqual(1 * p, p)
		self.assertEqual(p * 1, p)
		self.assertEqual(p * 0, Zero)
		self.assertEqual(0 * p, Zero)
		self.assertEqual(2 * p, p * 2)
		self.assertEqual(2 * (p/2), p)
		self.assertEqual((2*p == 3*p), (p == Zero)) # 2p == 3p <==> p, 0
		self.assertEqual((2*p != 3*p), (p != Zero)) # 2p != 3p <==> p != 0

	def test_operations(self):
		p = self.p
		self.assertEqual((p * p) * p, p * (p * p))
		self.assertEqual((p + p) + p, p + (p + p))
		self.assertEqual(p-p, Zero)

	def test_unitary(self):
		"""unitary operations"""
		p = self.p
		self.assertEqual(p + (-p), 0)
		self.assertEqual(p, +p)
		self.assertEqual(p, -(-p))

	def test_scalar_add(self):
		p = self.p
		self.assertEqual(p+p, 2*p)
		self.assertEqual(p + One, p + 1)
		self.assertEqual(p + 0, p)
		self.assertEqual((p+2) -2, p)
		self.assertEqual((2-p)+p, 2)


	def test_array_evaluation(self):
		"Evaluation should work on arrays"
		self.myLength = 10
		self.myArray = rng.random(self.myLength)
		self.assertIsInstance(self.p(self.myArray), np.ndarray)

	def test_zeros(self):
		if self.p:
			npt.assert_array_almost_equal(self.p(np.array(self.p.zeros())), 0)

	def test_zeros_is_list(self):
		if self.p:
			self.assertIsInstance(self.p.zeros(), list) # a list, not an array
	
	def test_set_coeff(self):
		N = 10
		p = Polynomial(self.p.coeffs)
		p[N] = 20.
		self.assertGreaterEqual(p.length(), N)

class Test_Zero(Test_One):
	def setUp(self):
		self.p = Zero

class Test_X(Test_One):
	def setUp(self):
		self.p = X

class Test_Random(Test_One):
	def setUp(self):
		self.p = get_random_polynomial(rng, real=True)

class Test_RandomComplex(Test_One):
	def setUp(self):
		self.p = get_random_polynomial(rng)

class Test_Simple(unittest.TestCase):
	def test_Zero_zeros(self):
		"""Asking for zeros of Zero raises an exception"""
		with self.assertRaises(Polynomial.ConstantPolynomialError):
			Zero.zeros()

	def test_simple_operations(self):
		"""tests with some specific polynomials"""
		p1 = Polynomial([2.,0,3.,0]) # 2 + 3x^2
		self.assertEqual(str(p1), '2.0 + 3.0X^2')
		self.assertEqual(repr(p1), 'Polynomial([2.0, 0.0, 3.0])')
		p2 = Polynomial([3.,2.]) #3 + 2x
		self.assertEqual(str(p2), '3.0 + 2.0X')
		self.assertEqual(p1.degree(), 2)
		self.assertEqual(p1[4], 0, "Index out of range should return zero")
		self.assertEqual(p1[0], 2)
		self.assertEqual(p1+p2, Polynomial([5,2,3]))
		self.assertEqual(p1.differentiate(), Polynomial([0,6]))
		self.assertEqual(p1(2), 14)
		self.assertEqual(p1*p2, Polynomial([6., 4., 9., 6.]))


# if __name__ == '__main__':
# 	unittest.main()
