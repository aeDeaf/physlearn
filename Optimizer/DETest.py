import unittest

from physlearn.Optimizer import DifferentialEvolution


class DETest(unittest.TestCase):
    def test_create_population_amount(self):
        for i in range(1000):
            with self.subTest(i=i):
                population = DifferentialEvolution.create_population(i, 2, -1, 1)
                self.assertEqual(i, population.shape[0])

    def test_create_population_dim(self):
        for i in range(1000):
            with self.subTest(i=i):
                population = DifferentialEvolution.create_population(1, i, -1, 1)
                self.assertEqual(i, population[0].shape[0])


if __name__ == '__main__':
    unittest.main()
