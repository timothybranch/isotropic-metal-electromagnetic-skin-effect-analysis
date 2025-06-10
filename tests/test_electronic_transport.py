import unittest
import numpy as np
from Libraries.electronic_transport import (
    q_prime_func,
    conductivity_relaxation,
    nonlocality_term_3D_isotropic_func,
    cond_spectrum_3D_isotropic
)

class TestElectronicTransport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.resistivity_residual_DC = 100*1e-9*1e-2
        cls.cond_DC = 1/cls.resistivity_residual_DC

        cls.v_F = 1e6
        cls.gamma_MR = 1e11
        cls.l_MR = cls.v_F / cls.gamma_MR

        cls.num_ql = 200
        cls.ql_min = -3
        cls.ql_max = 3
        cls.ql = np.logspace(cls.ql_min, cls.ql_max, cls.num_ql)  # Dimensionless product of wavevector and mean free path
        cls.q = cls.ql/cls.l_MR

        cls.num_freq = 100
        cls.freq_min = 10
        cls.freq_max = 15
        cls.freq = np.logspace(cls.freq_min, cls.freq_max, cls.num_freq)
        cls.omega = 2*np.pi*cls.freq
        # cls.l_MR = 1.0
        # cls.gamma_MR = 0.5
        # cls.resistivity_residual_DC = 0.01
        # cls.q_prime = np.array([[1.0, 2.0], [3.0, 4.0]])


    def test_q_prime_func(self):
        result = q_prime_func(self.q, self.omega, self.l_MR, self.gamma_MR)
        self.assertEqual(result.shape, (self.num_ql, self.num_freq))

    def test_conductivity_relaxation(self):
        result = conductivity_relaxation(self.cond_DC, self.omega, self.gamma_MR)
        self.assertEqual(result.shape, (self.num_freq,))

    def test_nonlocality_term_3D_isotropic_func(self):
        q_prime = q_prime_func(self.q, self.omega, self.l_MR, self.gamma_MR)
        result = nonlocality_term_3D_isotropic_func(q_prime)
        self.assertEqual(result.shape, (self.num_ql, self.num_freq))

    def test_cond_spectrum_3D_isotropic_func(self):
        result = cond_spectrum_3D_isotropic(self.q, self.omega, self.l_MR, self.gamma_MR, self.resistivity_residual_DC)
        self.assertEqual(result.shape, (self.num_ql, self.num_freq))

if __name__ == '__main__':
    unittest.main()