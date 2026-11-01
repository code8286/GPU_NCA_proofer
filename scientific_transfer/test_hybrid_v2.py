import unittest
import numpy as np

from scientific_transfer.materials import Material, WATER, maxwell, ratios
from scientific_transfer.hybrid_v2 import HybridParameters, coefficients, rhs, solve, wall
from scientific_transfer.physics import Parameters, initial_guess, rhs as bridge_rhs
from scientific_transfer.independent_reference import differentiation, homann


class HybridV2Tests(unittest.TestCase):
    def test_zero_loading_recovers_base(self):
        self.assertEqual(set(ratios().values()), {1.0})
        self.assertGreater(ratios(0.01, 0.02)["rho"], 1)
        self.assertGreater(ratios(0.01, 0.02)["k"], 1)

    def test_final_fraction_density_and_heat_capacity(self):
        from scientific_transfer.materials import ZINC, TITANIA
        c = ratios(.02, .03)
        self.assertAlmostEqual(c["rho"]*WATER.density,
                               .95*WATER.density+.02*ZINC.density+.03*TITANIA.density)
        self.assertAlmostEqual(c["rho_cp"]*WATER.density*WATER.heat_capacity,
                               .95*WATER.density*WATER.heat_capacity+.02*ZINC.density*ZINC.heat_capacity+
                               .03*TITANIA.density*TITANIA.heat_capacity)

    def test_invalid_material_fraction_and_parameter_rejected(self):
        for pair in ((-.01, 0), (.1, .01), (np.nan, 0)):
            with self.assertRaises(ValueError):
                ratios(*pair)
        with self.assertRaises(ValueError):
            ratios(base=Material(0, 1, 1, 1, 1))
        with self.assertRaises(ValueError):
            HybridParameters(temperature_ratio=.9)

    def test_maxwell_identity_and_single_particle_limits(self):
        self.assertEqual(maxwell(2, 7, 0), 2)
        self.assertAlmostEqual(maxwell(2, 7, 1), 7)
        c = ratios(.02, 0, zinc=WATER, titania=WATER)
        for key in ("rho", "rho_cp", "k", "sigma", "rho_beta"):
            self.assertAlmostEqual(c[key], 1)

    def test_v2_reduces_to_v1_rhs(self):
        x = np.linspace(0, 10, 100)
        y = initial_guess(x)
        p = dict(magnetic=.3, rotation=.4, buoyancy=.2, prandtl=2)
        np.testing.assert_allclose(rhs(x, y, HybridParameters(**p)), bridge_rhs(x, y, Parameters(**p)), atol=1e-14)

    def test_nonlinear_radiation_conservative_flux(self):
        x = np.linspace(0, 3, 100)
        y = initial_guess(x)
        p = HybridParameters(radiation=.3, temperature_ratio=1.5, phi_zinc=.01)
        dy = rhs(x, y, p)
        absolute_t = 1+.5*y[5]
        k = coefficients(p)["k"]+.3*absolute_t**3
        dk = 3*.3*.5*absolute_t**2
        residual = k*dy[6]+dk*y[6]**2+2*p.prandtl*coefficients(p)["rho_cp"]*y[0]*y[6]
        np.testing.assert_allclose(residual, 0, atol=1e-14)

    def test_wall_flux_includes_radiation(self):
        p = HybridParameters(radiation=.2, temperature_ratio=2)
        y = initial_guess(np.linspace(0, 10, 100))
        self.assertAlmostEqual(wall(y, p)["minus_total_conductivity_theta_p0"], 2.6)

    def test_spectral_polynomial_derivatives(self):
        x, d = differentiation(25, 4)
        np.testing.assert_allclose(d@(x**3), 3*x*x, atol=1e-9)
        np.testing.assert_allclose(d@d@(x**3), 6*x, atol=1e-8)

    def test_independent_homann_and_thermal_agree(self):
        independent = homann()
        numerical = solve(HybridParameters())
        self.assertTrue(independent["accepted"], independent)
        self.assertTrue(numerical["accepted"])
        self.assertAlmostEqual(independent["fpp0"], numerical["wall"]["mu_ratio_fpp0"], delta=1e-5)
        self.assertAlmostEqual(independent["minus_theta_p0"], numerical["wall"]["minus_total_conductivity_theta_p0"], delta=1e-5)


if __name__ == "__main__":
    unittest.main()
