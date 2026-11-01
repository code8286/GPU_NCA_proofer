"""Declared final-volume mixture convention; NOT the manuscript's printed Eq7."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class Material:
    density: float
    heat_capacity: float
    conductivity: float
    electrical: float
    expansion: float


# Imported Table1 inputs, visually inspected on PDF p11. Not measured here and
# not independently certified physical properties (especially TiO2 electrical).
WATER = Material(997.1, 4179.0, 0.613, 5.5e-6, 21e-5)
ZINC = Material(7140.0, 389.0, 116.0, 1.69e7, 3.5e-5)
TITANIA = Material(4250.0, 686.2, 8.9528, 2.6e6, 0.90e-5)


def maxwell(host, inclusion, fraction):
    return host * (inclusion+2*host+2*fraction*(inclusion-host)) / (
        inclusion+2*host-fraction*(inclusion-host))


def ratios(phi_zinc=0.0, phi_titania=0.0, base=WATER, zinc=ZINC, titania=TITANIA):
    if not all(math.isfinite(v) and 0 <= v for v in (phi_zinc, phi_titania)):
        raise ValueError("Finite nonnegative final volume fractions required")
    total = phi_zinc+phi_titania
    if total > 0.10:
        raise ValueError("Development dilute-mixture domain requires total fraction <=0.10")
    for material in (base, zinc, titania):
        if not all(math.isfinite(v) and v > 0 for v in asdict(material).values()):
            raise ValueError("Positive finite material properties required")
    fractions = (1-total, phi_zinc, phi_titania)
    materials = (base, zinc, titania)
    def average(prop):
        return sum(f*prop(m) for f, m in zip(fractions, materials))
    def sequential(prop):
        # Zn added first to residual host, then TiO2; fractions refer to FINAL
        # mixture. This convention is deliberately order-dependent.
        first = maxwell(prop(base), prop(zinc), phi_zinc/(1-phi_titania))
        return maxwell(first, prop(titania), phi_titania)/prop(base)
    return {"rho": average(lambda m: m.density)/base.density,
            "rho_cp": average(lambda m: m.density*m.heat_capacity)/(base.density*base.heat_capacity),
            "rho_beta": average(lambda m: m.density*m.expansion)/(base.density*base.expansion),
            "mu": (1-total)**(-2.5),
            "k": sequential(lambda m: m.conductivity),
            "sigma": sequential(lambda m: m.electrical)}
