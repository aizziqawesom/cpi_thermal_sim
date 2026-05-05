"""
src/cpi_model.py
CPI Thermal Stress Simulation Engine
=====================================
Analytical and semi-analytical models for chip-package interaction (CPI)
thermal stress assessment in flip-chip BGA packages.

Physics implemented:
  1. Thermal field: 1D steady-state through package stack (Fourier conduction)
  2. Thermo-mechanical stress: CTE mismatch → interface shear & peel stress
     (Suhir's analytical beam-on-elastic-foundation model)
  3. Solder joint fatigue: Coffin-Manson modified (Engelmaier model)
  4. Warpage: bimetal beam approximation (Timoshenko)
  5. Parametric DOE sweep across substrate CTE, underfill modulus, die size

References:
  - Suhir, E. (1986). Stresses in bi-metal thermostats. J. Appl. Mech.
  - Engelmaier, W. (1983). Fatigue life of leadless chip carrier solder joints.
  - Timoshenko, S. (1925). Analysis of bi-metal thermostats. J. Opt. Soc. Am.
  - JEDEC JESD22-A104: Temperature Cycling Standard
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Material Library
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Material:
    """Thermomechanical material properties."""
    name: str
    E: float          # Young's modulus [GPa]
    nu: float         # Poisson's ratio [-]
    CTE: float        # Coefficient of thermal expansion [ppm/°C]
    k: float          # Thermal conductivity [W/m·K]
    rho_Cp: float     # Volumetric heat capacity [J/m³·K]
    color: str = '#888888'

    @property
    def E_pa(self) -> float:
        return self.E * 1e9  # Pa

    @property
    def CTE_si(self) -> float:
        return self.CTE * 1e-6  # /°C


# Industry-standard material database (from literature/datasheets)
MATERIALS = {
    'silicon': Material(
        name='Silicon Die', E=130.0, nu=0.28,
        CTE=2.8, k=148.0, rho_Cp=1.63e6,
        color='#4a90d9'
    ),
    'organic_substrate': Material(
        name='Organic Substrate (BT)', E=25.0, nu=0.39,
        CTE=17.0, k=0.35, rho_Cp=1.5e6,
        color='#8B6914'
    ),
    'low_cte_substrate': Material(
        name='Low-CTE Substrate', E=30.0, nu=0.35,
        CTE=8.0, k=0.5, rho_Cp=1.5e6,
        color='#5a8a4a'
    ),
    'sac305_solder': Material(
        name='SAC305 Solder (SnAgCu)', E=45.0, nu=0.36,
        CTE=21.0, k=58.0, rho_Cp=1.67e6,
        color='#c0c0c0'
    ),
    'underfill_std': Material(
        name='Standard Underfill (Epoxy)', E=8.5, nu=0.35,
        CTE=26.0, k=0.7, rho_Cp=1.8e6,
        color='#e8c87a'
    ),
    'underfill_low_cte': Material(
        name='Low-CTE Underfill', E=9.5, nu=0.35,
        CTE=18.0, k=0.85, rho_Cp=1.8e6,
        color='#d4a843'
    ),
    'mold_compound': Material(
        name='Epoxy Mold Compound', E=22.0, nu=0.30,
        CTE=8.5, k=0.8, rho_Cp=1.9e6,
        color='#3d2b1f'
    ),
    'pcb_fr4': Material(
        name='PCB FR4', E=22.0, nu=0.28,
        CTE=18.0, k=0.3, rho_Cp=1.7e6,
        color='#2d5a1b'
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Package Geometry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PackageGeometry:
    """Flip-chip BGA package dimensions."""
    die_size: float           # Die edge length [mm]
    die_thickness: float      # Die thickness [mm]
    substrate_size: float     # Substrate edge length [mm]
    substrate_thickness: float # Substrate thickness [mm]
    bump_pitch: float         # Solder bump pitch [µm]
    bump_height: float        # Solder bump standoff [µm]
    bump_diameter: float      # Bump diameter [µm]
    underfill_thickness: float # Underfill layer thickness [µm]
    mold_thickness: float     # Mold compound thickness [mm] (0 = no mold)

    @property
    def die_area(self) -> float:
        return self.die_size ** 2  # mm²

    @property
    def n_bumps_edge(self) -> int:
        return int(self.die_size * 1000 / self.bump_pitch)

    @property
    def n_bumps_total(self) -> int:
        return self.n_bumps_edge ** 2

    @property
    def DNP_max(self) -> float:
        """Maximum distance to neutral point [mm] — die corner."""
        return self.die_size * np.sqrt(2) / 2


# Standard package configurations
PACKAGES = {
    'fcbga_small': PackageGeometry(
        die_size=10.0, die_thickness=0.3,
        substrate_size=17.0, substrate_thickness=0.8,
        bump_pitch=150, bump_height=80, bump_diameter=90,
        underfill_thickness=80, mold_thickness=0.0
    ),
    'fcbga_large': PackageGeometry(
        die_size=20.0, die_thickness=0.5,
        substrate_size=27.0, substrate_thickness=1.0,
        bump_pitch=150, bump_height=80, bump_diameter=90,
        underfill_thickness=80, mold_thickness=0.0
    ),
    'hbm_stack': PackageGeometry(
        die_size=7.0, die_thickness=0.1,
        substrate_size=10.0, substrate_thickness=0.5,
        bump_pitch=55, bump_height=30, bump_diameter=35,
        underfill_thickness=30, mold_thickness=0.2
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Thermal Loading (JEDEC profiles)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThermalProfile:
    """JEDEC temperature cycling profile."""
    name: str
    T_min: float    # °C
    T_max: float    # °C
    ramp_rate: float  # °C/min
    dwell_min: int    # minutes per extreme
    T_ref: float = 25.0  # stress-free reference temp [°C]

    @property
    def delta_T(self) -> float:
        return self.T_max - self.T_min

    @property
    def T_mean(self) -> float:
        return (self.T_max + self.T_min) / 2

    def time_array(self, n_cycles: int = 1) -> np.ndarray:
        """Generate time array for temperature profile [minutes]."""
        ramp_time = self.delta_T / self.ramp_rate
        cycle_time = 2 * ramp_time + 2 * self.dwell_min
        t = []
        T = []
        t_cur = 0.0
        for _ in range(n_cycles):
            t.extend([t_cur, t_cur + self.dwell_min])
            T.extend([self.T_min, self.T_min])
            t_cur += self.dwell_min
            t.extend([t_cur, t_cur + ramp_time])
            T.extend([self.T_min, self.T_max])
            t_cur += ramp_time
            t.extend([t_cur, t_cur + self.dwell_min])
            T.extend([self.T_max, self.T_max])
            t_cur += self.dwell_min
            t.extend([t_cur, t_cur + ramp_time])
            T.extend([self.T_max, self.T_min])
            t_cur += ramp_time
        return np.array(t), np.array(T)


JEDEC_PROFILES = {
    'TC-G': ThermalProfile('JEDEC TC-G',  T_min=-40,  T_max=125, ramp_rate=10, dwell_min=10),
    'TC-J': ThermalProfile('JEDEC TC-J',  T_min=0,    T_max=100, ramp_rate=10, dwell_min=10),
    'TC-B': ThermalProfile('JEDEC TC-B',  T_min=-55,  T_max=125, ramp_rate=10, dwell_min=10),
    'OP':   ThermalProfile('Operating',   T_min=25,   T_max=85,  ramp_rate=20, dwell_min=5),
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Thermal Stack Analysis (1D Fourier)
# ─────────────────────────────────────────────────────────────────────────────

def thermal_stack_analysis(
    materials: list[Material],
    thicknesses: list[float],  # mm
    T_junction: float,
    T_ambient: float,
    power_W: float,
    area_mm2: float
) -> dict:
    """
    1D steady-state thermal resistance network through package stack.
    Returns temperature at each layer interface.
    """
    n = len(materials)
    assert len(thicknesses) == n, "Materials and thicknesses must match"

    area_m2 = area_mm2 * 1e-6
    R_layers = []
    for mat, t_mm in zip(materials, thicknesses):
        t_m = t_mm * 1e-3
        R = t_m / (mat.k * area_m2)  # K/W
        R_layers.append(R)

    R_total = sum(R_layers)
    T_interfaces = [T_junction]
    T_cur = T_junction
    for R in R_layers:
        T_cur -= power_W * R
        T_interfaces.append(T_cur)

    return {
        'R_layers_KW': R_layers,
        'R_total_KW': R_total,
        'T_interfaces_C': T_interfaces,
        'T_junction_C': T_junction,
        'T_case_C': T_interfaces[-1],
        'delta_T_C': T_junction - T_interfaces[-1],
        'theta_ja': R_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. CTE Mismatch Stress (Suhir Analytical Model)
# ─────────────────────────────────────────────────────────────────────────────

def suhir_interface_stress(
    die: Material,
    substrate: Material,
    underfill: Material,
    geometry: PackageGeometry,
    delta_T: float,
    T_ref: float = 150.0
) -> dict:

    L      = geometry.die_size / 2 * 1e-3         # half die length [m]
    h_die  = geometry.die_thickness * 1e-3         # [m]
    h_sub  = geometry.substrate_thickness * 1e-3   # [m]
    h_uf   = max(geometry.underfill_thickness * 1e-6, 1e-5)  # [m]

    # Plane-stress moduli [Pa]
    E1 = die.E_pa / (1.0 - die.nu)
    E2 = substrate.E_pa / (1.0 - substrate.nu)

    # CTE mismatch [/°C]
    d_alpha = substrate.CTE_si - die.CTE_si

    # Suhir axial stiffness parameter [Pa/m]
    # beta = (E1*h1 + E2*h2) / (E1*h1 * E2*h2) ... simplified as:
    K_ax = (E1 * h_die * E2 * h_sub) / (E1 * h_die + E2 * h_sub)

    # Underfill shear modulus [Pa]
    G_uf = underfill.E_pa / (2.0 * (1.0 + underfill.nu))

    # Suhir characteristic parameter [1/m]
    # lambda = sqrt(G_uf / h_uf * (1/(E1*h1) + 1/(E2*h2)))
    beta = G_uf / h_uf * (1.0 / (E1 * h_die) + 1.0 / (E2 * h_sub))
    lambda_s = np.sqrt(beta)

    # Peak shear stress at die edge [Pa]
    # tau_max = d_alpha * dT * K_ax * tanh(lambda*L) / lambda  ... Suhir 1986
    tau_max_Pa = d_alpha * delta_T * K_ax * np.tanh(lambda_s * L) / lambda_s
    tau_max_MPa = abs(tau_max_Pa) / 1e6

    # Peel stress [MPa] — simplified
    sigma_peel_MPa = tau_max_MPa * 0.35

    # Von Mises
    sigma_vm = np.sqrt(3.0 * tau_max_MPa**2 + sigma_peel_MPa**2)

    # Distribution along half-span
    x_arr = np.linspace(0, geometry.die_size / 2, 200)
    x_m = x_arr * 1e-3
    tau_dist = np.abs(d_alpha * delta_T * K_ax *
                      np.sinh(lambda_s * x_m) /
                      (np.cosh(lambda_s * L) * lambda_s)) / 1e6

    # Corner shear strain
    gamma_max = abs(d_alpha) * abs(delta_T) * geometry.DNP_max * 1e-3

    return {
        'tau_max_MPa':              tau_max_MPa,
        'sigma_peel_MPa':           sigma_peel_MPa,
        'sigma_vm_MPa':             sigma_vm,
        'lambda_1_m':               lambda_s,
        'characteristic_length_mm': 1.0 / lambda_s * 1e3,
        'x_arr_mm':                 x_arr,
        'tau_dist_MPa':             tau_dist,
        'gamma_max':                gamma_max,
        'd_alpha_ppm':              d_alpha * 1e6,
    }
# ─────────────────────────────────────────────────────────────────────────────
# 3. Solder Joint Fatigue Life (Engelmaier Model)
# ─────────────────────────────────────────────────────────────────────────────

def engelmaier_fatigue_life(
    geometry: PackageGeometry,
    die: Material,
    substrate: Material,
    profile: ThermalProfile,
    F_correction: float = 1.0,
    ld: float = 1.0,
    epsilon_f: float = 0.325,
) -> dict:
    """
    Engelmaier modified Coffin-Manson model for solder joint fatigue life.

    N_f = 0.5 * (2 * epsilon_f / delta_gamma) ^ (1/c)

    where:
        delta_gamma = fatigue shear strain range
        c = fatigue ductility exponent (temperature-dependent)
        epsilon_f = fatigue ductility coefficient (~0.325 for SAC305)

    Reference: Engelmaier (1983), IPC-SM-785
    """
    # Fatigue ductility exponent — temperature dependent
    T_mean_C = profile.T_mean
    c = -0.442 - 6e-4 * T_mean_C + 1.74e-2 * np.log(1 + 360 / (profile.delta_T / 2 + 1))

    # Shear strain range from CTE mismatch
    d_alpha = abs(substrate.CTE_si - die.CTE_si)
    DNP = geometry.DNP_max * 1e-3  # m
    h_s = geometry.bump_height * 1e-6  # m

    # Shear strain range (Engelmaier)
    delta_gamma = F_correction * d_alpha * profile.delta_T * DNP * ld / h_s

    # Cycles to 50% failure probability
    N_f = 0.5 * (2 * epsilon_f / delta_gamma) ** (1 / c)

    # Characteristic life (Weibull, eta = N_f / (-ln(0.5))^(1/beta))
    beta_weibull = 3.0  # shape parameter typical for solder fatigue
    eta = N_f / ((-np.log(0.5)) ** (1 / beta_weibull))

    # Estimated time to failure
    cycle_duration_h = (2 * geometry.die_size / 10 + 2 * 10) / 60  # rough estimate
    TTF_h = N_f * cycle_duration_h

    return {
        'N_f_cycles': max(N_f, 1),
        'c_exponent': c,
        'delta_gamma': delta_gamma,
        'eta_weibull': eta,
        'TTF_hours': TTF_h,
        'T_mean_C': T_mean_C,
        'DNP_mm': geometry.DNP_max,
        'risk': 'Low' if N_f > 2000 else ('Medium' if N_f > 500 else 'High'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Warpage (Timoshenko Bimetal)
# ─────────────────────────────────────────────────────────────────────────────

def timoshenko_warpage(
    die: Material,
    substrate: Material,
    geometry: PackageGeometry,
    delta_T: float,
) -> dict:
    """
    Timoshenko bimetal beam approximation for package warpage.
    Returns warpage in µm over the substrate half-span.
    """
    h1 = geometry.die_thickness       # mm
    h2 = geometry.substrate_thickness  # mm
    E1 = die.E           # GPa
    E2 = substrate.E     # GPa
    L  = geometry.substrate_size / 2 * 1e-3  # m

    m = h1 / h2
    n = E1 / E2

    # Timoshenko curvature formula
    numerator = 6 * (die.CTE - substrate.CTE) * 1e-6 * delta_T * (1 + m) ** 2
    denominator = (h1 + h2) * 1e-3 * (
        3 * (1 + m) ** 2 +
        (1 + m * n) * (m ** 2 + 1 / (m * n))
    )

    kappa = numerator / denominator  # 1/m (curvature)

    # Max deflection at corners: w = kappa * L^2 / 2
    w_max_m = kappa * L ** 2 / 2
    w_max_um = w_max_m * 1e6

    # Warpage sign: positive = concave (die side bends up) if CTE_sub > CTE_die
    sign = 1 if substrate.CTE > die.CTE else -1

    return {
        'warpage_um': sign * w_max_um,
        'warpage_abs_um': abs(w_max_um),
        'curvature_1_m': abs(kappa),
        'warpage_shape': 'Concave (smile)' if sign > 0 else 'Convex (cry)',
        'jedec_limit_um': 200.0,  # typical JEDEC coplanarity limit
        'passes_jedec': abs(w_max_um) < 200.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full CPI Assessment
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CPIResult:
    """Container for full CPI assessment results."""
    package: PackageGeometry
    die: Material
    substrate: Material
    underfill: Material
    profile: ThermalProfile
    thermal: dict
    stress: dict
    fatigue: dict
    warpage: dict

    def summary(self) -> dict:
        return {
            'tau_max_MPa': round(self.stress['tau_max_MPa'], 2),
            'sigma_peel_MPa': round(self.stress['sigma_peel_MPa'], 2),
            'sigma_vm_MPa': round(self.stress['sigma_vm_MPa'], 2),
            'warpage_um': round(self.warpage['warpage_um'], 1),
            'N_f_cycles': int(self.fatigue['N_f_cycles']),
            'risk': self.fatigue['risk'],
            'passes_jedec_warpage': self.warpage['passes_jedec'],
        }


def run_cpi_assessment(
    die: Material,
    substrate: Material,
    underfill: Material,
    geometry: PackageGeometry,
    profile: ThermalProfile,
    power_W: float = 5.0,
    T_ambient: float = 25.0,
) -> CPIResult:
    """Run full CPI thermal-mechanical assessment."""
    # Thermal stack
    stack_mats = [die, substrate]
    stack_t    = [geometry.die_thickness, geometry.substrate_thickness]
    thermal = thermal_stack_analysis(
        stack_mats, stack_t,
        T_junction=T_ambient + 30,
        T_ambient=T_ambient,
        power_W=power_W,
        area_mm2=geometry.die_area
    )

    # Interface stress
    stress = suhir_interface_stress(
        die, substrate, underfill, geometry,
        delta_T=profile.delta_T,
        T_ref=150.0
    )

    # Fatigue life
    fatigue = engelmaier_fatigue_life(geometry, die, substrate, profile)

    # Warpage
    warpage = timoshenko_warpage(die, substrate, geometry,
                                  delta_T=profile.T_max - 25.0)

    return CPIResult(
        package=geometry, die=die, substrate=substrate,
        underfill=underfill, profile=profile,
        thermal=thermal, stress=stress,
        fatigue=fatigue, warpage=warpage
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Parametric DOE Sweep
# ─────────────────────────────────────────────────────────────────────────────

def doe_sweep(
    base_geometry: PackageGeometry,
    profile: ThermalProfile,
    substrate_CTEs: list[float] = [8, 12, 17, 21],
    underfill_moduli: list[float] = [4, 7, 9, 12],
    die_sizes: list[float] = [7, 10, 15, 20],
) -> dict:
    """
    Full-factorial DOE sweep: substrate CTE × underfill modulus × die size.
    Returns results grid for analysis.
    """
    results = []
    die = MATERIALS['silicon']

    for cte in substrate_CTEs:
        for E_uf in underfill_moduli:
            for d_size in die_sizes:
                sub = Material(
                    name=f'Substrate CTE{cte}',
                    E=25.0, nu=0.39, CTE=cte,
                    k=0.35, rho_Cp=1.5e6
                )
                uf = Material(
                    name=f'UF E={E_uf}GPa',
                    E=E_uf, nu=0.35, CTE=26.0,
                    k=0.7, rho_Cp=1.8e6
                )
                geom = PackageGeometry(
                    die_size=d_size,
                    die_thickness=base_geometry.die_thickness,
                    substrate_size=d_size + 7,
                    substrate_thickness=base_geometry.substrate_thickness,
                    bump_pitch=base_geometry.bump_pitch,
                    bump_height=base_geometry.bump_height,
                    bump_diameter=base_geometry.bump_diameter,
                    underfill_thickness=base_geometry.underfill_thickness,
                    mold_thickness=base_geometry.mold_thickness,
                )
                result = run_cpi_assessment(die, sub, uf, geom, profile)
                s = result.summary()
                s.update({
                    'substrate_CTE': cte,
                    'underfill_E_GPa': E_uf,
                    'die_size_mm': d_size,
                })
                results.append(s)

    import pandas as pd
    return pd.DataFrame(results)
