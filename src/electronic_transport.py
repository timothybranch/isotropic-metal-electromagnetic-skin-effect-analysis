"""
This module contains functions for calculating the skin effect response of a metal.

Includes functions for calculating conductivity spectra, surface impedance, skin effect regime boundaries, and skin effect regime asymptotic behaviours.

Most of the functions are designed to work with vectors of the wavevectors `q' and the angular frequencies `omega'.

Sources:

Baker, G. (2022). Non-Local Electrical Conductivity in PdCoO$_{2}$ [PhD thesis]. University of British Columbia. http://hdl.handle.net/2429/82849

Hein, M. A., Ormeno, R. J., & Gough, C. E. (2001). High-frequency electrodynamic response of strongly anisotropic clean normal and superconducting metals. Physical Review B, 64(2), 024529. https://doi.org/10.1103/PhysRevB.64.024529

3. Sondheimer, E. H. (1954). The theory of the anomalous skin effect in anisotropic metals. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 224(1157), 260-271. https://doi.org/10.1098/rspa.1954.0157

4. Reuter, G. E. H., & Sondheimer, E. H. (1948). The theory of the anomalous skin effect in metals. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 195(1042), 336-364. https://doi.org/10.1098/rspa.1948.0123
"""

import numpy as np
from numpy.typing import NDArray
from scipy.constants import mu_0, epsilon_0, pi, c
import mpmath as mp

beta_specular: float = 4/(3*np.sqrt(3))
"""
Constant prefactorfor the anomalous skin effect regime with specular boundary scattering.

From Graham Baker's PhD thesis, equation 1.45, page 21.

Notes
-----
.. math:: 
    \beta_{specular} = \frac{4}{3\sqrt{3}} \approx 0.7698

    Z_ASE = beta_specular*mu_0*((4*skin_depth_London**2*velocity_Fermi)/(3*pi))**(1/3)*omega**(2/3)*np.exp(-1j*pi/3)
"""
beta_diffuse: float = np.sqrt(3)/2
"""
Constant prefactor for the anomalous skin effect regime with diffuse boundary scattering.

From Graham Baker's PhD thesis, equation 1.45, page 21.

Notes
-----
.. math::
    \beta_{diffuse} = \frac{\sqrt{3}}{2} \approx 0.8660
"""

def skin_depth_London(freq_plasma: float) -> float:
    """
    The London skin depth as a function of plasma frequency.

    Represents the skin depth in the limit of no bulk scattering.

    From Graham Baker's PhD thesis.

    Parameters
    ----------
    freq_plasma : float
        Plasma frequency [Hz]

    Returns
    -------
    float
        Skin depth [m]

    Notes
    -----
    .. math::
        \lambda_{London} = \frac{c}{\omega_{plasma}}
    """
    return c/freq_plasma

# @np.vectorize
def q_prime_func(q: NDArray[np.float64], omega: NDArray[np.float64], mean_free_path_MR: float, rate_scattering_MR: float) -> NDArray[np.complex128]:
    """
    Calculate the dimensionless wavevector product, that include relaxation effects.

    Used for calculating the conductivity spectrum.

    From Hein et al. PRB 64 024529 (2001).
    
    Parameters
    ----------
    q : NDArray[np.float64] of shape (N,)
        Wavevector [1/m]
    omega : NDArray[np.float64] of shape (M,)
        Angular frequency [rad/s]
    mean_free_path_MR : float
        Mean free path l_MR [m]
    rate_scattering_MR : float
        Scattering rate gamma_MR [rad/s]

    Returns
    -------
    NDArray[np.complex128] of shape (N, M)
        q * mean_free_path_MR / (1 - i * omega / rate_scattering_MR) [1/m]

    Notes
    -----
    Returns the following expression:
    .. math::
        q' = \frac{q l_{MR}}{1 - i \omega / \gamma_{MR}}
    """

    assert mean_free_path_MR > 0, "Mean free path must be positive"
    assert rate_scattering_MR > 0, "Scattering rate must be positive"
    assert all(q > 0), "Wavevectors must be positive"
    assert all(omega > 0), "Angular frequencies must be positive"

    return np.outer((q*mean_free_path_MR), 1/(1-1j*omega/rate_scattering_MR))

@np.vectorize
def conductivity_relaxation(cond_DC: float, omega: NDArray[np.float64], rate_scattering_MR: float) -> NDArray[np.complex128]:
    """
    Calculate the frequency-dependent Drude response conductivity of a metal, using a momentum-relaxing scattering rate.

    From Hein et al. PRB 64 024529 (2001).
    
    Parameters
    ----------
    cond_DC : float
        Conductivity at zero frequency [S/m]
    omega : NDArray[np.float64] of shape (N,)
        Angular frequency [rad/s]
    rate_scattering_MR : float
        Momentum-relaxing scattering rate [rad/s]

    Returns
    -------
    NDArray[np.complex128] of shape (N,)
        Drude conductivity including relaxation effects [S/m]

    Notes
    -----
    Returns the following expression:
    .. math::
        \sigma(\omega) = \frac{\sigma_{DC}}{1 - i \omega / \gamma_{MR}}
    """

    assert cond_DC > 0, "Conductivity must be positive"
    assert rate_scattering_MR > 0, "Scattering rate must be positive"
    assert omega > 0, "Angular frequencies must be positive"

    return cond_DC/(1 - 1j*omega/rate_scattering_MR)

@np.vectorize
def nonlocality_term_3D_isotropic_func(q_prime: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Term that accounts for nonlocality of a 3D isotropic Fermi surface.

    Used for calculating the conductivity spectrum.

    From equation (3) in Hein et al. PRB 64 024529 (2001).

    Parameters
    ----------
    q_prime : NDArray[np.complex128] of shape (N,M)
        q_prime = q * l_MR / (1 - i * omega / gamma_MR) [1]

    Returns
    -------
    NDArray[np.complex128] of shape (N,M)
        Nonlocality term. [1]

    Notes
    -----
    Returns the following expression:
    .. math::
        \frac{3}{2} q'^{-3} \left( (1 + q'^2) \arctan(q') - q' \right)
    """

    ## I'm not sure why it works this way 
    @np.vectorize
    def nonlocality_term_func(q_prime):
        q_function = (3/2) * q_prime**-3 * ((1 + q_prime**2) * mp.atan(q_prime) - q_prime)
        nonlocality_term = complex(q_function)
        return nonlocality_term

    # q_function = (3/2) * q_prime**-3 * ((1 + q_prime**2) * mp.atan(q_prime) - q_prime)
    # nonlocality_term = complex(q_function)
    nonlocality_term = np.array(nonlocality_term_func(q_prime).tolist(), dtype=complex)
    return nonlocality_term

def cond_spectrum_3D_isotropic(q: NDArray[np.float64], omega: NDArray[np.float64], mean_free_path_MR: float, rate_scattering_MR: float, resistivity_residual_DC: float) -> NDArray[np.complex128]:
    """
    Calculate the conductivity spectrum for a 3D isotropic Fermi surface.

    From Hein et al. PRB 64 024529 (2001).
    
    Parameters
    ----------
    q : NDArray[np.float64] of shape (N,)
        Wavevector [1/m]
    omega : NDArray[np.float64] of shape (M,)
        Angular frequency [rad/s]
    mean_free_path_MR : float
        Mean free path l_MR [m]
    rate_scattering_MR : float
        Scattering rate gamma_MR [rad/s]
    resistivity_residual_DC : float
        Residual resistivity [Ohm m]

    Returns
    -------
    cond_spectrum : NDArray[np.complex128] of shape (N,M)
        Conductivity spectrum [S/m]
    """
    assert all(q > 0), "Wavevectors must be positive"
    assert all(omega > 0), "Angular frequencies must be positive"
    assert mean_free_path_MR > 0, "Mean free path must be positive"
    assert rate_scattering_MR > 0, "Scattering rate must be positive"
    assert resistivity_residual_DC > 0, "Residual resistivity must be positive"

    cond_spectrum_relaxation = conductivity_relaxation(resistivity_residual_DC**-1, omega, rate_scattering_MR)
    q_prime = q_prime_func(q, omega, mean_free_path_MR, rate_scattering_MR)
    nonlocality_term = nonlocality_term_3D_isotropic_func(q_prime)
    return np.multiply(cond_spectrum_relaxation, nonlocality_term)

## Analytic limits

def Z_CSE_limit(freq: NDArray[np.float64], resistivity_residual: float) -> NDArray[np.complex128]:
    """
    Surface impedance in the classical skin effect regime.

    Has a negative complex phase, and negative surface reactance.
    
    From Jake Bobowski's PhD thesis, eqn. 3.5, pg. 50 (2010).

    Parameters
    ----------
    freq : NDArray[np.float64] of shape (N,)
        Frequency [Hz]
    resistivity_residual : float
        Residual resistivity [Ohm m]

    Returns
    -------
    Z_CSE : NDArray[np.complex128] of shape (N,)
        Surface impedance in the classical skin effect regime [Ohm]

    Notes
    -----
    Returns the following expression:
    .. math::
        Z_{CSE} = \sqrt{-1j \mu_0 \omega \rho_{residual}}
    """
    assert all(freq > 0), "Frequency must be positive"
    assert resistivity_residual > 0, "Residual resistivity must be positive"

    omega = 2*pi*freq

    return np.sqrt(-1j*mu_0*omega*resistivity_residual)


def Z_ASE_limit(freq: NDArray[np.float64], freq_plasma: float, velocity_Fermi: float, surface_scattering_type: str) -> NDArray[np.complex128]:
    """
    Surface impedance in the anomalous skin effect regime, for either specular or diffuse surface scattering.

    Has a negative complex phase, and negative surface reactance.
    
    From Graham Baker's PhD thesis, eqn. 1.44, pg. 21 (2022).

    Parameters
    ----------
    freq : NDArray[np.float64] of shape (N,)
        Frequency [Hz]
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity [m/s]
    surface_scattering_type : str('specular' or 'diffuse')
        Surface scattering type

    Returns
    -------
    Z_ASE : NDArray[np.complex128] of shape (N,)
        Surface impedance in the anomalous skin effect regime [Ohm]

    Raises
    ------
    ValueError
        If the surface scattering type is not 'specular' or 'diffuse'.
        
    Notes
    -----
    Returns the following expression:
    .. math::
        Z_{ASE} = \beta \mu_0 \left( \frac{4 \lambda_{London}^2 velocity_Fermi}{3 \pi} \right)^{1/3} \omega^{2/3} e^{-i \pi/3}
    """

    assert all(freq > 0), "Frequency must be positive"
    assert freq_plasma > 0, "Plasma frequency must be positive"
    assert velocity_Fermi > 0, "Fermi velocity must be positive"
    assert surface_scattering_type in ['specular', 'diffuse'], "Surface scattering type must be 'specular' or 'diffuse'"

    omega = 2*pi*freq

    skin_depth_London = c/freq_plasma
    
    if surface_scattering_type == 'specular':
        beta = beta_specular
    elif surface_scattering_type == 'diffuse':
        beta = beta_diffuse
    else:
        ## This is redundant, but I'll keep it here
        raise ValueError("Invalid surface scattering type")

    return beta*mu_0*((4*skin_depth_London**2*velocity_Fermi)/(3*pi))**(1/3)*omega**(2/3)*np.exp(-1j*pi/3)

def Z_relaxation_limit(freq: NDArray[np.float64], freq_plasma: float, gamma_MR_average: float) -> NDArray[np.complex128]:
    """
    Surface impedance in the relaxation regime, to the leading order in both surface resistance and surface reactance.

    Has a negative complex phase, and negative surface reactance.
    
    From Graham Baker's PhD thesis, eqn. 1.44, pg. 21.

    Parameters
    ----------
    freq : NDArray[np.float64] of shape (N,)
        Frequency [Hz]
    freq_plasma : float
        Plasma frequency [Hz]
    gamma_MR_average : float
        Average scattering rate [rad/s]

    Returns
    -------
    Z_relaxation : NDArray[np.complex128] of shape (N,)
        Surface impedance in the relaxation regime limit [Ohm]

    Notes
    -----
    Returns the following expression:
    .. math::
        Z_{relaxation} = \mu_0 \lambda_{London} \omega e^{-i \pi/2}
    """

    assert all(freq > 0), "Frequency must be positive"
    assert freq_plasma > 0, "Plasma frequency must be positive"
    assert gamma_MR_average > 0, "Scattering rate must be positive"
    
    omega = 2*pi*freq

    skin_depth_London = c/freq_plasma

    R_s = mu_0*skin_depth_London*gamma_MR_average/2
    X_s = np.imag(mu_0*skin_depth_London*omega*np.exp(-1j*pi/2))

    Z_relaxation_limit = R_s + 1j*X_s
    return Z_relaxation_limit


def Z_anomalous_reflection_limit_diffuse(freq: NDArray[np.float64], freq_plasma: float, velocity_Fermi_J: float, gamma_MR_average: float) -> NDArray[np.complex128]:
    """
    Surface impedance in the anomalous reflection regime, for diffuse surface scattering.

    Has a negative complex phase, and negative surface reactance.

    From Graham Baker's PhD thesis, eqn. 1.44, pg. 21.

    The surface resistance is a combination of the surface resistance in the relaxation limit,
    but also includes a contribution due to surface scattering.

    This contribution is found from eqn. 46 in "The skin effect II. The skin effect at high frequencies" by Casimir et al. (1967).

    Parameters
    ----------
    freq : NDArray[np.float64] of shape (N,)
        Frequency [Hz]
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi_J : float
        Fermi velocity [m/s]
    gamma_MR_average : float
        Average scattering rate [rad/s]

    Returns
    -------
    Z_anomalous_reflection_limit : NDArray[np.complex128] of shape (N,)
        Surface impedance in the anomalous reflection regime [Ohm]
    """
    omega = 2*pi*freq

    skin_depth_London = c/freq_plasma

    ##TODO rewrite the R_s portion as the sum of the relaxation limit plus the surface scattering contribution
    R_s = mu_0*skin_depth_London*((3/4)*velocity_Fermi_J/(2*skin_depth_London) + gamma_MR_average)/2
    X_s = Z_relaxation_limit(freq, freq_plasma, gamma_MR_average).imag

    Z_anomalous_reflection_limit = R_s + 1j*X_s
    return Z_anomalous_reflection_limit

def Z_relaxation_full(freq: NDArray[np.float64], resistivity_residual: float, rate_scattering_MR: float) -> NDArray[np.complex128]:
    """ 
    Surface impedance in the local transport regime. Captures the transition from the classical skin effect regime to the relaxation regime.

    Has a negative complex phase, and negative surface reactance.

    Parameters
    ----------
    freq : NDArray[np.float64] of shape (N,)
        Frequency [Hz]
    resistivity_residual : float
        Residual resistivity [Ohm m]
    rate_scattering_MR : float
        Analytic scattering rate [rad/s]

    Returns
    -------
    NDArray[np.complex128] of shape (N,)
        Surface impedance in the local transport regime [Ohm]

    Notes
    -----
    Returns the following expression:
    .. math::
        Z_{local} = \sqrt{-1j \mu_0 \omega / \sigma_{local}}
    """

    omega = 2*pi*freq

    return np.sqrt(-1j*mu_0*omega/conductivity_relaxation(resistivity_residual**-1, omega, rate_scattering_MR))

##TODO: Include a way to add a Fermi velocity in the direction of the current, in addition to the average one.
##TODO: Check the averages for non-isotropic Fermi surfaces.
def get_skin_effect_regime_limits(freqs: NDArray[np.float64], resistivity_residual: float, freq_plasma: float, velocity_Fermi: float, rate_scattering_MR: float) -> dict:
    """
    Provides the limits of the surface impedance for each skin effect regime.

    Included: CSE, ASE (specular and diffuse), Relaxation, Local transport (Drude response).

    Parameters
    ----------
    freqs : NDArray[np.float64] of shape (N,)
        Frequency [Hz]
    resistivity_residual : float
        Residual resistivity [Ohm m]
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Average Fermi velocity [m/s]
    rate_scattering_MR : float
        Momentum-relaxing scattering rate (harmonic average)[rad/s]

    Returns
    -------
    limits : dict[str, NDArray[np.complex128] of shape (N,)]
        Dictionary with keys as regime names and values as surface impedance [Ohm]
        keys include:
            "CSE": Classical Skin Effect
            "ASE (specular surface scattering)": Anomalous Skin Effect, specular surface scattering
            "ASE (diffuse surface scattering)": Anomalous Skin Effect, diffuse surface scattering
            "Relaxation": Relaxation
            "Local transport (Drude response)": Local transport (Drude response)
            "Anomalous reflection (diffuse surface scattering)": Anomalous reflection, diffuse surface scattering
    """

    limits = {
        "CSE": Z_CSE_limit(freqs, resistivity_residual),
        "ASE (specular surface scattering)": Z_ASE_limit(freqs, freq_plasma, velocity_Fermi, 'specular'),
        "ASE (diffuse surface scattering)": Z_ASE_limit(freqs, freq_plasma, velocity_Fermi, 'diffuse'),
        "Relaxation": Z_relaxation_limit(freqs, freq_plasma, rate_scattering_MR),
        "Local transport (Drude response)": Z_relaxation_full(freqs, resistivity_residual, rate_scattering_MR),
        "Anomalous reflection (diffuse surface scattering)": Z_anomalous_reflection_limit_diffuse(freqs, freq_plasma, velocity_Fermi, rate_scattering_MR)
    }
    return limits


def frequency_boundary_CSE_to_relaxation(rate_scattering_MR: float) -> float:
    """
    Frequency at which the CSE to relaxation boundary occurs, defined by the rates.

    From Graham Baker's PhD thesis, Table 3.3, pg. 50.

    Parameters
    ----------
    rate_scattering_MR : float
        Momentum-relaxing scattering rate [rad/s]

    Returns
    -------
    float
        Frequency [Hz]

    Notes
    -----
    Returns the following expression:
    .. math::
        f_{CSE, relaxation} = \frac{\gamma_{MR}}{2\pi}
    """

    return rate_scattering_MR/(2*pi)

def frequency_boundary_CSE_to_ASE_length_scale(freq_plasma: float, velocity_Fermi: float, mean_free_path_MR: float) -> float:
    """
    Frequency at which the CSE to ASE boundary occurs, based on 
    the momentum-relaxing mean free path being equal to the classical skin depth.

    From Graham Baker's PhD thesis, Table 3.3, pg. 50.

    Slight adjustment made, adding a factor of 2 from the table.

    Parameters
    ----------
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity v_F [m/s]
    mean_free_path_MR : float
        Momentum-relaxing mean free path l_MR [m]

    Returns
    -------
    float
        Frequency boundary between CSE and ASE regimes [Hz]
        
    Notes
    -----
    Returns the following expression (the solution for l_MR = delta_classical):
    .. math::
        f_{CSE, ASE} = \frac{1}{\pi} \lambda_{London}^2 v_F \ell_{MR}^{-3}
    """
    skin_depth_London = c/freq_plasma

    return (1/pi)*skin_depth_London**2*velocity_Fermi*mean_free_path_MR**(-3)

def frequency_boundary_CSE_to_ASE_Rs_specular(freq_plasma: float, velocity_Fermi: float, mean_free_path_MR: float) -> float:
    """
    Frequency at which the CSE to ASE boundary occurs, based on the surface resistance being equal
    in the CSE limit and the specular surface scattering ASE limit.

    Calculated from the limits in eqn 1.44 in Graham Baker's PhD thesis.
    
    Parameters
    ----------
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity (v_F) [m/s]
    mean_free_path_MR : float
        Momentum-relaxing mean free path (l_MR) [m]

    Returns
    -------
    float
        Frequency [Hz]
    """

    skin_depth_London = c/freq_plasma

    return (1/(2*pi))*((2/beta_specular**2)**3*(3*pi/4)**2)*skin_depth_London**2*velocity_Fermi*mean_free_path_MR**(-3)

def frequency_boundary_CSE_to_ASE_Rs_diffuse(freq_plasma: float, velocity_Fermi: float, mean_free_path_MR: float) -> float:
    """
    Frequency at which the CSE to ASE boundary occurs, based on the surface resistance being equal
    in the CSE limit and the diffuse surface scattering ASE limit.

    Calculated from the limits in eqn 1.44 in Graham Baker's PhD thesis.
    
    Parameters
    ----------
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity (v_F) [m/s]
    mean_free_path_MR : float
        Momentum-relaxing mean free path (l_MR) [m]

    Returns
    -------
    float
        Frequency [Hz]
    """

    skin_depth_London = c/freq_plasma

    return (1/(2*pi))*((2/beta_diffuse**2)**3*(3*pi/4)**2)*skin_depth_London**2*velocity_Fermi*mean_free_path_MR**(-3)

def frequency_boundary_ASE_to_Anomalous_Reflection_rate_scale(freq_plasma: float, velocity_Fermi: float) -> float:
    """
    Frequency at which the ASE to the anomalous reflection regime occurs,
    based on scattering rates.

    From Graham Baker's PhD thesis, Table 3.3, pg. 50.

    Parameters
    ----------
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity (v_F) [m/s]

    Returns
    -------
    float
        Frequency [Hz]

    Notes
    -----
    Returns the following expression:
    .. math::
        f_{ASE, anomalous reflection} = \frac{v_F}{2\pi \lambda_{London}}
    """

    skin_depth_London = c/freq_plasma

    return (velocity_Fermi/skin_depth_London)*(1/(2*pi))

##TODO: Add average Fermi velocity or average scattering rate
def get_skin_effect_regime_frequency_boundaries(freq_plasma: float, velocity_Fermi: float, mean_free_path_MR: float) -> dict:
    """
    Returns all frequency boundaries in a dictionary.

    Parameters
    ----------
    freq_plasma : float
        Plasma frequency [Hz]
    velocity_Fermi : float
        Fermi velocity (v_F)[m/s]
    mean_free_path_MR : float
        Momentum-relaxing mean free path (l_MR) [m]

    Returns
    -------
    dict
        Dictionary with keys as boundary names and values as frequencies [Hz]
        keys include:
            "CSE, Relaxation (scattering rate)": CSE to relaxation boundary
            "CSE, ASE (l_MR = delta)": CSE to ASE boundary based on l_MR = delta
            "CSE, ASE (Rs_CSE = Rs_ASE_diffuse)": CSE to ASE boundary based on surface resistance
            "CSE, ASE (Rs_CSE = Rs_ASE_specular)": CSE to ASE boundary based on surface resistance
            "ASE, Anomalous Reflection (scattering rate)": ASE to anomalous reflection boundary
    """

    gamma_MR = velocity_Fermi/mean_free_path_MR
    boundaries = {
        "CSE, Relaxation (scattering rate)": frequency_boundary_CSE_to_relaxation(gamma_MR),
        "CSE, ASE (l_MR = delta)": frequency_boundary_CSE_to_ASE_length_scale(freq_plasma, velocity_Fermi, mean_free_path_MR),
        "CSE, ASE (Rs_CSE = Rs_ASE_diffuse)": frequency_boundary_CSE_to_ASE_Rs_diffuse(freq_plasma, velocity_Fermi, mean_free_path_MR),
        "CSE, ASE (Rs_CSE = Rs_ASE_specular)": frequency_boundary_CSE_to_ASE_Rs_specular(freq_plasma, velocity_Fermi, mean_free_path_MR),
        "ASE, Anomalous Reflection (scattering rate)": frequency_boundary_ASE_to_Anomalous_Reflection_rate_scale(freq_plasma, velocity_Fermi)
    }
    return boundaries



def conductivity_spectrum_Fermi_surface_properties(q: NDArray[np.float64], omega: NDArray[np.float64], conductivity_spectrum: NDArray[np.complex128]) -> tuple:
    """
    Calculate the Fermi velocity, plasma frequency, and other properties from the conductivity spectrum.

    Only valid in the limit of high wavevector magnitude, q l_MR >> 1.

    Parameters
    ----------
    q : NDArray[np.float64] of shape (N,)
        Wavevector [1/m]
    omega : NDArray[np.float64] of shape (M,)
        Angular frequency [rad/s]
    conductivity_spectrum : NDArray[np.complex128] of shape (N,M)
        Conductivity spectrum [S/m]

    Returns
    -------
    tuple
        Fermi velocity, plasma frequency, and other properties (See Notes)

    Notes
    -----
    Returns the following expressions, for each q_0, omega_0, and sigma(q_0, omega_0):
    .. math::
        v_F = A/B
        \omega_{plasma} = A/v_F
        A = \epsilon_0 \omega_{p}^2/v_F = (4/(3*\pi))*\Re(\sigma(q_0,\omega_0))*q_0
        B = \epsilon_0*\omega_{p}^2/v_F^2 = \Im(\sigma(q_0,\omega_0))*q_0^2/(3*\omega_0)
    """
    epsilon_freq_plasma_squared_over_vF = (4/(3*pi))*conductivity_spectrum.real*q
    epsilon_freq_plasma_squared_over_vF_squared = conductivity_spectrum.imag*q**2/(3*omega)

    velocity_Fermi_cond_calc = epsilon_freq_plasma_squared_over_vF/epsilon_freq_plasma_squared_over_vF_squared
    epsilon_freq_plasma_squared_cond_calc = epsilon_freq_plasma_squared_over_vF*velocity_Fermi_cond_calc

    freq_plasma_cond_calc = np.sqrt(epsilon_freq_plasma_squared_cond_calc/epsilon_0)

    return velocity_Fermi_cond_calc, freq_plasma_cond_calc, epsilon_freq_plasma_squared_over_vF, epsilon_freq_plasma_squared_over_vF_squared 

## Conductivity spectrum high-q limits.

def cond_spectra_3D_isotropic_high_q_complex(ql: NDArray[np.float64], freq: float, resistivity_residual: float, freq_plasma: float) -> NDArray[np.complex128]:
    """
    Conductivity spectrum limit for a 3D isotropic Fermi surface at high q.

    Parameters
    ----------
    ql : NDArray[np.float64] of shape (N,)
        Wavevector times mean free path [1]
    freq : float
        Frequency [Hz]
    resistivity_residual : float
        Residual resistivity (sigma_DC) [Ohm m]
    freq_plasma : float
        Plasma frequency [Hz]

    Returns
    -------
    cond_spectra_high_q_limit : NDArray[np.complex128] of shape (N,)
        Conductivity spectrum [S/m]

    Notes
    -----
    Returns the following expression:
    .. math::
        \Re(\sigma) = \sigma_{DC} \frac{3\pi}{4\epsilon_0 (q \ell_{MR})}
        \Im(\sigma) = 3 \sigma_{DC} \left( \frac{\gamma_{MR}}{\omega} \right) \frac{1}{(q \ell_{MR})^2}
    """

    gamma_MR = epsilon_0*freq_plasma**2*resistivity_residual
    omega = 2*pi*freq

    real_part = np.multiply(resistivity_residual**-1,(3*pi/(4*ql)))
    imag_part = np.multiply(resistivity_residual**-1*3*(omega/gamma_MR),ql**-2)

    return real_part + 1j*imag_part
