import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import os
import photutils
import math

from Tutorial_functions import *
from mag_zero_est import *
from JG_Streaktools import *
from interface_setup import *
from plate_solve import *
from effective_psf import *
from beacon import *
from haversine import haversine, Unit
from IPython.display import Markdown
from scipy.optimize import minimize_scalar


def V_lambda(lam_nm, lam_unc):
    """
    Returns (V, uncertainty) using linear interpolation
    and wavelength uncertainty lam_unc (in nm).
    """

    #sorted wavelength (nm) and V(lambda) table
    wv = np.array([600, 620, 630, 640, 650, 660, 670, 700])
    v  = np.array([0.631,0.381,0.265,0.175,0.107,0.061,0.032,0.004])

    #linear interpolation for central wavelengths
    V0 = np.interp(lam_nm, wv, v)

    #linar interpolation for wavelength bounds
    V_low  = np.interp(lam_nm - lam_unc, wv, v)
    V_high = np.interp(lam_nm + lam_unc, wv, v)

    #worst case deviation
    dV = max(abs(V0 - V_low), abs(V_high - V0))

    return V0, dV



def lumens_to_Irad(lumens, lumens_unc, lam_nm, lam_unc):
    """
    Convert luminous flux (lumens) to radiant intensity (W/sr)
    with wavelength and lumen uncertainties included.
    
    Uses:
        Radiant flux (Phi_rad) = lumens / (683 * V(lambda))
        Radiant intensity (I_rad) = Phi_rad / (4 * pi)
    """

    V0, dV = V_lambda(lam_nm, lam_unc) 

    #central radiant flux
    Phi_rad = lumens / (683 * V0)

    #error propagation
    rel_unc_lumens = lumens_unc / lumens
    rel_unc_V      = dV / V0

    Phi_unc = Phi_rad * np.sqrt(rel_unc_lumens**2 + rel_unc_V**2)

    #convert to radiant intensity (Assume isotropic!!)
    I_rad = Phi_rad / (4*np.pi)
    I_unc = Phi_unc / (4*np.pi)

    return I_rad, I_unc


def irradiance_at_distance(Irad, dIrad, dist, d_dist, theta, d_theta):
    """
    Computes irradiance F and its uncertainty dF at a distance
    from a source with radiant intensty Irad.

    Inputs:
      Irad = radiant intensity (W/sr)
      dIrad = uncertainty in radiant intensity (W/sr)
      dist = distance to observer (m)
      d_dist = distance uncertainty (m)
      theta = angle between source axis and observer (radians)
      d_theta = uncertainty in angle (radians)

    Returns:
      (F, dF) : irradiance and its propagated uncertainty
    """

    #central irradince value
    F = Irad * np.cos(theta) / dist**2

    #partial derivatives for uncertainty propagation
    dF_dI     = np.cos(theta) / dist**2
    dF_ddist  = -2 * Irad * np.cos(theta) / dist**3
    dF_dtheta = -Irad * np.sin(theta) / dist**2

    #total uncertainty (QUAD sum)
    dF = np.sqrt(
        (dF_dI     * dIrad)**2 +
        (dF_ddist  * d_dist)**2 +
        (dF_dtheta * d_theta)**2
    )

    return F, dF


def detected_irradiance(F, dF, S0, dS0):
    """
    Computes the detected irradiance after system throughput S0
    and propagates uncertainties in both F and S0.

    Inputs:
        F  = irradiance at the observer
        dF = uncertainty in that irradiance
        S0 = system throughput (fraction of flux detected)
        dS0 = uncertainty in system throughput

    Returns:
        (F_det, dF_det) - detected irradiance and its uncertainty
    """

    # central detected irradiance
    F_det = F * S0

    # relative uncertainties
    rel_F  = dF  / F
    rel_S0 = dS0 / S0

    # total uncertainty
    dF_det = F_det * np.sqrt(rel_F**2 + rel_S0**2)

    return F_det, dF_det


def ADU_per_pixel(Fdet, dFdet, lam_nm, dlam_nm, exp_time, gain_e_per_ADU, A_pix):
    """
    Computes ADU per pixel and uncertainty.

    Inputs:
        Fdet = detected irradiance at the sensor (W/m^2)
        dFdet = uncertainty in detected irradiance
        lam_nm = wavelength in nm
        dlam_nm = wavelength uncertainty in nm
        exp_time = exposure time (s)
        gain_e_per_ADU = electrons per ADU (cam gain)
        A_pix = pixel area (m^2) (given on CCD data sheet)

    Returns:
        (ADU, dADU)
    """

    #convert wavelength to m
    lam = lam_nm * 1e-9
    dlam = dlam_nm * 1e-9

    E_ph = h*c / lam

    #uncertainty handlng
    dE_dlam = -h*c / lam**2
    dE_ph = abs(dE_dlam) * dlam

    #optical power on a single pixle
    P_pix = Fdet * A_pix
    dP_pix = dFdet * A_pix

    #photons
    N_ph = (P_pix * exp_time) / E_ph

    #uncertainty in photons
    dN_ph = N_ph * ( (dP_pix / P_pix)**2 + (dE_ph / E_ph)**2 )**0.5

    #Losses accounted for in S0
    N_e = N_ph
    dN_e = dN_ph 

    #ADU convs
    ADU = N_e / gain_e_per_ADU
    dADU = dN_e / gain_e_per_ADU

    return ADU, dADU\


def total_ADU(ADU_pixel, dADU_pixel, Npix):
    """
    Computes total ADU in an aperture and its uncertainty.

    Inputs:
        ADU_pixel = ADU per pixel
        dADU_pixel = uncertainty in ADU per pixel
        Npix = number of pixels in the aperture (can get from Streak Tools after performing MCMC or pill)

    Returns:
        (Total_ADU, dTotal_ADU)
    """

    Total_ADU = ADU_pixel * Npix
    dTotal_ADU = dADU_pixel * Npix

    return Total_ADU, dTotal_ADU


def predict_total_ADU(lumens, d_lumens,dist, d_dist, exp_time, lam_nm, d_lam_nm,
                      S0, d_S0,
                      gain_e_per_ADU,
                      Npix,
                      A_pix):

    #luminous flux to rad intensity
    Irad, dIrad = lumens_to_Irad(lumens, d_lumens, lam_nm, d_lam_nm)

    #rad intensity to irradiance @ observer
    F, dF = irradiance_at_distance(Irad, dIrad, dist, d_dist, theta=0, d_theta=0)

    #irradiance to detected irradiance (system throughpit)
    Fdet, dFdet = detected_irradiance(F, dF, S0, d_S0)

    #detected irradiance to ADU per pixel
    ADU_pix, dADU_pix = ADU_per_pixel(Fdet, dFdet,
                                      lam_nm, d_lam_nm,
                                      exp_time,
                                      gain_e_per_ADU,
                                      A_pix)

    #ADU per pixel to total ADU in aperture
    Total_ADU, dTotal_ADU = total_ADU(ADU_pix, dADU_pix, Npix)

    return Total_ADU, dTotal_ADU


def S0_error(S0, lumens, exp_time, lam_nm, gain, A_pix, ADU_meas, distances, Npix_list):
    errors = []
    
    for ADU_obs, dist, Npix in zip(ADU_meas, distances, Npix_list):
        
        ADU_pred, _ = predict_total_ADU(
            lumens, 0,
            dist, 0,
            exp_time,
            lam_nm, 0,
            S0, 0,
            gain,
            Npix,
            A_pix
        )
        
        errors.append( (ADU_pred - ADU_obs)**2 )
    
    return sum(errors)


def fit_S0(lumens, exp_time, lam_nm, gain, A_pix, ADU_meas, distances, Npix_list):

    result = minimize_scalar(
        S0_error,
        bounds=(0, 1),
        method='bounded',
        args=(lumens, exp_time, lam_nm, gain, A_pix, ADU_meas, distances, Npix_list)
    )

    return result.x


