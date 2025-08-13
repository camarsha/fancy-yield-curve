from .fancy_yield_curve import calc_direct_capture_yield
import numpy as np


def read_azure_file(file_name):
    energies = []
    angle = []
    cs = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.split()
            if line:
                energies.append(float(line[0]) * 1000.0)
                angle.append(float(line[2]))
                cs.append(float(line[3]))
    return np.asarray(energies), np.asarray(angle), np.asarray(cs)


amu_to_kev = 931.4940954 * 1000.0


def calc_gamma(m_a, m_A, m_B, E_beam, E_x, theta):
    """Calculate energy of a gamma ray relativistically.
    See Iliadis C.14

    :param m_a: mass of projectile in u
    :param m_A: mass of target in u
    :param m_B: mass of recoil in u
    :param E_beam: Laboratory energy of the beam in keV
    :param E_x: Excitation energy of the recoil in keV
    :param theta: lab angle of the detector.
    :returns: gamma ray energy in MeV

    """

    # conversions
    theta *= np.pi / 180.0
    m_a *= amu_to_kev
    m_A *= amu_to_kev
    m_B *= amu_to_kev
    q = ((m_a + m_A) - m_B) - E_x
    numer = E_beam * m_A + (q * (m_a + m_A + m_B) / 2.0)

    denom = (
        m_a
        + m_A
        + E_beam
        - (np.cos(theta) * np.sqrt(E_beam * (2.0 * m_a + E_beam)))
    )
    return numer / denom


def com_cs_conversion(ma, mA, elab, theta):
    """
    I took all of this from Iliadis C.45.
    """
    theta = theta * (np.pi / 180.0)
    top = np.sqrt(elab * (elab + (2.0 * ma * amu_to_kev)))
    bottom = ma * amu_to_kev + mA * amu_to_kev + elab
    beta = np.sqrt(top) / bottom

    num = 1.0 - beta**2.0
    denom = (1.0 + beta * np.cos(theta)) ** 2.0
    return num / denom


class DirectCapture:
    def __init__(self, proj_mass, target_mass, recoil_mass, cs_file_name):
        self.energies_com, self.theta_com, self.cs_com = read_azure_file(
            cs_file_name
        )
        self.proj_mass = proj_mass
        self.target_mass = target_mass
        self.recoil_mass = recoil_mass
        self.energies = self.energies_com * (
            (proj_mass + target_mass) / target_mass
        )
        self.conv = com_cs_conversion(
            proj_mass,
            target_mass,
            self.energies,
            self.theta_com,
        )
        self.cs = self.cs_com * self.conv

    def yield_curve(
        self,
        e_beam,
        beam_fwhm,
        det_fwhm,
        height,
        dx,
        straggle_const,
        start,
        stop,
        step,
    ):
        return calc_direct_capture_yield(
            e_beam,
            beam_fwhm,
            det_fwhm,
            height,
            dx,
            straggle_const,
            self.energies,
            self.cs,
            start,
            stop,
            step,
        )

    def gamma_energy(self, e_p, angle):
        return calc_gamma(
            self.proj_mass,
            self.target_mass,
            self.recoil_mass,
            e_p,
            0.0,
            angle,
        )
