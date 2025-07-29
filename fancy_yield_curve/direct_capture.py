from .fancy_yield_curve import calc_direct_capture_yield
import numpy as np


def read_azure_file(file_name):
    energies = []
    cs = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.split()
            if line:
                energies.append(float(line[0]) * 1000.0)
                cs.append(float(line[3]))
    return energies, cs


amu_to_kev = 931.4940954 * 1000.0


def calc_gamma(m_a, m_A, m_B, E_beam, E_x, theta):
    """Calculate energy of a gamma ray relativistically.
    See Iliadis C.14

    :param m_a: mass of projectile in u
    :param m_A: mass of target in u
    :param m_B: mass of recoil in u
    :param E_beam: Laboratory energy of the beam in MeV
    :param E_x: Excitation energy of the recoil in MeV
    :param theta: Laboratory angle of the detector.
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


class DirectCapture:
    def __init__(self, proj_mass, target_mass, recoil_mass, cs_file_name):
        self.energies_com, self.values = read_azure_file(cs_file_name)
        self.proj_mass = proj_mass
        self.target_mass = target_mass
        self.recoil_mass = recoil_mass
        self.energies = np.asarray(self.energies_com) * (
            (proj_mass + target_mass) / target_mass
        )

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
            self.values,
            start,
            stop,
            step,
        )
