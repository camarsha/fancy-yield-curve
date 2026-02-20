from .fancy_yield_curve import (
    calc_direct_capture_yield,
    calc_direct_capture_yield_complete,
    calc_direct_capture_yield_target_profile,
    calc_direct_capture_yield_target_profile_complete,
    add_layer,
    replace_layer,
    remove_layer,
    clear_layers,
)
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
    E_beam = np.asarray(E_beam)
    # conversions
    theta *= np.pi / 180.0
    m_a *= amu_to_kev
    m_A *= amu_to_kev
    m_B *= amu_to_kev
    q = ((m_a + m_A) - m_B) - E_x
    numer = E_beam * m_A + (q * (m_a + m_A + m_B) / 2.0)

    denom = (
        m_a + m_A + E_beam - (np.cos(theta) * np.sqrt(E_beam * (2.0 * m_a + E_beam)))
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
    def __init__(
        self,
        proj_mass,
        target_mass,
        recoil_mass,
        cs_file_name,
    ):
        self.energies_com, self.theta_com, self.cs_com = read_azure_file(cs_file_name)
        self.proj_mass = proj_mass
        self.target_mass = target_mass
        self.recoil_mass = recoil_mass
        self.energies = self.energies_com * ((proj_mass + target_mass) / target_mass)
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

    def yield_curve_components(
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
        return calc_direct_capture_yield_complete(
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


class DirectCaptureDepthProfile(DirectCapture):
    def __init__(self, proj_mass, target_mass, recoil_mass, cs_file_name):
        super().__init__(proj_mass, target_mass, recoil_mass, cs_file_name)
        self._layers = {}

    def _normalize_layers(self, layers):
        s = sum(layers)
        return [x / s for x in layers]

    def add_layer(self, layer_name, dx, layer_composition, stopping_powers):
        layer_composition = self._normalize_layers(layer_composition)
        idx = add_layer(dx, layer_composition, stopping_powers)
        self._layers[layer_name] = (idx, dx, layer_composition, stopping_powers)

    def replace_layer(self, layer_name, dx, layer_composition, stopping_powers):
        layer_composition = self._normalize_layers(layer_composition)
        idx, *_ = self._layers[layer_name]
        self._layers[layer_name] = (idx, dx, layer_composition, stopping_powers)
        replace_layer(idx, dx, layer_composition, stopping_powers)

    def remove_layer(self, layer_name):
        idx, *_ = self._layers[layer_name]
        remove_layer(idx)
        del self._layers[layer_name]
        # We need to update the index for all layers that come after
        for k, (v, *others) in self._layers.items():
            if v > idx:
                self._layers[k] = (v - 1, *others)

    def clear_layers(self):
        clear_layers()
        self._layers = {}

    def get_layer(self, layer_name):
        return self._layers[layer_name]

    def yield_curve(
        self,
        e_beam,
        beam_fwhm,
        det_fwhm,
        height,
        straggle_const,
        start,
        stop,
        step,
    ):
        return calc_direct_capture_yield_target_profile(
            e_beam,
            beam_fwhm,
            det_fwhm,
            height,
            straggle_const,
            self.energies,
            self.cs,
            start,
            stop,
            step,
        )

    def yield_curve_components(
        self,
        e_beam,
        beam_fwhm,
        det_fwhm,
        height,
        straggle_const,
        start,
        stop,
        step,
    ):
        return calc_direct_capture_yield_target_profile_complete(
            e_beam,
            beam_fwhm,
            det_fwhm,
            height,
            straggle_const,
            self.energies,
            self.cs,
            start,
            stop,
            step,
        )
