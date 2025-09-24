from scipy.integrate import simpson
import numpy as np


def integrate(energy_grid, yield_grid):
    return simpson(yield_grid, x=energy_grid)
