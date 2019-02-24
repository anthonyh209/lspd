"""
This code contains sections from Title: <learned_primal_dual>
* Author: <Jonas Adler>
* Availability: <https://github.com/adler-j/learned_primal_dual>
"""

import numpy as np

import demandimport

with demandimport.enabled():
    import odl

__all__ = ('random_phantom', )

def random_ellipse(interior=False, spc=None):
    if interior:
        x_0 = np.random.rand() - 0.5
        y_0 = np.random.rand() - 0.5
    else:
        x_0 = 2 * np.random.rand() - 1.0
        y_0 = 2 * np.random.rand() - 1.0

    base_value = np.random.rand()
    if base_value < 0.5: # this value will determine if the ellipse is positive or negative (threshold value)
        base_value += 0.5

    return ((base_value - 0.5) * np.random.exponential(0.4),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            x_0, y_0,
            np.random.rand() * 2 * np.pi)


def random_phantom(spc, n_ellipse=50, interior=False):
    n = np.random.poisson(n_ellipse)
    ellipses = [random_ellipse(interior=interior, spc=spc) for _ in range(n)]

    return odl.phantom.ellipsoid_phantom(spc, ellipses)
