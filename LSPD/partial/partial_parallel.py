#
#
# This code contains sections from The ODL library
# * Availability: <https://github.com/odlgroup/odl>
#

"""Parallel beam geometries in 2 dimensions."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.discr import uniform_partition, nonuniform_partition
from odl.tomo.geometry.detector import Flat1dDetector, Flat2dDetector
from odl.tomo.geometry.geometry import Geometry, AxisOrientedGeometry
from odl.tomo.util import euler_matrix, transform_system, is_inside_bounds
from odl.util import signature_string, indent, array_str
from odl.tomo.geometry import ParallelBeamGeometry, Parallel2dGeometry, Parallel3dEulerGeometry, Parallel3dAxisGeometry, parallel_beam_geometry


__all__ = ('angle_beam_geometry', 'limited_beam_geometry', )

"""
This geometry is used for the partial operators creating a new geometry with projection angles given
"""
def angle_beam_geometry(space, angle_array, det_shape=None): #this function allows to specify a new geometry by providing the projection angles
    # Find maximum distance from rotation axis
    corners = space.domain.corners()[:, :2]
    rho = np.max(np.linalg.norm(corners, axis=1))

    # Find default values according to Nyquist criterion.

    # We assume that the function is bandlimited by a wave along the x or y
    # axis. The highest frequency we can measure is then a standing wave with
    # period of twice the inter-node distance.
    min_side = min(space.partition.cell_sides[:2])
    omega = np.pi / min_side
    num_px_horiz = 2 * int(np.ceil(rho * omega / np.pi)) + 1

    if space.ndim == 2:
        det_min_pt = -rho
        det_max_pt = rho
        if det_shape is None:
            det_shape = num_px_horiz
    elif space.ndim == 3:
        num_px_vert = space.shape[2]
        min_h = space.domain.min_pt[2]
        max_h = space.domain.max_pt[2]
        det_min_pt = [-rho, min_h]
        det_max_pt = [rho, max_h]
        if det_shape is None:
            det_shape = [num_px_horiz, num_px_vert]

    angle_partition= nonuniform_partition(angle_array, min_pt=0, max_pt=np.pi)
    det_partition = uniform_partition(det_min_pt, det_max_pt, det_shape)

    if space.ndim == 2:
        return Parallel2dGeometry(angle_partition, det_partition)
    elif space.ndim == 3:
        return Parallel3dAxisGeometry(angle_partition, det_partition)
    else:
        raise ValueError('``space.ndim`` must be 2 or 3.')


"""
This function creates a new geometry which has an upper bound possibly lower than 180 degrees
"""
def limited_beam_geometry(space, num_angles=None, max_degree=None, det_shape=None):
    # Find maximum distance from rotation axis
    corners = space.domain.corners()[:, :2] # corners come from the imagespace
    rho = np.max(np.linalg.norm(corners, axis=1))

    # Find default values according to Nyquist criterion.

    # We assume that the function is bandlimited by a wave along the x or y
    # axis. The highest frequency we can measure is then a standing wave with
    # period of twice the inter-node distance.
    min_side = min(space.partition.cell_sides[:2])
    omega = np.pi / min_side
    num_px_horiz = 2 * int(np.ceil(rho * omega / np.pi)) + 1

    if space.ndim == 2:
        det_min_pt = -rho
        det_max_pt = rho
        if det_shape is None:
            det_shape = num_px_horiz
    elif space.ndim == 3:
        num_px_vert = space.shape[2]
        min_h = space.domain.min_pt[2]
        max_h = space.domain.max_pt[2]
        det_min_pt = [-rho, min_h]
        det_max_pt = [rho, max_h]
        if det_shape is None:
            det_shape = [num_px_horiz, num_px_vert]

    if num_angles is None:
        num_angles = int(np.ceil(omega * rho))

    max_radian = max_degree * (np.pi / 180)

    angle_partition = uniform_partition(0, max_radian, num_angles)
    det_partition = uniform_partition(det_min_pt, det_max_pt, det_shape)

    if space.ndim == 2:
        return Parallel2dGeometry(angle_partition, det_partition)
    elif space.ndim == 3:
        return Parallel3dAxisGeometry(angle_partition, det_partition)
    else:
        raise ValueError('``space.ndim`` must be 2 or 3.')



if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
