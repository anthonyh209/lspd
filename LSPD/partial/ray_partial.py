#
#
# This code contains sections from The ODL library
# * Availability: <https://github.com/odlgroup/odl>
#


"""Partial and Total Ray transforms."""

from __future__ import print_function, division, absolute_import
import numpy as np
import warnings
import partial

from odl.discr import DiscreteLp
from partial import PartialBase
from odl.space import FunctionSpace
from odl.tomo.geometry import (
    Geometry, Parallel2dGeometry, Parallel3dAxisGeometry, parallel_beam_geometry)
from partial.partial_parallel import (angle_beam_geometry)
from odl.space.weighting import ConstWeighting
from odl.tomo.backends import (
    ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE, SKIMAGE_AVAILABLE,
    astra_supports, ASTRA_VERSION,
    astra_cpu_forward_projector, astra_cpu_back_projector,
    AstraCudaProjectorImpl, AstraCudaBackProjectorImpl,
    skimage_radon_forward, skimage_radon_back_projector)



ASTRA_CPU_AVAILABLE = ASTRA_AVAILABLE
_SUPPORTED_IMPL = ('astra_cpu', 'astra_cuda', 'skimage')
_AVAILABLE_IMPLS = []
if ASTRA_CPU_AVAILABLE:
    _AVAILABLE_IMPLS.append('astra_cpu')
if ASTRA_CUDA_AVAILABLE:
    _AVAILABLE_IMPLS.append('astra_cuda')
if SKIMAGE_AVAILABLE:
    _AVAILABLE_IMPLS.append('skimage')

__all__ = ('PartialRay', 'PartialBackRay')


class PartialTransformBase(PartialBase):
    """Base class for partial ray transforms containing common attributes."""

    def __init__(self, variant, imagespace, **kwargs):
        variant, variant_in = str(variant).lower(), variant
        if variant not in ('forward', 'backward'):
            raise ValueError('`variant` {!r} not understood'
                             ''.format(variant_in))

        self.__variant = variant  

        if not isinstance(imagespace, DiscreteLp):
            raise TypeError('`{}` must be a `DiscreteLp` instance, got '
                            '{!r}'.format(reco_name, imagespace))
        else:
            self.__imagespace = imagespace


        # Definition of which operation library we will use to compute this operation 
        # Handle backend choice
        if not _AVAILABLE_IMPLS:
            raise RuntimeError('no ray transform back-end available; '
                               'this requires 3rd party packages, please '
                               'check the install docs')
        impl = kwargs.pop('impl', None)

        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_IMPL:
            raise ValueError('`impl` {!r} not understood'.format(impl_in))
        if impl not in _AVAILABLE_IMPLS:
            raise ValueError('{!r} back-end not available'.format(impl))

        # Cache for input/output arrays of transforms
        self.use_cache = kwargs.pop('use_cache', True)

        self.__impl = impl # impl is simply defining what backend system we use

        # Reserve name for cached properties (used for efficiency reasons)
        self._adjoint = None # this will be populated once, the adjoint is called for the first time 
        self._astra_wrapper = None

        # Extra kwargs that can be reused for adjoint etc. These must
        # be retrieved with `get` instead of `pop` above.
        self._extra_kwargs = kwargs

        # with the given information, initialise the base structure operator
        if variant == 'forward':
            super(PartialTransformBase, self).__init__(linear=True)

        elif variant == 'backward':
            super(PartialTransformBase, self).__init__(linear=True)

    @property
    def impl(self):
        """Implementation back-end for the evaluation of this operator."""
        return self.__impl

    @property
    def variant(self):
        """Geometry of this operator."""
        return self.__variant

    @property
    def dataspace(self):
        """Geometry of this operator."""
        print("dataspace")
        return self.__dataspace

    @property
    def imagespace(self):
        """Geometry of this operator."""
        print("imagespace")
        return self.__imagespace

    def spacegeometry(self, angle_partition):
        print("spacegeometry")
        return angle_beam_geometry(self.imagespace, angle_partition)

    def generate_projectspace(self, geometry):
        print("generate_projectspace")
        dtype = self.imagespace.dtype
        print("holaa")
        print(dtype)
        proj_fspace = FunctionSpace(geometry.params, out_dtype=dtype)

        if not self.imagespace.is_weighted:
            weighting = None
        elif (isinstance(self.imagespace.weighting, ConstWeighting) and
              np.isclose(self.imagespace.weighting.const,
                         self.imagespace.cell_volume)):
            extent = float(geometry.partition.extent.prod())
            size = float(geometry.partition.size)
            weighting = extent / size
        else:
            raise NotImplementedError('unknown weighting of domain')

        proj_tspace = self.imagespace.tspace_type(geometry.partition.shape, weighting=weighting, dtype=dtype)

        if geometry.motion_partition.ndim == 0:
            angle_labels = []
        if geometry.motion_partition.ndim == 1:
            angle_labels = ['$\\varphi$']
        elif geometry.motion_partition.ndim == 2:
            # TODO: check order
            angle_labels = ['$\\vartheta$', '$\\varphi$']
        elif geometry.motion_partition.ndim == 3:
            # TODO: check order
            angle_labels = ['$\\vartheta$', '$\\varphi$', '$\\psi$']
        else:
            angle_labels = None

        if geometry.det_partition.ndim == 1:
            det_labels = ['$s$']
        elif geometry.det_partition.ndim == 2:
            det_labels = ['$u$', '$v$']
        else:
            det_labels = None

        if angle_labels is None or det_labels is None:
            # Fallback for unknown configuration
            axis_labels = None
        else:
            axis_labels = angle_labels + det_labels

        proj_interp = 'nearest'
        #            proj_interp = kwargs.get('interp', 'nearest')
        return DiscreteLp(proj_fspace, geometry.partition, proj_tspace, interp=proj_interp, axis_labels=axis_labels)

    def range(self, geometry=None):
        print("range")
        if self.variant == 'forward':
            return self.generate_projectspace(geometry)
        elif self.variant == 'backward':
            return self.imagespace

    def skeleton_range(self, model_number=None):
        if self.variant == 'forward':
            geometry = parallel_beam_geometry(self.imagespace, num_angles=model_number)
            return self.generate_projectspace(geometry)
        elif self.variant == 'backward':
            return self.imagespace

    def domain(self, geometry=None):
        if self.variant == 'forward':
            return self.imagespace
        elif self.variant == 'backward':
            return self.generate_projectspace(geometry)

    def skeleton_domain(self, model_number=None):
        if self.variant == 'forward':
            return self.imagespace
        elif self.variant == 'backward':
            geometry = parallel_beam_geometry(self.imagespace, num_angles=model_number)
            return self.generate_projectspace(geometry)

    def angle_range(self, angle_array=None):
        print("angle_range")
        geometry = self.spacegeometry(angle_array)
        if self.variant == 'forward':
            return self.generate_projectspace(geometry)
        elif self.variant == 'backward':
            return self.imagespace

    def angle_domain(self, angle_array=None):
        print("angle_domain")
        geometry = self.spacegeometry(angle_array)
        if self.variant == 'forward':
            return self.imagespace
        elif self.variant == 'backward':
            return self.generate_projectspace(geometry)



    def _call(self, x, angle_array, out=None):
        """Return ``self(x[, out])``."""
        #if self.domain.is_real:
            #return self._call_real(x, out)
        return self._call_real(x, angle_array, out)

        #else:
            #raise RuntimeError('bad domain {!r}'.format(self.domain))


class PartialRay(PartialTransformBase):
    """Discrete Ray transform between L^p spaces."""

    def __init__(self, imagespace, **kwargs):
        range = kwargs.pop('range', None)
        super(PartialRay, self).__init__(imagespace=imagespace, variant='forward', **kwargs)

    def _call_real(self, x_real, angle_array, out_real):
        """Real-space forward projection for the current set-up."""
        geometry = self.spacegeometry(angle_array)
        partialrange = self.range(geometry)
        partialdomain = self.domain(geometry)

        if self.impl.startswith('astra'):
            backend, data_impl = self.impl.split('_')

            if data_impl == 'cpu':
                return astra_cpu_forward_projector(
                    x_real, geometry, partialrange.real_space, out_real)

            elif data_impl == 'cuda':
                # if self._astra_wrapper is None:
                #     astra_wrapper = AstraCudaProjectorImpl(
                #         geometry, partialdomain.real_space,
                #         partialrange.real_space)
                #     if self.use_cache:
                #         self._astra_wrapper = astra_wrapper
                # else:
                #     print("ever here?")
                #     astra_wrapper = self._astra_wrapper
                astra_wrapper = AstraCudaProjectorImpl(
                         geometry, partialdomain.real_space,
                         partialrange.real_space)

                return astra_wrapper.call_forward(x_real, out_real)
            else:
                # Should never happen
                raise RuntimeError('bad `impl` {!r}'.format(self.impl))

        else:
            # Should never happen
            raise RuntimeError('bad `impl` {!r}'.format(self.impl))


    @property
    def adjoint(self):
        """Adjoint of this operator
        """
        kwargs = self._extra_kwargs.copy()
        self._adjoint = PartialBackRay(self.imagespace,
                                          impl=self.impl,
                                          use_cache=self.use_cache,
                                          **kwargs)
        return self._adjoint  


class PartialBackRay(PartialTransformBase):
    """Adjoint of the discrete Ray transform between L^p spaces."""

    def __init__(self, imagespace, **kwargs):
        # KEEP IN MIND DOMAIN/RANGE = IMAGE/DATA_SPACE
        super(PartialBackRay, self).__init__(imagespace=imagespace, variant='backward', **kwargs)

    def _call_real(self, x_real, angle_array, out_real):

        geometry = self.spacegeometry(angle_array)
        partialrange = self.range(geometry)
        partialdomain = self.domain(geometry)

        if self.impl.startswith('astra'):
            backend, data_impl = self.impl.split('_')
            if data_impl == 'cpu':
                return astra_cpu_back_projector(x_real, geometry,
                                                partialrange.real_space,
                                                out_real)
            elif data_impl == 'cuda':
                # if self._astra_wrapper is None:
                #     astra_wrapper = AstraCudaBackProjectorImpl(
                #         geometry, partialrange.real_space,
                #         partialdomain.real_space)
                #     if self.use_cache:
                #         self._astra_wrapper = astra_wrapper
                #
                # else:
                #     print("ever here?")
                #     astra_wrapper = self._astra_wrapper

                astra_wrapper = AstraCudaBackProjectorImpl(
                         geometry, partialrange.real_space,
                         partialdomain.real_space)
                return astra_wrapper.call_backward(x_real, out_real)
            else:
                # Should never happen
                raise RuntimeError('bad `impl` {!r}'.format(self.impl))

        else:
            # Should never happen
            raise RuntimeError('bad `impl` {!r}'.format(self.impl))

    @property
    def adjoint(self):
        if self._adjoint is not None:
            return self._adjoint

        kwargs = self._extra_kwargs.copy()
        kwargs['range'] = self.imagespace 
        self._adjoint = PartialRay(self.imagespace,
                                     impl=self.impl,
                                     use_cache=self.use_cache,
                                     **kwargs)
        return self._adjoint


