#
#
# This code contains sections from The ODL library
# * Availability: <https://github.com/odlgroup/odl>
#


"""Utilities for converting ODL operators to tensorflow layers and allowing projection angles as second input."""

from __future__ import print_function, division, absolute_import
import numpy as np
import odl
import uuid
import tensorflow as tf
from tensorflow.python.framework import ops


__all__ = ('tensor_partial_layer',)


def tensor_partial_layer(odl_op, name='ODLOperator', differentiable=True):
    
    default_name = name

    def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
        if grad is None:
            return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        else:
            if stateful:
                override_name = 'PyFunc'
            else:
                override_name = 'PyFuncStateless'

            # Need to generate a unique name to avoid duplicates:
            rnd_name = override_name + 'Grad' + str(uuid.uuid4())

            tf.RegisterGradient(rnd_name)(grad)
            g = tf.get_default_graph()

            with g.gradient_override_map({override_name: rnd_name}):
                return tf.py_func(func, inp, Tout, stateful=stateful,
                                  name=name)

    def tensorflow_layer_grad_impl(x, angle_array, dy, name):
        """
        The implementation of a custom gradient function
        https: // stackoverflow.com / questions / 39048984 / tensorflow - how - to - write - op -
        with-gradient - in -python / 39984513  # 39984513"""

        model_format = int(angle_array.shape.dims[0])

        with tf.name_scope(name):
            # Validate the input/output shape
            x_shape = x.get_shape()
            dy_shape = dy.get_shape()
            try:
                # Lazy check if the first dimension is dynamic
                n_x = int(x_shape[0])
                fixed_size = True
            except TypeError:
                n_x = x_shape[0]
                fixed_size = False

            print("here  go")
            in_shape = (n_x,) + space_shape(odl_op.skeleton_range(model_format)) + (1,)

            #in_shape = (n_x,) + (128, 128,) + (1,)

            out_shape = (n_x,) + space_shape(odl_op.skeleton_domain(model_format)) + (1,)


            assert x_shape[1:] == space_shape(odl_op.skeleton_domain(model_format)) + (1,)
            assert dy_shape[1:] == space_shape(odl_op.skeleton_range(model_format)) + (1,)

            grad_out_dtype = getattr(odl_op.skeleton_range(model_format), 'dtype',
                                odl_op.skeleton_domain(model_format).dtype)


            def _impl(x, angle_array, dy):
                # Validate the shape of the given input
                if fixed_size:
                    x_out_shape = out_shape
                    assert x.shape == out_shape
                    assert dy.shape == in_shape
                else:
                    x_out_shape = (x.shape[0],) + out_shape[1:]
                    assert x.shape[1:] == out_shape[1:]
                    assert dy.shape[1:] == in_shape[1:]

                # Evaluate the operator on all inputs in the batch.
                #out = np.empty(x_out_shape, odl_op.angle_domain(angle_array).dtype)
                out = np.empty(x_out_shape, np.float32)
                out_element = odl_op.angle_domain(angle_array).element() # defining the shape

                for i in range(x_out_shape[0]):
                    #xi = x[i, ..., 0]
                    dyi = dy[i, ..., 0]
                    result = odl_op.adjoint(dyi, angle_array, out=out_element) #derivative makes no difference with just taking directly the adjoint
                    out[i, ..., 0] = np.asarray(result)
                    #x = np.asarray(out_element)

                # Rescale the domain/range according to the weighting since
                # tensorflow does not have weighted spaces.
                try:
                    dom_weight = odl_op.angle_domain(angle_array).weighting.const
                except AttributeError:
                    dom_weight = 1.0

                try:
                    ran_weight = odl_op.angle_range(angle_array).weighting.const
                except AttributeError:
                    ran_weight = 1.0

                print("dom_weight")
                print(dom_weight)
                print("ran_weight")
                print(ran_weight)
                print(odl_op.angle_range(angle_array))
                scale = dom_weight / ran_weight
                out *= scale

                print("here I am")
                print(out.shape)
                return out

            # wrapping the function and calling it
            with ops.name_scope(name + '_pyfunc', values=[x, angle_array, dy]) as name_call:
                result = py_func(_impl,
                                 [x, angle_array, dy],
                                 [grad_out_dtype],
                                 name=name_call,
                                 stateful=False)

                # We must manually set the output shape since tensorflow cannot
                # figure it out
                result = result[0]
                result.set_shape(out_shape)
                return result

    def tensorflow_layer(x, angle_partition, name=None): # changed such that projection angles are allowed as input
        # this is called in the first iteration of this: in orer
        # using the angle_partition shape in order to call the order

        model_format = int(angle_partition.shape.dims[0])

        if name is None:
            name = default_name

        with tf.name_scope(name):
            # Validate input shape
            x_shape = x.get_shape()
            try:
                # Lazy check if the first dimension is dynamic
                n_x = int(x_shape[0])
                fixed_size = True
            except TypeError:
                n_x = x_shape[0]
                fixed_size = False

            in_shape = (n_x,) + space_shape(odl_op.skeleton_domain(model_format)) + (1,)
            if odl_op.is_functional:
                print("functional")
                out_shape = (n_x,)
            else:
                print("not functional")
                out_shape = (n_x,) + space_shape(odl_op.skeleton_range(model_format)) + (1,)

            assert x_shape[1:] == space_shape(odl_op.skeleton_domain(model_format)) + (1,)

            a = x_shape[1:]
            b = space_shape(odl_op.skeleton_domain(model_format)) + (1,)

            out_dtype = getattr(odl_op.skeleton_range(model_format), 'dtype',
                                odl_op.skeleton_domain(model_format).dtype)

            # actual content of the function (which is wrapped inside)
            def _impl(x, angle_partition):
                if fixed_size:
                    x_out_shape = out_shape
                    assert x.shape == in_shape
                else:

                    x_out_shape = (x.shape[0],) + out_shape[1:]
                    assert x.shape[1:] == in_shape[1:]

                # Evaluate the operator on all inputs in the batch.
                out = np.empty(x_out_shape, out_dtype)
                out_element = odl_op.angle_range(angle_partition).element()

                print("out_element")
                print(out_element.shape)
                print(odl_op.angle_range(angle_partition))

                for i in range(x_out_shape[0]):
                    print("hello")
                    hola = x[i,..., 0]
                    print(np.shape(hola))
                    odl_op(x[i, ..., 0], angle_partition, out=out_element)
                    out[i, ..., 0] = np.asarray(out_element)

                return out

            # layer check if differentiable (in our case always, taking the differentiable function)
            if differentiable:
                def tensorflow_layer_grad(op, grad):
                    """Thin wrapper for the gradient."""
                    x = op.inputs[0]
                    angle_array = op.inputs[1]
                    return tensorflow_layer_grad_impl(x, angle_array, grad, name=name + '_grad'), None
            else:
                tensorflow_layer_grad = None

            # preparing to exit the function call
            with ops.name_scope(name + '_pyfunc', values=[x]) as name_call:
                result = py_func(_impl,[x, angle_partition],[out_dtype],name=name_call,stateful=False,grad=tensorflow_layer_grad)

                # We must manually set the output shape since tensorflow cannot
                result = result[0]
                result.set_shape(out_shape)
                return result

    return tensorflow_layer

def space_shape(space):
    """Return ``space.shape``, including power space base shape.

    If ``space`` is a power space, return ``(len(space),) + space[0].shape``,
    otherwise return ``space.shape``.
    """
    if isinstance(space, odl.ProductSpace) and space.is_power_space:
        print("ever me")
        return (len(space),) + space[0].shape
    else:
        print("never me")
        return space.shape


if __name__ == '__main__':
    from odl.util.testutils import run_doctests

    run_doctests()
