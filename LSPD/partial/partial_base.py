#
#
# This code contains sections from The ODL library
# * Availability: <https://github.com/odlgroup/odl>
#

"""Abstract mathematical operators. The cut down version the mathematical base. Since not all aspects defined from the start"""

from __future__ import print_function, division, absolute_import
from builtins import object
import inspect
from numbers import Number, Integral
import sys
import odl.operator

from odl.set import LinearSpace, Set, Field
from odl.set.space import LinearSpaceElement
from odl.util import cache_arguments



__all__ = ('PartialBase',
           'OpTypeError', 'OpDomainError', 'OpRangeError',
           'OpNotImplementedError')


def _default_call_out_of_place(op, x, **kwargs):
    """Default out-of-place evaluation.
    """
    out = op.range.element()
    result = op._call_in_place(x, out, **kwargs)
    if result is not None and result is not out:
        raise ValueError('`op` returned a different value than `out`.'
                         'With in-place evaluation, the operator can '
                         'only return nothing (`None`) or the `out` '
                         'parameter.')
    return out


def _default_call_in_place(op, x, out, **kwargs):
    """Default in-place evaluation using ``Operator._call()``.
    """
    out.assign(op.range.element(op._call_out_of_place(x, **kwargs)))


def _function_signature(func):
    """Return the signature of a callable as a string.
    """
    if sys.version_info.major > 2:
        # Python 3 already implements this functionality
        return func.__name__ + str(inspect.signature(func))

    # In Python 2 we have to do it manually, unfortunately
    spec = inspect.getargspec(func)
    posargs = spec.args
    defaults = spec.defaults if spec.defaults is not None else []
    varargs = spec.varargs
    kwargs = spec.keywords
    deflen = 0 if defaults is None else len(defaults)
    nodeflen = 0 if posargs is None else len(posargs) - deflen

    args = ['{}'.format(arg) for arg in posargs[:nodeflen]]
    args.extend('{}={}'.format(arg, dval)
                for arg, dval in zip(posargs[nodeflen:], defaults))
    if varargs:
        args.append('*{}'.format(varargs))
    if kwargs:
        args.append('**{}'.format(kwargs))

    argstr = ', '.join(args)

    return '{}({})'.format(func.__name__, argstr)


@cache_arguments
def _dispatch_call_args(cls=None, bound_call=None, unbound_call=None,
                        attr='_call'):
    """Check the arguments of ``_call()`` or similar for conformity.
	"""
    py3 = (sys.version_info.major > 2)

    specs = ['_call(self, x[, **kwargs])',
             '_call(self, x, out[, **kwargs])',
             '_call(self, x, out=None[, **kwargs])']

    if py3:
        specs += ['_call(self, x, *, out=None[, **kwargs])']

    spec_msg = "\nPossible signatures are ('[, **kwargs]' means optional):\n\n"
    spec_msg += '\n'.join(specs)
    spec_msg += '\n\nStatic or class methods are not allowed.'

    if sum(arg is not None for arg in (cls, bound_call, unbound_call)) != 1:
        raise ValueError('exactly one object to check must be given')

    if cls is not None:
        # Get the actual implementation, including ancestors
        for parent in cls.mro():
            call = parent.__dict__.get(attr, None)
            if call is not None:
                break
        # Static and class methods are not allowed
        if isinstance(call, staticmethod):
            raise TypeError("'{}.{}' is a static method. "
                            "".format(cls.__name__, attr) + spec_msg)
        elif isinstance(call, classmethod):
            raise TypeError("'{}.{}' is a class method. "
                            "".format(cls.__name__, attr) + spec_msg)

    elif bound_call is not None:
        call = bound_call
        if not inspect.ismethod(call):
            raise TypeError('{} is not a bound method'.format(call))
    else:
        call = unbound_call

    if py3:
        # support kw-only args and annotations
        spec = inspect.getfullargspec(call)
        kw_only = spec.kwonlyargs
        kw_only_defaults = spec.kwonlydefaults
    else:
        spec = inspect.getargspec(call)
        kw_only = ()
        kw_only_defaults = {}

    signature = _function_signature(call)

    pos_args = spec.args
    if unbound_call is not None:
        # Add 'self' to positional arg list to satisfy the checker
        pos_args.insert(0, 'self')

    pos_defaults = spec.defaults
    varargs = spec.varargs

    # Variable args are not allowed
    if varargs is not None:
        raise ValueError("bad signature '{}': variable arguments not allowed"
                         "".format(signature) + spec_msg)

    if len(pos_args) not in (2, 3, 4):
        raise ValueError("bad signature '{}'".format(signature) + spec_msg)

    true_pos_args = pos_args[1:]
    if len(true_pos_args) == 1:  # 'out' kw-only
        if 'out' in true_pos_args:  # 'out' positional and 'x' kw-only -> no
            raise ValueError("bad signature '{}': `out` cannot be the only "
                             "positional argument"
                             "".format(signature) + spec_msg)
        else:
            if 'out' not in kw_only:
                has_out = out_optional = False
            elif kw_only_defaults['out'] is not None:
                raise ValueError(
                    "bad signature '{}': `out` can only default to "
                    "`None`, got '{}'"
                    "".format(signature, kw_only_defaults['out']) +
                    spec_msg)
            else:
                has_out = True
                out_optional = True

    elif len(true_pos_args) == 2:  # Both args positional
        if true_pos_args[0] == 'out':  # out must come second
            py3_txt = ' or keyword-only. ' if py3 else '. '
            raise ValueError("bad signature '{}': `out` can only be the "
                             "second positional argument".format(signature) +
                             py3_txt + spec_msg)
        elif true_pos_args[1] != 'out':  # 'out' must be 'out'
            raise ValueError("bad signature '{}': output parameter must "
                             "be called 'out', got '{}'"
                             "".format(signature, true_pos_args[1]) +
                             spec_msg)
        else:
            has_out = True
            out_optional = bool(pos_defaults)
            if pos_defaults and pos_defaults[-1] is not None:
                raise ValueError("bad signature '{}': `out` can only "
                                 "default to `None`, got '{}'"
                                 "".format(signature, pos_defaults[-1]) +
                                 spec_msg)

    elif len(true_pos_args) == 3:  # Three args positional
        if true_pos_args[0] == 'out':  # out must come third
            py3_txt = ' or keyword-only. ' if py3 else '. '
            raise ValueError("bad signature '{}': `out` can only be the "
                             "second positional argument".format(signature) +
                             py3_txt + spec_msg)
        elif true_pos_args[1] == 'out':  # 'out' must be 'out'
            py3_txt = ' or keyword-only. ' if py3 else '. '
            raise ValueError("bad signature '{}': `out` can only be the "
                             "second positional argument".format(signature) +
                             py3_txt + spec_msg)
        elif true_pos_args[2] != 'out':  # 'out' must be 'out'
            raise ValueError("bad signature '{}': output parameter must "
                             "be called 'out', got '{}'"
                             "".format(signature, true_pos_args[1]) +
                             spec_msg)
        else:
            has_out = True
            out_optional = bool(pos_defaults)
            if pos_defaults and pos_defaults[-1] is not None:
                raise ValueError("bad signature '{}': `out` can only "
                                 "default to `None`, got '{}'"
                                 "".format(signature, pos_defaults[-1]) +
                                 spec_msg)

    else:  # Too many positional args
        raise ValueError("bad signature '{}': too many positional arguments"
                         " ".format(signature) + spec_msg)

    return has_out, out_optional, spec


class PartialBase(object):

    def __new__(cls, *args, **kwargs):
        """Create a new instance."""
        call_has_out, call_out_optional, _ = _dispatch_call_args(cls)
        cls._call_has_out = call_has_out
        cls._call_out_optional = call_out_optional
        if not call_has_out:
            # Out-of-place _call
            cls._call_in_place = _default_call_in_place
            cls._call_out_of_place = cls._call
        elif call_out_optional:
            # Dual-use _call
            cls._call_in_place = cls._call_out_of_place = cls._call
        else:
            # In-place-only _call
            cls._call_in_place = cls._call
            cls._call_out_of_place = _default_call_out_of_place

        return object.__new__(cls)


    def __init__(self, linear=False):
        self.__is_linear = bool(linear)

        # Cache for efficiency since this is done in each call.
        self.__is_functional = isinstance(range, Field)

        # Mandatory out makes no sense for functionals.
        # However, we need to allow optional out to support vectorized
        # functions (which are functionals in the duck-typing sense).
        if (self.is_functional and self._call_has_out and
                not self._call_out_optional):
            raise ValueError('mandatory `out` parameter not allowed for '
                             'functionals')



    def _call(self, x, out=None, **kwargs): 
        raise NotImplementedError('this operator {!r} does not implement '
                                  '`_call`. See `Operator._call` for '
                                  'instructions on how to do this.'
                                  ''.format(self))

    @property
    def is_linear(self):
        """``True`` if this operator is linear."""
        return self.__is_linear

    @property
    def is_functional(self):
        """``True`` if this operator's range is a `Field`."""
        return self.__is_functional

    def derivative(self, point):
        """Return the operator derivative at ``point``.

        Raises
        ------
        OpNotImplementedError
            If the operator is not linear, the derivative cannot be
            default implemented.
        """
        if self.is_linear:
            return self
        else:
            raise OpNotImplementedError('derivative not implemented '
                                        'for operator {!r}'
                                        ''.format(self))


    def __call__(self, x, angle_array, out=None, **kwargs):
        """Return ``self(x[, out, **kwargs])``.
        """
        geometry = self.spacegeometry(angle_array)

        if x not in self.domain(geometry):
            try:
                x = self.domain(geometry).element(x)
            except (TypeError, ValueError):
                raise OpDomainError(
                    'unable to cast {!r} to an element of '
                    'the domain {!r}'.format(x, self.domain))

        if out is not None:  # In-place evaluation
            if out not in self.range(geometry):
                raise OpRangeError('`out` {!r} not an element of the range '
                                   '{!r} of {!r}'
                                   ''.format(out, self.range, self))

            if self.is_functional:
                raise TypeError('`out` parameter cannot be used '
                                'when range is a field')

            result = self._call_in_place(x, angle_array, out=out, **kwargs) #must include angle_partition
            if result is not None and result is not out:
                raise ValueError('`op` returned a different value than `out`. '
                                 'With in-place evaluation, the operator can '
                                 'only return nothing (`None`) or the `out` '
                                 'parameter.')

        else:  # Out-of-place evaluation
            out = self._call_out_of_place(x, angle_array, **kwargs)

            if out not in self.range(geometry):
                try:
                    out = self.range(geometry).element(out)
                except (TypeError, ValueError):
                    raise OpRangeError(
                        'unable to cast {!r} to an element of '
                        'the range {!r}'.format(out, self.range))
        return out



    def __repr__(self):
        """Return ``repr(self)``.

        The default `repr` implementation. Should be overridden by
        subclasses.
        """
        return '{}: {!r} -> {!r}'.format(self.__class__.__name__, self.domain,
                                         self.range)

    def __str__(self):
        """Return ``str(self)``.

        The default string implementation. Should be overridden by
        subclasses.
        """
        return self.__class__.__name__

    # Give a `Operator` a higher priority than any NumPy array type. This
    # forces the usage of `__op__` of `Operator` if the other operand
    # is a NumPy object (applies also to scalars!).
    # Set higher than LinearSpaceElement.__array_priority__ to handle
    # vector multiplication properly
    __array_priority__ = 2000000.0


class OpTypeError(TypeError):
    """Exception for operator type errors.

    Domain errors are raised by `Operator` subclasses when trying to call
    them with input not in the domain (`Operator.domain`) or with the wrong
    range (`Operator.range`).
    """


class OpDomainError(OpTypeError):
    """Exception for domain errors.

    Domain errors are raised by `Operator` subclasses when trying to call
    them with input not in the domain (`Operator.domain`).
    """


class OpRangeError(OpTypeError):
    """Exception for domain errors.

    Domain errors are raised by `Operator` subclasses when the returned
    value does not lie in the range (`Operator.range`).
    """


class OpNotImplementedError(NotImplementedError):
    """Exception for not implemented errors in `LinearSpace`'s.

    These are raised when a method in `LinearSpace` that has not been
    defined in a specific space is called.
    """


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()


