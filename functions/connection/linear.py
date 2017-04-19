from chainer import function
from chainer.utils import type_check

import numpy as np

from static_graph.static_graph import static_forward
from static_graph.static_graph import static_backward


# Notes on "static graph" feature.
# To make a function compatible with static graph, we only need to do two things:
# 1. Explicitly allocate any output arrays of `forward()` and `backward()`
# methods in the function body. (I assume/hope this can also work with cupy).
# 2. Write `static_*()` functions the implement the required computations for
# `forward()` and `backward()`. These `static_*()` function must be
# decorated with `@static_forward` (for forward function) or
# `@static_backward` (for backward function).
# These "static" functions must not return anything (only None). Therefore,
# any output arrays must be supplied as input arguments to the function as
# in this linear function below.
# That's all! Note that most of this function is unchanged from the existing
# Chainer version. Only a few minor changes were required to support static graph
# feature.

def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class LinearFunction(function.Function):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    @static_forward
    def static_linear(self, x, W, bias, y):
        """y = x*W^T + bias

        """
        x = _as_mat(x)
        np.dot(x, W.T, out=y)
        y += bias

    @static_forward
    def static_linear_no_bias(self, x, W, y):
        """y = x*W^T

        """
        np.dot(x, W.T, out=y)

    @static_backward
    def static_linear_backward(self, x, W, bias, gy, gx, gW, gbias):
        np.dot(gy, W, out=gx)
        np.dot(gy.T, x, out=gW)
        gbias[:] = gy.sum(0)

    @static_backward
    def static_linear_backward_no_bias(self, x, W, gy, gx, gW):
        # Input parameters: x, W, gy, gx.
        # Output parameters: gx, gW.
        #gx = gy.dot(W).astype(x.dtype)
        #gW = gy.T.dot(x).astype(W.dtype)
        np.dot(gy, W, out=gx)
        np.dot(gy.T, x, out=gW)


    def forward(self, inputs):
        # todo: This is only compatible with Numpy. Not yet compatible with cupy.
        x = inputs[0]
        W = inputs[1]
        # Notes:
        # In order to be compatible with the "static graph" feature, it is
        # required that all output arrays of this forward
        # function be allocated explicitly:
        y = np.empty((x.shape[0], W.shape[0])).astype(x.dtype)
        # This is required because all of the "static_*()" functions
        # use the convention that any output arrays are supplied
        # as input arguments to the function. That is because it is
        # not allowed for a "static_*()" function to return anything
        # other than `None`. The reason is to prevent dynamic allocation
        # of output arrays during execution of the static schedule
        # because it would break the model.
        if len(inputs) == 3:
            bias = inputs[2]
            # Note: `y` is the output array.
            self.static_linear(x, W, bias, y)
        else:
            # Note: `y` is the output array.
            self.static_linear_no_bias(x, W, y)
        return y,

    def backward(self, inputs, grad_outputs):
        #x = _as_mat(inputs[0])
        x = inputs[0]
        W = inputs[1]
        gy = grad_outputs[0]

        if x.ndim != 2:
            raise Exception('Only 2-dimensional x is currently supported')
        gx = np.empty(x.shape).astype(x.dtype)
        gW = np.empty(W.shape).astype(W.dtype)
        if len(inputs) == 3:
            bias = inputs[2]
            gbias = np.empty(bias.shape).astype(bias.dtype)
            self.static_linear_backward(x, W, bias, gy, gx, gW, gbias)
            return gx, gW, gbias
        else:
            self.static_linear_backward_no_bias(x, W, gy, gx, gW)
            return gx, gW

        #gx = gy.dot(W).astype(x.dtype, copy=False).reshape(inputs[0].shape)
        #gW = gy.T.dot(x).astype(W.dtype, copy=False)
        #if len(inputs) == 3:
        #    gb = gy.sum(0)
        #    return gx, gW, gb
        #else:
        #    return gx, gW


def linear(x, W, b=None):
    """Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    :math:`Y = xW^\\top + b`.

    Args:
        x (~chainer.Variable): Input variable. Its first dimension is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (~chainer.Variable): Weight variable of shape ``(M, N)``.
        b (~chainer.Variable): Bias variable (optional) of shape ``(M,)``.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Linear`

    """
    if b is None:
        return LinearFunction()(x, W)
    else:
        return LinearFunction()(x, W, b)
