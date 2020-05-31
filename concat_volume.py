from chainer import backend
from chainer import function_node
from chainer import functions as F
from chainer.utils import type_check

import numpy as np
from functools import reduce


class ConcatVolume(function_node.FunctionNode):

    """Function that separates a given array."""

    def __init__(self, block_size, stack_shape, pad):
        self.block_size = block_size
        self.stack_shape = stack_shape
        self.pad = pad

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        for i in range(1, type_check.eval(in_types.size())):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
                in_types[0].shape == in_types[i].shape
            )

    def forward(self, xs):
        self._xp = backend.get_array_module(xs[0])
        dtype = xs[0].dtype

        if self.block_size is None:
            self.block_size = tuple(s - self.pad * 2 for s in xs[0].shape[2:])

        out = self._xp.zeros(
            xs[0].shape[:2] + tuple(b * s for b, s in zip(self.block_size, self.stack_shape)),
            dtype=dtype
        )

        for i, x in enumerate(xs):
            in_slices = [slice(x.shape[0]), slice(x.shape[1])]
            out_slices = [slice(out.shape[0]), slice(out.shape[1])]

            index = np.unravel_index(i, self.stack_shape)
            for j, s in zip(index, self.block_size):
                in_slices.append(slice(
                    self.pad, self.pad + s
                ))
                out_slices.append(slice(
                    j * s,
                    (j + 1) * s
                ))
            out[tuple(out_slices)] = x[tuple(in_slices)]
        return out,

    def backward(self, indexes, grad_outputs):
        gx, = grad_outputs
        gys = []
        for i in range(reduce(lambda x, y: x * y, self.stack_shape)):
            in_slices = [slice(gx.shape[0]), slice(gx.shape[1])]
            out_slices = []
            index = np.unravel_index(i, self.stack_shape)
            for j, s in zip(index, self.block_size):
                in_slices.append(slice(
                    j * s,
                    (j + 1) * s
                ))

            crop_gx = gx[tuple(in_slices)]
            if self.pad != 0:
                crop_gx = F.pad(crop_gx, ((0, 0), (0,0)) + ((self.pad, self.pad),) * 3, 'edge')
            gys.append(crop_gx)

        return tuple(gys)


def concat_volume_slow(xs, block_size=None, stack_shape=(2, 2, 2), pad=2):
    n_volume = reduce(lambda x, y: x * y, stack_shape)
    assert n_volume == len(xs), 'lhs: {}, rhs: {}'.format(n_volume, len(xs))

    if block_size is None:
        block_size = tuple(s - pad * 2 for s in xs[0].shape[2:])

    def get_item(x):
        block_size
        in_slices = [
            slice(x.shape[0]), slice(x.shape[1]),
            slice(pad, pad + block_size[0]),
            slice(pad, pad + block_size[1]),
            slice(pad, pad + block_size[2]),
        ]
        return x[tuple(in_slices)]

    xs = [get_item(x) for x in xs]
    ys = []
    stride = n_volume
    for i in range(3):
        ys = []
        nx = stack_shape[i]
        for j in range(0, len(xs), nx):
            cxs = tuple(xs[k] for k in range(j, j + nx))
            ys.append(F.concat(cxs, axis=xs[0].ndim-i-1))
        xs = ys

    assert len(ys) == 1

    return ys[0]


def concat_volume(xs, block_size=None, stack_shape=(2, 2, 2), pad=2):
    #y, = ConcatVolume(block_size, stack_shape, pad).apply(xs)
    y = concat_volume_slow(xs, block_size, stack_shape, pad)
    return y
