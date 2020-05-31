import numpy as np

def get_hierarchy_slices(shape, pos, stride=1, stack_shape=(2, 2, 2)):
    slices = []
    block_shape = np.array(shape)
    stack_shape = np.array(stack_shape)
    for level, p in enumerate(pos):
        index = np.unravel_index(p, stack_shape)
        strides = (stride * 2 ** (len(pos) - level - 1),) * 3

        index = tuple(i * s for i, s in zip(index, strides))

        block_shape = block_shape // stack_shape

        slices.append(
            tuple(slice(i, i + s) for i, s in zip(index, block_shape))
        )
    return slices
