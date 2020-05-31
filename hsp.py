import random
from functools import reduce

import numpy as np

import chainer
import chainer.functions as F
from chainer.backends import cuda

from models.encoder import Encoder
from models.decoder import Decoder
from models.upsample_unit import UpsampleUnit
from models.output_unit import OutputUnit
from models.cascade_output_unit import CascadeOutputUnit

from models.concat_volume import concat_volume

from models.hierarchy_misc import get_hierarchy_slices


class HierarchicalSurfacePredictor(chainer.Chain):
    """
    Hierarchical volume reconstruction from 2d image.
    A detail is see the following paper:
        C. Hane, et. al., "Hierarchical surface prediction for 3D object reconstruction",
        Proc. - 2017 Int. Conf. 3D Vision, 3DV 2017, pp. 412–420, 2018.
    """
    pad = 2
    delta_train_boundary_dicision = 1e-4
    always_boudary_dicision = None

    Encoder = Encoder
    Decoder = Decoder
    CascadeOutputUnit = CascadeOutputUnit
    UpsampleUnit = UpsampleUnit
    OutputUnit = OutputUnit

    def __init__(
        self,
        in_ch=1, out_ch=1, latent_dim=128, n_level=5,
        block_size=(16, 16, 16),
        upsample_threshold=0.1,
        upsample_prob_rule=None,
        n_output_hidden_layers=2,
        encoder=None
    ):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        if out_ch == 1:
            self.cascade_out_ch = 3
        else:
            self.cascade_out_ch = 2 * out_ch - 1
        self.n_level = n_level - 1  # zero-based level
        self.n_output_hidden_layers = n_output_hidden_layers
        self.block_size = block_size
        self.out_size = tuple(b * (2 ** n_level) for b in block_size)
        self.upsample_threshold = upsample_threshold
        self.upsample_prob = 1e-3

        with self.init_scope():
            if encoder is not None:
                self.encoder = encoder
            else:
                self.encoder = self.Encoder(
                    in_ch, latent_dim
                )
            self.decoder = Decoder()
            setattr(self, 'O%02d' % 0, self.CascadeOutputUnit(
                out_ch=self.cascade_out_ch
            ))
            for i in range(1, self.n_level):
                setattr(self, 'O%02d' % i, self.CascadeOutputUnit(
                    out_ch=self.cascade_out_ch,
                    n_hidden_layers=n_output_hidden_layers
                ))
                setattr(self, 'U%02d' % i, self.UpsampleUnit())
            setattr(self, 'O%02d' % self.n_level, self.OutputUnit(
                out_ch=out_ch,
                n_hidden_layers=n_output_hidden_layers,
            ))

    def is_upsample(self, cascade):
        if self.always_boudary_dicision is not None:
            if self.always_boudary_dicision == 'boudary':
                return True
            if self.always_boudary_dicision == 'unboundary':
                return False
            if self.always_boudary_dicision == 'random':
                return random.random() > 0.5

        if chainer.config.train and random.random() > self.upsample_prob:
            return True
        else:
            n_elem = reduce(lambda x, y: x*y, cascade.shape[1:])
            return float(F.max(cascade).data) < self.upsample_threshold

    def compute_slices(self, index, size, pad=2, num_block=(2, 2, 2)):
        slices = []
        index = np.unravel_index(index, num_block)
        for j, s, nb in zip(index, size, num_block):
            slices.append(slice(
                j * s,
                (j + 1) * s + pad * 2
            ))
        return tuple(slices)

    def __call__(self, x, save_hierarchy=False):
        if chainer.config.train:
            self.upsample_prob += self.delta_train_boundary_dicision
        xp = cuda.get_array_module(x)
        latent_vector = self.encoder(x)
        f = self.decoder(latent_vector)

        if save_hierarchy:
            hierarchy_volumes = {
                i: self.get_hierarchy_output_array(i)
                for i in range(1, self.n_level)
            }
        else:
            hierarchy_volumes = None
        output_volume = self.upsample_cascade(f, hierarchy_volumes)

        if save_hierarchy:
            output_hierarchy_volumes = self.concat_hierarchy_volumes(
                hierarchy_volumes
            )
            output_hierarchy_volumes[0] = self.get_cascade_output(
                self.O00(f)
            )[:, :, 2:18, 2:18, 2:18]

            return output_volume, output_hierarchy_volumes
        else:
            return output_volume

    def upsample_cascade(self, f, hierarchy_volumes=None, level=1, pos=[], unboundary_volume=None):

        def get_item(x, region, channel=None):
            if channel is None:
                return x[(slice(x.shape[0]), slice(x.shape[1])) + region]
            else:
                return x[(slice(x.shape[0]), channel) + region]

        for_unboundary_upsampling = unboundary_volume is not None

        upsampled_feature = None
        if not for_unboundary_upsampling:
            pred_volume = getattr(self, 'O%02d' % level)(f)
        else:
            pred_volume = self.get_cascade_output(
                unboundary_volume) if level == self.n_level else unboundary_volume

        cascade_outputs = []
        hierarchy_outputs = []
        for i in range(8):

            if level != self.n_level:
                in_slices = self.compute_slices(i, self.block_size)
                is_boundary = level == 1 or any(self.is_upsample(
                    get_item(pred_volume, in_slices, channel=j)
                ) for j in range(1, pred_volume.shape[1], 2))

                if hierarchy_volumes is not None:
                    hierarchy_outputs.append(
                        self.get_cascade_output(pred_volume)
                    )

                if is_boundary and not for_unboundary_upsampling:
                    if upsampled_feature is None:
                        upsampled_feature = getattr(self, 'U%02d' % level)(f)

                    cascade_output = self.upsample_cascade(
                        get_item(upsampled_feature, in_slices),
                        hierarchy_volumes,
                        level=level+1,
                        pos=pos+[i, ]
                    )
                else:
                    in_slices = self.compute_slices(i, self.block_size, pad=0)
                    if not for_unboundary_upsampling:
                        slices = tuple(
                            slice(self.pad, pred_volume.shape[j+2] - self.pad)
                            for j in range(pred_volume.ndim - 2)
                        )
                        _unboundary_volume = get_item(pred_volume, slices)
                    else:
                        _unboundary_volume = unboundary_volume
                        # _unboundary_volume = F.unpooling_3d(
                        #     _unboundary_volume,
                        #     ksize=2, stride=2, cover_all=False
                        # )
                    cascade_output = self.upsample_cascade(
                        None,
                        hierarchy_volumes,
                        level=level+1,
                        pos=pos+[i, ],
                        unboundary_volume=_unboundary_volume
                    )

                cascade_outputs.append(cascade_output)
            else:  # not to upsample on the final level.
                cascade_outputs.append(pred_volume)
                # hierarchy_outputs.append(cascade_output)

        if self.n_level == level and not for_unboundary_upsampling:
            pad = 2
        else:
            pad = 0

        if hierarchy_volumes is not None and level != self.n_level and len(hierarchy_outputs) != 0:
            slices = get_hierarchy_slices(hierarchy_volumes[level].shape, pos)
            roi = hierarchy_volumes[level]
            for s in slices:
                roi = roi[s]
            roi[0, 0, 0] = concat_volume(
                hierarchy_outputs, pad=0 if for_unboundary_upsampling else 2)

        return concat_volume(cascade_outputs, pad=pad)

    def concat_hierarchy_volumes(self, hierarchy_volumes):
        output_hierarchy_volumes = {}
        for k, v in hierarchy_volumes.items():
            if len(v) == 1:
                output_hierarchy_volumes[k] = v[0, 0, 0]
            else:
                output_hierarchy_volumes[k] = concat_volume(
                    v.ravel().tolist(), pad=0,
                    stack_shape=(2**(k - 1), ) * 3
                )
        return output_hierarchy_volumes

    def get_cascade_output(self, pred_volume):
        volumes = (pred_volume[:, 0, ],) + tuple(
            pred_volume[:, i + 1, ...]
            for i in range(1, pred_volume.shape[1], 2)
        )
        volumes = tuple(F.expand_dims(x, axis=1) for x in volumes)
        return F.concat(volumes, axis=1)

    def get_hierarchy_output_array(self, level):
        shape = tuple(2 ** (level-1) for i in range(3))
        a = np.empty(shape, dtype=np.object)
        return a
