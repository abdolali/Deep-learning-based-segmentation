import chainer
import chainer.functions as F
import chainer.links as L
import math
from models.links import CBR

class SimpleUpsampleUnit(chainer.Chain):
    """
    Upsample unit described on the Table V
    This network upsamples a part of volume of the feature space.
    """
    def __init__(
        self,
        in_ch=32, out_ch=32,
        dropout='dropout',
        use_batch_norm=True
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        self.init_layers()

    def init_layers(self):
        i0 = self.in_ch
        i1 = self.in_ch // 2

        with self.init_scope():
            # setattr(self, 'd', CBR(
            #     3, self.in_ch, self.in_ch, ksize=4, stride=2, pad=2,
            #     activation=F.relu, sample='up',
            #     bn = self.use_batch_norm, dropout=self.dropout,
            #     ))
            setattr(self, 'c', CBR(
                3, self.in_ch, self.out_ch, ksize=3, stride=1, pad=0,
                activation=F.relu, sample='down',
                bn=self.use_batch_norm, dropout=self.dropout,
            ))

    def __call__(self, x):
        h = x
        h = F.unpooling_3d(
            h, ksize=4, stride=2, pad=2, cover_all=False
        )
        h = self.c(h)
        return h
