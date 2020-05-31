import chainer
import chainer.functions as F
import chainer.links as L
import math
from models.links import CBR

class CascadeOutputUnit(chainer.Chain):
    """
    Output unit described on the Table VI
    This network converts feature space of upsampled volume to usual volume.
    """
    def __init__(
        self,
        in_ch=32, out_ch=3,
        n_hidden_layers=2,
        dropout='dropout',
        use_batch_norm=True
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_hidden_layers = n_hidden_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        self.init_layers()

    def init_layers(self):
        i0 = self.in_ch
        i1 = self.in_ch // 2

        with self.init_scope():
            for i in range(self.n_hidden_layers):
                setattr(self, 'c%02d' % i, CBR(
                    3, i0, i1, ksize=3, stride=1,
                    activation=F.relu, sample='down',
                    bn = self.use_batch_norm, dropout=self.dropout,
                ))

                i0 = i1
                i1 = max(i1 // 2, self.out_ch)
            setattr(
                self, 'c%02d' % self.n_hidden_layers,
                L.Convolution3D(i0, self.out_ch, 3, 1, pad=1)
            )

    def __call__(self, x):
        h = x
        for i in range(self.n_hidden_layers + 1):
            h = getattr(self, 'c%02d' % i)(h)
        return h
