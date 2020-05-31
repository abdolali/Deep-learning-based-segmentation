import chainer
import chainer.functions as F
import chainer.links as L
import math
from models.links import CBR

class Decoder(chainer.Chain):
    def __init__(
        self,
        n_elem=1024, in_ch=128, out_ch=32,
        n_layers=4,
        upsample='conv',
        dropout='dropout',
        use_batch_norm=True
    ):
        super().__init__()
        self.n_elem = n_elem
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_layers = n_layers
        self.dropout = dropout
        self.upsample = upsample
        self.use_batch_norm = use_batch_norm

        self.init_layers()

    def init_layers(self):
        n_step = int(math.log2(self.in_ch / self.out_ch))
        i0 = self.in_ch
        i1 = self.in_ch if n_step > 0 else self.out_ch

        pad = 1
        with self.init_scope():
            for i in range(self.n_layers):
                index = self.n_layers - i - 1
                setattr(self, 'c%02d' % index, CBR(
                    3, i0, i1, ksize=3, stride=1, pad=pad,
                    activation=F.relu, sample='down',
                    bn = self.use_batch_norm, dropout=self.dropout,
                ))
                setattr(self, 'd%02d' % index, CBR(
                    3, i1, i1, ksize=4, stride=2, pad=1,
                    activation=F.relu, sample='up',
                    bn=self.use_batch_norm, dropout=self.dropout,
                ))

                i0 = i1
                if index <= n_step:
                    i1 = max(i1 // 2, self.out_ch)
                    pad = 0
            self.c = CBR(
                3, self.out_ch, self.out_ch, ksize=3, stride=1,
                activation=F.relu, sample='down',
                bn = self.use_batch_norm, dropout=self.dropout,
            )
            setattr(self, 'fc', L.Linear(None, self.n_elem))

    def __call__(self, x):
        h = F.leaky_relu(self.fc(F.reshape(x, (x.shape[0], -1))))
        h = F.reshape(h, (x.shape[0], -1, 2, 2, 2))
        for i in range(self.n_layers - 1, -1, -1):
            h = getattr(self, 'c%02d' % i)(h)
            h = getattr(self, 'd%02d' % i)(h)
        return self.c(h)
