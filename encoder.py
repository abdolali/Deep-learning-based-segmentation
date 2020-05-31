import chainer
import chainer.functions as F
import chainer.links as L
from functools import partial
from models.links import CBR

class Encoder(chainer.Chain):
    def __init__(
        self,
        in_ch, out_ch=128,
        n_layers=4,
        down='pooling',
        dropout='dropout',
        use_batch_norm=True
    ):
        """
        :param in_ch: Input channel
        :param out_ch: Output feature channel
        :param n_layers: N convolution layers
        :param down: Downsample method. this parameter support 'pooling' or 'conv'.
        :param dropout: Dropout type, currently only supported usually dropout.
        :param use_batch_norm: If true, use batch normalization after convolution.
        """
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_layers = n_layers
        self.dropout = dropout
        self.down = down
        self.use_batch_norm = use_batch_norm

        self.init_layers()

    def init_layers(self):
        i0 = self.in_ch
        i1 = 8

        if self.down == 'pooling':
            self.down_type = partial(F.max_pooling_2d, ksize=2, stride=2)
        elif self.down == 'conv':
            self.down_type = 'down'
        else:
            self.down_type = self.down

        with self.init_scope():
            for i in range(self.n_layers):
                setattr(self, 'c%02d' % i, CBR(
                    2, i0, i1, ksize=3, stride=1,
                    activation=F.leaky_relu, sample='down',
                    bn = self.use_batch_norm, dropout=self.dropout,
                ))
                setattr(self, 'd%02d' % i, CBR(
                    2, i1, i1, ksize=4, stride=2,
                    activation=F.leaky_relu, sample=self.down_type,
                    bn = self.use_batch_norm, dropout=self.dropout,
                ))
                i0 = i1
                i1 = min(i1 * 2, self.out_ch)

            setattr(self, 'fc', L.Linear(None, self.out_ch))

    def __call__(self, x):
        h = x
        for i in range(self.n_layers):
            h = getattr(self, 'c%02d' % i)(h)
            h = getattr(self, 'd%02d' % i)(h)
        return F.leaky_relu(self.fc(F.reshape(h, (x.shape[0], -1))))
