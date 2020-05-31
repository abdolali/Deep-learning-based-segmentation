import chainer
import chainer.functions as F
import chainer.links as L



class CBR(chainer.Chain):
    dropout_dict = dict(
        # bayesian=bayesian_dropout,
        dropout=F.dropout,
        none=None,
    )

    def __init__(
            self, n_dims,
            in_ch, out_ch,
            ksize=4, stride=2,
            pad=1,
            bn=True, sample='down',
            activation=F.relu,
            dropout='dropout'
    ):
        self.use_bn = bn
        self.activation = activation
        self.dropout = None if dropout in CBR.dropout_dict else CBR.dropout_dict[dropout]

        w = chainer.initializers.HeNormal(0.02)

        if n_dims == 2:
            convolution = L.Convolution2D
            deconvolution = L.Deconvolution2D
        if n_dims == 3:
            convolution = L.Convolution3D
            deconvolution = L.Deconvolution3D
        if n_dims >= 4:
            convolution = L.ConvolutionND
            deconvolution = L.DeconvolutionND

        super().__init__()
        with self.init_scope():
            if sample == 'down':
                self.c = convolution(
                    in_ch, out_ch, ksize=ksize, stride=stride, pad=pad, initialW=w,
                )
            elif sample == 'up':
                self.c = deconvolution(
                    in_ch, out_ch, ksize=ksize, stride=stride, pad=pad, initialW=w,
                )
            elif callable(sample):
                self.c = sample
            else:
                raise NotImplementedError('Unknown convolution type:' + sample)
            if bn:
                self.bn = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = self.c(x)
        if self.use_bn:
            h = self.bn(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h
