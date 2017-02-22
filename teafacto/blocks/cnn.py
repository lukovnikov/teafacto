from teafacto.core.base import Block, param, tensorops as T, Val
import numpy as np
from teafacto.util import issequence
from teafacto.blocks.activations import Tanh, ReLU
from teafacto.blocks.basic import VectorEmbed


class CNNEnc(Block):
    def __init__(self, indim=100, innerdim=200, window=5,
                 poolmode="max", activation=Tanh, stride=1, **kw):
        super(CNNEnc, self).__init__(**kw)
        self._rets = {"final"}
        self.layers = []
        if not issequence(innerdim):
            innerdim = [innerdim]
        if not issequence(window):
            window = [window] * len(innerdim)
        if not issequence(activation):
            activation = [activation()] * len(innerdim)
        else:
            activation = [act() for act in activation]
        if not issequence(stride):
            stride = [stride] * len(innerdim)
        assert(len(window) == len(innerdim))
        innerdim = [indim] + innerdim
        for i in range(1, len(innerdim)):
            layer = Conv1D(indim=innerdim[i-1], outdim=innerdim[i],
                           window=window[i-1], stride=stride[i-1])
            self.layers.append(layer)
            self.layers.append(activation[i-1])
        if poolmode and poolmode != "none":
            self.poollayer = GlobalPool1D(mode=poolmode)
        else:
            self.all_outputs()

    def all_outputs(self):
        self._rets = {"all"}
        return self

    def with_outputs(self):
        self._rets.add("all")
        return self

    def apply(self, x, mask=None):
        mask = x.mask if mask is None else mask
        acc = x
        acc.mask = mask
        for layer in self.layers:
            acc = layer(acc)
        return self._get_returnings(acc)

    def _get_returnings(self, x):
        ret = tuple()
        if "final" in self._rets:
            ret = ret + (self.poollayer(x),)
        if "all" in self._rets:
            ret = ret + (x,)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret


class CNNSeqEncoder(CNNEnc):
    def __init__(self, indim=500, inpembdim=100, inpemb=None, innerdim=200,
                 window=5, poolmode="max", activation=Tanh, maskid=None, **kw):
        if inpemb is None:
            self.embedder = VectorEmbed(indim, inpembdim, maskid=maskid)
        else:
            self.embedder = inpemb
            inpembdim = inpemb.outdim
        super(CNNSeqEncoder, self).__init__(indim=inpembdim, innerdim=innerdim,
                                            window=window, poolmode=poolmode,
                                            activation=activation, **kw)

    def apply(self, x, mask=None):
        acc = self.embedder(x)
        ret = super(CNNSeqEncoder, self).apply(acc, mask=mask)
        return ret


class Conv1D(Block):
    def __init__(self, indim=None, outdim=None, window=5,
                 pad_mode="match",     # "valid", "same"
                 stride=1, filter_flip=True, **kw):
        super(Conv1D, self).__init__(**kw)
        if pad_mode == "match":      # equivalent to border_mode "half"
            pad_mode = window // 2
        elif pad_mode == "none":    # equivalent to border_mode "valid"
            pad_mode = 0
        elif pad_mode == "full":    # equivalent to border_mode "full"
            pad_mode = window - 1
        if isinstance(pad_mode, int):
            border_mode = (pad_mode, 0)
        else:
            raise Exception("invalid pad mode")
        self.border_mode = border_mode
        self.stride = stride
        self.filter_flip = filter_flip
        self.filter_shape = (outdim, indim, window, 1)
        self.filter = param(self.filter_shape, name="conv_w").glorotuniform()
        self.maskfilter_shape = (1, 1, window, 1)
        self.maskfilter = Val(np.ones(self.maskfilter_shape, dtype="float32"))

    def apply(self, x, mask=None):     # (batsize, seqlen, dim)
        mask = x.mask if mask is None else mask
        if mask is not None:    # mask must be (batsize, seqlen)
            #realm = T.cast(T.tensordot(mask, T.ones((x.shape[-1],), dtype="int32"), 0), "float32")
            x = x * mask.dimadd(2)
        input = x.dimshuffle(0, 2, 1, 'x')
        input_shape = None #input.shape
        convout = T.nnet.conv2d(input, self.filter, input_shape, self.filter_shape,
                            border_mode=self.border_mode, subsample=(self.stride, 1),
                            filter_flip=self.filter_flip)
        ret = convout[:, :, :, 0].dimshuffle(0, 2, 1)
        if mask is not None:    # compute new mask
            mask_shape = None
            maskout = T.nnet.conv2d(T.cast(mask.dimshuffle(0, "x", 1, "x"), "float32"),
                                    self.maskfilter, mask_shape, self.maskfilter_shape,
                                    border_mode=self.border_mode, subsample=(self.stride, 1),
                                    filter_flip=self.filter_flip)
            maskcrit = self.filter_shape[2] - self.border_mode[0]
            mask = T.cast(maskout[:, 0, :, 0] >= maskcrit, "int32")
        ret.mask = mask
        return T.cast(ret, "float32")


class GlobalPool1D(Block):
    def __init__(self, mode="max", **kw):
        super(GlobalPool1D, self).__init__(**kw)
        self.mode = mode

    def apply(self, x, mask=None):  # (batsize, seqlen, dim)
        mask = x.mask if mask is None else mask
        if mask is not None:
            #realm = T.tensordot(mask, T.ones((x.shape[-1],)), 0)
            if self.mode == "max":
                xmin = T.min(x)
                x = ((x - xmin) * mask.dimadd(2)) + xmin
            else:
                x = x * mask.dimadd(2)
        if self.mode == "max":
            ret = T.max(x, axis=-2)
        elif self.mode == "sum":
            ret = T.sum(x, axis=-2)
        elif self.mode == "avg":
            div = x.shape[-2] if mask is None else T.sum(mask, axis=1).dimadd(1)
            ret = T.sum(x, axis=-2) / div
        else:
            raise Exception("unknown pooling mode: {:3s}".format(self.mode))
        # ret: (batsize, dim)
        return ret


