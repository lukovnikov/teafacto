from teafacto.core.base import Block, param, tensorops as T, Val
import numpy as np
from teafacto.util import issequence
from teafacto.blocks.activations import Tanh, ReLU
from teafacto.blocks.basic import VectorEmbed, Dropout


class CNNEnc(Block):
    def __init__(self, indim=100, innerdim=200, window=5, dropout=None,
                 poolmode="max", activation=Tanh, stride=1, **kw):
        super(CNNEnc, self).__init__(**kw)
        self._rets = {"final"}
        self.layers = []
        if not issequence(innerdim):
            innerdim = [innerdim]
        self.outdim = innerdim[-1]
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
                           window=window[i-1], stride=stride[i-1],
                           dropout=dropout)
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
    def __init__(self, inpvocsize=500, inpembdim=100, inpemb=None,
                 numpos=None, posembdim=None, posemb=None,
                 innerdim=200, stride=1,
                 window=5, poolmode="max", activation=Tanh, maskid=None,
                 dropout=None, **kw):
        if inpemb is None:
            self.embedder = VectorEmbed(inpvocsize, inpembdim, maskid=maskid)
        else:
            self.embedder = inpemb
            inpembdim = inpemb.outdim
        self.numpos = numpos
        indim = inpembdim
        if posemb is not False:
            if posemb is None or posemb is True:
                if posembdim is not None and numpos is not None:
                    self.posembmat = param((numpos, posembdim), name="pos_emb_mat").glorotuniform()
                    self.posemb = lambda x: self.posembmat
                    indim += posembdim
                else:
                    self.posemb = None
            else:
                self.posemb = posemb
                indim += self.posemb.outdim
        else:
            self.posemb = None

        super(CNNSeqEncoder, self).__init__(indim=indim, innerdim=innerdim,
                                            window=window, poolmode=poolmode, stride=stride,
                                            activation=activation, dropout=dropout, **kw)

    def apply(self, x, mask=None):
        xemb = self.embedder(x)     # (batsize, seqlen, embdim)
        xemb.mask = mask if mask is not None else xemb.mask
        xembndim = xemb.ndim
        origshape = None
        origmaskshape = None
        if xemb.ndim > 3:   # flatten
            flatmask = xemb.mask
            origshape = xemb.shape
            origmaskshape = flatmask.shape if flatmask is not None else None
            xemb = xemb.reshape((-1, xemb.shape[-2], xemb.shape[-1]))
            xemb.mask = flatmask.reshape((-1, flatmask.shape[-1])) if flatmask is not None else None
        if self.posemb is not None:
            mask = xemb.mask
            posids = Val(np.arange(0, self.numpos))
            posembone = self.posemb(posids)
            posemb = T.repeat(posembone.dimadd(0), xemb.shape[0], axis=0)
            xemb = T.concatenate([xemb, posemb], axis=2)
            xemb.mask = mask
        ret = super(CNNSeqEncoder, self).apply(xemb)
        if origshape is not None:
            newret = tuple()
            if not issequence(ret):
                ret = (ret,)
            for rete in ret:
                if rete.ndim == xemb.ndim - 1:  # lost last seq dim
                    goodshape = T.concatenate([origshape[:-2],
                                               rete.shape[-1:]], axis=0)
                    retek = rete.reshape(goodshape, ndim=xembndim - 1)
                    if origmaskshape is not None:
                        retek.mask = flatmask.reshape(origmaskshape)
                        retek.mask = T.sum(retek.mask, axis=-1) > 0   # agg over mask
                else:   # didn't lose
                    goodshape = T.concatenate([origshape[:-1],
                                               rete.shape[-1:]], axis=0)
                    retek = rete.reshape(goodshape, ndim=xembndim)
                    if origmaskshape is not None:
                        retek.mask = flatmask.reshape(origmaskshape)
                newret += (retek,)
            newret = newret[0] if len(newret) == 1 else newret
            ret = newret
        return ret


class Conv1D(Block):
    def __init__(self, indim=None, outdim=None, window=5,
                 pad_mode="match",     # "valid", "same"
                 stride=1, filter_flip=True,
                 dropout=None, **kw):
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
        self.dropout = Dropout(dropout)

    def apply(self, x, mask=None):     # (batsize, seqlen, dim)
        mask = x.mask if mask is None else mask
        x = self.dropout(x)
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
            mask = maskout[:, 0, :, 0] >= maskcrit
        #ret = T.cast(ret, "float32")
        ret.mask = mask
        return ret


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


