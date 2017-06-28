from teafacto.core.base import Block, tensorops as T, RVal, asblock
from teafacto.customops import STE_MaxHot, STE_Threshold

import numpy as np


class Activation(Block):
    def apply(self, x, mask=None, _trainmode=False, **kw):
        mask = x.mask if mask is None else mask
        x.mask = mask
        ret = self.innerapply(x, _trainmode=_trainmode, **kw)
        ret.mask = mask
        return ret

    def innerapply(self, x, _trainmode=False, **kw):
        raise NotImplementedError("use subclasses")


class Tanh(Activation):
    def innerapply(self, x, _trainmode=False, **kw):
        return T.tanh(x)


class Sigmoid(Activation):
    def __init__(self, slope=1, **kw):
        super(Sigmoid, self).__init__(**kw)
        self.slope = slope

    def innerapply(self, x, _trainmode=False, **kw):
        return T.nnet.sigmoid(x * self.slope)


class HardSigmoid(Activation):
    def __init__(self, slope=1, **kw):
        super(HardSigmoid, self).__init__(**kw)
        self.slope = slope

    def innerapply(self, x, _trainmode=False, **kw):
        return T.clip((self.slope * x + 1) / 2, 0, 1)


class Linear(Activation):
    def innerapply(self, x, _trainmode=False, **kw):
        return x


class ReLU(Activation):
    def innerapply(self, x, _trainmode=False, **kw):
        return T.nnet.relu(x)


class Softplus(Activation):
    def innerapply(self, x, _trainmode=False, **kw):
        return T.nnet.softplus(x)


class Threshold(Activation):
    def __init__(self, value=0.5, ste=False, slope=1, **kw):
        """
        Threshold function
        :param value:
        :param ste: use straight-through for diff: False, True, passthrough, sigmoid, hardsigmoid
        :param kw:
        """
        self.ste_f = None
        self.ste_a = None
        if ste is not False:
            self.ste_f = STE_Threshold(value)
        if ste is True:
            ste = "passthrough"
        if ste == "sigmoid":
            self.ste_a = Sigmoid(slope)
        elif ste == "hardsigmoid":
            self.ste_a = HardSigmoid(slope)
        elif ste == "passthrough" or ste is False:
            self.ste_a = None
        else:
            raise Exception("unknown ste option")
        self.value = value
        super(Threshold, self).__init__(**kw)

    def innerapply(self, x, _trainmode=False, **kw):
        if self.ste_f is None:
            return (x > self.value) * 1.
        else:
            if self.ste_a is not None:
                x = self.ste_a(x)
            return self.ste_f(x)


class StochasticThreshold(Activation):      # TODO: test for training
    def __init__(self, ste=False, slope=1, detexe=True, **kw):
        super(StochasticThreshold, self).__init__(**kw)
        if ste is True:
            ste = "passthrough"
        self.ste = ste
        self.rval = RVal()
        self.slope = slope
        self.detexe = detexe
        self.ste_f = STE_Threshold(threshold=0) if self.ste is not False else None

    def innerapply(self, x, _trainmode=False, **kw):
        if not _trainmode and self.detexe:
            noise = 0.5
        else:
            noise = self.rval.uniform(x.shape)
        if self.ste == "passthrough":
            noise = T.log(noise / (1 - noise))
            x = x - noise
        elif self.ste == "hardsigmoid":
            x = HardSigmoid(self.slope)(x) - noise
        elif self.ste == "sigmoid" or self.ste is False:
            x = Sigmoid(self.slope)(x) - noise
        else:
            raise Exception("unrecognized ste option")
        if self.ste_f is None:
            return (x > self.value) * 1.
        else:
            return self.ste_f(x)


'''
class Softmax(Activation):
    def innerapply(self, x):
        return T.nnet.softmax(x)'''


class Softmax(Activation):
    def __init__(self, temperature=1., maxhot=False, maxhot_ste=True, maxhot_pred=False, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.temp = temperature
        self.maxhot = maxhot
        if self.maxhot is True:
            self.maxhot = MaxHot(ste=maxhot_ste)
        self.maxhot_pred = maxhot_pred      # replaces by maxhot during prediction
        # TODO: test maxhot_pred and maxhot and maxhot_ste

    def innerapply(self, inptensor, _trainmode=False, **kw): # matrix
        if _trainmode is False and self.maxhot_pred:    # replace by maxhot during prediction
            return MaxHot()(inptensor)  # TODO masking properly
        x = T.softmax(inptensor, mask=inptensor.mask, temperature=self.temp)
        if self.maxhot is False:
            x = x
        else:
            x = self.maxhot(x)          # TODO masking properly
        x.mask = inptensor.mask
        return x


class GumbelSoftmax(Activation):
    def __init__(self, seed=None, shape=None, temperature=1.,
                 _alwaysrandom=False, deterministic_pred=False,
                 maxhot=False, ste=True, **kw):
        super(GumbelSoftmax, self).__init__(**kw)
        if seed is None:
            seed = np.random.randint(0, 1e6)
        self.seed = seed
        self.temp = temperature
        self._debug = _alwaysrandom
        self._det_sm_temp = 1e-2
        self._shape = shape
        self.rval = RVal(self.seed)
        self.detpred = deterministic_pred
        self.maxhot = maxhot
        if self.maxhot:
            self.maxhot = MaxHot(ste=ste)

    def innerapply(self, x, temps=None, _trainmode=False, **kw):        # x is probabilities??
        temp = temps.dimadd(temps.ndim) if temps is not None else self.temp
        if (not self.detpred) or (_trainmode or self._debug):
            shap = self._shape if self._shape is not None else x.shape
            g = self.rval.gumbel(shap)
            y = (x + g) / temp
            #y = x / self.temp
            ret = T.softmax(y, x.mask)
            ret.mask = x.mask
        else:   # deterministic prediction
            ret = T.softmax(x / temp, mask=x.mask, temperature=self._det_sm_temp)
            ret.mask = x.mask
        if self.maxhot is False:
            return ret
        else:
            ret = self.maxhot(ret)
            return ret


class MaxHot(Activation):
    EPS = 0
    def __init__(self, axes=-1, ste=False, **kw):
        self.axes = axes
        self.ste = ste
        if self.ste:
            self.ste = STE_MaxHot(axes=self.axes)
        super(MaxHot, self).__init__(**kw)

    def innerapply(self, x, _trainmode=False, **kw):
        mask = x.mask
        x = x - T.min(x, axis=0, keepdims=True)
        if mask is not None:
            assert(x.ndim == mask.ndim)
            x = x * mask
        if self.ste is False:
            retmax = T.max(x, axis=self.axes, keepdims=True)
            ret = T.eq(x, retmax)
            ret = T.cast(ret, x.dtype)
            if mask is not None:
                ret = ret * mask        # ensure when multiple selected, masked ones are 0
            return ret
        else:
            ret = self.ste(x)
            ret += self.EPS
            ret = T.clip(ret, 0, 1-self.EPS)
            return ret