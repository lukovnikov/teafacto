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
    def innerapply(self, x, _trainmode=False, **kw):
        return T.nnet.sigmoid(x)


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
    def __init__(self, value, ste=False, **kw):
        """
        Threshold function
        :param value:
        :param ste: use straight-through for diff
        :param kw:
        """
        self.ste = ste
        if self.ste:
            self.ste = STE_Threshold(value)
        self.value = value
        super(Threshold, self).__init__(**kw)

    def innerapply(self, x, _trainmode=False, **kw):
        if self.ste is False:
            return (x > self.value) * 1.
        else:
            return self.ste(x)

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
    def __init__(self, axes=-1, ste=False, **kw):
        self.axes = axes
        self.ste = ste
        if self.ste:
            self.ste = STE_MaxHot(axes=self.axes)
        super(MaxHot, self).__init__(**kw)

    def innerapply(self, x, _trainmode=False, **kw):
        if self.ste is False:
            retmax = T.max(x, axis=self.axes, keepdims=True)
            ret = T.eq(x, retmax)
            ret = T.cast(ret, x.dtype)
            return ret
        else:
            ret = self.ste(x)
            return ret