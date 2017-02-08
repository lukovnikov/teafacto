from teafacto.core.base import Block, tensorops as T, RVal
import numpy as np


class Activation(Block):
    def apply(self, x, mask=None, _trainmode=False):
        mask = x.mask if mask is None else mask
        x.mask = mask
        ret = self.innerapply(x, _trainmode=_trainmode)
        ret.mask = mask
        return ret

    def innerapply(self, x, _trainmode=False):
        raise NotImplementedError("use subclasses")


class Tanh(Activation):
    def innerapply(self, x, _trainmode=False):
        return T.tanh(x)


class Sigmoid(Activation):
    def innerapply(self, x, _trainmode=False):
        return T.nnet.sigmoid(x)


class Linear(Activation):
    def innerapply(self, x, _trainmode=False):
        return x


class ReLU(Activation):
    def innerapply(self, x, _trainmode=False):
        return T.nnet.relu(x)

'''
class Softmax(Activation):
    def innerapply(self, x):
        return T.nnet.softmax(x)'''


class Softmax(Activation):
    def __init__(self, temperature=None, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.temp = temperature

    def innerapply(self, inptensor, _trainmode=False): # matrix
        x = T.softmax(inptensor, mask=inptensor.mask, temperature=self.temp)
        x.mask = inptensor.mask
        return x


class GumbelSoftmax(Activation):
    def __init__(self, seed=None, temperature=0.3, _alwaysrandom=False, **kw):
        super(GumbelSoftmax, self).__init__(**kw)
        if seed is None:
            seed = np.random.randint(0, 1e6)
        self.seed = seed
        self.temp = temperature
        self._debug = _alwaysrandom
        self._det_sm_temp = 1e6

    def innerapply(self, x, _trainmode=False):        # x is probabilities??
        if _trainmode or self._debug:
            # sample from gumbel
            rng = RVal(self.seed)
            g = rng.gumbel(x.shape)
            y = (T.log(x) + g) / self.temp
            ret = T.softmax(y, x.mask)
            ret.mask = x.mask
            return ret
        else:
            return T.softmax(x, x.mask, temperature=self._det_sm_temp)
