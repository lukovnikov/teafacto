from teafacto.core.base import Block, tensorops as T, RVal
import numpy as np


class Activation(Block):
    def apply(self, x, mask=None):
        mask = x.mask if mask is None else mask
        x.mask = mask
        ret = self.innerapply(x)
        ret.mask = mask
        return ret

    def innerapply(self, x):
        raise NotImplementedError("use subclasses")


class Tanh(Activation):
    def innerapply(self, x):
        return T.tanh(x)


class Sigmoid(Activation):
    def innerapply(self, x):
        return T.nnet.sigmoid(x)


class Linear(Activation):
    def innerapply(self, x):
        return x


class ReLU(Activation):
    def innerapply(self, x):
        return T.nnet.relu(x)

'''
class Softmax(Activation):
    def innerapply(self, x):
        return T.nnet.softmax(x)'''


class Softmax(Activation):
    def innerapply(self, inptensor): # matrix
        x = T.softmax(inptensor, inptensor.mask)
        x.mask = inptensor.mask
        return x


class GumbelSoftmax(Activation):
    def __init__(self, seed=None, temperature=0.3, **kw):
        super(GumbelSoftmax, self).__init__(**kw)
        if seed is None:
            seed = np.random.randint(0, 1e6)
        self.seed = seed
        self.temp = temperature

    def innerapply(self, x):        # x is probabilities??
        # sample from gumbel
        rng = RVal(self.seed)
        g = rng.gumbel(x.shape)
        y = (T.log(x) + g) / self.temp
        ret = T.softmax(y, x.mask)
        ret.mask = x.mask
        return ret
