from teafacto.core.base import Block, tensorops as T, param, Val, Parameter
import numpy as np


class Softmax(Block):
    def apply(self, inptensor): # matrix
        return T.nnet.softmax(inptensor)


class MatDot(Block):
    def __init__(self, indim, dim, init="uniform", **kw):
        super(MatDot, self).__init__(**kw)
        self.indim = indim
        self.dim = dim
        self.W = param((self.indim, self.dim)).init(init)

    def apply(self, inptensor):
        return T.dot(inptensor, self.W)


class Linear(Block):
    def __init__(self, indim, dim, w_init="uniform", b_init="uniform", **kw):
        super(Linear, self).__init__(**kw)
        self.indim = indim
        self.dim = dim
        self.W = param((self.indim, self.dim)).init(w_init)
        self.b = param((self.dim,)).init(b_init)

    def apply(self, inp):
        return T.dot(inp, self.W) + self.b


class IdxToOneHot(Block):
    def __init__(self, vocsize, **kw):
        super(IdxToOneHot, self).__init__(**kw)
        self.W = Val(np.eye(vocsize, vocsize))

    def apply(self, inp):
        return self.W[inp, :]


class VectorEmbed(Block):
    def __init__(self, indim=1000, dim=50, value=None, normalize=False, **kw):
        super(VectorEmbed, self).__init__(**kw)
        self.dim = dim
        self.indim = indim
        if value is None:
            self.W = param((indim, dim), lrmul=1., name="embedder").uniform()
        else:
            self.W = Parameter(value, name="embedder")
        if normalize:
            self.W = self.W.normalize(axis=1)
        # assertions
        assert(self.W.d.get_value().shape == (self.indim, self.dim))

    def apply(self, inptensor):
        return self.W[inptensor, :]