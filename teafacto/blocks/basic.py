from teafacto.core.base import Block, tensorops as T, param, Val, Var, Parameter
from teafacto.util import issequence, isfunction
import numpy as np


class ConcatBlock(Block):
    def __init__(self, *blocks, **lw):
        super(ConcatBlock, self).__init__(**lw)
        self.blocks = blocks
        self.axis = lw["axis"] if "axis" in lw else 1
        self.argfun = lw["argfun"] if "argfun" in lw else None

    def apply(self, *args):  # args is a tuple of tuples of *args and **kwargs for each of the blocks in the concatenation
        res = []
        for block, arg in zip(self.blocks, args):
            if self.argfun is not None:
                arglist, argdic = self.argfun(arg)
            elif issequence(arg):
                assert(len(arg) < 3 and len(arg) > 0)
                arglist = arg[0]
                argdic = arg[1] if len(arg) > 1 else {}
            elif isinstance(arg, (Var, Val)):
                arglist = [arg]
                argdic = {}
            else:
                raise Exception("something wrong with concat's arguments: " + str(args))
            res.append(block(*arglist, **argdic))
        return T.concatenate(res, axis=self.axis)

class Softmax(Block):
    def apply(self, inptensor): # matrix
        return T.nnet.softmax(inptensor)


class MatDot(Block):
    def __init__(self, indim, dim, init="uniform", **kw):
        super(MatDot, self).__init__(**kw)
        self.indim = indim
        self.dim = dim
        self.W = param((self.indim, self.dim), name="matdot").init(init)

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


class Embedder(Block):
    def __init__(self, indim, outdim, **kw):
        super(Embedder, self).__init__(**kw)
        self.indim = indim
        self.outdim = outdim

    def apply(self, idxs):
        raise NotImplementedError("use subclass")


class IdxToOneHot(Embedder):
    def __init__(self, vocsize, **kw):
        super(IdxToOneHot, self).__init__(vocsize, vocsize, **kw)
        self.W = Val(np.eye(vocsize, vocsize))

    def apply(self, inp):
        return self.W[inp, :]


class VectorEmbed(Embedder):
    def __init__(self, indim=1000, dim=50, value=None, normalize=False, trainfrac=1.0, **kw):
        super(VectorEmbed, self).__init__(indim, dim, **kw)
        self.dim = dim
        self.indim = indim
        self.trainfrac = trainfrac
        if value is None:
            self.W = param((indim, dim), lrmul=self.trainfrac, name="embedder").glorotuniform()
        else:
            if trainfrac == 0.0:
                self.W = Val(value, name="embedder_val")
            else:
                self.W = Parameter(value, lrmul=self.trainfrac, name="embedder")
        if normalize:
            self.W = self.W.normalize(axis=1)
        # assertions
        assert(self.W.d.get_value().shape == (self.indim, self.dim))

    def apply(self, inptensor):
        return self.W[inptensor]