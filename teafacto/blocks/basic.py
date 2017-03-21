from teafacto.core.base import Block, tensorops as T, param, Val, Var, RVal, Parameter
from teafacto.util import issequence, isfunction
from teafacto.blocks.activations import Softmax, Tanh
import numpy as np

default_carry_bias = 1


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


class MatDot(Block):
    def __init__(self, indim, dim, value=None, init="glorotuniform", dropout=False, **kw):
        super(MatDot, self).__init__(**kw)
        self.indim = indim
        self.dim = dim
        if value is None:
            self.W = param((self.indim, self.dim), name="matdot").init(init)
        else:
            self.W = value
        self.dropout = Dropout(dropout)

    def apply(self, inptensor):
        inp = self.dropout(inptensor)
        return T.dot(inp, self.W)


class Linear(Block):
    def __init__(self, indim, dim, w_init="glorotuniform",
                 b_init="uniform", dropout=False, nobias=False, **kw):
        super(Linear, self).__init__(**kw)
        self.indim = indim
        self.dim = dim
        self.w_init = w_init
        self.b_init = b_init
        self.W = param((self.indim, self.dim), name="linear_W").init(w_init)
        if nobias:
            self.b = 0.
        else:
            self.b = param((self.dim,), name="linear_b").init(b_init)
        self.dropout = Dropout(dropout)

    def apply(self, inp):
        mask = inp.mask
        inp = self.dropout(inp)
        ret = T.dot(inp, self.W) + self.b
        ret.mask = mask
        return ret


class Forward(Linear):
    def __init__(self, indim, dim, activation=Tanh(), w_init="glorotuniform",
                 b_init="uniform", dropout=False, nobias=False, **kw):
        super(Forward, self).__init__(indim, dim, w_init=w_init, b_init=b_init, dropout=dropout, nobias=nobias, **kw)
        self.activation = activation

    def apply(self, inp):
        inp = super(Forward, self).apply(inp)
        return self.activation(inp)


class ForwardHighway(Forward):
    def __init__(self, indim, dim, w_init="glorotuniform", b_init="uniform",
                 init_carry_bias=True, dropout=False, carry_activation=T.nnet.sigmoid, **kw):
        """ init_carry_bias sets carry gate bias to negative value to encourage carry behavior (see Highway Networks paper) """
        super(ForwardHighway, self).__init__(indim, dim, w_init=w_init, b_init=b_init, dropout=dropout, **kw)
        self.carry_activation = carry_activation
        self.W_t = None
        if indim != dim:
            self.W_t = param((indim, dim), name="W_t").init(self.w_init)
        self.W_c = param((self.indim, self.dim), name="carry_W").init(self.w_init)
        if init_carry_bias > 0:
            amnt = default_carry_bias if init_carry_bias is True else init_carry_bias
            self.b_c = param((self.dim,), name="carry_b").constant(-amnt)
        else:
            self.b_c = param((self.dim,), name="carry_b").init(self.b_init)

    def apply(self, inp):
        carry = self.carry_activation(T.dot(inp, self.W_c) + self.b_c)
        pre = super(ForwardHighway, self).apply(inp)
        if self.W_t is not None:
            inp = T.dot(inp, self.W_t)
        ret = pre * carry + (1 - carry) * inp
        return ret


class Embedder(Block):
    def __init__(self, indim=None, outdim=None, normalize=False, trainfrac=1., **kw):
        super(Embedder, self).__init__(**kw)
        assert(indim is not None and outdim is not None)
        self.indim = indim
        self.outdim = outdim
        self.normalize = normalize
        self.trainfrac = trainfrac

    def apply(self, idxs):
        raise NotImplementedError("use subclass")


class IdxToOneHot(Embedder):
    def __init__(self, vocsize, maskid=None, **kw):
        super(IdxToOneHot, self).__init__(vocsize, vocsize, **kw)
        self.W = Val(np.eye(vocsize, vocsize))
        self.maskid = maskid

    def apply(self, inp):
        ret = self.W[inp, :]
        if self.maskid is not None:
            ret.mask = T.neq(inp, self.maskid)
        return ret


class Eye(Block):
    def __init__(self, dim=None, **kw):
        super(Eye, self).__init__(**kw)
        self.outdim = dim

    def apply(self, inp):
        return inp


class Masker(Block):
    def __init__(self, maskid=None, **kw):
        self.maskid = maskid
        super(Masker, self).__init__(**kw)

    def apply(self, x):
        if self.maskid is not None:
            return T.eq(x, self.maskid)
        else:
            return x


class Dropout(Block):
    def __init__(self, p=0.3, seed=None, rescale=True, _alwaysrandom=False, **kw):
        super(Dropout, self).__init__(**kw)
        if seed is None:
            seed = np.random.randint(0, 1e6)
        self.p = 0.0 if (p is False or p is None) else 0.3 if p is True else p
        self.rescale = rescale
        self.seed = seed
        self._debug = _alwaysrandom
        self.rval = RVal(self.seed)

    def apply(self, x, _trainmode=False):
        if (_trainmode or self._debug) and self.p > 0:
            xmask = x.mask
            if self.rescale:
                one = T.constant(1)
                x /= one - self.p
            #rng = RVal(self.seed)
            rv = self.rval.binomial(x.shape, p=1-self.p, dtype=x.dtype)
            x = x * rv
            #print "done dropout"
            x.mask = xmask
            # x.push_extra_outs({"dropout{}".format(np.random.randint(100, 199)): rv})
            return x
        else:
            return x


class VectorEmbed(Embedder):
    def __init__(self, indim=None, dim=None, value=None,
                 normalize=False, trainfrac=1.0, init=None, maskid=None, **kw):
        super(VectorEmbed, self).__init__(indim, dim, normalize=normalize,
                                          trainfrac=trainfrac, **kw)
        self.maskid = maskid
        if value is None:
            self.W = param((indim, dim), lrmul=self.trainfrac, name="embedder")
            if init == "zero":
                self.W = self.W.constant(0.0)
            elif init in ["glorot", None]:
                self.W = self.W.glorotuniform()
            elif init == "uniform":
                self.W = self.W.uniform()
        elif value is False:
            self.W = None       # no initialization
        else:
            self.setvalue(value)
        if self.normalize:
            self.W = self.W.normalize(axis=1)
        # assertions
        if isinstance(self.W, (Parameter, Val)):
            assert(self.W.d.get_value().shape == (self.indim, self.outdim))
        elif isinstance(self.W, Var):
            assert(self.indim is not None and self.outdim is not None)
        else:
            assert(self.W is None)

    def setvalue(self, v):
        if isinstance(v, Var):
            self.W = v
        else:
            if self.trainfrac == 0.0:
                self.W = Val(v, name="embedder_val")
            else:
                self.W = Parameter(v, lrmul=self.trainfrac, name="embedder")
            self.indim, self.outdim = v.shape

    def apply(self, inptensor):
        ret = self.W[inptensor]
        self._maskfrom(ret, inptensor)
        return ret

    def _maskfrom(self, ret, x):
        if self.maskid is not None:
            mask = T.neq(x, self.maskid)
        else:
            mask = None
        ret.mask = mask

    @property
    def w(self):
        return self.W


class SMO(Block):
    def __init__(self, indim, outdim, nobias=False, **kw):
        super(SMO, self).__init__(**kw)
        self.indim = indim
        self.outdim = outdim
        self.l = Linear(indim, outdim) if not nobias else MatDot(indim, outdim)

    def apply(self, x):
        mask = x.mask
        ret = self.l(x)
        ret = Softmax()(ret)
        ret.mask = mask
        return ret


class SMOWrap(Block):   # softmax output layer
    def __init__(self, inner, outdim=2, inneroutdim=None, nobias=False, **kw):
        super(SMOWrap, self).__init__(**kw)
        self.inner = inner
        self.outdim = outdim
        inneroutdim = inner.outdim if inneroutdim is None else inneroutdim
        self.outl = Linear(inneroutdim, outdim) if not nobias else MatDot(inneroutdim, outdim)

    def apply(self, *args, **kwargs):
        vec = self.inner(*args, **kwargs)
        ret = self.outl(vec)
        return Softmax()(ret)


class Switch(Block):
    def __init__(self, a, b, mask, **kw):
        super(Switch, self).__init__(**kw)
        self.a, self.b, self.mask = a, b, mask

    def apply(self):
        return T.switch(self.mask, self.a, self.b)
