import theano
from theano import tensor as T

from teafacto.core.init import *
from lasagne.init import *
from lasagne.updates import norm_constraint


class Parameter(object):
    '''
    A parameter wraps a shared variable and can optionally have a different learning rate and regularization multiplier
    '''
    def __init__(self, value, name=None, lrmul=1., regmul=1., shape=None):
        if isinstance(value, theano.compile.sharedvalue.SharedVariable):
            self.value = value
            self.shape = value.get_value().shape
        elif isinstance(value, Initializer):
            self.initializer = value
            self.shape = shape
            self.reset()
        else:
            self.value = theano.shared(value)
            self.shape = value.shape
        self.value = value
        self.lrmul = lrmul
        self.regmul = regmul
        self.name = str(name) if name is not None else "auto:" + str(np.random.randint(0, 10000))
        self.initializer = None
        self.constraints = []

    def reset(self):
        if isinstance(self.initializer, Initializer):
            self.value = self.initializer.sample(self.shape)
        else:
            self.value = self.initializer()

    @property
    def d(self):
        return self.value

    def __repr__(self):
        return "param::%s=%s:%s-%.1f:%.1f" % (self.name, str(self.value.dtype), str(self.value.get_value().shape), self.lrmul, self.regmul)

    ############## VALUE CONSTRAINTS ############### --> applied in the order that the were added
    def clip(self, a, b):
        self.constraints.append(lambda x: T.clip(x, a, b))
        return self

    def normalize(self, axis=0, norm=2, epsilon=1e-7):
        # TODO
        return self

    def norm_constraint(self, max_norm, norm_axes=None, epsilon=1e-7):
        self.constraints.append(lambda x: norm_constraint(x, max_norm=max_norm, norm_axes=norm_axes, epsilon=epsilon))
        return self

    def constraintf(self):
        def innerconstraintf(x):
            ret = x
            for cf in self.constraints:
                ret = cf(ret)
            return ret
        return innerconstraintf



class param(object):
    def __init__(self, shape, lrmul=1., regmul=1.):
        self.shape = shape
        self.lrmul = lrmul
        self.regmul = regmul
        self.value = None

    def random(self, offset=0.5, scale=0.1):
        init = lambda: random(self.shape, offset, scale)
        ret = Parameter(init, self.lrmul, self.regmul)
        ret.initializer = init
        return ret

    ############## LASAGE INITS ################
    def _lasagne_init(self, initializer):
        return Parameter(initializer, self.lrmul, self.regmul)

    def uniform(self, range=0.01, std=None, mean=0.0):
        return self._lasagne_init(Uniform(range, std, mean))

    def normal(self, std=0.01, mean=0.0):
        return self._lasagne_init(Normal(std, mean))

    def glorotnormal(self, gain=1.0, c01b=False):
        return self._lasagne_init(GlorotNormal(gain, c01b))

    def glorotuniform(self, gain=1.0, c01b=False):
        return self._lasagne_init(GlorotUniform(gain, c01b))

    def henormal(self, gain=1.0, c01b=False):
        return self._lasagne_init(HeNormal(gain, c01b))

    def heuniform(self, gain=1.0, c01b=False):
        return self._lasagne_init(HeUniform(gain, c01b))

    def constant(self, val=0.0):
        return self._lasagne_init(Constant(val))

    def sparse(self, sparsity=0.1, std=0.01):
        return self._lasagne_init(Sparse(sparsity, std))

    def orthogonal(self, gain=1.0):
        return self._lasagne_init(Orthogonal(gain))


class Elem(object):    # carries output shape information
    def __init__(self, shape=None, name=None):
        self._shape = shape
        self._name = name
        self.parents = set()
        self.params = set()

    @property
    def dshape(self): # returns declared shape
        return self._shape

    @property
    def allparams(self):
        acc = set()
        acc.update(self.params)
        for parent in self.parents:
            acc.update(parent.allparams)
        return acc

    @property
    def allinputs(self):
        acc = set()
        if isinstance(self, Var) and len(self.parents) == 0:
            acc.update([self])
        for parent in self.parents:
            acc.update(parent.allinputs)
        return acc


class Var(Elem): # result of applying a block on theano variables
    def __init__(self, tvar, parents=None, **kw):
        super(Var, self).__init__(name=tvar.name, **kw)
        assert(isinstance(tvar, theano.Variable))
        self.tvar = tvar
        self.parents = parents if parents is not None else set()

    @property
    def d(self):
        return self.tvar

    def __repr__(self):
        return "var::%s-%s:%s" % (self._name, self.tvar.dtype, str(self._shape))


class input(Var): # generates feed + creates symbolic vars for input
    def __init__(self, ndim, dtype, name=None, **kw): # data source (numpy array)
        value = T.TensorType(dtype, (False,)*ndim)(name=name)
        super(input, self).__init__(value, parents=None, **kw)
        self.ndim = ndim # store number of dimensions

    def dimswap(self, a, b):
        dims = range(self.ndim)
        dims[a] = b
        dims[b] = a
        ret = self.d.dimshuffle(*dims)
        return Var(ret, [self])


class BlockInput(object):
    def __init__(self, name, ndim, dtype, shape=None):
        self.name = name
        self.ndim = ndim
        self.dtype = dtype
        self.shape = shape


class Block(Elem): # block with parameters
    def __init__(self, **kw):
        super(Block, self).__init__(**kw)
        self.initparams()
        self.inputs = {}

    def initparams(self):
        '''init params here'''

    def initinputs(self):
        pass

    def add_input(self, inp):
        if inp.name in self.inputs:
            raise Exception("input with that name already exists in this block")
        self.inputs[inp.name] = BlockInput(name, ndim, dtype)
        return self.inp

    def add_params(self, params):
        for param in params:
            self.add_param(param)

    def add_param(self, p): # always returns a Parameter
        if isinstance(p, Parameter):
            p = p
        elif isinstance(p, theano.compile.sharedvalue.SharedVariable): # if shared var --> wrap in a param
            p = Parameter(p)
        elif isinstance(p, np.ndarray): # numpy array
            p = Parameter(param(p))
        elif isinstance(p, tuple): # try to decode as a list of (param, lrmul, regmul) entries --> wrap in a param
            assert(isinstance(p[0], theano.compile.sharedvalue.SharedVariable))
            lrmul = 1.
            regmul = 1.
            p = p[0]
            if len(p) > 1:
                lrmul = p[1]
            if len(p) > 2:
                regmul = p[2]
            p = Parameter(p, lrmul=lrmul, regmul=regmul)
        self.params.add(p)
        return p

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def apply(self, *blocks):
        for block in blocks:
            self.parents.add(block)
        trueargs = [x.d for x in blocks]
        result = self._apply(*trueargs)
        return Var(result, parents=[self])


class wrap(Block): # wraps a theano symbolic expression into a block
    def __init__(self, fun, *params, **kw):
        super(wrap, self).__init__(**kw)
        self.add_params(params)
        assert(hasattr(fun, "__call__"))
        self.opfun = fun

    def _apply(self, *tvars):
        return self.opfun(*tvars)


class FeedForward(Block): # feedforward
    def __init__(self, indim, dim, activation=None, **kw):
        super(FeedForward, self).__init__(**kw)
        self.W = self.add_param(random((indim, dim))).d
        self.b = self.add_param(random((dim, ))).d

    def _apply(self, tvar):
        return T.dot(tvar, self.W) + self.b


if __name__ == "__main__":
    x = input(2, "int32", name="x")
    E = param((10, 10)).uniform()
    W = param((10, 10)).uniform()
    y = wrap(lambda x: E[x, :], E)(x)
    y = FeedForward(11, 12)(y)

    '''
    model = Model(y, [x])
    errors = model.train([xval], gval).cross_entropy().l2(0.001).sgd(lr) \
                  .cross_validate(5).cross_entropy.accuracy(y, g) \
                  .train()
    prediction = model.predict([xval])
    '''
    print y.allinputs
    print y.allparams
    print x

