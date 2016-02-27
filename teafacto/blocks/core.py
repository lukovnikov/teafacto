import theano
from theano import tensor
from theano.tensor.var import _tensor_py_operators
from types import ModuleType

from teafacto.blocks.util import *
from teafacto.blocks.trainer import ModelTrainer, ContrastModelTrainer

from lasagne.init import *
from lasagne.updates import norm_constraint

### DON'T WORRY ABOUT THIS
class TWrapper(type):
    def __getattr__(cls, item):
        top = getattr(tensor, item)
        return wrapf(top)

    def scan(cls, fn, sequences=None, outputs_info=None, non_sequences=None, n_steps=None, truncate_gradient=-1, go_backwards=False,mode=None, name=None, profile=False, allow_gc=None, strict=False):
        return scan()(fn, sequences=sequences, outputs_info=outputs_info, non_sequences=non_sequences, n_steps=n_steps,
                      truncate_gradient=truncate_gradient, go_backwards=go_backwards,mode=mode, name=name, profile=profile,
                      allow_gc=allow_gc, strict=strict)

    def until(cls, expr):
        return until(expr)

    def as_long_as(cls, expr):
        return until(cls.xor(expr, 1))     # xor?


def wrapf(attr, root=None):
    if isfunction(attr): # real function
        innerwrap = lambda *args, **kwargs: fwrap(attr, root, *args, **kwargs)
    elif isnumber(attr) or isstring(attr): # or other literals/non-syms/modules/properties/...
        return attr
    elif isinstance(attr, ModuleType):
        innerwrap = pwrap(attr)
    elif isinstance(attr, theano.Variable):
        innerwrap = vwrap(attr, root)
    else:
        innerwrap = attr
    return innerwrap


def vwrap(attr, root):
    return Var(attr, parent=root)


def fwrap(attr, root, *args, **kwargs):
    params = [x for x in args if isinstance(x, Parameter)]
    kwparams = [x for x in kwargs.values() if isinstance(x, Parameter)]
    wrapper = wrap(lambda *args, **kwargs: attr(*args, **kwargs), *(params+kwparams))
    ret = wrapper(*args, **kwargs)
    if root is not None:
        if isinstance(root, (Var, Val)):
            wrapper.add_parent(root)
        elif isinstance(root, Parameter):
            wrapper.add_param(root)
    return ret

def pwrap(attr):
    return WrappedAttr(attr)

class WrappedAttr():
    def __init__(self, attr):
        self.attr = attr

    def __getattr__(self, item):
        return wrapf(getattr(self.attr, item))


class tensorops:
    __metaclass__ = TWrapper

class TensorWrapper(type):
    """Wrapper class that provides proxy access to an instance of some
       internal instance."""

    __ignore__ = "class mro new init setattr getattr getattribute getstate setstate"

    def __init__(cls, name, bases, dct):

        def make_proxy(name):
            def proxy(self, *args):
                attr = getattr(self.d, name)
                return wrapf(attr, root=self)
            return proxy

        ignore = set("__%s__" % n for n in cls.__ignore__.split())
        for name in dir(_tensor_py_operators):
            if name.startswith("__"):
                if name not in ignore and name not in dct:
                    setattr(cls, name, property(make_proxy(name)))
        type.__init__(cls, name, bases, dct)


class TensorWrapped(object):
    __metaclass__ = TensorWrapper

    def __getattr__(self, item):
        if item in ["__%s__" % a for a in self.__metaclass__.__ignore__.split(" ")]:
            raise AttributeError()
        ret = getattr(self.d, item)

        return wrapf(ret, root=self)

    def dimswap(self, a, b):
        def tinner(v, a, b):
            dims = range(v.ndim)
            dims[a] = b
            dims[b] = a
            return v.dimshuffle(*dims)
        return wrap(tinner)(self, a, b)



### WORRY ABOUT THIS
class Parameter(TensorWrapped):
    '''
    A parameter wraps a shared variable and can optionally have a different learning rate and regularization multiplier
    '''
    def __init__(self, value, name=None, lrmul=1., regmul=1., shape=None):
        self.initializer = None
        if isinstance(value, theano.compile.sharedvalue.SharedVariable):
            self.value = value
            self.shape = value.get_value().shape
            self.initializer = lambda: value.get_values()
        elif isinstance(value, Initializer):
            self.shape = shape
            self.initializer = lambda: value.sample(shape)
            self.value = theano.shared(np.zeros(shape))
            self.reset()
        elif isinstance(value, Val):
            self.value = value.d
            self.shape = value.d.get_value().shape
            self.initializer = lambda: value.d.get_value()
        else:
            self.value = theano.shared(value)
            self.initializer = lambda: value
            self.shape = value.shape
        self.lrmul = lrmul
        self.regmul = regmul
        self.name = str(name) if name is not None else "auto" + str(np.random.randint(0, 10000))
        self.constraints = []

    def applyonval(self, f):
        self.value.set_value(f(self.value.get_value()))
        return self

    def reset(self):
        self.value.set_value(self.initializer())

    @property
    def d(self):
        return self.value

    def __repr__(self):
        return "param::'%s':%s%s" % (str(self.name), str(self.value.dtype), str(self.value.get_value().shape))

    ############## VALUE CONSTRAINTS ############### --> applied in the order that the were added
    def clip(self, a, b):
        self.constraints.append(lambda x: tensor.clip(x, a, b))
        return self

    def normalize(self, axis=0, norm=2, epsilon=1e-7):
        self.constraints.append(lambda x: (x.T/(x.norm(norm, axis=axis)+epsilon)).T) # TODO
        return self

    def norm_constraint(self, max_norm, norm_axes=None, epsilon=1e-7):
        self.constraints.append(lambda x: norm_constraint(x, max_norm=max_norm, norm_axes=norm_axes, epsilon=epsilon))
        return self

    def constraintf(self):
        cs = self.constraints
        def innerconstraintf(x):
            ret = x
            for cf in cs:
                ret = cf(ret)
            return ret
        return innerconstraintf

    @property
    def allparams(self):
        return {self}


class param(object):
    def __init__(self, shape, lrmul=1., regmul=1., name=None):
        self.shape = shape
        self.lrmul = lrmul
        self.regmul = regmul
        self.value = None
        self.name = name

    def _init_help(self, f):
        ret = Parameter(f(self.shape), lrmul=self.lrmul, regmul=self.regmul, name=self.name)
        ret.initializer = f
        return ret

    def init(self, arg, *args, **kwargs):
        if isstring(arg):
            assert hasattr(self, arg)
            return getattr(self, arg)(*args, **kwargs)
        elif isfunction(arg):
            return self._init_help(arg)

    ############## OWN INITS ###################
    def random(self, offset=0.5, scale=0.1):
        return self._init_help(lambda shape: (np.random.random(shape).astype("float32")-offset)*scale)

    def eye(self, offset=0):
        return self._init_help(lambda shape: np.eye(shape[0], shape[1], k=offset, dtype="float32"))

    ############## LASAGE INITS ################
    def _lasagne_init(self, initializer):
        return Parameter(initializer, lrmul=self.lrmul, regmul=self.regmul, shape=self.shape, name=self.name)

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


class Val(TensorWrapped):
    def __init__(self, value, name=None, **kw):
        super(Val, self).__init__(**kw)
        self.name = name
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        self.value = theano.shared(value, name=name)

    @property
    def d(self):
        return self.value

    @property
    def v(self):
        return self.value.get_value()

    @property
    def allparams(self): # TODO: can Vals have parents?
        return set()


### DON'T WORRY ABOUT THIS
class Elem(object):    # carries output shape information
    def __init__(self, shape=None, name=None, **kw):
        super(Elem, self).__init__()
        self._shape = shape
        self._name = name
        self.parents = []

    @property
    def dshape(self): # returns declared shape
        return self._shape

    def add_parent(self, p):
        self.parents.append(p)

    def getparents(self):
        return self.parents

    @property
    def allparams(self):
        acc = set()
        if hasattr(self, "params"):
            acc.update(set(self.params))
        for parent in self.getparents():
            acc.update(parent.allparams)
        return acc


### WORRY ABOUT THIS
class Var(Elem, TensorWrapped): # result of applying a block on theano variables
    def __init__(self, value, parent=None, **kw):
        super(Var, self).__init__(name=value.name, **kw)
        assert(isinstance(value, theano.Variable))
        self.value = value
        if parent is not None:
            self.add_parent(parent)

    def eval(self, argdic={}):
        return self.d.eval(dict(map(lambda (x, y): (x.d, y), argdic.items())))

    @property
    def d(self):
        return self.value

    def __repr__(self):
        return "var::%s-%s:%s" % (self._name, self.value.dtype, str(self._shape))


class Input(Var): # generates feed + creates symbolic vars for input
    def __init__(self, ndim, dtype, name=None, **kw): # data source (numpy array)
        value = tensor.TensorType(dtype, (False,) * ndim)(name=name)
        super(Input, self).__init__(value, parent=None, **kw)
        self.ndim = ndim # store number of dimensions


def recurmap(fun, data):
    if isinstance(data, dict):
        return type(data)(dict([(recurmap(fun, item[0]), recurmap(fun, item[1])) for item in data.items()]))
    elif isinstance(data, (tuple, list, set)):
        return type(data)([recurmap(fun, elem) for elem in data])
    else:
        return fun(data)


class Block(Elem, Saveable): # block with parameters
    def __init__(self, **kw):
        super(Block, self).__init__(**kw)
        self.inputs = []
        self.params = []
        self.output = None
        self._predictf = None

    def apply(self, *vars, **kwargs):
        trueargs = recurmap(lambda x: x.d if hasattr(x, "d") else x, vars)
        truekwargs = recurmap(lambda x: x.d if hasattr(x, "d") else x, kwargs)
        result = self._apply(*trueargs, **truekwargs)
        return Var(result)#, parent=self)

    # may override: -------------------------------------------------
    def predict(self, *inputdata):
        if self._predictf is None:
            #if False or len(self.inputs) == 0 or self.output is None:
            self.autobuild(*inputdata)
            self._predictf = theano.function(outputs=self.output.d, inputs=[x.d for x in self.inputs])
        args = [np.asarray(x) if not isinstance(x, np.ndarray) else x for x in inputdata]
        return self._predictf(*args)

    def gettrainer(self, goldvar):
        return ModelTrainer(self, goldvar)

    # do not override ------------------------------------------------
    def wrapply(self, *args, **kwargs): # is this multi-output compatible?
        self.parents.extend(recurfilter(lambda x: isinstance(x, (Var, Val)), args))
        self.parents.extend(recurfilter(lambda x: isinstance(x, (Var, Val)), kwargs))
        ret = self.apply(*args, **kwargs)
        possiblechildren = recurfilter(lambda x: isinstance(x, Elem), ret)
        for p in possiblechildren:
            p.add_parent(self)
        return ret

    def build(self): # stores block inputs and block output
        self.inputs = self.initinputs()
        self._build(*self.inputs)

    def _build(self, *inps):
        self.output = self.wrapply(*inps)

    def autobuild(self, *inputdata):
        inputdata = map(lambda x: x if isinstance(x, np.ndarray) else np.asarray(x), inputdata)
        self.inputs = [Input(ndim=td.ndim, dtype=td.dtype) for td in inputdata]
        self._build(*self.inputs)

    def __call__(self, *args, **kwargs):
        return self.wrapply(*args, **kwargs)

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
        self.params.append(p)
        return p

    def train(self, inputdata, gold):
        # wrap data in datafeeds, generate gold var
        goldvar = Input(gold.ndim, gold.dtype, name="gold")
        self.autobuild(*inputdata)
        trainer = self.gettrainer(goldvar.d)
        trainer.traindata = inputdata
        trainer.traingold = gold
        return trainer


def asblock(f):
    def inner(*args):
        return f(*args)
    return inner



def recurfilter(fun, data):
    acc = []
    if isinstance(data, dict):
        data = data.items()
    if isinstance(data, (tuple, list, set)):
        for elem in data:
            ret = recurfilter(fun, elem)
            acc.extend(ret)
    else:
        if fun(data):
            acc.append(data)
        else:
            acc.append(None)
    return filter(lambda x: x is not None, acc)


class wrap(Block): # wraps a theano symbolic expression into a block
    def __init__(self, fun, *params, **kw):
        super(wrap, self).__init__(**kw)
        self.add_params(params)
        assert(hasattr(fun, "__call__"))
        self.opfun = fun

    def _apply(self, *tvars, **kwargs):
        return self.opfun(*tvars, **kwargs)


class scan(Block):
    def __init__(self, **kw):
        super(scan, self).__init__(**kw)
        # set params

    def fnwrap(self, fn): # enables writing fn in blocks level
        scanblock = self
        def fwrapper(*args): # theano vars
            trueargs = [Var(x) for x in args]
            res = fn(*trueargs)
            ret = tuple(recurmap(lambda x: x.d if hasattr(x, "d") else x, res))
            newparents = recurfilter(lambda x: isinstance(x, Var) or isinstance(x, Val), res) + \
                         recurfilter(lambda x: isinstance(x, until), res)
            for npa in newparents:
                scanblock.add_parent(npa)
            #self.add_params(reduce(lambda x, y: set(x).union(set(y)),
            #                       map(lambda x: x.allparams, recurfilter(lambda x: isinstance(x, Var), res)), set()))
            #self.add_params(recurfilter(lambda x: isinstance(x, Parameter), res))
            return ret
        return fwrapper

    def apply(self, fn, **kwargs):
        self.params.extend(recurfilter(lambda x: isinstance(x, Parameter), kwargs))
        trueargs = recurmap(lambda x: x.d if hasattr(x, "d") else x, kwargs)
        o, updates = theano.scan(self.fnwrap(fn), **trueargs)
        ret = [Var(oe) for oe in o] if issequence(o) else Var(o)
        return ret, updates

class until(Elem):
    def __init__(self, expr, **kw):
        super(until, self).__init__(**kw)
        self.add_parent(expr)
        self.expr = expr

    @property
    def d(self): # wrap theano.scan_module.until(cond)
        return theano.scan_module.until(self.expr.d)


class Embed(Block):
    def __init__(self, indim=1000, dim=50, **kw):
        super(Embed, self).__init__(**kw)
        self.dim = dim
        self.indim = indim
        self.W = param((indim, dim), lrmul=1., name="embedder").uniform().normalize(axis=1)

    def apply(self, inptensor):
        return self.W[inptensor, :]
