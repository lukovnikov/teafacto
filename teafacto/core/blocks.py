import theano
from theano import tensor as T


class Parameter(object):
    '''
    A parameter wraps a shared variable and can optionally have a different learning rate and regularization multiplier
    '''
    def __init__(self, param, lrmul=1., regmul=1.):
        self.param = param
        self.lrmul = lrmul
        self.regmul = regmul


class Block(object): # block can be applied, may have parameters
    def __init__(self, indim=None, outdim=None, **kw): #
        super(Block, self).__init__()
        assert(outdim is not None and indim is not None)
        self.outdim = outdim
        self.indim = indim
        self.params = set()

    def add_params(self, params):
        for param in params:
            if isinstance(param, Parameter):
                self.params.add(param)
            elif isinstance(param, theano.compile.sharedvalue.SharedVariable): # if shared var --> wrap in a param
                self.params.add(Parameter(param))
            elif isinstance(param, tuple): # try to decode as a list of (param, lrmul, regmul) entries --> wrap in a param
                assert(isinstance(param[0], theano.compile.sharedvalue.SharedVariable))
                lrmul = 1.
                regmul = 1.
                p = param[0]
                if len(param) > 1:
                    lrmul = param[1]
                if len(param) > 2:
                    regmul = param[2]
                self.params.add(Parameter(p, lrmul, regmul))

    def apply(self, *arg): # takes arguments (theano variables or applications), outputs application
        raise NotImplementedError("use subclass")

    @property
    def parameters(self):
        return self.params


class Application(object): # result of applying a block on theano variables
    def __init__(self, block, res):
        assert(isinstance(block, Block))
        self.block = block
        self.result = res


class BlockWrapper(Block): # wraps a theano symbolic expression into a block

    def __init__(self, params, **kw):
        super(BlockWrapper, self).__init__(**kw)
        self.add_params(params)
        self.opfun = None

    def set_operation(self, opfun):
        self.opfun = opfun

    def apply(self, *args): # takes Application/TensorVariable arguments, returns application
        trueargs = [x.result if isinstance(x, Application) else x for x in args]
        result = self.opfun(*trueargs)
        return Application(self, result)
