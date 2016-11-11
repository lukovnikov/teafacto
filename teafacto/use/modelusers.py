import theano, numpy as np

from teafacto.core.base import Input, Var, Val
from teafacto.util import issequence


class ModelUser(object):
    def __init__(self, model, **kw):
        super(ModelUser, self).__init__(**kw)
        self.model = model
        self.f = None


class RecPredictor(ModelUser):
    def __init__(self, model, **kw):
        super(RecPredictor, self).__init__(model, **kw)
        self._transf = None
        self.statevars, self.statevals, self.nonseqvars, self.nonseqvals = None, None, None, None

    def init(self, *initargs):
        # pre-build
        inits = self.model.get_init_info(*initargs)
        nonseqs = []
        if isinstance(inits, tuple):
            nonseqs = inits[1]
            inits = inits[0]
        self.statevars = [self.wrapininput(x) for x in inits]
        self.nonseqvars = [self.wrapininput(x) for x in nonseqs]
        self.statevals = [self.evalstate(x) for x in inits]
        self.nonseqvals = [self.evalstate(x) for x in nonseqs]
        return self

    def transf(self, transf):
        self._transf = transf
        return self

    def build(self, inps):
        inpvars = [Input(ndim=inp.ndim, dtype=inp.dtype) for inp in inps]
        if self._transf is not None:
            tinpvars = self._transf(*inpvars)
            if not issequence(tinpvars):
                tinpvars = (tinpvars,)
            tinpvars = list(tinpvars)
        else:
            tinpvars = inpvars
        out = self.model.rec(*(tinpvars + self.statevars + self.nonseqvars))
        alloutvars = out
        self.f = theano.function(inputs=[x.d for x in inpvars + self.statevars + self.nonseqvars],
                                 outputs=[x.d for x in alloutvars],
                                 on_unused_input="warn")

    def wrapininput(self, x):
        if isinstance(x, (Var, Val)):
            return Input(ndim=x.d.ndim, dtype=x.d.dtype)
        elif isinstance(x, int):
            return Input(ndim=0, dtype="int32")

    def evalstate(self, x):
        if isinstance(x, (Var, Val)):
            return x.d.eval()
        else:
            return x

    # feed
    def feed(self, *inps):  # inps: (batsize, ...)
        if self.f is None:      # build
            self.build(inps)
        inpvals = list(inps) + self.statevals + self.nonseqvals
        outpvals = self.f(*inpvals)
        self.statevals = outpvals[1:]
        return outpvals[0]

