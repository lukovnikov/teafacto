import theano, numpy as np

from teafacto.core.base import Input, Var, Val
from teafacto.util import issequence


class ModelUser(object):
    def __init__(self, model, **kw):
        super(ModelUser, self).__init__(**kw)
        self.model = model
        self.f = None


class RecPredictor(ModelUser):
    def __init__(self, model, *buildargs, **kw):
        super(RecPredictor, self).__init__(model, **kw)
        self.statevals = None
        self.nonseqvals = None
        self.buildargs = buildargs
        self.buildkwargs = kw
        self.transf = None
        self.startsym = kw["startsym"] if "startsym" in kw else 0

    def reset(self):
        self.buildargs = []
        self.buildkwargs = {}
        self.statevals = None
        self.nonseqvals = None
        self.transf = None
        self.f = None

    def setbuildargs(self, *args):
        self.buildargs = args

    def setbuildkwargs(self, **kwargs):
        self.buildkwargs = kwargs

    def settransform(self, f):
        self.transf = f

    def build(self, inps):  # data: (batsize, ...)
        batsize = inps[0].shape[0]
        inits = self.model.get_init_info(*(list(self.buildargs)+[batsize]))
        nonseqs = []
        if isinstance(inits, tuple):
            nonseqs = inits[1]
            inits = inits[0]
        inpvars = [Input(ndim=inp.ndim, dtype=inp.dtype) for inp in inps]
        if self.transf is not None:
            tinpvars = self.transf(*inpvars)
            if not issequence(tinpvars):
                tinpvars = (tinpvars,)
            tinpvars = list(tinpvars)
        else:
            tinpvars = inpvars
        statevars = [self.wrapininput(x) for x in inits]
        nonseqvars = [self.wrapininput(x) for x in nonseqs]
        out = self.model.rec(*(tinpvars + statevars + nonseqvars))
        alloutvars = out
        self.f = theano.function(inputs=[x.d for x in inpvars + statevars + nonseqvars],
                                 outputs=[x.d for x in alloutvars],
                                 on_unused_input="warn")
        self.statevals = [self.evalstate(x) for x in inits]
        self.nonseqvals = [self.evalstate(x) for x in nonseqs]

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

    def feed(self, *inps):  # inps: (batsize, ...)
        if self.f is None:      # build
            self.build(inps)
        inpvals = list(inps) + self.statevals + self.nonseqvals
        outpvals = self.f(*inpvals)
        self.statevals = outpvals[1:]
        return outpvals[0]

