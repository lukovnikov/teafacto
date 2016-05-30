import theano

from teafacto.core.base import Input, Var


class ModelUser(object):
    def __init__(self, model, **kw):
        super(ModelUser, self).__init__(**kw)
        self.model = model
        self.f = None


class RecPredictor(ModelUser):
    def __init__(self, model, *buildargs, **kw):
        super(RecPredictor, self).__init__(model, **kw)
        self.statevals = None
        self.buildargs = buildargs

    def setbuildargs(self, *args):
        self.buildargs = args

    def build(self, inps):  # data: (batsize, ...)
        batsize = inps[0].shape[0]
        inits = self.model.get_init_info(*(list(self.buildargs)+[batsize]))
        inpvars = [Input(ndim=inp.ndim, dtype=inp.dtype) for inp in inps]
        statevars = [self.wrapininput(x) for x in inits]
        allinpvars = inpvars + statevars
        out = self.model.rec(*(inpvars+statevars))
        alloutvars = out
        self.f = theano.function(inputs=[x.d for x in allinpvars], outputs=[x.d for x in alloutvars], on_unused_input="warn")
        self.statevals = [self.evalstate(x) for x in inits]

    def wrapininput(self, x):
        if isinstance(x, Var):
            return Input(ndim=x.d.ndim, dtype=x.d.dtype)
        elif isinstance(x, int):
            return Input(ndim=0, dtype="int32")

    def evalstate(self, x):
        if isinstance(x, Var):
            return x.d.eval()
        else:
            return x

    def feed(self, *inps):  # inps: (batsize, ...)
        if self.f is None:      # build
            self.build(inps)
        inpvals = list(inps) + self.statevals
        outpvals = self.f(*inpvals)
        self.statevals = outpvals[1:]
        return outpvals[0]