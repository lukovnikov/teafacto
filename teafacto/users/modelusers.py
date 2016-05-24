import theano

from teafacto.core.base import Input


class ModelUser(object):
    def __init__(self, model, **kw):
        super(ModelUser, self).__init__(**kw)
        self.model = model
        self.f = None


class RecApplicator(ModelUser):
    def __init__(self, model, **kw):
        super(RecApplicator, self).__init__(model, **kw)
        self.statevals = None

    def build(self, inps):  # data: (batsize, ...)
        batsize = inps[0].shape[0]
        inits, _ = self.model.recappl_init(batsize)
        inpvars = [Input(ndim=inp.ndim, dtype=inp.dtype) for inp in inps]
        statevars = [Input(ndim=x.d.ndim, dtype=x.d.dtype) for x in inits]
        allinpvars = inpvars + statevars
        out, states, tail = self.model.recappl(inpvars, statevars)
        alloutvars = out + states
        assert(len(tail) == 0)
        self.f = theano.function(inputs=[x.d for x in allinpvars], outputs=[x.d for x in alloutvars])
        self.statevals = [x.d.eval() for x in inits]

    def feed(self, *inps):  # inps: (batsize, ...)
        if self.f is None:      # build
            self.build(inps)
        inpvals = list(inps) + self.statevals
        outpvals = self.f(*inpvals)
        self.statevals = outpvals[1:]
        return outpvals[0]