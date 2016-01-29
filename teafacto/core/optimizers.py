import theano, numpy as np
from theano import tensor as T

class Optimizer(object):
    def getupdates(self, params, grads):
        raise NotImplementedError("use a subclass")

    def onattach(self, base):
        pass

class SGD(Optimizer):
    def __init__(self, lr=0.1, **kw):
        self.lr = lr
        self.tlr = theano.shared(self.lr)
        self.numbats = 1
        super(SGD, self).__init__(**kw)

    def onattach(self, main): # called when this optimizer is attached to a base
        self.numbats = main.numbats

    def getupdates(self, params, grads):
        updates = []
        for (p, g) in zip(params, grads):
            update = (p, p - (self.lr * self.numbats * g).astype("float32"))
            updates.append(update)
        return updates

class RMSProp(Optimizer):
    def __init__(self, decay=0.9, lr=0.1, epsilon=1e-8, **kw):
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon
        super(RMSProp, self).__init__(**kw)

    def getupdates(self, params, grads):
        cache = [theano.shared(np.zeros_like(param.get_value()).astype("float32")) for param in params]
        updates = []
        for (p, g, c) in zip(params, grads, cache):
            cn = self.decay * c + (1-self.decay) * g**2
            updates.append((c, cn.astype("float32")))
            pn = p - self.lr * g / T.sqrt(cn + self.epsilon)
            updates.append((p, pn.astype("float32")))
        return updates

class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, epsilon=0.000001, **kw):
        self.lr = 1.0
        self.rho = rho
        self.epsilon = epsilon
        super(AdaDelta, self).__init__(**kw)

    def getupdates(self, params, grads):
        adadelta_egs = [theano.shared(np.zeros_like(param.get_value()).astype("float32")) for param in params]
        adadelta_edxs= [theano.shared(np.zeros_like(param.get_value()).astype("float32")) for param in params]
        updates = []
        for (p, g, eg, ed) in zip(params, grads, adadelta_egs, adadelta_edxs):
            egp = self.rho * eg + (1 - self.rho) * (g**2)
            updates.append((eg, egp.astype("float32")))
            deltap = - (T.sqrt(ed + self.epsilon) / T.sqrt(egp + self.epsilon)) * g
            updates.append((p, (p + self.lr * deltap).astype("float32")))
            edp = self.rho * ed + (1 - self.rho) * (deltap**2)
            updates.append((ed, edp.astype("float32")))
        return updates