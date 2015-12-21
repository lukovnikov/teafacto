import theano, numpy as np
from theano import tensor as T
import inspect

class RNUBase(object):
    def __init__(self, dim=20, innerdim=20, wreg=0.0001, initmult=0.1, **kw): # dim is input dimensions, innerdim = dimension of internal elements
        super(RNUBase, self).__init__(**kw)
        self.dim = dim
        self.innerdim = innerdim
        self.wreg = wreg
        self.initmult = initmult
        self.initparams()

    def initparams(self):
        params = {}
        indim = self.innerdim
        for paramname in self.paramnames:
            if paramname[0] == "b" or paramname[0] == "p": # bias or peepholes, internal weights
                shape = (self.innerdim,)
            elif paramname[0] == "w": #input processing matrices
                shape = (self.dim, self.innerdim)
            else: # internal recurrent matrices
                shape = (self.innerdim, self.innerdim)
            paramval = (np.random.random(shape).astype("float32")-0.5)*self.initmult
            params[paramname] = theano.shared(paramval, name=paramname)
            setattr(self, paramname, params[paramname])
        #self.initstate = T.zeros((self.indim,), dtype="float32")

    def getreg(self, regf=lambda x: T.sum(x**2), mult=1./2):
        return mult * reduce(lambda x, y: x+y,
                               map(lambda x: regf(getattr(self, x))*self.wreg,
                                   self.paramnames))

    def recur(self, x):
        inputs = x.dimshuffle(1, 0, 2) # inputs is (seq_len, nb_samples, dim)
        numstates = len(inspect.getargspec(self.rec).args) - 2
        initstate = T.zeros((inputs.shape[1], self.innerdim)) # (nb_samples, dim)
        outputs, _ = theano.scan(fn=self.rec,
                                 sequences=inputs,
                                 outputs_info=[None]+[initstate]*numstates)
        output = outputs[0]
        return output.dimshuffle(1, 0, 2) #output is (nb_samples, seq_len, nb_feats)

    def rec(self, *args):
        raise NotImplementedError("use subclass")

    def __call__(self, x):
        return self.getoutput(x)

    def getoutput(self, x):
        '''
        :param x: symbolic input tensor for shape (nb_samples, seq_len, nb_feats) where
            nb_samples is the number of samples (number of sequences) in the current input
            seq_len is the maximum length of the sequences
            nb_feats is the number of features per sequence element
        :return: symbolic output tensor for shape (nb_samples, seq_len, out_dim) where
            nb_samples is the number of samples (number of sequences) in the original input
            seq_len is the maximum length of the sequences
            out_dim is the dimension of the output vector as specified by the dim argument in the constructor
        '''
        return self.recur(x)

    @property
    def parameters(self):
        return map(lambda x: getattr(self, x), self.paramnames)

class RNU(RNUBase):
    def __init__(self, **kw):
        self.paramnames = ["u", "w"]
        super(RNU, self).__init__(**kw)

    def rec(self, x_t, h_tm1):
        inp = T.dot(x_t, self.w)
        rep = T.dot(h_tm1, self.u)
        h = inp + rep
        h = T.tanh(h)
        return [h, h] #T.tanh(inp+rep)


class GRU(RNUBase):
    def __init__(self, gateactivation=T.nnet.sigmoid, outpactivation=T.tanh, **kw):
        self.gateactivation = gateactivation
        self.outpactivation = outpactivation
        self.paramnames = ["uz", "wz", "ur", "wr", "u", "w", "br", "bz", "b"]
        super(GRU, self).__init__(**kw)

    def _getz(self, x_t, h_tm1):
        return self.gateactivation(T.dot(h_tm1, self.uz) + T.dot(x_t, self.wz) + self.bz)
    def _getr(self, x_t, h_tm1):
        return self.gateactivation(T.dot(h_tm1, self.ur) + T.dot(x_t, self.wr) + self.br)
    def _gethh(self, x_t, h_tm1, r):
        return self.outpactivation(T.dot(h_tm1 * r, self.u) + T.dot(x_t, self.w) + self.b)
    def _geth(self, z, hh, h_tm1):
        return z * h_tm1 + (1-z) * hh

    def rec(self, x_t, h_tm1):
        '''
        :param x_t: input values (nb_samples, nb_feats) for this recurrence step
        :param h_tm1: previous states (nb_samples, out_dim)
        :return: new state (nb_samples, out_dim)
        '''
        z = self._getz(x_t, h_tm1)
        r = self._getr(x_t, h_tm1)
        hh = self._gethh(x_t, h_tm1, r)
        h = self._geth(z, hh, h_tm1)
        return [h, h]


class LSTM(RNUBase):
    def __init__(self, gateactivation=T.nnet.sigmoid, outpactivation=T.tanh, **kw):
        self.gateactivation = gateactivation
        self.outpactivation = outpactivation
        self.paramnames = ["wf", "rf", "bf", "wi", "ri", "bi", "wo", "ro", "bo", "w", "r", "b", "pf", "pi", "po"]
        super(LSTM, self).__init__(**kw)

    def rec(self, x_t, c_tm1, y_tm1):
        fgate = self.gateactivation(c_tm1*self.pf + self.bf + T.dot(x_t, self.wf) + T.dot(y_tm1, self.rf))
        igate = self.gateactivation(c_tm1*self.pi + self.bi + T.dot(x_t, self.wi) + T.dot(y_tm1, self.ri))
        cf = c_tm1 * fgate
        ifi = self.outpactivation(T.dot(x_t, self.w) + T.dot(y_tm1, self.r) + self.b) * igate
        c_t = cf + ifi
        ogate = self.gateactivation(c_t*self.po + self.bo + T.dot(x_t, self.wo) + T.dot(y_tm1, self.ro))
        y_t = ogate * self.outpactivation(c_t)
        return [y_t, c_t, y_t]