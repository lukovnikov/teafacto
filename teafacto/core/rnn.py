import theano, numpy as np
from theano import tensor as T
import inspect
from teafacto.core.trainutil import Parameterized
from teafacto.core.init import random
from IPython import embed

class RNUBase(Parameterized):
    def __init__(self, dim=20, innerdim=20, wreg=0.0001, initmult=0.1, nobias=False, **kw): # dim is input dimensions, innerdim = dimension of internal elements
        super(RNUBase, self).__init__(**kw)
        self.dim = dim
        self.innerdim = innerdim
        self.wreg = wreg
        self.initmult = initmult
        self.nobias = nobias
        self.initparams()
        self.initstates = None

    def initparams(self):
        params = {}
        indim = self.innerdim
        for paramname in self.paramnames:
            if paramname[0] == "b" and self.nobias is True:
                setattr(self, paramname, 0)
                continue
            if paramname[0] == "b" or paramname[0] == "p": # bias or peepholes, internal weights
                shape = (self.innerdim,)
            elif paramname[0] == "w": #input processing matrices
                shape = (self.dim, self.innerdim)
            else: # internal recurrent matrices
                shape = (self.innerdim, self.innerdim)
            paramval = self._inittensor(shape)
            params[paramname] = theano.shared(paramval, name=paramname)
            setattr(self, paramname, params[paramname])
        #self.initstate = T.zeros((self.indim,), dtype="float32")

    def _inittensor(self, dims):
        return random(dims, offset=0.5, initmult=self.initmult)

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
        return output.dimshuffle(1, 0, 2)[:, -1, :] #output is last state (nb_samples, nb_feats)

    def rec(self, *args):
        raise NotImplementedError("use subclass")

    def __call__(self, x):
        return self.getoutput(x)

    def set_init_states(self, values):
        self.initstates = values

    def get_init_info(self, batsize):
        raise NotImplementedError("use subclass")

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
    def ownparameters(self):
        params = [param for param in self.paramnames if self.nobias is False or not param[0] == "b"]
        return set(map(lambda x: getattr(self, x), params))


class RNU(RNUBase):
    def __init__(self, **kw):
        self.paramnames = ["u", "w"]
        super(RNU, self).__init__(**kw)

    def get_init_info(self, batsize):
        h_t0 = self.initstates
        if h_t0 is None:
            h_t0 = T.zeros((batsize, self.innerdim))
        return [h_t0]

    def rec(self, x_t, h_tm1):      # x_t: (batsize, dim), h_tm1: (batsize, innerdim)
        inp = T.dot(x_t, self.w)    # w: (dim, innerdim) ==> inp: (batsize, innerdim)
        rep = T.dot(h_tm1, self.u)  # u: (innerdim, innerdim) ==> rep: (batsize, innerdim)
        h = inp + rep               # h: (batsize, innerdim)
        h = T.tanh(h)               #
        return [h, h] #T.tanh(inp+rep)


class GatedRNU(RNUBase):
    def __init__(self, gateactivation=T.nnet.sigmoid, outpactivation=T.tanh, **kw):
        self.gateactivation = gateactivation
        self.outpactivation = outpactivation
        super(GatedRNU, self).__init__(**kw)


class GRU(GatedRNU):
    def __init__(self, **kw):
        self.paramnames = ["um", "wm", "uhf", "whf", "u", "w", "bm", "bhf", "b"]
        super(GRU, self).__init__(**kw)

    def rec(self, x_t, h_tm1):
        '''
        :param x_t: input values (nb_samples, nb_feats) for this recurrence step
        :param h_tm1: previous states (nb_samples, out_dim)
        :return: new state (nb_samples, out_dim)
        '''
        mgate =  self.gateactivation(T.dot(h_tm1, self.um)  + T.dot(x_t, self.wm)  + self.bm)
        hfgate = self.gateactivation(T.dot(h_tm1, self.uhf) + T.dot(x_t, self.whf) + self.bhf)
        canh = self.outpactivation(T.dot(h_tm1 * hfgate, self.u) + T.dot(x_t, self.w) + self.b)
        h = mgate * h_tm1 + (1-mgate) * canh
        return [h, h]

class IEGRU(GRU): # self-input-embedding GRU
    def rec(self, x_t, h_tm1):
        mgate =  self.gateactivation(T.dot(h_tm1, self.um)  + self.wm[x_t, :] + self.bm)
        hfgate = self.gateactivation(T.dot(h_tm1, self.uhf) + self.whf[x_t, :] + self.bhf)
        canh = self.outpactivation(T.dot(h_tm1 * hfgate, self.u) + self.w[x_t, :] + self.b)
        h = mgate * h_tm1 + (1-mgate) * canh
        return [h, h]

class FullEGRU(IEGRU):
    def __init__(self, **kw):
        super(FullEGRU, self).__init__(**kw)
        self.um = theano.shared(self._inittensor((self.dim, self.innerdim, self.innerdim)))
        self.uhf = theano.shared(self._inittensor((self.dim, self.innerdim, self.innerdim)))
        self.u = theano.shared(self._inittensor((self.dim, self.innerdim, self.innerdim)))

    def rec(self, x_t, h_tm1):
        mgate =  self.gateactivation(T.batched_dot(h_tm1, self.um[x_t, :, :])  + self.wm[x_t, :] + self.bm)
        hfgate = self.gateactivation(T.batched_dot(h_tm1, self.uhf[x_t, :, :]) + self.whf[x_t, :] + self.bhf)
        canh = self.outpactivation(T.batched_dot(h_tm1 * hfgate, self.u[x_t, :, :]) + self.w[x_t, :] + self.b)
        h = mgate * h_tm1 + (1-mgate) * canh
        return [h, h]


class IFGRU(GatedRNU):
    def __init__(self, **kw):
        self.paramnames = ["um", "wm", "uhf", "whf", "uif", "wif", "u", "w", "bm", "bhf", "bif", "b"]
        super(IFGRU, self).__init__(**kw)

    def rec(self, x_t, h_tm1):
        '''
        :param x_t: input values (nb_samples, nb_feats) for this recurrence step
        :param h_tm1: previous states (nb_samples, out_dim)
        :return: new state (nb_samples, out_dim)
        '''
        mgate =  self.gateactivation(T.dot(h_tm1, self.um)  + T.dot(x_t, self.wm)  + self.bm)
        hfgate = self.gateactivation(T.dot(h_tm1, self.uhf) + T.dot(x_t, self.whf) + self.bhf)
        ifgate = self.gateactivation(T.dot(h_tm1, self.uif) + T.dot(x_t, self.wif) + self.bif)
        canh = self.outpactivation(T.dot(h_tm1 * hfgate, self.u) + T.dot(x_t * ifgate, self.w) + self.b)
        h = mgate * h_tm1 + (1-mgate) * canh
        return [h, h]

class IFGRUTM(GatedRNU):
    def __init__(self, **kw):
        self.paramnames = ["ucf, uyf, uxf, uof, ucm, uc, rcf, ryf, rxf, rof, rcm, rc, wcf, wyf, wxf, wof, wcm, wc, wo, bcf, byf, bxf, bcm, bof, bc"]
        super(IFGRUTM, self).__init__(**kw)

    def get_init_info(self, batsize):
        y_t0 = T.zeros((batsize, self.innerdim))
        c_t0 = self.initstates
        if c_t0 is None:
            c_t0 = T.zeros((batsize, self.innerdim))
        return [y_t0, c_t0]

    def rec(self, x_t, y_tm1, c_tm1):
        cfgate = self.gateactivation(T.dot(c_tm1, self.ucf) + T.dot(y_tm1, self.rcf) + T.dot(x_t, self.wcf) + self.bcf)
        yfgate = self.gateactivation(T.dot(c_tm1, self.uyf) + T.dot(y_tm1, self.ryf) + T.dot(x_t, self.wyf) + self.byf)
        xfgate = self.gateactivation(T.dot(c_tm1, self.uxf) + T.dot(y_tm1, self.rxf) + T.dot(x_t, self.wxf) + self.bxf)
        mgate = self.gateactivation(T.dot(c_tm1, self.ucm) + T.dot(y_tm1, self.rcm) + T.dot(x_t, self.wcm) + self.bcm)
        cft = T.dot(c_tm1 * cfgate, self.uc)
        yft = T.dot(y_tm1 * yfgate, self.rc)
        xft = T.dot(x_t * xfgate, self.wc)
        canct = self.outpactivation(cft + yft + xft + self.bc)
        c_t = mgate * c_tm1 + (1-mgate) * canct
        ofgate = self.gateactivation(T.dot(c_t, self.uof) + T.dot(y_tm1, self.rof) + T.dot(x_t, self.wof) + self.bof)
        y_t = self.outpactivation(T.dot(c_t * ofgate, self.wo))
        return [y_t, y_t, c_t]


class LSTM(RNUBase):
    def __init__(self, **kw):
        self.paramnames = ["wf", "rf", "bf", "wi", "ri", "bi", "wo", "ro", "bo", "w", "r", "b", "pf", "pi", "po"]
        super(LSTM, self).__init__(**kw)

    def get_init_info(self, batsize):
        y_t0 = T.zeros((batsize, self.innerdim))
        c_t0 = self.initstates
        if c_t0 is None:
            c_t0 = T.zeros((batsize, self.innerdim))
        return [y_t0, c_t0]

    def rec(self, x_t, y_tm1, c_tm1):
        fgate = self.gateactivation(c_tm1*self.pf + self.bf + T.dot(x_t, self.wf) + T.dot(y_tm1, self.rf))
        igate = self.gateactivation(c_tm1*self.pi + self.bi + T.dot(x_t, self.wi) + T.dot(y_tm1, self.ri))
        cf = c_tm1 * fgate
        ifi = self.outpactivation(T.dot(x_t, self.w) + T.dot(y_tm1, self.r) + self.b) * igate
        c_t = cf + ifi
        ogate = self.gateactivation(c_t*self.po + self.bo + T.dot(x_t, self.wo) + T.dot(y_tm1, self.ro))
        y_t = ogate * self.outpactivation(c_t)
        return [y_t, y_t, c_t]


class RNUParameterized(Parameterized):
    def __init__(self, **kw):
        self.rnu = None
        super(RNUParameterized, self).__init__(**kw)

    def __add__(self, other):
        self.attach(other)

    def attach(self, other):
        if isinstance(other, RNUBase):
            self.rnu = other
            self.afterRNUattach()
        return self

    def afterRNUattach(self):
        raise NotImplementedError("use subclass")

    @property
    def depparameters(self):
        return self.rnu.parameters


class RNNEncoder(RNUParameterized):
    '''
    Encodes a sequence of vectors into a vector, input dims and output dims specified by the RNU unit
    '''

    def encode(self, seq): # seq: (batsize, seqlen, dim)
        inp = seq.dimshuffle(1, 0, 2)
        outputs, _ = theano.scan(fn=self.recwrap,
                                 sequences=inp,
                                 outputs_info=[None]+self.rnu.get_init_info(seq.shape[0]))
        output = outputs[0]
        return output[-1, :, :] #output is (batsize, innerdim)

    def recwrap(self, x_t, *args): # x_t: (batsize, dim)      if input is all zeros, just return previous state
        mask = x_t.norm(2, axis=1) > 0 # (batsize, )
        rnuret = self.rnu.rec(x_t, *args) # list of matrices (batsize, **somedims**)
        ret = map(lambda (a, r): (a.T * (1-mask) + r.T * mask).T, zip([args[0]] + list(args), rnuret))
        return ret


class RNNDecoder(RNUParameterized):
    '''
    Decodes a sequence given initial state
    output: probabilities over symbol space float: (batsize, seqlen, dim) where dim is number of symbols

    TERMINUS SYMBOL = 0
    ! first input is TERMINUS ==> suggest to set TERMINUS(0) embedding to all zeroes (in s2vf)
    '''
    # TODO: test
    def __init__(self, hlimit=50, s2vf=None, v2pf=None, **kw): # limit says at most how many is produced
        super(RNNDecoder, self).__init__(**kw)
        self.limit = hlimit
        # outdims are the same as RNU's input dims
        self.s2vf = s2vf
        self.v2pf = v2pf
        self.O = None
        self.W = None

    @property
    def ownparameters(self):
        ret = set()
        if self.O is not None:
            ret.add(self.O)
        return ret

    def afterRNUattach(self):
        if self.v2pf is None:
            self.O = theano.shared(random((self.rnu.innerdim, self.rnu.dim)))
            self.v2pf = lambda v: T.nnet.softmax(T.dot(v, self.O))
        if self.s2vf is None:
            self.W = T.eye(self.rnu.dim, self.rnu.dim) # one-hot embedding matrix
            self.s2vf = lambda s: self.W[s, :]

    def decode(self, initstates, initprobs=None): # initstate: (batsize, innerdim)
        if initprobs is None:
            initprobs = T.eye(1, self.rnu.dim).repeat(initstates.shape[0], axis=0) # all TERMINUS (batsize, dim)
        self.rnu.set_init_states(initstates)
        outputs, _ = theano.scan(fn=self.recwrap,
                                 outputs_info=[initprobs, 0]+self.rnu.get_init_info(),
                                 n_steps=self.limit)
        return outputs[0] # returns probabilities of symbols --> (batsize, seqlen, vocabsize)

    def recwrap(self, x_t, i, *args): # once output is terminus, always terminus and previous state is returned
        chosen = x_t.argmax(axis=1, keepdims=True) # x_t = probs over symbols:: f32-(batsize, dim) ==> int32-(batsize,)
        mask = T.clip(chosen + T.clip(1-i, 0, 1), 0, 1) # (batsize, ) --> only make mask if not in first iter
        newinp = self.s2vf(chosen) #
        rnuret = self.rnu.rec(newinp, *args) # list of matrices (batsize, **somedims**)
        outprobs = self.v2pf(rnuret[0])
        ret = map(lambda (a, r): (a.T * (1-mask) + r.T * mask).T, zip([x_t] + list(args), [outprobs]+rnuret[1:]))
        i = i + 1
        return [ret[0], i] + ret[1:], {}, theano.scan_module.until( (i > 1) * T.eq(mask.norm(1), 0) )


class RNNMask(RNUParameterized):
    '''
    Puts an entry-wise RNN mask on a sequence of vectors.
    RNU output dims must be the same as input dims
    '''
    def mask(self, seq): # seq: (batsize, seqlen, dim)
        inp = seq.dimshuffle(1, 0, 2)
        # initialize hidden states, depending on the used RNN (numstates)
        numstates = len(inspect.getargspec(self.rnu.rec).args) - 2
        initstate = T.zeros((inp.shape[1], self.rnu.innerdim)) # (nb_samples, dim)
        # run RNU over it
        outputs, _ = theano.scan(fn=self.recwrap,
                                 sequences=inp,
                                 outputs_info=[None]+[initstate]*numstates)
        outputs = T.nnet.sigmoid(outputs[0]).dimshuffle(1, 0, 2) # outputs: [0, 1] of (batsize, seqlen, dim)
        mseq = seq * outputs # apply the mask
        #embed()
        return mseq

    def recwrap(self, x_t, *args): # x_t: (batsize, dim)
        return self.rnu.rec(x_t, *args) # list of matrices (batsize, **somedims**)


class RNNVectorMask(RNUParameterized):
    '''
    Puts a element-wise RNN weights on a sequence of vectors.
    '''
    pass
    # TODO
