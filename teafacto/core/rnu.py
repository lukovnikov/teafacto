import inspect
from base import *
from base import tensorops as T


class RecurrentBlock(Block):
    def __init__(self, **kw):
        super(RecurrentBlock, self).__init__(**kw)

    def rec(self, *args):
        raise NotImplementedError("use subclass")

    def get_init_info(self, initstates):
        info, red = self.do_get_init_info(initstates)
        assert((issequence(red) and len(red) == 0) or (not issequence(red)))
        return info

    def get_states_from_outputs(self, outputs):     # topmost layer --> first in list of states (reverse)
        raise NotImplementedError("use subclass")


class RNUBase(RecurrentBlock):
    paramnames = []

    def __init__(self, dim=20, innerdim=20, wreg=0.0001, initmult=0.1, nobias=False, **kw): # dim is input dimensions, innerdim = dimension of internal elements
        super(RNUBase, self).__init__(**kw)
        self.dim = dim
        self.innerdim = innerdim
        self.wreg = wreg
        self.initmult = initmult
        self.nobias = nobias
        self.initparams()

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
            params[paramname] = param(shape, name=paramname).uniform()
            setattr(self, paramname, params[paramname])

    def apply(self, x, initstates=None):
        if initstates is None:
            infoarg = x.shape[0]    # batsize
        else:
            infoarg = initstates
            assert(issequence(infoarg))
        inputs = x.dimswap(1, 0) # inputs is (seq_len, nb_samples, dim)
        outputs, _ = T.scan(fn=self.rec,
                            sequences=inputs,
                            outputs_info=[None]+self.get_init_info(infoarg))
        output = outputs[0]
        return output.dimswap(1, 0)


class RNU(RNUBase):
    paramnames = ["u", "w", "b"]

    def __init__(self, outpactivation=T.tanh, **kw):
        super(RNU, self).__init__(**kw)
        self.outpactivation = outpactivation

    def do_get_init_info(self, initstates):    # either a list of init states or the batsize
        if issequence(initstates):
            return [initstates[0]], initstates[1:]
        else:
            return [T.zeros((initstates, self.innerdim))], initstates

    def get_states_from_outputs(self, outputs):
        assert(len(outputs) == 1)
        return [outputs[0]]

    def rec(self, x_t, h_tm1):      # x_t: (batsize, dim), h_tm1: (batsize, innerdim)
        inp = T.dot(x_t, self.w)    # w: (dim, innerdim) ==> inp: (batsize, innerdim)
        rep = T.dot(h_tm1, self.u)  # u: (innerdim, innerdim) ==> rep: (batsize, innerdim)
        h = inp + rep + self.b               # h: (batsize, innerdim)
        h = self.outpactivation(h)               #
        return [h, h] #T.tanh(inp+rep)


class GatedRNU(RNU):
    def __init__(self, gateactivation=T.nnet.sigmoid, **kw):
        self.gateactivation = gateactivation
        super(GatedRNU, self).__init__(**kw)

    def rec(self, *args):
        raise NotImplementedError("use subclass")


class GRU(GatedRNU):
    paramnames = ["um", "wm", "uhf", "whf", "u", "w", "bm", "bhf", "b"]

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
        self.um = param((self.dim, self.innerdim, self.innerdim)).uniform()
        self.uhf = param((self.dim, self.innerdim, self.innerdim)).uniform()
        self.u = param((self.dim, self.innerdim, self.innerdim)).uniform()

    def rec(self, x_t, h_tm1):
        mgate =  self.gateactivation(T.batched_dot(h_tm1, self.um[x_t, :, :])  + self.wm[x_t, :] + self.bm)
        hfgate = self.gateactivation(T.batched_dot(h_tm1, self.uhf[x_t, :, :]) + self.whf[x_t, :] + self.bhf)
        canh = self.outpactivation(T.batched_dot(h_tm1 * hfgate, self.u[x_t, :, :]) + self.w[x_t, :] + self.b)
        h = mgate * h_tm1 + (1-mgate) * canh
        return [h, h]


class IFGRU(GatedRNU):
    paramnames = ["um", "wm", "uhf", "whf", "uif", "wif", "u", "w", "bm", "bhf", "bif", "b"]

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
    paramnames = ["ucf, uyf, uxf, uof, ucm, uc, rcf, ryf, rxf, rof, rcm, rc, wcf, wyf, wxf, wof, wcm, wc, wo, bcf, byf, bxf, bcm, bof, bc"]

    def do_get_init_info(self, initstates):
        if issequence(initstates):
            c_t0 = initstates[0]
            red = initstates[1:]
            y_t0 = T.zeros((c_t0.shape[0], self.innerdim))
        else:
            c_t0 = T.zeros((initstates, self.innerdim))
            red = initstates
            y_t0 = T.zeros((initstates, self.innerdim))
        return [y_t0, c_t0], red

    def get_states_from_outputs(self, outputs):
        assert(len(outputs) == 2)
        return [outputs[1]]

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


class LSTM(GatedRNU):
    paramnames = ["wf", "rf", "bf", "wi", "ri", "bi", "wo", "ro", "bo", "w", "r", "b", "pf", "pi", "po"]

    def do_get_init_info(self, initstates):
        if issequence(initstates):
            c_t0 = initstates[0]
            red = initstates[1:]
            y_t0 = T.zeros((c_t0.shape[0], self.innerdim))
        else:
            c_t0 = T.zeros((initstates, self.innerdim))
            red = initstates
            y_t0 = T.zeros((initstates, self.innerdim))
        return [y_t0, c_t0], red

    def get_states_from_outputs(self, outputs):
        assert(len(outputs) == 2)
        return [outputs[1]]

    def rec(self, x_t, y_tm1, c_tm1):
        fgate = self.gateactivation(c_tm1*self.pf + self.bf + T.dot(x_t, self.wf) + T.dot(y_tm1, self.rf))
        igate = self.gateactivation(c_tm1*self.pi + self.bi + T.dot(x_t, self.wi) + T.dot(y_tm1, self.ri))
        cf = c_tm1 * fgate
        ifi = self.outpactivation(T.dot(x_t, self.w) + T.dot(y_tm1, self.r) + self.b) * igate
        c_t = cf + ifi
        ogate = self.gateactivation(c_t*self.po + self.bo + T.dot(x_t, self.wo) + T.dot(y_tm1, self.ro))
        y_t = ogate * self.outpactivation(c_t)
        return [y_t, y_t, c_t]
