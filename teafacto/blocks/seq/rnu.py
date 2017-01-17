from teafacto.core.base import Block, param
from teafacto.core.base import tensorops as T
from teafacto.util import getnumargs
from teafacto.util import issequence
from teafacto.blocks.basic import Dropout
from IPython import embed

default_init_carry_gate_bias = 1

class RecurrentBlock(Block):     # ancestor class for everything that consumes sequences f32~(batsize, seqlen, ...)
    def __init__(self, reverse=False, **kw):
        super(RecurrentBlock, self).__init__(**kw)
        self._reverse = reverse

    @property
    def reverse(self):
        self._reverse = True
        return self

    @property
    def numstates(self):
        raise NotImplementedError("use subclass")

    def apply_argspec(self):
        return ((3, "float"),)

    # FWD API
    def apply(self, x, mask=None, initstates=None):
        final, output, states = self.innerapply(x, mask, initstates)
        return output  # return is (batsize, seqlen, dim)

    def innerapply(self, x, mask=None, initstates=None):
        raise NotImplementedError("use subclass")


class ReccableBlock(RecurrentBlock):    # exposes a rec function
    def __init__(self, **kw):
        super(ReccableBlock, self).__init__(**kw)

    @property
    def numstates(self):
        return getnumargs(self.rec) - 2

    # REC API
    def rec(self, *args):
        raise NotImplementedError("use subclass")

    def get_init_info(self, initstates):
        raise NotImplementedError("use subclass")

    def get_inits(self, initstates):
        return self.get_init_info(initstates)

    # FWD API IMPLEMENTATION USING REC API
    def innerapply(self, x, mask=None, initstates=None):
        assert(x.ndim == 3 and (mask is None or mask.ndim == 2))
        if initstates is None:
            infoarg = x.shape[0]    # batsize
        else:
            infoarg = initstates
            assert(issequence(infoarg))
        inputs = x.dimswap(1, 0) # inputs is (seq_len, batsize, dim)
        init_info = self.get_init_info(infoarg)
        if mask is None:
            outputs = T.scan(fn=self.rec,
                                sequences=inputs,
                                outputs_info=[None]+init_info,
                                go_backwards=self._reverse)
        else:
            outputs = T.scan(fn=self.recwmask,
                                sequences=[inputs, mask.dimswap(1, 0)],
                                outputs_info=[None] + init_info,
                                go_backwards=self._reverse)
        outputs = [x.dimswap(1, 0) for x in outputs]
        return outputs[0][:, -1, :], outputs[0], outputs[1:]

    def recwmask(self, x_t, m_t, *states):   # m_t: (batsize, ), x_t: (batsize, dim), states: (batsize, **somedim**)
        recout = self.rec(x_t, *states)
        y_t = recout[0]
        newstates = recout[1:]
        y_tm1 = T.zeros_like(y_t)
        y_tm1 = states[0]               # TODO: beware with multiple layers (here will be the bottom first)
        y_t_out = (y_t.T * m_t + y_tm1.T * (1 - m_t)).T
        states_out = [(a.T * m_t + b.T * (1 - m_t)).T for a, b in zip(newstates, states)]   # TODO: try replace with switch expression
        return [y_t_out] + states_out


class RNUBase(ReccableBlock):

    def __init__(self, dim=20, innerdim=20, wreg=0.0001,
                 initmult=0.1, nobias=False, paraminit="glorotuniform", biasinit="uniform",
                 dropout_in=False, dropout_h=False, **kw): #layernormalize=False): # dim is input dimensions, innerdim = dimension of internal elements
        super(RNUBase, self).__init__(**kw)
        self.indim = dim
        self.innerdim = innerdim
        self.wreg = wreg
        self.initmult = initmult
        self.nobias = nobias
        self.paraminit = paraminit
        self.biasinit = biasinit
        '''self.layernormalize = layernormalize
        if self.layernormalize:
            self.layernorm_gain = param((innerdim,), name="layer_norm_gain").uniform()
            self.layernorm_bias = param((innerdim,), name="layer_norm_bias").uniform()
            #self.nobias = True'''
        self.rnuparams = {}
        self.dropout_in = Dropout(dropout_in)
        self.dropout_h = Dropout(dropout_h)
    '''
    def normalize_layer(self, vec):     # (batsize, hdim)
        if self.layernormalize:
            fshape = T.cast(vec.shape[1], "float32")
            mean = (T.sum(vec, axis=1) / fshape).dimshuffle(0, "x")
            sigma = (T.sqrt(T.sum((vec - mean)**2, axis=1) / fshape)).dimshuffle(0, "x")
            ret = (self.layernorm_gain / sigma) * (vec - mean) + self.layernorm_bias
            #ret = T.cast(ret, "float32")
        else:
            ret = vec
        return ret
    '''

    def recappl(self, inps, states):
        numrecargs = getnumargs(self.rec) - 2       # how much to pop from states
        mystates = states[:numrecargs]
        tail = states[numrecargs:]
        inps = [inps] if not issequence(inps) else inps
        outs = self.rec(*(inps + mystates))
        return outs[0], outs[1:], tail


class RNU(RNUBase):

    def __init__(self, outpactivation=T.tanh, **kw):
        self.outpactivation = outpactivation
        super(RNU, self).__init__(**kw)
        self.makeparams()

    def makeparams(self):
        self.w = param((self.indim, self.innerdim), name="w").init(self.paraminit)
        self.u = param((self.innerdim, self.innerdim), name="u").init(self.paraminit)
        if self.nobias is False:
            self.b = param((self.innerdim,), name="b").init(self.biasinit)
        else:
            self.b = 0

    def get_init_info(self, initstates):    # either a list of init states or the batsize
        if not issequence(initstates):
            initstates = [initstates] * self.numstates
        acc = []
        for initstate in initstates:
            if isinstance(initstate, int) or initstate.ndim == 0:
                #embed()
                acc.append(T.zeros((initstate, self.innerdim)))
            else:
                acc.append(initstate)
        return acc

    def rec(self, x_t, h_tm1):      # x_t: (batsize, dim), h_tm1: (batsize, innerdim)
        x_t = self.dropout_in(x_t)
        h_tm1 = self.dropout_h(h_tm1)
        inp = T.dot(x_t, self.w)    # w: (dim, innerdim) ==> inp: (batsize, innerdim)
        rep = T.dot(h_tm1, self.u)  # u: (innerdim, innerdim) ==> rep: (batsize, innerdim)
        h = inp + rep + self.b               # h: (batsize, innerdim)
        '''h = self.normalize_layer(h)'''
        h = self.outpactivation(h)               #
        return [h, h] #T.tanh(inp+rep)


class GatedRNU(RNU):
    def __init__(self, gateactivation=T.nnet.sigmoid, init_carry_bias=False, **kw):
        self.gateactivation = gateactivation
        self._init_carry_bias = init_carry_bias
        super(GatedRNU, self).__init__(**kw)

    def rec(self, *args):
        raise NotImplementedError("use subclass")


class GRU(GatedRNU):

    def makeparams(self):
        self.w = param((self.indim, self.innerdim), name="w").init(self.paraminit)
        self.wm = param((self.indim, self.innerdim), name="wm").init(self.paraminit)
        self.whf = param((self.indim, self.innerdim), name="whf").init(self.paraminit)
        self.u = param((self.innerdim, self.innerdim), name="u").init(self.paraminit)
        self.um = param((self.innerdim, self.innerdim), name="um").init(self.paraminit)
        self.uhf = param((self.innerdim, self.innerdim), name="uhf").init(self.paraminit)
        if not self.nobias:
            self.b = param((self.innerdim,), name="b").init(self.biasinit)
            if self._init_carry_bias > 0:
                amnt = default_init_carry_gate_bias\
                    if self._init_carry_bias is True else self._init_carry_bias
                self.bm = param((self.innerdim,), name="bm").constant(amnt)
            else:
                self.bm = param((self.innerdim,), name="bm").init(self.biasinit)
            self.bhf = param((self.innerdim,), name="bhf").init(self.biasinit)
        else:
            self.b, self.bm, self.bhf = 0, 0, 0

    def rec(self, x_t, h_tm1):
        '''
        :param x_t: input values (nb_samples, nb_feats) for this recurrence step
        :param h_tm1: previous states (nb_samples, out_dim)
        :return: new state (nb_samples, out_dim)
        '''
        x_t = self.dropout_in(x_t)
        h_tm1_i = self.dropout_h(h_tm1)
        mgate =  self.gateactivation(T.dot(h_tm1_i, self.um)  + T.dot(x_t, self.wm)  + self.bm)
        hfgate = self.gateactivation(T.dot(h_tm1_i, self.uhf) + T.dot(x_t, self.whf) + self.bhf)
        canh = T.dot(h_tm1_i * hfgate, self.u) + T.dot(x_t, self.w) + self.b
        '''canh = self.normalize_layer(canh)'''
        canh = self.outpactivation(canh)
        h = mgate * h_tm1 + (1-mgate) * canh
        #h = self.normalize_layer(h)
        return [h, h]


class RHN(GatedRNU):
    pass    # TODO implement
            # TODO maybe move one abstraction layer higher


class IFGRU(GRU):      # input-modulating GRU

    def makeparams(self):
        super(IFGRU, self).makeparams()
        self.uif = param((self.innerdim, self.indim), name="uif").init(self.paraminit)
        self.wif = param((self.indim, self.indim), name="wif").init(self.paraminit)
        if not self.nobias:
            self.bif = param((self.indim,), name="bif").init(self.biasinit)
        else:
            self.bif = 0

    def rec(self, x_t, h_tm1):
        '''
        :param x_t: input values (nb_samples, nb_feats) for this recurrence step
        :param h_tm1: previous states (nb_samples, out_dim)
        :return: new state (nb_samples, out_dim)
        '''
        x_t = self.dropout_in(x_t)
        h_tm1 = self.dropout_h(h_tm1)
        mgate =  self.gateactivation(T.dot(h_tm1, self.um)  + T.dot(x_t, self.wm)  + self.bm)
        hfgate = self.gateactivation(T.dot(h_tm1, self.uhf) + T.dot(x_t, self.whf) + self.bhf)
        ifgate = self.gateactivation(T.dot(h_tm1, self.uif) + T.dot(x_t, self.wif) + self.bif)
        canh = self.outpactivation(T.dot(h_tm1 * hfgate, self.u) + T.dot(x_t * ifgate, self.w) + self.b)
        h = mgate * h_tm1 + (1-mgate) * canh
        return [h, h]


class LSTM(GatedRNU):
    def makeparams(self):
        self.w = param((self.indim, self.innerdim), name="w").init(self.paraminit)
        self.wf = param((self.indim, self.innerdim), name="wf").init(self.paraminit)
        self.wi = param((self.indim, self.innerdim), name="wi").init(self.paraminit)
        self.wo = param((self.indim, self.innerdim), name="wo").init(self.paraminit)
        self.r = param((self.innerdim, self.innerdim), name="r").init(self.paraminit)
        self.rf = param((self.innerdim, self.innerdim), name="rf").init(self.paraminit)
        self.ri = param((self.innerdim, self.innerdim), name="ri").init(self.paraminit)
        self.ro = param((self.innerdim, self.innerdim), name="ro").init(self.paraminit)
        if not self.nobias:
            self.b = param((self.innerdim,), name="b").init(self.biasinit)
            if self._init_carry_bias > 0:
                amnt = default_init_carry_gate_bias\
                    if self._init_carry_bias is True else self._init_carry_bias
                self.bf = param((self.innerdim,), name="bf").constant(amnt)
                self.bi = param((self.innerdim,), name="bi").constant(-amnt)
            else:
                self.bf = param((self.innerdim,), name="bf").init(self.biasinit)
                self.bi = param((self.innerdim,), name="bi").init(self.biasinit)
            self.bo = param((self.innerdim,), name="bo").init(self.biasinit)
        else:
            self.b, self.bf, self.bi, self.bo = 0, 0, 0, 0
        self.p = param((self.innerdim,), name="p").init(self.biasinit)
        self.pf = param((self.innerdim,), name="pf").init(self.biasinit)
        self.pi = param((self.innerdim,), name="pi").init(self.biasinit)
        self.po = param((self.innerdim,), name="po").init(self.biasinit)

    def rec(self, x_t, y_tm1, c_tm1):
        x_t = self.dropout_in(x_t)
        c_tm1 = self.dropout_h(c_tm1)
        fgate = self.gateactivation(c_tm1*self.pf + self.bf + T.dot(x_t, self.wf) + T.dot(y_tm1, self.rf))
        igate = self.gateactivation(c_tm1*self.pi + self.bi + T.dot(x_t, self.wi) + T.dot(y_tm1, self.ri))
        cf = c_tm1 * fgate
        ifi = self.outpactivation(T.dot(x_t, self.w) + T.dot(y_tm1, self.r) + self.b) * igate
        c_t = cf + ifi
        ogate = self.gateactivation(c_t*self.po + self.bo + T.dot(x_t, self.wo) + T.dot(y_tm1, self.ro))
        y_t = ogate * self.outpactivation(c_t)
        return [y_t, y_t, c_t]

'''
class XRU(RNU):
    pass


class RXRU(XRU):
    paramnames = "ui wi bi uh bh uo bo wt".split()

    def rec(self, x_t, h_tm1):
        x_t_i = self.dropout_in(x_t)
        h_tm1_i = self.dropout_h(h_tm1)
        l_t = self.outpactivation(T.dot(x_t_i, self.wi) + T.dot(h_tm1_i, self.ui) + self.bi)
        o_t_i = self.outpactivation(T.dot(l_t, self.uo) + self.bo)
        h_t_i = self.outpactivation(T.dot(l_t, self.uh) + self.bh)
        x_t_o = T.dot(x_t, self.wt)
        y_t = x_t_o + o_t_i
        h_t = h_t_i + h_tm1_i
        return [y_t, h_t]


class GXRU(XRU, GatedRNU):
    paramnames = "um wm bm uhf whf bhf w u b uo wo bo wio".split()

    def rec(self, x_t, h_tm1):
        x_t_i = self.dropout_in(x_t)
        x_t_o = T.dot(x_t, self.wio)
        h_tm1_i = self.dropout_h(h_tm1)
        mgate = self.gateactivation(T.dot(h_tm1_i, self.um) + T.dot(x_t_i, self.wm) + self.bm)
        ogate = self.gateactivation(T.dot(h_tm1_i, self.uo) + T.dot(x_t_i, self.wo) + self.bo)
        hfgate = self.gateactivation(T.dot(h_tm1_i, self.uhf) + T.dot(x_t_i, self.whf) + self.bhf)
        canh = T.dot(h_tm1_i * hfgate, self.u) + T.dot(x_t, self.w) + self.b
        canh = self.outpactivation(canh)
        h_t = mgate * h_tm1 + (1 - mgate) * canh
        y_t = ogate * x_t_o + (1 - ogate) * canh
        return [y_t, h_t]


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
'''


