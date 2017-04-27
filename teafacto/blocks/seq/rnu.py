from teafacto.core.base import Block, param, Val
from teafacto.core.base import tensorops as T
from teafacto.util import getnumargs
from teafacto.util import issequence
from teafacto.blocks.basic import Dropout
from teafacto.blocks.activations import Tanh, Sigmoid
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

    def get_statespec(self, flat=False):
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
        assert(x.ndim == 3)
        assert(mask is None or mask.ndim == 2)
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
        if not issequence(outputs):
            outputs = [outputs]
        outputs = [x.dimswap(1, 0) for x in outputs]
        return outputs[0][:, -1, :], outputs[0], outputs[1:]

    def recwmask(self, x_t, m_t, *states):   # m_t: (batsize, ), x_t: (batsize, dim), states: (batsize, **somedim**)
        recout = self.rec(x_t, *states)
        y_t = recout[0]
        newstates = recout[1:]
        if len(states) > 0:
            states_out = [(a.T * m_t + b.T * (1 - m_t)).T for a, b in
                          zip(newstates, states)]
        else:
            states_out = []
        ''' TODO: WRONG!!! assumes previous output is first state; this is only for GRU
        y_tm1 = states[0] if len(states) > 0 else None
        if y_tm1 is not None:
            y_t_out_a = y_t.T * m_t
            y_t_out_b = y_tm1.T * (1 - m_t)
            y_t_out = (y_t_out_a + y_t_out_b).T
            states_out = [(a.T * m_t + b.T * (1 - m_t)).T for a, b in zip(newstates, states)]   # TODO: try replace with switch expression
        else:
            y_t_out = y_t
            states_out = []     '''
        return [y_t] + states_out


class ReccableWrapper(ReccableBlock):
    """ wraps a non-recurrent block to be reccable """
    def __init__(self, block, **kw):
        super(ReccableWrapper, self).__init__(**kw)
        self.block = block

    @property
    def numstates(self):
        return 0

    def get_statespec(self, flat=False):
        return tuple()

    def rec(self, x_t):
        return [self.block(x_t)]

    def get_init_info(self, arg):
        return []


class RNUBase(ReccableBlock):

    def __init__(self, dim=20, innerdim=20, wreg=0.0, noinput=False,
                 initmult=0.1, nobias=False, paraminit="glorotuniform", biasinit="uniform",
                 dropout_in=False, dropout_h=False, zoneout=False, **kw): #layernormalize=False): # dim is input dimensions, innerdim = dimension of internal elements
        super(RNUBase, self).__init__(**kw)
        self.indim = dim
        self.innerdim = innerdim
        self.wreg = wreg
        self.initmult = initmult
        self.nobias = nobias
        self.noinput = noinput
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
        self.zoneout = Dropout(zoneout)
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

    def __init__(self, outpactivation=T.tanh, param_init_states=False, **kw):
        self.outpactivation = outpactivation
        super(RNU, self).__init__(**kw)
        self.initstateparams = None
        if param_init_states:
            self.initstateparams = []
            for spec in self.get_statespec():
                if spec[0] == "state":
                    initstateparam = param(spec[1], name="init_state").uniform()
                    self.initstateparams.append(initstateparam)
                else:
                    self.initstateparams.append(None)
        self.param_init_states = param_init_states
        self.makeparams()

    def makeparams(self):
        if not self.noinput:
            self.w = param((self.indim, self.innerdim), name="w").init(self.paraminit)
        else:
            self.w = 0
        self.u = param((self.innerdim, self.innerdim), name="u").init(self.paraminit)
        if self.nobias is False:
            self.b = param((self.innerdim,), name="b").init(self.biasinit)
        else:
            self.b = 0

    def get_statespec(self, flat=False):
        return (("state", (self.innerdim,)),)

    def get_init_info(self, initstates):    # either a list of init states or the batsize
        if not issequence(initstates):
            initstates = [initstates] * self.numstates
        acc = []
        if self.initstateparams is None:
            initstateparams = [None] * self.numstates
        else:
            initstateparams = self.initstateparams
        for initstate, initstateparam in zip(initstates, initstateparams):
            if isinstance(initstate, int) or initstate.ndim == 0:
                #embed()
                if initstateparam is not None:
                    toapp = T.repeat(initstateparam.dimadd(0), initstate, axis=0)
                    acc.append(toapp)
                else:
                    acc.append(T.zeros((initstate, self.innerdim)))
            else:
                acc.append(initstate)
        return acc          # left is bottom

    def rec(self, x_t, h_tm1):      # x_t: (batsize, dim), h_tm1: (batsize, innerdim)
        x_t = self.dropout_in(x_t) if not self.noinput else 0
        inp = T.dot(x_t, self.w)    # w: (dim, innerdim) ==> inp: (batsize, innerdim)
        h_tm1 = self.dropout_h(h_tm1)
        rep = T.dot(h_tm1, self.u)  # u: (innerdim, innerdim) ==> rep: (batsize, innerdim)
        h = inp + rep + self.b               # h: (batsize, innerdim)
        '''h = self.normalize_layer(h)'''
        h = self.outpactivation(h)               #
        return [h, h] #T.tanh(inp+rep)


class GatedRNU(RNU):
    def __init__(self, gateactivation=T.nnet.sigmoid,
                 outpactivation=T.tanh,
                 param_init_states=False,
                 init_carry_bias=False, **kw):
        self.gateactivation = gateactivation
        self._init_carry_bias = init_carry_bias
        super(GatedRNU, self).__init__(outpactivation=outpactivation,
                                       param_init_states=param_init_states,
                                       **kw)

    def rec(self, *args):
        raise NotImplementedError("use subclass")


class Gate(Block):
    def __init__(self, indims, outdim, activation=Sigmoid(), nobias=False,
                 paraminit="glorotuniform", biasinit="uniform", **kw):
        super(Gate, self).__init__(**kw)
        self.outdim = outdim
        self.biasinit = biasinit
        self.activation = activation
        indim = sum(indims)
        self.W = param((indim, outdim), name="gate_W").init(paraminit)
        if not nobias:
            self.b = param((outdim,), name="gate_b").init(biasinit) + 0
        else:
            self.b = T.zeros((outdim,))

    def apply(self, *inps):
        inp = T.concatenate(list(inps), axis=-1)
        val = T.dot(inp, self.W) + self.b.dimadd(0)
        ret = self.activation(val)
        return ret


class GRU(GatedRNU):

    def makeparams(self):
        if not self.noinput:
            self.w = param((self.indim, self.innerdim), name="w").init(self.paraminit)
            self.wm = param((self.indim, self.innerdim), name="wm").init(self.paraminit)
            self.whf = param((self.indim, self.innerdim), name="whf").init(self.paraminit)
        else:
            self.w , self.wm, self.whf = 0, 0, 0
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
        x_t = self.dropout_in(x_t) if not self.noinput else T.zeros_like(x_t)
        h_tm1_i = self.dropout_h(h_tm1)
        mgate =  self.gateactivation(T.dot(h_tm1_i, self.um)  + T.dot(x_t, self.wm)  + self.bm)
        hfgate = self.gateactivation(T.dot(h_tm1_i, self.uhf) + T.dot(x_t, self.whf) + self.bhf)
        canh = T.dot(h_tm1_i * hfgate, self.u) + T.dot(x_t, self.w) + self.b
        '''canh = self.normalize_layer(canh)'''
        canh = self.outpactivation(canh)
        mgate = self.zoneout(mgate)
        h = (1 - mgate) * h_tm1 + mgate * canh
        #h = self.normalize_layer(h)
        return [h, h]


class mGRU(GatedRNU):       # multiplicative GRU: https://arxiv.org/pdf/1609.07959.pdf
    def makeparams(self):
        if not self.noinput:
            self.w = param((self.indim, self.innerdim), name="w").init(self.paraminit)
            self.w_xm = param((self.indim, self.innerdim), name="w_xm").init(self.paraminit)
            self.w_xhf = param((self.indim, self.innerdim), name="w_xhf").init(self.paraminit)
            self.w_xmg = param((self.indim, self.innerdim), name="w_xmg").init(self.paraminit)
        else:
            self.w , self.w_xm, self.w_xhf, self.w_xmg = 0, 0, 0, 0
        self.u = param((self.innerdim, self.innerdim), name="u").init(self.paraminit)
        self.w_hm = param((self.innerdim, self.innerdim), name="w_hm").init(self.paraminit)
        self.w_hmg = param((self.innerdim, self.innerdim), name="w_hmg").init(self.paraminit)
        self.w_hhf = param((self.innerdim, self.innerdim), name="w_hhf").init(self.paraminit)
        if not self.nobias:
            self.b = param((self.innerdim,), name="b").init(self.biasinit)
            if self._init_carry_bias > 0:
                amnt = default_init_carry_gate_bias\
                    if self._init_carry_bias is True else self._init_carry_bias
                self.b_mg = param((self.innerdim,), name="b_mg").constant(amnt)
            else:
                self.b_mg = param((self.innerdim,), name="b_mg").init(self.biasinit)
            self.b_hf = param((self.innerdim,), name="b_hf").init(self.biasinit)
        else:
            self.b, self.b_mg, self.b_hf = 0, 0, 0

    def rec(self, x_t, h_tm1):
        x_t = self.dropout_in(x_t) if not self.noinput else 0
        h_tm1_i = self.dropout_h(h_tm1)
        m_t = T.dot(x_t, self.w_xm) * T.dot(h_tm1_i, self.w_hm)
        mgate = self.gateactivation(T.dot(x_t, self.w_xmg) + T.dot(m_t, self.w_hmg) + self.b_mg)
        hfgate = self.gateactivation(T.dot(x_t, self.w_xhf) + T.dot(m_t, self.w_hhf) + self.b_hf)
        canh = T.dot(m_t * hfgate, self.u) + T.dot(x_t, self.w) + self.b
        canh = self.outpactivation(canh)
        mgate = self.zoneout(mgate)
        h = (1 - mgate) * h_tm1 + mgate * canh
        return [h, h]


class MIGRU(GatedRNU):      # multiplicative integration GRU: https://arxiv.org/pdf/1606.06630.pdf

    def makeparams(self):
        if not self.noinput:
            self.w = param((self.indim, self.innerdim), name="w").init(self.paraminit)
            self.w_m = param((self.indim, self.innerdim), name="w_m").init(self.paraminit)
            self.w_h = param((self.indim, self.innerdim), name="w_h").init(self.paraminit)
        else:
            self.w , self.w_m, self.w_h = 0, 0, 0
        self.u = param((self.innerdim, self.innerdim), name="u").init(self.paraminit)
        self.u_m = param((self.innerdim, self.innerdim), name="u_m").init(self.paraminit)
        self.u_h = param((self.innerdim, self.innerdim), name="u_h").init(self.paraminit)
        if not self.nobias:
            self.b_1 = param((self.innerdim,), name="b_1").init(self.biasinit)
            self.b_2 = param((self.innerdim,), name="b_2").init(self.biasinit)
            self.b_3 = param((self.innerdim,), name="b_3").init(self.biasinit)
            self.b_4 = param((self.innerdim,), name="b_4").init(self.biasinit)
            self.b_m1 = param((self.innerdim,), name="b_m1").init(self.biasinit)
            self.b_m2 = param((self.innerdim,), name="b_m2").init(self.biasinit)
            self.b_m3 = param((self.innerdim,), name="b_m3").init(self.biasinit)
            if self._init_carry_bias > 0:
                amnt = default_init_carry_gate_bias\
                    if self._init_carry_bias is True else self._init_carry_bias
                self.b_m4 = param((self.innerdim,), name="b_m4").constant(amnt)
            else:
                self.b_m4 = param((self.innerdim,), name="b_m4").init(self.biasinit)
            self.b_h1 = param((self.innerdim,), name="b_h1").init(self.biasinit)
            self.b_h2 = param((self.innerdim,), name="b_h2").init(self.biasinit)
            self.b_h3 = param((self.innerdim,), name="b_h3").init(self.biasinit)
            self.b_h4 = param((self.innerdim,), name="b_h4").init(self.biasinit)
        else:
            self.b_1, self.b_2, self.b_3, self.b_4, \
            self.b_m1, self.b_m2, self.b_m3, self.b_m4, \
            self.b_h1, self.b_h2, self.b_h3, self.b_h4 = \
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    def rec(self, x_t, h_tm1):
        x_t = self.dropout_in(x_t) if not self.noinput else 0
        h_tm1_i = self.dropout_h(h_tm1)
        mgate = self.gateactivation(self.b_m1 * T.dot(x_t, self.w_m) * T.dot(h_tm1_i, self.u_m) +
                                    self.b_m2 * T.dot(x_t, self.w_m) +
                                    self.b_m3 * T.dot(h_tm1_i, self.u_m) +
                                    self.b_m4)
        hfgate = self.gateactivation(self.b_h1 * T.dot(x_t, self.w_h) * T.dot(h_tm1_i, self.u_h) +
                                    self.b_h2 * T.dot(x_t, self.w_h) +
                                    self.b_h3 * T.dot(h_tm1_i, self.u_h) +
                                    self.b_h4)
        canh = self.b_1 * T.dot(x_t, self.w) * T.dot(h_tm1_i * hfgate, self.u) + \
               self.b_2 * T.dot(x_t, self.w) + \
               self.b_3 * T.dot(h_tm1_i * hfgate, self.u) + \
               self.b_4
        canh = self.outpactivation(canh)
        mgate = self.zoneout(mgate)
        h = (1 - mgate) * h_tm1 + mgate * canh
        return [h, h]


class MuFuRU(GatedRNU):     # https://arxiv.org/pdf/1606.03002.pdf
    def makeparams(self):
        if not self.noinput:
            self.w_v = param((self.indim, self.innerdim), name="w_v").init(self.paraminit)
            self.w_r = param((self.indim, self.innerdim), name="w_r").init(self.paraminit)
            self.w_u = param((self.indim, self.innerdim, 7), name="w_u").init(self.paraminit)
        else:
            self.w_v , self.w_r, self.w_u = 0, 0, 0
        self.u_v = param((self.innerdim, self.innerdim), name="u_v").init(self.paraminit)
        self.u_r = param((self.innerdim, self.innerdim), name="u_r").init(self.paraminit)
        self.u_u = param((self.innerdim, self.innerdim, 7), name="u_u").init(self.paraminit)
        if not self.nobias:
            self.b_v = param((self.innerdim,), name="b_v").init(self.biasinit)
            self.b_r = param((self.innerdim,), name="b_r").init(self.biasinit)
            self.b_u = param((self.innerdim, 7), name="b_u").init(self.biasinit)
        else:
            self.b_v, self.b_r, self.b_u = 0, 0, 0

    def rec(self, x_t, h_tm1):
        x_t = self.dropout_in(x_t) if not self.noinput else T.zeros_like(x_t)
        h_tm1_i = self.dropout_h(h_tm1)
        r_t = self.gateactivation(T.dot(h_tm1_i, self.u_r) + T.dot(x_t, self.w_r) + self.b_r)
        v_t = self.outpactivation(T.dot(h_tm1_i * r_t, self.u_v) + T.dot(x_t, self.w_v) + self.b_v)

        u_t = T.tensordot(h_tm1_i, self.u_u, axes=([1], [0])) \
              + T.tensordot(x_t, self.w_u, axes=([1], [0]))\
              + self.b_u
        u_t = T.softmax(u_t)        # (batsize, dim, numops)

        # ops
        keep_t = h_tm1_i
        repl_t = v_t
        temp_t = T.concatenate([v_t.dimadd(0), h_tm1_i.dimadd(0)], axis=0)
        max_t = T.max(temp_t, axis=0)
        min_t = T.min(temp_t, axis=0)
        mul_t = v_t * h_tm1_i
        diff_t = 0.5 * abs(v_t - h_tm1_i)
        forg_t = T.zeros_like(v_t)

        h = keep_t * u_t[:, :, 0] \
            + repl_t * u_t[:, :, 1] \
            + max_t * u_t[:, :, 2] \
            + min_t * u_t[:, :, 3] \
            + mul_t * u_t[:, :, 4] \
            + diff_t * u_t[:, :, 5] \
            + forg_t * u_t[:, :, 6]
        zoneout = self.zoneout(T.ones_like(h))
        h = h * zoneout + (1 - zoneout) * h_tm1_i
        return [h, h]


class QRNU(GatedRNU):       # QRNN: https://arxiv.org/pdf/1611.01576.pdf

    def __init__(self, window_size=3, gateactivation=T.nnet.sigmoid,
                 outpactivation=T.tanh,
                 param_init_states=False,
                 init_carry_bias=False,
                 nobias=True, **kw):
        self.window_size = window_size
        super(QRNU, self).__init__(gateactivation=gateactivation,
                                   outpactivation=outpactivation,
                                   param_init_states=param_init_states,
                                   init_carry_bias=init_carry_bias,
                                   nobias=nobias,
                                   **kw)

    def makeparams(self):
        self.w_z = param((self.indim * self.window_size, self.innerdim), name="w_z").init(self.paraminit)
        self.w_f = param((self.indim * self.window_size, self.innerdim), name="w_f").init(self.paraminit)
        self.w_o = param((self.indim * self.window_size, self.innerdim), name="w_o").init(self.paraminit)
        if not self.nobias:
            self.b_z = param((self.innerdim,), name="b_z").init(self.biasinit)
            if self._init_carry_bias > 0:
                amnt = default_init_carry_gate_bias\
                    if self._init_carry_bias is True else self._init_carry_bias
                self.b_f = param((self.innerdim,), name="b_f").constant(amnt)
            else:
                self.b_f = param((self.innerdim,), name="b_f").init(self.biasinit)
            self.b_o = param((self.innerdim,), name="b_o").init(self.biasinit)
        else:
            self.b_z, self.b_f, self.b_o = 0, 0, 0

    def get_statespec(self, flat=False):
        return (("state", (self.innerdim,)),)   # TODO

    def get_init_info(self, initstates):  # either a list of init states or the batsize
        sinit = super(QRNU, self).get_init_info(initstates)
        assert(len(sinit) == 1)
        add = T.zeros((sinit[0].shape[0], self.indim * self.window_size))
        ret = T.concatenate([sinit[0], add], axis=1)
        return [ret]

    def rec(self, x_t, h_tm1):  # h_tm1: (batsize, innerdim + windowsize * indim)
        x_t = self.dropout_in(x_t)
        x_tms = h_tm1[:, self.innerdim+self.indim:]
        h_tm1 = h_tm1[:, :self.innerdim]
        h_tm1_i = self.dropout_h(h_tm1)
        # prepare previous x's
        x_tms = T.concatenate([x_tms, x_t], axis=1) # (batsize, indim * windowsize)
        z_t = self.outpactivation(T.dot(x_tms, self.w_z) + self.b_z)
        f_t = self.gateactivation(T.dot(x_tms, self.w_f) + self.b_f)
        o_t = self.gateactivation(T.dot(x_tms, self.w_o) + self.b_o)
        f_t = self.zoneout(f_t)
        h_t = (1 - f_t) * h_tm1_i + f_t * z_t
        y_t = o_t * h_t
        h_ret = T.concatenate([h_t, x_tms], axis=1)
        return [y_t, h_ret]


class RHN(GatedRNU):
    pass    # TODO implement
            # TODO maybe move one abstraction layer higher


class IFGRU(GRU):      # input-modulating GRU

    def makeparams(self):
        super(IFGRU, self).makeparams()
        self.uif = param((self.innerdim, self.indim), name="uif").init(self.paraminit)
        if not self.noinput:
            self.wif = param((self.indim, self.indim), name="wif").init(self.paraminit)
        else:
            self.wif = 0
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
        x_t = self.dropout_in(x_t) if not self.noinput else 0
        h_tm1 = self.dropout_h(h_tm1)
        mgate =  self.gateactivation(T.dot(h_tm1, self.um)  + T.dot(x_t, self.wm)  + self.bm)
        hfgate = self.gateactivation(T.dot(h_tm1, self.uhf) + T.dot(x_t, self.whf) + self.bhf)
        ifgate = self.gateactivation(T.dot(h_tm1, self.uif) + T.dot(x_t, self.wif) + self.bif)
        canh = self.outpactivation(T.dot(h_tm1 * hfgate, self.u) + T.dot(x_t * ifgate, self.w) + self.b)
        h = mgate * h_tm1 + (1-mgate) * canh
        return [h, h]


class LSTM(GatedRNU):
    def makeparams(self):
        if not self.noinput:
            self.w = param((self.indim, self.innerdim), name="w").init(self.paraminit)
            self.wf = param((self.indim, self.innerdim), name="wf").init(self.paraminit)
            self.wi = param((self.indim, self.innerdim), name="wi").init(self.paraminit)
            self.wo = param((self.indim, self.innerdim), name="wo").init(self.paraminit)
        else:
            self.w, self.wf, self.wi, self.wo = 0, 0, 0, 0
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

    def rec(self, x_t, c_tm1, y_tm1):
        x_t = self.dropout_in(x_t) if not self.noinput else 0
        c_tm1 = self.dropout_h(c_tm1)
        fgate = self.gateactivation(c_tm1*self.pf + self.bf + T.dot(x_t, self.wf) + T.dot(y_tm1, self.rf))
        igate = self.gateactivation(c_tm1*self.pi + self.bi + T.dot(x_t, self.wi) + T.dot(y_tm1, self.ri))
        cf = c_tm1 * fgate
        ifi = self.outpactivation(T.dot(x_t, self.w) + T.dot(y_tm1, self.r) + self.b) * igate
        c_t = cf + ifi
        ogate = self.gateactivation(c_t*self.po + self.bo + T.dot(x_t, self.wo) + T.dot(y_tm1, self.ro))
        y_t = ogate * self.outpactivation(c_t)
        return [y_t, c_t, y_t]

    def get_statespec(self, flat=False):
        return (("state", (self.innerdim,)), ("output", (self.innerdim,)))

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


