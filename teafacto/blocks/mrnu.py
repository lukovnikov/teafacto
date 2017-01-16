from teafacto.blocks.seq.attention import WeightedSumAttCon, AttGen
from teafacto.blocks.match import DotDistance
from teafacto.blocks.seq.rnu import GatedRNU
from teafacto.core.base import param, tensorops as T
from teafacto.util import issequence


class XLTM(GatedRNU):
    def __init__(self, discrete=True, memsize=50, **kw):
        super(XLTM, self).__init__(**kw)
        self.discrete = discrete
        self.memsize = memsize
        self.attgen = AttGen(DotDistance())
        self.attcon = WeightedSumAttCon() if self.discrete else WeightedSumAttCon()

    def makeparams(self):
        self.usf = param((self.innerdim, self.innerdim), name="usf").init(self.paraminit)
        self.umf = param((self.innerdim, self.innerdim), name="umf").init(self.paraminit)
        self.u = param((self.innerdim, self.innerdim), name="u").init(self.paraminit)
        self.uug = param((self.innerdim, self.innerdim), name="uug").init(self.paraminit)
        self.uwf = param((self.innerdim, self.innerdim), name="uwf").init(self.paraminit)
        self.uma = param((self.innerdim, self.innerdim), name="uma").init(self.paraminit)
        self.uma2 = param((self.innerdim, self.innerdim), name="uma2").init(self.paraminit)
        self.uif = param((self.innerdim, self.indim), name="uif").init(self.paraminit)
        self.w = param((self.indim, self.innerdim), name="w").init(self.paraminit)
        self.wsf = param((self.indim, self.innerdim), name="wsf").init(self.paraminit)
        self.wmf = param((self.indim, self.innerdim), name="wmf").init(self.paraminit)
        self.wug = param((self.indim, self.innerdim), name="wug").init(self.paraminit)
        self.wwf = param((self.indim, self.innerdim), name="wwf").init(self.paraminit)
        self.wma = param((self.indim, self.innerdim), name="wma").init(self.paraminit)
        self.wma2 = param((self.indim, self.innerdim), name="wma2").init(self.paraminit)
        self.wif = param((self.indim, self.indim), name="wif").init(self.paraminit)
        self.m = param((self.innerdim, self.innerdim), name="m").init(self.paraminit)
        self.msf = param((self.innerdim, self.innerdim), name="msf").init(self.paraminit)
        self.mmf = param((self.innerdim, self.innerdim), name="mmf").init(self.paraminit)
        self.mug = param((self.innerdim, self.innerdim), name="mug").init(self.paraminit)
        self.mwf = param((self.innerdim, self.innerdim), name="mwf").init(self.paraminit)
        self.mma = param((self.innerdim, self.innerdim), name="mma").init(self.paraminit)
        self.mma2 = param((self.innerdim, self.innerdim), name="mma2").init(self.paraminit)
        self.mif = param((self.innerdim, self.indim), name="mif").init(self.paraminit)
        if not self.nobias:
            self.b = param((self.innerdim,), name="b").init(self.biasinit)
            self.bsf = param((self.innerdim,), name="bsf").init(self.biasinit)
            self.bmf = param((self.innerdim,), name="bmf").init(self.biasinit)
            self.bug = param((self.innerdim,), name="bug").init(self.biasinit)
            self.bwf = param((self.innerdim,), name="bwf").init(self.biasinit)
            self.bma = param((self.innerdim,), name="bma").init(self.biasinit)
            self.bma2 = param((self.innerdim,), name="bma2").init(self.biasinit)
            self.bif = param((self.indim,), name="bif").init(self.biasinit)
        else:
            self.b, self.bsf, self.bmf, self.bug, self.bwf, self.bma, self.bma2, self.bif = \
                0, 0, 0, 0, 0, 0, 0, 0


    def do_get_init_info(self, initstates):
        if issequence(initstates):
            h_t0 = initstates[0]
            mem_t0 = initstates[1]
            red = initstates[2:]
            m_t0 = T.zeros((h_t0.shape[0], self.innerdim))
        else:       # initstates is batchsize scalar
            h_t0 = T.zeros((initstates, self.innerdim))
            mem_t0 = T.zeros((initstates, self.memsize, self.innerdim))
            red = initstates
            m_t0 = T.zeros((initstates, self.innerdim))
        return [m_t0, mem_t0, h_t0], red

    def get_states_from_outputs(self, outputs):
        assert (len(outputs) == 2)
        return [outputs[[1, 2]]]

    def rec(self, x_t, m_tm1, mem_tm1, h_tm1):
        """
        :param x_t:     current input vector: (batsize, inp_dim)
        :param h_tm1:   previous state vector: (batsize, state_dim)
        :param m_tm1:   previous memory content vector: (batsize, state_dim)
        :param mem_tm1: previous memory state: (batsize, mem_size, state_dim)
        :return:    (y_t, h_t, m_t, mem_t)
        """
        # read memory
        memory_addr_gate1 =     self.gateactivation(T.dot(h_tm1, self.uma) + T.dot(x_t, self.wma) + T.dot(m_tm1, self.mma) + self.bma)
        memory_addr_gate2 =     self.gateactivation(T.dot(h_tm1, self.uma2) + T.dot(x_t, self.wma2) + T.dot(m_tm1, self.mma2) + self.bma2)
        memaddrcan         =    memory_addr_gate1 * h_tm1 +      (1 - memory_addr_gate1) * m_tm1
        memaddr =               memory_addr_gate2 * memaddrcan + (1 - memory_addr_gate2) * x_t      # TODO: ERROR HERE: x_t shape incompatible with internal shapes
        memsel = self.attgen(memaddr, mem_tm1)
        m_t = self.attcon(mem_tm1, memsel)

        # update inner stuff
        state_filter_gate =     self.gateactivation(T.dot(h_tm1, self.usf) + T.dot(x_t, self.wsf) + T.dot(m_t, self.msf) + self.bsf)
        memory_filter_gate =    self.gateactivation(T.dot(h_tm1, self.umf) + T.dot(x_t, self.wmf) + T.dot(m_t, self.mmf) + self.bmf)
        input_filter_gate =     self.gateactivation(T.dot(h_tm1, self.uif) + T.dot(x_t, self.wif) + T.dot(m_t, self.mif) + self.bif)
        update_gate     =       self.gateactivation(T.dot(h_tm1, self.uug) + T.dot(x_t, self.wug) + T.dot(m_t, self.mug) + self.bug)

        # compute new state
        h_tm1_filtered = T.dot(state_filter_gate * h_tm1, self.u)
        x_t_filtered =   T.dot(input_filter_gate * x_t, self.w)
        m_t_filtered = T.dot(memory_filter_gate * m_t, self.m)
        h_t_can = self.outpactivation(h_tm1_filtered + x_t_filtered + m_t_filtered + self.b)
        h_t = update_gate * h_tm1 + (1 - update_gate) * h_t_can

        # write memory
        memory_write_filter=    self.gateactivation(T.dot(h_tm1, self.uwf) + T.dot(x_t, self.wwf) + T.dot(m_t, self.mwf) + self.bwf)    # (batsize, state_dim)
        if self.discrete:       # memsel: (batsize, mem_size)
            memseln = T.zeros_like(memsel)
            memsel = T.argmax(memsel, axis=1)
            memseln[T.arange(memsel.shape[0]), memsel] = 1.0        # TODO: doesn't work
            memsel = memseln

        memwritesel = T.batched_tensordot(memsel, memory_write_filter, axes=0)  # (batsize, mem_size, state_dim)
        h_t_rep = h_t.reshape((h_t.shape[0], 1, h_t.shape[1])).repeat(mem_tm1.shape[1], axis=1)
        mem_t = memwritesel * mem_tm1 + (1 - memwritesel) * h_t_rep
        return [h_t, m_t, mem_t, h_t]