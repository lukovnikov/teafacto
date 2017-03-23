from teafacto.blocks.seq.rnu import GRU, ReccableBlock, Gate
from teafacto.blocks.seq.attention import WeightedSumAttCon
from teafacto.blocks.match import DotDistance
from teafacto.core import T, param
from teafacto.blocks.activations import GumbelSoftmax, Softplus
from teafacto.util import argprun
from IPython import embed


class ReGRU(ReccableBlock):
    def __init__(self, indim=100, outdim=100,
                 memsize=10, mem_sample_temp=None,
                 wreg=0.0, dropout_in=False, dropout_h=False,
                 mem_gates=False, nobias=False,
                 _debug=False, **kw):
        super(ReGRU, self).__init__(**kw)
        self.gru = GRU(dim=indim, innerdim=outdim * 2,
                       wreg=wreg, dropout_h=dropout_h, dropout_in=dropout_in)
        self.memsize = memsize
        self.outdim = outdim
        self.att_dist = DotDistance()
        self.att_norm = GumbelSoftmax(temperature=mem_sample_temp if mem_sample_temp is not None else 1.0)
        self.att_cons = WeightedSumAttCon()
        self.temp = mem_sample_temp
        if self.temp is None:
            self.temp_W = param((outdim,), name="temp_W").uniform()
            self.temp_b = param((1,), name="temp_b").uniform()
        self.mem_gates = mem_gates
        self.lookup_gate_mem = True
        if self.mem_gates:
            self.lookupgate = Gate([indim, outdim, outdim], outdim, nobias=nobias)
            self.writegate = Gate([indim, outdim, outdim, outdim], outdim, nobias=nobias)
        if _debug:
            self.att_norm = _debug

    def get_init_info(self, initstates):
        gru_h_0 = self.gru.get_init_info(initstates)
        batsize = gru_h_0[0].shape[0]
        M_0 = T.zeros((batsize, self.memsize, self.outdim))
        return [M_0] + gru_h_0

    def rec(self, x_t, M_tm1, s_tm1):
        h_tm1 = s_tm1[:, :self.outdim]
        m_tm1 = s_tm1[:, self.outdim:]
        m_t, M_t, w = self._swap_mem(x_t, h_tm1, m_tm1, M_tm1)
        s_tm1bis = T.concatenate([h_tm1, m_t], axis=-1)
        s_t, _ = self.gru.rec(x_t, s_tm1bis)
        h_t = s_t[:, :self.outdim]
        return [h_t, M_t, s_t]

    def _swap_mem(self, x_t, h_tm1, m_tm1, M_tm1):   # h_tm1: (batsize, outdim),
                                        # m_tm1: (batsize, outdim),
                                        # M_tm1: (batsize, memsize, outdim)
        temps = None
        if self.temp is None:
            temps = Softplus()(T.dot(h_tm1, self.temp_W) + self.temp_b[0]) + 1
        # place attention over M using h
        h_lookup = h_tm1
        M_lookup = M_tm1
        if self.mem_gates:
            lgate = self.lookupgate(x_t, h_tm1, m_tm1)  # (batsize, outdim)
            h_lookup *= lgate
            if self.lookup_gate_mem:
                M_lookup *= lgate.dimadd(1)
        w = self.att_dist(h_lookup, M_lookup)   # (batsize, memsize)
        w = self.att_norm(w, temps=temps)
        # retrieve m_t using sampled attention from M_tm1
        m_t = self.att_cons(M_tm1, w)
        # store m_tm1 in attended M cells by weight -> M_t
        m_write = m_tm1
        if self.mem_gates:
            wgate = self.writegate(x_t, h_tm1, m_tm1, m_t)
            m_write *= wgate
        M_t = M_tm1 * (1 - w.dimadd(2)) + m_write.dimadd(1) * w.dimadd(2)
        return m_t, M_t, w


def run(lr=0.5):
    m = ReGRU()
    inits = m.get_init_info(5)
    embed()


if __name__ == "__main__":
    argprun(run)