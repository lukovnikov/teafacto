from teafacto.blocks.seq.rnu import GRU, ReccableBlock
from teafacto.blocks.seq.attention import Attention
from teafacto.core import T
from teafacto.blocks.activations import GumbelSoftmax
from teafacto.util import argprun
from IPython import embed


class ReGRU(ReccableBlock):
    def __init__(self, indim=100, outdim=100,
                 memsize=10, mem_sample_temp=0.1,
                 wreg=0.0, dropout_in=False, dropout_h=False,
                 _debug=False, **kw):
        super(ReGRU, self).__init__(**kw)
        self.gru = GRU(dim=indim, innerdim=outdim * 2,
                       wreg=wreg, dropout_h=dropout_h, dropout_in=dropout_in)
        self.memsize = memsize
        self.outdim = outdim
        self.att = Attention().dot_gen()
        self.att.attentiongenerator.normalizer = GumbelSoftmax(temperature=mem_sample_temp)
        if _debug:
            self.att.attentiongenerator.sampler._debug = _debug

    def get_init_info(self, initstates):
        gru_h_0 = self.gru.get_init_info(initstates)
        batsize = gru_h_0[0].shape[0]
        M_0 = T.zeros((batsize, self.memsize, self.outdim))
        return [M_0] + gru_h_0

    def rec(self, x_t, M_tm1, s_tm1):
        h_tm1 = s_tm1[:, :self.outdim]
        m_tm1 = s_tm1[:, self.outdim:]
        m_t, M_t, w = self._swap_mem(h_tm1, m_tm1, M_tm1)
        s_tm1bis = T.concatenate([h_tm1, m_t], axis=-1)
        s_t, _ = self.gru.rec(x_t, s_tm1bis)
        h_t = s_t[:, :self.outdim]
        return [h_t, M_t, s_t]

    def _swap_mem(self, h_tm1, m_tm1, M_tm1):   # h_tm1: (batsize, outdim),
                                        # m_tm1: (batsize, outdim),
                                        # M_tm1: (batsize, memsize, outdim)
        # place attention over M using h
        w = self.att.attentiongenerator(h_tm1, M_tm1)   # (batsize, memsize)
        # retrieve m_t using sampled attention from M_tm1
        m_t = self.att.attentionconsumer(M_tm1, w)
        # store m_tm1 in attended M cells by weight -> M_t
        M_t = M_tm1 * (1 - w.dimadd(2)) + m_tm1.dimadd(1) * w.dimadd(2)
        return m_t, M_t, w


def run(lr=0.5):
    m = ReGRU()
    inits = m.get_init_info(5)
    embed()


if __name__ == "__main__":
    argprun(run)