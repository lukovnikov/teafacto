from teafacto.blocks.basic import MatDot as Lin, Softmax
from teafacto.blocks.rnn import RNNDecoder
from teafacto.blocks.rnu import GRU
from teafacto.core.base import Block


class vec2sec(Block):
    def __init__(self, indim=50, innerdim=300, seqlen=20, vocsize=27, **kw):
        super(vec2sec, self).__init__(**kw)
        self.indim = indim
        self.innerdim=innerdim
        self.seqlen = seqlen
        self.vocsize = vocsize
        self.lin = Lin(indim=self.indim, dim=self.innerdim)
        self.dec = RNNDecoder(                                  # IdxToOneHot inserted automatically
            GRU(dim=self.vocsize, innerdim=self.innerdim),      # the decoding RNU
            Lin(indim=self.innerdim, dim=self.vocsize),         # transforms from RNU inner dims to vocabulary
            Softmax(),                                          # softmax
                indim=self.vocsize, seqlen=self.seqlen)

    def apply(self, vec):
        return self.dec(self.lin(vec))