from teafacto.core.base import param, Block, tensorops as T
from teafacto.core.stack import stack
from teafacto.blocks.basic import Softmax, MatDot, IdxToOneHot
from teafacto.blocks.rnn import RNNDecoder, RNNEncoder
from teafacto.blocks.rnu import GRU, LSTM


class RNNAutoEncoder(Block):    # tries to decode original sequence
    def __init__(self, vocsize=25, innerdim=200, seqlen=50, **kw):
        super(RNNAutoEncoder, self).__init__(**kw)
        self.seqlen = seqlen
        self.encstack = stack(
            IdxToOneHot(vocsize=vocsize),
            GRU(dim=vocsize, innerdim=innerdim)
        )
        self.encoder = RNNEncoder(self.encstack)
        self.decstack = stack(
            GRU(dim=vocsize, innerdim=innerdim),
            MatDot(indim=innerdim, dim=vocsize),
            Softmax()
        )
        self.decoder = RNNDecoder(self.decstack, indim=vocsize, seqlen=seqlen)

    def apply(self, inpseq):
        enc = self.encoder(inpseq)
        dec = self.decoder(enc, seqlen=inpseq.shape[1])
        return dec


