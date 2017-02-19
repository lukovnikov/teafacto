from teafacto.core.base import Block
from teafacto.blocks.seq.rnn import SeqEncoder, MakeRNU, MaskMode
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.basic import VectorEmbed, Linear as Lin, Softmax
from teafacto.util import issequence


class SeqTrans(Block):
    def __init__(self, embedder, *layers, **kw):
        super(SeqTrans, self).__init__(**kw)
        self.enc = SeqEncoder(embedder, *layers)
        self.enc.all_outputs().maskoption(MaskMode.NONE)

    def apply(self, x):
        return self.enc(x)


class SimpleSeqTrans(SeqTrans):
    def __init__(self, indim=400, embdim=50, inpemb=None, bidir=False,
                 innerdim=100, outdim=50, rnu=GRU, dropout=False, **kw):
        if inpemb is None:
            emb = VectorEmbed(indim=indim, dim=embdim)
        else:
            emb = inpemb
            embdim = emb.outdim
        if not issequence(innerdim):
            innerdim = [innerdim]
        innerdim = [embdim] + innerdim
        rnn, lastdim = MakeRNU.fromdims(innerdim, rnu=rnu,
                                        dropout_h=dropout,
                                        dropout_in=dropout,
                                        bidir=bidir)
        smo = Lin(indim=lastdim, dim=outdim)
        super(SimpleSeqTrans, self).__init__(emb, *(rnn + [smo, Softmax()]), **kw)
