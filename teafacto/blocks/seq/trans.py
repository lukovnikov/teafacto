from teafacto.core.base import Block
from teafacto.blocks.seq.rnn import SeqEncoder, MakeRNU
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.basic import VectorEmbed, Linear as Lin, Softmax
from teafacto.util import issequence


class SeqTrans(SeqEncoder):
    def __init__(self, embedder, *layers, **kw):
        super(SeqTrans, self).__init__(embedder, *layers, **kw)
        self.all_outputs


class SimpleSeqTrans(SeqTrans):
    def __init__(self, indim=400, embdim=50, inpemb=None,
                 innerdim=100, outdim=50, rnu=GRU, **kw):
        if inpemb is None:
            self.emb = VectorEmbed(indim=indim, dim=embdim)
        else:
            self.emb = inpemb
            embdim = self.emb.outdim
        if not issequence(innerdim):
            innerdim = [innerdim]
        innerdim = [embdim] + innerdim
        self.rnn, _ = MakeRNU.fromdims(innerdim, rnu=rnu)
        self.smo = Lin(indim=innerdim[-1], dim=outdim)
        super(SimpleSeqTrans, self).__init__(self.emb,
                                             *(self.rnn
                                             + [self.smo,
                                                Softmax()]), **kw)
