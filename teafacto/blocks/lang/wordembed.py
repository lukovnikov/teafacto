from teafacto.core.base import *
from teafacto.core.base import tensorops as T
from teafacto.blocks.basic import IdxToOneHot, Embedder, VectorEmbed
from teafacto.blocks.rnn import SeqEncoder
from teafacto.blocks.rnu import RecurrentBlock, GRU
from teafacto.blocks.lang.wordvec import Glove


class WordEmbedGlove(Embedder):
    def __init__(self, indim=1000, outdim=50, trainfrac=0.0, **kw):
        super(WordEmbedGlove, self).__init__(indim, outdim, **kw)
        self.emb = Glove(outdim, vocabsize=indim, trainfrac=trainfrac).block

    def apply(self, idxs):
        return self.emb(idxs)


class WordEncoder(Block):
    def __init__(self, indim=220, outdim=200, **kw):    # indim is number of characters
        super(WordEncoder, self).__init__(**kw)
        self.enc = SeqEncoder(IdxToOneHot(indim),
                              GRU(dim=indim, innerdim=outdim))

    def apply(self, seq):       # seq: (batsize, maxwordlen) of character idxs
        enco = self.enc(seq)    # enco: (batsize, outdim) of floats
        return enco


class WordEmbedPlusGlove(Embedder):
    def __init__(self, indim=4000, outdim=100, embdim=50, embtrainfrac=0.0, **kw):
        super(WordEmbedPlusGlove, self).__init__(indim, outdim+embdim, **kw)
        self.glove = Glove(embdim, vocabsize=indim, trainfrac=embtrainfrac).block
        self.emb = VectorEmbed(indim=indim, dim=outdim)

    def apply(self, idxs):  # (batsize,) word idxs
        gemb = self.glove(idxs)         # (batsize, embdim)
        oemb = self.emb(idxs)           # (batsize, outdim),
        return T.concatenate([gemb, oemb], axis=1)  # (batsize, outdim+embdim)


class WordEncoderPlusGlove(Block):
    def __init__(self, numchars=220, numwords=4e5, encdim=100, embdim=50, embtrainfrac=0.0, glovepath=None, **kw):
        super(WordEncoderPlusGlove, self).__init__(**kw)
        self.glove = Glove(embdim, vocabsize=numwords, trainfrac=embtrainfrac, path=glovepath).block
        self.enc = WordEncoder(indim=numchars, outdim=encdim)

    def apply(self, seq):       # seq: (batsize, 1+maxwordlen): first column: Glove idxs, subsequent cols: char ids
        emb = self.glove(seq[:, 0])                 # (batsize, embdim)
        enc = self.enc(seq[:, 1:])                  # (batsize, encdim)
        return T.concatenate([emb, enc], axis=1)    # (batsize, embdim + encdim)