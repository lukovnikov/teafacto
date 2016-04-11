from teafacto.core.base import Block
from teafacto.blocks.basic import Softmax, MatDot as Lin, VectorEmbed
from teafacto.blocks.rnn import SeqEncoder, SeqDecoder, InConcatCRex
from teafacto.blocks.rnu import GRU
from teafacto.blocks.lang.wordembed import WordEncoderPlusGlove, WordEncoderPlusEmbed


class FBBasicCompositeEncoder(Block):    # SeqEncoder of WordEncoderPlusGlove, fed to single-layer Softmax output
    def __init__(self, wordembdim=50, wordencdim=100, innerdim=200, outdim=1e4, numwords=4e5, numchars=128, glovepath=None, **kw):
        super(FBBasicCompositeEncoder, self).__init__(**kw)
        self.indim = wordembdim + wordencdim
        self.outdim = outdim
        self.wordembdim = wordembdim
        self.wordencdim = wordencdim
        self.innerdim = innerdim

        self.enc = SeqEncoder(
            WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=self.wordencdim, embdim=self.wordembdim, embtrainfrac=0.0, glovepath=glovepath),
            GRU(dim=self.wordembdim + self.wordencdim, innerdim=self.innerdim)
        )

        self.out = Lin(indim=self.innerdim, dim=self.outdim)

    def apply(self, inp):
        enco = self.enc(inp)
        ret = Softmax()(self.out(enco))
        return ret


class FBSeqCompositeEncDec(Block):
    def __init__(self, wordembdim=50, wordencdim=100, entembdim=200, innerdim=200, outdim=1e4, numwords=4e5, numchars=128, glovepath=None, **kw):
        super(FBSeqCompositeEncDec, self).__init__(**kw)
        self.indim = wordembdim + wordencdim
        self.outdim = outdim
        self.wordembdim = wordembdim
        self.wordencdim = wordencdim
        self.encinnerdim = innerdim
        self.entembdim = entembdim
        self.decinnerdim = innerdim

        self.enc = SeqEncoder(
            WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=self.wordencdim, embdim=self.wordembdim, embtrainfrac=0.0, glovepath=glovepath),
            GRU(dim=self.wordembdim + self.wordencdim, innerdim=self.encinnerdim)
        )

        self.dec = SeqDecoder(
            VectorEmbed(indim=self.outdim, dim=self.entembdim),
            InConcatCRex(GRU(dim=self.entembdim+self.encinnerdim, innerdim=self.decinnerdim), outdim=self.decinnerdim)
        )

    def apply(self, inpseq, outseq):
        enco = self.enc(inpseq)
        deco = self.dec(enco, outseq)
        return deco