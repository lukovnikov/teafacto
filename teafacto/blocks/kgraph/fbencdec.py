from teafacto.core.base import Block
from teafacto.core.stack import stack
from teafacto.blocks.basic import Softmax, MatDot as Lin, VectorEmbed, ConcatBlock
from teafacto.blocks.rnn import SeqEncoder, SeqDecoder, InConcatCRex, RecurrentStack
from teafacto.blocks.rnu import GRU
from teafacto.blocks.lang.wordembed import WordEncoderPlusGlove
from teafacto.blocks.attention import LinearGateAttentionGenerator
from teafacto.blocks.memory import MemoryBlock, LinearGateMemAddr, GeneralDotMemAddr


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
    '''
    The input sequence is encoded into a vector with a GRU.
    Each input sequence element is mapped to a vector with the composite Glove + character encoding block.
    The encoding is passed to the decoder, as part of the decoder RNN's input.
    No attention in this model.

    '''
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
            InConcatCRex(
                GRU(dim=self.entembdim+self.encinnerdim, innerdim=self.decinnerdim),
                outdim=self.decinnerdim)
        )

    def apply(self, inpseq, outseq):
        enco = self.enc(inpseq)
        deco = self.dec(enco, outseq)
        return deco


class FBSeqCompositeEncMemDec(Block):
    def __init__(self,  wordembdim=50,
                        wordencdim=100,
                        entembdim=200,
                        innerdim=200,
                        outdim=1e4,
                        numwords=4e5,
                        numchars=128,
                        glovepath=None,
                        memdata=None,
                        attdim=100,
                        memaddr=GeneralDotMemAddr, **kw):
        super(FBSeqCompositeEncMemDec, self).__init__(**kw)
        self.indim = wordembdim + wordencdim
        self.outdim = outdim
        self.wordembdim = wordembdim
        self.wordencdim = wordencdim
        self.encinnerdim = innerdim
        self.entembdim = entembdim
        self.decinnerdim = innerdim

        #memory
        wencpg = WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=self.wordencdim, embdim=self.wordembdim, embtrainfrac=0.0, glovepath=glovepath)
        self.memenco = SeqEncoder(
            wencpg,
            GRU(dim=self.wordembdim + self.wordencdim, innerdim=self.encinnerdim)
        )

        entemb = VectorEmbed(indim=self.outdim, dim=self.entembdim)
        self.mempayload = ConcatBlock(entemb, self.memenco)
        self.memblock = MemoryBlock(self.mempayload, memdata, indim=self.outdim, outdim=self.encinnerdim+self.entembdim)

        #encoder
        #wencpg2 = WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=self.wordencdim, embdim=self.wordembdim, embtrainfrac=0.0, glovepath=glovepath)
        self.enc = SeqEncoder(
            wencpg,
            GRU(dim=self.wordembdim + self.wordencdim, innerdim=self.encinnerdim)
        )

        #decoder
        self.softmaxoutblock = stack(memaddr(self.memblock, indim=self.decinnerdim, memdim=self.memblock.outdim, attdim=attdim), Softmax())
        self.dec = SeqDecoder(
            self.memblock,
            InConcatCRex(
                GRU(dim=self.memblock.outdim + self.encinnerdim, innerdim=self.decinnerdim),
                outdim=self.decinnerdim),
            softmaxoutblock=self.softmaxoutblock
        )

    def apply(self, inpseq, outseq):
        enco = self.enc(inpseq)
        deco = self.dec(enco, outseq)
        return deco

