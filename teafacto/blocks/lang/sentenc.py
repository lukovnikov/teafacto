from teafacto.blocks.seq.enc import SimpleSeqStar2Vec, SeqEncoder, MaskMode, EncLastDim
from teafacto.blocks.seq.rnn import MakeRNU
from teafacto.blocks.basic import VectorEmbed
from teafacto.core.base import Block, tensorops as T
from teafacto.util import issequence

# sentence encoders

class CharWordSentEnc(SimpleSeqStar2Vec):
    def __init__(self, numchars=256, charembdim=50, wordembdim=100, innerdim=200, maskid=None, **kw):
        super(CharWordSentEnc, self).__init__(indim=numchars, inpembdim=charembdim, innerdim=[wordembdim, innerdim], maskid=maskid)


class WordCharSentEnc(Block):
    def __init__(self, numchars=256, charembdim=50, charemb=None, charinnerdim=100,
                 numwords=1000, wordembdim=100, wordemb=None, wordinnerdim=200,
                 maskid=None, bidir=False, returnall=False, **kw):
        super(WordCharSentEnc, self).__init__(**kw)
        self.maskid = maskid
        # char level inits
        if charemb is None:
            self.charemb = VectorEmbed(indim=numchars, dim=charembdim)
        else:
            self.charemb = charemb
            charembdim = charemb.outdim
        if not issequence(charinnerdim):
            charinnerdim = [charinnerdim]
        charlayers, lastchardim = MakeRNU.make(charembdim, charinnerdim, bidir=bidir)
        self.charenc = SeqEncoder(self.charemb, *charlayers).maskoptions(maskid, MaskMode.AUTO)
        # word level inits
        if wordemb is None:
            self.wordemb = VectorEmbed(indim=numwords, dim=wordembdim)
        elif wordemb is False:
            self.wordemb = None
            wordembdim = 0
        else:
            self.wordemb = wordemb
            wordembdim = wordemb.outdim
        if not issequence(wordinnerdim):
            wordinnerdim = [wordinnerdim]
        wordlayers, outdim = MakeRNU.make(wordembdim + lastchardim, wordinnerdim, bidir=bidir)
        self.wordenc = SeqEncoder(None, *wordlayers).maskoptions(MaskMode.NONE)
        if returnall:
            self.wordenc.all_outputs()
        self.outdim = outdim

    def apply(self, x):     # (batsize, numwords, 1 + numcharsperword)
        if self.wordemb is not None:
            chartensor = x[:, :, 1:]
            wordencs = EncLastDim(self.charenc)(chartensor)     # (batsize, numwords, wordencdim)
            wordmat = x[:, :, 0]
            assert(wordmat.ndim == 2)
            wordembs = self.wordemb(wordmat)    # (batsize, numwords, wordembdim)
            wordvecs = T.concatenate([wordencs, wordembs], axis=2)
            wmask = T.neq(wordmat, self.maskid) if self.maskid is not None else None
        else:
            wordvecs = EncLastDim(self.charenc)(x)
            wmask = T.gt(T.sum(T.eq(x, self.maskid), axis=2), 0)
        sentenc = self.wordenc(wordvecs, mask=wmask)
        return sentenc