from teafacto.core.base import Block, asblock, Val, issequence
from teafacto.blocks.seq.rnn import SeqEncoder, MaskSetMode, SeqDecoderAtt
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.seq.attention import Attention, WeightedSumAttCon, AttGen
from teafacto.blocks.basic import MatDot
from teafacto.blocks.match import CosineDistance


class SeqEncDec(Block):
    def __init__(self, enc, dec, statetrans=None, **kw):
        super(SeqEncDec, self).__init__(**kw)
        self.enc = enc
        self.dec = dec
        if isinstance(statetrans, Block):
            self.statetrans = asblock(lambda x, y: statetrans(x))
        elif statetrans is True:
            self.statetrans = asblock(lambda x, y: x)
        else:
            self.statetrans = statetrans

    def apply(self, inpseq, outseq, inmask=None, outmask=None):
        enco, allenco = self.enc(inpseq, mask=inmask)
        if self.statetrans is not None:
            topstate = self.statetrans(enco, allenco)
            deco = self.dec(allenco, outseq, initstates=[topstate], mask=outmask)
        else:
            deco = self.dec(allenco, outseq, mask=outmask)      # no state transfer
        return deco

    # DON'T USE THIS --> USE DEDICATED SAMPLER INSTEAD this should be called "start()" <-- REFACTOR
    def get_init_info(self, inpseq, batsize, maskseq=None):     # must evaluate enc here, in place, without any side effects
        """
        VERY DIFFERENT FROM THE PURELY SYMBOLIC GET_INIT_INFO IN REAL REC BLOCKS !!!
        This one is used in decoder/prediction
        """
        enco, allenco, _ = self.enc.predict(inpseq, mask=maskseq)

        if self.statetrans is not None:
            topstate = self.statetrans.predict(enco, allenco)   # this gives unused input warning in theano - it's normal
            initstates = [topstate]
        else:
            initstates = batsize
        return self.dec.get_init_info([Val(x) for x in initstates]
                                            if issequence(initstates)
                                            else initstates)

    def rec(self, x_t, *states):
        return self.dec.rec(x_t, *states)


class SimpleSeqEncDecAtt(SeqEncDec):
    def __init__(self,
                 inpvocsize=400,
                 inpembdim=None,
                 inpemb=None,
                 outvocsize=100,
                 outembdim=None,
                 outemb=None,
                 encdim=100,
                 decdim=100,
                 attdim=100,
                 bidir=False,
                 rnu=GRU,
                 statetrans=None,
                 vecout=None,
                 inconcat=True,
                 outconcat=False,
                 maskid=-1,
                 dropout=False,
                 attdist=CosineDistance(),
                 **kw):
        encinnerdim = [encdim] if not issequence(encdim) else encdim
        decinnerdim = [decdim] if not issequence(decdim) else decdim

        # encoder
        enc = SeqEncoder.RNN(indim=inpvocsize, inpembdim=inpembdim, inpemb=inpemb,
                            innerdim=encinnerdim, bidir=bidir, maskid=maskid,
                            dropout_in=dropout, dropout_h=False, rnu=rnu) \
            .with_outputs().maskoptions(MaskSetMode.ZERO)
        lastencinnerdim = enc.outdim

        # attention
        lastdecinnerdim = decinnerdim[-1]
        attgen = AttGen(attdist)
        attcon = WeightedSumAttCon()

        # decoder
        dec = SeqDecoderAtt.RNN(
            emb=outemb, embdim=outembdim, embsize=outvocsize, maskid=maskid,
            ctxdim=lastencinnerdim, attention=Attention(attgen, attcon),
            innerdim=decinnerdim, inconcat=inconcat,
            softmaxoutblock=vecout, outconcat=outconcat,
            dropout=dropout, rnu=rnu
        )

        # initial decoder state
        if statetrans is True:
            if lastencinnerdim != lastdecinnerdim:  # state shape mismatch
                statetrans = MatDot(lastencinnerdim, lastdecinnerdim)
        elif statetrans == "matdot":
            statetrans = MatDot(lastencinnerdim, lastdecinnerdim)

        super(SimpleSeqEncDecAtt, self).__init__(enc, dec, statetrans=statetrans, **kw)