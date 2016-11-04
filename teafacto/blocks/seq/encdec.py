from teafacto.core.base import Block, asblock, Val, issequence
from teafacto.blocks.seq.rnn import SeqEncoder, MaskMode, MaskSetMode, SeqDecoder, SeqDecoderAtt, BiRNU
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.seq.attention import Attention, WeightedSumAttCon, DotprodAttGen, GenDotProdAttGen, ForwardAttGen
from teafacto.blocks.basic import VectorEmbed, IdxToOneHot, MatDot


class SeqEncDec(Block):
    def __init__(self, enc, dec, statetrans=None, dropout=False, **kw):
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

    # TODO: DON'T USE THIS --> USE DEDICATED SAMPLER INSTEAD this should be called "start()" <-- REFACTOR
    def get_init_info(self, inpseq, batsize, maskseq=None):     # TODO: must evaluate enc here, in place, without any side effects
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


class SeqEncDecAtt(SeqEncDec):
    def __init__(self, enclayers, declayers, attgen, attcon,
                 decinnerdim, statetrans=None, vecout=False,
                 inconcat=True, outconcat=False, maskid=-1, dropout=False, **kw):
        enc = SeqEncoder(*enclayers, dropout=dropout)\
            .with_outputs()\
            .maskoptions(maskid, MaskMode.AUTO, MaskSetMode.ZERO)
        smo = False if vecout else None
        dec = SeqDecoderAtt(
            declayers,
            attention=Attention(attgen, attcon),
            innerdim=decinnerdim, inconcat=inconcat,
            softmaxoutblock=smo, outconcat=outconcat,
            dropout=dropout
        )
        super(SeqEncDecAtt, self).__init__(enc, dec, statetrans=statetrans, dropout=dropout, **kw)


class SimpleSeqEncDecAtt(SeqEncDecAtt):
    def __init__(self,
                 inpvocsize=400,
                 inpembdim=None,
                 outvocsize=100,
                 outembdim=None,
                 encdim=100,
                 decdim=100,
                 attdim=100,
                 bidir=False,
                 rnu=GRU,
                 statetrans=None,
                 vecout=False,
                 inconcat=True,
                 outconcat=False,
                 maskid=-1,
                 dropout=False,
                 **kw):
        encinnerdim = [encdim] if not issequence(encdim) else encdim
        decinnerdim = [decdim] if not issequence(decdim) else decdim

        self.enclayers, lastencinnerdim = \
            self.getenclayers(inpembdim, inpvocsize, encinnerdim, bidir, rnu, maskid=maskid, dropout=dropout)

        self.declayers = \
            self.getdeclayers(outembdim, outvocsize, lastencinnerdim,
                              decinnerdim, rnu, inconcat, maskid=maskid, dropout=dropout)

        # attention
        lastdecinnerdim = decinnerdim[-1]
        argdecinnerdim = lastdecinnerdim if outconcat is False else lastencinnerdim + lastdecinnerdim
        attgen = DotprodAttGen()
        attcon = WeightedSumAttCon()

        if statetrans is True:
            if lastencinnerdim != lastdecinnerdim:  # state shape mismatch
                statetrans = MatDot(lastencinnerdim, lastdecinnerdim)
        elif statetrans == "matdot":
            statetrans = MatDot(lastencinnerdim, lastdecinnerdim)

        super(SimpleSeqEncDecAtt, self).__init__(self.enclayers, self.declayers,
            attgen, attcon, argdecinnerdim, statetrans=statetrans, vecout=vecout,
            inconcat=inconcat, outconcat=outconcat, dropout=dropout, **kw)

    def getenclayers(self, inpembdim, inpvocsize, encinnerdim, bidir, rnu, maskid=-1, dropout=False):
        if inpembdim is None:
            inpemb = IdxToOneHot(inpvocsize)
            inpembdim = inpvocsize
        elif isinstance(inpembdim, Block):
            inpemb = inpembdim
            inpembdim = inpemb.outdim
        else:
            inpemb = VectorEmbed(indim=inpvocsize, dim=inpembdim, maskid=maskid)
        encrnus = []
        dims = [inpembdim] + encinnerdim
        #print dims
        i = 1
        lastencinnerdim = dims[-1] if not bidir else dims[-1] * 2
        while i < len(dims):
            if bidir:
                newrnu = BiRNU.fromrnu(rnu, dim=dims[i - 1], innerdim=dims[i], dropout_in=dropout, dropout_h=False)
            else:
                newrnu = rnu(dim=dims[i - 1], innerdim=dims[i], dropout_in=dropout, dropout_h=False)
            encrnus.append(newrnu)
            i += 1
        enclayers = [inpemb] + encrnus
        return enclayers, lastencinnerdim

    def getdeclayers(self, outembdim, outvocsize, lastencinnerdim,
                     decinnerdim, rnu, inconcat, maskid=-1, dropout=False):
        if outembdim is None:
            outemb = IdxToOneHot(outvocsize)
            outembdim = outvocsize
        elif isinstance(outembdim, Block):
            outemb = outembdim
            outembdim = outemb.outdim
        else:
            outemb = VectorEmbed(indim=outvocsize, dim=outembdim, maskid=maskid)
        decrnus = []
        firstdecdim = outembdim + lastencinnerdim if inconcat else outembdim
        dims = [firstdecdim] + decinnerdim
        i = 1
        while i < len(dims):
            decrnus.append(rnu(dim=dims[i - 1], innerdim=dims[i], dropout_in=dropout, dropout_h=False))
            i += 1
        declayers = [outemb] + decrnus
        return declayers