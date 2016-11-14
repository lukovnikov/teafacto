from teafacto.core.base import Block, asblock, Val, issequence, tensorops as T
from teafacto.blocks.seq.rnn import SeqEncoder, MaskSetMode, SeqDecoder
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.seq.attention import Attention, WeightedSumAttCon, AttGen
from teafacto.blocks.basic import MatDot
from teafacto.blocks.match import CosineDistance
from teafacto.use.recsearch import SeqEncDecWrapper


class SeqEncDec(Block):
    searchwrapper = SeqEncDecWrapper

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
        initstates, allenco = self.preapply(inpseq, inmask)
        deco = self.dec(allenco, outseq, initstates=initstates, mask=outmask)      # no state transfer
        return deco

    def preapply(self, inpseq, inmask=None):
        enco, allenco = self.enc(inpseq, mask=inmask)
        initstates = None
        if self.statetrans is not None:
            topstate = self.statetrans(enco, allenco)
            initstates = [topstate]
        return initstates, allenco

    def get_inits(self, inpseq, batsize, maskseq=None):
        initstates, allenco = self.preapply(inpseq, inmask=maskseq)
        return self.dec.get_inits(initstates, batsize, allenco)

    def userec(self, x_t, *states):
        return self.dec.userec(x_t, *states)


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
                 bidir=False,
                 rnu=GRU,
                 statetrans=None,
                 vecout=None,
                 inconcat=True,
                 outconcat=False,
                 maskid=-1,
                 dropout=False,
                 attdist=CosineDistance(),
                 sepatt=False,
                 **kw):
        encinnerdim = [encdim] if not issequence(encdim) else encdim
        decinnerdim = [decdim] if not issequence(decdim) else decdim

        # encoder
        if sepatt:
            enc = self._getencoder_sepatt(indim=inpvocsize, inpembdim=inpembdim, inpemb=inpemb,
                            innerdim=encinnerdim, bidir=bidir, maskid=maskid,
                            dropout_in=dropout, dropout_h=False, rnu=rnu)
        else:
            enc = self._getencoder(indim=inpvocsize, inpembdim=inpembdim, inpemb=inpemb,
                                innerdim=encinnerdim, bidir=bidir, maskid=maskid,
                                dropout_in=dropout, dropout_h=False, rnu=rnu)
        lastencinnerdim = enc.outdim

        # attention
        lastdecinnerdim = decinnerdim[-1]
        attgen = AttGen(attdist)
        attcon = WeightedSumAttCon()
        attention = Attention(attgen, attcon, separate=sepatt)

        # decoder
        dec = SeqDecoder.RNN(
            emb=outemb, embdim=outembdim, embsize=outvocsize, maskid=maskid,
            ctxdim=lastencinnerdim, attention=attention,
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

    def _getencoder(self, indim=None, inpembdim=None, inpemb=None, innerdim=None,
                    bidir=False, maskid=-1, dropout_in=False, dropout_h=False, rnu=GRU):
        enc = SeqEncoder.RNN(indim=indim, inpembdim=inpembdim, inpemb=inpemb,
                             innerdim=innerdim, bidir=bidir, maskid=maskid,
                             dropout_in=dropout_in, dropout_h=dropout_h, rnu=rnu) \
            .with_outputs().maskoptions(MaskSetMode.ZERO)
        return enc

    def _getencoder_sepatt(self, indim=None, inpembdim=None, inpemb=None, innerdim=None,
                    bidir=False, maskid=-1, dropout_in=False, dropout_h=False, rnu=GRU):
        enc_a = SeqEncoder.RNN(indim=indim, inpembdim=inpembdim, inpemb=inpemb,
                             innerdim=innerdim, bidir=bidir, maskid=maskid,
                             dropout_in=dropout_in, dropout_h=dropout_h, rnu=rnu) \
            .with_outputs().maskoptions(MaskSetMode.ZERO)

        enc_c = SeqEncoder.RNN(indim=indim, inpembdim=inpembdim, inpemb=inpemb,
                             innerdim=innerdim, bidir=bidir, maskid=maskid,
                             dropout_in=dropout_in, dropout_h=dropout_h, rnu=rnu) \
            .with_outputs().maskoptions(MaskSetMode.ZERO)
        return SepAttEncoders(enc_a, enc_c)


class SepAttEncoders(Block):
    def __init__(self, enc_a, enc_c, **kw):
        super(SepAttEncoders, self).__init__(**kw)
        self.enc_a = enc_a
        self.enc_c = enc_c

    def __getattr__(self, item):
        return getattr(self.enc_c, item)

    def apply(self, seq, weights=None, mask=None):
        att_enc_final, att_enc_all = self.enc_a(seq, weights=weights, mask=mask)
        con_enc_final, con_enc_all = self.enc_c(seq, weights=weights, mask=mask)
        encmask = con_enc_all.mask
        att_enc_all = att_enc_all.dimshuffle(0, 1, "x", 2)
        con_enc_all = con_enc_all.dimshuffle(0, 1, "x", 2)
        ret = T.concatenate([con_enc_all, att_enc_all], axis=2)
        ret.mask = encmask
        return con_enc_final, ret



# TODO: add sep-attentional enc-dec