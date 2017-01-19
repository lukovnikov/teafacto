from teafacto.core.base import Block, asblock, Val, issequence, tensorops as T
from teafacto.blocks.seq.rnn import SeqEncoder, MaskSetMode, SeqDecoder
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.seq.attention import Attention, WeightedSumAttCon, AttGen
from teafacto.blocks.basic import MatDot
from teafacto.blocks.match import CosineDistance, DotDistance
from teafacto.use.recsearch import SeqEncDecWrapper
from IPython import embed


class SeqEncDec(Block):
    searchwrapper = SeqEncDecWrapper

    def __init__(self, enc, dec, statetrans=None, **kw):
        super(SeqEncDec, self).__init__(**kw)
        self.enc = enc
        self.dec = dec
        self.statetrans = None
        self.set_statetrans(statetrans)

    def set_statetrans(self, statetrans):
        if isinstance(statetrans, Block):
            self.statetrans = asblock(lambda x, y: statetrans(x))
        elif statetrans is True:
            self.statetrans = asblock(lambda x, y: x)
        else:
            self.statetrans = statetrans

    def apply(self, inpseq, outseq, inmask=None, outmask=None):
        initstates, allenco = self.preapply(inpseq, inmask)
        deco = self.dec(allenco, outseq, initstates=initstates, mask=outmask)
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

    def rec(self, x_t, *states):
        return self.dec.rec(x_t, *states)


class SimpleSeqEncDecAtt(SeqEncDec):
    """ RNN encoder decoder with attention """
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
        self.encinnerdim = [encdim] if not issequence(encdim) else encdim
        self.decinnerdim = [decdim] if not issequence(decdim) else decdim

        # encoder
        if sepatt:
            enc = self._getencoder_sepatt(indim=inpvocsize, inpembdim=inpembdim, inpemb=inpemb,
                            innerdim=self.encinnerdim, bidir=bidir, maskid=maskid,
                            dropout_in=dropout, dropout_h=dropout, rnu=rnu)
        else:
            enc = self._getencoder(indim=inpvocsize, inpembdim=inpembdim, inpemb=inpemb,
                                innerdim=self.encinnerdim, bidir=bidir, maskid=maskid,
                                dropout_in=dropout, dropout_h=dropout, rnu=rnu)
        self.lastencinnerdim = enc.outdim

        self.dropout = dropout
        # attention
        self.lastdecinnerdim = self.decinnerdim[-1]
        attgen = AttGen(attdist)
        attcon = WeightedSumAttCon()
        attention = Attention(attgen, attcon, separate=sepatt)

        # decoder
        dec = SeqDecoder.RNN(
            emb=outemb, embdim=outembdim, embsize=outvocsize, maskid=maskid,
            ctxdim=self.lastencinnerdim, attention=attention,
            innerdim=self.decinnerdim, inconcat=inconcat,
            softmaxoutblock=vecout, outconcat=outconcat,
            dropout=dropout, rnu=rnu, dropout_h=dropout,
        )
        self.statetrans_setting = statetrans

        statetrans = self._build_state_trans(self.statetrans_setting, self.lastencinnerdim, self.lastdecinnerdim)

        super(SimpleSeqEncDecAtt, self).__init__(enc, dec, statetrans=statetrans, **kw)

    def _build_state_trans(self, statetrans, encdim, decdim):
        # initial decoder state
        if statetrans is True:
            if encdim != decdim:  # state shape mismatch
                statetrans = MatDot(encdim, decdim)
        elif statetrans == "matdot":
            statetrans = MatDot(encdim, decdim)
        return statetrans

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

    def remake_encoder(self, inpvocsize=None, inpembdim=None, inpemb=None, innerdim=None,
                       bidir=False, maskid=-1, dropout_in=False, dropout_h=False, rnu=GRU,
                       sepatt=False):
        innerdim = ([innerdim] if not issequence(innerdim) else innerdim) if innerdim is not None else self.encinnerdim
        if sepatt:
            enc = self._getencoder_sepatt(indim=inpvocsize, inpembdim=inpembdim, inpemb=inpemb,
                                    innerdim=innerdim, bidir=bidir, maskid=maskid,
                                    dropout_in=dropout_in, dropout_h=dropout_h, rnu=rnu)
        else:
            enc = self._getencoder(indim=inpvocsize, inpembdim=inpembdim, inpemb=inpemb,
                             innerdim=innerdim, bidir=bidir, maskid=maskid,
                             dropout_in=dropout_in, dropout_h=dropout_h, rnu=rnu)
        lastencinnerdim = enc.outdim
        lastdecinnerdim = self.lastdecinnerdim
        statetrans = self._build_state_trans(self.statetrans_setting, lastencinnerdim, lastdecinnerdim)
        self.set_statetrans(statetrans)
        self.enc = enc



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
