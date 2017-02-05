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
        self.statetrans = statetrans

    def apply(self, inpseq, outseq, inmask=None, outmask=None):
        initstates, allenco = self.preapply(inpseq, inmask)
        deco = self.dec(allenco, outseq, initstates=initstates, mask=outmask)
        return deco

    def preapply(self, inpseq, inmask=None):
        enco, allenco, states = self.enc(inpseq, mask=inmask)   # bottom states first
        states = [state[:, -1, :] for state in states]
        initstates = self.make_initstates(states)
        return initstates, allenco

    def make_initstates(self, encstates):
        if self.statetrans != None:
            return self.statetrans(encstates, self.enc.get_statespec(), self.dec.get_statespec())
        else:
            return None

    def get_inits(self, inpseq, batsize, maskseq=None):
        initstates, allenco = self.preapply(inpseq, inmask=maskseq)
        return self.dec.get_inits(initstates, batsize, allenco)

    def rec(self, x_t, *states):
        return self.dec.rec(x_t, *states)


class SimpleSeqEncDec(SeqEncDec):
    def __init__(self, inpvocsize=400,
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
                 maskid=-1,
                 dropout=False,
                 encoder=None,
                 decoder=None,
                 **kw):
        pass


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
                 encoder=None,
                 decoder=None,
                 attention=None,
                 **kw):
        self.encinnerdim = [encdim] if not issequence(encdim) else encdim
        self.decinnerdim = [decdim] if not issequence(decdim) else decdim
        self.dropout = dropout

        # encoder
        if encoder is None:
            if sepatt:
                enc = self._getencoder_sepatt(indim=inpvocsize, inpembdim=inpembdim, inpemb=inpemb,
                                innerdim=self.encinnerdim, bidir=bidir, maskid=maskid,
                                dropout_in=dropout, dropout_h=dropout, rnu=rnu)
            else:
                enc = self._getencoder(indim=inpvocsize, inpembdim=inpembdim, inpemb=inpemb,
                                    innerdim=self.encinnerdim, bidir=bidir, maskid=maskid,
                                    dropout_in=dropout, dropout_h=dropout, rnu=rnu)
        else:
            enc = encoder

        if attention is None:
            attention = self._getattention(attdist, sepatt=sepatt)

        self.lastencinnerdim = enc.outdim
        if decoder is None:
            dec = self._getdecoder(outvocsize=outvocsize, outembdim=outembdim, outemb=outemb,
                                   maskid=maskid, attention=attention, lastencinnerdim=self.lastencinnerdim,
                                   decinnerdim=self.decinnerdim, inconcat=inconcat, outconcat=outconcat,
                                   softmaxout=vecout, dropout=dropout, rnu=rnu)
        else:
            dec = decoder

        self.lastdecinnerdim = self.decinnerdim[-1]
        self.statetrans_setting = statetrans
        statetrans = self._build_state_trans(self.statetrans_setting)

        super(SimpleSeqEncDecAtt, self).__init__(enc, dec, statetrans=statetrans, **kw)

    def _getattention(self, attdist, sepatt=False):
        attgen = AttGen(attdist)
        attcon = WeightedSumAttCon()
        attention = Attention(attgen, attcon, separate=sepatt)
        return attention

    def _build_state_trans(self, statetrans):
        if statetrans is None:
            return None

        def innerf(encstates, encspec, decspec):
            decspec = reduce(lambda x, y: list(x) + list(y), decspec, [])
            encspec = reduce(lambda x, y: list(x) + list(y), encspec, [])
            assert(len(decspec) == len(encspec))
            ret = []
            for i in range(len(encspec)):
                if encspec[i][0] == "state" and decspec[i][0] == "state":
                    if decspec[i][1][0] != encspec[i][1][0] or statetrans == "matdot":
                        t = MatDot(encspec[i][1][0], decspec[i][1][0])
                    else:
                        t = asblock(lambda x: x)
                elif encspec[i][0] == decspec[i][0]:
                    t = None
                else:
                    raise Exception()
                ret.append(t)
            assert(len(encstates) == len(ret))
            out = []
            for encstate, rete in zip(encstates, ret):
                if rete is None:
                    out.append(None)
                else:
                    out.append(rete(encstate))
            return out
        return innerf

    def _getencoder(self, indim=None, inpembdim=None, inpemb=None, innerdim=None,
                    bidir=False, maskid=-1, dropout_in=False, dropout_h=False, rnu=GRU):
        enc = SeqEncoder.RNN(indim=indim, inpembdim=inpembdim, inpemb=inpemb,
                             innerdim=innerdim, bidir=bidir, maskid=maskid,
                             dropout_in=dropout_in, dropout_h=dropout_h, rnu=rnu) \
            .with_outputs().with_states().maskoptions(MaskSetMode.ZERO)
        return enc

    def _getencoder_sepatt(self, indim=None, inpembdim=None, inpemb=None, innerdim=None,
                    bidir=False, maskid=-1, dropout_in=False, dropout_h=False, rnu=GRU):
        enc_a = SeqEncoder.RNN(indim=indim, inpembdim=inpembdim, inpemb=inpemb,
                             innerdim=innerdim, bidir=bidir, maskid=maskid,
                             dropout_in=dropout_in, dropout_h=dropout_h, rnu=rnu) \
            .with_outputs().with_states().maskoptions(MaskSetMode.ZERO)

        enc_c = SeqEncoder.RNN(indim=indim, inpembdim=inpembdim, inpemb=inpemb,
                             innerdim=innerdim, bidir=bidir, maskid=maskid,
                             dropout_in=dropout_in, dropout_h=dropout_h, rnu=rnu) \
            .with_outputs().with_states().maskoptions(MaskSetMode.ZERO)
        return SepAttEncoders(enc_a, enc_c)

    def _getdecoder(self, outvocsize=None, outembdim=None, outemb=None, maskid=-1,
                    attention=None, lastencinnerdim=None, decinnerdim=None, inconcat=False,
                    outconcat=True, softmaxout=None, dropout=None, rnu=None):
        lastencinnerdim = self.lastencinnerdim if lastencinnerdim is None else lastencinnerdim
        decinnerdim = self.decinnerdim if decinnerdim is None else decinnerdim
        rnu = GRU if rnu is None else rnu
        dec = SeqDecoder.RNN(
            emb=outemb, embdim=outembdim, embsize=outvocsize, maskid=maskid,
            ctxdim=lastencinnerdim, attention=attention,
            innerdim=decinnerdim, inconcat=inconcat,
            softmaxoutblock=softmaxout, outconcat=outconcat,
            dropout=dropout, rnu=rnu, dropout_h=dropout,
        )
        return dec

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
        att_enc_final, att_enc_all, att_enc_states = self.enc_a(seq, weights=weights, mask=mask)
        con_enc_final, con_enc_all, con_enc_states = self.enc_c(seq, weights=weights, mask=mask)
        encmask = con_enc_all.mask
        att_enc_all = att_enc_all.dimshuffle(0, 1, "x", 2)
        con_enc_all = con_enc_all.dimshuffle(0, 1, "x", 2)
        ret = T.concatenate([con_enc_all, att_enc_all], axis=2)
        ret.mask = encmask
        return con_enc_final, ret, con_enc_states
