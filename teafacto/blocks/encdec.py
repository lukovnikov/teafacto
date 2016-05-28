# TODO move all encoder/decoders here
from teafacto.blocks.attention import Attention, LinearGateAttentionGenerator, WeightedSumAttCon
from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.rnn import SeqTransducer, RecStack, SeqDecoder, BiRNU
from teafacto.blocks.rnu import GRU
from teafacto.core.base import Block
from teafacto.util import issequence


class SeqEncDec(Block):
    def __init__(self, enc, dec, **kw):
        super(SeqEncDec, self).__init__(**kw)
        self.enc = enc
        self.dec = dec

    def apply(self, inpseq, outseq, maskseq=None):
        enco = self.enc(inpseq, mask=maskseq)
        deco = self.dec(enco, outseq)
        ret = SeqTransducer.applymask(deco, maskseq)
        return ret

    def get_init_info(self, inpseq, initstates, maskseq=None):
        enco = self.enc(inpseq, mask=maskseq)
        return self.dec.get_init_info(enco, None, initstates)

    def rec(self, x_t, *states):
        return self.dec.rec(x_t, *states)


class SeqEncDecAtt(SeqEncDec):
    def __init__(self, enclayers, declayers, attgen, attcon, decinnerdim, inconcat, outconcat, **kw):
        enc = RecStack(*enclayers)
        dec = SeqDecoder(
            declayers,
            attention=Attention(attgen, attcon),
            innerdim=decinnerdim,
            outconcat=outconcat,
            inconcat=inconcat
        )
        super(SeqEncDecAtt, self).__init__(enc, dec, **kw)


class SimpleSeqEncDecAtt(SeqEncDecAtt):
    def __init__(self,
                 inpvocsize=400,
                 inpembdim=50,
                 outvocsize=100,
                 outembdim=50,
                 encdim=100,
                 decdim=100,
                 attdim=100,
                 bidir=False,
                 rnu=GRU,
                 outconcat=True,
                 inconcat=True,
                 **kw):
        encinnerdim = [encdim] if not issequence(encdim) else encdim
        decinnerdim = [decdim] if not issequence(decdim) else decdim

        # encoder stack
        inpemb = VectorEmbed(indim=inpvocsize, dim=inpembdim)
        encrnus = []
        dims = [inpembdim] + encinnerdim
        i = 1
        while i < len(dims):
            if bidir:
                newrnu = BiRNU.fromrnu(rnu, dim=dims[i-1], innerdim=dims[i])
            else:
                newrnu = rnu(dim=dims[i-1], innerdim=dims[i])
            encrnus.append(newrnu)
            i += 1
        enclayers = [inpemb] + encrnus

        # attention
        attgen = LinearGateAttentionGenerator(indim=encinnerdim[-1] + decinnerdim[-1], attdim=attdim)
        attcon = WeightedSumAttCon()

        # decoder
        outemb = VectorEmbed(indim=outvocsize, dim=outembdim)
        decrnus = []
        firstdecdim = outembdim if inconcat is False else outembdim + encinnerdim[-1]
        dims = [firstdecdim] + decinnerdim
        i = 1
        while i < len(dims):
            decrnus.append(rnu(dim=dims[i-1], innerdim=dims[i]))
            i += 1
        declayers = [outemb] + decrnus
        decinnerdim = encinnerdim[-1] if outconcat is False else encinnerdim[-1] + decinnerdim[-1]
        super(SimpleSeqEncDecAtt, self).__init__(enclayers, declayers, attgen, attcon, decinnerdim, inconcat, outconcat, **kw)