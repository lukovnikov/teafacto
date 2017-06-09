from teafacto.blocks.basic import MatDot
from teafacto.blocks.match import DotDistance
from teafacto.blocks.seq.attention import Attention, WeightedSumAttCon, AttGen
from teafacto.blocks.seq.rnn import SeqEncoder, SeqDecoder
from teafacto.blocks.seq.rnu import GRU
from teafacto.core.base import Block, asblock, issequence, tensorops as T
from teafacto.use.recsearch import SeqEncDecWrapper


class SeqEncDec(Block):
    searchwrapper = SeqEncDecWrapper

    def __init__(self, enc, dec, statetrans=None, **kw):
        super(SeqEncDec, self).__init__(**kw)
        self.enc = enc
        self.dec = dec
        self.statetrans = statetrans

    def apply(self, inpseq, outseq):
        initstates, allenco = self.preapply(inpseq)
        deco = self.dec(allenco, outseq, initstates=initstates)
        return deco

    def preapply(self, inpseq):
        enco, allenco, states = self.enc(inpseq)   # bottom states first
        states = [state[:, -1, :] for state in states]
        initstates = self.make_initstates(states)
        return initstates, allenco

    def make_initstates(self, encstates):
        if self.statetrans != None:
            return self.statetrans(encstates, self.enc.get_statespec(), self.dec.get_statespec())
        else:
            return None

    def get_inits(self, inpseq, batsize):
        initstates, allenco = self.preapply(inpseq)
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
                 attdist=DotDistance(),
                 splitatt=None,
                 encoder=None,
                 decoder=None,
                 attention=None,
                 **kw):
        self.encinnerdim = [encdim] if not issequence(encdim) else encdim
        self.decinnerdim = [decdim] if not issequence(decdim) else decdim
        self.dropout = dropout

        # encoder
        if encoder is None:
            enc = self._getencoder(indim=inpvocsize, inpembdim=inpembdim, inpemb=inpemb,
                                innerdim=self.encinnerdim, bidir=bidir, maskid=maskid,
                                dropout_in=dropout, dropout_h=dropout, rnu=rnu)
        else:
            enc = encoder

        if attention is None:
            attention = self._getattention(attdist, splitatt=splitatt)
            self.attention = attention

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

    def _getattention(self, attdist, splitatt=None):
        attgen = AttGen(attdist)
        attcon = WeightedSumAttCon()
        attention = Attention(attgen, attcon, splitters=splitatt)
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
            .with_outputs().with_states()
        return enc

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
        enc = self._getencoder(indim=inpvocsize, inpembdim=inpembdim, inpemb=inpemb,
                         innerdim=innerdim, bidir=bidir, maskid=maskid,
                         dropout_in=dropout_in, dropout_h=dropout_h, rnu=rnu)
        self.statetrans = self._build_state_trans(self.statetrans_setting)
        self.enc = enc


# BELOW NEW PARALLEL ENCDEC STUFF (see also rnn.py)
from teafacto.blocks.seq.rnn import MakeRNU, RecStack


class EncDec(Block):        # NOT FOR STATIC CONTEXT TODO
                            # EXPLICIT STATE TRANSFER (by init state gen)
    def __init__(self,
                 encoder=None,      # encoder block
                 attention=None,    # attention block
                 attentiontransformer=None,  # transforms state to criterion vector used in attention
                 inconcat=True,     # concat att result to decoder inp
                 outconcat=False,   # concat att result to smo
                 stateconcat=True,  # concat top state to smo
                 updatefirst=False, # perform RNN update before attention
                 concatdecinp=False,    # directly concat(feed) dec input to att gen and to smo
                 inpemb=None,       # decoder input embedding block
                 indim=None,        # decoder input dim
                 innerdim=None,     # internal dim of dec rnn
                 rnu=GRU,           # dec rnn type
                 dropout_in=False,  # dropout dec rnn in
                 dropout_h=False,   # dropout dec rnn h
                 zoneout=False,     # zoneout dec rnn
                 smo=None,          # dec smo block
                 init_state_gen=None,   # block that generates initial state of topmost layer that produces addressing vectors
                 **kw):
        super(EncDec, self).__init__(**kw)
        self.inconcat = inconcat
        self.outconcat = outconcat
        self.stateconcat = stateconcat
        self.updatefirst = updatefirst
        self.concatdecinp = concatdecinp
        self.encoder = encoder
        self.inpemb = inpemb
        innerdim = innerdim if issequence(innerdim) else [innerdim]
        self.outdim = innerdim[-1]
        if indim is None:
            indim = inpemb.outdim
            if self.inconcat:
                indim += encoder.outdim
        self.init_state_gen = init_state_gen
        paraminitstates = init_state_gen is None
        layers, lastdim = MakeRNU.fromdims([indim] + innerdim, rnu=rnu,
                                  dropout_in=dropout_in,
                                  dropout_h=dropout_h,
                                  zoneout=zoneout,
                                  param_init_states=paraminitstates)
        self.block = RecStack(*layers)
        self.attention = attention
        self.attentiontransformer = attentiontransformer
        self.smo = smo              # softmax out block on decoder
        self.lastdim = lastdim
        self.outconcatdim = 0
        if self.stateconcat:    self.outconcatdim += lastdim
        if self.outconcat:      self.outconcatdim += self.encoder.outdim
        if self.concatdecinp:   self.outconcatdim += indim

    def apply(self, decinp, encinp):
        inpenc = self.encoder(encinp)   # 2D or 3D
        batsize = decinp.shape[0]
        init_info, nonseqs = self.get_inits(batsize, inpenc)
        decinpemb = self.inpemb(decinp)
        mask = decinpemb.mask
        outputs = T.scan(fn=self.inner_rec,
                         sequences=decinpemb.dimswap(1, 0),
                         outputs_info=[None] + init_info,
                         non_sequences=list(nonseqs),
                         )
        ret = outputs[0].dimswap(1, 0)
        ret.mask = mask
        return ret

    @property
    def numstates(self):
        return self.block.numstates

    def get_inits(self, batsize, ctx):
        init_state = self.init_state_gen(ctx) if self.init_state_gen is not None else None
        init_info = self._get_init_states(init_state, batsize)
        if ctx.ndim == 3:       # attention
            ctx_0 = T.zeros((batsize, ctx.shape[2]))
        else:
            ctx_0 = ctx
        nonseqs = self.get_nonseqs(ctx)
        return [ctx_0] + init_info, nonseqs

    def _get_init_states(self, initstates, batsize):
        if initstates is None:
            initstates = batsize
        elif issequence(initstates):
            if len(initstates) < self.numstates:  # fill up with batsizes for lower layers
                initstates = [batsize] * (self.numstates - len(initstates)) + initstates
        return self.get_init_info(initstates)

    def get_init_info(self, initstates):
        return self.block.get_init_info(initstates)

    def get_nonseqs(self, inpenc):
        ctx = inpenc
        ctxmask = ctx.mask if ctx.mask is not None else T.ones(ctx.shape[:2], dtype="float32")
        return ctxmask, ctx

    def inner_rec(self, x_t_emb, ctx_tm1, *args):
        ctx = args[-1]
        ctxmask = args[-2]
        states_tm1 = args[:-2]
        y_tm1 = states_tm1[-1]
        if self.updatefirst:
            i_t = T.concatenate([x_t_emb, ctx_tm1], axis=1) if self.inconcat else x_t_emb
            rnuret = self.block.rec(i_t, *states_tm1)
            o_t = rnuret[0]
            states_t = rnuret[1:]
            y_t = states_t[-1]
            ctx_t = self._get_ctx_t(ctx, y_t, x_t_emb, self.attention, ctxmask)
        else:
            ctx_t = self._get_ctx_t(ctx, y_tm1, x_t_emb, self.attention, ctxmask)
            i_t = T.concatenate([x_t_emb, ctx_t], axis=1) if self.inconcat else x_t_emb
            rnuret = self.block.rec(i_t, *states_tm1)
            o_t = rnuret[0]
            states_t = rnuret[1:]
        # output
        concatthis = []
        if self.stateconcat:            concatthis.append(o_t)
        if self.outconcat:              concatthis.append(ctx_t)
        if self.concatdecinp:           concatthis.append(x_t_emb)
        _y_t = T.concatenate(concatthis, axis=1) if len(concatthis) > 1 else concatthis[0]
        y_t = self.smo(_y_t) if self.smo is not None else _y_t        # TODO: smo == None --> return raw y_t
        return [y_t, ctx_t] + states_t

    def _get_ctx_t(self, ctx, h, x_t_emb, att, ctxmask):
        # ctx: 3D if attention, 2D otherwise
        if ctx.ndim == 2:
            return ctx      # no attention, return ctx as-is
        else:
            assert(ctx.ndim == 3)
            assert(att is not None)
        if self.concatdecinp:
            h = T.concatenate([h, x_t_emb], axis=-1)
        if self.attentiontransformer is not None:
            h = self.attentiontransformer(h)
        ret = att(h, ctx, mask=ctxmask)
        return ret


class MultiEncDec(Block):
    def __init__(self, encoders=None, slices=None, attentions=None,
                 indim=None,
                 inpemb=None, inconcat=True, outconcat=False,
                 innerdim=None, rnu=GRU, dropout_in=False, dropout_h=False,
                 smo=None, **kw):
        super(MultiEncDec, self).__init__(**kw)
        self.inconcat = inconcat
        self.outconcat = outconcat
        self.encoders = encoders
        self.inpemb = inpemb
        innerdim = innerdim if issequence(innerdim) else [innerdim]
        self.outdim = innerdim[-1]
        layers, lastdim = MakeRNU.fromdims([indim] + innerdim, rnu=rnu,
                                  dropout_in=dropout_in, dropout_h=dropout_h,
                                  param_init_states=True)
        self.block = RecStack(*layers)
        if not issequence(slices):
            slices = [slices]
        self.slices = (smo.indim,) + tuple(slices)        # slices to take from final layer's vector to feed to each attention
        self.attentions = attentions
        self.smo = smo              # softmax out block on decoder

    def apply(self, decinp, *encinps):
        inpencs = []
        for encinp, encoder in zip(encinps, self.encoders):
            inpencs.append(encoder(encinp))
        batsize = decinp.shape[0]
        init_info = self.get_init_info(batsize)   # blank init info
        nonseqs = self.get_nonseqs(*inpencs)
        self._numnonseqs = len(nonseqs)
        decinpemb = self.inpemb(decinp)
        mask = decinpemb.mask
        outputs = T.scan(fn=self.inner_rec,
                        sequences=decinpemb.dimswap(1, 0),
                        outputs_info=[None] + init_info,
                        non_sequences=nonseqs,
                        )
        ret = outputs[0].dimswap(1, 0)
        ret.mask = mask
        return ret

    def get_init_info(self, initstates):
        return self.block.get_init_info(initstates)

    def get_nonseqs(self, *inpencs):
        ctxs = []
        ctxmasks = []
        for inpenc in inpencs:
            ctx = inpenc
            ctxmask = ctx.mask if ctx.mask is not None else T.ones(ctx.shape[:2], dtype="float32")
            ctxs.append(ctx)
            ctxmasks.append(ctxmask)
        return ctxs + ctxmasks

    def inner_rec(self, x_t_emb, *args):
        ctxs = args[-self._numnonseqs:-self._numnonseqs/2]
        ctxmasks = args[-self._numnonseqs/2:]
        states_tm1 = args[:-self._numnonseqs]
        ctxs_t = []
        sliceleft = self.slices[0]
        h_tm1 = states_tm1[-1]          # left is bottom, left is inner
        for ctx, ctxmask, attention, slice in zip(ctxs, ctxmasks, self.attentions, self.slices[1:]):
            h_tm1_slice = h_tm1[:, sliceleft:sliceleft+slice]
            sliceleft += slice
            ctx_t = self._get_ctx_t(ctx, h_tm1_slice, attention, ctxmask)
            ctxs_t.append(ctx_t)
        ctx_t = T.concatenate(ctxs_t, axis=1)
        i_t = T.concatenate([x_t_emb, ctx_t], axis=1) if self.inconcat else x_t_emb
        rnuret = self.block.rec(i_t, *states_tm1)
        h_t = rnuret[0]
        states_t = rnuret[1:]
        _y_t_slice = h_t[:, :self.slices[0]]
        _y_t = T.concatenate([_y_t_slice, ctx_t], axis=1) if self.outconcat else _y_t_slice
        y_t = self.smo(_y_t)
        return [y_t] + states_t

    def _get_ctx_t(self, ctx, h, att, ctxmask):
        ret = att(h, ctx, mask=ctxmask)
        return ret
