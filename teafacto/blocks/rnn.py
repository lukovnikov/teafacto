import inspect

from teafacto.blocks.attention import WeightedSumAttCon, Attention, AttentionConsumer, LinearGateAttentionGenerator
from teafacto.blocks.basic import IdxToOneHot, VectorEmbed, Softmax, MatDot
from teafacto.blocks.basic import MatDot as Lin
from teafacto.blocks.rnu import GRU, ReccableBlock, RecurrentBlock, RNUBase
from teafacto.core.base import Block, tensorops as T, asblock
from teafacto.util import issequence, getnumargs


class RecStack(ReccableBlock):
    # must handle RecurrentBlocks ==> can not recappl, if all ReccableBlocks ==> can do recappl
    # must give access to final states of internal layers
    # must give access to all outputs of top layer
    # must handle masks
    def __init__(self, *layers, **kw):
        super(RecStack, self).__init__(**kw)
        self.layers = layers

    @property
    def numstates(self):
        return reduce(lambda x, y: x.numstates + y.numstates, [x for x in self.layers if isinstance(x, RecurrentBlock)], 0)

    # FWD API. initial states can be set, mask is accepted, everything is returned. Works for all RecurrentBlocks
    # FWD API IMPLEMENTED USING FWD API
    def innerapply(self, seq, mask=None, initstates=None):
        states = []
        for layer in self.layers:
            if isinstance(layer, RecurrentBlock):
                if initstates is not None:
                    layerinpstates = initstates[:layer.numstates]
                    initstates = initstates[layer.numstates:]
                else:
                    layerinpstates = None
                seq, layerstates = layer.innerapply(seq, mask=mask, initstates=layerinpstates)
                states.extend(layerstates)
            elif isinstance(layer, Block):
                seq = self.recurnonreclayer(seq, layer)
            else:
                raise Exception("can not apply this layer: " + str(layer) + " in RecStack")
        return seq, states           # full history of final output and all states (ordered from bottom layer to top)

    @classmethod
    def apply_mask(cls, xseq, maskseq=None):
        if maskseq is None:
            ret = xseq
        else:
            mask = T.tensordot(maskseq, T.ones((xseq.shape[2],)), 0)  # f32^(batsize, seqlen, outdim) -- maskseq stacked
            ret = mask * xseq
        return ret

    @classmethod
    def dummyrec(cls, layer):
        def innerrec(x_t):
            return layer(x_t)

        return innerrec

    @classmethod
    def recurnonreclayer(cls, x, layer):
        y, _ = T.scan(fn=cls.dummyrec(layer),
                      sequences=x.dimswap(1, 0),
                      outputs_info=None)
        return y.dimswap(1, 0)

    # REC API: only works with ReccableBlocks
    def get_init_info(self, initstates):
        recurrentlayers = list(filter(lambda x: isinstance(x, ReccableBlock), self.layers))
        assert (len(filter(lambda x: isinstance(x, RecurrentBlock) and not isinstance(x, ReccableBlock),
                           self.layers)) == 0)  # no non-reccable blocks allowed
        init_infos = []
        for recurrentlayer in recurrentlayers:
            if issequence(initstates):
                arg = initstates[:recurrentlayer.numstates]
                initstates = initstates[recurrentlayer.numstates:]
            else:
                arg = initstates
            initinfo = recurrentlayer.get_init_info(arg)
            init_infos.extend(initinfo)
        return init_infos

    def rec(self, x_t, *states):
        # apply each block on x_t to get next-level input, consume states in the process
        nextinp = x_t
        nextstates = []
        for block in self.layers:
            if isinstance(block, ReccableBlock):
                numstates = block.numstates
                recstates = states[:numstates]
                states = states[numstates:]
                rnuret = block.rec(nextinp, *recstates)
                nextstates.extend(rnuret[1:])
                nextinp = rnuret[0]
            elif isinstance(block, Block): # block is a function
                nextinp = block(nextinp)
        return [nextinp] + nextstates


class BiRNU(RecurrentBlock): # TODO: optimizer can't process this
    def __init__(self, fwd=None, rew=None, **kw):
        super(BiRNU, self).__init__(**kw)
        assert(isinstance(fwd, RNUBase) and isinstance(rew, RNUBase))
        self.fwd = fwd
        self.rew = rew
        assert(self.fwd.indim == self.rew.indim)

    @classmethod
    def fromrnu(cls, rnucls, *args, **kw):
        assert(issubclass(rnucls, RNUBase))
        kw["reverse"] = False
        fwd = rnucls(*args, **kw)
        kw["reverse"] = True
        rew = rnucls(*args, **kw)
        return cls(fwd=fwd, rew=rew)

    @property
    def numstates(self):
        return self.fwd.numstates + self.rew.numstates

    def innerapply(self, seq, mask=None, initstates=None):
        initstatesfwd = initstates[:self.fwd.numstates] if initstates is not None else initstates
        initstates = initstates[self.fwd.numstates:] if initstates is not None else initstates
        assert(initstates is None or len(initstates) == self.rew.numstates)
        initstatesrew = initstates
        fwdout, fwdstates = self.fwd.innerapply(seq, mask=mask, initstates=initstatesfwd)
        rewout, rewstates = self.rew.innerapply(seq, mask=mask, initstates=initstatesrew)
        # concatenate: fwdout, rewout: (batsize, seqlen, feats) ==> (batsize, seqlen, feats_fwd+feats_rew)
        out = T.concatenate([fwdout, rewout], axis=2)
        return out, fwdstates+rewstates


class SeqEncoder(AttentionConsumer, Block):
    '''
    Encodes a sequence of vectors into a vector, input dims and output dims specified by the RNU unit
    Returns multiple outputs, multiple states
    Builds for one output
    '''
    _withoutput = False
    _all_states = False
    _weighted = False
    _nomask = False
    _zeromask = False

    def __init__(self, embedder, *layers, **kw):
        super(SeqEncoder, self).__init__(**kw)
        self.embedder = embedder
        if len(layers) > 0:
            if len(layers) == 1:
                self.block = layers[0]
                assert(isinstance(self.block, RecurrentBlock))
            else:
                self.block = RecStack(*layers)
        else:
            self.block = None

    def apply(self, seq, weights=None, mask="auto"): # seq: (batsize, seqlen, dim), weights: (batsize, seqlen) OR (batsize, seqlen, seqlen*, dim) ==> reduce the innermost seqlen
        if self.embedder is not None:
            seqemb = self.embedder(seq)
        else:
            seqemb = seq
        if weights is None:
            weights = T.ones(seqemb.shape[:-1]) # (batsize, seqlen)
        else:
            self._weighted = True
        mask = self._generate_mask(mask, seq, seqemb)
        fullmask = mask * weights
        outputs, states = self.block.innerapply(seqemb, mask=fullmask)
        return self._get_apply_outputs(outputs, states, mask)

    def _generate_mask(self, maskinp, seq, seqemb): # seq: (batsize, seqlen, dim)
        if maskinp is None:     # generate default all-ones mask
            mask = T.ones(seqemb.shape[:2])
            self._nomask = True
        elif maskinp is "auto": # generate mask based on seq data and seqemb's shape
            axes = range(2, seq.ndim)       # mask must be 2D
            if "int" in seq.dtype:       # ==> indexes  # mask must be 2D
                mask = seq.sum(axis=axes) > 0      # 0 is TERMINUS
            else:
                mask = seq.norm(2, axis=axes) > 0
        else:
            mask = maskinp
        return mask     # (batsize, seqlen)

    def _get_apply_outputs(self, outputs, states, mask):
        if self._withoutput:
            ret = outputs       # (batsize, seqlen, dim) --> zero-fy according to mask
            if self._zeromask:
                fmask = T.tensordot(mask, T.ones((outputs.shape[2],)), 0)
                ret = ret * fmask
        else:
            ret = outputs[:, -1, :]
        return ret

    def _get_apply_outputs_old(self, outputs):
        output = outputs[0]
        if self._all_states:    # get the last values of each of the states
            states = self.block.get_states_from_outputs(outputs[1:])
            ret = [s[-1, :, :] for s in states]
        else:                   # get the last state of the final state
            ret = [output[:, -1, :]] #output is (batsize, innerdim)
        if self._withoutput:    # include stack output
            return ret + [output]
        else:
            if len(ret) == 1:   # if only topmost last state or only one stateful layer --> one output
                return ret[0]
            else:               # else: a list of outputs
                return ret

    def _build(self, *inps):
        res = self.wrapply(*inps)
        if issequence(res):
            output = res[0]
        else:
            output = res
        return output

    @property
    def all_states(self):       # TODO
        '''Call this switch to get the final states of all recurrent layers'''
        self._all_states = True
        return self

    @property
    def zeromask(self):
        self._zeromask = True
        return self

    @property
    def all_outputs(self):
        '''Call this switch to get the actual output of top layer as the last outputs'''
        self._withoutput = True
        return self


class SeqDecoder(Block):
    '''
    Decodes a sequence of symbols given context
    output: probabilities over symbol space: float: (batsize, seqlen, vocabsize)

    ! must pass in a recurrent block that takes two arguments: context_t and x_t
    ! first input is TERMINUS ==> suggest to set TERMINUS(0) embedding to all zeroes (in s2vf)
    ! first layer must be an embedder or IdxToOneHot, otherwise, an IdxToOneHot is created automatically based on given dim
    '''
    def __init__(self, layers, softmaxoutblock=None, innerdim=None, attention=None, inconcat=False, outconcat=False, **kw): # limit says at most how many is produced
        super(SeqDecoder, self).__init__(**kw)
        self.embedder = layers[0]
        self.block = RecStack(*layers[1:])
        self.outdim = innerdim
        self.inconcat = inconcat
        self.outconcat = outconcat
        self.attention = attention
        self._mask = False
        self._attention = None
        assert(isinstance(self.block, ReccableBlock))
        if softmaxoutblock is None: # default softmax out block
            sm = Softmax()
            self.lin = MatDot(indim=self.outdim, dim=self.embedder.indim)
            self.softmaxoutblock = asblock(lambda x: sm(self.lin(x)))
        else:
            self.softmaxoutblock = softmaxoutblock

    def apply(self, context, seq, context_0=None, initstates=None, mask=None, **kw):    # context: (batsize, enc.innerdim), seq: idxs-(batsize, seqlen)
        if initstates is None:
            initstates = seq.shape[0]
        init_info = self.get_init_info(context, context_0, initstates)  # sets init states to provided ones
        outputs, _ = T.scan(fn=self.rec,
                            sequences=seq.dimswap(1, 0),
                            outputs_info=[None] + init_info)
        ret = outputs[0].dimswap(1, 0)     # returns probabilities of symbols --> (batsize, seqlen, vocabsize)
        ret = self.applymask(ret, mask)
        return ret

    @classmethod
    def applymask(cls, xseq, maskseq):
        if maskseq is None:
            return xseq
        else:
            mask = T.tensordot(maskseq, T.ones((xseq.shape[2],)), 0)  # f32^(batsize, seqlen, outdim) -- maskseq stacked
            masker = T.concatenate(
                [T.ones((xseq.shape[0], xseq.shape[1], 1)), T.zeros((xseq.shape[0], xseq.shape[1], xseq.shape[2] - 1))],
                axis=2)  # f32^(batsize, seqlen, outdim) -- gives 100% prob to output 0
            ret = xseq * mask + masker * (1.0 - mask)
            return ret

    def get_init_info(self, context, context_0, initstates):
        ret = self.block.get_init_info(initstates)
        context_0 = self._get_ctx_t0(context, context_0)
        return [context, context_0, 0] + ret

    def _get_ctx_t0(self, ctx, ctx_0=None):
        if ctx_0 is None:
            if ctx.d.ndim == 2:     # static context
                ctx_0 = ctx
            elif ctx.d.ndim > 2:   # dynamic context (batsize, inseqlen, inencdim)
                assert(self.attention is not None)      # 3D context only processable with attention (dynamic context)
                w_0 = T.ones((ctx.shape[0], ctx.shape[1]), dtype=T.config.floatX) / ctx.shape[1].astype(T.config.floatX)    #  ==> make uniform weights (??)
                ctx_0 = self.attention.attentionconsumer(ctx, w_0)
                '''else:
                    ctx_0 = ctx[:, -1, :]       # take the last context'''
            else:
                print "sum ting wong in SeqDecoder _get_ctx_t0()"
        return ctx_0

    def rec(self, x_t, ctx, ctx_tm1, t, *states_tm1):  # x_t: (batsize), context: (batsize, enc.innerdim)
        i_t = self.embedder(x_t)                             # i_t: (batsize, embdim)
        j_t = self._get_j_t(i_t, ctx_tm1)
        rnuret = self.block.rec(j_t, *states_tm1)     # list of matrices (batsize, **somedims**)
        ret = rnuret
        t = t + 1
        h_t = ret[0]
        states_t = ret[1:]
        ctx_t = self._gen_context(ctx, h_t)
        g_t = self._get_g_t(h_t, ctx_t)
        y_t = self.softmaxoutblock(g_t)
        return [y_t, ctx, ctx_t, t] + states_t #, {}, T.until( (i > 1) * T.eq(mask.norm(1), 0) )

    def _get_j_t(self, i_t, ctx_tm1):
        return T.concatenate([i_t, ctx_tm1], axis=1) if self.inconcat else i_t

    def _get_g_t(self, h_t, ctx_t):
        return T.concatenate([h_t, ctx_t], axis=1) if self.outconcat else h_t

    def _gen_context(self, multicontext, criterion):
        return self.attention(criterion, multicontext) if self.attention is not None else multicontext

# ----------------------------------------------------------------------------------------------------------------------

# TODO: travis error messages about theano optimization and shapes only involve things below
class SimpleEncoderDecoder(Block):  # gets two sequences of indexes for training
    def __init__(self, innerdim=50, input_vocsize=100, output_vocsize=100, **kw):
        super(SimpleEncoderDecoder, self).__init__(**kw)
        input_embedder = IdxToOneHot(input_vocsize)
        output_embedder = IdxToOneHot(output_vocsize)
        encrec = GRU(dim=input_vocsize, innerdim=innerdim)
        decrecrnu = GRU(dim=output_vocsize, innerdim=innerdim)
        self.enc = SeqEncoder(input_embedder, encrec)
        self.dec = SeqDecoder([output_embedder, decrecrnu], outconcat=True, innerdim=innerdim+innerdim)

    def apply(self, inpseq, outseq):
        enco = self.enc(inpseq)
        deco = self.dec(enco, outseq)
        return deco


class RNNAutoEncoder(Block):    # tries to decode original sequence
    def __init__(self, vocsize=25, encdim=200, innerdim=200, seqlen=50, **kw):
        super(RNNAutoEncoder, self).__init__(**kw)
        self.seqlen = seqlen
        self.encoder = SeqEncoder(
            IdxToOneHot(vocsize=vocsize),
            GRU(dim=vocsize, innerdim=encdim))
        self.decoder = SeqDecoder([IdxToOneHot(vocsize), GRU(dim=vocsize+encdim, innerdim=innerdim)],
                                  innerdim=innerdim,
                                  inconcat=True
                                  )

    def apply(self, inpseq, outseq):    # inpseq: (batsize, seqlen), indexes
        enc = self.encoder(inpseq)
        dec = self.decoder(enc, outseq)
        return dec


class RewAttRNNEncDecoder(Block):
    '''
    Take the input index sequence as-is, transform to one-hot, feed to gate AttentionGenerator, encode with weighted SeqEncoder,
    put everything as attention inside SeqDecoder
    '''
    def __init__(self, vocsize=25, outvocsize=20, encdim=200, innerdim=200, attdim=50, **kw):
        super(RewAttRNNEncDecoder, self).__init__(**kw)
        self.emb = IdxToOneHot(vocsize)
        attgen = LinearGateAttentionGenerator(indim=innerdim+vocsize, innerdim=attdim)
        attcon = SeqEncoder(None,
            GRU(dim=vocsize, innerdim=encdim))
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize+encdim, innerdim=innerdim)],
                              attention=Attention(attgen, attcon),
                              inconcat=True,
                              innerdim=innerdim)

    def apply(self, inpseq, outseq):    # inpseq: indexes~(batsize, seqlen)
        inp = self.emb(inpseq)  # inp:    floats~(batsize, seqlen, vocsize)
        return self.dec(inp, outseq)


class FwdAttRNNEncDecoder(Block):
    '''
    Take the input index sequence as-is, transform to one-hot, feed to gate AttentionGenerator, encode with weighted SeqEncoder,
    put everything as attention inside SeqDecoder
    '''
    def __init__(self, vocsize=25, outvocsize=20, encdim=200, innerdim=200, attdim=50, **kw):
        super(FwdAttRNNEncDecoder, self).__init__(**kw)
        self.emb = IdxToOneHot(vocsize)
        attgen = LinearGateAttentionGenerator(indim=innerdim+vocsize, innerdim=attdim)
        attcon = SeqEncoder(None,
            GRU(dim=vocsize, innerdim=encdim))
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize, innerdim=innerdim)],
                              outconcat=True,
                              attention=Attention(attgen, attcon),
                              innerdim=innerdim+encdim)

    def apply(self, inpseq, outseq):    # inpseq: indexes~(batsize, seqlen)
        inp = self.emb(inpseq)  # inp:    floats~(batsize, seqlen, vocsize)
        return self.dec(inp, outseq)


class RewAttSumDecoder(Block):
    def __init__(self, vocsize=25, outvocsize=25, encdim=200, innerdim=200, attdim=50, **kw):
        super(RewAttSumDecoder, self).__init__(**kw)
        self.rnn = SeqEncoder(IdxToOneHot(vocsize), GRU(dim=vocsize, innerdim=encdim)).all_outputs
        attgen = LinearGateAttentionGenerator(indim=innerdim+encdim, innerdim=attdim)
        attcon = WeightedSumAttCon()
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize+encdim, innerdim=innerdim)],
                              inconcat=True,
                              attention=Attention(attgen, attcon),
                              innerdim=innerdim)

    def apply(self, inpseq, outseq):        # inpseq: indexes~(batsize, seqlen)
        rnnout = self.rnn(inpseq)   # (batsize, seqlen, encdim)
        return self.dec(rnnout, outseq)     # (batsize, seqlen, vocsize)


class FwdAttSumDecoder(Block):
    def __init__(self, vocsize=25, outvocsize=25, encdim=300, innerdim=200, attdim=50, **kw):
        super(FwdAttSumDecoder, self).__init__(**kw)
        self.rnn = SeqEncoder(IdxToOneHot(vocsize), GRU(dim=vocsize, innerdim=encdim)).all_outputs
        attgen = LinearGateAttentionGenerator(indim=innerdim+encdim, innerdim=attdim)
        attcon = WeightedSumAttCon()
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize, innerdim=innerdim)],
                              outconcat=True,
                              attention=Attention(attgen, attcon),
                              innerdim=innerdim+encdim
                              )

    def apply(self, inpseq, outseq):
        rnnout = self.rnn(inpseq)
        return self.dec(rnnout, outseq)


class BiFwdAttSumDecoder(Block):
    def __init__(self, vocsize=25, outvocsize=25, encdim=300, innerdim=200, attdim=50, **kw):
        super(BiFwdAttSumDecoder, self).__init__(**kw)
        self.rnn = SeqEncoder(IdxToOneHot(vocsize),
                                  BiRNU.fromrnu(GRU, dim=vocsize, innerdim=encdim)).all_outputs
        attgen = LinearGateAttentionGenerator(indim=innerdim+encdim*2, innerdim=attdim)
        attcon = WeightedSumAttCon()
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize, innerdim=innerdim)],
                              outconcat=True,
                              attention=Attention(attgen, attcon),
                              innerdim=innerdim+encdim*2
                              )

    def apply(self, inpseq, outseq):
        rnnout = self.rnn(inpseq)
        return self.dec(rnnout, outseq)


class BiRewAttSumDecoder(Block):
    def __init__(self, vocsize=25, outvocsize=25, encdim=300, innerdim=200, attdim=50, **kw):
        super(BiRewAttSumDecoder, self).__init__(**kw)
        self.rnn = SeqEncoder(IdxToOneHot(vocsize),
                                  BiRNU.fromrnu(GRU, dim=vocsize, innerdim=encdim)).all_outputs
        attgen = LinearGateAttentionGenerator(indim=innerdim+encdim*2, innerdim=attdim)
        attcon = WeightedSumAttCon()
        self.dec = SeqDecoder([IdxToOneHot(outvocsize), GRU(dim=outvocsize+encdim*2, innerdim=innerdim)],
                              inconcat=True,
                              attention=Attention(attgen, attcon),
                              innerdim=innerdim
                              )

    def apply(self, inpseq, outseq):
        rnnout = self.rnn(inpseq)
        return self.dec(rnnout, outseq)










class SeqTransducer(Block):
    def __init__(self, embedder, *layers, **kw):
        """ layers must have an embedding layers first, final softmax layer is added automatically"""
        assert("smodim" in kw and "outdim" in kw)
        self.embedder = embedder
        smodim = kw["smodim"]
        outdim = kw["outdim"]
        del kw["smodim"]; del kw["outdim"]
        super(SeqTransducer, self).__init__(**kw)
        self.block = RecStack(*(layers + (Lin(indim=smodim, dim=outdim), Softmax())))

    def apply(self, inpseq, maskseq=None):    # inpseq: idx^(batsize, seqlen), maskseq: f32^(batsize, seqlen)
        embseq = self.embedder(inpseq)
        res = self.block(embseq, mask=maskseq)            # f32^(batsize, seqlen, outdim)
        ret = self.applymask(res, maskseq=maskseq)
        return ret

    @classmethod
    def applymask(cls, xseq, maskseq=None):
        if maskseq is None:
            ret = xseq
        else:
            mask = T.tensordot(maskseq, T.ones((xseq.shape[2],)), 0)  # f32^(batsize, seqlen, outdim) -- maskseq stacked
            masker = T.concatenate([T.ones((xseq.shape[0], xseq.shape[1], 1)), T.zeros((xseq.shape[0], xseq.shape[1], xseq.shape[2] - 1))], axis=2)  # f32^(batsize, seqlen, outdim) -- gives 100% prob to output 0
            ret = xseq * mask + masker * (1.0 - mask)
        return ret


class SimpleSeqTransducer(SeqTransducer):
    def __init__(self, indim=400, embdim=50, innerdim=100, outdim=50, **kw):
        self.emb = VectorEmbed(indim=indim, dim=embdim)
        if not issequence(innerdim):
            innerdim = [innerdim]
        innerdim = [embdim] + innerdim
        self.rnn = self.getrnnfrominnerdim(innerdim)
        super(SimpleSeqTransducer, self).__init__(self.emb, *self.rnn, smodim=innerdim[-1], outdim=outdim, **kw)

    @classmethod
    def getrnnfrominnerdim(self, innerdim, rnu=GRU):
        rnn = []
        assert(len(innerdim) >= 2)
        i = 1
        while i < len(innerdim):
            rnn.append(rnu(dim=innerdim[i-1], innerdim=innerdim[i]))
            i += 1
        return rnn


class SeqTransDec(Block):
    def __init__(self, *layers, **kw):
        """ first two layers must be embedding layers. Final softmax is added automatically"""
        assert("smodim" in kw and "outdim" in kw)
        smodim = kw["smodim"]
        outdim = kw["outdim"]
        del kw["smodim"]; del kw["outdim"]
        super(SeqTransDec, self).__init__(**kw)
        self.inpemb = layers[0]
        self.outemb = layers[1]
        self.block = RecStack(*(layers[2:] + (Lin(indim=smodim, dim=outdim), Softmax())))

    def apply(self, inpseq, outseq, maskseq=None):
        # embed with the two embedding layers
        emb = self._get_emb(inpseq, outseq)
        res = self.block(emb)
        ret = SeqTransducer.applymask(res, maskseq=maskseq)
        return ret

    def _get_emb(self, inpseq, outseq):
        iemb = self.inpemb(inpseq)     # (batsize, seqlen, inpembdim)
        oemb = self.outemb(outseq)     # (batsize, seqlen, outembdim)
        emb = T.concatenate([iemb, oemb], axis=iemb.ndim-1)                       # (batsize, seqlen, inpembdim+outembdim)
        return emb

    def rec(self, inpa, inpb, *states):
        emb = self._get_emb(inpa, inpb)
        return self.block.rec(emb, *states)

    def get_init_info(self, initstates):
        return self.block.get_init_info(initstates)


class SimpleSeqTransDec(SeqTransDec):
    def __init__(self, indim=400, outdim=50, inpembdim=50, outembdim=50, innerdim=100, **kw):
        self.inpemb = VectorEmbed(indim=indim, dim=inpembdim)
        self.outemb = VectorEmbed(indim=outdim, dim=outembdim)
        self.rnn = []
        if not issequence(innerdim):
            innerdim = [innerdim]
        innerdim = [inpembdim+outembdim] + innerdim
        self.rnn = SimpleSeqTransducer.getrnnfrominnerdim(innerdim)
        super(SimpleSeqTransDec, self).__init__(self.inpemb, self.outemb, *self.rnn, smodim=innerdim[-1], outdim=outdim, **kw)