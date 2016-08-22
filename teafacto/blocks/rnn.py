from teafacto.blocks.attention import WeightedSumAttCon, Attention, AttentionConsumer, LinearGateAttentionGenerator
from teafacto.blocks.basic import IdxToOneHot, Softmax, MatDot
from teafacto.blocks.rnu import GRU, ReccableBlock, RecurrentBlock, RNUBase
from teafacto.core.base import Block, tensorops as T, asblock
from teafacto.util import issequence
from enum import Enum


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
        return reduce(lambda x, y: x + y, [x.numstates for x in self.layers if isinstance(x, RecurrentBlock)], 0)

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
                final, seq, layerstates = layer.innerapply(seq, mask=mask, initstates=layerinpstates)
                states.extend(layerstates)
            elif isinstance(layer, Block):
                seq = self.recurnonreclayer(seq, layer)
                final = seq[:, -1, :]
            else:
                raise Exception("can not apply this layer: " + str(layer) + " in RecStack")
        return final, seq, states           # full history of final output and all states (ordered from bottom layer to top)

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
        if issequence(initstates):  # fill up init state args so that layers for which no init state is specified get default arguments that lets them specify a default init state
                                    # if is a sequence, expecting a value, not batsize
            if len(initstates) < self.numstates:    # top layers are being given the given init states, bottoms make their own default
                initstates = [initstates.shape[0]] * (self.numstates - len(initstates)) + initstates
        else:   # expecting a batsize as initstate arg
            initstates = [initstates] * self.numstates
        init_infos = []
        for recurrentlayer in recurrentlayers:  # from bottom layers to top
            arg = initstates[:recurrentlayer.numstates]
            initstates = initstates[recurrentlayer.numstates:]
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
        fwdfinal, fwdout, fwdstates = self.fwd.innerapply(seq, mask=mask, initstates=initstatesfwd)   # (batsize, seqlen, innerdim)
        rewfinal, rewout, rewstates = self.rew.innerapply(seq, mask=mask, initstates=initstatesrew) # TODO: reverse?
        # concatenate: fwdout, rewout: (batsize, seqlen, feats) ==> (batsize, seqlen, feats_fwd+feats_rew)
        finalout = T.concatenate([fwdfinal, rewfinal], axis=1)
        out = T.concatenate([fwdout, rewout.reverse(1)], axis=2)
        return finalout, out, fwdstates+rewstates



class MaskMode(Enum):
    NONE = 0
    AUTO = 1
    AUTO_FORCE = 2

class MaskSetMode(Enum):
    NONE = 0
    ZERO = 1
    MASKID = 2


class MaskConfig(object):
    def __init__(self, maskmode=MaskMode.NONE, maskid=0, maskset=MaskSetMode.NONE):
        self.maskmode = maskmode
        self.maskid = maskid
        self.maskset = maskset

    def option(self, o):
        if isinstance(o, MaskSetMode):
            self.maskset = o
        elif isinstance(o, MaskMode):
            self.maskmode = o
        elif isinstance(o, int):
            self.maskid = o
        else:
            raise NotImplementedError("unrecognized mask configuration option")


class SeqEncoder(AttentionConsumer, Block):
    '''
    Encodes a sequence of vectors into a vector, input dims and output dims specified by the RNU unit
    Returns multiple outputs, multiple states
    Builds for one output
    '''

    def __init__(self, embedder, *layers, **kw):
        super(SeqEncoder, self).__init__(**kw)
        self._return = {"enc"}
        self._nomask = False
        self._maskconfig = kw["maskcfg"] if "maskcfg" in kw else MaskConfig(MaskMode.AUTO, 0, MaskSetMode.NONE)
        self.embedder = embedder
        if len(layers) > 0:
            if len(layers) == 1:
                self.block = layers[0]
                assert(isinstance(self.block, RecurrentBlock))
            else:
                self.block = RecStack(*layers)
        else:
            self.block = None

    def apply(self, seq, weights=None, mask=None): # seq: (batsize, seqlen, dim), weights: (batsize, seqlen) OR (batsize, seqlen, seqlen*, dim) ==> reduce the innermost seqlen
        # embed
        if self.embedder is not None:
            seqemb = self.embedder(seq)
        else:
            seqemb = seq
        # compute full mask
        if self._maskconfig.maskmode == MaskMode.AUTO_FORCE or \
                (mask is None and self._maskconfig.maskmode == MaskMode.AUTO) or \
                mask == "auto":
            mask = self._autogenerate_mask(seq, seqemb)

        fullmask = None
        if mask is not None:
            fullmask = mask
        if weights is not None:
            fullmask = weights if fullmask is None else weights * fullmask
        final, outputs, states = self.block.innerapply(seqemb, mask=fullmask)
        return self._get_apply_outputs(final, outputs, states, mask)

    def _autogenerate_mask(self, seq, seqemb):
        assert(seqemb.ndim == 3)
        print "automasking in SeqEncoder (%s)" % __file__
        axes = range(2, seq.ndim)       # mask must be 2D
        if "int" in seq.dtype:       # ==> indexes  # mask must be 2D
            if seq.ndim < 2:
                raise AttributeError("CAN NOT GENERATE MASK FOR NON-SEQUENCE")
            elif seq.ndim == 2:
                seqformask = seq
            else:
                print "generating default mask for non-standard seq shape (SeqEncoder, %s)" % __file__
                seqformask = seq[(slice(None, None, None),) * 2 + (0,) * (seq.ndim-2)]
                #if self._maskconfig.maskid != 0:
                #    raise AttributeError("CAN NOT CREATE MASK USING CUSTOM MASKID %d BECAUSE OF NON-STANDARD SEQ (%d dims, %s)" % (self._maskconfig.maskid, seq.ndim, str(seq.dtype)))
                #mask = T.gt(seq.sum(axis=axes), 0)      # 0 is TERMINUS
            assert(seqformask.ndim == 2)
            mask = T.neq(seqformask, self._maskconfig.maskid)
        else:
            #TODO raise AttributeError("CAN NOT GENERATE MASK FOR NON-INT SEQ")
            mask = T.gt(seq.norm(2, axis=axes), 0)
        return mask

    def _get_apply_outputs(self, final, outputs, states, mask):
        ret = []
        if "enc" in self._return:       # final states of topmost layer
            ret.append(final)
        if "all" in self._return:       # states (over all time) of topmost layer
            rete = outputs       # (batsize, seqlen, dim) --> zero-fy according to mask
            if self._maskconfig.maskset == MaskSetMode.ZERO and mask is not None:
                fmask = T.tensordot(mask, T.ones((outputs.shape[2],)), 0)
                rete = rete * fmask
            ret.append(rete)
        if "states" in self._return:    # final states (over all layers)???
            pass # TODO: do we need to support this?
        if "mask" in self._return:
            ret.append(mask)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def _get_apply_outputs_old(self, outputs):
        output = outputs[0]
        if self._all_states:    # get the last values of each of the states
            states = self.block.get_states_from_outputs(outputs[1:])
            ret = [s[-1, :, :] for s in states]
        else:                   # get the last state of the final state
            ret = [output[:, -1, :]] #output is (batsize, innerdim)
        if self._alloutput:    # include stack output
            return ret + [output]
        else:
            if len(ret) == 1:   # if only topmost last state or only one stateful layer --> one output
                return ret[0]
            else:               # else: a list of outputs
                return ret

    ### FLUENT MASK SETTINGS
    def mask(self, maskcfg):
        assert(isinstance(maskcfg, MaskConfig))
        self._maskconfig = maskcfg
        return self

    def maskoptions(self, *opt):
        for o in opt:
            self.maskoption(o)
        return self

    def maskoption(self, maskoption):
        self._maskconfig.option(maskoption)
        return self

    ### FLUENT OUTPUT SETTINGS
    @property
    def reset_return(self):
        self._return = set()
        return self

    @property
    def with_states(self):       # TODO
        '''Call this switch to get the final states of all recurrent layers'''
        self._return.add("states")
        return self

    @property
    def all_outputs(self):
        '''Call this switch to get the actual output of top layer as the last outputs'''
        self._return = {"all"}
        return self

    @property
    def with_outputs(self):
        self._return.add("all")
        return self

    @property
    def with_mask(self):
        ''' Calling this switch also returns the mask on original idx input sequence'''
        self._return.add("mask")
        return self

    def setreturn(self, *args):
        self._return = args
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
        elif softmaxoutblock is False:
            self.softmaxoutblock = asblock(lambda x: x)
        else:
            self.softmaxoutblock = softmaxoutblock

    @property
    def numstates(self):
        return self.block.numstates

    def apply(self, context, seq, context_0=None, initstates=None, mask=None, encmask=None, **kw):    # context: (batsize, enc.innerdim), seq: idxs-(batsize, seqlen)
        if initstates is None:
            initstates = seq.shape[0]
        elif issequence(initstates):
            if len(initstates) < self.numstates:    # fill up with batsizes for lower layers
                initstates = [seq.shape[0]]*(self.numstates - len(initstates)) + initstates
        init_info, ctx = self.get_init_info(context, context_0, initstates, encmask=encmask)  # sets init states to provided ones
        outputs, _ = T.scan(fn=self.rec,
                            sequences=seq.dimswap(1, 0),
                            outputs_info=[None] + init_info,
                            non_sequences=ctx)
        ret = outputs[0].dimswap(1, 0)     # returns probabilities of symbols --> (batsize, seqlen, vocabsize)
        if mask == "auto":
            mask = (seq > 0).astype("int32")
        ret = self.applymask(ret, mask)
        return ret

    @classmethod
    def applymask(cls, xseq, maskseq):
        if maskseq is None:
            return xseq
        else:
            mask = T.tensordot(maskseq, T.ones((xseq.shape[2],)), 0)  # f32^(batsize, seqlen, outdim) -- maskseq stacked
            masker = T.concatenate(
                [T.ones((xseq.shape[0], xseq.shape[1], 1)),
                 T.zeros((xseq.shape[0], xseq.shape[1], xseq.shape[2] - 1))],
                axis=2)  # f32^(batsize, seqlen, outdim) -- gives 100% prob to output 0
            ret = xseq * mask + masker * (1.0 - mask)
            return ret

    def get_init_info(self, context, context_0, initstates, encmask=None):
        ret = self.block.get_init_info(initstates)
        context_0 = self._get_ctx_t0(context, context_0)
        return [context_0, 0] + ret, [encmask, context]

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

    def rec(self, x_t, ctx_tm1, t, *args):  # x_t: (batsize), context: (batsize, enc.innerdim)
        states_tm1 = args[:-1]
        ctx = args[-1]
        encmask = args[-2]
        i_t = self.embedder(x_t)                             # i_t: (batsize, embdim)
        j_t = self._get_j_t(i_t, ctx_tm1)
        rnuret = self.block.rec(j_t, *states_tm1)     # list of matrices (batsize, **somedims**)
        ret = rnuret
        t = t + 1
        h_t = ret[0]
        states_t = ret[1:]
        ctx_t = self._gen_context(ctx, h_t, encmask)
        g_t = self._get_g_t(h_t, ctx_t)
        y_t = self.softmaxoutblock(g_t)
        return [y_t, ctx_t, t] + states_t #, {}, T.until( (i > 1) * T.eq(mask.norm(1), 0) )

    def _get_j_t(self, i_t, ctx_tm1):
        return T.concatenate([i_t, ctx_tm1], axis=1) if self.inconcat else i_t

    def _get_g_t(self, h_t, ctx_t):
        return T.concatenate([h_t, ctx_t], axis=1) if self.outconcat else h_t

    def _gen_context(self, multicontext, criterion, encmask):
        return self.attention(criterion, multicontext, mask=encmask) if self.attention is not None else multicontext

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


class MakeRNU(object):
    ''' generates a list of RNU's'''
    @staticmethod
    def make(initdim, specs, rnu=GRU, bidir=False):
        if not issequence(specs):
            specs = [specs]
        rnns = []
        prevdim = initdim
        for spec in specs:
            fspec = {"dim": None, "bidir": bidir, "rnu": rnu}
            if isinstance(spec, int):
                fspec["dim"] = spec
            elif isinstance(spec, dict):
                assert(hasattr(spec, "dim")
                       and
                       set(spec.keys()).union(set(fspec.keys()))
                            == set(fspec.keys()))
                fspec.update(spec)
            if fspec["bidir"] == True:
                rnn = BiRNU.fromrnu(fspec["rnu"], dim=prevdim, innerdim=fspec["dim"])
                prevdim = fspec["dim"] * 2
            else:
                rnn = fspec["rnu"](dim=prevdim, innerdim=fspec["dim"])
                prevdim = fspec["dim"]
            rnns.append(rnn)
        return rnns, prevdim