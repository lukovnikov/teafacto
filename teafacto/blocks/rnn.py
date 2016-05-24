from teafacto.blocks.basic import MatDot as Lin, Softmax, VectorEmbed
from teafacto.blocks.rnu import GRU
from teafacto.core.base import Block, tensorops, tensorops, tensorops, tensorops, tensorops, tensorops
from teafacto.blocks.rnu import GRU, ReccableBlock, RecurrentBlock, RNUBase
from teafacto.core.base import Block, tensorops as T, asblock
from teafacto.blocks.basic import IdxToOneHot, VectorEmbed, Embedder, Softmax, MatDot, ConcatBlock
from teafacto.blocks.attention import WeightedSumAttCon, LinearSumAttentionGenerator, Attention, AttentionConsumer, LinearGateAttentionGenerator
from teafacto.users.modelusers import RecUsable
from teafacto.util import issequence
import inspect
from IPython import embed


class ReccableStack(ReccableBlock):
    def __init__(self, *args, **kw): # layer can be a layer or function
        super(ReccableStack, self).__init__(**kw)
        self.layers = args

    def __getitem__(self, idx):
        return self.layers[idx]

    def get_states_from_outputs(self, outputs):
        # outputs are ordered from topmost recurrent layer first ==> split and delegate
        states = []
        for recurrentlayer in filter(lambda x: isinstance(x, ReccableBlock), self.layers): # from bottom -> eat from behind; insert to the front
            numstates = len(inspect.getargspec(recurrentlayer.rec).args) - 2
            layerstates = recurrentlayer.get_states_from_outputs(outputs[-numstates:]) # might be more than one
            i = 0
            for layerstate in layerstates:
                states.insert(i, layerstate)
                i += 1
            outputs = outputs[:-numstates]
        assert(len(outputs) == 0)
        return states

    def do_get_init_info(self, initstates):    # if initstates is not a list, it must be batsize
        recurrentlayers = list(filter(lambda x: isinstance(x, ReccableBlock), self.layers))
        recurrentlayers.reverse()
        init_infos = []
        for recurrentlayer in recurrentlayers:
            initinfo, initstates = recurrentlayer.do_get_init_info(initstates)
            init_infos.extend(initinfo)
        return init_infos, initstates   # layerwise in reverse

    def rec(self, x_t, *states):
        # apply each block on x_t to get next-level input, consume states in the process
        nextinp = x_t
        nextstates = []
        for block in self.layers:
            if isinstance(block, ReccableBlock):
                numstates = len(inspect.getargspec(block.rec).args) - 2
                # eat from behind
                recstates = states[-numstates:]
                states = states[:-numstates]
                rnuret = block.rec(nextinp, *recstates)
                # insert from behind
                i = 0
                for nextstate in rnuret[1:]:
                    nextstates.insert(i, nextstate)
                    i += 1
                nextinp = rnuret[0]
            elif isinstance(block, Block): # block is a function
                nextinp = block(nextinp)
        return [nextinp] + nextstates

    def recappl(self, inp, *states):
        return self.rec(inp, *states)

    def apply(self, se, initstates=None):
        seq = se.dimswap(1, 0)
        initstatearg = initstates if initstates is not None else seq.shape[1]
        outputs, _ = T.scan(fn=self.rec,
                            sequences=seq,
                            outputs_info=[None]+self.get_init_info(initstatearg))
        output = outputs[0]
        return output.dimswap(1, 0)


class RecurrentStack(Block):       # TODO: setting init states of contained recurrent blocks
    def __init__(self, *layers, **kw):
        super(RecurrentStack, self).__init__(**kw)
        self.layers = layers

    def apply(self, seq):   # layer-wise processing of input sequence
        acc = seq
        for layer in self.layers:
            if isinstance(layer, RecurrentBlock):
                acc = layer(acc)
            elif isinstance(layer, Block): # non-recurrent ==> recur
                acc = self.recurnonreclayer(acc, layer)
            else:
                raise Exception("can not apply this layer: " + str(layer))
        return acc

    def recappl_init(self, ist):
        return self.get_init_info(ist)

    def get_init_info(self, initstates):
        recurrentlayers = list(filter(lambda x: isinstance(x, ReccableBlock), self.layers))
        assert(len(filter(lambda x: isinstance(x, RecurrentBlock) and not isinstance(x, ReccableBlock), self.layers)) == 0)       # no non-reccable blocks allowed
        init_infos = []
        for recurrentlayer in recurrentlayers:
            initinfo, initstates = recurrentlayer.do_get_init_info(initstates)
            init_infos.extend(initinfo)
        return init_infos, initstates   # layerwise in reverse


    def recappl(self, inps, states):       # what happens in one iteration ==> inside the scan #TODO: REMOVE
        # first inp is a var or a tuple of vars, after that follows layer-wise state vars
        # each block gives only one output or a tuple of outputs
        heads = []
        tail = states
        for layer in self.layers:
            if isinstance(layer, ReccableBlock):
                inps, head, tail = layer.recappl(inps, tail)   # flattened
                heads.extend(head)
            elif isinstance(layer, RecurrentBlock):
                raise AssertionError("no non-reccable recurrent blocks allowed")
            elif isinstance(layer, Block):
                inps = layer(*inps)
            if not issequence(inps):
                inps = [inps]
        return inps, heads, tail

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




class ReccableBlockParameterized(object):       # superclass for classes that take a reccable block as init param
    def __init__(self, *layers, **kw):
        self._reverse = False
        if "reverse" in kw:
            self._reverse = kw["reverse"]
            del kw["reverse"]
        super(ReccableBlockParameterized, self).__init__(**kw)
        if len(layers) > 0:
            if len(layers) == 1:
                self.block = layers[0]
                assert(isinstance(self.block, ReccableBlock))
            else:
                self.block = ReccableStack(*layers)
        else:
            self.block = None


'''class AttentionParameterized(object):
    def __add__(self, other):
        assert(self.block is None)  # this block should not be parameterized already in order to parameterize it
        self.attach(other)
        return self

    def attach(self, other):
        if isinstance(other, RecurrentBlock):
            self.block = other
            self.onAttach()
        return self

    def onAttach(self):
        raise NotImplementedError("use subclass")
'''

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

    def apply(self, seq, init_states=None):
        fwdout = self.fwd(seq)
        rewout = self.rew(seq)
        # concatenate: fwdout, rewout: (batsize, seqlen, feats) ==> (batsize, seqlen, feats_fwd+feats_rew)
        out = T.concatenate([fwdout, rewout], axis=2)
        return out


class SeqEncoder(ReccableBlockParameterized, AttentionConsumer, Block):
    '''
    Encodes a sequence of vectors into a vector, input dims and output dims specified by the RNU unit
    Returns multiple outputs, multiple states
    Builds for one output
    '''
    _withoutput = False
    _all_states = False
    _weighted = False
    _nomask = False

    @property
    def nomask(self):
        self._nomask = True

    def apply(self, seq, weights=None): # seq: (batsize, seqlen, dim), weights: (batsize, seqlen)
        inp = seq.dimswap(1, 0)         # inp: (seqlen, batsize, dim)
        if weights is None:
            w = T.ones((inp.shape[0], inp.shape[1])) # (seqlen, batsize)
        else:
            self._weighted = True
            w = weights.dimswap(1, 0)
        outputs, _ = T.scan(fn=self.recwrap,
                            sequences=[inp, w],
                            outputs_info=[None]+self.block.get_init_info(seq.shape[0]),
                            go_backwards=self._reverse)
        return self._get_apply_outputs(outputs)

    def _get_apply_outputs(self, outputs):
        output = outputs[0]
        if self._all_states:    # get the last values of each of the states
            states = self.block.get_states_from_outputs(outputs[1:])
            ret = [s[-1, :, :] for s in states]
        else:                   # get the last state of the final state
            ret = [output[-1, :, :]] #output is (batsize, innerdim)
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

    def recwrap(self, x_t, w_t, *args): # x_t: (batsize, dim)      if input is all zeros, just return previous state
        if not self._nomask:
            if x_t.ndim == 1:       # ==> indexes
                mask = x_t > 0      # 0 is TERMINUS
            else:
                mask = x_t.norm(2, axis=1) > 0 # mask: (batsize, )
        rnuret = self.block.rec(x_t, *args) # list of matrices (batsize, **somedims**)
        if self._weighted:
            rnuret = map(lambda (origarg, rnuretarg): (origarg.T * (1 - w_t) + rnuretarg.T * w_t).T, zip([args[0]] + list(args), rnuret))
        if not self._nomask:
            ret = map(lambda (origarg, rnuretarg): (origarg.T * (1 - mask) + rnuretarg.T * mask).T, zip([args[0]] + list(args), rnuret)) # TODO mask breaks multi-layered encoders (order is reversed)
        else:
            ret = rnuret
        #ret = rnuret
        return ret

    @property
    def all_states(self):
        '''Call this switch to get the final states of all recurrent layers'''
        self._all_states = True
        return self

    @property
    def with_outputs(self):
        '''Call this switch to get the actual output of top layer as the last outputs'''
        self._withoutput = True
        return self


class SeqDecoder(ReccableBlockParameterized, Block):        # TODO: make recappl-able
    '''
    Decodes a sequence of symbols given context
    output: probabilities over symbol space: float: (batsize, seqlen, vocabsize)

    ! must pass in a recurrent block that takes two arguments: context_t and x_t
    ! first input is TERMINUS ==> suggest to set TERMINUS(0) embedding to all zeroes (in s2vf)
    ! first layer must be an embedder or IdxToOneHot, otherwise, an IdxToOneHot is created automatically based on given dim
    '''
    def __init__(self, layers, softmaxoutblock=None, innerdim=None, attention=None, inconcat=False, outconcat=False, **kw): # limit says at most how many is produced
        self.embedder = layers[0]
        self.outdim = innerdim
        self.inconcat = inconcat
        self.outconcat = outconcat
        self.attention = attention
        super(SeqDecoder, self).__init__(*layers[1:], **kw)     # puts layers into a ReccableBlock
        self._mask = False
        self._attention = None
        assert(isinstance(self.block, ReccableBlock))
        if softmaxoutblock is None: # default softmax out block
            sm = Softmax()
            self.lin = MatDot(indim=self.outdim, dim=self.embedder.indim)
            self.softmaxoutblock = asblock(lambda x: sm(self.lin(x)))
        else:
            self.softmaxoutblock = softmaxoutblock
        self.init_states = None

    def set_init_states(self, *states):
        self.init_states = states

    def apply(self, context, seq, context_0=None, **kw):    # context: (batsize, enc.innerdim), seq: idxs-(batsize, seqlen)
        sequences = [seq.dimswap(1, 0)]     # sequences: (seqlen, batsize)
        context_0 = self._get_ctx_t0(context, context_0)
        if self.init_states is not None:
            init_info = self.block.get_init_info(self.init_states)  # sets init states to provided ones
        else:
            init_info = self.block.get_init_info(seq.shape[0])           # initializes zero init states
        outputs, _ = T.scan(fn=self.recwrap,
                            sequences=sequences,
                            outputs_info=[None, context, context_0, 0] + init_info)
        return outputs[0].dimswap(1, 0)     # returns probabilities of symbols --> (batsize, seqlen, vocabsize)

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

    def recwrap(self, x_t, ctx, ctx_tm1, t, *states_tm1):  # x_t: (batsize), context: (batsize, enc.innerdim)
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
        attcon = SeqEncoder(
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
        attcon = SeqEncoder(
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
        self.rnn = ReccableStack(IdxToOneHot(vocsize), GRU(dim=vocsize, innerdim=encdim))
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
        self.rnn = RecurrentStack(IdxToOneHot(vocsize), GRU(dim=vocsize, innerdim=encdim))
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
        self.rnn = RecurrentStack(IdxToOneHot(vocsize),
                                  BiRNU.fromrnu(GRU, dim=vocsize, innerdim=encdim))
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
        self.rnn = RecurrentStack(IdxToOneHot(vocsize),
                                  BiRNU.fromrnu(GRU, dim=vocsize, innerdim=encdim))
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
        self.block = RecurrentStack(*(layers + (Lin(indim=smodim, dim=outdim), Softmax())))

    def apply(self, inpseq, maskseq=None):    # inpseq: idx^(batsize, seqlen), maskseq: f32^(batsize, seqlen)
        embseq = self.embedder(inpseq)
        res = self.block(embseq)            # f32^(batsize, seqlen, outdim)
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


class SeqTransDec(Block, RecUsable):
    def __init__(self, *layers, **kw):
        """ first two layers must be embedding layers. Final softmax is added automatically"""
        assert("smodim" in kw and "outdim" in kw)
        smodim = kw["smodim"]
        outdim = kw["outdim"]
        del kw["smodim"]; del kw["outdim"]
        super(SeqTransDec, self).__init__(**kw)
        self.inpemb = layers[0]
        self.outemb = layers[1]
        self.block = RecurrentStack(*(layers[2:] + (Lin(indim=smodim, dim=outdim), Softmax())))

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

    def recappl(self, inps, states):
        emb = self._get_emb(*inps)
        inps, heads, tail = self.block.recappl(emb, states)
        return inps, heads, tail

    def recappl_init(self, ist):
        return self.block.get_init_info(ist)

    def get_init_info(self, initstates):
        return self.block.get_init_info(initstates)


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