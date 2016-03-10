from teafacto.blocks.rnu import GRU
from teafacto.blocks.rnu import RecurrentBlock
from teafacto.core.base import Block, tensorops as T
from teafacto.blocks.basic import IdxToOneHot, VectorEmbed, Embedder, Softmax, MatDot
from teafacto.blocks.attention import WeightedSum, LinearSumAttentionGenerator, Attention, AttentionConsumer, LinearGateAttentionGenerator
from teafacto.util import issequence
import inspect
from IPython import embed


class RecurrentStack(RecurrentBlock):
    def __init__(self, *args, **kw): # layer can be a layer or function
        super(RecurrentStack, self).__init__(**kw)
        self.layers = args

    def __getitem__(self, idx):
        return self.layers[idx]

    def get_states_from_outputs(self, outputs):
        # outputs are ordered from topmost recurrent layer first ==> split and delegate
        states = []
        for recurrentlayer in filter(lambda x: isinstance(x, RecurrentBlock), self.layers): # from bottom -> eat from behind; insert to the front
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
        recurrentlayers = list(filter(lambda x: isinstance(x, RecurrentBlock), self.layers))
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
            if isinstance(block, RecurrentBlock):
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

    def apply(self, seq, initstates=None):
        seq = seq.dimswap(1, 0)
        initstatearg = initstates if initstates is not None else seq.shape[1]
        outputs, _ = T.scan(fn=self.rec,
                            sequences=seq,
                            outputs_info=[None]+self.get_init_info(initstatearg))
        output = outputs[0]
        return output.dimswap(1, 0)


class RecurrentBlockParameterized(object):
    def __init__(self, *layers, **kw):
        super(RecurrentBlockParameterized, self).__init__(**kw)
        if len(layers) > 0:
            if len(layers) == 1:
                self.block = layers[0]
                assert(isinstance(self.block, RecurrentBlock))
            else:
                self.block = RecurrentStack(*layers)
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


class SeqEncoder(RecurrentBlockParameterized, AttentionConsumer, Block):
    '''
    Encodes a sequence of vectors into a vector, input dims and output dims specified by the RNU unit
    Returns multiple outputs, multiple states
    Builds for one output
    '''
    _withoutput = False
    _all_states = False
    _weighted = False

    def apply(self, seq, weights=None): # seq: (batsize, seqlen, dim), weights: (batsize, seqlen)
        inp = seq.dimswap(1, 0)
        if weights is None:
            weights = T.ones((inp.shape[0], inp.shape[1])) # (seqlen, batsize)
        else:
            self._weighted = True
            weights = weights.dimswap(1, 0)
        outputs, _ = T.scan(fn=self.recwrap,
                            sequences=[inp, weights],
                            outputs_info=[None]+self.block.get_init_info(seq.shape[0]))
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
        if x_t.ndim == 1:       # ==> indexes
            mask = x_t > 0      # 0 is TERMINUS
        else:
            mask = x_t.norm(2, axis=1) > 0 # mask: (batsize, )
        rnuret = self.block.rec(x_t, *args) # list of matrices (batsize, **somedims**)
        if self._weighted:
            rnuret = map(lambda (origarg, rnuretarg): (origarg.T * (1 - w_t) + rnuretarg.T * w_t).T, zip([args[0]] + list(args), rnuret))
        ret = map(lambda (origarg, rnuretarg): (origarg.T * (1 - mask) + rnuretarg.T * mask).T, zip([args[0]] + list(args), rnuret)) # TODO mask breaks multi-layered encoders (order is reversed)
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


class SeqDecoder(RecurrentBlockParameterized, Block):
    '''
    Decodes a sequence of symbols given context
    output: probabilities over symbol space: float: (batsize, seqlen, vocabsize)

    Supports attention

    TERMINUS SYMBOL = 0
    ! first input is TERMINUS ==> suggest to set TERMINUS(0) embedding to all zeroes (in s2vf)
    ! first layer must be an embedder or IdxToOneHot, otherwise, an IdxToOneHot is created automatically based on given dim
    '''
    def __init__(self, *layers, **kw): # limit says at most how many is produced
        if "seqlen" in kw:
            self.limit = kw["seqlen"]
            del kw["seqlen"]
        else:
            self.limit = 50
        try:
            self.indim = kw["indim"]
            del kw["indim"]
        except Exception:
            raise Exception("must provide input dimension with indim=<inp_dim>")
        self.idxtovec = layers[0]
        if not isinstance(self.idxtovec, Embedder):
            raise AssertionError("first layer must be an embedding block")
        super(SeqDecoder, self).__init__(*layers[1:], **kw)
        self._mask = False
        self._attention = None

    def set_attention(self, att):
        self._attention = att

    @property
    def has_attention(self):
        return self._attention is not None and isinstance(self._attention, Attention)

    @property
    def attention(self):
        return self._attention

    def apply(self, context, **kw): # encoding: Var: (batsize, innerdim) OR a block with attention that produces (batsize, innerdim) based on what decoder provides
        if isinstance(context, Attention):
            self.set_attention(context)
            return self
        initprobs, batsize, seqlen, context = self._get_scan_args(context, **kw)
        outputs, _ = T.scan(fn=self.recwrap,
                            outputs_info=[initprobs, 0, context]+self.block.get_init_info(batsize),
                            n_steps=seqlen)
        return outputs[0].dimshuffle(1, 0, 2) # returns probabilities of symbols --> (batsize, seqlen, vocabsize)

    def _get_scan_args(self, context, **kw):
        if "seqlen" in kw:
            seqlen = kw["seqlen"]
        else:
            seqlen = self.limit
        batsize = context.shape[0]
        if "initprobs" in kw:
            initprobs = kw["initprobs"]
            del kw["initprobs"]
        else:
            initprobs = T.eye(1, self.indim).repeat(batsize, axis=0) # all TERMINUS (batsize, dim)
        return initprobs, batsize, seqlen, context

    def recwrap(self, x_t, t, context, *states):   # once output is terminus, always terminus and previous state is returned
        chosen = x_t.argmax(axis=1, keepdims=False)                 # x_t = probs over symbols:: f32-(batsize, dim) ==> int32-(batsize,)
        chosenvec = self.idxtovec(chosen)                           # chosenvec: (batsize, embdim)
        context_t = self._gen_context(context, *states)
        blockarg = T.concatenate([context_t, chosenvec], axis=1)     # concat encdim of encoding and embdim of chosenvec
        if self._mask:
            mask = T.clip(chosen.reshape(chosen.shape[0], 1) + T.clip(1-t, 0, 1), 0, 1)     # (batsize,) --> only make mask if not in first iter
        rnuret = self.block.rec(blockarg, *states)                                          # list of matrices (batsize, **somedims**)
        if self._mask:
            ret = map(lambda (prevval, newval): (prevval.T * (1-mask) + newval.T * mask).T, zip([x_t] + list(states), rnuret))
        else:
            ret = rnuret
        t = t + 1
        return [ret[0], t, context] + ret[1:]#, {}, T.until( (i > 1) * T.eq(mask.norm(1), 0) )

    def _gen_context(self, context, *states):
        if self.has_attention:
            criterion = T.concatenate(self.block.get_states_from_outputs(states), axis=1)   # states are (batsize, statedim)
            return self.attention(criterion, context)   # ==> criterion is (batsize, sum_of_statedim), context is (batsize, ...)
            # output should be (batsize, block_input_dims)
        else:
            return context


class RNNAutoEncoder(Block):    # tries to decode original sequence
    def __init__(self, vocsize=25, encdim=200, innerdim=200, seqlen=50, **kw):
        super(RNNAutoEncoder, self).__init__(**kw)
        self.seqlen = seqlen
        self.encoder = SeqEncoder(
            IdxToOneHot(vocsize=vocsize),
            GRU(dim=vocsize, innerdim=encdim))
        self.decoder = SeqDecoder(IdxToOneHot(vocsize),
                                  GRU(dim=vocsize+encdim, innerdim=innerdim),
                                  MatDot(indim=innerdim, dim=vocsize),
                                  Softmax(), indim=vocsize, seqlen=seqlen)

    def apply(self, inpseq):
        enc = self.encoder(inpseq)
        dec = self.decoder(enc, seqlen=inpseq.shape[1])
        return dec


class AttentionRNNAutoEncoder(Block):
    '''
    Take the input index sequence as-is, transform to one-hot, feed to gate AttentionGenerator, encode with weighted SeqEncoder,
    put everything as attention inside SeqDecoder
    '''
    def __init__(self, vocsize=25, encdim=200, innerdim=200, attdim=50, seqlen=50, **kw):
        super(AttentionRNNAutoEncoder, self).__init__(**kw)
        self.seqlen = seqlen
        self.emb = IdxToOneHot(vocsize)
        attgen = LinearGateAttentionGenerator(indim=innerdim+vocsize, innerdim=attdim)
        attcon = SeqEncoder(
            GRU(dim=vocsize, innerdim=encdim))
        self.dec = SeqDecoder(IdxToOneHot(vocsize),
                              GRU(dim=vocsize+encdim, innerdim=innerdim),
                              MatDot(indim=innerdim, dim=vocsize),
                              Softmax(), indim=vocsize, seqlen=seqlen)
        self.dec(Attention(attgen, attcon))

    def apply(self, inpseq):    # inpseq: indexes~(batsize, seqlen)
        inp = self.emb(inpseq)  # inp:    floats~(batsize, seqlen, vocsize)
        return self.dec(inp)

