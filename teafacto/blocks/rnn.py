from teafacto.blocks.core import *
from teafacto.blocks.rnu import *
from teafacto.blocks.core import tensorops as T


class RecurrentStack(RecurrentBlock):
    def __init__(self, *args, **kw): # layer can be a layer or function
        super(RecurrentStack, self).__init__(**kw)
        self.layers = args
        self.initstates = None

    def set_init_states(self, values): # one state per RNU, non-RNU's are ignored
        self.initstates = values

    def do_set_init_states(self):
        recurrentlayers = filter(lambda x: isinstance(x, RecurrentBlock), self.layers)
        if self.initstates is None:
            self.initstates = [None] * len(recurrentlayers)
        if len(recurrentlayers) != len(self.initstates):
            raise AssertionError("number of states should be the same as number of stateful layers in stack")
        z = zip(recurrentlayers, self.initstates)
        for l, s in z:
            l.set_init_states(s)

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

    def get_init_info(self, batsize):
        self.do_set_init_states()
        recurrentlayers = filter(lambda x: isinstance(x, RecurrentBlock), self.layers)
        init_infos = []
        for recurrentlayer in recurrentlayers: # insert in the front
            i = 0
            for initinfo in recurrentlayer.get_init_info(batsize):
                init_infos.insert(i, initinfo)
                i += 1
        return init_infos   # layerwise in reverse

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
                nextinp = block(*nextinp)
        return [nextinp] + nextstates

    def apply(self, seq):
        seq = seq.dimswap(1, 0)
        outputs, _ = T.scan(fn=self.rec,
                            sequences=seq,
                            outputs_info=[None]+self.get_init_info(seq.shape[1]))
        output = outputs[0]
        return output.dimswap(1, 0)


class RecurrentBlockParameterized(object):
    def __init__(self, *layers, **kw):
        super(RecurrentBlockParameterized, self).__init__(**kw)
        if len(layers) > 0:
            self.block = RecurrentStack(*layers)
        else:
            self.block = None

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


class RNNEncoder(RecurrentBlockParameterized, Block):
    '''
    Encodes a sequence of vectors into a vector, input dims and output dims specified by the RNU unit
    Returns multiple outputs, multiple states
    Builds for one output
    '''

    def apply(self, seq): # seq: (batsize, seqlen, dim)
        inp = seq.dimswap(1, 0)
        outputs, _ = T.scan(fn=self.recwrap,
                            sequences=inp,
                            outputs_info=[None]+self.block.get_init_info(seq.shape[0]))
        output = outputs[0]
        states = self.block.get_states_from_outputs(outputs[1:])
        #return output[-1, :, :] #output is (batsize, innerdim)
        return [s[-1, :, :] for s in states]

    def _build(self, *inps):
        self.output = self.wrapply(*inps)[0]

    def recwrap(self, x_t, *args): # x_t: (batsize, dim)      if input is all zeros, just return previous state
        mask = x_t.norm(2, axis=1) > 0 # (batsize, )
        rnuret = self.block.rec(x_t, *args) # list of matrices (batsize, **somedims**)
        ret = map(lambda (origarg, rnuretarg): (origarg.T * (1 - mask) + rnuretarg.T * mask).T, zip([args[0]] + list(args), rnuret)) # TODO mask breaks multi-layered encoders (order is reversed)
        return ret

    def onAttach(self):
        pass


class RNNDecoder(RecurrentBlockParameterized, Block):
    '''
    Decodes a sequence given initial state
    output: probabilities over symbol space float: (batsize, seqlen, dim) where dim is number of symbols

    TERMINUS SYMBOL = 0
    ! first input is TERMINUS ==> suggest to set TERMINUS(0) embedding to all zeroes (in s2vf)
    '''
    # TODO: test
    def __init__(self, hlimit=50, **kw): # limit says at most how many is produced
        super(RNNDecoder, self).__init__(**kw)
        self.limit = hlimit

    def onAttach(self):
        pass

    def apply(self, initstates, initprobs=None): # initstates: list of (batsize, innerdim)
        batsize = initstates[0].shape[0]
        if initprobs is None:
            initprobs = T.eye(1, self.block.dim).repeat(batsize, axis=0) # all TERMINUS (batsize, dim)
        self.block.set_init_states(initstates)
        outputs, _ = T.scan(fn=self.recwrap,
                                 outputs_info=[initprobs, 0]+self.block.get_init_info(batsize),
                                 n_steps=self.limit)
        return outputs[0].dimshuffle(1, 0, 2) # returns probabilities of symbols --> (batsize, seqlen, vocabsize)

    def recwrap(self, x_t, i, *args): # once output is terminus, always terminus and previous state is returned
        chosen = x_t.argmax(axis=1, keepdims=False) # x_t = probs over symbols:: f32-(batsize, dim) ==> int32-(batsize,)
        mask = T.clip(chosen.reshape(chosen.shape[0], 1) + T.clip(1-i, 0, 1), 0, 1) # (batsize, ) --> only make mask if not in first iter
        rnuret = self.block.rec(chosen, *args) # list of matrices (batsize, **somedims**)
        outprobs = rnuret[0]
        ret = map(lambda (a, r): (a.T * (1-mask) + r.T * mask).T, zip([x_t] + list(args), [outprobs]+rnuret[1:]))
        i = i + 1
        return [ret[0], i] + ret[1:]#, {}, T.until( (i > 1) * T.eq(mask.norm(1), 0) )


class RNNMask(RecurrentBlockParameterized):
    '''
    Puts an entry-wise RNN mask on a sequence of vectors.
    RNU output dims must be the same as input dims
    '''
    def mask(self, seq): # seq: (batsize, seqlen, dim)
        inp = seq.dimshuffle(1, 0, 2)
        # initialize hidden states, depending on the used RNN (numstates)
        numstates = len(inspect.getargspec(self.rnu.rec).args) - 2
        initstate = T.zeros((inp.shape[1], self.rnu.innerdim)) # (nb_samples, dim)
        # run RNU over it
        outputs, _ = T.scan(fn=self.recwrap,
                                 sequences=inp,
                                 outputs_info=[None]+[initstate]*numstates)
        outputs = T.nnet.sigmoid(outputs[0]).dimshuffle(1, 0, 2) # outputs: [0, 1] of (batsize, seqlen, dim)
        mseq = seq * outputs # apply the mask
        #embed()
        return mseq

    def recwrap(self, x_t, *args): # x_t: (batsize, dim)
        return self.rnu.rec(x_t, *args) # list of matrices (batsize, **somedims**)
