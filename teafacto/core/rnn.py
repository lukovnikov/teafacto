from base import *
from rnu import *
from base import tensorops as T


class RecurrentStack(RecurrentBlock):
    def __init__(self, *args, **kw): # layer can be a layer or function
        super(RecurrentStack, self).__init__(**kw)
        self.layers = args

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
    _withoutput = False
    _all_states = False

    def apply(self, seq): # seq: (batsize, seqlen, dim)
        inp = seq.dimswap(1, 0)
        outputs, _ = T.scan(fn=self.recwrap,
                            sequences=inp,
                            outputs_info=[None]+self.block.get_init_info(seq.shape[0]))
        output = outputs[0]
        if self._all_states:
            states = self.block.get_states_from_outputs(outputs[1:])
            ret = [s[-1, :, :] for s in states]
        else:
            ret = [output[-1, :, :]] #output is (batsize, innerdim)
        if self._withoutput:
            return ret + [output]
        else:
            if len(ret) == 1:
                return ret[0]
            else:
                return ret

    def _build(self, *inps):
        res = self.wrapply(*inps)
        if issequence(res):
            output = res[0]
        else:
            output = res
        return output

    def recwrap(self, x_t, *args): # x_t: (batsize, dim)      if input is all zeros, just return previous state
        mask = x_t.norm(2, axis=1) > 0 # (batsize, )
        rnuret = self.block.rec(x_t, *args) # list of matrices (batsize, **somedims**)
        ret = map(lambda (origarg, rnuretarg): (origarg.T * (1 - mask) + rnuretarg.T * mask).T, zip([args[0]] + list(args), rnuret)) # TODO mask breaks multi-layered encoders (order is reversed)
        return ret

    def onAttach(self):
        pass

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


class RNNDecoder(RecurrentBlockParameterized, Block):
    '''
    Decodes a sequence of symbols given initial state
    output: probabilities over symbol space float: (batsize, seqlen, vocabsize)

    TERMINUS SYMBOL = 0
    ! first input is TERMINUS ==> suggest to set TERMINUS(0) embedding to all zeroes (in s2vf)
    '''
    # TODO: test
    def __init__(self, *layers, **kw): # limit says at most how many is produced
        if "hlimit" in kw:
            self.limit = kw["hlimit"]
            del kw["hlimit"]
        else:
            self.limit = 50
        try:
            self.dim = kw["dim"]
            del kw["dim"]
        except Exception:
            raise Exception("must provide input dimension with dim=<inp_dim>")
        super(RNNDecoder, self).__init__(*layers, **kw)

    def onAttach(self):
        pass

    def apply(self, *initstates, **kw): # initstates: list of (batsize, innerdim)
        batsize = initstates[0].shape[0]
        if "initprobs" in kw:
            initprobs = kw["initprobs"]
            del kw["initprobs"]
        else:
            initprobs = T.eye(1, self.dim).repeat(batsize, axis=0) # all TERMINUS (batsize, dim)
        outputs, _ = T.scan(fn=self.recwrap,
                            outputs_info=[initprobs, 0]+self.block.get_init_info(initstates),
                            n_steps=self.limit)
        return outputs[0].dimshuffle(1, 0, 2) # returns probabilities of symbols --> (batsize, seqlen, vocabsize)

    def recwrap(self, x_t, i, *args): # once output is terminus, always terminus and previous state is returned
        chosen = x_t.argmax(axis=1, keepdims=False) # x_t = probs over symbols:: f32-(batsize, dim) ==> int32-(batsize,)
        eye = T.eye(x_t.shape[1], x_t.shape[1])
        chosen = eye[chosen]    # TODO: fix the mask
        #mask = T.clip(chosen.reshape(chosen.shape[0], 1) + T.clip(1-i, 0, 1), 0, 1) # (batsize, ) --> only make mask if not in first iter
        rnuret = self.block.rec(chosen, *args) # list of matrices (batsize, **somedims**)
        outprobs = rnuret[0]
        #ret = map(lambda (a, r): (a.T * (1-mask) + r.T * mask).T, zip([x_t] + list(args), [outprobs]+rnuret[1:]))
        ret = rnuret
        i = i + 1
        return [ret[0], i] + ret[1:]#, {}, T.until( (i > 1) * T.eq(mask.norm(1), 0) )
