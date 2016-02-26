from teafacto.blocks.core import *
from teafacto.blocks.rnu import *
from teafacto.blocks.core import tensorops as T


class RecurrentStack(RNUParam):
    def __init__(self, dim, *blocks, **kw): # layer can be a layer or function
        # TODO --> to support different kinds of RNU's and intermediate transformations (incl. dropout)
        self.dim = dim
        self.blocks = blocks
        self.initstates = None
        super(RecurrentStack, self).__init__(**kw)

    def set_init_states(self, values): # one state per RNU, non-RNU's are ignored
        self.initstates = values

    def do_set_init_states(self):
        rnulayers = filter(lambda x: isinstance(x, RNUBase), self.blocks)
        if self.initstates is None:
            self.initstates = [None] * len(rnulayers)
        if len(rnulayers) != len(self.initstates):
            raise AssertionError("number of states should be the same as number of stateful layers in stack")
        z = zip(rnulayers, self.initstates)
        for l, s in z:
            l.set_init_states(s)

    def get_init_info(self, batsize):
        self.do_set_init_states()
        rnulayers = filter(lambda x: isinstance(x, RNUBase), self.blocks)
        init_infos = []
        for rnul in rnulayers:
            init_infos.extend(rnul.get_init_info(batsize))
        return init_infos

    def rec(self, x_t, *states):
        # apply each block on x_t to get next-level input, consume states in the process
        nextinp = x_t
        nextstates = []
        for block in self.blocks:
            numstates = 0
            if isinstance(block, RNUBase): # real RNU # TODO: also accept RecurrentStacks here
                numstates = len(inspect.getargspec(block.rec).args) - 2
                rnuret = block.rec(nextinp, *states[0:numstates])
                nextinp = rnuret[0]
                nextstates.extend(rnuret[1:])
                states = states[numstates:]
            else: # block is a function
                nextinp = block(nextinp)
        return [nextinp] + nextstates

    @property
    def depparameters(self):
        ret = set()
        for block in self.blocks:
            if isinstance(block, Parameterized):
                ret.update(block.parameters)
        return ret

class RNUParameterized(Parameterized):
    def __init__(self, **kw):
        self.rnu = None
        super(RNUParameterized, self).__init__(**kw)

    def __add__(self, other):
        self.attach(other)
        return self

    def attach(self, other):
        if isinstance(other, RNUParam):
            self.rnu = other
            self.afterRNUattach()
        return self

    def afterRNUattach(self):
        raise NotImplementedError("use subclass")

    @property
    def depparameters(self):
        return self.rnu.parameters


class RNNEncoder(RNUParameterized):
    '''
    Encodes a sequence of vectors into a vector, input dims and output dims specified by the RNU unit
    '''

    def encode(self, seq): # seq: (batsize, seqlen, dim)
        inp = seq.dimshuffle(1, 0, 2)
        outputs, _ = T.scan(fn=self.recwrap,
                            sequences=inp,
                            outputs_info=[None]+self.rnu.get_init_info(seq.shape[0]))
        output = outputs[0]
        return output[-1, :, :] #output is (batsize, innerdim)

    def recwrap(self, x_t, *args): # x_t: (batsize, dim)      if input is all zeros, just return previous state
        mask = x_t.norm(2, axis=1) > 0 # (batsize, )
        rnuret = self.rnu.rec(x_t, *args) # list of matrices (batsize, **somedims**)
        ret = map(lambda (a, r): (a.T * (1-mask) + r.T * mask).T, zip([args[0]] + list(args), rnuret))
        return ret

    def afterRNUattach(self):
        pass


class RNNDecoder(RNUParameterized):
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
        # outdims are the same as RNU's input dims
        self.O = None
        self.W = None

    @property
    def ownparameters(self):
        ret = set()
        if self.O is not None:
            ret.add(self.O)
        return ret

    def afterRNUattach(self):
        if not isinstance(self.rnu, RecurrentStack): # ==> create our default recurrent stack
            self.W = T.eye(self.rnu.dim, self.rnu.dim) # one-hot embedding matrix
            self.O = param((self.rnu.innerdim, self.rnu.dim)).uniform()
            rnu = self.rnu
            rs = RecurrentStack(lambda x: self.W[x, :],
                                rnu,
                                lambda x: T.dot(x, self.O),
                                lambda x: T.nnet.softmax(x))
            self.rnu = rs

    def decode(self, initstates, initprobs=None): # initstates: list of (batsize, innerdim)
        batsize = initstates[0].shape[0]
        if initprobs is None:
            initprobs = T.eye(1, self.rnu.dim).repeat(batsize, axis=0) # all TERMINUS (batsize, dim)
        self.rnu.set_init_states(initstates)
        outputs, _ = T.scan(fn=self.recwrap,
                                 outputs_info=[initprobs, 0]+self.rnu.get_init_info(batsize),
                                 n_steps=self.limit)
        return outputs[0].dimshuffle(1, 0, 2) # returns probabilities of symbols --> (batsize, seqlen, vocabsize)

    def recwrap(self, x_t, i, *args): # once output is terminus, always terminus and previous state is returned
        chosen = x_t.argmax(axis=1, keepdims=False) # x_t = probs over symbols:: f32-(batsize, dim) ==> int32-(batsize,)
        mask = T.clip(chosen.reshape(chosen.shape[0], 1) + T.clip(1-i, 0, 1), 0, 1) # (batsize, ) --> only make mask if not in first iter
        rnuret = self.rnu.rec(chosen, *args) # list of matrices (batsize, **somedims**)
        outprobs = rnuret[0]
        ret = map(lambda (a, r): (a.T * (1-mask) + r.T * mask).T, zip([x_t] + list(args), [outprobs]+rnuret[1:]))
        i = i + 1
        return [ret[0], i] + ret[1:]#, {}, T.until( (i > 1) * T.eq(mask.norm(1), 0) )


class RNNMask(RNUParameterized):
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