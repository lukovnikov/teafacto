from teafacto.blocks.seq.rnu import ReccableBlock, RecurrentBlock, ReccableWrapper
from teafacto.core.base import Block
from teafacto.util import issequence


class RecStack(ReccableBlock):
    # must handle RecurrentBlocks ==> can not recappl, if all ReccableBlocks ==> can do recappl
    # must give access to final states of internal layers
    # must give access to all outputs of top layer
    # must handle masks
    def __init__(self, *layers, **kw):
        super(RecStack, self).__init__(**kw)
        self.layers = []
        for l in layers:
            if isinstance(l, RecurrentBlock):
                self.layers.append(l)
            elif isinstance(l, Block):
                self.layers.append(ReccableWrapper(l))
            else:
                raise Exception("cannot apply this layer")

    @property
    def numstates(self):
        return reduce(lambda x, y: x + y, [x.numstates for x in self.layers if isinstance(x, RecurrentBlock)], 0)

    def get_statespec(self):
        return [l.get_statespec() for l in self.layers]

    # FWD API. initial states can be set, mask is accepted, everything is returned. Works for all RecurrentBlocks
    # FWD API IMPLEMENTED USING FWD API
    def innerapply(self, seq, mask=None, initstates=None):
        states = []     # bottom states first
        for layer in self.layers:
            if initstates is not None:
                layerinpstates = initstates[:layer.numstates]
                initstates = initstates[layer.numstates:]
            else:
                layerinpstates = None
            final, seq, layerstates = layer.innerapply(seq, mask=mask, initstates=layerinpstates)
            states.extend(layerstates)
        return final, seq, states           # full history of final output and all states (ordered from bottom layer to top)

    @classmethod
    def apply_mask(cls, xseq, maskseq=None):
        if maskseq is None:
            ret = xseq
        else:
            mask = T.tensordot(maskseq, T.ones((xseq.shape[2],)), 0)  # f32^(batsize, seqlen, outdim) -- maskseq stacked
            ret = mask * xseq
        return ret

    # REC API: only works with ReccableBlocks
    def get_init_info(self, initstates):
        recurrentlayers = list(filter(lambda x: isinstance(x, ReccableBlock), self.layers))
        assert (len(filter(lambda x: isinstance(x, RecurrentBlock) and not isinstance(x, ReccableBlock),
                           self.layers)) == 0)  # no non-reccable blocks allowed
        if issequence(initstates):  # fill up init state args so that layers for which no init state is specified get default arguments that lets them specify a default init state
                                    # if is a sequence, expecting a value, not batsize
            if len(initstates) < self.numstates:    # top layers are being given the given init states, bottoms make their own default
                initstates = [None] * (self.numstates - len(initstates)) + initstates
            batsize = 0
            for initstate in initstates:
                if initstate is not None:
                    batsize = initstate.shape[0]
            initstates = [batsize if initstate is None else initstate for initstate in initstates]
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