from teafacto.blocks.basic import VectorEmbed, Softmax, Embedder
from teafacto.core.base import tensorops as T
from teafacto.core.base import Block, Val, Var, param
from teafacto.util import issequence


class MemoryBlock(Embedder):
    """
    Memory Blocks are preloaded with a collection of items on which the defined transformations are defined.
    The transformation should generate a vector representation for each element.
    Further, they act as Vector Embedding blocks.

    If data argument is not specified, a variable will be assumed to be given to the apply() function.
    """
    def __init__(self, block=None, data=None, indim=200, outdim=50, **kw):
        assert(block is not None)
        ourdata = []
        if not issequence(data):
            data = [data]
        for datae in data:
            if not isinstance(datae, (Var, Val)) and datae is not None:
                ourdata.append(Val(datae))
            else:
                ourdata.append(datae)
        assert(isinstance(block, Block))
        self.data = ourdata
        super(MemoryBlock, self).__init__(indim, outdim, **kw)      # outdim = outdim of the contained block
        self.payload = block
        self.innervar = self.payload(*self.data) if None not in data else None    # innervar: (indim, outdim)

    def apply(self, idxs, *datavar):     # idxs: ints of (batsize,)
        datavars = list(datavar)
        if self.innervar is None:   # not all of data vars provided during construction ==> fill up the Nones
            for i in range(len(self.data)):
                datae = self.data[i]
                if datae is None:
                    if len(datavars) < 1:
                        raise AssertionError()
                    y = datavars.pop(0)
                    self.data[i] = y
            assert(len(datavars) == 0 and self.data.count(None) == 0)
            self.innervar = self.payload(*self.data)
        else:       # all vars should've been provided during construction already
            assert(len(datavar) == 0)
        return self.innervar[idxs, :]


class MemoryAddress(Block):
    def __init__(self, memblock, **kw):
        super(MemoryAddress, self).__init__(**kw)
        self.memblock = memblock    # (mem_size, mem_dim)

    def apply(self, criterion):     # gets a criterion vector (batsize, crit_dim), outputs (batsize, mem_size)
        raise NotImplementedError("use subclass")


class LinearGateMemAddr(MemoryAddress):
    """
    Wraps a memory block
    """
    def __init__(self, memblock, memdim=None, indim=None, attdim=None, **kw):
        assert (indim is not None and memdim is not None and attdim is not None)
        indim = memdim + indim
        innerdim = attdim
        super(LinearGateMemAddr, self).__init__(memblock, **kw)
        self.W = param((indim, innerdim), name="attention_ff").uniform()
        self.U = param((innerdim,), name="attention_agg").uniform()

    def apply(self, criterion):     # criterion: (batsize, crit_dim), self.mem: (mem_size, mem_dim), out: (batsize, mem_size)
        def rec(x_t, crit):                     # x_t: (mem_dim),   crit: (batsize, crit_dim)
            combo = self._get_combo(x_t, crit)  # (batsize, crit_dim + datadim)
            trans = T.dot(combo, self.W)        # (batsize, outdim)
            trans = T.tanh(trans)                                       # apply tanh
            ret = T.dot(trans, self.U)                                  # (batsize, )
            return T.nnet.sigmoid(ret)                                  # apply sigmoid
        o, _ = T.scan(fn=rec, sequences=self.memblock.innervar, non_sequences=criterion)    # (memsize, batsize)
        return o.dimswap(1, 0)                                                              # (batsize, memsize)

    def _get_combo(self, x_t, crit):    # x_t: (mem_dim),   crit: (batsize, crit_dim), out: (batsize, mem_dim + crit_dim)
        x_t_repped = T.repeat(x_t.reshape((x_t.shape[0], 1)), crit.shape[0], axis=1).T    # (batsize, mem_dim)
        return T.concatenate([x_t_repped, crit], axis=1)


class GeneralDotMemAddr(MemoryAddress):
    def __init__(self, memblock, memdim=None, indim=None, attdim=None, **kw):  # indim should be mem_dim, innerdim should be crit_dim
        assert(indim is not None and memdim is not None)     # can not specify separate attention dimensions
        super(GeneralDotMemAddr, self).__init__(memblock, **kw)
        self.W = param((memdim, indim), name="attention").uniform()

    def apply(self, criterion):     # criterion: (batsize, innerdim), self.mem: (mem_size, mem_dim), out: (batsize, mem_size)
        memdot = T.dot(self.memblock.innervar, self.W)  # (mem_size, innerdim)
        def rec(x_t, crit):         # x_t: (indim),   crit: (batsize, indim)
            d = T.dot(crit, x_t)    # (batsize, )
            return T.nnet.sigmoid(d)
        o, _ = T.scan(fn=rec, sequences=memdot, non_sequences=criterion)
        return o.dimswap(1, 0)
