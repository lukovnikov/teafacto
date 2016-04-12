from teafacto.blocks.basic import VectorEmbed, Softmax, Embedder
from teafacto.core.base import Block, Val, Var


class MemoryBlock(Embedder):
    """
    Memory Blocks are preloaded with a collection of items on which the defined transformations are defined.
    The transformation should generate a vector representation for each element.
    Further, they act as Vector Embedding blocks.

    If data argument is not specified, a variable will be assumed to be given to the apply() function.
    """
    def __init__(self, block=None, data=None, indim=200, outdim=50, **kw):
        assert(block is not None)
        self.datavar = data is None
        if not isinstance(data, (Var, Val)) and data is not None:
            data = Val(data)
        assert((isinstance(data, (Var, Val)) or data is None) and isinstance(block, Block))
        self.data = data
        super(MemoryBlock, self).__init__(indim, outdim, **kw)      # outdim = outdim of the contained block
        self.payload = block
        self.innervar = self.payload(data) if data is not None else None    # innervar: (indim, outdim)

    def apply(self, idxs, datavar=None):     # idxs: ints of (batsize,)
        if self.innervar is None:
            assert(isinstance(datavar, (Var, Val)))
            self.innervar = self.payload(datavar)
        else:
            assert(datavar is None)
        return self.innervar[idxs, :]