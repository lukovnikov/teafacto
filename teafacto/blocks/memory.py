from teafacto.blocks.basic import VectorEmbed, Softmax, Embedder
from teafacto.core.base import Block, Val, Var


class MemoryBlock(Embedder):
    """
    Memory Blocks are preloaded with a collection of items on which the defined transformations are defined.
    The transformation should generate a vector representation for each element.
    Further, they act as Vector Embedding blocks.

    """
    def __init__(self, block=None, data=None, indim=200, outdim=50, **kw):
        assert(block is not None and data is not None)
        # TODO: convert non-Var or non-Val data to Val
        if not isinstance(data, (Var, Val)):
            data = Val(data)
        assert(isinstance(data, (Var, Val)) and isinstance(block, Block))
        super(MemoryBlock, self).__init__(indim, outdim, **kw)      # outdim = outdim of the contained block
        self.payload = block
        self.innervar = block(data)     # innervar: (indim, outdim)
        pass

    def apply(self, idxs):  # idxs: ints of (batsize,)
        return self.innervar[idxs, :]