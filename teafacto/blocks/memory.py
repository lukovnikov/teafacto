from teafacto.blocks.basic import VectorEmbed, Softmax
from teafacto.core.base import Block

class MemoryBlock(Block):
    """
    Memory Blocks are preloaded with a collection of items on which the defined transformations are defined.
    The transformation should generate a vector representation for each element.
    Further, they act as Vector Embedding blocks.

    """
    def __init__(self, **kw):
        super(MemoryBlock, self).__init__(**kw)