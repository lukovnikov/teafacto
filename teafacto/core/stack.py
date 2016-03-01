from base import *
from rnu import RNUBase
from rnn import RecurrentStack


def stack(*layers, **kw):
    rec = False
    for layer in layers:
        if isinstance(layer, RNUBase):
            rec = True
            break
    if rec is True:
        return RecurrentStack(*layers, **kw)
    else:
        return BlockStack(*layers, **kw)


class BlockStack(Block):
    def __init__(self, *layers, **kw):
        super(BlockStack, self).__init__(**kw)
        self.layers = layers

    # TODO: if one of the layers is recurrent, wrap all non-recurrent layers in a scan
    def apply(self, *vars):
        ret = vars
        for layer in self.layers:
            ret = [layer(ret[0])]
            assert(len(ret) == 1)
        return ret[0]