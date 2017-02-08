from teafacto.blocks.basic import MatDot as Lin, Softmax, VectorEmbed, IdxToOneHot
from teafacto.core.base import Block, param, Val
from teafacto.blocks.match import Distance
from teafacto.core.base import tensorops as T
import numpy as np
import theano, theano.tensor


############################## ATTENTION GENERATORS ###############################

class AttGen(Block):
    """ wraps a distance """
    def __init__(self, distance, normalizer=Softmax(), **kw):
        super(AttGen, self).__init__(**kw)
        self.dist = distance
        self.normalizer = normalizer

    def apply(self, criterion, data, mask=None):
        mask = data.mask if mask is None else mask
        o = self.dist(criterion, data)
        o_out = self.normalizer(o, mask=mask)
        o_out.mask = mask
        return o_out


################################ ATTENTION CONSUMERS #####################################

class AttentionConsumer(Block):
    '''
     A block that consumes some data and associated attention weights to generate a *context* that can be fed further.
     Subclass this for attention-consuming blocks.
    '''
    def apply(self, data, weights):
        """
        :param data:    data to aggregate ->            (batsize, seqlen, memdim)
        :param weights: weights to use to aggregate ->  (batsize, seqlen)
        """
        raise NotImplementedError("use subclass")


class WeightedSumAttCon(AttentionConsumer):    # applies attention to sequence while summing up
    def apply(self, data, weights):   # data: (batsize, seqlen, elem_dim)
                                      # weights: (batsize, seqlen)
        w = weights.dimshuffle(0, 1, 'x')
        ret = data * w
        return T.sum(ret, axis=1)


# ATTENTIONS

class Attention(Block):
    def __init__(self, attentiongenerator, attentionconsumer=WeightedSumAttCon(), separate=False, **kw):
        super(Attention, self).__init__(**kw)
        if isinstance(attentiongenerator, AttGen):
            self.attentiongenerator = attentiongenerator
        elif isinstance(attentiongenerator, Distance):
            self.attentiongenerator = AttGen(attentiongenerator)
        self.attentionconsumer = attentionconsumer
        self.separate = separate

    def apply(self, criterion, data, mask=None):
        if not self.separate:
            ret = self._apply_normal(criterion, data, mask=mask)
        else:
            ret = self._apply_separate(criterion, data, mask=mask)
        return ret

    def _apply_normal(self, criterion, data, mask=None):
        weights = self.attentiongenerator(criterion, data, mask=mask)
        weights.output_as("attention_weights")
        ret = self.attentionconsumer(data, weights)
        return ret

    def _apply_separate(self, criterion, data, mask=None):    # data: (batsize, seqlen, 2, dim)
        weights = self.attentiongenerator(criterion, data[:, :, 1, :], mask=mask)
        weights.output_as("attention_weights")
        ret = self.attentionconsumer(data[:, :, 0, :], weights)
        return ret
