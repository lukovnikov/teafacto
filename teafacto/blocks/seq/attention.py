from teafacto.blocks.basic import MatDot as Lin, Softmax, VectorEmbed, IdxToOneHot
from teafacto.core.base import Block, param, Val
from teafacto.blocks.match import Distance
from teafacto.core.base import tensorops as T
import numpy as np
import theano, theano.tensor

############################## ATTENTION GENERATORS ###############################

class AttGen(Block):
    """ wraps a distance """
    def __init__(self, distance, **kw):
        super(AttGen, self).__init__(**kw)
        self.dist = distance

    def apply(self, criterion, data, mask=None):
        o = self.dist(criterion, data)
        #o_out = Softmax()(o, mask=mask)
        #return o_out
        return o



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


class Attention(Block):
    '''
    Block wraps both an AttentionGenerator and AttentionConsumer.
    '''
    def __init__(self, attentiongenerator, attentionconsumer=WeightedSumAttCon, **kw):
        super(Attention, self).__init__(**kw)
        if isinstance(attentiongenerator, AttGen):
            self.attentiongenerator = attentiongenerator
        elif isinstance(attentiongenerator, Distance):
            self.attentiongenerator = AttGen(attentiongenerator)
        self.attentionconsumer = attentionconsumer

    def apply(self, criterion, data, mask=None):
        attention = self.attentiongenerator(criterion, data, mask=mask)
        return self.attentionconsumer(data, attention)


