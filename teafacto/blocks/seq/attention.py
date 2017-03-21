from teafacto.blocks.basic import MatDot as Lin, VectorEmbed, IdxToOneHot
from teafacto.core.base import Block, param, Val
from teafacto.blocks.match import Distance, CosineDistance, DotDistance, ForwardDistance
from teafacto.core.base import tensorops as T
from teafacto.blocks.activations import GumbelSoftmax, Softmax
import numpy as np
import theano, theano.tensor


############################## ATTENTION GENERATORS ###############################

class AttGen(Block):
    """ wraps a distance """
    def __init__(self, distance, normalizer=Softmax(),
                 sampler=None, sample_temperature=0.2, **kw):
        super(AttGen, self).__init__(**kw)
        self.dist = distance
        self.normalizer = normalizer
        self.sampler = sampler
        self.set_sampler(sampler, sample_temperature)

    def set_sampler(self, sampler=None, sample_temperature=0.2):
        if sampler == "gumbel":
            self.sampler = GumbelSoftmax(temperature=sample_temperature)

    def apply(self, criterion, data, mask=None):
        mask = data.mask if mask is None else mask
        o = self.dist(criterion, data)
        o_out = self.normalizer(o, mask=mask)
        if self.sampler is not None:
            o_out = self.sampler(o_out, mask=mask)
        o_out.mask = mask
        return o_out


################################ ATTENTION CONSUMERS #####################################

class AttentionConsumer(Block):
    '''
     A block that consumes some data and associated attention weights to generate a *context* that can be fed further.
     Subclass this for attention-consuming blocks.
    '''
    def apply(self, data, weights, mask=None):
        """
        :param data:    data to aggregate ->            (batsize, seqlen, memdim)
        :param weights: weights to use to aggregate ->  (batsize, seqlen)
        """
        mask = data.mask if mask is None else mask
        weights = weights * mask if mask is not None else weights
        return self._inner_apply(data, weights)


class WeightedSumAttCon(AttentionConsumer):    # applies attention to sequence while summing up
    def _inner_apply(self, data, weights):   # data: (batsize, seqlen, elem_dim)
                                      # weights: (batsize, seqlen)
        w = weights.dimshuffle(0, 1, 'x')
        ret = data * w
        return T.sum(ret, axis=1)


class WeightedMaxPoolAttCon(AttentionConsumer):     # <-- does it even make sense?
    def _inner_apply(self, data, weights):
        w = weights.dimshuffle(0, 1, 'x')
        mindata = T.min(data)
        wdata = ((data - mindata) * w) + mindata
        return T.max(wdata, axis=-2)


# ATTENTIONS

class Attention(Block):
    def __init__(self, attentiongenerator=None, attentionconsumer=None,
                 splitters=None,        # two blocks, each applied to data, first used for addr, second used for content
                 **kw):
        super(Attention, self).__init__(**kw)
        attentiongenerator = AttGen(DotDistance()) if attentiongenerator is None else attentiongenerator
        attentionconsumer = "sum" if attentionconsumer is None else attentionconsumer
        if isinstance(attentiongenerator, AttGen):
            self.attentiongenerator = attentiongenerator
        elif isinstance(attentiongenerator, Distance):
            self.attentiongenerator = AttGen(attentiongenerator)
        if attentionconsumer == "sum":
            attentionconsumer = WeightedSumAttCon()
        elif attentionconsumer == "max":
            attentionconsumer = WeightedMaxPoolAttCon()
        self.attentionconsumer = attentionconsumer
        self.splitters = splitters

    # fluent
    def dot_gen(self):
        self.attentiongenerator = AttGen(DotDistance())
        return self

    def forward_gen(self, ldim, rdim, innerdim=100):
        self.attentiongenerator = AttGen(ForwardDistance(ldim, rdim, aggdim=innerdim))
        return self

    def apply(self, criterion, data, mask=None):
        mask = data.mask if mask is None else mask
        addrdata = data if self.splitters is None else self.splitters[0](data)
        weights = self.attentiongenerator(criterion, addrdata, mask=mask)
        weights.output_as("attention_weights")
        contdata = data if self.splitters is None else self.splitters[1](data)
        ret = self.attentionconsumer(contdata, weights, mask=mask)
        return ret
