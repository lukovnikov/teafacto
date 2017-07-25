from teafacto.blocks.basic import MatDot as Lin, VectorEmbed, IdxToOneHot, Forward
from teafacto.core.base import Block, param, Val
from teafacto.blocks.match import Distance, CosineDistance, DotDistance, ForwardDistance, EuclideanDistance, LNormSimilarity
from teafacto.core.base import tensorops as T
from teafacto.blocks.activations import GumbelSoftmax, Softmax, Sigmoid, Tanh
from teafacto.blocks.seq.rnu import Gate
import numpy as np
import theano, theano.tensor


############################## ATTENTION GENERATORS ###############################

class AttGen(Block):
    """ wraps a distance
     gates are computed by forward layers from criterion fed into attgen
        gating vector is applied by distance used
    """
    EPS = 1e-4

    def __init__(self, distance=None, normalizer=Softmax(),
                 sampler=None, sample_temperature=0.2, **kw):
        super(AttGen, self).__init__(**kw)
        self.dist = distance
        self.normalizer = normalizer
        self.sampler = sampler
        if sampler is not None:
            self.set_sampler(sampler, sample_temperature)
        self._gated_crit = None
        self._gated_gate = None
        self._crit_trans = None

    def apply(self, criterion, data, mask=None):
        mask = data.mask if mask is None else mask
        distance_gating = None
        if self._gated_gate is not None and self._gated_crit is not None:
            distance_gating = self._gated_gate(criterion)  # (batsize, datadim)
            criterion = self._gated_crit(criterion) # replace criterion
        if self._crit_trans is not None:
            criterion = self._crit_trans(criterion)
        o = self.dist(criterion, data, gates=distance_gating)
        o += self.EPS
        o_out = self.normalizer(o, mask=mask)
        if self.sampler is not None:
            o_out = self.sampler(o_out, mask=mask)
        o_out.mask = mask
        return o_out

    # fluent API for distances
    def eucl_dist(self):
        self.dist = EuclideanDistance()
        return self

    def lnorm_dist(self, L=2):
        self.dist = LNormSimilarity(L=L)
        return self

    def dot_dist(self):
        self.dist = DotDistance()
        return self

    def cosine_dist(self):
        self.dist = CosineDistance()
        return self

    def set_sampler(self, sampler=None, sample_temperature=0.2):
        if sampler == "gumbel":
            print "DO NOT USE THIS"
            self.sampler = GumbelSoftmax(temperature=sample_temperature)
        return self

    #fluent API for gating in attention addressing
    def gated(self, critdim, datadim):
        self._gated_crit = Forward(critdim, datadim, activation=Tanh(), nobias=False)
        self._gated_gate = Forward(critdim, datadim, activation=Sigmoid(), nobias=False)
        return self

    def crit_trans(self, critdim, datadim):
        self._crit_trans = Forward(critdim, datadim, activation=Tanh(), nobias=False)
        return self


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


# ATTENTION
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

    def apply(self, criterion, data, mask=None):
        weights = self.get_attention_weights(criterion, data, mask=mask)
        ret = self.get_attention_results(data, weights, mask=mask)
        return ret

    def get_attention_weights(self, criterion, data, mask=None):
        mask = data.mask if mask is None else mask
        addrdata = data if self.splitters is None else self.splitters[0](data)
        weights = self.attentiongenerator(criterion, addrdata, mask=mask)
        weights.output_as("attention_weights")
        return weights

    def get_attention_results(self, data, weights, mask=None):
        contdata = data if self.splitters is None else self.splitters[1](data)
        ret = self.attentionconsumer(contdata, weights, mask=mask)
        return ret

    # fluent
    def dot_gen(self):
        self.attentiongenerator.dot_dist()
        return self

    def eucl_gen(self):
        self.attentiongenerator.eucl_dist()
        return self

    def lnorm_gen(self, L=2):
        self.attentiongenerator.lnorm_dist(L=L)
        return self

    def forward_gen(self, ldim, rdim, innerdim=100):
        self.attentiongenerator.dist = ForwardDistance(ldim, rdim, aggdim=innerdim)
        return self

    def gated_gen(self, critdim, datadim):
        self.attentiongenerator.gated(critdim, datadim)
        return self

    def crit_trans_gen(self, critdim, datadim):
        self.attentiongenerator.crit_trans(critdim, datadim)
        return self
