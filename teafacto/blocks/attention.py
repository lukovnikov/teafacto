from teafacto.blocks.basic import MatDot as Lin, Softmax, VectorEmbed, IdxToOneHot
from teafacto.core.base import Block, param, Val
from teafacto.core.base import tensorops as T
import numpy as np
import theano, theano.tensor


class Attention(Block):
    '''
    Block wraps both an AttentionGenerator and AttentionConsumer.
    '''
    def __init__(self, attentiongenerator, attentionconsumer, **kw):
        super(Attention, self).__init__(**kw)
        self.attentiongenerator = attentiongenerator
        self.attentionconsumer = attentionconsumer

    def apply(self, criterion, data):
        attention = self.attentiongenerator(criterion, data)
        return self.attentionconsumer(data, attention)


############################## ATTENTION GENERATORS ###############################

class AttentionGenerator(Block):
    '''
     An attention block takes as *data input* a data structure (Var) to whose elements it will assign a weight,
     based on the *criterion input* Var.
     The *output* of this block for one element of the input should be a float between 0.0 and 1.0.
     In the basic setup, a single element is represented by a vector of features, for which one float (the weight) is returned.
     But it should also be possible to consider elements to be higher-mode tensors to, for example, assign weights to
     matrices in a sequence.

     So the attention block itself defines what it understands under one element of the whole
     *direct* input (vector/matrix/...) and how to use the *criterion* input to get scalar weights for each element.

     Given **M**, the number of dimensions in the data input, and **N**, the number of dimensions of what is considered one
     element of data input, the dimensions of the output are **M-N**. There are no principal restrictions on what the
     dimensionality of the *criterion input*.

     For example, consider a sequence of words that is represented by a Var with the following shape: (num_words, num_feats).
     If one word is considered one element, the element dimensions are (num_feats,), thus the output dimension is (num_words,).
     Alternatively, consider the same sequence of words, except this time we go to character level with the following
     shape: (num_words, num_chars_per_word, num_chars), using one-hot character encoding. In this case, we might consider
     the element to be a matrix of shape (num_chars_per_word, num_chars). Then, the output dimension is still (num_words,).
     However, if we considered a letter as a single element (shape: (num_chars,)), the output would have shape (num_words, num_chars_per_word).
    '''
    def __init__(self, indim=50, attdim=50, memdim=50, **kw):
        """
        :param indim:       dimension of the input (criterion vectors)
        :param attdim:      inner dimension of attention generator
        :param memdim:      dimension of memory elements
        """
        super(AttentionGenerator, self).__init__(**kw)
        self.indim = indim
        self.attdim = attdim
        self.memdim = memdim

    def apply(self, criterion, data):   # criterion: (batsize, indim), data: (batsize, seqlen, memdim)
        raise NotImplementedError("use subclass")


class DotprodAttGen(AttentionGenerator):
    """
    indim and memdim should be the same, attdim is not used
    no parameters here
    """
    def apply(self, criterion, data):
        return T.batched_dot(data, criterion)


class LinearSumAttentionGenerator(AttentionGenerator):    # simple feedforward
    def __init__(self, **kw):
        super(LinearSumAttentionGenerator, self).__init__(**kw)
        self.W = param((self.indim, self.attdim), name="attention_ff").uniform()

    def apply(self, criterion, data):   # data is (batsize, seqlen, elem_dim)
        def rec(x_t, crit):     # x_t is (batsize, elem_dim), crit is (batsize, crit_dim)
            ret = T.dot(T.concatenate([x_t, crit], axis=1), self.W)     # (batsize, innerdim)
            return T.sum(ret, axis=1)       # (batsize, )
        o, _ = T.scan(fn=rec, sequences=data.dimswap(1, 0), non_sequences=criterion)    # o is (seqlen, batsize)
        return Softmax()(o.dimswap(1, 0))       # returns (batsize, seqlen), softmaxed on seqlen


class LinearGateAttentionGenerator(AttentionGenerator):
    def __init__(self, **kw):
        super(LinearGateAttentionGenerator, self).__init__(**kw)
        self.W = param((self.indim, self.attdim), name="attention_ff").uniform()
        self.U = param((self.attdim,), name="attention_agg").uniform()

    def apply(self, criterion, data):   # criterion: (batsize, crit_dim), data: (batsize, seqlen, datadim)
        def rec(x_t, crit):
            combo = self._get_combo(x_t, crit)  # (batsize, crit_dim + datadim)
            trans = T.dot(combo, self.W)        # (batsize, innerdim)
            trans = T.tanh(trans)                                       # apply tanh
            ret = T.dot(trans, self.U)                                  # (batsize, )
            return T.nnet.sigmoid(ret)                                  # apply sigmoid
        o, _ = T.scan(fn=rec, sequences=data.dimswap(1, 0), non_sequences=criterion)
        return o.dimswap(1, 0)

    def _get_combo(self, x_t, crit):
        return T.concatenate([x_t, crit], axis=1)


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
        def rec(x_t, att_t, acc):       # x_t: (batsize, elem_dim), att_t: (batsize, ), acc: (batsize, elem_dim)
            acc += T.batched_dot(x_t, att_t)
            return acc  # (batsize, elem_dim)
        o, _ = T.scan(fn=rec, sequences=[data.dimswap(1, 0), weights.T], outputs_info=T.zeros((data.shape[0], data.shape[2])))
        return o[-1, :, :]


class ArgmaxAttCon(AttentionConsumer):
    def apply(self, data, weights):
        bestidx = T.argmax(weights, axis=1)
        return data[T.arange(bestidx.shape[0]), bestidx, :]

