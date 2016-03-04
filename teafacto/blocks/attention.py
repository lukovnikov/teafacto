from teafacto.blocks.basic import MatDot as Lin, Softmax, VectorEmbed, IdxToOneHot
from teafacto.blocks.rnn import RNNDecoder, RNNEncoder
from teafacto.blocks.rnu import GRU
from teafacto.core.base import Block


class vec2seq(Block):
    def __init__(self, indim=50, innerdim=300, seqlen=20, vocsize=27, **kw):
        super(vec2seq, self).__init__(**kw)
        self.indim = indim
        self.innerdim=innerdim
        self.seqlen = seqlen
        self.vocsize = vocsize
        self.lin = Lin(indim=self.indim, dim=self.innerdim)
        self.dec = RNNDecoder(                                  # IdxToOneHot inserted automatically
            GRU(dim=self.vocsize, innerdim=self.innerdim),      # the decoding RNU
            Lin(indim=self.innerdim, dim=self.vocsize),         # transforms from RNU inner dims to vocabulary
            Softmax(),                                          # softmax
                indim=self.vocsize, seqlen=self.seqlen)

    def apply(self, vec):
        return self.dec(self.lin(vec))


class idx2seq(Block):
    def __init__(self, invocsize=500, outvocsize=27, innerdim=300, seqlen=20, **kw):
        super(idx2seq, self).__init__(**kw)
        self.invocsize = invocsize
        self.outvocsize = outvocsize
        self.innerdim = innerdim
        self.seqlen = seqlen
        self.emb = VectorEmbed(indim=self.invocsize, dim=innerdim)
        self.dec = RNNDecoder(
            GRU(dim=self.outvocsize, innerdim=self.innerdim),
            Lin(indim=self.innerdim, dim=self.outvocsize),
            Softmax(),
                indim=self.outvocsize, seqlen=self.seqlen)

    def apply(self, idx):
        return self.dec(self.emb(idx))


class seq2idx(Block):
    def __init__(self, invocsize=27, outvocsize=500, innerdim=300, **kw):
        super(seq2idx, self).__init__(**kw)
        self.invocsize = invocsize
        self.outvocsize = outvocsize
        self.innerdim = innerdim
        self.enc = RNNEncoder(
            VectorEmbed(indim=self.invocsize, dim=self.invocsize),
            GRU(dim=self.invocsize, innerdim=self.innerdim)
        )
        self.outlin = Lin(indim=self.innerdim, dim=self.outvocsize)

    def apply(self, seqs):
        return Softmax()(self.outlin(self.enc(seqs)))



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


class AttentionConsumer(Block):
    '''
     A block that consumes some data and associated attention weights. Subclass this for attention-consuming blocks.
    '''
    def apply(self, attention, *vars):
        raise NotImplementedError("use subclass")


class AttentionCombo(Block):
    '''
    Block wraps both an AttentionGenerator and AttentionConsumer.
    '''


class RecurrentAttentionWrap(AttentionConsumer):
    '''
     Wraps a recurrent block to consume both input and associated attention weights. The wrapped block should be able to
     consume the input. The wrapper modulates between the normal current output of the wrapped block and the previous output,
       based on the attention weight for that element.
    '''