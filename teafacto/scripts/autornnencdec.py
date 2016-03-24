import pandas as pd, numpy as np
import re, math
from IPython import embed

from teafacto.core.base import Block, tensorops as T
from teafacto.core.stack import stack
from teafacto.blocks.rnn import RNNAutoEncoder, SeqDecoder, SeqEncoder, InConcatCRex, RewAttRNNEncDecoder, FwdAttRNNEncDecoder, RewAttSumDecoder, FwdAttSumDecoder
from teafacto.blocks.rnu import GRU
from teafacto.blocks.basic import IdxToOneHot, VectorEmbed, Softmax, MatDot as Lin
from teafacto.blocks.embed import Glove
from teafacto.util import argprun


class idx2seq(Block):
    def __init__(self, encdim=44, invocsize=500, outvocsize=27, innerdim=300, seqlen=20, **kw):
        super(idx2seq, self).__init__(**kw)
        self.invocsize = invocsize
        self.outvocsize = outvocsize
        self.innerdim = innerdim
        self.seqlen = seqlen
        self.encdim = encdim
        self.emb = VectorEmbed(indim=self.invocsize, dim=self.encdim, normalize=False)
        self.dec = SeqDecoder(IdxToOneHot(self.outvocsize),
                              InConcatCRex(
                                  GRU(dim=self.outvocsize+self.encdim, innerdim=self.innerdim, nobias=True),
                                  outdim=self.innerdim
                                )
                              )

    def apply(self, idxs, seq): # seq: (batsize, seqlen)
        return self.dec(self.emb(idxs), seq)


class seq2idx(Block):
    def __init__(self, invocsize=27, outvocsize=500, innerdim=300, **kw):
        super(seq2idx, self).__init__(**kw)
        self.invocsize = invocsize
        self.outvocsize = outvocsize
        self.innerdim = innerdim
        self.enc = SeqEncoder(
            VectorEmbed(indim=self.invocsize, dim=self.invocsize),
            GRU(dim=self.invocsize, innerdim=self.innerdim)
        )
        self.outlin = Lin(indim=self.innerdim, dim=self.outvocsize)

    def apply(self, seqs):
        return Softmax()(self.outlin(self.enc(seqs)))

''' THIS SCRIPT TESTS TRAINING RNN ENCODER, RNN DECODER AND RNN AUTOENCODER '''


def word2int(word):
    return [ord(letter)-96 if letter is not " " else 0 for letter in word]


def words2ints(words):
    wldf = pd.DataFrame(map(word2int, words)).fillna(0)
    data = wldf.values.astype("int32")
    del wldf
    return data


def int2word(ints):
    chars = [chr(i+96) if i > 0 else " " for i in ints]
    return "".join(chars)


def ints2words(ints):
    return [int2word(x) for x in ints]


def run_RNNAutoEncoder(         # works after refactoring
        wreg=0.000001,
        epochs=50,
        numbats=20,
        lr=0.1,
        statedim=70,
        encdim=70
    ):
    # get words
    vocsize = 27
    embdim = 50
    lm = Glove(embdim, 1000)
    words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
    #wldf = pd.DataFrame(map(word2int, words)).fillna(0)
    #data = wldf.values.astype("int32")
    data = words2ints(words)
    sdata = shiftdata(data)
    embs = lm.W[map(lambda x: lm * x, words), :]
    print embs.shape, data.shape
    #embed()
    print "random seq neg log prob %.3f" % math.log(vocsize**data.shape[1])
    testneglogprob = 17
    print "%.2f neg log prob for a whole sequence is %.3f prob per slot" % (testneglogprob, math.exp(-testneglogprob*1./data.shape[1]))

    testpred = ["the", "alias", "mock", "test", "stalin"]
    testpred = words2ints(testpred)

    block = RNNAutoEncoder(vocsize=vocsize, encdim=70, innerdim=statedim, seqlen=data.shape[1])
    block.train([data, sdata], data).seq_neg_log_prob().grad_total_norm(1.0).adagrad(lr=lr).l2(wreg)\
         .autovalidate().seq_accuracy().validinter(4)\
         .train(numbats=numbats, epochs=epochs)

    pred = block.predict(testpred, shiftdata(testpred))
    print ints2words(np.argmax(pred, axis=2))


def run_attentionseqdecoder(        # seems to work
        wreg=0.00001,       # TODO: regularization other than 0.0001 first stagnates, then goes down
        epochs=100,
        numbats=20,
        lr=0.1,
        statedim=50,
        encdim=50,
        attdim=50
    ):

    # get words
    vocsize = 27
    embdim = 50
    lm = Glove(embdim, 2000)
    allwords = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
    words = allwords[:1000]
    vwords = allwords[1000:]
    data = words2ints(words)
    sdata = shiftdata(data)
    vdata = words2ints(vwords)
    svdata = shiftdata(vdata)
    print "random seq neg log prob %.3f" % math.log(vocsize**data.shape[1])
    testneglogprob = 17
    print "%.2f neg log prob for a whole sequence is %.3f prob per slot" % (testneglogprob, math.exp(-testneglogprob*1./data.shape[1]))

    testpred = ["the", "alias", "mock", "test", "stalin", "allahuakbar", "python", "pythonista", " "*(data.shape[1])]
    testpred = words2ints(testpred)
    print testpred

    block = FwdAttRNNEncDecoder(vocsize=vocsize, outvocsize=vocsize, encdim=encdim, innerdim=statedim, attdim=attdim)
    block.train([data, sdata], data).seq_neg_log_prob().grad_total_norm(1.0).adagrad(lr=lr).l2(wreg)\
         .validate_on([vdata, svdata], vdata).seq_accuracy().validinter(4)\
         .train(numbats=numbats, epochs=epochs)

    pred = block.predict(testpred, shiftdata(testpred))
    print ints2words(np.argmax(pred, axis=2))

    embed()


def run_idx2seq(        # works after refactor
        wreg=0.000001,
        epochs=150,
        numbats=10,
        lr=0.1,
        statedim=70,
        encdim=100,
    ):
    # get words
    numchars = 27
    embdim = 50
    lm = Glove(embdim, 1000)
    words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
    data = words2ints(words)
    sdata = shiftdata(data)
    wordidxs = np.arange(0, len(words))
    numwords = wordidxs.shape[0]
    print "random seq neg log prob %.3f" % math.log(numchars**data.shape[1])
    testneglogprob = 17
    print "%.2f neg log prob for a whole sequence is %.3f prob per slot" % (testneglogprob, math.exp(-testneglogprob*1./data.shape[1]))

    testpred = wordidxs[:15]
    testdata = data[:15]
    testsdata = sdata[:15]
    print testpred
    print testdata
    print testsdata
    #testpred = words2ints(testpred)

    block = idx2seq(encdim=encdim, invocsize=numwords, outvocsize=numchars, innerdim=statedim, seqlen=data.shape[1])
    print np.argmax(block.predict(testpred, testsdata), axis=2)
    print block.output.allparams
    block.train([wordidxs, sdata], data).seq_neg_log_prob().grad_total_norm(0.5).adagrad(lr=lr).l2(wreg)\
         .autovalidate().seq_accuracy().validinter(5)\
         .train(numbats=numbats, epochs=epochs)

    pred = block.predict(testpred, testsdata)
    for word in ints2words(np.argmax(pred, axis=2)):
        print word
    embed()


def shiftdata(x):
    return np.concatenate([np.zeros_like(x[:, 0:1]), x[:, :-1]], axis=1)


def run_seq2idx(        # works after refactoring (with adagrad)
        wreg=0.0,
        epochs=75,
        numbats=20,
        lr=1.,
        statedim=100,
    ):
    # get words
    numchars = 27
    embdim = 50
    lm = Glove(embdim, 1000)
    words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
    #wldf = pd.DataFrame(map(word2int, words)).fillna(0)
    #data = wldf.values.astype("int32")
    data = words2ints(words)
    #embs = lm.W[map(lambda x: lm * x, words), :]
    #embed()
    wordidxs = np.arange(0, len(words))
    print wordidxs[:5]
    print data[:5]
    print words[:5]
    numwords = wordidxs.shape[0]
    print "random seq neg log prob %.3f" % math.log(numchars**data.shape[1])
    testneglogprob = 17
    print "%.2f neg log prob for a whole sequence is %.3f prob per slot" % (testneglogprob, math.exp(-testneglogprob*1./data.shape[1]))

    testpred = ["the", "of", "to", "their", "in"]
    testpred = words2ints(testpred)
    print words[:5], words[35]
    ####testpred = np.eye(numchars, numchars)[testpred, :]

    wordidxsonehot = np.eye(numwords, numwords)[wordidxs, :]

    ####data = np.eye(numchars, numchars)[data, :]

    block = seq2idx(invocsize=numchars, outvocsize=numwords, innerdim=statedim)
    '''gru = GRU(innerdim=statedim, dim=numchars)
    lin = Lin(indim=statedim, dim=numwords)
    lin2 = Lin(indim=numwords, dim=numwords)
    block = asblock(lambda x: Softmax()(lin(gru(x)[:, -1, :])))'''
    ###block = asblock(lambda x: Softmax()(lin2(x)))
    '''
    print testpred
    probepred = np.argmax(block.predict(testpred), axis=1)
    print probepred

    for p in block.output.allparams:
        print p
    '''
    block.train([data], wordidxs).neg_log_prob().adagrad(lr=lr).autovalidate().accuracy().validinter(5)\
         .train(numbats=numbats, epochs=epochs)

    #embed()
    pred = block.predict(testpred)
    print pred.shape
    print np.argmax(pred, axis=1)
    #'''




if __name__ == "__main__":
    argprun(run_attentionseqdecoder)
    #print ints2words(np.asarray([[20,8,5,0,0,0], [1,2,3,0,0,0]]))