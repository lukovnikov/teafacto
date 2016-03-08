import pandas as pd, numpy as np
import re, math
from IPython import embed

from teafacto.blocks.attention import vec2seq, idx2seq, seq2idx
from teafacto.core.stack import stack
from teafacto.blocks.rnn import RNNAutoEncoder
from teafacto.blocks.rnu import GRU
from teafacto.blocks.basic import IdxToOneHot, VectorEmbed, Softmax, MatDot as Lin
from teafacto.blocks.embed import Glove
from teafacto.core.base import asblock
from teafacto.util import argparsify

''' THIS SCRIPT TESTS TRAINING RNN ENCODER, RNN DECODER AND RNN AUTOENCODER '''

def word2int(word):
    return [ord(letter)-96 for letter in word]

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

def run1(
        wreg=0.0,
        epochs=20,
        numbats=20,
        lr=0.1,
        statedim=50,
    ):
    # get words
    vocsize = 27
    embdim = 50
    lm = Glove(embdim, 1000)
    words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
    wldf = pd.DataFrame(map(lambda word: [ord(letter)-96 for letter in word], words)).fillna(0)
    data = wldf.values.astype("int32")
    embs = lm.W[map(lambda x: lm * x, words), :]
    print embs.shape, data.shape
    #embed()
    del wldf
    print "random seq neg log prob %.3f" % math.log(vocsize**data.shape[1])

    block = vec2seq(indim=embdim, innerdim=statedim, vocsize=vocsize, seqlen=data.shape[1])
    block.train([embs], data).seq_neg_log_prob().adadelta(lr=lr)\
         .train(numbats=numbats, epochs=epochs)


def run2(
        wreg=0.0,
        epochs=200,
        numbats=20,
        lr=0.003,
        statedim=200,
    ):
    # get words
    vocsize = 27
    embdim = 50
    lm = Glove(embdim, 1000)
    words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
    #wldf = pd.DataFrame(map(word2int, words)).fillna(0)
    #data = wldf.values.astype("int32")
    data = words2ints(words)
    embs = lm.W[map(lambda x: lm * x, words), :]
    print embs.shape, data.shape
    #embed()
    print "random seq neg log prob %.3f" % math.log(vocsize**data.shape[1])
    testneglogprob = 17
    print "%.2f neg log prob for a whole sequence is %.3f prob per slot" % (testneglogprob, math.exp(-testneglogprob*1./data.shape[1]))

    testpred = ["the", "alias", "mock", "test", "stalin"]
    testpred = words2ints(testpred)

    block = RNNAutoEncoder(vocsize=vocsize, innerdim=statedim, seqlen=data.shape[1])
    block.train([data], data).seq_neg_log_prob().sgd(lr=lr)\
         .train(numbats=numbats, epochs=epochs)

    pred = block.predict(testpred)
    print ints2words(np.argmax(pred, axis=2))


def runidx2seq(
        wreg=0.001,
        epochs=100,
        numbats=10,
        lr=0.1,
        statedim=70,
        encdim=70,
    ):
    # get words
    numchars = 27
    embdim = 50
    lm = Glove(embdim, 1000)
    words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
    data = words2ints(words)
    wordidxs = np.arange(0, len(words))
    print wordidxs[:15]
    print data[:15]
    numwords = wordidxs.shape[0]
    print "random seq neg log prob %.3f" % math.log(numchars**data.shape[1])
    testneglogprob = 17
    print "%.2f neg log prob for a whole sequence is %.3f prob per slot" % (testneglogprob, math.exp(-testneglogprob*1./data.shape[1]))

    testpred = wordidxs[:15]
    #testpred = words2ints(testpred)

    block = idx2seq(encdim=encdim, invocsize=numwords, outvocsize=numchars, innerdim=statedim, seqlen=data.shape[1])
    print np.argmax(block.predict(testpred), axis=2)
    print block.output.allparams
    block.train([wordidxs], data).seq_neg_log_prob().grad_total_norm(0.5).adagrad(lr=lr).l2(wreg)\
         .autovalidate().seq_accuracy().validinter(5)\
         .train(numbats=numbats, epochs=epochs)

    pred = block.predict(testpred)
    print np.argmax(pred, axis=2)
    embed()


def runidx2firstletter(
        wreg=0.0,
        epochs=1000,
        numbats=10,
        lr=0.1,
        statedim=100,
        encdim=100,
    ):
    # get words
    numchars = 27
    embdim = 50
    lm = Glove(embdim, 1000)
    words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
    data = words2ints(words)
    wordidxs = np.arange(0, len(words))
    print wordidxs[:15]
    print data[:15]
    numwords = wordidxs.shape[0]
    print "random seq neg log prob %.3f" % math.log(numchars**data.shape[1])
    testneglogprob = 17
    print "%.2f neg log prob for a whole sequence is %.3f prob per slot" % (testneglogprob, math.exp(-testneglogprob*1./data.shape[1]))

    testpred = wordidxs[:15]
    #testpred = words2ints(testpred)

    block = stack(VectorEmbed(indim=numwords, dim=encdim), Lin(indim=encdim, dim=numchars), Softmax())
    print np.argmax(block.predict(testpred), axis=1)
    print block.output.allparams
    block.train([wordidxs], data[:, 0]).neg_log_prob().sgd(lr=lr)\
         .train(numbats=numbats, epochs=epochs)

    pred = block.predict(testpred)
    print np.argmax(pred, axis=1)
    print data[:15, 0]
    embed()


def run(
        wreg=0.0,
        epochs=300,
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
    block.train([data], wordidxs).neg_log_prob().sgd(lr=lr).autovalidate().accuracy().validinter(5)\
         .train(numbats=numbats, epochs=epochs)

    #embed()
    pred = block.predict(testpred)
    print pred.shape
    print np.argmax(pred, axis=1)
    #'''




if __name__ == "__main__":
    runidx2seq(**argparsify(runidx2seq()))
    #print ints2words(np.asarray([[20,8,5,0,0,0], [1,2,3,0,0,0]]))