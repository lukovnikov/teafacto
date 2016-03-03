import pandas as pd
import re, math

from teafacto.blocks.memnet import vec2sec
from teafacto.blocks.rnn import RNNAutoEncoder
from teafacto.blocks.embed import Glove
from teafacto.util import argparsify

''' THIS SCRIPT TESTS TRAINING RNN ENCODER, RNN DECODER AND RNN AUTOENCODER '''


def run2(
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

    block = vec2sec(indim=embdim, innerdim=statedim, vocsize=vocsize, seqlen=data.shape[1])
    block.train([embs], data).seq_neg_log_prob().adadelta(lr=lr)\
         .train(numbats=numbats, epochs=epochs)

def run(
        wreg=0.0,
        epochs=200,
        numbats=20,
        lr=0.1,
        statedim=200,
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
    testneglogprob = 22
    print "%.2f neg log prob for a whole sequence is %.3f prob per slot" % (testneglogprob, math.exp(-testneglogprob/data.shape[1]))

    block = RNNAutoEncoder(vocsize=vocsize, innerdim=statedim, seqlen=data.shape[1])
    block.train([data], data).seq_neg_log_prob().adadelta(lr=lr)\
         .train(numbats=numbats, epochs=epochs)


if __name__ == "__main__":
    run(**argparsify(run))