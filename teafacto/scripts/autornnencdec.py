import pandas as pd
import re

from teafacto.blocks.memnet import vec2sec
from teafacto.blocks.embed import Glove
from teafacto.util import argparsify

''' THIS SCRIPT TESTS TRAINING RNN ENCODER, RNN DECODER AND RNN AUTOENCODER '''


def run(
        wreg=0.0,
        epochs=20,
        numbats=100,
        lr=0.01,
        dims=27,
        predicton=None,
        statedim = 300,
    ):
    # get words
    embdim = 50
    lm = Glove(embdim, 10000)
    words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
    wldf = pd.DataFrame(map(lambda word: [ord(letter)-96 for letter in word], words)).fillna(0)
    data = wldf.values.astype("int32")
    embs = lm.W[map(lambda x: lm * x, words), :]
    print embs.shape, data.shape
    #embed()
    del wldf

    block = vec2sec(indim=embdim, innerdim=statedim, vocsize=dims, seqlen=data.shape[1])
    block.train([embs], data).seq_neg_log_prob().sgd(lr=lr)\
         .train(numbats=numbats, epochs=epochs)


if __name__ == "__main__":
    run(**argparsify(run))