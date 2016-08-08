import sys, re, os.path, numpy as np
from IPython import embed
from teafacto.util import argprun, tokenize, ticktock
from teafacto.blocks.match import MatchScore, CosineDistance, DotDistance, EuclideanDistance
from teafacto.blocks.lang.wordvec import Glove
from teafacto.blocks.seqproc import SeqEncoder, SimpleSeq2Vec
from teafacto.core.base import Block
import pickle


def run(
        epochs=10,
        numbats=100,
        negrate=1,
        lr=0.1,
        embdim=50,
        encdim=50,
        wreg=0.00005,
        marginloss=False,
        margin=1.,
        cosine=False,
    ):
    tt = ticktock("script")
    # get glove words
    g = Glove(encdim)
    words = g.D.keys()
    maxwordlen = 0
    for word in words:
        maxwordlen = max(maxwordlen, len(word))
    chars = set("".join(words))
    chars.add(" ")
    print "{} words, maxlen {}, {} characters in words".format(len(words), maxwordlen, len(chars))
    # get char word matrix
    chardic = dict(zip(chars, range(len(chars))))
    pickle.dump(chardic, open("glove2c2w.chardic.pkl", "w"))
    charwordmat = np.ones((len(words)+1, maxwordlen), dtype="int32") * (-1.0 if cosine else 1.0)
    charwordmat[0, 0] = chardic[" "]
    for i in range(0, len(words)):
        word = words[i]
        charwordmat[i+1, :len(word)] = [chardic[x] for x in word]
    print charwordmat[0]
    # encode characters
    cwenc = SimpleSeq2Vec(indim=len(chars),
                          inpembdim=embdim,
                          innerdim=encdim/2,
                          maskid=-1,
                          bidir=True)
    dist = CosineDistance() if cosine else EuclideanDistance() #DotDistance()
    scorer = MatchScore(cwenc, g.block, scorer=dist)

    scorer.train([charwordmat, np.arange(len(words)+1)], -np.ones((charwordmat.shape[0],)))\
        .linear_objective().adagrad(lr=lr).l2(wreg)\
        .train(numbats=numbats, epochs=epochs)

    cwenc.save("glove2c2w.block")


    # NEGATIVE SAMPLING ::::::::::::::::::::::::::::::::::
    sys.exit()  # don't

    #embed()

    class NegIdxGen(object):
        def __init__(self, rng):
            self.min = 0
            self.max = rng

        def __call__(self, datas, gold):
            return datas, np.random.randint(self.min, self.max, gold.shape).astype("int32")

    if marginloss:
        obj = lambda p, n: (n-p+margin).clip(0, np.infty)
    else:
        obj = lambda p, n: n - p

    nscorer = scorer.nstrain([charwordmat, np.arange(len(words)+1)])\
        .negsamplegen(NegIdxGen(len(words))).negrate(negrate)\
        .objective(obj).adagrad(lr=lr).l2(wreg)\
        .train(numbats=numbats, epochs=epochs)



if __name__ == "__main__":
    argprun(run)