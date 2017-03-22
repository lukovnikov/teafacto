from teafacto.blocks.seq.memnn import AutoMorph
from teafacto.blocks.basic import VectorEmbed, SMO, SMOWrap
from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.core import Block, param, T
from teafacto.blocks.word import Glove
from teafacto.blocks.activations import GumbelSoftmax, Softmax, ReLU, Sigmoid, Softplus
from teafacto.util import argprun, ticktock
from IPython import embed
import numpy as np


def loaddata(splits=10):
    g = Glove(50, 10000)
    words = g.D.keys()
    maxlen = max([len(x) for x in words])
    mat = np.zeros((len(words), maxlen), dtype="int32")
    nwo = []
    for i, word in enumerate(words):
        mat[i, :len(word)] = [ord(x) for x in word]
        nwo.append(word)
    wmat = np.arange(0, len(nwo))
    wdic = dict(zip(nwo, range(len(nwo))))
    splitidxs = np.arange(len(wmat))
    np.random.shuffle(splitidxs)
    testidxs = splitidxs[:len(splitidxs)/splits]
    trainidxs = splitidxs[len(splitidxs)/splits:]
    trainmat = mat[trainidxs]
    testmat = mat[testidxs]
    traingold = wmat[trainidxs]
    testgold = wmat[testidxs]
    return (trainmat, traingold), (testmat, testgold), wdic


class DWVAE(Block):
    def __init__(self, enc, encdim, numwords, N=200, K=10, temp=1.0, **kw):
        super(DWVAE, self).__init__(**kw)
        self.enc = enc
        self.numwords = numwords
        self.encdim = encdim
        self.N = N
        self.K = K
        self.temp = temp
        self.a = param((encdim, N*K), name="dv").glorotuniform()
        self.b = param((N*K, self.numwords), name="outlin").glorotuniform()

    def apply(self, x):
        enc = self.enc(x)
        a = T.dot(enc, self.a)      # (batsize, N*K)
        a = Softplus()(a)
        a = a.reshape((a.shape[0], self.N, self.K))
        a_sm = GumbelSoftmax(temperature=self.temp)(a)
        x = a_sm.reshape((a_sm.shape[0], -1))
        x_o = T.dot(x, self.b)
        out = Softmax()(x_o)
        return out



def run(lr=0.5,
        epochs=10,
        numbats=5000,
        charembdim=100,
        temp=5.,
        ):
    tt = ticktock("script")
    tt.tick("loading data")
    (traindata, traingold), (validdata, validgold), wdic = loaddata()
    tt.tock("data loaded")

    # MODEL
    numchars = max(np.max(traindata), np.max(validdata)) + 1
    numwords = max(np.max(traingold), np.max(validgold)) + 1
    charenc = RNNSeqEncoder.fluent()\
        .vectorembedder(numchars, charembdim, maskid=0)\
        .addlayers([200, 200]).make()

    wordemb = VectorEmbed(numwords, 200)

    m = DWVAE(wordemb, 200, numwords, temp=temp)

    m.train([traingold], traingold).adadelta(lr=lr).cross_entropy().accuracy()\
        .validate_on([validgold], validgold).cross_entropy().accuracy()\
        .train(numbats=numbats, epochs=epochs)

    predf = m.predict
    rwd = {v: k for k, v in wdic.items()}
    def play(word):
        x = np.array([[ord(x) for x in word]])
        pred = predf(x)
        print rwd[np.argmax(pred[0])]

    embed()


if __name__ == "__main__":
    argprun(run)
