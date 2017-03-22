from teafacto.blocks.seq.memnn import AutoMorph
from teafacto.blocks.basic import VectorEmbed, SMO, SMOWrap
from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.core import Block, param, T
from teafacto.blocks.word import Glove
from teafacto.blocks.activations import GumbelSoftmax, Softmax, ReLU, Sigmoid
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
    def __init__(self, enc, encdim, numwords, **kw):
        super(DWVAE, self).__init__(**kw)
        self.enc = enc
        self.numwords = numwords
        self.encdim = encdim
        self.numdc = 100
        self.a = param((encdim, self.numdc), name="dv_1").glorotuniform()
        self.b = param((encdim, self.numdc), name="dv_2").glorotuniform()
        self.c = param((encdim, self.numdc), name="dv_3").glorotuniform()
        self.d = param((self.numdc, encdim), name="vd_1").glorotuniform()
        self.e = param((self.numdc, encdim), name="vd_2").glorotuniform()
        self.f = param((self.numdc, encdim), name="vd_3").glorotuniform()
        self.x = param((encdim * 3, self.numwords), name="outlin").glorotuniform()

    def apply(self, x):
        enc = self.enc(x)
        a = T.dot(enc, self.a)
        b = T.dot(enc, self.b)
        c = T.dot(enc, self.c)
        a = ReLU()(a)
        b = ReLU()(b)
        c = ReLU()(c)
        a_sm = GumbelSoftmax()(a)
        b_sm = GumbelSoftmax()(b)
        c_sm = GumbelSoftmax()(c)
        d = T.dot(a_sm, self.d)
        e = T.dot(b_sm, self.e)
        f = T.dot(c_sm, self.f)
        x = T.concatenate([d, e, f], axis=-1)
        x_o = T.dot(x, self.x)
        out = Softmax()(x_o)
        return out



def run(lr=0.5,
        epochs=10,
        numbats=5000,
        charembdim=100,
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

    m = DWVAE(charenc, 200, numwords)

    m.train([traindata], traingold).adadelta(lr=lr).cross_entropy().accuracy()\
        .validate_on([validdata], validgold).cross_entropy().accuracy()\
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
