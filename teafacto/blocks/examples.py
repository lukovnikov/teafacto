from teafacto.blocks.core import tensorops as T
from teafacto.blocks.core import *
from teafacto.blocks.datafeed import *
from teafacto.blocks.trainer import *
from teafacto.blocks.util import argparsify
from teafacto.lm import Glove


class Embed(Block):
    def __init__(self, indim=1000, dim=50, **kw):
        super(Embed, self).__init__(**kw)
        self.dim = dim
        self.indim = indim
        self.W = param((indim, dim)).uniform().normalize(axis=1)

    def initinputs(self):
        return [Input(ndim=1, dtype="int32", name="emb_inp")]

    def apply(self, inptensor):
        return self.W[inptensor, :]

class Softmax(Block):
    def apply(self, inptensor): # matrix
        return T.nnet.softmax(inptensor)


class Dummy(Block):
    def __init__(self, indim=1000, dim=50, **kw):
        super(Dummy, self).__init__(**kw)
        self.dim = dim
        self.indim = indim
        self.W = Embed(indim=indim, dim=dim)
        self.O = param((dim, indim)).uniform()

    def initinputs(self):
        return [Input(ndim=1, dtype="int32", name="autoenc_inp")]

    def apply(self, inptensor):
        emb = self.W(inptensor)
        out = T.dot(emb, self.O)
        probs = Softmax()(out)
        return probs


def run(
        epochs=20,
        dim=15,
        vocabsize=2000,
        lr=0.1,
        numbats=100
    ):
    data = np.arange(0, vocabsize).astype("int32")
    ae = Dummy(indim=vocabsize, dim=dim)
    ae  .train([data], data).rmsprop().cross_entropy().autovalidate().cross_entropy().accuracy()\
        .train(numbats=numbats, epochs=epochs)

    pdata = [0, 1, 2]
    pembs = ae.W.predict(pdata)
    print pembs, pembs.shape
    print np.linalg.norm(pembs, axis=1)
    pred = ae.predict(pdata)
    print pred.shape


if __name__ == "__main__":
    run(**argparsify(run))
