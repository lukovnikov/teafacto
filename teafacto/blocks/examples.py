from teafacto.blocks.core import *
from teafacto.blocks.datafeed import *
from teafacto.blocks.trainer import *
from teafacto.core.utils import argparsify
from teafacto.lm import Glove
from teafacto.blocks.core import tensorops as T


class Embed(Block):
    def __init__(self, indim=1000, dim=50, **kw):
        super(Embed, self).__init__(**kw)
        self.dim = dim
        self.indim = indim
        self.W = param((indim, dim)).uniform()

    def apply(self, inptensor):
        return self.W[inptensor, :]

class Softmax(Block):
    def apply(self, inptensor): # matrix
        return T.nnet.softmax(inptensor)


class AutoEncoder(Block):
    def __init__(self, indim=1000, dim=50, **kw):
        super(AutoEncoder, self).__init__(**kw)
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
        epochs=100,
        dim=20,
        vocabsize=2000,
        lr=1,
        numbats=100
    ):
    data = np.arange(0, vocabsize).astype("int32")
    ae = AutoEncoder(indim=vocabsize, dim=dim)
    ae  .train([data], data).rmsprop().cross_entropy().validate(5, random=True).cross_entropy()\
        .train(numbats=numbats, epochs=epochs)

    pred = ae.predict([0, 1, 2])
    print pred.shape
    print pred


if __name__ == "__main__":
    run(**argparsify(run))
