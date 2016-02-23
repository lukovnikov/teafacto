from teafacto.blocks.core import *
from teafacto.blocks.datafeed import *
from teafacto.blocks.trainer import *
from teafacto.core.utils import argparsify

from teafacto.lm import Glove

from theano import tensor as T


class Embed(Block):
    def __init__(self, indim=1000, dim=50, **kw):
        super(Embed, self).__init__(**kw)
        self.dim = dim
        self.indim = indim
        self.W = self.add_param(param((indim, dim)).uniform()).d

    def _apply(self, inptensor):
        return self.W[inptensor, :]

class Softmax(Block):
    def _apply(self, inptensor): # matrix
        return T.nnet.softmax(inptensor)


class AutoEncoder(Block):
    def __init__(self, indim=1000, dim=50, **kw):
        super(AutoEncoder, self).__init__(**kw)
        self.dim = dim
        self.indim = indim
        self.W = Embed(indim=indim, dim=dim)
        self.O = self.add_param(param((dim, indim)).uniform()).d

    def initinputs(self):
        return [Input(ndim=1, dtype="int32")]

    def apply(self, inptensor):
        emb = self.W(inptensor)
        out = wrap(lambda x: T.dot(x, self.O), self.O)(emb)
        probs = Softmax()(out)
        return probs


def run(
        epochs=100,
        dim=50,
        vocabsize=2000,
        lr=1,
        numbats=100
    ):
    data = np.arange(0, vocabsize).astype("int32")
    ae = AutoEncoder(indim=vocabsize, dim=dim)
    ae  .train(data, data).rmsprop().cross_entropy().validate(5, True).cross_entropy()\
        .train(numbats=numbats, epochs=epochs)

    ae.predict([0])

    return epochs


if __name__ == "__main__":
    run(**argparsify(run))
