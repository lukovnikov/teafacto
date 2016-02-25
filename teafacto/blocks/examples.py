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
        self.W = param((indim, dim), lrmul=1.).uniform().normalize(axis=1)

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
        self.O = param((dim, indim), lrmul=1.).uniform()

    def initinputs(self):
        return [Input(ndim=1, dtype="int32", name="autoenc_inp")]

    def apply(self, inptensor):
        emb = self.W(inptensor)
        out = T.dot(emb, self.O)
        probs = Softmax()(out)
        return probs


def run(
        epochs=100,
        dim=10,
        vocabsize=2000,
        lr=0.02,
        numbats=100
    ):
    lr *= numbats
    data = np.arange(0, vocabsize).astype("int32")
    ae = Dummy(indim=vocabsize, dim=dim)
    err, verr, _, _ = \
        ae  .train([data], data).adadelta(lr=lr).neg_log_prob().validate(5, random=True).neg_log_prob().accuracy()\
        .train(numbats=numbats, epochs=epochs)

    pdata = range(100)
    pembs = ae.W.predict(pdata)
    print np.linalg.norm(pembs, axis=1)
    pred = ae.predict(pdata)
    print pred.shape
    print np.argmax(pred, axis=1)
    #print err, verr


if __name__ == "__main__":
    run(**argparsify(run))
