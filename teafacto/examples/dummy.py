from teafacto.core.base import *
from teafacto.core.base import tensorops as T
from teafacto.core.trainer import *
from teafacto.blocks.basic import VectorEmbed, Softmax
from teafacto.util import argparsify


class Dummy(Block):
    def __init__(self, indim=1000, dim=50, outdim=None, normalize=False, **kw):
        super(Dummy, self).__init__(**kw)
        self.dim = dim
        self.indim = indim
        self.W = VectorEmbed(indim=indim, dim=dim, normalize=normalize)
        self.outdim = indim if outdim is None else outdim
        self.O = param((dim, self.outdim), lrmul=1.).glorotuniform()

    def apply(self, inptensor):
        emb = self.W(inptensor)
        out = T.dot(emb, self.O)
        probs = Softmax()(out)
        return probs


def run(
        epochs=1,
        dim=10,
        vocabsize=2000,
        lr=0.02,
        numbats=100
    ):
    lr *= numbats
    data = np.arange(0, vocabsize).astype("int32")
    ae = Dummy(indim=vocabsize, dim=dim)
    ae = ae.train([data], data).adadelta(lr=lr).dlr_thresh().cross_entropy()\
            .split_validate(5, random=True).cross_entropy().accuracy().autosave\
        .train(numbats=numbats, epochs=epochs)

    pdata = range(100)
    pembs = ae.W.predict(pdata)
    #print np.linalg.norm(pembs, axis=1)
    pred = ae.predict(pdata)
    print pred.shape
    #print np.argmax(pred, axis=1)
    #print err, verr
    return pred


if __name__ == "__main__":
    run(**argparsify(run))
