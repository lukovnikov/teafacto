from teafacto.core.base import Block, param, tensorops as T
from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.match import MatchScore, EuclideanDistance, DotDistance
from teafacto.util import argprun, ticktock
import numpy as np
import pickle
from IPython import embed


class Rescal(Block):
    def __init__(self, embdim, numents, numrels, **kw):
        self.A = VectorEmbed(indim=numents, dim=embdim)
        self.R = param((numrels, embdim, embdim), name="rel_embed").glorotuniform()
        self.scorer = DotDistance()
        super(Rescal, self).__init__(**kw)

    def apply(self, sp, o):
        entembs = self.A(sp[:, 0])
        relembs = self.R[sp[:, 1], :, :]
        ret = T.batched_dot(entembs, relembs)
        return self.scorer(ret, self.A(o))


class TransE(Block):
    def __init__(self, embdim, numents, numrels, **kw):
        self.A = VectorEmbed(indim=numents, dim=embdim)
        self.R = VectorEmbed(indim=numrels, dim=embdim)
        self.scorer = EuclideanDistance()
        super(TransE, self).__init__(**kw)

    def apply(self, sp, o):
        entembs = self.A(sp[:, 0])
        relembs = self.R(sp[:, 1])
        ret = entembs + relembs
        return self.scorer(ret, self.A(o)) ** 2


def run(
        embdim=50,
        epochs=10,
        numbats=1000,
        wreg=0.00000000000000001,
        margin=0.,
        rankingloss=False,
        negrate=1,
        lr=0.1,
        mode="rescal",  # also: "transe"
    ):
    wnbinp = "../../data/wn18.bin"
    with open(wnbinp) as wnbin:
        data = pickle.load(wnbin)
    numents = len(data["entities"])
    numrels = len(data["relations"])
    xs = data["train_subs"]
    x = np.asarray(xs)
    print x.shape, numents, numrels, np.max(x, axis=0)

    if mode == "rescal":
        model = Rescal
    elif mode == "transe":
        model = TransE
    else:
        raise Exception("unknown mode")

    scorer = model(embdim, numents, numrels)

    class NegIdxGen(object):
        def __init__(self, rng):
            self.min = 0
            self.max = rng

        def __call__(self, datas, gold):
            return datas, np.random.randint(self.min, self.max, gold.shape).astype("int32")

    # trainer config and training
    obj = lambda p, n: n - p
    if rankingloss:
        obj = lambda p, n: (n - p + margin).clip(0, np.infty)

    nscorer = scorer.nstrain([x[:, [0, 2]], x[:, 1]])\
        .negsamplegen(NegIdxGen(numents)).negrate(negrate).objective(obj)\
        .adagrad(lr=lr).l2(wreg)\
        .train(numbats=numbats, epochs=epochs)

    #scorer.save("rescal.scorer.block")


if __name__ == "__main__":
    argprun(run)