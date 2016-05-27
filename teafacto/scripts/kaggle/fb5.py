from teafacto.core.base import param, Block, tensorops as T
from teafacto.blocks.basic import Softmax
from teafacto.util import argprun, ticktock as TT
from os.path import expanduser
import numpy as np
from pympler.asizeof import asizeof


def loaddict(p):
    dic = {}
    with open(p) as f:
        for line in f:
            ns = map(int, line[:-1].split("\t"))
            dic[ns[0]] = ns[1]
    revdic = {v: k for k, v in dic.items()}
    return dic, revdic


def loaddata(p, top=np.infty):
    tt = TT("Dataloader")
    traindata = None
    golddata = None
    i = 0
    tt.tick("loading")
    with open(p) as f:
        numsam = 1
        for line in f:
            if traindata is None and golddata is None:  # first line
                numsam, numcol = map(int, line[:-1].split(" "))
                traindata = np.zeros((min(numsam, top), numcol-1)).astype("float32")
                golddata = np.zeros((min(numsam, top),)).astype("int32")
            else:
                ns = line[:-1].split("\t")
                traindata[i, :] = map(float, ns[:-1])
                golddata[i] = int(ns[-1])
                i += 1
                tt.progress(i, numsam, live=True)
            if top is not None and i >= top:
                break
    tt.tock("loaded")
    return traindata, golddata


class SpatialEmb(Block):
    def __init__(self, dim=500, sharpness=1, **kw):
        super(SpatialEmb, self).__init__(**kw)
        self.xes = param((dim,), name="xes").constant(5000.0)
        self.yes = param((dim,), name="yes").constant(5000.0)
        self.divmul = param((), name="divmul").constant(1.0)
        self.e = sharpness

    def rec(self, x, y, div, xes, yes, divmul):     # x: (4,), xes, yes: (outdim, ) accmul: ()
        xdiff = xes - x
        ydiff = yes - y
        dist = T.sqrt(T.sqr(xdiff) + T.sqr(ydiff))
        score = self.distscorefun(dist, div, divmul)
        return score

    def _distscorefun(self, dist, div, divmul):
        a = 1 / (1 + abs((dist / (divmul * div))) ** (2 * self.e))
        b = 1 / (1 + abs((dist / (divmul * div))) ** 2)
        return a * b

    def _sdistscorefun(self, dist, div, divmul):
        return 1 / (1 + T.log(1 + T.exp(dist)))

    def distscorefun(self, dist, div, divmul):
        return -divmul * dist

    def apply(self, x):     # x: (batsize, 4)
        o, _ = T.scan(fn=self.rec, sequences=[x[:, 0], x[:, 1], x[:, 2]], non_sequences=[self.xes, self.yes, self.divmul], outputs_info=None)    # (batsize, outdim)
        #axes = T.tile(x[:, 0], (self.xes.shape[0], 1)).T
        #ayes = T.tile(x[:, 1], (self.xes.shape[0], 1)).T
        #adivs = T.tile(x[:, 2], (self.xes.shape[0], 1)).T
        #bxes = T.tile(self.xes, (x.shape[0], 1))
        #byes = T.tile(self.yes, (x.shape[0], 1))
        #o = self.rec(axes, ayes, adivs, bxes, byes, self.divmul)
        ret = Softmax()(o)
        return ret


def run(dicp="~/dev/kaggle/fb5/pdic.map", datap="~/dev/kaggle/fb5/train.tab", lr=1., numbats=100, epochs=10):
    dic, revdic = loaddict(expanduser(dicp))
    print len(dic)
    traindata, golddata = loaddata(expanduser(datap), top=10000)
    print asizeof(traindata), golddata.dtype
    m = SpatialEmb(dim=len(dic))

    m.train([traindata], golddata).adagrad(lr=lr).cross_entropy()\
        .split_validate(splits=100, random=True).cross_entropy().accuracy()\
        .train(numbats, epochs)



if __name__ == "__main__":
    argprun(run)