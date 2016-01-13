import os, numpy as np, pandas as pd
from teafacto.core.utils import ticktock as TT
from IPython import embed

class WordEmb(object):
    def __init__(self):
        self.W = []
        self.D = {}
        self.tt = TT(self.__class__.__name__)

    def load(self, **kw):
        self.tt.tick("loading")
        if "path" not in kw:
            raise Exception("path must be specified")
        else:
            path = kw["path"]
        self.W = []
        i = 0
        for line in open(path):
            ls = line.split(" ")
            word = ls[0]
            self.D[word] = i
            self.W.append(map(lambda x: float(x), ls[1:]))
            i += 1
        self.tt.tock("loaded in list").tick()
        self.W = np.asarray(self.W)
        self.tt.tock("loaded")

    def getvector(self, word):
        return self.W[self.D[word]]

    def getdistance(self, A, B, distance=None):
        if distance is None:
            distance = self.cosine
        return distance(self.getvector(A), self.getvector(B))

    def cosine(self, A, B):
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def __call__(self, A, B):
        return self.getdistance(A, B)

    def __getitem__(self, word):
        return self.getvector(word)


class Glove(WordEmb):
    def __init__(self, dims):
        super(Glove, self).__init__()
        self.load(dims=dims)

    def load(self, **kw):
        if "dims" not in kw:
            raise Exception("dims must be specified")
        else:
            dims = kw["dims"]
        path = os.path.join(os.path.dirname(__file__), "../data/glove/glove.6B.%dd.txt" % dims)
        super(Glove, self).load(path=path)


if __name__ == "__main__":
    glove = Glove(50)
    embed()