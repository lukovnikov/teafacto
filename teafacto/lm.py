import os, numpy as np, pandas as pd

from collections import OrderedDict
from teafacto.core.utils import ticktock as TT
from IPython import embed
import theano
from theano import tensor as T

class WordEmb(object): # unknown words are mapped to index 0, their embedding is a zero vector
    def __init__(self, asparam=False):
        self.W = []
        self.D = OrderedDict()
        self.tt = TT(self.__class__.__name__)
        self.isparam = asparam

    @property
    def shape(self):
        return self.W.shape

    def load(self, **kw):
        self.tt.tick("loading")
        if "path" not in kw:
            raise Exception("path must be specified")
        else:
            path = kw["path"]
        self.W = [None] # to be replaced by all zero's
        i = 1
        for line in open(path):
            ls = line.split(" ")
            word = ls[0]
            self.D[word] = i
            self.W.append(map(lambda x: float(x), ls[1:]))
            i += 1
        self.W[0] = np.zeros_like(self.W[1])
        self.tt.tock("loaded in list").tick()
        self.W = np.asarray(self.W)
        self.tt.tock("loaded")

    def d(self, word):
        return self.D[word] if word in self.D else 0

    def __mul__(self, other):
        return self.d(other)

    def __mod__(self, other):
        if isinstance(other, (tuple, list)): # distance
            assert len(other) > 1
            if len(other) == 2:
                return self.getdistance(other[0], other[1])
            else:
                y = other[0]
                return map(lambda x: self.getdistance(y, x), other[1:])
        else:   # embed
            return self.__getitem__(other)

    def getvector(self, word):
        try:
            return self.W[self.D[word]]
        except Exception:
            return None

    def getdistance(self, A, B, distance=None):
        if distance is None:
            distance = self.cosine
        return distance(self.getvector(A), self.getvector(B))

    def cosine(self, A, B):
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def __call__(self, A, B):
        return self.getdistance(A, B)

    def __getitem__(self, word):
        if isinstance(word, theano.Variable): # word should contain indexes
            return self.theanovar[word, :]
        else:
            v = self.getvector(word)
            return v if v is not None else np.zeros_like(self.W[0])

    @property
    def theanovar(self):
        if self._theanovar is None:
            self._theanovar = self.gettheanovariable()
        return self._theanovar

    def gettheanovariable(self):
        return theano.shared(np.asarray(self.W).astype("float32"), name=self.__class__.__name__)

    @property
    def theano(self):
        return self.gettheanovariable()


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