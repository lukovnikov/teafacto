from collections import OrderedDict

import numpy as np
import os

from teafacto.core.base import Block, Val
from teafacto.blocks.basic import VectorEmbed
from teafacto.util import ticktock as TT, isnumber, isstring


class WordEmbBase(object):
    def __init__(self, worddic, **kw):
        super(WordEmbBase, self).__init__(**kw)
        self.D = OrderedDict() if worddic is None else worddic

    # region NON-BLOCK API :::::::::::::::::::::::::::::::::::::
    def getindex(self, word):
        return self.D[word] if word in self.D else 0

    def __mul__(self, other):
        return self.getindex(other)

    def __contains__(self, word):
        return word in self.D

    def getvector(self, word):
        try:
            if isstring(word):
                return self.w[self.idtrans(self.D[word])]
            elif isnumber(word):
                return self.w[self.idtrans(word), :]
        except Exception:
            return None

    def idtrans(self, x):
        return x

    def __getitem__(self, word):
        v = self.getvector(word)
        return v if v is not None else self.w[0, :]

    @property
    def w(self):
        return self.W.d.get_value()

    @property
    def shape(self):
        return self.W.shape

    def cosine(self, A, B):
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def getdistance(self, A, B, distance=None):
        if distance is None:
            distance = self.cosine
        return distance(self.getvector(A), self.getvector(B))

    def __mod__(self, other):
        if isinstance(other, (tuple, list)):  # distance
            assert len(other) > 1
            if len(other) == 2:
                return self.getdistance(other[0], other[1])
            else:
                y = other[0]
                return map(lambda x: self.getdistance(y, x), other[1:])
        else:  # embed
            return self.__getitem__(other)
    # endregion

    @property
    def block(self):
        return self


class WordEmb(WordEmbBase, VectorEmbed): # unknown words are mapped to index 0, their embedding is a zero vector
    """ is a VectorEmbed with a dictionary to map words to ids """
    def __init__(self, dim=50, indim=1000, value=None, worddic=None,
                 normalize=False, trainfrac=1.0, init=None, **kw):
        if isstring(value):     # path
            assert(init is None and worddic is None)
            value, worddic = self.loadvalue(value, dim, indim=indim)
        super(WordEmb, self).__init__(indim=indim, dim=dim, value=value,
                                      normalize=normalize, worddic=worddic,
                                      trainfrac=trainfrac, init=init, **kw)

    def adapt(self, wdic):
        return AdaptedWordEmb(self, wdic)

    def loadvalue(self, path, dim, indim=None):
        tt = TT(self.__class__.__name__)
        tt.tick()
        W = [np.zeros((1, dim))]
        D = OrderedDict()
        i = 1
        for line in open(path):
            if indim is not None and i >= (indim+1):
                break
            ls = line.split(" ")
            word = ls[0]
            D[word] = i
            W.append(np.asarray([map(lambda x: float(x), ls[1:])]))
            i += 1
        W = np.concatenate(W, axis=0)
        tt.tock("loaded")
        return W, D


class AdaptedWordEmb(WordEmbBase, Block):
    def __init__(self, wordemb, wdic, **kw):
        D = wordemb.D
        super(AdaptedWordEmb, self).__init__(wdic, **kw)
        self.inner = wordemb
        self.ad = {v: D[k] if k in D else 0 for k, v in wdic.items()}

        valval = np.zeros((max(self.ad.keys()) + 1,), dtype="int32")
        for i in range(valval.shape[0]):
            valval[i] = self.ad[i] if i in self.ad else 0
        self.adb = Val(valval)

    @property
    def W(self):
        return self.inner.W

    def idtrans(self, x):
        return self.ad[x]

    def apply(self, x):
        x = self.adb[x]
        return self.inner(x)


class Glove(WordEmb):
    defaultpath = "../../../data/glove/glove.6B.%dd.txt"

    def __init__(self, dim, vocabsize=None, path=None, **kw):     # if dim=None, load all
        path = self.defaultpath if path is None else path
        relpath = path % dim
        path = os.path.join(os.path.dirname(__file__), relpath)
        super(Glove, self).__init__(dim=dim, indim=vocabsize, value=path, **kw)
