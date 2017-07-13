from collections import OrderedDict

import numpy as np
import os, pickle as pkl
from IPython import embed

from teafacto.core.base import Block, Val, tensorops as T
from teafacto.blocks.basic import VectorEmbed, Embedder, Switch
from teafacto.util import ticktock as TT, isnumber, isstring
from teafacto.blocks.seq.enc import SimpleSeqStar2Vec


class WordEmbBase(object):
    masktoken = "<MASK>"
    raretoken = "<RARE>"

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
                return self.w[self.D[word]]
            elif isnumber(word):
                return self.w[word, :]
        except Exception:
            return None

    def __getitem__(self, word):
        v = self.getvector(word)
        return v if v is not None else self.w[0, :]

    @property
    def w(self):
        return self.W.d.get_value()

    @property
    def shape(self):
        return self.w.shape

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


class WordEmb(WordEmbBase, VectorEmbed):
    """ is a VectorEmbed with a dictionary to map words to ids """
    def __init__(self, dim=50, value=None, worddic=None,
                 normalize=False, trainfrac=1.0, init=None,
                 **kw):
        assert(worddic is not None)     # always needs a dictionary
        wdvals = worddic.values()
        assert(min(wdvals) >= 0)     # word ids must be positive non-zero

        # extract maskid and rareid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        rareid = worddic[self.raretoken] if self.raretoken in worddic else None

        indim = max(worddic.values())+1        # to init from worddic
        super(WordEmb, self).__init__(indim=indim, dim=dim, value=value,
                                      normalize=normalize, worddic=worddic,
                                      trainfrac=trainfrac, init=init,
                                      maskid=maskid,
                                      **kw)

    def adapt(self, wdic):      # adapts to given word-idx dictionary
        return AdaptedWordEmb(self, wdic)

    def override(self, wordemb, which=None):    # uses override vectors instead of base vectors if word in override dictionary
        return NewOverriddenWordEmb(self, wordemb, which=which)

    def augment(self, wordemb):
        return AugmentedWordEmb(self, wordemb)


class AdaptedWordEmb(WordEmb):  # adapt to given dictionary, map extra words to rare
    def __init__(self, wordemb, wdic, **kw):
        D = wordemb.D
        assert(wordemb.raretoken in D)     # must have rareid in D to map extra words to it
        super(AdaptedWordEmb, self).__init__(worddic=wdic, value=False,
                dim=wordemb.outdim, normalize=wordemb.normalize,
                trainfrac=wordemb.trainfrac, **kw)
        self.inner = wordemb

        self.ad = {v: D[k] if k in D else D[self.raretoken]
                   for k, v in wdic.items()}

        valval = np.zeros((max(self.ad.keys()) + 1,), dtype="int32")
        for i in range(valval.shape[0]):
            valval[i] = self.ad[i] if i in self.ad else 0
        self.adb = Val(valval)

    @property
    def w(self):
        return self.inner.W.d.get_value()[self.adb.d.get_value()]

    def apply(self, inp):
        x = self.adb[inp]
        ret = self.inner(x)
        self._maskfrom(ret, inp)
        return ret


class OverriddenWordEmb(WordEmb): # TODO: RARE TOKEN MGMT
    """
    Overrides every word from base's dictionary with override's vectors,
    if in override's dictionary
    """
    def __init__(self, base, override, **kw):
        assert(base.outdim == override.outdim)
        super(OverriddenWordEmb, self).__init__(worddic=base.D, value=False,
                dim=base.outdim, normalize=base.normalize,
                trainfrac=base.trainfrac, **kw)

        self.base = base
        self.override = override
        self.ad = {v: override.D[k] if k in override.D else 0 for k, v in base.D.items()}
        valval = np.zeros((max(self.ad.keys()) + 1,), dtype="int32")
        for i in range(valval.shape[0]):
            valval[i] = self.ad[i] if i in self.ad else 0
        self.adb = Val(valval)

    def apply(self, x):     # (batsize,)int
        overx = self.adb[x]
        mask = overx > 0
        mask = T.outer(mask, T.ones((self.outdim,)))
        ret = T.switch(mask, self.override(overx), self.base(x))
        self._maskfrom(ret, x)
        return ret

    @property
    def w(self):
        return None         # TODO


class NewOverriddenWordEmb(WordEmb):
    def __init__(self, base, override, which=None, **kw):
        assert(base.outdim == override.outdim)  # ensure same output dimension
        assert(override.raretoken in override.D)
        baseindexes = Val(np.asarray(sorted(base.D.values()), dtype="int32"))
        basevar = base(baseindexes)     # slicing out base vectors
        if which is None:   # which: list of words to override
            ad = {v: override.D[k]
                    if k in override.D
                    else override.D[override.raretoken]
                  for k, v in base.D.items()}
        else:
            ad = {base.D[k]: override.D[k]
                    if k in override.D
                    else override.D[override.raretoken]
                  for k in which}
        valval = np.zeros((max(ad.keys()) + 1,), dtype="int32")
        for i in range(valval.shape[0]):
            valval[i] = ad[i] if i in ad else 0
        overrideindexes = Val(valval)
        # a zero in override indexes means basevar will be used and override's rare will be invoked but not used
        overridevar = override(overrideindexes)
        overridemask = np.repeat((valval[:, None] != override.D[override.raretoken]) * 1,
                                 base.outdim, axis=1)
        v = T.switch(overridemask, overridevar, basevar)
        super(NewOverriddenWordEmb, self).__init__(worddic=base.D, value=v,
               dim=base.outdim, **kw)


class AugmentedWordEmb(WordEmb):    # TODO: RARE TOKEN MGMT
    def __init__(self, base, augment, **kw):
        assert(base.outdim == augment.outdim)
        super(AugmentedWordEmb, self).__init__(worddic=base.D, value=False,
                dim=base.outdim, normalize=base.normalize,
                trainfrac=base.trainfrac, **kw)
        self.base = base
        self.augment = augment
        self.ad = {v: augment.D[k]
                    if k in augment.D
                    else 0
                   for k, v in base.D.items()}
        valval = np.zeros((max(self.ad.keys()) + 1,), dtype="int32")
        for i in range(valval.shape[0]):
            valval[i] = self.ad[i] if i in self.ad else 0
        self.adb = Val(valval)

    def apply(self, x):
        baseemb = self.base(x)
        augmemb = self.augment(self.adb[x])
        ret = T.concatenate([baseemb, augmemb], axis=1)
        self._maskfrom(ret, x)
        return ret

    @property
    def w(self):
        return None         # TODO


class Glove(WordEmb):
    defaultpath = "../../../data/glove/glove.%dd"
    maskid = 0
    rareid = 1

    def __init__(self, dim, vocabsize=None, path=None, trainfrac=0.0,
                 **kw):
        path = self._get_path(dim, path=path)
        value, wdic = self.loadvalue(path, dim, indim=vocabsize, maskid=self.maskid, rareid=self.rareid)
        self.allwords = wdic.keys()
        super(Glove, self).__init__(dim=dim, value=value,
                                    worddic=wdic, trainfrac=trainfrac, **kw)

    @classmethod
    def _get_path(cls, dim, path=None):
        # if dim=None, load all
        path = cls.defaultpath if path is None else path
        relpath = path % dim
        path = os.path.join(os.path.dirname(__file__), relpath)
        return path

    def loadvalue(self, path, dim, indim=None, maskid=None, rareid=None):
        tt = TT(self.__class__.__name__)
        tt.tick()
        W = np.load(open(path+".npy"))
        if indim is not None:
            W = W[:indim, :]
        if rareid is not None:
            W = np.concatenate([np.zeros_like(W[0, :])[np.newaxis, :], W], axis=0)
        if maskid is not None:
            W = np.concatenate([np.zeros_like(W[0, :])[np.newaxis, :], W], axis=0)
        tt.tock("vectors loaded")
        tt.tick()
        # dictionary
        words = pkl.load(open(path+".words"))
        D = OrderedDict()
        i = 0
        if maskid is not None:
            D["<MASK>"] = i; i+=1
        if rareid is not None:
            D["<RARE>"] = i; i+=1
        for j, word in enumerate(words):
            if indim is not None and j >= indim:
                break
            D[word] = i
            i += 1
        tt.tock("dictionary created")
        return W, D



