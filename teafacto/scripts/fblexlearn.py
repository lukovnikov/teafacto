from teafacto.core.base import tensorops as T
from teafacto.core.base import Block
from teafacto.core.datafeed import DataFeed
from teafacto.blocks.rnn import SeqEncoder
from teafacto.blocks.lang.wordembed import IdxToOneHot, WordEncoderPlusGlove, WordEmbedPlusGlove
from teafacto.blocks.lang.wordvec import Glove
from teafacto.util import argprun, ticktock, issequence
import numpy as np, pandas as pd
from IPython import embed


class FBLexLearn(Block):
    def __init__(self, **kw):
        super(FBLexLearn, self).__init__(**kw)

    def apply(self, inp):
        pass


class FBLexDataFeed(DataFeed):
    def __init__(self, data, worddic, numwords=10, numchars=30, **kw):
        super(FBLexDataFeed, self).__init__(data, **kw)
        self.worddic = worddic
        self._shape = (data.shape[0], numwords, numchars)

    @property
    def dtype(self):
        return np.dtype("int32")

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, item):
        ret = self.data.__getitem__(item)
        return self.transform(ret)

    def transform(self, x):
        def transinner(word):
            skip = False
            if word is None:
                retword = [0]*(self.shape[2]+1)         # missing word ==> all zeros
                skip = True
            else:
                if word in self.worddic:
                    retword = [self.worddic[word]]      # get word index
                else:
                    retword = [1]                       # unknown word
                retword.extend(map(ord, word))
                retword.extend([0]*(self.shape[2]-len(retword)+1))
            return retword, skip #np.asarray(retword, dtype="int32")
        '''print x, type(x), x.dtype, x.shape
        ret = np.zeros((x.shape + (self.shape[1], self.shape[2]+1)))
        retv = np.vectorize(transinner)(x)
        print retv
        for i in range(self.shape[2]):
            ret[..., i] = np.vectorize(lambda z: transinner(z))(x)[i]
        return ret'''

        print type(x), x.dtype
        ret = np.zeros((x.shape[0], self.shape[1], self.shape[2]+1), dtype="int32")
        i = 0
        while i < x.shape[0]:
            j = 0
            while j < x.shape[1]:
                word = x[i, j]
                retword, skip = transinner(word)
                if skip:
                    j = x.shape[1]
                else:
                    ret[i, j, :] = retword
                j += 1
            i += 1
        return ret


class FBLexDataFeedsMaker(object):
    def __init__(self, datapath, worddic, entdic, numwords=10, numchars=30):
        self.path = datapath
        self.trainingdata = []
        self.golddata = []
        self.worddic = worddic
        self.numwords = numwords
        self.numchars = numchars
        self.load(entdic)

    def load(self, entdic):
        self.trainingdata = []
        self.golddata = []
        tt = ticktock(self.__class__.__name__)
        tt.tick("loading freebase lex")
        with open(self.path) as f:
            c = 0
            for line in f:
                ns = line[:-1].split("\t")
                if len(ns) is not 2:
                    print line, c
                    continue
                sf, fb = ns
                self.trainingdata.append(self._process_sf(sf))
                self.golddata.append(entdic[fb])
                if c % 1e6 == 0:
                    tt.tock("%.0fM" % (c/1e6)).tick()
                c += 1
        self.golddata = np.asarray(self.golddata, dtype="int32")
        self.trainingdata = np.array(self.trainingdata)

    @property
    def trainfeed(self):
        return FBLexDataFeed(self.trainingdata, worddic=self.worddic, numwords=self.numwords, numchars=self.numchars)

    @property
    def goldfeed(self):
        return self.golddata    # already np array of int32

    def _process_sf(self, sf):
        words = sf.split(" ")
        if len(words) > self.numwords:
            words = words[:self.numwords]
        i = 0
        while i < len(words):
            if len(words[i]) > self.numchars:
                words[i] = words[i][:self.numchars]
            i += 1
        words.extend([None]*max(0, (self.numwords - len(words))))
        return words


def getglovedict(path, offset=2):
    gd = {}
    with open(path) as f:
        c = offset
        for line in f:
            ns = line.split(" ")
            w = ns[0]
            gd[w] = c
            c += 1
    return gd


def getentdict(path, offset=2):
    ed = {}
    with open(path) as f:
        for line in f:
            e, i = line[:-1].split("\t")
            ed[e] = int(i) + offset
    return ed


def run(
        epochs=1000,
        lr=0.1,
        wreg=0.0001,
        fblexpath="/media/denis/My Passport/data/freebase/labelsrevlex.map",
        glovepath="../../data/glove/glove.6B.50d.txt",
        fbentdicp="../../data/freebase/entdic.map",
        numwords=10,
        numchars=30,
    ):
    gd = getglovedict(glovepath)
    print gd["alias"]
    ed = getentdict(fbentdicp)
    print ed["m.0ndj09y"]

    indata = FBLexDataFeedsMaker(fblexpath, gd, ed, numwords=numwords, numchars=numchars)
    print indata.goldfeed.shape
    tt = ticktock("fblextranstimer")
    tt.tick()
    print indata.trainfeed[0:90000].shape
    tt.tock("transformed")
    #embed()


if __name__ == "__main__":
    argprun(run)
