from teafacto.core.datafeed import DataFeed
import numpy as np
from teafacto.util import ticktock


class FBLexDataFeed(DataFeed):
    def __init__(self, data, worddic, unkwordid=1, numwords=10, numchars=30, **kw):
        super(FBLexDataFeed, self).__init__(data, **kw)
        self.worddic = worddic
        self._shape = (data.shape[0], numwords, numchars+1)
        self.unkwordid = unkwordid

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
                    retword = [self.unkwordid]                       # unknown word
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

        #print type(x), x.dtype
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
    def __init__(self, datapath, worddic, entdic, numwords=10, numchars=30, unkwordid=1):
        self.path = datapath
        self.trainingdata = []
        self.golddata = []
        self.worddic = worddic
        self.numwords = numwords
        self.numchars = numchars
        self.unkwordid = unkwordid
        self.load(entdic)

    def load(self, entdic):
        self.trainingdata = []
        self.golddata = []
        tt = ticktock(self.__class__.__name__)
        tt.tick("loading kgraph lex")
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
        return FBLexDataFeed(self.trainingdata, worddic=self.worddic, unkwordid=self.unkwordid, numwords=self.numwords, numchars=self.numchars)

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


def getglovedict(path, offset=2, top=None):
    gd = {}
    maxid = 0
    with open(path) as f:
        c = offset
        for line in f:
            if not top is None and c - offset > top:
                break
            ns = line.split(" ")
            w = ns[0]
            gd[w] = c
            maxid = max(maxid, c)
            c += 1
    return gd, maxid


def getentdict(path, offset=2, top=None):
    ed = {}
    maxid = 0
    with open(path) as f:
        c = 0
        for line in f:
            if not top is None and top < c:
                break
            e, i = line[:-1].split("\t")
            ed[e] = int(i) + offset
            maxid = max(ed[e], maxid)
            c += 1
    return ed, maxid

