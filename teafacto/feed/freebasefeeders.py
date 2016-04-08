import numpy as np

from teafacto.feed.langfeeds import WordSeqFeed
from teafacto.feed.langtransform import WordToWordCharTransform
from teafacto.util import ticktock


def iden(x):
    return x


class FreebaseEntFeedsMaker(object):
    def __init__(self, datapath, worddic, entdic, numwords=10, numchars=30, unkwordid=1, unkentid=0):
        self.path = datapath
        self.trainingdata = []
        self.golddata = []
        self.worddic = worddic
        self.numwords = numwords
        self.numchars = numchars
        self.unkwordid = unkwordid
        self.unkentid = unkentid
        self.load(entdic)

    def load(self, entdic):
        self.trainingdata = []
        self.golddata = []
        tt = ticktock(self.__class__.__name__)
        tt.tick("loading kgraph")
        with open(self.path) as f:
            c = 0
            for line in f:
                ns = line[:-1].split("\t")
                if len(ns) is not 2:
                    print line, c
                    continue
                sf, fb = ns
                self.trainingdata.append(self._process_sf(sf))
                entids = self._process_ent(fb, entdic)
                self.golddata.append(entids)
                if c % 1e6 == 0:
                    tt.tock("%.0fM" % (c/1e6)).tick()
                c += 1
        self.golddata = np.asarray(self.golddata, dtype="int32")
        self.trainingdata = np.array(self.trainingdata)

    @property
    def trainfeed(self):
        t = WordToWordCharTransform(self.worddic, unkwordid=self.unkwordid, numwords=self.numwords, numchars=self.numchars)
        return WordSeqFeed(self.trainingdata, t)

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

    def _process_ent(self, fb, entdic):
        if fb in entdic:
            ret = entdic[fb]
        else:
            ret = self.unkentid
        return ret


class FreebaseSeqFeedMaker(FreebaseEntFeedsMaker):
    def _process_ent(self, fb, entdic):
        ret = []
        for e in fb.split(" "):
            if fb in entdic:
                ret.append(entdic[fb])
            else:
                ret.append(self.unkentid)
        return ret


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

