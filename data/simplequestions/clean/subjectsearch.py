import re
from teafacto.util import tokenize, ticktock, isstring
from nltk.corpus import stopwords
from teafacto.procutil import wordids2string
from IPython import embed


class SubjectSearch(object):
    def __init__(self, subjinfop="subjs-counts-labels-types.fb2m.tsv"):
        self.indexdict = {}
        self.ignoresubgrams = True
        self.stops = set(stopwords.words("english"))
        if isstring(subjinfop):
            self.build(subjinfop)
        elif isinstance(subjinfop, dict):
            self.indexdict = subjinfop
        else:
            raise Exception("unknown stuff")

    def build(self, p):
        i = 0
        tt = ticktock("builder")
        tt.tick("building")
        for line in open(p):
            sline = line[:-1].split("\t")
            fb_id = sline[0]
            triplecount = int(sline[1]) + int(sline[2])
            name = " ".join(tokenize(sline[3]))
            type_id = sline[4]
            type_id = type_id if type_id != "<UNK>" else None
            type_name = " ".join(tokenize(sline[5]))
            type_name = type_name if type_name != " ".join(tokenize("<UNK>")) else None
            if name not in self.indexdict:
                self.indexdict[name] = []
            self.indexdict[name].append({"fb_id": fb_id,
                    "triplecount": triplecount, "type_id": type_id,
                    "type_name": type_name})
            i += 1
            if i % 1000 == 0:
                tt.live("{}k".format(i//1000))
        tt.tock("built")

    def save(self, p):
        with open(p, "w", 1) as f:
            for item in self.indexdict.items():
                f.write("::{}\n".format(item[0]))
                for ve in item[1]:
                    f.write("{}\t{}\t{}\t{}\n".format(
                        *[ve[x] for x in "fb_id triplecount type_id type_name".split()]
                    ))

    @staticmethod
    def load(p):
        tt = ticktock("SubjectSearch")
        tt.tick("loading")
        d = {}
        l = []
        k = None
        with open(p) as f:
            for line in f:
                if line[:2] == "::":
                    if k is None:
                        assert(l == [])
                    else:
                        d[k] = l
                        l = []
                    k = line[2:-1]
                else:
                    splits = line[:-1].split("\t")
                    le = dict(zip("fb_id triplecount type_id type_name".split(),
                          [splits[0], int(splits[1])] + splits[2:]))
                    l.append(le)
        d[k] = l
        tt.tock("loaded")
        return SubjectSearch(subjinfop=d)

    def search(self, s, top=5):
        ss = " ".join(tokenize(s))
        return self._search(ss, top=top)

    def _search(self, ss, top=5):
        res = self.indexdict[ss] if ss in self.indexdict else []
        sres = sorted(res, key=lambda x: x["triplecount"], reverse=True)
        return sres[:min(top, len(sres))]

    def searchsentence(self, sentence, top=5):
        words = tokenize(sentence)
        res = self._recurngramsearch(words, top=top)
        return res

    def _recurngramsearch(self, seq, top=5, right=False):
        searchterm = " ".join(seq)
        res = self._search(searchterm, top=top)
        if len(res) > 0 and self.ignoresubgrams:
            return res
        elif len(seq) == 1:
            return res if seq[0] not in self.stops else []
        else:
            lres = self._recurngramsearch(seq[:-1], top=top, right=False) if not right else []
            rres = self._recurngramsearch(seq[1:], top=top, right=True)
            res = res + lres + rres
        return res

    def searchwordmat(self, wordmat, wd, top=5):
        cans = []
        rwd = {v: k for k, v in wd.items()}
        for i in range(wordmat.shape[0]):
            sentence = wordids2string(wordmat[i], rwd=rwd)
            res = self.searchsentence(sentence, top=top)
            cans.append([r["fb_id"] for r in res])
        return cans


if __name__ == "__main__":
    #s = SubjectSearch(); s.save("subjinfo.idxdic")
    #embed()
    s = SubjectSearch.load("subjinfo.idxdic")
    #s.searchsentence("what is the book e about")
    import pickle
    print "loading datamat"
    x = pickle.load(open("datamat.word.fb2m.pkl"))
    print "datamat loaded"
    testdata = x["test"][0]
    testgold = x["test"][1]
    wd = x["worddic"]
    ed = x["entdic"]
    ne = x["numents"]
    del x
    print "generating cans"
    testcans = s.searchwordmat(testdata, wd, top=10)
    embed()
