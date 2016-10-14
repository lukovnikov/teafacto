import re, pickle, editdistance
from teafacto.util import tokenize, ticktock, isstring, argprun
from nltk.corpus import stopwords
from nltk.stem import porter
from teafacto.procutil import wordids2string
from IPython import embed

class Processor(object):
    def __init__(self):
        self.stemmer = porter.PorterStemmer()

    def stemmedprocessline(self, x):
        x = x.replace("'s", "")
        x = x.replace("' s", "")
        x = x.replace("'", "")
        tokens = tokenize(x)
        #print tokens
        stokens = [self.stemmer.stem(token) for token in tokens]
        return " ".join(stokens)

    def processline(self, x):
        return " ".join(tokenize(x))


class SubjectSearch(object):
    stops = stopwords.words("english")
    customstops = set("the a an of on is at in by did do not does had has have for what which when where why who whom how".split())
    smallstops = set("the a an of on at by".split())

    def __init__(self, subjinfop="subjs-counts-labels-types.fb2m.tsv", revind=None):
        self.indexdict = {}
        self.ignoresubgrams = True
        self.processor = Processor()
        self.revind = revind
        self.maxeditdistance = 1
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
            name = self.processor.processline(sline[3])
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
        ret = SubjectSearch(subjinfop=d, revind=SubjectSearch.buildrevindex(d))
        return ret

    @staticmethod
    def buildrevindex(d):
        revind = {}
        for k in d.keys():
            words = k.split()
            for word in words:
                if len(word) < 2:
                    continue
                if word not in revind:
                    revind[word] = []
                revind[word].append(k)
        return revind

    def search(self, s, top=5):
        ss = self.processor.processline(s)
        return self._search(ss, top=top)

    def _search(self, ss, top=5, edsearch=True):
        res = self.indexdict[ss] if ss in self.indexdict else []
        sres = sorted(res, key=lambda x: x["triplecount"], reverse=True)
        ret = sres[:min(top, len(sres))]
        for x in ret:
            x.update({"name": ss})
        if len(ret) == 0 and self.revind is not None and edsearch and self.maxeditdistance > 0:   # no exact matches
            nonexactsearchstrings = set()
            words = ss.split()
            if len(words) >= 2:
                for word in words:
                    if len(word) < 2 or word in self.customstops:
                        continue
                    if word not in self.revind:
                        continue
                    for nonexcan in self.revind[word]:
                        if abs(len(nonexcan) - len(ss)) > 3:
                            continue
                        nonexcanred = nonexcan.replace(" '", "")
                        #embed()
                        if editdistance.eval(nonexcanred, ss) <= self.maxeditdistance:
                            nonexactsearchstrings.add(nonexcan)
                for nonexactsearchstring in nonexactsearchstrings:
                    edsearchres = self._search(nonexactsearchstring, top=top, edsearch=False)
                    #embed()
                    ret.extend(edsearchres)
        return ret

    def searchsentence(self, sentence, top=5):
        if sentence[-1] == "?":
            sentence = sentence[:-1]
        words = self.processor.processline(sentence).split()
        if self.ignoresubgrams:
            res = self._searchngrams(words, top=top)
        else:
            res = self._recurngramsearch(words, top=top)
        return res

    def _searchngrams(self, words, top=5):
        ngramsize = len(words)
        bannedpos = set()
        ret = []
        while ngramsize > 0:
            for i in range(0, len(words) - ngramsize + 1):
                coveredpos = set(range(i, i + ngramsize))
                if len(coveredpos.difference(bannedpos)) == 0 \
                        and self.ignoresubgrams:
                    continue
                #elif i in bannedpos and self.ignoresubgrams:
                #    continue
                else:
                    ss = words[i: i + ngramsize]
                    if len(ss) == 1 and (ss[0] in self.stops):
                        res = []
                    else:
                        res = self._search(" ".join(ss), top=top)
                    if len(res) > 0 and self.ignoresubgrams:
                        if ss[0] in self.smallstops:
                            if False and ngramsize > 1:
                                coveredpos = set(range(i+2, i + ngramsize))
                                coveredpos.add(i)
                            else:
                                coveredpos = set([i])
                        bannedpos.update(coveredpos)
                    ret += res
            ngramsize -= 1
        return ret

    def _recurngramsearch(self, seq, top=5, right=False):
        searchterm = " ".join(seq)
        res = self._search(searchterm, top=top)
        if len(seq) == 1:
            return res if seq[0] not in self.stops else []
        else:
            lres = self._recurngramsearch(seq[:-1], top=top, right=False) if not right else []
            rres = self._recurngramsearch(seq[1:], top=top, right=True)
            res = res + lres + rres
        return res

    def searchwordmat(self, wordmat, wd, top=5):
        cans = []
        rwd = {v: k for k, v in wd.items()}
        tt = ticktock("wordmatsearcher")
        tt.tick("started searching")
        for i in range(wordmat.shape[0]):
            sentence = wordids2string(wordmat[i], rwd=rwd)
            #ssentence.replace(" '", "")
            res = self.searchsentence(sentence, top=top)
            cans.append([r["fb_id"] for r in res])
            tt.progress(i, wordmat.shape[0], live=True)
        tt.tock("done searching")
        return cans


def gensubjclose(cansp="traincans10c.pkl"):
    traincans = pickle.load(open(cansp))
    allcans = set()
    for traincane in traincans:
        allcans.update(traincane)
    print len(allcans)
    qsofcans = {k: set() for k in allcans}
    for i in range(len(traincans)):
        traincane = traincans[i]
        for traincan in traincane:
            qsofcans[traincan].add(i)
    cansofcans = {k: set() for k in allcans}
    for k, v in qsofcans.items():
        for ve in v:
            cansofcans[k].update(traincans[ve])
    for k in cansofcans:
        cansofcans[k].remove(k)
        cansofcans[k] = set(list(cansofcans[k])[:500]) if len(cansofcans[k]) > 500 else cansofcans[k]
    return cansofcans


def run(numcans=10,
        build=False,
        load=False,
        gencan=False,
        genclose=False,
        test=False,
        editdistance=False):
    if False:
        p = Processor()
        o = p.processline("porter ' s stemmer works ' in united states")
        print o
    if build:
        s = SubjectSearch(); s.save("subjinfo.idxdic")
        embed()
    if load:
        s = SubjectSearch.load("subjinfo.idxdic")
        if editdistance:
            s.maxeditdistance = 1
        else:
            s.maxeditdistance = 0
        embed()
    #s.searchsentence("what is the book e about")
    if gencan:
        import pickle
        print "loading datamat"
        x = pickle.load(open("datamat.word.fb2m.pkl"))
        print "datamat loaded"
        testdata = x["test"][0]
        testgold = x["test"][1]
        wd = x["worddic"]
        ed = x["entdic"]
        del x
        print "generating cans"
        testcans = s.searchwordmat(testdata, wd, top=numcans)
        testcanids = [[ed[x] for x in testcan] for testcan in testcans]
        acc = 0
        for i in range(testgold.shape[0]):
            if testgold[i, 0] in testcanids[i]:
                acc += 1
        print acc * 1. / testgold.shape[0]
        embed()
    if False:
        print s.searchsentence("2 meter sessies?")
    if genclose:
        subjclose = gensubjclose("traincans10c.pkl")
        embed()



if __name__ == "__main__":
    argprun(run)
