import sys, re
from IPython import embed
from teafacto.util import argprun, tokenize, ticktock
from teafacto.blocks.memory import LinearGateMemAddr, DotMemAddr
from teafacto.blocks.match import MatchScore
from teafacto.blocks.lang.wordvec import Glove
from teafacto.blocks.basic import VectorEmbed
from teafacto.core.base import tensorops as T, Val
from collections import OrderedDict
import numpy as np, pickle
from teafacto.blocks.seqproc import SimpleSeq2Idx, SimpleSeq2Vec, SimpleVec2Idx, MemVec2Idx, Seq2Idx
from multiprocessing import Pool, cpu_count
from contextlib import closing
from teafacto.datahelp.labelsearch import SimpleQuestionsLabelIndex
from teafacto.eval.metrics import ClassAccuracy, RecallAt

# persistent memoization
from tempfile import mkdtemp
cachedir = mkdtemp()
from joblib import Memory
memory = Memory(cachedir=cachedir, verbose=0)

""" SUBJECT PREDICTION TRAINING WITH NEGATIVE SAMPLING """

class SubjRankEval(object):
    def __init__(self, scorer, host="localhost", index="sq_subjnames_fb2m",
                 worddic=None, entdic=None, metrics=None):
        self.scorer = scorer
        self.host = host
        self.index = index
        self.wd = worddic
        self.rwd = {v: k for k, v in self.wd.items()}
        self.ed = entdic
        self.metrics = metrics if metrics is not None else []
        #embed()

    def eval(self, data, gold, transform=None):     # data: wordidx^(batsize, seqlen), gold: entidx^(batsize)
        # generate candidates
        cans = gencans(data, host=self.host, index=self.index, rwd=self.rwd, ed=self.ed)           # list of lists of entidx
        assert len(cans) == data.shape[0] == gold.shape[0]
        #
        embed()
        predictor = self.scorer.predict.transform(transform)
        tt = ticktock("evaluator")
        tt.tick("evaluating...")
        nocans = 0
        for i in range(data.shape[0]):
            numcans = len(cans[i])
            predinp = [np.repeat(np.expand_dims(data[i, :], axis=0), numcans, axis=0),
                       np.asarray(cans[i], dtype="int32")]
            #print predinp, "%d/%d" % (i, data.shape[0]), numcans
            if numcans > 0:
                predinpscores = predictor(*predinp)      # (numcans,)
                ranking = sorted(zip(cans[i], list(predinpscores)),
                                 key=lambda (x, y): y)
                for metric in self.metrics:
                    metric.accumulate([gold[i]], ranking)
            else:
                nocans += 1
            if i % 1000 == 0:
                tt.live("evaluated: %.2f%%" % (i*100./data.shape[0]))
        tt.tock("evaluated")
        print "no cans for %d questions" % nocans
        return self.metrics

@memory.cache(ignore=["data", "rwd", "ed"])
def gencans(data, top=50, exact=True, rwd=None, ed=None, host=None, index=None):
    idx = SimpleQuestionsLabelIndex(host=host, index=index)
    # transform data using worddic and search
    sentences = []
    cans = []
    tt = ticktock("candidate generator")
    tt.tick("generating cans")
    for i in range(data.shape[0]):
        sentence = " ".join(
                        map(lambda x: rwd[x],
                            filter(lambda x: x in rwd, data[i, :])))
        sentences.append(sentence)
        searchres = idx.searchsentence(sentence, exact=exact, top=top)
        scans = map(lambda (x, (y, z)): ed[x], searchres.items())
        if i % 10 == 0:
            tt.live("%d of %d" % (i, data.shape[0]))
        cans.append(scans)
    tt.stoplive()
    tt.tock("generated cans")
    return cans



def _readdata(p):
    x = pickle.load(open(p))
    def preprocessforsubjdet(x):
        goldmat = x[1]
        return x[0], goldmat[:, 0]
    entdic = x["entdic"]
    numents = x["numents"]
    newdic = {}
    for k, v in entdic.items():
        if v < numents:
            newdic[k] = v
    train = preprocessforsubjdet(x["train"])
    valid = preprocessforsubjdet(x["valid"])
    test  = preprocessforsubjdet(x["test"])
    return train, valid, test, x, newdic


def readdata(p):
    x = pickle.load(open(p))
    def preprocessforsubjdet(x):
        goldmat = x[1]
        return x[0], goldmat[:, 0]
    worddic = x["worddic"]
    entdic = x["entdic"]
    numents = x["numents"]
    newdic = {}
    for k, v in entdic.items():
        if v < numents:
            newdic[k] = v
    entmat = x["entmat"]
    entmat = entmat[:numents, :]
    train = preprocessforsubjdet(x["train"])
    valid = preprocessforsubjdet(x["valid"])
    test  = preprocessforsubjdet(x["test"])
    return train, valid, test, worddic, newdic, entmat

#region junk
def ents2labels(labelp, entdic, maxwords=50, parallel=True):
    labeldic = loadlabels(labelp)
    wolabels = set()
    ents = sorted(entdic.items(), key=lambda (x, y): y)
    if parallel:
        with closing(Pool(cpu_count() - 1)) as p:
            ents = p.map(MapEnts2labels(labeldic, wolabels=wolabels, maxlen=maxwords), ents)
    else:
        ents = map(MapEnts2labels(labeldic, wolabels=wolabels, maxlen=maxwords), ents)
        print "%d entities have no labels" % len(wolabels)
    return ents


class MapEnts2labels():
    def __init__(self, labeldic, wolabels=set(), maxlen=50):
        self.labeldic = labeldic
        self.wolabels = wolabels
        self.maxlen = maxlen

    def __call__(self, x):
        ret = tokenize(self.labelfy(x[0]))
        ret = ret[:min(self.maxlen, len(ret))]
        return ret, x[1]

    def labelfy(self, x):
        if x in self.labeldic:
            return self.labeldic[x]
        else:
            self.wolabels.add(x)
            return x


def getmemdata(entdic, worddic,
               labelp="../../../data/simplequestions/labels.map",
               maxwords=30):    # updates worddic with words found in entity labels
    ents = ents2labels(labelp, entdic, maxwords=maxwords)
    allentwords = set()
    maxlen = 0
    prevc = -1
    for ent, c in ents:
        assert(c-1 == prevc)
        prevc = c
        maxlen = max(maxlen, len(ent))
        for entw in ent:
            allentwords.add(entw)
    maxlen = min(maxlen, maxwords)
    entwordsnotinworddic = allentwords.difference(set(worddic.keys()))
    for ewniw in entwordsnotinworddic:
        worddic[ewniw] = len(worddic)
    ret = [[worddic[w] for w in ent] for (ent, _) in ents]
    retmat = np.zeros((len(ents), maxlen)).astype("int32") - 1
    i = 0
    for r in ret:
        retmat[i, :len(r)] = r
        i += 1
    return retmat


def toglove(wordmat, worddic, dim=50):
    g = Glove(dim)
    gws = set(g.D.keys())
    wdws = set(worddic.keys())
    diff = wdws.difference(gws)
    # gather states about diff
    diffcounts = {worddic[k]: 0 for k in diff}
    total = 0
    moretal = 0
    for i in range(wordmat.shape[0]):
        for j in range(wordmat.shape[1]):
            if wordmat[i, j] >= 0:
                total += 1
                if wordmat[i, j] in diffcounts:
                    diffcounts[wordmat[i, j]] += 1
                    moretal += 1
    diffcounts = sorted(diffcounts.items(), key=lambda (k, v): v, reverse=True)
    print "%d words unknown by Glove of %d total words" % (moretal, total)
    revdic = {v: k for k, v in worddic.items()}
    d2g = lambda x: g * revdic[x] if x in revdic else x
    newdic = {k: d2g(v) for k, v in worddic.items()}
    newmat = np.vectorize(d2g)(wordmat)
    revgdic = {v: k for k, v in g.D.items()}
    embed()

def getdic2glove(worddic, dim=50):
    g = Glove(dim)
    revdic = {v: k for k, v in worddic.items()}
    d2g = lambda x: g * revdic[x] if x in revdic else x
    newdic = {k: d2g(v) for k, v in worddic.items()}
    return d2g, newdic

def getcharmemdata(reldic):
    rels = sorted(reldic.items(), key=lambda (x, y): y)
    maxlen = 0
    prevc = -1
    for rel, c in rels:
        assert(c-1 == prevc)
        prevc = c
        maxlen = max(maxlen, len(rel))
    retmat = np.zeros((len(rels), maxlen)).astype("int32") - 1
    for rel, c in rels:
        retmat[c, :len(rel)] = map(ord, rel)
    return retmat


def evaluate(pred, gold):
    return np.sum(gold == pred) * 100. / gold.shape[0]
#endregion

def run(
        epochs=10,
        numbats=100,
        numsam=10000,
        negrate=1,
        lr=0.1,
        datap="../../../data/simplequestions/datamat.word.mem.fb2m.pkl",
        embdim=100,
        innerdim=200,
        wreg=0.00005,
        bidir=False,
        keepmincount=5,
        mem=False,
        dynmem=False,
        sameenc=False,
        memaddr="dot",
        memattdim=100,
        membidir=False,
        memlayers=1,
        memmaxwords=5,
        layers=1,
        testfirst=False,
        ):

    tt = ticktock("script")
    tt.tick()
    (traindata, traingold), (validdata, validgold), (testdata, testgold), \
    worddic, entdic, entmat\
        = readdata(datap)

    print entmat.shape
    print traindata.shape, traingold.shape

    tt.tock("data loaded")

    # *data: matrix of word ids (-1 filler), example per row
    # *gold: vector of true entity ids
    # entmat: matrix of word ids (-1 filler), entity label per row, indexes according to *gold
    # *dic: from word/ent-fbid to integer id, as used in data

    numwords = max(worddic.values()) + 1
    numents = max(entdic.values()) + 1
    print "%d words, %d entities" % (numwords, numents)

    if bidir:
        encinnerdim = [innerdim/2]*layers
    else:
        encinnerdim = [innerdim]*layers

    # question representation:
    # encodes question sequence to vector
    qenc = SimpleSeq2Vec(indim=numwords,
                        inpembdim=embdim,
                        innerdim=encinnerdim,
                        maskid=-1,
                        bidir=bidir)

    # entity representation:
    if mem:
        # encodes label to vector
        if membidir:
            innerdim = [innerdim/2]*memlayers
        else:
            innerdim = [innerdim]*memlayers
        memembdim = embdim
        lenc = SimpleSeq2Vec(indim=numwords,
                                inpembdim=memembdim,
                                innerdim=innerdim,
                                maskid=-1,
                                bidir=membidir)
    else:
        # embeds entity id to vector
        lenc = VectorEmbed(indim=numents, dim=innerdim)

    # question-entity score computation:
    scorer = MatchScore(qenc, lenc)       # batched dot

    # trainer config preparation
    class PreProcf(object):
        def __init__(self, entmat):
            self.em = Val(entmat)                # entmat: idx[word]^(numents, len(ent.name))

        def __call__(self, datas, gold):    # gold: idx^(batsize, )
            return (datas, self.em[gold, :]), {}

    class NegIdxGen(object):
        def __init__(self, rng):
            self.min = 0
            self.max = rng

        def __call__(self, datas, gold):    # gold: idx^(batsize,)
            return datas, np.random.randint(self.min, self.max, gold.shape).astype("int32")

    if testfirst:
        eval = SubjRankEval(scorer, worddic=worddic, entdic=entdic, metrics=[ClassAccuracy(), RecallAt(10)])
        evalres = eval.eval(testdata, testgold, transform=PreProcf(entmat))
        print evalres
        tt.msg("tested dummy")
        sys.exit()
    #embed()
    # trainer config and training
    nscorer = scorer.nstrain([traindata, traingold]).transform(PreProcf(entmat))\
        .negsamplegen(NegIdxGen(numents)).negrate(negrate).objective(lambda p, n: p - n)\
        .adagrad(lr=lr).l2(wreg).grad_total_norm(1.0)\
        .validate_on([validdata, validgold]).takebest()\
        .train(numbats=numbats, epochs=epochs)

    # evaluation
    eval = SubjRankEval(scorer, worddic=worddic, entdic=entdic, metrics=[ClassAccuracy(), RecallAt(10)])

    evalres = eval.eval(testdata, testgold, transform=PreProcf(entmat))
    for evalre in evalres:
        print evalre


if __name__ == "__main__":
    argprun(run)