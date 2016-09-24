import cPickle as pickle, sys, os, numpy as np
from IPython import embed

from teafacto.util import ticktock, tokenize, argprun
from teafacto.blocks.seq.enc import SimpleSeq2Vec, SimpleSeqStar2Vec
from teafacto.blocks.seq.rnn import SeqEncoder, MaskMode, EncLastDim
from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.lang.sentenc import WordCharSentEnc
from teafacto.blocks.lang.wordvec import Glove
from teafacto.procutil import wordmat2charmat


def readdata(mode="char",
             p="../../../../data/simplequestions/clean/datamat.word.fb2m.pkl",
             entinfp="../../../../data/simplequestions/clean/subjs-counts-labels-types.fb2m.tsv",
             maskid=-1,
             cachep="subjpredcharns.readdata.cache.pkl"):
    tt = ticktock("dataloader")
    if os.path.isfile(cachep):      # load
        tt.tick("loading from cache")
        ret = pickle.load(open(cachep))
        entdic, worddic = ret[-1], ret[-2]
        tt.tock("loaded from cache")
    else:
        # everything in word space !!!!!
        tt.tick("loading datamat")
        x = pickle.load(open(p))
        tt.tock("datamat loaded")
        worddic = x["worddic"]
        entdic = x["entdic"]
        numents = x["numents"]
        entmat = x["entmat"]
        traindata, traingold = x["train"]
        validdata, validgold = x["valid"]
        testdata, testgold = x["test"]

        if mode == "char":
            tt.tick("transforming to chars")
            rwd = {v: k for k, v in worddic.items()}
            traindata = wordmat2charmat(traindata, rwd=rwd, maxlen=110)
            validdata = wordmat2charmat(validdata, rwd=rwd, maxlen=110)
            testdata = wordmat2charmat(testdata, rwd=rwd, maxlen=110)
            entmat = wordmat2charmat(entmat, rwd=rwd, maxlen=75)
            tt.tick()
            allchars = set(list(np.unique(traindata)))\
                .union(set(list(np.unique(validdata))))\
                .union(set(list(np.unique(testdata))))\
                .union(set(list(np.unique(entmat))))
            tt.tock("collected unique chars").tick()
            chardic = dict(zip(allchars, range(len(allchars))))
            chardic[maskid] = maskid
            dicmap = np.vectorize(lambda x: chardic[x])
            traindata = dicmap(traindata)
            validdata = dicmap(validdata)
            testdata = dicmap(testdata)
            entmat = dicmap(entmat)
            tt.tock("transformed to chars")
            chardic = {chr(k): v for k, v in chardic.items() if k in range(256)}
            worddic = chardic
        ret = ((traindata, traingold), (validdata, validgold), (testdata, testgold),
               entmat, worddic, entdic)
        if cachep is not None:
            tt.tick("dumping to cache")
            pickle.dump(ret, open(cachep, "w"))
            tt.tock("dumped to cache")

    subjinfo = loadsubjinfo(entinfp, entdic)
    testcans = loadtestcans()

    debug = True
    if debug:
        rcd = {v: k for k, v in worddic.items()}
        def cpp(x):
            print "".join([rcd[xe] if xe in rcd else "" for xe in x])
        embed()
    return ret + (subjinfo, testcans)


def loadtestcans(p="../../../../data/simplequestions/clean/testcans.pkl"):
    tt = ticktock("test subjects candidate loader")
    tt.tick("loading candidates")
    ret = pickle.load(open(p))
    tt.tock("canddiates loaded")
    return ret


def loadsubjinfo(entinfp, entdic, cachep=None):#"subjinfo.cache.pkl"):
    tt = ticktock("subjinfoloader")
    def make():
        tt.tick("making subject info from file")
        subjinf = {}
        c = 0
        for line in open(entinfp):
            subjuri, subjc, objc, subjname, typuri, typname = line[:-1].split("\t")
            subjinf[entdic[subjuri]] = (subjname, typname.lower().split(), typuri, subjc, objc)
            if c % 1000 == 0:
                tt.live(str(c))
            c += 1
        tt.tock("made subject info from file")
        return subjinf
    if cachep is not None:
        if os.path.isfile(cachep):      # load
            tt.tick("loading cached subject info")
            subjinfo = pickle.load(open(cachep))
            tt.tock("loaded cached subject info")
        else:                           # make  and dump
            subjinfo = make()
            tt.tick("dumping subject info in cache")
            pickle.dump(subjinfo, open(cachep, "w"))
            tt.tock("dumped subject info in cache")
    else:       # just make
        subjinfo = make()
    return subjinfo


if __name__ == "__main__":
    x = np.random.randint(0, 50, (5, 4, 3))
    x = np.concatenate([np.random.randint(0, 1000, (5, 4, 1)), x], axis=2)
    x = np.concatenate([x, np.zeros_like(x)], axis=1)
    print x, x.shape
    m = WordCharSentEnc(numchars=50, charembdim=10, charinnerdim=20,
                        wordemb=Glove(50, 1000), wordinnerdim=3,
                        maskid=0, returnall=True)
    pred = m.predict(x)
    print pred, pred.shape