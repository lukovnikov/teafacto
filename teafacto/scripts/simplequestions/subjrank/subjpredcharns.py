import pickle, sys, os, numpy as np
from IPython import embed
from teafacto.util import ticktock, tokenize


def readdata(mode="char",
             p="../../../../data/simplequestions/clean/datamat.word.fb2m.pkl",
             entinfp="../../../../data/simplequestions/clean/subjs-counts-labels-types.fb2m.tsv",
             maskid=-1,
             cachep="subjpredcharns.readdata.cache.pkl"):
    tt = ticktock("dataloader")
    if os.path.isfile(cachep):      # load
        tt.tick("loading from cache")
        ret = pickle.load(open(cachep))
        entdic = ret[-1]
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
            traindata = wordmat2charmat(traindata, rwd)
            validdata = wordmat2charmat(validdata, rwd)
            testdata = wordmat2charmat(testdata, rwd)
            entmat = wordmat2charmat(entmat, rwd)
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
            rcd = {v: k for k, v in chardic.items()}
            def cpp(x):
                print "".join([rcd[xe] for xe in x])
            worddic = chardic
        ret = ((traindata, traingold), (validdata, validgold), (testdata, testgold),
               entmat, worddic, entdic)
        if cachep is not None:
            tt.tick("dumping to cache")
            pickle.dump(ret, open(cachep, "w"))
            tt.tock("dumped to cache")

    subjinfo = loadsubjinfo(entinfp, entdic)
    testcans = loadtestcans()
    embed()
    return ret


def wordmat2charmat(wm, rwd, maxmaxlen=120, maskid=-1):       # wm: (numsam, len)
    tt = ticktock("chartransformer")
    tt.tick("transforming word mat to char mat")
    maxlen = 0
    toolong = 0
    cm = maskid * np.ones((wm.shape[0], maxmaxlen), dtype="int32")
    for i in range(wm.shape[0]):
        string = " ".join([rwd[x] for x in wm[i] if x != maskid])
        maxlen = max(maxlen, len(string))
        if len(string) > maxmaxlen:
            toolong += 1
            string = string[:maxmaxlen]
        cm[i, :len(string)] = [ord(ch) for ch in string]
        tt.progress(i, wm.shape[0], live=True)
    if maxlen < maxmaxlen:
        cm = cm[:, :maxlen]
    print "{} too long".format(toolong)
    tt.tock("transformed")
    return cm


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
            subjinf[entdic[subjuri]] = (typname.lower().split(), typuri, subjc, objc)
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
    readdata()