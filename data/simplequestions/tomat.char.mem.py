import pickle, sys
from IPython import embed
import numpy as np

from teafacto.util import argprun, tokenize


def run(trainp="fb_train.tsv",
        testp="fb_test.tsv",
        validp="fb_valid.tsv",
        outp="datamat.char.mem.fb2m.pkl",
        entnames="subjnames_fb2m.map",
        rellist="rels_fb2m.list",
        maxnamelen=100):
    # worddic
    chardic = {" ": 0}
    def updatechardic(*chars):
        for char in chars:
            if char not in chardic:
                chardic[char] = len(chardic)

    # process entity names and relation list
    entdic = {}
    entmatr = []
    entswonames = set()
    c = 0
    maxlen = 0
    for line in open(entnames):
        e, n = line.split("\t")
        nt = " ".join(tokenize(n))
        nt = nt[:min(len(nt), maxnamelen)]
        maxlen = max(maxlen, len(nt))
        updatechardic(*set(nt))
        entmatr.append(nt)
        if e not in entdic:
            entdic[e] = len(entdic)
        if c % 1e3 == 0:
            print "%.0fk" % (c/1e3)
        c += 1

    def updateentk(*ents):  #ents have not been seen during initial population ==> no titles
        for ent in ents:
            assert(ent not in entdic)
            entdic[ent] = len(entdic)
            entswonames.add(ent)
            entmatr.append([" "])

    reldic = {}
    relmatr = []
    for line in open(rellist):
        r = line[:-1]
        rt = " ".join(tokenize(r))
        rt = rt[:min(len(rt), maxnamelen)]
        maxlen = max(maxlen, len(rt))
        updatechardic(*set(rt))
        relmatr.append(rt)
        r = "/" + r.replace(".", "/")
        if r not in reldic:
            reldic[r] = len(reldic)

    maxnamelen = min(maxlen, maxnamelen)

    print len(entdic), len(reldic), len(chardic), maxnamelen

    # process data
    def getdata(p, maxc=np.infty):
        data = []
        gold = []
        maxlen = 0
        c = 0
        for line in open(p):
            q, a = (line[:-1] if line[-1] == "\n" else line).split("\t")
            s, p = a.split()
            chars = " ".join(tokenize(q))
            updatechardic(*set(chars))
            maxlen = max(maxlen, len(chars))
            if s not in entdic:
                updateentk(s)
            if p not in reldic:
                raise Exception("impossibru!")
            charidx = map(lambda x: chardic[x] if x in chardic else chardic[" "], chars)
            data.append(charidx)
            gold.append([entdic[s], reldic[p]])
            c += 1
            if c % 100 == 0:
                print c
            if c > maxc:
                break
        datamat = np.zeros((c, maxlen)).astype("int32") - 1
        goldmat = np.zeros((c, 2)).astype("int32")
        i = 0
        for x in data:
            datamat[i, :len(x)] = x
            i += 1
        i = 0
        for x in gold:
            goldmat[i, :] = x
            i += 1
        return datamat, goldmat

    traindata = getdata(trainp)
    validdata = getdata(validp)
    testdata = getdata(testp)

    # build ent mat
    entmat = np.zeros((len(entmatr), maxnamelen), dtype="int32") - 1
    for i in range(len(entmatr)):
        x = entmatr[i]
        entmat[i, :len(x)] = map(lambda a: chardic[a], x)
    # build rel mat
    relmat = np.zeros((len(relmatr), maxnamelen), dtype="int32") - 1
    for i in range(len(relmatr)):
        x = relmatr[i]
        relmat[i, :len(x)] = map(lambda a: chardic[a], x)

    # pre-package tests:
    print entmat.shape[0], len(entdic)
    assert(entmat.shape[0] == len(entdic))
    # package
    entmat = np.concatenate([entmat, relmat], axis=0)
    numents = len(entdic)
    traindata[1][:, 1] += numents
    validdata[1][:, 1] += numents
    testdata[1][:, 1] += numents
    reldic = {k: v+numents for k, v in reldic.items()}
    entdic.update(reldic)

    # save
    acc = {
        "train": traindata,
        "valid": validdata,
        "test":  testdata,
        "chardic": chardic,
        "entdic": entdic,
        "entmat": entmat,
        "numents": numents
    }

    print "%d entities without names in datasets" % len(entswonames)

    pickle.dump(acc, open(outp, "w"))


if __name__ == "__main__":
    argprun(run)
    #getwords("What's is plaza-midwood (wood) a type of?")