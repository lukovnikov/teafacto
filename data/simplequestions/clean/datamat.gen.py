import pickle, sys, numpy as np, re
from IPython import embed
from teafacto.util import argprun, tokenize


def reluri_tokenize(reluri):
    return tokenize(reluri.replace("/", " ").replace("_", " "))


def run(trainp="../fb_train.tsv",
        testp="../fb_test.tsv",
        validp="../fb_valid.tsv",
        outp="datamat.word.fb2m.pkl",
        entnames="../subjnames_fb2m.map",
        rellist="../predicates_fb2m.list",
        maxnamelen=30):
    # worddic
    worddic = {"<RARE>": 0}
    wordcounts = {"<RARE>": 0}
    def addwords(*words):       # adds a word to the worddic
        for word in words:
            if word not in worddic:
                worddic[word] = len(worddic)
            if word not in wordcounts:
                wordcounts[word] = 0
            wordcounts[word] += 1

    # entity names
    entdic = {}
    entmatr = []
    c = 0       # counter
    maxlen = 0
    for line in open(entnames):
        entity_id, entity_label = line[:-1].split("\t")
        entity_label_tokens = tokenize(entity_label)
        entity_label_tokens = entity_label_tokens[:min(len(entity_label_tokens), maxnamelen)]    # ensure taken entity label has no more than maximum allowed number of tokens
        maxlen = max(maxlen, len(entity_label_tokens))
        addwords(*entity_label_tokens)
        entmatr.append(entity_label_tokens)
        if entity_id not in entdic:
            entdic[entity_id] = len(entdic)
        if c % 1e3 == 0:
            print("{}k".format(c/1e3))
        c += 1

    entswonames = set()
    def add_entity_wo_label(entid):
        assert(entid not in entdic)
        entdic[entid] = len(entdic)
        entswonames.add(entid)
        entmatr.append(["<RARE>"])

    # relation uri's
    reldic = {}
    relmatr = []
    for line in open(rellist):
        relation_uri = line[:-1]
        relation_uri_tokens = reluri_tokenize(relation_uri)
        relation_uri_tokens = relation_uri_tokens[:min(len(relation_uri_tokens), maxnamelen)]   # ensure max len
        maxlen = max(maxlen, len(relation_uri_tokens))
        addwords(*relation_uri_tokens)
        relmatr.append(relation_uri_tokens)
        if relation_uri not in reldic:
            reldic[relation_uri] = len(reldic)

    maxnamelen = min(maxlen, maxnamelen)

    print len(entdic), len(reldic), len(worddic), maxnamelen

    def getdata(p, maxc=np.infty):
        data = []
        gold = []
        maxlen = 0
        c = 0
        for line in open(p):
            question, answer = line[:-1].split("\t")
            subject, predicate = answer.split(" ")
            question_words = tokenize(question)
            addwords(*question_words)
            maxlen = max(maxlen, len(question_words))
            if subject not in entdic:
                add_entity_wo_label(subject)
            if predicate not in reldic:
                raise Exception("predicate should be there")
            wordidx = [worddic[x] if x in worddic else worddic["<RARE>"] for x in question_words]
            data.append(wordidx)
            gold.append([entdic[subject], reldic[predicate]])
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
        entmat[i, :len(x)] = [worddic[a] for a in x]
    # build rel mat
    relmat = np.zeros((len(relmatr), maxnamelen), dtype="int32") - 1
    for i in range(len(relmatr)):
        x = relmatr[i]
        relmat[i, :len(x)] = [worddic[a] for a in x]

    # package
    entmat = np.concatenate([entmat, relmat], axis=0)
    numents = len(entdic)
    traindata[1][:, 1] += numents
    validdata[1][:, 1] += numents
    testdata[1][:, 1] += numents
    entdic.update({k: v+numents for k, v in reldic.items()})

    # save
    acc = {
        "train": traindata,
        "valid": validdata,
        "test": testdata,
        "worddic": worddic,
        "entdic": entdic,
        "entmat": entmat,
        "numents": numents,
        "wordcounts": wordcounts,
    }

    print("{} entities without names in datasets".format(len(entswonames)))

    pickle.dump(acc, open(outp, "w"))


if __name__ == "__main__":
    argprun(run)