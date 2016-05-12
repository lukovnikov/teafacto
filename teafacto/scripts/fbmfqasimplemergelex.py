from teafacto.blocks.kgraph.fbencdec import FBSeqSimpEncDecAtt
from teafacto.feed.freebasefeeders import FBSeqFeedsMaker, getentdict
from teafacto.feed.langtransform import WordToWordId
from teafacto.util import argprun, ticktock
import numpy as np
from collections import OrderedDict
from IPython import embed


def makeworddict(lexp, datap):
    worddic = OrderedDict()
    maxentid = 0
    with open(datap) as f:
        for line in f:
            ns = line[:-1].lower().split("\t")
            words = ns[0].split(" ")
            for word in words:
                if word not in worddic:
                    worddic[word] = len(worddic) + 1
            ents = map(int, ns[1].split(" "))
            for ent in ents:
                maxentid = max(maxentid, ent)
    with open(lexp) as f:
        for line in f:
            ns = line[:-1].lower().split("\t")
            ent = int(ns[0])
            if ent > maxentid:
                break
            words = ns[1].split(" ")
            for word in words:
                if word not in worddic:
                    worddic[word] = len(worddic) + 1
    return worddic


def loaddata(worddic, fbentdicp, fblexpath, wordoffset, numwords):
    tt = ticktock("fblexdataloader") ; tt.tick()
    ed, vocnuments = getentdict(fbentdicp, offset=0)
    tt.tock("loaded %d entdic" % len(ed)).tick()

    indata = FBSeqFeedsMaker(fblexpath, ed, worddic, numwords=numwords)
    datanuments = np.max(indata.goldfeed)+1
    tt.tick()
    indata.trainfeed[0:9000]
    tt.tock("transformed")
    #embed()

    traindata = indata.trainfeed
    golddata = indata.goldfeed + 1  # no entity = id 0

    return traindata, golddata, vocnuments, len(worddic)+1, datanuments+1, ed


def shiftdata(x, right=1):
    if isinstance(x, np.ndarray):
        return np.concatenate([np.zeros_like(x[:, 0:right]), x[:, :-right]], axis=1)
    else:
        raise Exception("can not shift this")


def load_lex_data(lexp, maxid, worddic):     # load the provided (id-based) lexical data up to maxid
    sftrans = WordToWordId(worddic, numwords=20)
    procsf = lambda x: FBSeqFeedsMaker._process_sf(x, 20)
    c = 0
    with open(lexp) as f:
        coll = [[None]*20]
        entids = [0]
        for line in f:
            line = line.lower()
            idx, sf = line[:-1].split("\t")
            idx = int(idx)+1
            if idx >= maxid:
                break
            entids.append(idx)
            coll.append(procsf(sf))
            c += 1
    ret = sftrans.transform(np.asarray(coll))
    return np.asarray(entids), ret

def run(
        epochs=100,
        lr=0.03,
        wreg=0.0001,
        numbats=10,
        fbdatapath="../../data/mfqa/mfqa.tsv.sample.small",
        fblexpath="../../data/mfqa/mfqa.labels.idx.map",
        fbentdicp="../../data/mfqa/mfqa.dic.map",
        numwords=20,
        wordembdim=50,
        entembdim=101,
        innerdim=100,
        attdim=100,
        wordoffset=1,
        validinter=1,
        gradnorm=1.0,
        validsplit=5,
        model="lex",
    ):
    tt = ticktock("fblextransrun")

    worddic = makeworddict(fblexpath, fbdatapath)

    traindata, golddata, vocnuments, vocnumwords, datanuments, entdic = \
        loaddata(worddic, fbentdicp, fbdatapath, wordoffset, numwords)
    tt.tock("made data").tick()
    entids, lexdata = load_lex_data(fblexpath, datanuments, worddic)


    # manual split # TODO: do split in feeder
    splitpoint = int(traindata.shape[0]*(1. - 1./validsplit))
    print splitpoint
    validdata = traindata[splitpoint:]
    validgold = golddata[splitpoint:]
    traindata = traindata[:splitpoint]
    golddata = golddata[:splitpoint]

    print traindata.shape, golddata.shape
    print validdata.shape, validgold.shape

    if "lex" in model:      # append lexdata
        traindata = np.concatenate([traindata, lexdata], axis=0)
        print traindata.shape
        entids = entids.reshape((entids.shape[0], 1))
        golddata = np.concatenate([golddata, np.concatenate([entids, np.zeros_like(entids, dtype="int32")], axis=1)], axis=0)
        print golddata.shape
    #exit()
    m = FBSeqSimpEncDecAtt(
        wordembdim=wordembdim,
        entembdim=entembdim,
        innerdim=innerdim,
        attdim=attdim,
        outdim=datanuments,
        numwords=vocnumwords,
    )
    tt.tock("model defined")

    reventdic = {}
    for k, v in entdic.items():
        reventdic[v] = k


    # embed()
    outdata = shiftdata(golddata)

    tt.tick("predicting")
    print traindata[:5].shape, outdata[:5].shape
    #print golddata[:5]  ; exit()
    pred = m.predict(traindata[:5], outdata[:5])
    print np.argmax(pred, axis=2) - 1
    print np.vectorize(lambda x: reventdic[x])(np.argmax(pred, axis=2) - 1)
    tt.tock("predicted sample")

    tt.tick("training")
    m.train([traindata, outdata], golddata).adagrad(lr=lr).l2(wreg).grad_total_norm(gradnorm).seq_cross_entropy() \
        .validate_on([validdata, shiftdata(validgold)], validgold).validinter(validinter).seq_accuracy().seq_cross_entropy() \
        .train(numbats, epochs)
    # embed()

    tt.tock("trained").tick("predicting")
    pred = m.predict(validdata, shiftdata(validgold))
    print np.argmax(pred, axis=2) - 1
    #print np.vectorize(lambda x: reventdic[x])(np.argmax(pred, axis=2) - 1)
    tt.tock("predicted sample")


if __name__ == "__main__":
    argprun(run, model="lex")
