from teafacto.blocks.kgraph.fbencdec import FBSeqCompositeEncDec, FBSeqCompositeEncMemDec
from teafacto.blocks.memory import LinearGateMemAddr
from teafacto.feed.freebasefeeders import getentdict, getglovedict, FreebaseSeqFeedMaker, FreebaseSeqFeedMakerEntidxs
from teafacto.feed.langtransform import WordToWordCharTransform
from teafacto.util import argprun, ticktock
import numpy as np
from IPython import embed


def loaddata(glovepath, fbentdicp, fblexpath, wordoffset, numwords, numchars):
    tt = ticktock("fblexdataloader") ; tt.tick()
    gd, vocnumwords = getglovedict(glovepath, offset=wordoffset)
    tt.tock("loaded %d worddic" % len(gd)).tick()
    ed, vocnuments = getentdict(fbentdicp, offset=0)
    tt.tock("loaded %d entdic" % len(ed)).tick()

    indata = FreebaseSeqFeedMakerEntidxs(fblexpath, gd, ed, numwords=numwords, numchars=numchars, unkwordid=wordoffset - 1)
    datanuments = np.max(indata.goldfeed)+1
    tt.tick()
    indata.trainfeed[0:9000]
    tt.tock("transformed")
    #embed()

    traindata = indata.trainfeed
    golddata = indata.goldfeed + 1  # no entity = id 0

    return traindata, golddata, vocnuments, vocnumwords, datanuments+1, ed, gd


def shiftdata(x, right=1):
    if isinstance(x, np.ndarray):
        return np.concatenate([np.zeros_like(x[:, 0:right]), x[:, :-right]], axis=1)
    else:
        raise Exception("can not shift this")


def load_lex_data(lexp, maxid, worddic):     # load the provided (id-based) lexical data up to maxid
    sftrans = WordToWordCharTransform(worddic, unkwordid=1, numwords=20, numchars=30)
    procsf = lambda x: FreebaseSeqFeedMakerEntidxs._process_sf(x, 20, 30)
    c = 0
    with open(lexp) as f:
        coll = [[""]*20]
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
    ret[0, :, :] = 0
    return np.asarray(entids), ret


def run(
        epochs=100,
        lr=0.1,
        wreg=0.0001,
        numbats=50,
        fbdatapath="../../data/mfqa/mfqa.tsv.sample.small",
        fblexpath="../../data/mfqa/mfqa.labels.idx.map",
        glovepath="../../data/glove/glove.6B.50d.txt",
        fbentdicp="../../data/mfqa/mfqa.dic.map",
        numwords=20,
        numchars=30,
        wordembdim=50,
        wordencdim=10,
        entembdim=30,
        innerdim=40,
        attdim=20,
        wordoffset=1,
        validinter=3,
        gradnorm=1.0,
        validsplit=1,
        model="mem",
    ):
    tt = ticktock("fblextransrun")

    traindata, golddata, vocnuments, vocnumwords, datanuments, entdic, worddic = \
        loaddata(glovepath, fbentdicp, fbdatapath, wordoffset, numwords, numchars)
    outdata = shiftdata(golddata)
    tt.tock("made data").tick()
    if model == "mem":
        entids, lexdata = load_lex_data(fblexpath, datanuments, worddic)
        print lexdata.shape
        print datanuments
        #embed()

    # define model
        m = FBSeqCompositeEncMemDec(
            wordembdim=wordembdim,
            wordencdim=wordencdim,
            entembdim=entembdim,
            innerdim=innerdim,
            outdim=datanuments,
            numchars=128,               # ASCII
            numwords=vocnumwords,
            memdata=[entids, lexdata],
            attdim=attdim,
            memaddr=LinearGateMemAddr,
        )
    else:
        m = FBSeqCompositeEncDec(           # compiles, errors go down
            wordembdim=wordembdim,
            wordencdim=wordencdim,
            entembdim=entembdim,
            innerdim=innerdim,
            outdim=datanuments,
            numchars=128,
            numwords=vocnumwords
        )

    reventdic = {}
    for k, v in entdic.items():
        reventdic[v] = k

    #wenc = WordEncoderPlusGlove(numchars=numchars, numwords=vocnumwords, encdim=wordencdim, embdim=wordembdim)
    tt.tock("model defined")
    #embed()
    tt.tick("predicting")
    print traindata[:5].shape, outdata[:5].shape
    pred = m.predict(traindata[:5], outdata[:5])
    print np.argmax(pred, axis=2)-1
    print np.vectorize(lambda x: reventdic[x])(np.argmax(pred, axis=2)-1)
    tt.tock("predicted sample")

    tt.tick("training")
    m.train([traindata, outdata], golddata).adagrad(lr=lr).grad_total_norm(gradnorm).seq_cross_entropy()\
        .autovalidate(splits=validsplit, random=True).validinter(validinter).seq_accuracy()\
        .train(numbats, epochs)
    #embed()

    tt.tock("trained").tick("predicting")
    pred = m.predict(traindata[:50], outdata[:50])
    print np.vectorize(lambda x: reventdic[x])(np.argmax(pred, axis=2)-1)
    tt.tock("predicted sample")


if __name__ == "__main__":
    argprun(run, model="mem")
