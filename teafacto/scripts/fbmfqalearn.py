from teafacto.blocks.kgraph.fbencdec import FBSeqCompositeEncDec, FBSeqCompositeEncMemDec, FBMemMatch
from teafacto.blocks.memory import LinearGateMemAddr, GeneralDotMemAddr
from teafacto.blocks.lang.wordvec import Glove
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
    ret[0, :, :] = 0
    return np.asarray(entids), ret


def run(
        epochs=100,
        lr=0.01,
        wreg=0.0001,
        numbats=10,
        fbdatapath="../../data/mfqa/mfqa.tsv.sample.small",
        fblexpath="../../data/mfqa/mfqa.labels.idx.map",
        glovepath="../../data/glove/glove.6B.50d.txt",
        fbentdicp="../../data/mfqa/mfqa.dic.map",
        numwords=20,
        numchars=30,
        wordembdim=50,
        wordencdim=100,
        entembdim=100,
        innerdim=200,
        attdim=200,
        wordoffset=1,
        validinter=1,
        gradnorm=1.0,
        validsplit=1,
        vocnumwordsres=50e3,
        model="mem",
    ):
    tt = ticktock("fblextransrun")

    traindata, golddata, vocnuments, vocnumwords, datanuments, entdic, worddic = \
        loaddata(glovepath, fbentdicp, fbdatapath, wordoffset, numwords, numchars)
    outdata = shiftdata(golddata)
    tt.tock("made data").tick()
    entids, lexdata = load_lex_data(fblexpath, datanuments, worddic)
    if model == "mem":
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
    elif model=="lex":          # for testing purposes
        print lexdata.shape
        print datanuments
        #vocnumwords = 4000
        #exit()
        #embed()
        m = FBMemMatch(
            wordembdim=wordembdim,
            wordencdim=wordencdim,
            entembdim=entembdim,
            innerdim=innerdim,
            outdim=datanuments,
            numchars=128,
            numwords=vocnumwords,
            memdata=[entids, lexdata],
            attdim=attdim,
        )

    elif model=="nomem":
        m = FBSeqCompositeEncDec(           # compiles, errors go down
            wordembdim=wordembdim,
            wordencdim=wordencdim,
            entembdim=entembdim,
            innerdim=innerdim,
            outdim=datanuments,
            numchars=128,
            numwords=vocnumwords
        )
    else:
        m = None
        print "no such model"
    reventdic = {}
    for k, v in entdic.items():
        reventdic[v] = k

    #wenc = WordEncoderPlusGlove(numchars=numchars, numwords=vocnumwords, encdim=wordencdim, embdim=wordembdim)
    tt.tock("model defined")
    if model == "lex": # for testing purposes
        tt.tick("predicting")
        print lexdata[1:5].shape, entids[1:5].shape
        #print lexdata[1:5]
        print entids[1:5]
        pred = m.predict(lexdata[1:5])
        print pred.shape
        print np.argmax(pred, axis=1)-1
        print np.vectorize(lambda x: reventdic[x] if x in reventdic else None)(np.argmax(pred, axis=1)-1)
        tt.tock("predicted sample")
        tt.tick("training")
        m.train([lexdata[1:151]], entids[1:151]).adagrad(lr=lr).cross_entropy().grad_total_norm(0.5)\
            .split_validate(5, random=True).validinter(validinter).accuracy().train(numbats, epochs)
    else:
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
    argprun(run, model="nomem")
