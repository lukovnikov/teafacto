from teafacto.blocks.kgraph.fbencdec import FBSeqCompositeEncDec
from teafacto.feed.freebasefeeders import getentdict, getglovedict, FreebaseSeqFeedMaker
from teafacto.util import argprun, ticktock
import numpy as np


def loaddata(glovepath, fbentdicp, fblexpath, wordoffset, numwords, numchars):
    tt = ticktock("fblexdataloader") ; tt.tick()
    gd, vocnumwords = getglovedict(glovepath, offset=wordoffset)
    tt.tock("loaded %d worddic" % len(gd)).tick()
    ed, vocnuments = getentdict(fbentdicp, offset=0)
    tt.tock("loaded %d entdic" % len(ed)).tick()

    indata = FreebaseSeqFeedMaker(fblexpath, gd, ed, numwords=numwords, numchars=numchars, unkwordid=wordoffset - 1)
    datanuments = np.max(indata.goldfeed)+1
    tt.tick()
    print "max entity id+1: %d" % datanuments
    indata.trainfeed[0:9000]
    tt.tock("transformed")
    #embed()

    traindata = indata.trainfeed
    golddata = indata.goldfeed

    return traindata, golddata, vocnuments, vocnumwords, datanuments

def shiftdata(x, right=1):
    if isinstance(x, np.ndarray):
        return np.concatenate([np.zeros_like(x[:, 0:right]), x[:, :-right]], axis=1)
    else:
        raise Exception("can not shift this")

def run(
        epochs=100,
        lr=0.005,
        wreg=0.0001,
        numbats=10,
        fblexpath="../../data/mfqa/mfqa.tsv.sample.small",
        glovepath="../../data/glove/glove.6B.50d.txt",
        fbentdicp="../../data/mfqa/mfqa.dic.map",
        numwords=20,
        numchars=30,
        wordembdim=50,
        wordencdim=100,
        entembdim=300,
        innerdim=300,
        wordoffset=1,
        validinter=3,
        gradnorm=1.0,
        validsplit=100,
    ):
    tt = ticktock("fblextransrun")

    traindata, golddata, vocnuments, vocnumwords, datanuments = \
        loaddata(glovepath, fbentdicp, fblexpath, wordoffset, numwords, numchars)
    golddata = golddata + 1
    datanuments += 1
    outdata = shiftdata(golddata)
    tt.tock("made data").tick()

    # define model
    m = FBSeqCompositeEncDec(
        wordembdim=wordembdim,
        wordencdim=wordencdim,
        entembdim=entembdim,
        innerdim=innerdim,
        outdim=datanuments,
        numchars=128,               # ASCII
        numwords=vocnumwords,
    )

    #wenc = WordEncoderPlusGlove(numchars=numchars, numwords=vocnumwords, encdim=wordencdim, embdim=wordembdim)
    tt.tock("model defined")
    # train model   TODO
    tt.tick("training")
    m.train([traindata, outdata], golddata).adagrad(lr=lr).grad_total_norm(gradnorm).seq_neg_log_prob()\
        .autovalidate(splits=validsplit, random=True).validinter(validinter).seq_accuracy()\
        .train(numbats, epochs)
    #embed()
    tt.tock("trained").tick("predicting")
    print m.predict(traindata).shape
    tt.tock("predicted sample")


if __name__ == "__main__":
    argprun(run)
