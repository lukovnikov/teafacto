from teafacto.blocks.kgraph.fbencdec import FBBasicCompositeEncoder
from teafacto.feeders.freebasefeeders import getentdict, getglovedict, FBLexDataFeedsMaker
from teafacto.util import argprun, ticktock


def run(
        epochs=100,
        lr=0.5,
        wreg=0.0001,
        numbats=100,
        fblexpath="../../data/freebase/labelsrevlex.map.sample",
        glovepath="../../data/glove/glove.6B.50d.txt",
        fbentdicp="../../data/freebase/entdic.all.map",
        numwords=10,
        numchars=30,
        wordembdim=50,
        wordencdim=100,
        innerdim=300,
        wordoffset=1,
        validinter=3,
        gradnorm=1.0,
    ):
    gd, vocnumwords = getglovedict(glovepath, offset=wordoffset)
    print gd["alias"]
    ed, vocnuments = getentdict(fbentdicp, offset=0)
    print ed["m.0ndj09y"]

    indata = FBLexDataFeedsMaker(fblexpath, gd, ed, numwords=numwords, numchars=numchars, unkwordid=wordoffset-1)
    datanuments = max(indata.goldfeed)+1
    tt = ticktock("fblextransrun")
    tt.tick()
    print "max entity id+1: %d" % datanuments
    indata.trainfeed[0:9]
    tt.tock("transformed")
    #embed()

    traindata = indata.trainfeed
    golddata = indata.goldfeed

    # define model
    m = FBBasicCompositeEncoder(
        wordembdim=wordembdim,
        wordencdim=wordencdim,
        innerdim=innerdim,
        outdim=datanuments,
        numchars=128,               # ASCII
        numwords=vocnumwords,
    )

    #wenc = WordEncoderPlusGlove(numchars=numchars, numwords=vocnumwords, encdim=wordencdim, embdim=wordembdim)

    # train model   TODO
    print "training"
    m.train([traindata], golddata).adagrad(lr=lr).grad_total_norm(gradnorm).neg_log_prob()\
        .autovalidate().validinter(validinter).accuracy()\
        .train(numbats, epochs)
    #embed()
    tt.tick("predicting")
    print m.predict(traindata).shape
    tt.tock("predicted sample")


if __name__ == "__main__":
    argprun(run)
