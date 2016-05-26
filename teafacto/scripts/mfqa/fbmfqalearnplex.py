import numpy as np

from scripts.mfqa.fbmfqalearn import load_lex_data, loaddata, shiftdata
from teafacto.blocks.kgraph.fbencdec import FBSeqCompositeEncDec, FBSeqCompEncDecAtt
from teafacto.util import argprun, ticktock


def run(
        epochs=100,
        epochsp=10,
        lr=0.03,
        wreg=0.001,
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
        innerdim=400,
        attdim=200,
        wordoffset=1,
        validinter=1,
        gradnorm=1.0,
        validsplit=5,
        vocnumwordsres=50e3,
        model="nomem",
    ):
    tt = ticktock("fblextransrun")

    traindata, golddata, vocnuments, vocnumwords, datanuments, entdic, worddic = \
        loaddata(glovepath, fbentdicp, fbdatapath, wordoffset, numwords, numchars)
    tt.tock("made data").tick()
    entids, lexdata = load_lex_data(fblexpath, datanuments, worddic)

    # manual split # TODO: do split in feeder
    splitpoint = int(traindata.shape[0]*(1. - 1./validsplit))
    print splitpoint
    validdata = traindata[splitpoint:]
    validgold = golddata[splitpoint:]
    traindata = traindata[:splitpoint]
    golddata = golddata[:splitpoint]

    if "att" in model:
        m = FBSeqCompEncDecAtt(
            wordembdim=wordembdim,
            wordencdim=wordencdim,
            entembdim=entembdim,
            innerdim=innerdim,
            outdim=datanuments,
            numchars=128,
            attdim=attdim,
            numwords=vocnumwords
        )
    else:
        m = FBSeqCompositeEncDec(  # compiles, errors go down
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
    if "lex" in model:
        tt.tick("predicting lexicon")
        print lexdata[1:5].shape, entids[1:5].shape, golddata[:5].shape
        #print lexdata[1:5]
        #print entids[:5]; exit()
        pred = m.predict(lexdata[1:5], np.zeros((entids[1:5].shape[0], 1), dtype="int32"))
        print pred.shape
        print np.argmax(pred, axis=2)-1
        print np.vectorize(lambda x: reventdic[x] if x in reventdic else None)(np.argmax(pred, axis=2)-1)
        tt.tock("predicted sample")
        tt.tick("training")
        lextrain = lexdata
        print lextrain.shape
        lexgold = entids.reshape((entids.shape[0], 1))
        print lexgold.shape
        lexgoldshifted = shiftdata(lexgold)
        m.train([lextrain, lexgoldshifted], lexgold).adagrad(lr=lr).seq_cross_entropy().grad_total_norm(gradnorm)\
            .autovalidate(validsplit, random=True).validinter(validinter).seq_accuracy().seq_cross_entropy()\
            .train(numbats, epochsp)

        tt.tick("predicting")
        print lexdata[1:5].shape, entids[1:5].shape, golddata[:5].shape
        # print lexdata[1:5]
        # print entids[:5]; exit()
        pred = m.predict(lexdata[1:5], np.zeros((entids[1:5].shape[0], 1), dtype="int32"))
        print pred.shape
        print np.argmax(pred, axis=2) - 1
        print np.vectorize(lambda x: reventdic[x] if x in reventdic else None)(np.argmax(pred, axis=2) - 1)
        tt.tock("predicted sample")

        m.fixO(lr=0.01)

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
    pred = m.predict(traindata[:50], outdata[:50])
    print np.argmax(pred, axis=2) - 1
    #print np.vectorize(lambda x: reventdic[x])(np.argmax(pred, axis=2) - 1)
    tt.tock("predicted sample")


if __name__ == "__main__":
    argprun(run, model="att lex")
