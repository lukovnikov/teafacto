from teafacto.blocks.kgraph.fbencdec import FBSeqCompositeEncDec, FBSeqCompEncDecAtt
from teafacto.util import argprun, ticktock
import numpy as np
from IPython import embed

from teafacto.scripts.fbmfqalearn import load_lex_data, loaddata, shiftdata

def run(
        epochs=100,
        lr=0.03,
        wreg=0.0001,
        numbats=10,
        fbdatapath="../../data/mfqa/mfqa.tsv.sample",
        fblexpath="../../data/mfqa/mfqa.labels.idx.map",
        glovepath="../../data/glove/glove.6B.50d.txt",
        fbentdicp="../../data/mfqa/mfqa.dic.map",
        numwords=20,
        numchars=30,
        wordembdim=50,
        wordencdim=50,
        entembdim=101,
        innerdim=100,
        attdim=100,
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

    print traindata.shape, golddata.shape
    print validdata.shape, validgold.shape

    if "lex" in model:      # append lexdata
        traindata = np.concatenate([traindata, lexdata], axis=0)
        print traindata.shape
        entids = entids.reshape((entids.shape[0], 1))
        golddata = np.concatenate([golddata, np.concatenate([entids, np.zeros_like(entids, dtype="int32")], axis=1)], axis=0)
        print golddata.shape
    #exit()

    if "att" in model:
        m = FBSeqCompEncDecAtt(
            wordembdim=wordembdim,
            wordencdim=wordencdim,
            entembdim=entembdim,
            innerdim=innerdim,
            outdim=datanuments,
            numchars=128,
            numwords=vocnumwords,
            attdim=attdim,
        )

    else:
        m = FBSeqCompositeEncDec(
            wordembdim=wordembdim,
            wordencdim=wordencdim,
            entembdim=entembdim,
            innerdim=innerdim,
            outdim=datanuments,
            numchars=128,
            numwords=vocnumwords,
        )

    reventdic = {}
    for k, v in entdic.items():
        reventdic[v] = k

    prelex = "lex" in model

    #wenc = WordEncoderPlusGlove(numchars=numchars, numwords=vocnumwords, encdim=wordencdim, embdim=wordembdim)
    tt.tock("model defined")

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
    argprun(run, model="att lex")
