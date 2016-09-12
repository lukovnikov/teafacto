import pickle, numpy as np

from teafacto.util import argprun, ticktock
from teafacto.scripts.simplequestions.classify.predpredkeras import readdata
from teafacto.blocks.seq.enc import SimpleSeq2Idx
from IPython import embed

def run(epochs=10,
        numbats=700,
        lr=0.1,
        embdim=200,
        encdim=300,
        layers=1,
        clean=False,
        rarefreq=4,
        type="rnn",  # rnn or cnn
        p="../../data/simplequestions/datamat.word.mem.fb2m.pkl",
        ):
    # load data for classification
    tt = ticktock("script")
    tt.tick("loading data")
    (traindata, traingold), (validdata, validgold), (testdata, testgold), \
        entdic, entmat, worddic, numents = readdata(p, clean=clean, rarefreq=rarefreq)
    tt.tock("loaded data")
    # model
    tt.tick("building model")

    m = SimpleSeq2Idx(indim=len(worddic)+1, inpembdim=embdim, numclasses=len(entdic),
                      innerdim=encdim, maskid=0, layers=layers)
    tt.tock("built model")
    tt.tick("training")
    m.train([traindata], traingold).adadelta(lr=lr).cross_entropy().grad_total_norm(1.)\
        .validate_on([validdata], validgold).cross_entropy().accuracy().takebest()\
        .train(numbats=numbats, epochs=epochs)
    tt.tock("trained")
    tt.tick("testing")
    preds = m.predict(testdata)
    acc = preds == testgold
    acc = np.sum(acc) * 1.0 / testdata.shape[0]
    print("Accuracy: {}".format(acc))
    tt.tock("tested")

if __name__ == "__main__":
    argprun(run)
