import pickle, numpy as np

from keras.models import Sequential
from keras.layers import LSTM, GRU, Embedding, Dense, Activation
from keras.optimizers import Adadelta

from teafacto.util import argprun
from teafacto.scripts.simplequestions.fullrank import readdata
from IPython import embed

def run(epochs=10,
        batsize=100,
        lr=0.1,
        embdim=200,
        encdim=300,
        layers=1,
        p="../../data/simplequestions/datamat.word.mem.fb2m.pkl",
        ):
    # load data for classification
    with open(p) as f:
        x = pickle.load(f)
        traindata, traingold = x["train"]
        traindata += 1
        validdata, validgold = x["valid"]
        validdata += 1
        testdata, testgold = x["test"]
        testdata += 1
        worddic = x["worddic"]
        numents = x["numents"]
        worddic = {k: v+1 for k, v in worddic.items()}
        rwd = {v: k for k, v in worddic.items()}
        entmat = x["entmat"]
        entmat = entmat[numents:, :]
        entmat += 1
        entdic = x["entdic"]
        entdic = {k: v - numents for k, v in entdic.items() if v >= numents}
        traingold = traingold[:, [1]] - numents
        validgold = validgold[:, 1] - numents
        testgold = testgold[:, 1] - numents
        def pp(idseq):
            print " ".join([rwd[k] if k in rwd else ""
            if k == 0 else "<???>" for k in idseq])

        #embed()
        print traindata.shape, traingold.shape

    # model
    m = Sequential()
    m.add(Embedding(len(worddic), embdim, mask_zero=True))
    for i in range(layers - 1):
        m.add(GRU(encdim, return_sequences=True))
    m.add(GRU(encdim, return_sequences=False))
    m.add(Dense(len(entdic)))
    m.add(Activation("softmax"))

    m.compile(loss="categorical_crossentropy",
              optimizer=Adadelta(),
              metrics=["accuracy"])

    m.fit(traindata, traingold, nb_epoch=epochs, batch_size=batsize,
          validation_data=(validdata, validgold))


if __name__ == "__main__":
    argprun(run)
