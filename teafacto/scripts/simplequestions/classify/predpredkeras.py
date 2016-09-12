import pickle, numpy as np

from keras.models import Sequential
from keras.layers import LSTM, GRU, Embedding, Dense, Activation, Convolution1D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.engine import Layer, InputSpec
from keras import backend as K

from teafacto.util import argprun, ticktock
from IPython import embed

class _GlobalPooling1D(Layer):

    def __init__(self, **kwargs):
        super(_GlobalPooling1D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]
        #self.supports_masking = True

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        raise NotImplementedError


class GlobalMaxPooling1D(_GlobalPooling1D):
    '''Global average pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    '''

    def compute_mask(self, x, mask):
        return None

    def call(self, x, mask=None):
        ret = K.max(x, axis=1)
        return ret
        #sum = K.sum(x, axis=1)      # (samples, features)
        #tot = K.sum(mask, axis=1) if mask is not None else x.shape[1]  # (samples, )
        #return sum / tot


def cleandata(traindata, validdata=None, testdata=None, dic=None, rarefreq=5):
    counts = np.bincount(traindata.flatten())     # !no neg ids! counts of words in traindata
    retain = set(list(np.argwhere(counts > rarefreq)[:, 0]))
    dropper = np.vectorize(lambda x: x if x in retain else dic["<RARE>"])
    traindata = dropper(traindata)
    validdata = dropper(validdata) if validdata is not None else validdata
    testdata = dropper(testdata) if testdata is not None else testdata
    outdic = {k: v for k, v in dic.items() if v in retain}
    #embed()
    return traindata, validdata, testdata, outdic


def readdata(p, clean=False, rarefreq=4):
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
        worddic = {k: v + 1 for k, v in worddic.items()}
        rwd = {v: k for k, v in worddic.items()}
        entmat = x["entmat"]
        entmat = entmat[numents:, :]
        entmat += 1
        entdic = x["entdic"]
        entdic = {k: v - numents for k, v in entdic.items() if v >= numents}
        traingold = traingold[:, 1] - numents
        validgold = validgold[:, 1] - numents
        testgold = testgold[:, 1] - numents

        def pp(idseq):
            print " ".join([rwd[k] if k in rwd else ""
            if k == 0 else "<???>" for k in idseq])

        # embed()
        print traindata.shape, traingold.shape
        if clean:
            traindata, validdata, testdata, worddic = cleandata(traindata, validdata=validdata, testdata=testdata, dic=worddic, rarefreq=rarefreq)
        return (traindata, traingold), (validdata, validgold), (testdata, testgold), entdic, entmat, worddic, numents

def run(epochs=10,
        batsize=100,
        lr=0.1,
        embdim=200,
        encdim=300,
        layers=1,
        type="rnn",  # rnn or cnn
        clean=False,
        rarefreq=4,
        p="../../data/simplequestions/datamat.word.mem.fb2m.pkl",
        ):
    # load data for classification
    tt = ticktock("script")
    tt.tick("loading data")
    (traindata, traingold), (validdata, validgold), (testdata, testgold), \
        entdic, entmat, worddic, numents = readdata(p, clean=clean, rarefreq=rarefreq)
    numrels = len(entdic)
    traingold = np_utils.to_categorical(traingold, nb_classes=numrels)
    validgold = np_utils.to_categorical(validgold, nb_classes=numrels)
    testgold = np_utils.to_categorical(testgold, nb_classes=numrels)
    tt.tock("loaded data")
    # model
    tt.tick("building model")
    m = Sequential()
    m.add(Embedding(len(worddic)+1, embdim, mask_zero=True))
    if type == "rnn":
        print("doing RNN")
        for i in range(layers - 1):
            m.add(GRU(encdim, return_sequences=True))
        m.add(GRU(encdim, return_sequences=False))
    elif type == "cnn":
        print("doing CNN")
        for i in range(layers):
            m.add(Convolution1D(encdim, encdim))
        m.add(GlobalMaxPooling1D())
    m.add(Dense(len(entdic)))
    m.add(Activation("softmax"))

    m.compile(loss="categorical_crossentropy",
              optimizer=Adadelta(lr=lr),
              metrics=["accuracy"])
    tt.tock("built model")
    tt.tick("training")
    m.fit(traindata, traingold, nb_epoch=epochs, batch_size=batsize,
          validation_data=(validdata, validgold))
    tt.tock("trained")
    tt.tick("testing")
    score, acc = m.evaluate(testdata, testgold, batch_size=batsize)
    print("Score: {}\nAccuracy: {}".format(score, acc))
    tt.tock("tested")

if __name__ == "__main__":
    argprun(run)
