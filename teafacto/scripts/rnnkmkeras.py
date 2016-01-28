__author__ = 'denis'

from datetime import datetime
import pickle
import os
import math

from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.regularizers import l2
from keras.optimizers import SGD, Adadelta
from keras.callbacks import Callback
from matplotlib import pyplot as plt
import numpy as np
from IPython import embed

from teafacto.core.sptensor import SparseTensor


def loaddata(file):
    file = os.path.join(os.path.dirname(__file__), file)
    st = SparseTensor.from_ssd(file)
    return st

def loadmeta(dfile):
    dfile = os.path.join(os.path.dirname(__file__), dfile)
    meta = pickle.load(open(dfile))
    return meta

def getsavepath():
    dfile = os.path.join(os.path.dirname(__file__), "../../models/%s.%s" %
                         (os.path.splitext(os.path.basename(__file__))[0], datetime.now().strftime("%Y-%m-%d=%H:%M")))
    return dfile

def save(model):
    sp = getsavepath()
    try:
        model.save_weights(sp+".h5")
        open(sp+".json", "w").write(model.to_json())
    except Exception:
        print "some ting wong"

def run():
    # params
    numbats = 1 # 100
    epochs = 5000 #20
    lr = 2./numbats #0.0001 # for SGD
    lr2 = 0.01
    evalinter = 1


    dims = 5#100
    wreg = 0.0# 0.00001

    datafileprefix = "../../data/"
    #datafileprefix = "../../data/nycfilms/"
    tensorfile = "toy.ssd"
    #tensorfile = "tripletensor.ssd"

    # get the data and split
    start = datetime.now()
    data = loaddata(datafileprefix+tensorfile)
    data.threshold(0.5)
    maxentid = max(data.maxid(1), data.maxid(2))
    #data.shiftids(0, maxentid+1)

    vocabsize = data.maxid(0)+1
    data = data.keys.lok

    trainX = data[:, [1, 0]]
    labels = data[:, 2]
    trainY = np.zeros((labels.shape[0], vocabsize)).astype("float32")
    trainY[np.arange(labels.shape[0]), labels] = 1
    batsize=int(math.ceil(data.shape[0]*1.0/numbats))

    print "source data loaded in %f seconds" % (datetime.now() - start).total_seconds()

    # train model
    print "training model"
    start = datetime.now()
    model = Sequential()
    model.add(Embedding(vocabsize, dims, W_regularizer=l2(wreg)))
    model.add(GRU(dims, activation="tanh", ))
    model.add(Dense(vocabsize, W_regularizer=l2(wreg), activation="softmax"))
    opt = SGD(lr=lr2, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adadelta()
    model.compile(optimizer=opt, loss="categorical_crossentropy")
    w = model.get_weights()
    print "model %s defined in %f" % (model.__class__.__name__, (datetime.now() - start).total_seconds())

    start = datetime.now()
    losses = LossHistory()
    model.fit(trainX, trainY, nb_epoch=epochs, batch_size=batsize, verbose=1, callbacks=[losses])
    print "model trained in %f" % (datetime.now() - start).total_seconds()

    print model.predict(np.asarray([[0, 10]]).astype("int32"))

    #print losses.losses
    plt.plot(losses.losses, "r")
    plt.show(block=False)

    save(model)

    embed()


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_begin(self, epoch, logs={}):
        self.losses.append(0)

    def on_batch_end(self, batch, logs={}):
        self.losses[-1] += logs.get('loss')

if __name__ == "__main__":
    run()
