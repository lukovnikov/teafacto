import re

import pandas as pd
from teafacto.lm import Glove

from blocks.rnn import *
from teafacto.core.base import *
from teafacto.core.base import tensorops as T
from util import argparsify


class RNNEncDecSM(SeqSMBase, Predictor, Saveable):
    def __init__(self, indim=50, innerdim=50, hlimit=50, **w):
        super(RNNEncDecSM, self).__init__(**w)
        self.indim = indim
        self.innerdim = innerdim
        encrnu = GRU(dim=indim, innerdim=innerdim, wreg=self.wreg)
        decrnu = GRU(dim=indim, innerdim=innerdim, wreg=self.wreg)
        self.W = T.eye(indim, indim)
        self.O = theano.shared(random((innerdim, indim))) # from vector to symbols
        self.encoder = RNNEncoder() + encrnu
        decstack = RecurrentStack(indim,
            lambda x: self.W[x, :],
            decrnu,
            lambda x: T.dot(x, self.O),
            lambda x: T.nnet.softmax(x),
        )
        self.decoder = RNNDecoder(hlimit=hlimit) + decstack

    def defmodel(self):
        inp = T.imatrix("inp") # indexes (batsize, seqlen)
        outp = T.imatrix("outp") # indexes (batsize, seqlen)
        encoding = self.encoder.encode(self.W[inp, :])
        decoding = self.decoder.decode([encoding])
        return [decoding, outp, [inp, outp]]


    def getpredictfunction(self):
        pass

    def getsamplegen(self, data, labels, onebatch=False):
        if onebatch:                                      # works correctly (DONE: shapes inspected)
            batsize = data.shape[0]
        else:
            batsize = self.batsize
        dataidxs = np.arange(data.shape[0])
        def samplegen():
            np.random.shuffle(dataidxs)
            sampleidxs = dataidxs[:batsize]
            return data[sampleidxs, :], labels[sampleidxs, :]
        return samplegen

    @property
    def depparameters(self):
        return self.encoder.parameters.union(self.decoder.parameters)

    @property
    def ownparameters(self):
        return {self.O}



def run(
        wreg=0.0,
        epochs=20,
        numbats=100,
        lr=0.01,
        dims=27,
        predicton=None # "../../../data/kaggleai/validation_set.tsv"
    ):
    # get words
    lm = Glove(50)
    words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys()[:10000])
    del lm
    wldf = pd.DataFrame(map(lambda word: [ord(letter)-96 for letter in word], words)).fillna(0)
    data = wldf.values.astype("int32")
    #embed()
    del wldf

    encdec = RNNEncDecSM(indim=dims, innerdim=100, numbats=numbats, maxiter=epochs, hlimit=data.shape[1])\
             + SGD(lr=lr)
    encdec.train(data, data)

if __name__ == "__main__":
    run(**argparsify(run))