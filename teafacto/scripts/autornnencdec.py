from teafacto.blocks.embed import Glove
from teafacto.blocks.memnet import RNNAutoEncoder
from teafacto.blocks.rnn import RNNDecoder, RNNEncoder
from teafacto.blocks.rnu import GRU
from teafacto.core.base import Block
from teafacto.blocks.basic import MatDot as Lin, Softmax
from teafacto.util import argparsify
import pandas as pd, numpy as np, re
from IPython import embed

''' THIS SCRIPT TESTS TRAINING RNN ENCODER, RNN DECODER AND RNN AUTOENCODER '''

class vec2sec(Block):
    def __init__(self, indim=50, innerdim=300, seqlen=20, vocsize=27, **kw):
        super(vec2sec, self).__init__(**kw)
        self.indim = indim
        self.innerdim=innerdim
        self.seqlen = seqlen
        self.vocsize = vocsize
        self.lin = Lin(indim=self.indim, dim=self.innerdim)
        self.dec = RNNDecoder(                                  # IdxToOneHot inserted automatically
            GRU(dim=self.vocsize, innerdim=self.innerdim),      # the decoding RNU
            Lin(indim=self.innerdim, dim=self.vocsize),         # transforms from RNU inner dims to vocabulary
            Softmax(), dim=self.vocsize)                        # softmax

    def apply(self, vec):
        return self.dec(self.lin(vec))


def run(
        wreg=0.0,
        epochs=20,
        numbats=100,
        lr=0.01,
        dims=27,
        predicton=None,
        statedim = 300,
    ):
    # get words
    lm = Glove(50, 10000)
    words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
    wldf = pd.DataFrame(map(lambda word: [ord(letter)-96 for letter in word], words)).fillna(0)
    data = wldf.values.astype("int32")
    embs = lm.W
    del lm
    embed()
    del wldf


if __name__ == "__main__":
    run(**argparsify(run))