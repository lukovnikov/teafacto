from teafacto.scripts.semparse.semparse import loadgeo, preprocess, split_train_test
from teafacto.util import argprun, isstring, issequence, ticktock
from teafacto.procutil import wordids2string
import numpy as np, re, random
from IPython import embed

from teafacto.blocks.cnn import CNNSeqEncoder
from teafacto.blocks.seq.encdec import EncDec
from teafacto.blocks.word import WordEmb
from teafacto.blocks.seq.attention import Attention
from teafacto.blocks.basic import SMO

def run(numbats=50,
        epochs=10,
        lr=0.5,
        embdim=50,
        encdim=200,
        dropout=0.2,
        inconcat=True,
        outconcat=True,
        preproc=True,
        posembdim=50,
        ):
    maskid = 0
    qmat, amat, qdic, adic, qwc, awc = loadgeo(reverse=False)

    def pp(i):
        print wordids2string(qmat[i], {v: k for k, v in qdic.items()}, 0)
        print wordids2string(amat[i], {v: k for k, v in adic.items()}, 0)

    if preproc:
        qmat, amat, qdic, adic, qwc, awc = preprocess(qmat, amat, qdic, adic, qwc, awc, maskid, qreversed=False,
                                                  dorare=True)

    qmat_train, qmat_test = split_train_test(qmat)
    amat_train, amat_test = split_train_test(amat)

    inpemb = WordEmb(worddic=qdic, maskid=maskid, dim=embdim)
    outemb = WordEmb(worddic=adic, maskid=maskid, dim=embdim)

    encoder = CNNSeqEncoder(inpemb=inpemb,
                            numpos=qmat.shape[1],
                            posembdim=posembdim,
                            innerdim=[encdim] * 3,
                            window=[3, 3, 4],
                            dropout=dropout)

    decoder = EncDec(encoder=encoder,
                     attention=Attention().dot_gen(),
                     impemb=outemb,
                     inconcat=inconcat, outconcat=outconcat,
                     innerdim=encdim,
                     dropout_h=dropout,
                     dropout_in=dropout,
                     smo=SMO(encdim+encdim, max(adic.values()) + 1))

    decoder.train([qmat_train, amat_train[:, :-1]], amat_train[:, 1:]) \
        .cross_entropy().adadelta(lr=lr).grad_total_norm(5.) \
        .validate_on([qmat_test, amat_test[:, :-1]], amat_test[:, 1:]) \
        .cross_entropy().accuracy() \
        .train(numbats, epochs)

    embed()

if __name__ == "__main__":
    argprun(run)