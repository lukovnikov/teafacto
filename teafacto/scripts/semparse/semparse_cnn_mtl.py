from teafacto.scripts.semparse.semparse import loadgeo, preprocess, split_train_test
from teafacto.util import argprun, isstring, issequence, ticktock
from teafacto.procutil import wordids2string
import numpy as np, re, random
from IPython import embed

from teafacto.blocks.cnn import CNNSeqEncoder
from teafacto.blocks.seq.encdec import EncDec
from teafacto.blocks.word import WordEmb
from teafacto.blocks.seq.attention import Attention
from teafacto.blocks.basic import SMO, Linear
from teafacto.blocks.activations import ReLU, Tanh

from teafacto.core import asblock, param, T, Block
from teafacto.blocks.loss import CrossEntropy
from teafacto.blocks.activations import GumbelSoftmax, Softmax


class TranslatingAutoencoder(Block):    # block produces losses
    def __init__(self, encdec_one, encdec_two, interembmat, **kw):
        super(TranslatingAutoencoder, self).__init__(**kw)
        self.encdec_one = encdec_one
        self.encdec_two = encdec_two
        self.interembmat = interembmat

    def apply(self, inpseq, outseq, _trainmode=False):
        outdec = self.encdec_one(outseq[:, :-1], inpseq)
        outdec_to_loss = Softmax()(outdec)
        if _trainmode:
            outdec_to_indec = GumbelSoftmax(temperature=0.3)(outdec_to_loss)
            #outdec_to_indec = Softmax()(outdec)
            outdec_to_indec_mask = outdec_to_indec.mask
            outdec_to_indec = T.dot(outdec_to_indec, self.interembmat)
            preconc = self.interembmat[outseq[:, 0]].dimadd(1)
            outdec_to_indec = T.concatenate([preconc, outdec_to_indec], axis=1)
            outdec_to_indec_mask = T.concatenate([T.ones((outdec_to_indec.shape[0], 1), dtype="int8"), outdec_to_indec_mask], axis=1)
            outdec_to_indec.mask = outdec_to_indec_mask
            indec = self.encdec_two(inpseq[:, :-1], outdec_to_indec)
            #indec_to_loss = Softmax()(indec)
            outdec_loss = CrossEntropy()(outdec_to_loss, outseq[:, 1:])
            indec_loss = CrossEntropy()(indec, inpseq[:, 1:])
            loss = outdec_loss + indec_loss
            return loss, outdec_loss, indec_loss
        else:
            return outdec_to_loss


def run(numbats=50,
        epochs=10,
        lr=0.5,
        embdim=50,
        encdim=200,
        dropout=0.3,
        inconcat=True,
        outconcat=True,
        concatdecinp=False,
        forwardattention=False,
        splitatt=False,
        statetransfer=False,
        preproc=True,
        posembdim=50,
        userelu=False,
        numdeclayers=1,
        inspectdata=False,
        ):
    tt = ticktock("script")

    tt.tick("loading data")
    maskid = 0
    qmat, amat, qdic, adic, qwc, awc = loadgeo(reverse=False)
    tt.tock("data loaded")

    def pp(i):
        print wordids2string(qmat[i], {v: k for k, v in qdic.items()}, 0)
        print wordids2string(amat[i], {v: k for k, v in adic.items()}, 0)

    if preproc:
        tt.tick("preprocessing")
        qmat, amat, qdic, adic, qwc, awc = preprocess(qmat, amat, qdic, adic, qwc, awc, maskid, qreversed=False,
                                                  dorare=True)
        tt.tock("preprocessed")

    qmat_train, qmat_test = split_train_test(qmat)
    amat_train, amat_test = split_train_test(amat)

    if inspectdata:
        def pp_train(i):
            print wordids2string(qmat_train[i], {v: k for k, v in qdic.items()}, 0)
            print wordids2string(amat_train[i], {v: k for k, v in adic.items()}, 0)
        embed()

    inpemb = WordEmb(worddic=qdic, maskid=maskid, dim=embdim)
    outemb = WordEmb(worddic=adic, maskid=maskid, dim=embdim)

    encoder = CNNSeqEncoder(inpemb=inpemb,
                            numpos=qmat_train.shape[1],
                            posembdim=posembdim,
                            innerdim=[encdim] * 4 if not splitatt else [encdim*2] * 4,
                            window=[3, 3, 5, 5],
                            activation=ReLU if userelu else Tanh,
                            dropout=dropout).all_outputs()

    smodim = encdim+encdim if not concatdecinp else encdim+encdim+embdim
    ctxdim = encdim
    critdim = encdim if not concatdecinp else encdim + embdim
    splitters = (asblock(lambda x: x[:, :, :encdim]), asblock(lambda x: x[:, :, encdim:encdim*2]))
    attention = Attention(splitters=splitters) if splitatt else Attention()
    attention.forward_gen(critdim, ctxdim, encdim) if forwardattention else attention.dot_gen()

    init_state_gen_mat = param((encdim, encdim), name="init_state_gen_mat").glorotuniform()
    addrportion = slice(None, None, None) if splitatt is False else slice(encdim, encdim*2, None)
    init_state_gen = asblock(lambda x: T.dot(x[:, 0, addrportion], init_state_gen_mat)) if statetransfer else None

    decoder = EncDec(encoder=encoder,
                     attention=attention,
                     inpemb=outemb,
                     indim=embdim+encdim,
                     inconcat=inconcat, outconcat=outconcat, concatdecinp=concatdecinp,
                     innerdim=[encdim]*numdeclayers,
                     init_state_gen=init_state_gen,
                     dropout_h=dropout,
                     dropout_in=dropout,
                     smo=Linear(smodim, max(adic.values()) + 1, nobias=True))

    interembmat = outemb.W

    encoder2 = CNNSeqEncoder(inpemb=False, inpembdim=embdim,
                            numpos=amat_train.shape[1],
                            posembdim=posembdim,
                            innerdim=[encdim] * 4 if not splitatt else [encdim*2] * 4,
                            window=[3, 3, 5, 5],
                            activation=ReLU if userelu else Tanh,
                            dropout=dropout).all_outputs()

    smodim = encdim + encdim if not concatdecinp else encdim + encdim + embdim
    ctxdim = encdim
    critdim = encdim if not concatdecinp else encdim + embdim
    splitters = (asblock(lambda x: x[:, :, :encdim]), asblock(lambda x: x[:, :, encdim:encdim * 2]))
    attention2 = Attention(splitters=splitters) if splitatt else Attention()
    attention2.forward_gen(critdim, ctxdim, encdim) if forwardattention else attention2.dot_gen()

    init_state_gen_mat2 = param((encdim, encdim), name="init_state_gen_mat").glorotuniform()
    addrportion = slice(None, None, None) if splitatt is False else slice(encdim, encdim * 2, None)
    init_state_gen2 = asblock(lambda x: T.dot(x[:, 0, addrportion], init_state_gen_mat2)) if statetransfer else None

    decoder2 = EncDec(encoder=encoder2,
                     attention=attention2,
                     inpemb=inpemb,
                     indim=embdim + encdim,
                     inconcat=inconcat, outconcat=outconcat, concatdecinp=concatdecinp,
                     innerdim=[encdim] * numdeclayers,
                     init_state_gen=init_state_gen2,
                     dropout_h=dropout,
                     dropout_in=dropout,
                     smo=SMO(smodim, max(qdic.values()) + 1))

    b = TranslatingAutoencoder(decoder, decoder2, interembmat)

    tt.tick("training")

    b.train([qmat_train, amat_train]) \
        .model_losses(3).adadelta(lr=lr).grad_total_norm(1.) \
        .validate_on([qmat_test, amat_test], amat_test[:, 1:]) \
        .cross_entropy().seq_accuracy() \
        .train(numbats, epochs)

    tt.tock("trained")
    embed()

if __name__ == "__main__":
    argprun(run)