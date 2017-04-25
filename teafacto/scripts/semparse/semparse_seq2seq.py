from teafacto.scripts.semparse.semparse import loadgeo, preprocess, split_train_test
from teafacto.util import argprun, isstring, issequence, ticktock
from teafacto.procutil import wordids2string
import numpy as np, re, random
from IPython import embed

from teafacto.blocks.cnn import CNNSeqEncoder
from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.blocks.seq.encdec import EncDec
from teafacto.blocks.word import WordEmb
from teafacto.blocks.seq.attention import Attention
from teafacto.blocks.basic import SMO
from teafacto.blocks.activations import ReLU, Tanh

from teafacto.core.base import asblock, param, tensorops as T


def concat_generate(qmat, amat, rate=1, maskid=0, concat_original=False):
    questions = []
    answers = []
    maxlen = (0, 0)
    for i in range(len(qmat)):
        for j in range(rate):
            new_q = filter(lambda x: x != maskid, list(qmat[i]))
            new_a = filter(lambda x: x != maskid, list(amat[i]))
            concatid = random.sample(set(range(len(qmat))).difference({i}), 1)[0]
            new_q.extend(filter(lambda x: x != maskid, list(qmat[concatid])))
            new_a.extend(filter(lambda x: x != maskid, list(amat[concatid])))
            maxlen = (max(maxlen[0], len(new_q)), max(maxlen[1], len(new_a)))
            questions.append(new_q)
            answers.append(new_a)
    mat_q = np.ones((len(questions), maxlen[0]), dtype="int32") * maskid
    mat_a = np.ones((len(answers), maxlen[1]), dtype="int32") * maskid
    for i in range(len(questions)):
        q = questions[i]
        a = answers[i]
        mat_q[i, :len(q)] = q
        mat_a[i, :len(a)] = a
    if concat_original:
        qmat = np.concatenate([qmat,
            np.ones((qmat.shape[0], mat_q.shape[1] - qmat.shape[1]), dtype="int32") * maskid], axis=1)
        amat = np.concatenate([amat,
            np.ones((amat.shape[0], mat_a.shape[1] - amat.shape[1]), dtype="int32") * maskid], axis=1)
        qmat = np.concatenate([mat_q, qmat], axis=0)
        amat = np.concatenate([mat_a, amat], axis=0)
    else:
        qmat = mat_q
        amat = mat_a
    return qmat, amat


def run(numbats=50,
        epochs=10,
        lr=0.5,
        expmovavg=0.95,
        embdim=50,
        encdim=200,
        dropout=0.3,
        inconcat=True,
        outconcat=True,
        concatdecinp=False,
        forwardattention=False,     # use forward layer in attention (instead of just dot/cosine)
        splitatt=False,
        gumbelatt=False,
        statetransfer=False,
        preproc=True,
        posembdim=50,
        userelu=False,
        numdeclayers=1,
        concatgen=0,
        inspectdata=False,
        gatedattention=False,
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
    if concatgen > 0:
        tt.tick("concat gen")
        qmat_train, amat_train = concat_generate(qmat_train, amat_train, rate=concatgen, maskid=maskid)
        qmat_test = np.concatenate([qmat_test, np.ones((qmat_test.shape[0], qmat_train.shape[1] - qmat_test.shape[1]),
                                                       dtype="int32") * maskid], axis=1)
        amat_test = np.concatenate([amat_test, np.ones((amat_test.shape[0], amat_train.shape[1] - amat_test.shape[1]),
                                                       dtype="int32") * maskid], axis=1)
        tt.tock("concat genned")

    if inspectdata:
        def pp_train(i):
            print wordids2string(qmat_train[i], {v: k for k, v in qdic.items()}, 0)
            print wordids2string(amat_train[i], {v: k for k, v in adic.items()}, 0)
        embed()

    inpemb = WordEmb(worddic=qdic, maskid=maskid, dim=embdim)
    outemb = WordEmb(worddic=adic, maskid=maskid, dim=embdim)

    encoder = RNNSeqEncoder.fluent()\
        .setembedder(inpemb)\
        .addlayers(dim=encdim, bidir=True, zoneout=dropout)\
        .addlayers(dim=encdim, bidir=False, zoneout=dropout)\
        .make().all_outputs()

    '''
    encoder = CNNSeqEncoder(inpemb=inpemb,
                            numpos=qmat_train.shape[1],
                            posembdim=posembdim,
                            innerdim=[encdim] * 4 if not splitatt else [encdim*2] * 4,
                            window=[3, 3, 5, 5],
                            activation=ReLU if userelu else Tanh,
                            dropout=dropout).all_outputs()'''

    smodim = encdim+encdim if not concatdecinp else encdim+encdim+embdim
    ctxdim = encdim
    critdim = encdim if not concatdecinp else encdim + embdim
    splitters = (asblock(lambda x: x[:, :, :encdim]), asblock(lambda x: x[:, :, encdim:encdim*2]))
    attention = Attention(splitters=splitters) if splitatt else Attention()
    if gumbelatt:
        attention.attentiongenerator.set_sampler("gumbel")
    attention.forward_gen(critdim, ctxdim, encdim) if forwardattention else attention.dot_gen()
    if gatedattention:
        attention.gated_gen(encdim, encdim)

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
                     smo=SMO(smodim, max(adic.values()) + 1))

    tt.tick("training")

    decoder.train([amat_train[:, :-1], qmat_train], amat_train[:, 1:]) \
        .cross_entropy().cross_entropy(cemode="allmean").seq_accuracy().adadelta(lr=lr).grad_total_norm(1.) \
        .validate_on([amat_test[:, :-1], qmat_test], amat_test[:, 1:]) \
        .cross_entropy(cemode="allmean").cross_entropy().seq_accuracy() \
        .train(numbats, epochs)

    tt.tock("trained")
    #embed()

if __name__ == "__main__":
    argprun(run)