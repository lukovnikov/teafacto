from teafacto.scripts.semparse.semparse import loadgeo, preprocess, split_train_test
from teafacto.util import argprun, isstring, issequence, ticktock
from teafacto.procutil import wordids2string
import numpy as np, re, random
from IPython import embed

from teafacto.blocks.cnn import CNNSeqEncoder
from teafacto.blocks.seq.rnu import QRNU, MIGRU, mGRU, GRU, FlatMuFuRU, MuFuRU, PPGRU
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
        epochs=100,
        lr=0.1,
        expmovavg=0.95,
        embdim=50,
        encdim=300,
        decdim=300,
        dropout=0.2,
        gradnorm=5.,
        inconcat=True,
        outconcat=True,
        concatdecinp=False,
        attdist="dot",     # "dot" or "fwd" or "eucl" or "l1norm"
        splitatt=False,
        gumbelatt=False,
        hardatt=False,
        gatedattention=False,       # uses gated euclidean distance in attgen
        transattention=False,
        statetransfer=False,
        preproc=True,
        posembdim=50,
        userelu=False,
        numdeclayers=1,
        concatgen=0,
        inspectdata=False,
        mode="gru",      # "gru" or "qrnn" or "cnn" or "migru" or "mgru" or "mufuru" or "ppgru"
        ):
    tt = ticktock("script")

    np.random.seed(1337)

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

    if mode == "qrnn":
        encoder = RNNSeqEncoder.fluent()\
            .setembedder(inpemb)\
            .add_layer(QRNU(window_size=3, dim=embdim, innerdim=encdim,
                            zoneout=dropout), encdim)\
            .add_layer(QRNU(window_size=3, dim=encdim, innerdim=encdim,
                            zoneout=dropout), encdim)\
            .make().all_outputs()
    elif mode == "gru" or mode == "migru" or mode == "mgru" or mode == "mufuru" or mode == "ppgru":
        rnu = GRU
        if mode == "migru":
            rnu = MIGRU
        elif mode == "mgru":
            rnu = mGRU
        elif mode == "mufuru":
            rnu = MuFuRU
        elif mode == "ppgru":
            rnu = PPGRU
        encoder = RNNSeqEncoder.fluent()\
            .setembedder(inpemb)\
            .addlayers(dim=encdim, bidir=True, zoneout=dropout, rnu=rnu)\
            .addlayers(dim=encdim, bidir=False, zoneout=dropout, rnu=rnu)\
            .make().all_outputs()
    elif mode == "cnn":
        encoder = CNNSeqEncoder(inpemb=inpemb,
                                numpos=qmat_train.shape[1],
                                posembdim=posembdim,
                                innerdim=[encdim] * 4 if not splitatt else [encdim*2] * 4,
                                window=[3, 3, 5, 5],
                                activation=ReLU if userelu else Tanh,
                                dropout=dropout).all_outputs()
    else:
        raise Exception("unknown encoder mode")

    smodim = encdim+decdim if not concatdecinp else encdim+decdim+embdim
    ctxdim = encdim
    critdim = decdim if not concatdecinp else decdim + embdim
    splitters = (asblock(lambda x: x[:, :, :encdim]), asblock(lambda x: x[:, :, encdim:encdim*2]))
    attention = Attention(splitters=splitters) if splitatt else Attention()
    if gumbelatt:
        attention.attentiongenerator.set_sampler("gumbel")
    if gatedattention:
        attention.gated_gen(critdim, ctxdim)
    if transattention:
        attention.crit_trans_gen(critdim, ctxdim)
    if attdist == "dot":
        attention.dot_gen()
    elif attdist == "fwd":
        attention.forward_gen(critdim, ctxdim, decdim)
    elif attdist == "eucl":
        attention.eucl_gen()
    elif attdist == "l1norm":
        attention.lnorm_gen(L=1)
    else:
        raise Exception("unrecognized attention distance")

    init_state_gen_mat = param((encdim, decdim), name="init_state_gen_mat").glorotuniform()
    addrportion = slice(None, None, None) if splitatt is False else slice(encdim, encdim*2, None)
    init_state_gen = asblock(lambda x: T.dot(x[:, 0, addrportion], init_state_gen_mat)) if statetransfer else None

    rnu = GRU
    if mode == "migru":
        rnu = MIGRU
    elif mode == "mgru":
        rnu = mGRU
    elif mode == "mufuru":
        rnu = MuFuRU
    elif mode == "ppgru":
        rnu = GRU

    decoder = EncDec(encoder=encoder,
                     attention=attention,
                     inpemb=outemb,
                     indim=embdim+decdim,
                     inconcat=inconcat, outconcat=outconcat, concatdecinp=concatdecinp,
                     innerdim=[decdim]*numdeclayers,
                     init_state_gen=init_state_gen,
                     zoneout=dropout,
                     dropout_in=dropout,
                     rnu=rnu,
                     smo=SMO(smodim, max(adic.values()) + 1))

    tt.tick("training")

    decoder.train([amat_train[:, :-1], qmat_train], amat_train[:, 1:]) \
        .cross_entropy().cross_entropy(cemode="allmean").seq_accuracy()\
        .adadelta(lr=lr).grad_total_norm(gradnorm) \
        .validate_on([amat_test[:, :-1], qmat_test], amat_test[:, 1:]) \
        .cross_entropy().cross_entropy(cemode="allmean").seq_accuracy() \
        .train(numbats, epochs)

    tt.tock("trained")
    #embed()

if __name__ == "__main__":
    argprun(run)