from teafacto.util import argprun, isstring, issequence
from teafacto.procutil import wordids2string, wordmat2charmat
import numpy as np, re, random
from IPython import embed

from teafacto.core.base import Val, Block, tensorops as T
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.blocks.basic import VectorEmbed, Linear, MatDot
from teafacto.blocks.activations import Softmax, Tanh
from teafacto.blocks.lang.wordvec import WordEmb, Glove
from teafacto.query.lbd2tree import LambdaParser
from teafacto.scripts.semparse.semparse import loadgeo, loadgeoauto, \
    VectorPosEmb, SoftMaxOut, preprocess, gentypdic, generate, GenSample, \
    add_pos_indexes, RandomCorrupt, to_char_level, compute_overlap, \
    do_custom_emb, split_train_test


def run(
        numbats=50,
        epochs=10,
        lr=0.5,
        embdim=50,
        encdim=400,
        dropout=0.2,
        layers=1,
        inconcat=True,
        outconcat=True,
        posemb=False,
        customemb=False,
        preproc="none",     # "none" or "generate" or "abstract" or "gensample"
        bidir=False,
        corruptnoise=0.0,
        inspectdata=False,
        relinearize="none",
        wreg=0.0,
        testmode=False,
        autolr=0.5,
        autonumbats=2000,
        **kw):

    ######### DATA LOADING AND TRANSFORMATIONS ###########
    srctransformer = None
    if relinearize != "none":
        lambdaparser = LambdaParser()
        if relinearize == "greedy":
            def srctransformer(x): return lambdaparser.parse(x).greedy_linearize(deeppref=True)
        elif relinearize == "deep":
            def srctransformer(x): return lambdaparser.parse(x).deep_linearize()
        else:
            raise Exception("unknown linearization")
    adic = {}
    ### AUTO DATA LOAD ###
    qmat_auto, amat_auto, qdic_auto, adic, qwc_auto, awc_auto = \
        loadgeoauto(reverse=True, transformer=srctransformer)
    def pp(i):
        print wordids2string(qmat_auto[i], {v: k for k, v in qdic_auto.items()}, 0)
        print wordids2string(amat_auto[i], {v: k for k, v in adic.items()}, 0)
    if inspectdata:
        print "auto data inspect"
        embed()

    ### TRAIN DATA LOAD ###
    qmat, amat, qdic, adic, qwc, awc = loadgeo(customemb=customemb, reverse=True,
                                               transformer=srctransformer, adic=adic)

    maskid = 0
    typdic = None
    oqmat = qmat.copy()
    oamat = amat.copy()
    print "{} is preproc".format(preproc)
    if preproc != "none":
        qmat, amat, qdic, adic, qwc, awc = preprocess(qmat, amat, qdic, adic, qwc, awc, maskid, qreversed=True, dorare=preproc != "generate")
        if preproc == "generate":   # alters size
            print "generating"
            qmat, amat = generate(qmat, amat, qdic, adic, oqmat, oamat, reversed=True)
            #embed()
        elif preproc == "gensample":
            typdic = gentypdic(qdic, adic)

    ######### train/test split from here #########
    qmat_t, qmat_x = split_train_test(qmat)
    amat_t, amat_x = split_train_test(amat)
    oqmat_t, oqmat_x = split_train_test(oqmat)
    oamat_t, oamat_x = split_train_test(oamat)

    qoverlap, aoverlap, overlap = compute_overlap(qmat_t, amat_t, qmat_x, amat_x)
    print "overlaps: {}, {}: {}".format(len(qoverlap), len(aoverlap), len(overlap))

    if inspectdata:
        embed()

    np.random.seed(12345)

    encdimi = [encdim/2 if bidir else encdim] * layers
    decdimi = [encdim] * layers

    amati_t, amati_x = amat_t, amat_x
    oamati_t, oamati_x = oamat_t, oamat_x
    amati_auto = amat_auto

    if posemb:      # add positional indexes to datamatrices
        qmat_t, oqmat_t, amat_t, oamat_t = add_pos_indexes(qmat_t, oqmat_t, amat_t, oamat_t)
        qmat_x, oqmat_x, amat_x, oamat_x = add_pos_indexes(qmat_x, oqmat_x, amat_x, oamat_x)

    if preproc == "gensample":
        qmat_x, amat_x, amati_x = oqmat_x, oamat_x, oamati_x

    rqdic = {v: k for k, v in qdic.items()}
    radic = {v: k for k, v in adic.items()}

    def tpp(i):
        print wordids2string(qmat_t[i], rqdic, 0)
        print wordids2string(amat_t[i], radic, 0)

    def xpp(i):
        print wordids2string(qmat_x[i], rqdic, 0)
        print wordids2string(amat_x[i], radic, 0)

    if inspectdata:
        embed()
    print "{} training examples".format(qmat_t.shape[0])

    ################## MODEL DEFINITION ##################
    inpemb = WordEmb(worddic=qdic, maskid=maskid, dim=embdim)
    outemb = WordEmb(worddic=adic, maskid=maskid, dim=embdim)

    inpemb_auto = WordEmb(worddic=qdic_auto, maskid=maskid, dim=embdim)

    if customemb:
        inpemb, outemb = do_custom_emb(inpemb, outemb, awc, embdim)
        inpemb_auto, outemb = do_custom_emb(inpemb_auto, outemb, awc_auto, embdim)

    if posemb:  # use custom emb layers, with positional embeddings
        posembdim = 50
        inpemb = VectorPosEmb(inpemb, qmat_t.shape[1], posembdim)
        outemb = VectorPosEmb(outemb, amat_t.shape[1], posembdim)

        inpemb_auto = VectorPosEmb(inpemb_auto, qmat_auto.shape[1], posembdim)
        outemb = VectorPosEmb(outemb, max(amat_auto.shape[1], amat_t.shape[1]), posembdim)

    smodim = embdim
    smo = SoftMaxOut(indim=encdim + encdim, innerdim=smodim,
                     outvocsize=len(adic) + 1, dropout=dropout)

    if customemb:
        smo.setlin2(outemb.baseemb.W.T)

    # main encdec model
    encdec = SimpleSeqEncDecAtt(inpvocsize=max(qdic.values()) + 1,
                                inpembdim=embdim,
                                inpemb=inpemb,
                                outvocsize=max(adic.values()) + 1,
                                outembdim=embdim,
                                outemb=outemb,
                                encdim=encdimi,
                                decdim=decdimi,
                                maskid=maskid,
                                statetrans=True,
                                dropout=dropout,
                                inconcat=inconcat,
                                outconcat=outconcat,
                                rnu=GRU,
                                vecout=smo,
                                bidir=bidir,
                                )

    encdec_auto = SimpleSeqEncDecAtt(inpvocsize=max(qdic_auto.values())+1,
                                     inpembdim=embdim,
                                     inpemb=inpemb_auto,
                                     encdim=encdimi,
                                     decdim=decdimi,
                                     maskid=maskid,
                                     statetrans=True,
                                     dropout=dropout,
                                     inconcat=inconcat,
                                     outconcat=outconcat,
                                     rnu=GRU,
                                     bidir=bidir,
                                     decoder=encdec.dec)

    encdec_params = encdec.get_params()
    encdec_auto_params = encdec_auto.get_params()
    dec_params = encdec.dec.get_params()
    assert(len(encdec_params.intersection(encdec_auto_params).difference(dec_params)) == 0)

    ################## INTERLEAVED TRAINING ##################

    main_trainer = encdec.train([qmat_t, amat_t[:, :-1]], amati_t[:, 1:])\
        .sampletransform(GenSample(typdic),
                         RandomCorrupt(corruptdecoder=(2, max(adic.values()) + 1),
                                       corruptencoder=(2, max(qdic.values()) + 1),
                                       maskid=maskid, p=corruptnoise))\
        .cross_entropy().adadelta(lr=lr).grad_total_norm(5.) \
        .l2(wreg).exp_mov_avg(0.8) \
        .validate_on([qmat_x, amati_x[:, :-1]], amat_x[:, 1:]) \
        .cross_entropy().seq_accuracy()\
        .train_lambda(numbats, 1)

    auto_trainer = encdec_auto.train([qmat_auto, amat_auto[:, :-1]], amati_auto[:, 1:]) \
        .cross_entropy().adadelta(lr=autolr).grad_total_norm(5.) \
        .l2(wreg).exp_mov_avg(0.95) \
        .split_validate(splits=50, random=True).cross_entropy().seq_accuracy()\
        .train_lambda(autonumbats, 1)

    #embed()

    main_trainer.interleave(auto_trainer).train(epochs=100)

    qrwd = {v: k for k, v in qdic.items()}
    arwd = {v: k for k, v in adic.items()}

    def play(*x, **kw):
        hidecorrect = False
        if "hidecorrect" in kw:
            hidecorrect = kw["hidecorrect"]
        if len(x) == 1:
            x = x[0]
            q = wordids2string(qmat_x[x], rwd=qrwd, maskid=maskid, reverse=True)
            ga = wordids2string(amat_x[x, 1:], rwd=arwd, maskid=maskid)
            pred = encdec.predict(qmat_x[x:x+1], amati_x[x:x+1, :-1])
            pa = wordids2string(np.argmax(pred[0], axis=1), rwd=arwd, maskid=maskid)
            if hidecorrect and ga == pa[:len(ga)]:  # correct
                return False
            else:
                print "{}: {}".format(x, q)
                print ga
                print pa
                return True
        elif len(x) == 0:
            for i in range(0, qmat_x.shape[0]):
                r = play(i)
                if r:
                    raw_input()
        else:
            raise Exception("invalid argument to play")
    embed()


if __name__ == "__main__":
    argprun(run)