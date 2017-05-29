# TODO: pretraining entity, relation etc. representations

from teafacto.blocks.basic import VectorEmbed, SMO
from teafacto.blocks.seq.autoencoder import *
from teafacto.blocks.loss import *
from teafacto.blocks.seq.rnn import SeqEncoder, SimpleRNNSeqDecoder
from teafacto.util import argprun, ticktock
import numpy as np, pickle
from IPython import embed


def build_model(vocsize=None, embdim=None, innerdim=None, bidir=False,
                dropout_in=False, dropout_h=False, zoneout=False,
                dropout_auto=False, maskid=0):
    embedder = VectorEmbed(vocsize, embdim, maskid=maskid)
    smo = SMO(innerdim[-1], vocsize)
    encoder = SeqEncoder.fluent()\
        .setembedder(embedder)\
        .addlayers(dim=innerdim, bidir=bidir,
            dropout_in=dropout_in, dropout_h=dropout_h, zoneout=zoneout)\
        .make()

    ctxdim = innerdim[-1] if not bidir else innerdim[-1] * 2
    decoder = SimpleRNNSeqDecoder(emb=embedder, ctxdim=ctxdim, softmaxoutblock=smo,
                                  inconcat=True, outconcat=False, innerdim=innerdim,
                                  dropout_h=dropout_h, dropout=dropout_in, zoneout=zoneout)

    autoencoder = SeqAutoEncoder(encoder, decoder, dropout=dropout_auto)
    return autoencoder


def make_dummy_data():
    vocsize = 200
    seqlen = 10
    numex = 1000
    data = np.random.randint(0, vocsize, (numex, seqlen))
    traindata, validdata, testdata = data[:800], data[800:900], data[900:]
    return traindata, validdata, testdata, vocsize


def run_dummy(
        lr=0.1,
        embdim=100,
        innerdim=200,
        numlayers=1,
        dropout=0.2,
        numbats=100,
        epochs=100,
        inspectdata=False,
        ):
    traindata1, validdata1, testdata1, vocsize = make_dummy_data()
    traindata2, validdata2, testdata2, vocsize = make_dummy_data()

    if inspectdata:
        embed()

    # make model
    m1 = build_model(vocsize, embdim=embdim, innerdim=[innerdim]*numlayers,
                    dropout_auto=dropout)
    m2 = build_model(vocsize, embdim=embdim, innerdim=[innerdim] * numlayers,
                     dropout_auto=dropout)
    encoders = [m1.encoder, m2.encoder]
    decoders = [m1.decoder, m2.decoder]

    m = MultiSeqAutoEncoder(encoders, decoders, dropout=dropout, mode="sum")

    loss = MTL_Loss([CrossEntropy(), CrossEntropy()])

    b = loss.apply_on(m)

    predm = m.get_encoder()

    prediction = predm.predict(traindata1[:5], traindata2[:5])
    print prediction.shape

    try:
        b.train([traindata1, traindata2, traindata1[:, 1:], traindata2[:, 1:]])\
            .model_loss()\
            .adadelta(lr=lr).grad_total_norm(5.)\
            .validate_on([validdata1, validdata2, validdata1[:, 1:], validdata2[:, 1:]])\
            .model_loss()\
            .train(numbats, epochs)
    except KeyboardInterrupt, e:
        print "keyboard interrupt"
        embed()


def compute_encodings(predm, data, batsize=100):
    predf = predm.predict
    ptr = 0
    data0 = data[0]
    ret = []
    while ptr < len(data0):
        ptrto = min(batsize + ptr, len(data0))
        dataslices = [d[ptr:ptrto] for d in data]
        slicepred = predf(*dataslices)
        ret.append(slicepred)   # (batsize, ...)
        ptr += batsize
    ret = np.concatenate(ret, axis=0)
    return ret


def load_data(p="../../../data/WebQSP/data/WebQSP.canids.allinfo.pkl"):
    tt = ticktock("data loader")
    tt.tick("loading")
    info = pickle.load(open(p))
    tt.tock("loaded")
    seqs = []
    uniquewords = {"<MASK>", "<RARE>"}
    tt.tick("processing")
    maxlabelcharlen = 0
    maxlabelwordlen = 0
    maxtypelabelwordlen = 0
    for sf in info.keys():
        label = sf[1]
        if len(label) == 0:
            label = "<RARE>"
            print "label is rare: {}".format(sf[0])
        maxlabelcharlen = max(maxlabelcharlen, len(label))
        label = label.split()
        maxlabelwordlen = max(maxlabelwordlen, len(label))
        typelabel = sf[3]
        if len(typelabel) == 0:
            typelabel = sf[4]
        if len(typelabel) == 0:
            typelabel = "<RARE>"
        typelabel = typelabel.split()
        seqs.append((label, typelabel))
        maxtypelabelwordlen = max(maxtypelabelwordlen, len(typelabel))
        uniquewords.update(set(label))
        uniquewords.update(set(typelabel))
    tt.tock("processed")
    dic = {"<MASK>": 0, "<RARE>": 1}
    uniquewords = uniquewords.difference(dic.keys())
    dic.update(dict(zip(uniquewords, range(2, len(uniquewords) + 2))))
    # init matrices
    tt.tick("building matrices")
    labelcharmat = np.ones((len(info), maxlabelcharlen), dtype="int32") * dic["<MASK>"]
    labelwordmat = np.ones((len(info), maxlabelwordlen), dtype="int32") * dic["<MASK>"]
    typelabelwordmat = np.ones((len(info), maxtypelabelwordlen), dtype="int32") * dic["<MASK>"]
    for c, seq in enumerate(seqs):
        labelchars = [ord(x) for x in " ".join(seq[0])]
        labelwords = [dic[x] for x in seq[0]]
        typewords = [dic[x] for x in seq[1]]
        labelcharmat[c, :len(labelchars)] = labelchars
        labelwordmat[c, :len(labelwords)] = labelwords
        typelabelwordmat[c, :len(typewords)] = typewords
    tt.tock("matrices built")
    rdic = {v: k for k, v in dic.items()}
    def pp(c):
        chars = "".join([chr(x) for x in labelcharmat[c] if x != dic["<MASK>"]])
        words = " ".join([rdic[x] for x in labelwordmat[c] if x != dic["<MASK>"]])
        typew = " ".join([rdic[x] for x in typelabelwordmat[c] if x != dic["<MASK>"]])
        return chars, words, typew
    return labelcharmat, labelwordmat, typelabelwordmat, dic


def run(
        lr=0.1,
        wordembdim=200,
        charembdim=100,
        innerdim=400,
        numlayers=1,
        charnumlayers=2,
        dropout=0.2,
        batsize=100,
        epochs=100,
        inspectdata=False,
):
    labelchars, labelwords, typewords, worddic = load_data()
    print labelchars.shape, labelwords.shape, typewords.shape
    numwords = max(worddic.values()) + 1
    numchars = 128
    maskid = worddic["<MASK>"]
    numex = len(labelchars)
    numbats = numex // batsize + (1 if numex % batsize > 0 else 0)
    if inspectdata:
        embed()

    labelcharautoenc = build_model(numchars, charembdim, [innerdim]*charnumlayers,
                                   bidir=False, zoneout=dropout, dropout_in=dropout,
                                   maskid=maskid)

    labelwordautoenc = build_model(numwords, wordembdim, [innerdim]*numlayers,
                                   bidir=False, zoneout=dropout, dropout_in=dropout,
                                   maskid=maskid)

    typelabelwordautoenc = build_model(numwords, wordembdim, [innerdim]*numlayers,
                                       bidir=False, zoneout=dropout, dropout_in=dropout,
                                       maskid=maskid)

    encoders = [labelcharautoenc.encoder,
                labelwordautoenc.encoder,
                typelabelwordautoenc.encoder]

    decoders = [labelcharautoenc.decoder,
                labelwordautoenc.decoder,
                typelabelwordautoenc.decoder]

    autoenc = MultiSeqAutoEncoder(encoders, decoders,
                                  dropout=dropout, mode="sum")

    loss = MTL_Loss([CrossEntropy(), CrossEntropy(), CrossEntropy()])

    m = loss.apply_on(autoenc)

    predm = autoenc.get_encoder()

    try:
        m.train([labelchars, labelwords, typewords,
                 labelchars[:, 1:], labelwords[:, 1:], typewords[:, 1:]])\
            .model_loss()\
            .adadelta(lr=lr).grad_total_norm(5.)\
            .train(numbats, epochs)
    except KeyboardInterrupt, e:
        print "KEYBOARD INTERRUPT"
        embed()

    # TODO: start from pretrained glove embeddings (all unknown words to <RARE>)

    embed()


if __name__ == "__main__":
    argprun(run_dummy)


