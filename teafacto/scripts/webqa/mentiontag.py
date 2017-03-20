import json, numpy as np
from teafacto.util import argprun, ticktock
from teafacto.procutil import wordmat2wordchartensor
from IPython import embed
from teafacto.blocks.word import Glove

from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.blocks.basic import SMO
from teafacto.blocks.word import Glove
from teafacto.blocks.word.wordrep import *
from teafacto.core.trainer import ExternalObjective


def loaddata(p="../../../data/semparse/WebQSP/data/"):
    trainp = p + "WebQSP.train.json"
    testp = p + "WebQSP.test.json"
    trainquestions, train_tem = loaddatajson(trainp)
    testquestions, test_tem = loaddatajson(testp)
    trainquestions, train_spans = get_tem_span(trainquestions, train_tem)
    testquestions, test_spans = get_tem_span(testquestions, test_tem)
    wdic = {"<MASK>": 0, "<RARE>": 1}
    wdic, trainquestions, testquestions = dictify(trainquestions, testquestions, wdic=wdic)
    tdic = {"<MASK>": 0, "I": 1, "O": 2}
    train_spans = get_tag_mat(train_spans, tdic)
    test_spans = get_tag_mat(test_spans, tdic)
    #embed()
    trainquestions = wordmat2wordchartensor(trainquestions, worddic=wdic, maskid=0)
    testquestions = wordmat2wordchartensor(testquestions, worddic=wdic, maskid=0)
    return (trainquestions, train_spans), (testquestions, test_spans), (wdic, tdic)


def get_tag_mat(spans, tdic):
    maxlen = max([len(a) for a in spans])
    o = np.ones((len(spans), maxlen), dtype="int32") * (tdic["<MASK>"] if "<MASK>" in tdic else 0)
    for i, span in enumerate(spans):
        o[i, :len(span)] = [tdic[x] for x in span]
    return o


def dictify(*x, **d):
    wdic = d["wdic"] if "wdic" in d else {}
    y = []
    for xe in x:
        ye_maxlen = max([len(a) for a in xe])
        ye = np.ones((len(xe), ye_maxlen), dtype="int32") * (wdic["<MASK>"] if "<MASK>" in wdic else 0)
        for i, xe_s in enumerate(xe):
            for xe_s_w in xe_s:
                if xe_s_w not in wdic:
                    wdic[xe_s_w] = len(wdic)
            ye_s = [wdic[xe_s_w] for xe_s_w in xe_s]
            ye[i, :len(ye_s)] = ye_s
        y.append(ye)
    return (wdic,) + tuple(y)


def get_tem_span(q, tems):
    outq = []
    spans = []
    for i in range(len(q)):
        question = q[i]
        tem = tems[i]
        j = 0
        span = []
        while j <= len(question) - len(tem):
            if question[j: j + len(tem)] == tem:
                span.append(j)
            j += 1
        if len(span) != 1:
            #embed()
            continue
        spann = span[0]
        span = ["O"] * len(question)
        span[spann:spann+len(tem)] = ["I"] * len(tem)
        outq.append(question)
        spans.append(span)
    return outq, spans


def loaddatajson(p):
    j = json.load(open(p))
    j = j["Questions"]
    trainquestions = []
    topicentitymentions = []
    for question in j:
        try:
            trainquestion = question["ProcessedQuestion"]
            topicentitymention = question["Parses"][0]["PotentialTopicEntityMention"]
            if trainquestion is None or topicentitymention is None:
                continue
            trainquestions.append(trainquestion.split())
            topicentitymentions.append(topicentitymention.split())
        except KeyError, e:
            continue
    return trainquestions, topicentitymentions


def processrare(traindata, testdata, glove, rarefreq=1, embtrainfrac=0.0):
    counts = np.unique(traindata, return_counts=True)
    rarewords = set(counts[0][counts[1] <= rarefreq])
    goodwords = set(counts[0][counts[1] > rarefreq])
    traindata[:, :, 0] = np.vectorize(lambda x: glove.D["<RARE>"] if x in rarewords else x)(traindata[:, :, 0])
    if embtrainfrac == 0.0:
        goodwords = goodwords.union(glove.allwords)
    testdata[:, :, 0] = np.vectorize(lambda x: glove.D["<RARE>"] if x not in goodwords else x)(testdata[:, :, 0])
    return traindata, testdata


class SeqTagger(Block):
    def __init__(self, enc, out, **kw):
        super(SeqTagger, self).__init__(**kw)
        self.enc = enc
        self.out = out

    def apply(self, x):     # (batsize, seqlen)
        enco = self.enc(x)  # (batsize, seqlen, dim)
        outo = self.out(enco)
        return outo


class CharEncWrap(Block):
    def __init__(self, charenc, fromdim, todim, **kw):
        super(CharEncWrap, self).__init__(**kw)
        self.enc = charenc
        self.tra = Forward(fromdim, todim, nobias=True)

    def apply(self, x):
        enco = self.enc(x)
        ret = self.tra(enco)
        return ret


def run(
        epochs=25,
        numbats=50,
        lr=0.5,
        gradnorm=5.0,
        dorare=True,
        embdim=50,
        charembdim=50,
        dropout=0.3,
        encdim=200,
        layers=2,
        bidir=True,
        embtrainfrac=1.0,
        mode="words",
        debugvalid=False,
        inspectdata=False,
        skiptraining=False,
    ):
    tt = ticktock("script")
    tt.tick("loading data")
    (traindata, traingold), (testdata, testgold), (wdic, tdic) = loaddata()
    g = Glove(embdim, trainfrac=embtrainfrac, worddic=wdic, maskid=0)
    if dorare:
        traindata, testdata = processrare(traindata, testdata, g, rarefreq=1)
    tt.tock("data loaded")
    if inspectdata:
        embed()

    # BUILD MODEL
    # Choice of word representation
    if mode != "words":
        numchars = traindata[:, :, 1:].max() + 1
        charenc = RNNSeqEncoder.fluent() \
            .vectorembedder(numchars, charembdim, maskid=0) \
            .addlayers(embdim, bidir=True) \
            .make()
        charenc = CharEncWrap(charenc, embdim * 2, embdim)
    if mode == "words":
        emb = g
    elif mode == "concat":
        emb = WordEmbCharEncConcat(g, charenc)
    elif mode == "gate":
        emb = WordEmbCharEncGate(g, charenc, gatedim=embdim, dropout=dropout)
    elif mode == "ctxgate":
        gate_enc = RNNSeqEncoder.fluent() \
            .noembedder(embdim * 2) \
            .addlayers(embdim, bidir=True) \
            .add_forward_layers(embdim, activation=Sigmoid) \
            .make().all_outputs()
        emb = WordEmbCharEncCtxGate(g, charenc, gate_enc=gate_enc)
    else:
        raise Exception("unknown mode in script")
    # tagging model
    enc = RNNSeqEncoder.fluent().setembedder(emb) \
        .addlayers([encdim] * layers, bidir=bidir, dropout_in=dropout).make() \
        .all_outputs()

    # output tagging model
    encoutdim = encdim if not bidir else encdim * 2
    out = SMO(encoutdim, len(tdic), nobias=True)

    # final
    m = SeqTagger(enc, out)

    # TRAINING
    if mode == "words":
        traindata = traindata[:, :, 0]
        testdata = testdata[:, :, 0]
    elif mode == "concat" or mode == "gate" or mode == "ctxgate":
        tt.msg("character-level included")
    else:
        raise Exception("unknown mode in script")

    if not skiptraining:
        m = m.train([traindata], traingold) \
            .cross_entropy().seq_accuracy() \
            .adadelta(lr=lr).grad_total_norm(gradnorm).exp_mov_avg(0.99) \
            .split_validate(splits=10) \
            .cross_entropy().seq_accuracy() \
            .earlystop(select=lambda x: -x[2],
                       stopcrit=7) \
            .train(numbats=numbats, epochs=epochs, _skiptrain=debugvalid)
    else:
        tt.msg("skipping training")

    # EVAL
    pred = m.predict(testdata)
    pred = np.argmax(pred, axis=-1)
    comp = 1 - (pred == testgold)
    comp[testgold == 0] = 0
    aggs = np.sum(comp, axis=-1) == 0
    print "test accuracy:\n {}".format(np.sum(aggs)*1. / len(aggs))

    rwd = {v: k for k, v in wdic.items()}

    testdata = testdata[:, :, 0]

    last_i = [0]

    def play(i=None):
        if i is None:
            i = last_i[0] + 1
        print " ".join([rwd[x] for x in list(testdata[i]) if x != 0])
        print pred[i]
        spanwords = testdata[i] * (pred[i] == 1)
        print " ".join([rwd[x] for x in list(spanwords) if x != 0])
        last_i[0] = i

    embed()




if __name__ == "__main__":
    argprun(run)