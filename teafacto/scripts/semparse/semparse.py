from teafacto.util import argprun
import numpy as np
from IPython import embed

from teafacto.core.base import Val
from teafacto.blocks.seq.rnn import SeqEncoder, RNNSeqEncoder, SeqDecoder, SeqDecoderAtt
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt


def loadgeo(p="../../../data/semparse/geoquery.txt"):
    qss, ass = [], []
    maxqlen, maxalen = 0, 0
    qwords, awords = {}, {}

    for line in open(p):
        q, a = [x.split(" ") for x in line[:-1].split("\t")]
        q = ["<s>"] + q + ["</s>"]
        a = ["<s>"] + a + ["</s>"]
        qss.append(q)
        ass.append(a)
        maxqlen = max(len(q), maxqlen)
        maxalen = max(len(a), maxalen)
        for qw in q:
            if qw not in qwords:
                qwords[qw] = 0
            qwords[qw] += 1
        for aw in a:
            if aw not in awords:
                awords[aw] = 0
            awords[aw] += 1
    qmat = np.zeros((len(qss), maxqlen), dtype="int32")
    amat = np.zeros((len(ass), maxalen), dtype="int32")
    qdic = dict(zip([x for x, y in sorted(qwords.items(), reverse=True, key=lambda (x, y): y)],
                    range(1, len(qwords) + 1)))
    adic = dict(zip([x for x, y in sorted(awords.items(), reverse=True, key=lambda (x, y): y)],
                    range(1, len(awords) + 1)))
    for i in range(len(qss)):
        q = qss[i]
        a = ass[i]
        qx = [qdic[x] for x in q]
        qx.reverse()
        qx.reverse() #
        qmat[i, :len(q)] = qx
        amat[i, :len(a)] = [adic[x] for x in a]
    return qmat, amat, qdic, adic, qwords, awords


def run(p="m",
        numbats=50,
        epochs=10,
        lr=1.,
        embdim=200,
        encdim=400):
    # loaddata
    qmat, amat, qdic, adic, qwc, awc = loadgeo()

    np.random.seed(1234)

    # make seq/dec+att
    encdec = SimpleSeqEncDecAtt(inpvocsize=len(qdic)+1,
                                inpembdim=embdim,
                                outvocsize=len(qdic)+1, #
                                outembdim=embdim,
                                encdim=encdim,
                                decdim=encdim,
                                maskid=0,
                                statetrans=True)

    encdec.train([qmat, qmat[:, :-1]], qmat[:, 1:])\
        .cross_entropy().adadelta(lr=lr/numbats).grad_total_norm(1.)\
        .split_validate(5).cross_entropy().seq_accuracy()\
        .train(numbats, epochs)

    embed()

if __name__ == "__main__":
    argprun(run)