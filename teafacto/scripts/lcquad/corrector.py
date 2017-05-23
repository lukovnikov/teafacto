from teafacto.util import argprun, ticktock
import json, numpy as np
from unidecode import unidecode
from IPython import embed

from teafacto.blocks.seq.encdec import EncDec
from teafacto.blocks.seq.rnn import SeqEncoder
from teafacto.blocks.basic import VectorEmbed, SMO
from teafacto.blocks.seq.attention import Attention


def load_data(p="../../../data/lcquad/data_v2.json"):
    data = json.load(open(p))
    l = []
    for question in data:
        q, a = question["verbalized_question"], question["corrected_answer"]
        q = unidecode(q)
        a = unidecode(a)
        l.append((q, a))
    rows = []
    maxes = [0, 0]
    for q, a in l:
        q = [1] + map(ord, q) + [2]
        a = [1] + map(ord, a) + [2]
        maxes = [max(maxes[0], len(q)), max(maxes[1], len(a))]
        rows.append((q, a))
    def pp(inp):
        return "".join(map(lambda x: chr(x) if x != 0 else "", inp))
    qmat, amat = 0 * np.ones((len(rows), maxes[0]), dtype="int32"), 0 * np.ones((len(rows), maxes[1]), dtype="int32")
    c = 0
    for q, a in rows:
        qmat[c, :len(q)] = q
        amat[c, :len(a)] = a
        c += 1
    # dictionary
    uniquechars = set(np.unique(qmat)) & set(np.unique(amat))
    dic = dict([(chr(char), char) for char in uniquechars])

    # split
    idx = len(qmat) // 5 * 4
    qmat_train = qmat[:idx]
    amat_train = amat[:idx]
    qmat_test = qmat[idx:]
    amat_test = amat[idx:]

    return qmat_train, amat_train, qmat_test, amat_test, dic, pp


def run(
        lr=0.1,
        numbats=600,
        epochs=100,
        embdim=100,
        encdim=200,
        dropout=0.2,
        inspectdata=False,
):
    qmat_train, amat_train, qmat_test, amat_test, dic, pp = load_data()
    vocsize = max(dic.values())
    if inspectdata:
        embed()

    # build model
    emb = VectorEmbed(indim=vocsize, dim=embdim, maskid=0)
    encoder = SeqEncoder.fluent()\
        .setembedder(emb)\
        .addlayers(dim=encdim, bidir=True, dropout_in=dropout, zoneout=dropout)\
        .addlayers(dim=encdim, bidir=False, dropout_in=dropout, zoneout=dropout)\
        .make().all_outputs()
    attention = Attention()
    decoder = EncDec(encoder=encoder,
                     attention=attention,
                     inconcat=False,
                     outconcat=True,
                     inpemb=emb,
                     innerdim=[encdim, encdim],
                     dropout_in=dropout,
                     zoneout=dropout,
                     smo=SMO(encdim, vocsize)
                     )

    # train
    decoder.train([qmat_train, amat_train[:, :-1]], amat_train[:, 1:])\
        .seq_cross_entropy().seq_accuracy()\
        .adadelta(lr=lr).grad_total_norm(5.)\
        .validate_on([qmat_test, amat_test[:, :-1]], amat_test[:, 1:])\
        .seq_cross_entropy().seq_accuracy()\
        .train(numbats, epochs)



if __name__ == "__main__":
    argprun(run)