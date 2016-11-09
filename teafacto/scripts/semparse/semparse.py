from teafacto.util import argprun
import numpy as np, re
from IPython import embed

from teafacto.core.base import Val, Block, tensorops as T
from teafacto.blocks.seq.rnn import SeqEncoder, RNNSeqEncoder, SeqDecoder, SeqDecoderAtt
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.blocks.basic import VectorEmbed, Linear
from teafacto.blocks.activations import Softmax, Tanh


def loadgeo(p="../../../data/semparse/geoquery.txt"):
    qss, ass = [], []
    maxqlen, maxalen = 0, 0
    qwords, awords = {}, {}

    for line in open(p):
        q, a = [re.split("[\s-]", x) for x in line[:-1].split("\t")]
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
        qmat[i, :len(q)] = qx
        amat[i, :len(a)] = [adic[x] for x in a]
    return qmat, amat, qdic, adic, qwords, awords


class VectorPosEmb(Block):
    def __init__(self, vocsize, embdim, numpos, posembdim, maskid=-1, **kw):
        super(VectorPosEmb, self).__init__(**kw)
        self.wemb = VectorEmbed(indim=vocsize, dim=embdim, maskid=maskid)
        self.pemb = VectorEmbed(indim=numpos, dim=posembdim)
        self.outdim = self.wemb.outdim + self.pemb.outdim
        self.indim = self.wemb.indim

    def apply(self, x):     # (batsize, seqlen, 2)
        wembeddings = self.wemb(x[:, :, 0])
        pembeddings = self.pemb(x[:, :, 1])
        ret = T.concatenate([wembeddings, pembeddings], axis=2)     # (batsize, seqlen, wembdim+pembdim)
        ret.mask = wembeddings.mask
        return ret


class SoftMaxOut(Block):
    def __init__(self, indim=None, innerdim=None, outvocsize=None, dropout=None, **kw):
        super(SoftMaxOut, self).__init__(**kw)
        self.lin1 = Linear(indim=indim, dim=innerdim, dropout=dropout)
        self.lin2 = Linear(indim=innerdim, dim=outvocsize)

    def apply(self, x):
        a = self.lin1(x)
        b = Tanh()(a)
        c = self.lin2(b)
        d = Softmax()(c)
        return d


def run(
        numbats=50,
        epochs=10,
        lr=1.,
        embdim=200,
        encdim=400,
        dropout=0.5,
        layers=1,
        posemb=False,
        inconcat=True):
    # loaddata
    qmat, amat, qdic, adic, qwc, awc = loadgeo()

    #embed()

    np.random.seed(1234)
    encdimi = [encdim] * layers
    decdimi = [encdim] * layers

    inpemb = None   # normal args are used
    outemb = None   # normal args are used

    maskid = 0

    if posemb:      # custom emb layers, with positional embeddings
        posembdim = 50
        inpemb = VectorPosEmb(len(qdic)+1, embdim, qmat.shape[1], posembdim, maskid=maskid)
        outemb = VectorPosEmb(len(adic)+1, embdim, amat.shape[1], posembdim, maskid=maskid)

    smo = SoftMaxOut(indim=encdim+encdim, innerdim=encdim, outvocsize=len(adic)+1, dropout=dropout)

    # make seq/dec+att
    encdec = SimpleSeqEncDecAtt(inpvocsize=len(qdic)+1,
                                inpembdim=embdim,
                                inpemb=inpemb,
                                outvocsize=len(adic)+1,
                                outembdim=embdim,
                                outemb=outemb,
                                encdim=encdimi,
                                decdim=decdimi,
                                maskid=maskid,
                                statetrans=True,
                                dropout=dropout,
                                inconcat=inconcat,
                                outconcat=True,
                                rnu=GRU,
                                vecout=smo,
                                )

    amati = amat

    if posemb:
        qposmat = np.arange(0, qmat.shape[1])[None, :]
        qposmat = np.repeat(qposmat, qmat.shape[0], axis=0)
        qmat = np.concatenate([qmat[:, :, None], qposmat[:, :, None]], axis=2)
        aposmat = np.arange(0, amat.shape[1])[None, :]
        aposmat = np.repeat(aposmat, amat.shape[0], axis=0)
        amati = np.concatenate([amat[:, :, None], aposmat[:, :, None]], axis=2)

    encdec.train([qmat, amati[:, :-1]], amat[:, 1:])\
        .cross_entropy().rmsprop(lr=lr/numbats).grad_total_norm(1.)\
        .split_validate(5).cross_entropy().seq_accuracy()\
        .train(numbats, epochs)

    embed()

if __name__ == "__main__":
    argprun(run)