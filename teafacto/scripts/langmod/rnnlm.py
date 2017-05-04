# TODO: CHAR LEVEL RNN LM
import h5py, numpy as np
from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.blocks.seq.rnn import MakeRNU, RecStack
from teafacto.blocks.seq.rnu import GRU, PPGRU
from teafacto.blocks.basic import VectorEmbed, SMO
from teafacto.core.base import Block
from teafacto.util import ticktock, argprun
from IPython import embed


def loaddata(p="../../../data/hutter/enwik8.h5", window=200, subsample=1000):
    tt = ticktock("dataloader")
    tt.tick("loading data")
    with h5py.File(p, "r") as f:
        charlist = [x.decode("unicode-escape") for x in list(f["dict"][:, 0])]
        chardic = dict(zip(range(len(charlist)), charlist))
        train, valid, test = f["train"][:], f["valid"][:], f["test"][:]
    tt.tock("data loaded")
    tt.tick("making mats")

    def makemat(data, window, subsample):
        startpositions = np.arange(0, data.shape[0] - window)
        np.random.shuffle(startpositions)
        numex = startpositions.shape[0] // subsample
        startpositions = startpositions[:numex]
        mat = np.zeros((startpositions.shape[0], window), dtype="int32")
        for i in range(startpositions.shape[0]):
            startpos = startpositions[i]
            mat[i, :] = data[startpos:startpos+window]
        return mat

    def pp(charseq):
        if charseq.ndim == 1:
            return "".join([chardic[x] for x in charseq])
        elif charseq.ndim == 2:
            ret = []
            for i in range(len(charseq)):
                ret.append(pp(charseq[i]))
            return ret

    ret = (makemat(train, window, subsample),
           makemat(valid, window, subsample),
           makemat(test, window, subsample),
           chardic, pp)
    tt.tock("made mats")
    return ret


class CLM(Block):
    def __init__(self, emb=None, rnn=None, smo=None, **kw):
        super(CLM, self).__init__(**kw)
        self.emb = emb
        self.rnn = rnn
        self.smo = smo

    def apply(self, x):     # (batsize, seqlen)-charids
        embs = self.emb(x)  # (batsize, seqlen, embdim)
        encos = self.rnn(embs)  # (batsize, seqlen, encdim)
        enco = encos[0][:, -1, :]  # (batsize, encdim)
        out = self.smo(enco)
        return out


def run(window=100, subsample=10000, inspectdata=False,
        embdim=200,
        encdim=300,
        layers=2,
        rnu="gru",      # "gru" or "ppgru"
        zoneout=0.2,
        dropout=0.1,
        lr=0.1,
        gradnorm=5.,
        numbats=100,
        epochs=100,
        ):
    trainmat, validmat, testmat, rcd, pp = loaddata(window=window, subsample=subsample)
    #for x in pp(trainmat[:10]): print x
    if inspectdata:
        embed()

    # config
    vocsize = max(rcd.keys()) + 1
    if rnu == "gru":
        rnu = GRU
    elif rnu == "ppgru":
        rnu = PPGRU

    # make model
    emb = VectorEmbed(vocsize, embdim)
    rnnlayers, lastdim = MakeRNU.fromdims([embdim]+[encdim]*layers, rnu=rnu, zoneout=zoneout, dropout_in=dropout)
    rnn = RecStack(*rnnlayers)
    smo = SMO(encdim, vocsize, dropout=dropout)
    m = CLM(emb, rnn, smo)

    # train model
    m.train([trainmat[:, :-1]], trainmat[:, -1])\
        .cross_entropy().perplexity()\
        .adadelta(lr=lr).grad_total_norm(gradnorm)\
        .validate_on([validmat[:, :-1]], validmat[:, -1])\
        .cross_entropy().perplexity()\
        .train(numbats=numbats, epochs=epochs)


if __name__ == "__main__":
    argprun(run)