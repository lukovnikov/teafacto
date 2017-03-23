from teafacto.blocks.seq.memnn import AutoMorph
from teafacto.blocks.seq.marnu import ReGRU
from teafacto.blocks.basic import VectorEmbed, SMO, SMOWrap
from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.util import argprun, ticktock
from IPython import embed
import numpy as np


def loaddata(p="../../../data/langmod/shakespear.txt", window=50, split=10):
    oneline = ""
    with open(p) as f:
        for line in f:
            oneline += line
    allchars = set(oneline)
    dic = dict(zip(list(allchars), range(len(allchars))))
    mat = np.array([dic[x] for x in oneline])

    # GENERATE WINDOWS
    outmat = np.zeros((len(mat) - window, window), dtype="int32")
    for i in range(len(outmat)):
        outmat[i, :] = mat[i:i+window]

    splitidxs = np.arange(0, len(outmat))
    #np.random.shuffle(splitidxs)

    testmat = outmat[splitidxs[:len(splitidxs)/split]]
    trainmat = outmat[splitidxs[len(splitidxs)/split:]]

    return trainmat, testmat, dic


def run(epochs=10,
        numbats=1000,
        window=50,
        inspectdata=False,
        lr=0.5,
        memlen=1000,
        chardims="50,100,100",
        morfdims="100,200",
        dropout=0.3,
        mode="morf",     # "rnn" or "morf" or "retardis"
        memsize=10,
        sampleaddr=False,
        cemode="mean",   # "mean" or "sum"
        ):
    traindata, testdata, cdic = loaddata(window=window)
    numchars = max(np.max(traindata), np.max(testdata)) + 1
    rcd = {v: k for k, v in cdic.items()}

    def pp(s):
        print "".join([rcd[x] for x in s])

    if inspectdata:
        embed()

    keydims = [int(x) for x in chardims.split(",")]
    valdims = [int(x) for x in morfdims.split(",")]

    embdim = keydims[0]
    memkeydim = keydims[-1]
    keydims = keydims[1:-1]

    memvaldim = valdims[0]
    outdim = valdims[-1]
    valdims = valdims[1:-1]

    charemb = VectorEmbed(numchars, embdim)
    if mode == "morf":
        print "doing automorf"
        am = AutoMorph(memlen=memlen, memkeydim=memkeydim, memvaldim=memvaldim,
                      charencdim=keydims, morfencdim=valdims,
                      charemb=charemb, outdim=outdim, sampleaddr=sampleaddr)
    elif mode == "rnn":
        am = RNNSeqEncoder.fluent().setembedder(charemb)\
            .addlayers(keydims+[memvaldim]+valdims+[outdim])\
            .make().all_outputs()
    elif mode == "retardis":
        dims = [embdim] + keydims + [memvaldim] + valdims + [outdim]
        rs = []
        for i in range(len(dims) - 1):
            rs.append(ReGRU(dims[i], dims[i+1], memsize=memsize, dropout_h=False, dropout_in=dropout))
        am = RNNSeqEncoder.fluent().setembedder(charemb)\
            .setlayers(*rs).make().all_outputs()
    else:
        raise Exception("unknown mode: {}".format(mode))

    m = SMOWrap(am, outdim=numchars, inneroutdim=outdim, nobias=True)

    m.train([traindata[:, :-1]], traindata[:, 1:])\
        .cross_entropy(cemode=cemode).cross_entropy().cross_entropy(cemode="mean")\
        .adadelta(lr=lr)\
        .validate_on([testdata[:, :-1]], testdata[:, 1:])\
        .cross_entropy(cemode=cemode).cross_entropy().cross_entropy(cemode="mean")\
        .train(numbats=numbats, epochs=epochs)


if __name__ == "__main__":
    argprun(run)

