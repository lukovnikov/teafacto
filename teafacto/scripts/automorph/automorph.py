from teafacto.blocks.seq.memnn import AutoMorph
from teafacto.blocks.basic import VectorEmbed, SMO, SMOWrap
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
        embdim=50,
        keyencdim="100",
        valencdim="200",
        outdim=300,
        memdim=100,
        dropout=0.3,
        ):
    traindata, testdata, cdic = loaddata(window=window)
    numchars = max(np.max(traindata), np.max(testdata)) + 1
    rcd = {v: k for k, v in cdic.items()}

    def pp(s):
        print "".join([rcd[x] for x in s])

    if inspectdata:
        embed()

    keydims = [int(x) for x in keyencdim.split(",")]
    valdims = [int(x) for x in valencdim.split(",")]

    charemb = VectorEmbed(numchars, embdim)
    am = AutoMorph(memlen=memlen, memkeydim=memdim, memvaldim=memdim,
                  charencdim=keydims, morfencdim=valdims,
                  charemb=charemb, outdim=outdim)

    m = SMOWrap(am, outdim=numchars, nobias=True)

    m.train([traindata[:, :-1]], traindata[:, 1:])\
        .cross_entropy(cemode="mean").adadelta(lr=lr)\
        .validate_on([testdata[:, :-1]], testdata[:, 1:])\
        .cross_entropy(cemode="mean")\
        .train(numbats=numbats, epochs=epochs)


if __name__ == "__main__":
    argprun(run)

