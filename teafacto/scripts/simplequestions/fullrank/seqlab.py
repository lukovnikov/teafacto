from teafacto.blocks.seq.trans import SimpleSeqTrans
from teafacto.core.base import Block, asblock
from teafacto.blocks.basic import VectorEmbed
from teafacto.blocks.lang.wordvec import Glove
import numpy as np, pickle
from IPython import embed
from teafacto.scripts.simplequestions.fullrank.fullrank import readdata
from teafacto.util import argprun


def preproc():
    ((traindata, traingold), (validdata, validgold), (testdata, testgold),
     (subjmat, relmat), (subjdic, reldic), worddic, subjinfo, (testsubjcans, relspersubj)) \
        = readdata(numtestcans=5, wordlevel=True)

    maskid = -1

    def getlabeldatafrom(data, gold):
        inp = []
        out = []
        for i in range(data.shape[0]):
            goldname = subjmat[gold[i, 0]]
            goldname = filter(lambda x: x!=maskid, list(goldname))
            for j in range(data.shape[1]):
                if len(goldname) + j > data.shape[1]:
                    continue
                datapart = list(data[i, j:j+len(goldname)])
                if datapart == goldname:
                    inprow = data[i]
                    outrow = inprow + 0
                    outrow[outrow > -1] = 0
                    outrow[j:j+len(goldname)] = 1
                    inp.append(inprow[np.newaxis, :])
                    out.append(outrow[np.newaxis, :])
                    break
        inp = np.concatenate(inp, axis=0)
        out = np.concatenate(out, axis=0).astype("int8")
        return inp, out

    trainlabeldata, trainlabelgold = getlabeldatafrom(traindata, traingold)
    validlabeldata, validlabelgold = getlabeldatafrom(validdata, validgold)
    testlabeldata, testlabelgold = getlabeldatafrom(testdata, testgold)

    #embed()
    save = {"train": (trainlabeldata, trainlabelgold),
            "valid": (validlabeldata, validlabelgold),
            "test": (testlabeldata, testlabelgold),
            "worddic": worddic}
    pickle.dump(save, open("seqlab.data", "w"))


def run(lr=1.0, epochs=50, numbats=700):
    d = pickle.load(open("seqlab.data"))
    worddic = d["worddic"]
    embdim = 50
    maskid = -1
    emb = Glove(embdim, maskid=maskid).adapt(worddic)
    b = SimpleSeqTrans(inpemb=emb, bidir=True, innerdim=[100, 100], outdim=2)

    traindata, traingold = d["train"]
    validdata, validgold = d["valid"]
    traingold, validgold = traingold.astype("int32"), validgold.astype("int32")

    b.train([traindata], traingold).seq_cross_entropy().adadelta(lr=lr)\
        .validate_on([validdata], validgold).seq_cross_entropy().seq_accuracy()\
        .train(epochs=epochs, numbats=numbats)


if __name__ == "__main__":
    preproc()
    argprun(run)
