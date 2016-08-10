from teafacto.util import argprun, ticktock
from teafacto.blocks.seqproc import SimpleSeq2Vec, SeqEncDecAtt, SimpleSeqEncDecAtt, SeqUnroll
from teafacto.blocks.match import SeqMatchScore
from teafacto.core.base import Val, tensorops as T
import pickle, numpy as np
from IPython import embed


def readdata(mode):
    if mode == "char":
        p = "../../../data/simplequestions/datamat.char.mem.fb2m.pkl"
    elif mode == "word":
        p = "../../../data/simplequestions/datamat.word.mem.fb2m.pkl"
    elif mode == "charword":
        p = "../../../data/simplequestions/datamat.charword.mem.fb2m.pkl"
    else:
        raise Exception("unknown mode")
    x = pickle.load(open(p))
    worddic = x["worddic"] if mode == "word" else x["chardic"]
    worddic = {k: v+1 for k, v in worddic.items()}
    worddic["|"] = 0
    worddic2 = x["worddic"] if mode == "charword" else None
    entdic = x["entdic"]
    entdic = {k: v+1 for k, v in entdic.items()}
    entdic["|"] = 0
    numents = x["numents"]+1
    entmat = x["entmat"]
    addtoentmat = -np.ones_like(entmat[[0]], dtype="int32")
    addtoentmat[0] = 0
    entmat = np.concatenate([addtoentmat, entmat], axis=0)
    train = x["train"]
    valid = x["valid"]
    test  = x["test"]
    return train, valid, test, worddic, entdic, entmat


def shiftdata(d):  # idx (batsize, seqlen)
    ds = np.zeros_like(d)
    ds[:, 1:] = d[:, :-1]
    return ds

def run(
        epochs=50,
        mode="char",    # or "word" or "charword"
        numbats=100,
        lr=0.1,
        wreg=0.000001,
        bidir=False,
        layers=1,
        innerdim=200,
        embdim=100,
        negrate=1,
        margin=1.,
        rankingloss=False,
    ):
    # load the right file
    tt = ticktock("script")
    tt.tick()
    (traindata, traingold), (validdata, validgold), (testdata, testgold), \
    worddic, entdic, entmat\
        = readdata(mode)

    reventdic = {v: k for k, v in entdic.items()}
    revworddic = {v: k for k, v in worddic.items()}
    print entmat.shape, reventdic[0], revworddic[0]
    print traindata.shape, traingold.shape, testdata.shape, testgold.shape

    tt.tock("data loaded")

    # *data: matrix of word ids (-1 filler), example per row
    # *gold: vector of true entity ids
    # entmat: matrix of word ids (-1 filler), entity label per row, indexes according to *gold
    # *dic: from word/ent-fbid to integer id, as used in data

    numwords = max(worddic.values()) + 1
    numents = max(entdic.values()) + 1
    print "%d words, %d entities" % (numwords, numents)

    if bidir:
        encinnerdim = [innerdim / 2] * layers
    else:
        encinnerdim = [innerdim] * layers

    memembdim = embdim
    memlayers = layers
    membidir = bidir
    if membidir:
        innerdim = [innerdim/2]*memlayers
    else:
        innerdim = [innerdim]*memlayers

    entenc = SimpleSeq2Vec(indim=numwords,
                         inpembdim=memembdim,
                         innerdim=innerdim,
                         maskid=-1,
                         bidir=membidir)

    encdec = SimpleSeqEncDecAtt(inpvocsize=numwords, inpembdim=embdim,
                    encdim=encinnerdim, bidir=bidir, outembdim=entenc,
                    decdim=innerdim, outconcat=False, vecout=True,
                    statetrans=True)

    scorer = SeqMatchScore(encdec, SeqUnroll(entenc))
    # TODO: test dummy prediction shapes
    dummydata = np.random.randint(0, numwords, (10, 5))
    dummygold = np.random.randint(0, numents, (10, 2))
    dummygoldshifted = shiftdata(dummygold)
    dummypred = scorer.predict((dummydata, dummygoldshifted), dummygold)
    print "DUMMY PREDICTION !!!:"
    print dummypred

    # TODO: below this line, check and test
    class PreProc(object):
        def __init__(self, entmat):
            self.em = Val(entmat)

        def __call__(self, datas, gold):        # gold: idx^(batsize, seqlen)
            x = self.em[gold, :]                # idx^(batsize, seqlen, ...)
            encd = datas[0]
            decd = datas[1]                     # idx (batsize, seqlen, ...)
            y = self.em[decd, :]
            return ((encd, y), x), {}

    class NegIdxGen(object):
        def __init__(self, rng):
            self.min = 0
            self.max = rng

        def __call__(self, datas, gold):    # the whole target sequence is corrupted, corruption targets the whole set of entities and relations together
            return datas, np.random.randint(self.min, self.max, gold.shape).astype("int32")

    obj = lambda p, n: n - p
    if rankingloss:
        obj = lambda p, n: (n - p + margin).clip(0, np.infty)

    traingoldshifted = shiftdata(traingold)
    validgoldshifted = shiftdata(validgold)
    testgoldshifted = shiftdata(testgold)

    nscorer = scorer.nstrain([(traindata, traingoldshifted), traingold]).transform(PreProc(entmat)) \
        .negsamplegen(NegIdxGen(numents)).negrate(negrate).objective(obj) \
        .adagrad(lr=lr).l2(wreg).grad_total_norm(1.0) \
        .validate_on([(validdata, validgoldshifted), validgold]) \
        .train(numbats=numbats, epochs=epochs)

    embed()


if __name__ == "__main__":
    argprun(run)