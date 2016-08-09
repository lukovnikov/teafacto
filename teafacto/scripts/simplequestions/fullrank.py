from teafacto.util import argprun, ticktock
import pickle
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
    worddic2 = x["worddic"] if mode == "charword" else None
    entdic = x["entdic"]
    numents = x["numents"]
    entmat = x["entmat"]
    train = x["train"]
    valid = x["valid"]
    test  = x["test"]
    return train, valid, test, worddic, entdic, entmat


def run(
        epochs=50,
        mode="char",    # or "word" or "charword"
        numbats=100,
        lr=0.1,
        wreg=0.000001,
    ):
    # load the right file
    tt = ticktock("script")
    tt.tick()
    (traindata, traingold), (validdata, validgold), (testdata, testgold), \
    worddic, entdic, entmat\
        = readdata(mode)

    print entmat.shape
    print traindata.shape, traingold.shape, testdata.shape, testgold.shape

    tt.tock("data loaded")
    embed()


if __name__ == "__main__":
    argprun(run)