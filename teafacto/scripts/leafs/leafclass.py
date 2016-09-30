import pandas as pd, numpy as np
from teafacto.util import argprun, issequence

from teafacto.blocks.basic import Linear, Softmax
from teafacto.core.base import Block

class Model(Block):
    def __init__(self, numin, *dims, **kw):
        super(Model, self).__init__(**kw)
        self.layers = []
        dims = list(dims)
        dims = [numin] + dims
        for i in range(1, len(dims)):
            self.layers.append(Linear(indim=dims[i-1], dim=dims[i]))
        self.layers.append(Softmax())

    def apply(self, x):
        acc = x
        for layer in self.layers:
            acc = layer(acc)
        return acc



def run(datap="../../../data/leafs/train.csv",
        testp="../../../data/leafs/test.csv",
        lr=0.1,
        numbats=100,
        epochs=100):
    df = pd.DataFrame.from_csv(datap)
    ul = df["species"].unique()
    labeldic = dict(zip(sorted(ul), range(len(ul))))
    print labeldic
    labels = np.vectorize(lambda x: labeldic[x])(df["species"]).astype("int32")
    print labels.shape
    featuremat = df.values[:, 1:].astype("float32")
    print featuremat.shape

    m = Model(featuremat.shape[1], 100)

    m.train([featuremat], labels).adagrad(lr=lr).cross_entropy()\
        .split_validate(splits=5, random=True).cross_entropy().accuracy()\
        .train(numbats=numbats, epochs=epochs)

    df = pd.DataFrame.from_csv(testp)
    featuremat = df.values.astype("float32")

    predprobs = m.predict(featuremat)
    preds = np.argmax(predprobs, axis=1)

    outdf = pd.DataFrame(data=predprobs)
    outdf.columns = sorted(labeldic.keys())
    outdf.index = df.index

    print outdf



if __name__ == "__main__":
    argprun(run)