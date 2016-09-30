import pandas as pd, numpy as np
from teafacto.util import argprun, issequence

from teafacto.blocks.basic import Linear, Softmax
from teafacto.core.base import Block

class Model(Block):
    def __init__(self, numin, dims, **kw):
        self.layers = []
        if not issequence(dims):
            dims = [dims]

    def apply(self):



def run(datap="../../../data/leafs/train.csv"):
    df = pd.DataFrame.from_csv(datap)
    ul = df["species"].unique()
    labeldic = dict(zip(sorted(ul), range(len(ul))))
    print labeldic
    labels = np.vectorize(lambda x: labeldic[x])(df["species"])
    print labels.shape
    featuremat = df.values[:, 1:]
    print featuremat.shape



if __name__ == "__main__":
    argprun(run)