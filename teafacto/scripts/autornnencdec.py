from teafacto.blocks.embed import Glove
import pandas as pd, numpy as np, re


def run(
        wreg=0.0,
        epochs=20,
        numbats=100,
        lr=0.01,
        dims=27,
        predicton=None # "../../../data/kaggleai/validation_set.tsv"
    ):
    # get words
    lm = Glove(50)
    words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys()[:10000])
    del lm
    wldf = pd.DataFrame(map(lambda word: [ord(letter)-96 for letter in word], words)).fillna(0)
    data = wldf.values.astype("int32")
    #embed()
    del wldf