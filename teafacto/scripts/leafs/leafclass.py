import pandas as pd
from teafacto.util import argprun


def run(datap="../../../data/leafs/train.csv"):
    df = pd.DataFrame.from_csv(datap)
    print df["species"].value_counts()



if __name__ == "__main__":
    argprun(run)