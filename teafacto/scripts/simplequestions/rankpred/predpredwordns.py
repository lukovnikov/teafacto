from teafacto.util import argprun, ticktock
import numpy as np, os, sys, math, pickle
from IPython import embed

def readdata(p="../../../data/simplequestions/clean/datamat.word.fb2m.pkl"):
    x = pickle.load(open(p))
    worddic = x["worddic"]
    entdic = x["entdic"]
    numents = x["numents"]
    entmat = x["entmat"]
    train = x["train"]
    valid = x["valid"]
    test = x["test"]

    return train, valid, test, worddic, entdic, entmat, numents

def run(epochs=50,
        numbats=700,
        lr=1.,
        wreg=0.000001,
        bidir=False,
        layers=1,
        embdim=200,
        encdim=200,
        negrate=1.,
        margin=1.,
        hingeloss=False,
        debug=False,
        ):
