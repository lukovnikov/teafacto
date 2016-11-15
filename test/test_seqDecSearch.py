from unittest import TestCase

import numpy as np
import pandas as pd
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.use.recsearch import GreedySearch


def word2int(word):
    return [ord(letter)-96 if letter is not " " else 0 for letter in word]


def words2ints(words):
    wldf = pd.DataFrame(map(word2int, words)).fillna(0)
    data = wldf.values.astype("int32")
    del wldf
    return data


def int2word(ints):
    chars = [chr(i+96) if i > 0 else " " for i in ints]
    return "".join(chars)


def ints2words(ints):
    return [int2word(x) for x in ints]


def shiftdata(x):
    return np.concatenate([np.zeros_like(x[:, 0:1]), x[:, :-1]], axis=1)


class TestSeqDecSearch(TestCase):
    def test_seqdecatt(self,
            statedim=50,
            encdim=50,
            attdim=50,
            startsym=0,
    ):
        # get words
        vocsize = 27

        testpred = ["the", "alias", "mock", "test", "stalin", "allahuakbar", "python", "pythonista",
                    " "]
        testpred = words2ints(testpred)
        print testpred

        block = SimpleSeqEncDecAtt(inpvocsize=vocsize, outvocsize=vocsize,
                                   encdim=encdim, decdim=statedim,
                                   attdim=attdim, inconcat=False,
                                   maskid=0)

        s = GreedySearch(block, startsymbol=startsym, maxlen=testpred.shape[1])
        s.init(testpred, testpred.shape[0])
        ctxmask, ctx = s.wrapped.recpred.nonseqvals
        print ctxmask
        self.assertTrue(np.all(ctxmask == (testpred > 0)))
        pred, probs = s.search(testpred.shape[0])
        print ints2words(pred), probs
