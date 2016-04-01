from unittest import TestCase

import math
import numpy as np
import pandas as pd
import re

from teafacto.blocks.lang.wordembed import Glove
from teafacto.scripts.autornnencdec import seq2idx, idx2seq, shiftdata


class TestVecSeqIdx(TestCase):
    def setUp(self):
        self.encdim = 44
        self.innerdim = 50
        self.seqlen = 5
        self.numchars = 7
        self.numwords = 15
        self.indim = 20
        self.batsize = 100

    def test_seq2idx_shape(self):
        b = seq2idx(invocsize=self.numchars, outvocsize=self.numwords, innerdim=self.innerdim)
        data = np.random.randint(0, self.numchars, (self.batsize, self.seqlen))
        p = b.predict(data)
        self.assertEqual(p.shape, (self.batsize, self.numwords))

    def test_idx2seq_shape(self):
        b = idx2seq(encdim=self.encdim, invocsize=self.numwords, outvocsize=self.numchars, seqlen=self.seqlen, innerdim=self.innerdim)
        data = np.random.randint(0, self.numwords, (self.batsize, ))
        pdata = np.random.randint(0, self.numchars, (self.batsize, self.seqlen))
        p = b.predict(data, pdata)
        self.assertEqual(p.shape, (self.batsize, self.seqlen, self.numchars))

    def test_seq2idx_params(self):
        b = seq2idx(invocsize=self.numchars, outvocsize=self.numwords, innerdim=self.innerdim)
        data = np.random.randint(0, self.numchars, (self.batsize, self.seqlen))
        p = b.predict(data)
        allparams = b.output.allparams
        for param in allparams:
            print param


def words2ints(words):
    wldf = pd.DataFrame(map(word2int, words)).fillna(0)
    data = wldf.values.astype("int32")
    del wldf
    return data


def word2int(word):
    return [ord(letter)-96 for letter in word]


class TestIdx2SeqTraining(TestCase):
    def setUp(self):
        wreg=0.001
        epochs=3
        numbats=10
        lr=0.1
        statedim=70
        encdim=70
        # get words
        numchars = 27
        embdim = 50
        Glove.defaultpath = "../../data/glove/miniglove.%dd.txt"
        lm = Glove(embdim, 1000)
        words = filter(lambda x: re.match("^[a-z]+$", x), lm.D.keys())
        data = words2ints(words)
        sdata = shiftdata(data)
        wordidxs = np.arange(0, len(words))
        numwords = wordidxs.shape[0]
        print "random seq neg log prob %.3f" % math.log(numchars**data.shape[1])
        testneglogprob = 17
        print "%.2f neg log prob for a whole sequence is %.3f prob per slot" % (testneglogprob, math.exp(-testneglogprob*1./data.shape[1]))

        testpred = wordidxs[:15]
        testdata = data[:15]
        testsdata = sdata[:15]
        print testpred
        print testdata
        print testsdata
        #testpred = words2ints(testpred)
        block = idx2seq(encdim=encdim, invocsize=numwords, outvocsize=numchars, innerdim=statedim, seqlen=data.shape[1])
        print np.argmax(block.predict(testpred, testsdata), axis=2)
        self.block_before_training_frozen = block.freeze()
        block.train([wordidxs, sdata], data).seq_neg_log_prob().grad_total_norm(0.5).adagrad(lr=lr).l2(wreg)\
             .autovalidate().seq_accuracy().validinter(5)\
             .train(numbats=numbats, epochs=epochs)
        self.block_after_training_frozen = block.freeze()
        pred = block.predict(testpred, testsdata)

    def test_parameters_changed(self):
        self.assertNotEqual(self.block_after_training_frozen, self.block_before_training_frozen)
        vanillablock = idx2seq.unfreeze(self.block_before_training_frozen)
        trainedblock = idx2seq.unfreeze(self.block_after_training_frozen)
        vanillaparams = vanillablock.output.allparams
        trainedparams = trainedblock.output.allparams
        self.assertSetEqual(set(map(lambda x: x.name, vanillaparams)), set(map(lambda x: x.name, trainedparams)))
        for vanillaparam, trainedparam in zip(sorted(vanillaparams, key=lambda x: x.name), sorted(trainedparams, key=lambda x: x.name)):
            self.assertTrue(not np.allclose(vanillaparam.d.get_value(), trainedparam.d.get_value()))
            print "diff: %s %f" % (vanillaparam.name,
                                   (np.sum(np.abs(vanillaparam.d.get_value() - trainedparam.d.get_value()))
                                     /(vanillaparam.d.get_value().shape[0]*vanillaparam.d.get_value().shape[1])))