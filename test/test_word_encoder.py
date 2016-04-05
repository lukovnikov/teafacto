from unittest import TestCase
from teafacto.blocks.lang.wordembed import WordEncoder, WordEncoderPlusGlove, WordEmbedPlusGlove
from teafacto.blocks.lang.wordvec import Glove
import numpy as np


class TestWordEncoder(TestCase):
    def test_word_encoder_output_shape(self):
        batsize = 111
        seqlen = 13
        wordlen = 37
        numchars = 200
        data = np.random.randint(0, numchars, (batsize, wordlen))
        encdim = 100
        block = WordEncoder(indim=numchars, outdim=encdim)
        pred = block.predict(data)
        self.assertEqual(pred.shape, (batsize, encdim))


class TestWordEncoderPlusGlove(TestCase):
    def test_word_encoder_output_shape(self):
        Glove.defaultpath = "../../../data/glove/miniglove.%dd.txt"
        batsize = 111
        seqlen = 13
        wordlen = 37
        numchars = 200
        numwords = 1000
        worddata = np.random.randint(0, numwords, (batsize, 1))
        chardata = np.random.randint(0, numchars, (batsize, wordlen))
        data = np.concatenate([worddata, chardata], axis=1)
        encdim = 100
        embdim = 50
        block = WordEncoderPlusGlove(numchars=numchars, numwords=numwords, encdim=encdim, embdim=embdim)
        pred = block.predict(data)
        self.assertEqual(pred.shape, (batsize, encdim+embdim))


class TestWordEmbedPlusGlove(TestCase):
    def test_word_embed_plus_glove_output_shape(self):
        Glove.defaultpath = "../../../data/glove/miniglove.%dd.txt"
        batsize = 111
        seqlen = 13
        wordlen = 37
        numchars = 200
        numwords = 1000
        data = np.random.randint(0, numwords, (batsize,))
        encdim = 100
        embdim = 50
        block = WordEmbedPlusGlove(indim=numwords, outdim=encdim, embdim=embdim)
        pred = block.predict(data)
        self.assertEqual(pred.shape, (batsize, encdim+embdim))

