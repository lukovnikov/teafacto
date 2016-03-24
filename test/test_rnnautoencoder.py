from unittest import TestCase

from teafacto.blocks.rnn import RNNAutoEncoder, RewAttRNNEncDecoder, RewAttSumDecoder
import numpy as np


def shiftdata(x):
    return np.concatenate([np.zeros_like(x[:, 0:1]), x[:, :-1]], axis=1)


class TestRNNAutoEncoder(TestCase):
    def setUp(self):
        vocsize = 25
        innerdim = 200
        encdim = 200
        batsize = 500
        seqlen = 5
        self.exppredshape = (batsize, seqlen, vocsize)
        self.rae = self.get_rae(vocsize=vocsize, innerdim=innerdim, encdim=encdim, seqlen=seqlen)
        self.dummydata = np.random.randint(0, vocsize, (batsize, seqlen))
        self.dummypred = self.rae.predict(self.dummydata, shiftdata(self.dummydata))

    def get_rae(self, **kwargs):
        return RNNAutoEncoder(**kwargs)

    def test_dummy_prediction_output_shape(self):
        self.assertEqual(self.dummypred.shape, self.exppredshape)


class AttentionRNNAutoEncoderTest(TestRNNAutoEncoder):
    def get_rae(self, **kwargs):
        kwargs["attdim"] = 33
        return RewAttRNNEncDecoder(**kwargs)

    def test_params(self):
        print self.rae.output.allparams


class RNNAttWSumDecoderTest(TestRNNAutoEncoder):
    def get_rae(self, **kwargs):
        kwargs["attdim"] = 33
        return RewAttSumDecoder(**kwargs)

    def test_params(self):
        print self.rae.output.allparams