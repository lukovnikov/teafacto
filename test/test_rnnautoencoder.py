from unittest import TestCase

from teafacto.blocks.rnn import RNNAutoEncoder, AttentionRNNAutoEncoder, RNNAttWSumDecoder
import numpy as np

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
        self.dummypred = self.rae.predict(self.dummydata)

    def get_rae(self, **kwargs):
        return RNNAutoEncoder(**kwargs)

    def test_dummy_prediction_output_shape(self):
        self.assertEqual(self.dummypred.shape, self.exppredshape)


class AttentionRNNAutoEncoderTest(TestRNNAutoEncoder):
    def get_rae(self, **kwargs):
        kwargs["attdim"] = 33
        return AttentionRNNAutoEncoder(**kwargs)

    def test_params(self):
        print self.rae.output.allparams


class RNNAttWSumDecoderTest(TestRNNAutoEncoder):
    def get_rae(self, **kwargs):
        kwargs["attdim"] = 33
        return RNNAttWSumDecoder(**kwargs)

    def test_params(self):
        print self.rae.output.allparams