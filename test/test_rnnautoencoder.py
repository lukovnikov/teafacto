from unittest import TestCase

from teafacto.blocks.rnn import RNNAutoEncoder
import numpy as np

class TestRNNAutoEncoder(TestCase):
    def setUp(self):
        vocsize = 25
        innerdim = 200
        encdim = 200
        batsize = 500
        seqlen = 5
        self.uparamshape = (innerdim, innerdim)
        self.wparamshape = (vocsize, innerdim)
        self.bparamshape = (innerdim,)
        self.oparamshape = (innerdim, vocsize)
        self.exppredshape = (batsize, seqlen, vocsize)
        self.rae = RNNAutoEncoder(vocsize=vocsize, innerdim=innerdim, encdim=encdim)
        self.dummydata = np.random.randint(0, vocsize, (batsize, seqlen))
        self.dummypred = self.rae.predict(self.dummydata)

    def test_dummy_prediction_output_shape(self):
        self.assertEqual(self.dummypred.shape, self.exppredshape)