from unittest import TestCase

from teafacto.blocks.memnet import RNNAutoEncoder
import numpy as np

class TestRNNAutoEncoder(TestCase):
    def setUp(self):
        vocsize = 25
        innerdim = 300
        batsize = 500
        seqlen = 5
        self.exppredshape = (batsize, seqlen, vocsize)
        self.rae = RNNAutoEncoder(vocsize=vocsize, innerdim=innerdim)
        self.dummydata = np.random.randint(0, vocsize, (batsize, seqlen))
        self.dummypred = self.rae.predict(self.dummydata)

    def test_dummy_prediction_output_shape(self):
        self.assertEqual(self.dummypred.shape, self.exppredshape)