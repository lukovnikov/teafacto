from unittest import TestCase

from teafacto.blocks.rnn import RNNAutoEncoder
import numpy as np

class TestRNNAutoEncoder(TestCase):
    def setUp(self):
        vocsize = 25
        innerdim = 300
        batsize = 500
        seqlen = 5
        self.uparamshape = (innerdim, innerdim)
        self.wparamshape = (vocsize, innerdim)
        self.bparamshape = (innerdim,)
        self.oparamshape = (innerdim, vocsize)
        self.exppredshape = (batsize, seqlen, vocsize)
        self.rae = RNNAutoEncoder(vocsize=vocsize, innerdim=innerdim)
        self.dummydata = np.random.randint(0, vocsize, (batsize, seqlen))
        self.dummypred = self.rae.predict(self.dummydata)

    def test_dummy_prediction_output_shape(self):
        self.assertEqual(self.dummypred.shape, self.exppredshape)

    def test_parameter_propagation(self):
        params = self.rae.output.allparams
        paramcounts = {}
        for param in params:
            if param.name[0] == "b":
                self.assertEqual(param.shape, self.bparamshape)
            elif param.name[0] == "w":
                self.assertEqual(param.shape, self.wparamshape)
            elif param.name[0] == "u":
                self.assertEqual(param.shape, self.uparamshape)
            elif param.name[0] == "a":
                self.assertEqual(param.shape, self.oparamshape)
            if param.name not in paramcounts:
                paramcounts[param.name] = 0
            paramcounts[param.name] += 1
        for k, v in paramcounts.items():
            if k[:4] == "auto":
                self.assertEqual(v, 1)
            else:
                self.assertEqual(v, 2)