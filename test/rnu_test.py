from unittest import TestCase

from teafacto.blocks.rnu import GRU, LSTM
import numpy as np


class TestGRU(TestCase):
    def setUp(self):
        self.dim = 20
        self.innerdim = 50
        self.rnu = self.makernu()
        self.batsize = 50
        self.seqlen = 10
        self.datashape = (self.batsize, self.seqlen, self.dim)
        self.testdata = np.random.random(self.datashape)
        self.paramnames = self.getparamnames()
        self.wshape = (self.dim, self.innerdim)
        self.ushape = (self.innerdim, self.innerdim)
        self.bshape = (self.innerdim, )
        self.rnu.autobuild(self.testdata)

    def makernu(self):
        return GRU(dim=self.dim, innerdim=self.innerdim)

    def getparamnames(self):
        return ["um", "wm", "uhf", "whf", "u", "w", "bm", "bhf", "b"]

    def getwparamnames(self):
        return ["wm", "whf", "w"]

    def getuparamnames(self):
        return ["um", "uhf", "u"]

    def getbparamnames(self):
        return ["bm", "bhf", "b"]

    def test_params_owned(self):
        for param in self.paramnames:
            self.assertTrue(hasattr(self.rnu, param))

    def test_param_dims(self):
        for param in self.getwparamnames():
            self.assertEqual(getattr(self.rnu, param).shape, self.wshape)
        for param in self.getuparamnames():
            self.assertEqual(getattr(self.rnu, param).shape, self.ushape)
        for param in self.getbparamnames():
            self.assertEqual(getattr(self.rnu, param).shape, self.bshape)

    def test_params_propagated_to_outvar(self):
        outpvar = self.rnu.output
        gruparamset = set([getattr(self.rnu, paramname) for paramname in self.paramnames])
        varparamset = set(outpvar.allparams)
        self.assertSetEqual(gruparamset, varparamset)

    def test_output_shape_predict(self):
        outpv = self.rnu.predict(self.testdata)
        self.assertEqual(outpv.shape, (self.batsize, self.seqlen, self.innerdim))

    def test_input_other_batsize(self):
        othershape = (self.batsize*25, self.seqlen, self.dim) # 25 times more batches
        data = np.random.random(othershape)
        outpv = self.rnu.predict(data)
        self.assertEqual(outpv.shape, (self.batsize*25, self.seqlen, self.innerdim))

    def test_input_other_seqlen(self):
        othershape = (self.batsize, self.seqlen*25, self.dim) # 25 times longer sequences
        data = np.random.random(othershape)
        outpv = self.rnu.predict(data)
        self.assertEqual(outpv.shape, (self.batsize, self.seqlen*25, self.innerdim))

    def test_input_fail_other_dims(self):
        othershape = (self.batsize, self.seqlen, self.dim*25)
        data = np.random.random(othershape)
        self.assertRaises(Exception, self.rnu.predict, data)



class TestLSTM(TestGRU):
    def makernu(self):
        return LSTM(dim=self.dim, innerdim=self.innerdim)

    def getparamnames(self):
        return ["wf", "rf", "bf", "wi", "ri", "bi", "wo", "ro", "bo", "w", "r", "b", "pf", "pi", "po"]

    def getwparamnames(self):
        return ["wf", "wi", "wo", "w"]

    def getuparamnames(self):
        return ["rf", "ri", "ro", "r"]

    def getbparamnames(self):
        return ["bf", "bi", "bo", "b"]


class TestGRUnobias(TestGRU):
    def makernu(self):
        return GRU(dim=self.dim, innerdim=self.innerdim, nobias=True)

    def getparamnames(self):
        return ["um", "wm", "uhf", "whf", "u", "w"]

    def getbparamnames(self):
        return []

    def test_has_zero_bias_params(self):
        for bias in super(self.__class__, self).getbparamnames():
            self.assertEqual(getattr(self.rnu, bias), 0)

    def test_output_var_has_no_bias_params(self):
        outpvarparamnames = [x.name for x in self.rnu.output.allparams]
        for bias in super(self.__class__, self).getbparamnames():
            self.assertNotIn(bias, outpvarparamnames)


class TestLSTMnobias(TestLSTM, TestGRUnobias):
    def makernu(self):
        return LSTM(dim=self.dim, innerdim=self.innerdim, nobias=True)

    def getparamnames(self):
        return ["wf", "rf", "wi", "ri", "wo", "ro", "w", "r", "pf", "pi", "po"]

    def getbparamnames(self):
        return []





