import os
from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.rnu import GRU, LSTM, IFGRU, QRNU
from teafacto.core.base import param, Input

import theano
from theano import tensor as T


class TestGRU(TestCase):
    def setUp(self):
        self.dim = 20
        self.innerdim = 50
        self.rnu = self.makernu()
        self.batsize = 50
        self.seqlen = 10
        self.datashape = (self.batsize, self.seqlen, self.dim)
        self.testdata = np.random.random(self.datashape).astype("float32")
        self.paramnames = self.getparamnames()
        self.wshape = (self.dim, self.innerdim)
        self.ushape = (self.innerdim, self.innerdim)
        self.bshape = (self.innerdim, )
        inps, outps = self.rnu.autobuild(self.testdata)
        self.outp = outps[0]
        self.toremovefiles = []

    def tearDown(self):
        for p in self.toremovefiles:
            os.remove(p)

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
        outpvar = self.outp
        gruparamset = set([getattr(self.rnu, paramname) for paramname in self.paramnames])
        varparamset = set(outpvar.allparams)
        self.assertSetEqual(gruparamset, varparamset)

    def test_params_propagated_through_rnu(self):
        O = param((self.dim, self.dim), name="bigo").uniform()
        i = Input(ndim=2, dtype="int32")
        x = O[i, :]
        out = self.rnu(x)
        self.assertIn(O, out[0].allparams)

    def test_output_shape_predict(self):
        outpv = self.rnu.predict(self.testdata)
        self.assertEqual(outpv.shape, (self.batsize, self.seqlen, self.innerdim))

    def test_input_other_batsize(self):
        othershape = (self.batsize*25, self.seqlen, self.dim) # 25 times more batches
        data = np.random.random(othershape).astype("float32")
        outpv = self.rnu.predict(data)
        self.assertEqual(outpv.shape, (self.batsize*25, self.seqlen, self.innerdim))

    def test_input_other_seqlen(self):
        othershape = (self.batsize, self.seqlen*25, self.dim) # 25 times longer sequences
        data = np.random.random(othershape).astype("float32")
        outpv = self.rnu.predict(data)
        self.assertEqual(outpv.shape, (self.batsize, self.seqlen*25, self.innerdim))

    def test_input_fail_other_dims(self):
        othershape = (self.batsize, self.seqlen, self.dim*25)
        data = np.random.random(othershape)
        self.assertRaises(Exception, self.rnu.predict, data)

    def test_save_load_predict(self):
        outpv = self.rnu.predict(self.testdata)
        path = self.rnu.save()
        self.toremovefiles.append(path)
        loaded = self.rnu.__class__.load(path)
        self.assertTrue(np.allclose(outpv, loaded.predict(self.testdata)))


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
        outpvarparamnames = [x.name for x in self.outp.allparams]
        for bias in super(self.__class__, self).getbparamnames():
            self.assertNotIn(bias, outpvarparamnames)


class TestLSTMnobias(TestLSTM, TestGRUnobias):
    def makernu(self):
        return LSTM(dim=self.dim, innerdim=self.innerdim, nobias=True)

    def getparamnames(self):
        return ["wf", "rf", "wi", "ri", "wo", "ro", "w", "r", "pf", "pi", "po"]

    def getbparamnames(self):
        return []


class TestIFGRU(TestGRU):
    def makernu(self):
        return IFGRU(dim=self.dim, innerdim=self.innerdim)

    def getparamnames(self):
        return ["um", "wm", "uhf", "whf", "uif", "wif", "u", "w", "bm", "bhf", "bif", "b"]

    def getwparamnames(self):
        return ["wm", "whf", "w"]

    def getuparamnames(self):
        return ["uhf", "um", "u"]

    def getbparamnames(self):
        return ["bhf", "bm", "b"]

    def test_special_param_shapes(self):
        self.assertEqual(self.rnu.bif.shape, (self.dim,))
        self.assertEqual(self.rnu.wif.shape, (self.dim, self.dim))
        self.assertEqual(self.rnu.uif.shape, (self.innerdim, self.dim))


class TestIFGRUnobias(TestIFGRU, TestGRUnobias):
    def makernu(self):
        return IFGRU(dim=self.dim, innerdim=self.innerdim, nobias=True)

    def getparamnames(self):
        return ["um", "wm", "uhf", "whf", "uif", "wif", "u", "w"]

    def getbparamnames(self):
        return []

    def test_special_param_shapes(self):
        self.assertEqual(self.rnu.bif, 0)
        self.assertEqual(self.rnu.wif.shape, (self.dim, self.dim))
        self.assertEqual(self.rnu.uif.shape, (self.innerdim, self.dim))


class TestQRNU(TestCase):
    def setUp(self):
        self.dim = 20
        self.innerdim = 50
        self.rnu = self.makernu()
        self.batsize = 50
        self.seqlen = 10
        self.datashape = (self.batsize, self.seqlen, self.dim)
        self.testdata = np.random.random(self.datashape).astype("float32")

    def makernu(self):
        return QRNU(dim=self.dim, innerdim=self.innerdim, window_size=3)

    def test_shapes(self):
        pred = self.rnu.predict(self.testdata)
        self.assertEqual(pred.shape, (self.batsize, self.seqlen, self.innerdim))


class TestGRUBasic(TestCase):
    def test_output_shape(self):
        indim = 20
        innerdim = 50
        batsize = 200
        seqlen = 5
        data = np.random.random((batsize, seqlen, indim)).astype("float32")
        gru = GRU(innerdim=innerdim, dim=indim)
        grupred = gru.predict(data)
        self.assertEqual(grupred.shape, (batsize, seqlen, innerdim))

    def test_get_params(self):
        gru = GRU(innerdim=100, dim=20)
        params = {gru.um, gru.wm, gru.uhf, gru.whf, gru.u, gru.w, gru.bm, gru.bhf, gru.b}
        self.assertEqual(params, gru.get_params())

    def test_gru_with_mask(self):
        indim = 2
        innerdim = 5
        batsize = 4
        seqlen = 3
        data = np.random.random((batsize, seqlen, indim)).astype("float32")
        mask = np.zeros((batsize, seqlen)).astype("float32")
        mask[:, 0] = 1.
        mask[0, :] = 1.
        gru = GRU(innerdim=innerdim, dim=indim)
        grupred = gru.predict(data, mask)
        print grupred

        self.assertEqual(grupred.shape, (batsize, seqlen, innerdim))
        #self.assertTrue(np.allclose(grupred[1:, 1:, :], np.zeros_like(grupred[1:, 1:, :])))
        self.assertTrue(np.all(abs(grupred[0, ...]) > 0))
        self.assertTrue(np.all(abs(grupred[:, 0, :]) > 0))

    def test_gru_noinput(self):
        gru = GRU(innerdim=50, dim=0, noinput=True)


class TestGRU2(TestCase):

    def test_if_prediction_is_equivalent_to_manually_constructed_theano_graph(self):
        indim = 20
        innerdim = 50
        batsize = 200
        seqlen = 5
        data = np.random.random((batsize, seqlen, indim)).astype("float32")
        gru = GRU(innerdim=innerdim, dim=indim)
        grupred = gru.predict(data)[:, -1, :]
        print grupred.shape
        tgru_in, tgru_out = self.build_theano_gru(innerdim, indim, batsize, gru)
        tgrupred = tgru_out.eval({tgru_in: data.astype("float32")})
        print np.sum(np.abs(tgrupred-grupred))
        self.assertTrue(np.allclose(grupred, tgrupred))

    def build_theano_gru(self, innerdim, indim, batsize, gru):
        u = theano.shared(gru.u.d.get_value())
        w = theano.shared(gru.w.d.get_value())
        um = theano.shared(gru.um.d.get_value())
        wm = theano.shared(gru.wm.d.get_value())
        uhf = theano.shared(gru.uhf.d.get_value())
        whf = theano.shared(gru.whf.d.get_value())
        b = theano.shared(gru.b.d.get_value())
        bm = theano.shared(gru.bm.d.get_value())
        bhf = theano.shared(gru.bhf.d.get_value())

        def rec(x_t, h_tm1):
            mgate =  T.nnet.sigmoid(T.dot(h_tm1, um)  + T.dot(x_t, wm)  + bm)
            hfgate = T.nnet.sigmoid(T.dot(h_tm1, uhf) + T.dot(x_t, whf) + bhf)
            canh = T.tanh(T.dot(h_tm1 * hfgate, u) + T.dot(x_t, w) + b)
            h = (1 - mgate) * h_tm1 + mgate * canh
            return [h, h]

        def apply(x):
            inputs = x.dimshuffle(1, 0, 2) # inputs is (seq_len, batsize, dim)
            init_h = T.zeros((batsize, innerdim))
            outputs, _ = theano.scan(fn=rec,
                                sequences=inputs,
                                outputs_info=[None, init_h])
            output = outputs[0]
            return output[-1, :, :] #.dimshuffle(1, 0, 2) # return is (batsize, seqlen, dim)

        inp = T.ftensor3()
        return inp, apply(inp)







