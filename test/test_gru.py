from unittest import TestCase

from teafacto.blocks.rnu import GRU
from theano import tensor as T
import theano, numpy as np


class TestGRU(TestCase):
    def test_if_prediction_is_equivalent_to_manually_constructed_theano_graph(self):
        indim = 20
        innerdim = 50
        batsize = 200
        seqlen = 5
        data = np.random.random((batsize, seqlen, indim))
        gru = GRU(innerdim=innerdim, dim=indim)
        grupred = gru.predict(data)[:, -1, :]
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
            h = mgate * h_tm1 + (1-mgate) * canh
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



