from unittest import TestCase
import numpy as np
from teafacto.blocks.seq.marnu import ReGRU
from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.core import Val


class TestReGRU(TestCase):
    def test_mem_swap(self):
        m = ReGRU(2, 2, 3, _debug=False)
        m.att.attentiongenerator.normalizer.detpred = True
        h_tm1 = Val(np.array([[1.0, 0.0], [0.0, 1.0]])) + 0
        m_tm1 = Val(np.array([[0.5, 0.5], [0.5, 0.5]])) + 0
        M_tm1 = Val(np.array([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                              [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])) + 0
        #print M_tm1.eval()

        m_t, M_t, w = m._swap_mem(h_tm1, m_tm1, M_tm1)
        np.set_printoptions(suppress=True, precision=3)
        print w.eval()
        print m_t.eval()
        print M_t.eval()
        self.assertTrue(np.allclose(m_t.eval(), np.array([[1, 0], [0, 1]])))
        self.assertTrue(np.allclose(w.eval(), np.array([[1, 0, 0], [0, 0, 1]])))
        self.assertTrue(np.allclose(M_t.eval(),
                                    np.array([
                                        [[0.5, 0.5], [0, 1], [0, 0]],
                                        [[1, 0], [0, 0], [0.5, 0.5]]])))

    def test_init_info(self):
        batsize = 5
        m = ReGRU(7, 2, 3)
        inits = m.get_init_info(batsize)
        print inits[0].eval()
        print inits[1].eval()
        self.assertTrue(np.allclose(inits[0].eval(), np.zeros((batsize, 3, 2))))
        self.assertTrue(np.allclose(inits[1].eval(), np.zeros((batsize, 2 * 2))))

    def test_in_seqencoder(self):
        rnu = ReGRU(10, 11, 12)
        rnu2 = ReGRU(11, 13, 5)
        m = RNNSeqEncoder.fluent().vectorembedder(100, 10)\
            .setlayers(rnu, rnu2).make()
        data = np.random.randint(0, 100, (7, 8))
        pred = m.predict(data)
        print pred.shape
        self.assertEqual(pred.shape, (7, 13))

        m = RNNSeqEncoder.fluent().vectorembedder(100, 10) \
            .setlayers(rnu, rnu2).make().all_outputs()
        data = np.random.randint(0, 100, (7, 8))
        pred = m.predict(data)
        print pred.shape
        self.assertEqual(pred.shape, (7, 8, 13))

