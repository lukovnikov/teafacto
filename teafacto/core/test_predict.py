from unittest import TestCase

from teafacto.examples.dummy import Dummy
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.plot.attention import AttentionPlotter
import numpy as np


class TestBlockPredictor(TestCase):
    def test_dummy_pred(self):
        m = Dummy(20, 5)
        data = np.random.randint(0, 20, (50,))
        pred = m.predict(data)
        print pred

    def test_dummy_pred_extra_outputs(self):
        m = Dummy(20, 5)
        data = np.random.randint(0, 20, (50,))
        pred = m.predict.return_extra_outs(["out"])(data)
        self.assertEqual(len(pred), 2)
        self.assertEqual(pred[0].shape, pred[1]["out"].shape)
        self.assertEqual(pred[0].shape, (50, 20))
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            ret = e_x / (np.sum(e_x, axis=-1, keepdims=True))
            return ret
        self.assertTrue(np.allclose(softmax(pred[1]["out"]), pred[0]))

    def test_encdec_attention_output_extra(self):
        m = SimpleSeqEncDecAtt()
        xdata = np.random.randint(0, 400, (50, 13))
        ydata = np.random.randint(0, 100, (50, 7))
        pred, extra = m.predict.return_extra_outs(["attention_weights", "i_t"])(xdata, ydata)
        self.assertEqual(pred.shape, (50, 7, 100))
        self.assertEqual(extra["attention_weights"].shape, (7, 50, 13))

        #AttentionPlotter.plot(extra["attention_weights"][:, 0, :].T)
        attw = extra["attention_weights"][:, 0, :].T
        print np.sum(attw, axis=0)
        print np.sum(attw, axis=1)
        self.assertTrue(np.allclose(np.ones_like(np.sum(attw, axis=0)), np.sum(attw, axis=0)))


