from unittest import TestCase
import numpy as np
from teafacto.blocks.cnn import Conv1D, GlobalPool1D, CNNSeqEncoder
from teafacto.blocks.seq.rnn import EncLastDim
from teafacto.core.base import Val

class TestConv1D(TestCase):
    def test_output_shape(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        conv = Conv1D(indim=50, outdim=40, window=5)
        pred = conv.predict(xval)
        self.assertEqual(pred.shape[:2], xval.shape[:2])
        self.assertEqual(pred.shape[2], 40)

    def test_output_shape_strided_no_pad(self):
        xval = np.random.random((100, 13, 50)).astype("float32")
        conv = Conv1D(indim=50, outdim=40, window=7, stride=2, pad_mode="none")
        pred = conv.predict(xval)
        print pred.shape
        self.assertEqual(pred.shape, (100, 4, 40))

    def test_output_shape_strided_match(self):
        xval = np.random.random((100, 13, 50)).astype("float32")
        conv = Conv1D(indim=50, outdim=40, window=7, stride=2, pad_mode="match")
        pred = conv.predict(xval)
        print pred.shape
        self.assertEqual(pred.shape, (100, 7, 40))

    def test_output_shape_strided_full(self):
        xval = np.random.random((100, 13, 50)).astype("float32")
        conv = Conv1D(indim=50, outdim=40, window=7, stride=2, pad_mode="full")
        pred = conv.predict(xval)
        print pred.shape
        self.assertEqual(pred.shape, (100, 10, 40))

    def test_output_shape_masked_match_no_stride(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        maskid = np.random.randint(3, 20, (100,))
        mask = np.ones((xval.shape[:2]))
        for i in range(mask.shape[0]):
            mask[i, maskid[i]:] = 0
        conv = Conv1D(indim=50, outdim=40, window=5)
        x = Val(xval)
        x.mask = Val(mask)
        pred = conv(x)
        predmask = pred.mask
        predval = pred.eval()
        predvalmask = (predval != 0.0) * 1
        predvalexpmask = np.ones_like(predvalmask)
        for i in range(predvalexpmask.shape[0]):
            predvalexpmask[i, min(maskid[i]+2, predvalexpmask.shape[1]):, :] = 0
        self.assertTrue(np.sum(predvalexpmask - predvalmask) == 0)
        self.assertEqual(predval.shape[:2], xval.shape[:2])
        self.assertEqual(predval.shape[2], 40)

    def test_output_mask_strided_nopad(self):
        np.random.seed(1337)
        xval = np.random.random((9, 13, 50)).astype("float32")
        mask = np.asarray([
            [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0],
            [ 1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
        conv = Conv1D(indim=50, outdim=40, window=4, stride=3, pad_mode="none")
        x = Val(xval)
        x.mask = Val(mask)
        pred = conv(x)
        predmask = pred.mask
        print predmask.eval().shape
        print predmask.eval()
        print mask
        expmask = np.asarray([
                    [1, 1, 1, 1],
                    [1, 1, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
        self.assertTrue(np.allclose(expmask, predmask.eval()))


    def test_output_mask_strided_match_pad(self):
        np.random.seed(1337)
        xval = np.random.random((9, 13, 50)).astype("float32")
        mask = np.asarray([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        conv = Conv1D(indim=50, outdim=40, window=4, stride=3,
                      pad_mode="match")
        x = Val(xval)
        x.mask = Val(mask)
        pred = conv(x)
        predmask = pred.mask
        print predmask.eval().shape
        print predmask.eval()
        print mask
        expmask = np.asarray([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]])
        self.assertTrue(np.allclose(expmask, predmask.eval()))

    def test_output_mask_strided_full_pad(self):
        np.random.seed(1337)
        xval = np.random.random((9, 13, 50)).astype("float32")
        mask = np.asarray([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        conv = Conv1D(indim=50, outdim=40, window=4, stride=3,
                      pad_mode="full")
        x = Val(xval)
        x.mask = Val(mask)
        pred = conv(x)
        predmask = pred.mask
        print predmask.eval().shape
        print predmask.eval()
        print mask
        expmask = np.asarray([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]])
        self.assertTrue(np.allclose(expmask, predmask.eval()))


class TestConv1DTooManyDims(TestCase):
    def test_shapes(self):
        xval = np.random.random((100, 20, 25, 50)).astype("float32")
        conv = Conv1D(indim=50, outdim=40, window=5)
        pred = conv.predict(xval)
        self.assertEqual(pred.shape[:2], xval.shape[:2])
        self.assertEqual(pred.shape[2], 40)


class TestPool1D(TestCase):
    def test_max_pool(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        pool = GlobalPool1D(mode="max")
        pred = pool.predict(xval)
        nppred = np.max(xval, axis=1)
        self.assertTrue(np.allclose(nppred, pred))

    def test_avg_pool(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        pool = GlobalPool1D(mode="avg")
        pred = pool.predict(xval)
        nppred = np.sum(xval, axis=1) / xval.shape[1]
        self.assertTrue(np.allclose(nppred, pred))

    def test_max_pool_masked(self):
        xval = np.random.random((100, 20, 50)).astype("float32") - 0.5
        maskid = np.random.randint(1, 18, (100,))
        mask = np.ones((xval.shape[:2]))
        for i in range(mask.shape[0]):
            mask[i, maskid[i]:] = 0
        #xval[:, :, -1] = 100
        x = Val(xval)
        x.mask = Val(mask)
        pool = GlobalPool1D(mode="max")
        pred = pool(x)
        predval = pred.eval()
        xval = xval - 1e9 * np.tensordot(1-mask, np.ones((xval.shape[-1],)), 0)
        predvalexp = np.max(xval, axis=1)
        print predvalexp[:5, :5]
        print predval[:5, :5]
        self.assertTrue(np.allclose(predval, predvalexp, atol=1e-6))

    def test_avg_pool_masked(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        maskid = np.random.randint(1, 18, (100,))
        mask = np.ones((xval.shape[:2]))
        for i in range(mask.shape[0]):
            mask[i, maskid[i]:] = 0
        # xval[:, :, -1] = 100
        x = Val(xval)
        x.mask = Val(mask)
        pool = GlobalPool1D(mode="avg")
        pred = pool(x)
        predval = pred.eval()
        predvalexp = np.sum(xval * mask[:, :, np.newaxis], axis=1) / np.sum(mask, axis=1)[:, np.newaxis]
        self.assertTrue(np.allclose(predval, predvalexp))

    def test_sum_pool_masked(self):
        xval = np.random.random((100, 20, 50)).astype("float32")
        maskid = np.random.randint(1, 18, (100,))
        mask = np.ones((xval.shape[:2]))
        for i in range(mask.shape[0]):
            mask[i, maskid[i]:] = 0
        # xval[:, :, -1] = 100
        x = Val(xval)
        x.mask = Val(mask)
        pool = GlobalPool1D(mode="sum")
        pred = pool(x)
        predval = pred.eval()
        predvalexp = np.sum(xval * mask[:, :, np.newaxis], axis=1)
        self.assertTrue(np.allclose(predval, predvalexp))


class TestCNNEnc(TestCase):
    def test_enc(self):
        xval = np.random.randint(0, 200, (100, 20)).astype("int32")
        enc = CNNSeqEncoder(indim=200, inpembdim=50, innerdim=[30, 40])
        pred = enc.predict(xval)
        print pred.dtype
        self.assertEqual(pred.shape, (100, 40))

    def test_cnnenc_in_dimred(self):
        xval = np.random.randint(0, 200, (3, 100, 20)).astype("int32")
        enc = EncLastDim(CNNSeqEncoder(indim=200, inpembdim=50, innerdim=[30, 40]))
        pred = enc.predict(xval)
        print pred.dtype
        self.assertEqual(pred.shape, (3, 100, 40))

    def test_cnnenc_ret_all(self):
        xval = np.random.randint(0, 200, (100, 20)).astype("int32")
        enc = CNNSeqEncoder(indim=200, inpembdim=50, innerdim=[30, 40]).all_outputs()
        pred = enc.predict(xval)
        print pred.dtype
        self.assertEqual(pred.shape, (100, 20, 40))

    def test_cnnenc_ret_all_pos_emb(self):
        xval = np.random.randint(0, 200, (100, 20)).astype("int32")
        enc = CNNSeqEncoder(indim=200, inpembdim=50, innerdim=[30, 40],
                            posembdim=37, numpos=20).all_outputs()
        pred = enc.predict(xval)
        print pred.dtype
        self.assertEqual(pred.shape, (100, 20, 40))

    def test_cnnenc_ret_all_pos_emb_with_dropout(self):
        xval = np.random.randint(0, 200, (100, 20)).astype("int32")
        enc = CNNSeqEncoder(indim=200, inpembdim=50, innerdim=[30, 40],
                            posembdim=37, numpos=20, dropout=0.1).all_outputs()
        pred = enc.predict(xval)
        print pred.dtype
        self.assertEqual(pred.shape, (100, 20, 40))

    def test_enc_mask(self):
        xval = np.random.randint(1, 200, (100, 20)).astype("int32")
        maskid = np.random.randint(0, 5, (100,))
        for i in range(xval.shape[0]):
            xval[i, maskid[i]:] = 0
        x = Val(xval)
        enc = CNNSeqEncoder(indim=200, inpembdim=50, innerdim=5, maskid=0)
        pred = enc(x)
        #print pred.mask.eval().shape
        predval = pred.eval()
        print predval.shape

    def test_enc_mask_ret_all(self):
        xval = np.random.randint(1, 200, (100, 20)).astype("int32")
        maskid = np.random.randint(5, 10, (100,))
        for i in range(xval.shape[0]):
            xval[i, maskid[i]:] = 0
        x = Val(xval)
        enc = CNNSeqEncoder(indim=200, inpembdim=50, innerdim=5, maskid=0).all_outputs()
        pred = enc(x)
        # print pred.mask.eval().shape
        predval = pred.eval()
        print predval.shape
        print pred.mask.eval()
        self.assertTrue(np.allclose(pred.mask.eval(), xval != 0))





