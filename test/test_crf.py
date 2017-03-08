from unittest import TestCase
from teafacto.blocks.crf import log_sum_exp, forward
from teafacto.core.base import Val
import numpy as np


class TestCRFUtils(TestCase):
    def test_log_sum_exp(self):
        x = Val(np.random.random((10, 5, 6)))
        y = log_sum_exp(x, axis=-1)
        yv = y.eval()
        print yv.shape

    def test_forward_fixed_data(self):
        sma = -1e3
        obs = Val(np.array([
            [
                [sma, 1.0, sma],
                [sma, sma, 1.0],
                [1.0, sma, sma]
            ]
        ], dtype="float32"))
        trans = Val(np.array([
            [sma, 1.0, sma],
            [sma, sma, 1.0],
            [1.0, sma, sma]
        ], dtype="float32"))
        out = forward(obs, trans)
        outv = out.eval()
        print outv

        transval = trans.d.eval()
        obsval = obs.d.eval()[0]

        def sxy(y):
            acc = 0
            for i in range(1, len(y)):
                acc += transval[y[i - 1], y[i]]
                acc += obsval[i - 1, y[i]]
            return acc

        bigacc = []
        # enumerate all possible seqs of 4 with 3 vals
        for i in range(3 ** 3):
            k = i
            y = [0]
            for j in range(3):
                y.append(k % 3)
                k = k // 3
            bigacc.append(sxy(y))

        sumexp = np.log(np.sum(np.exp(bigacc)))
        print sumexp

        self.assertTrue(np.allclose([sumexp], outv))
        self.assertTrue(np.allclose(outv, [6.]))

    def test_forward_fixed_data_masked(self):
        sma = -1e3
        obs = Val(np.array([
            [
                [sma, 1.0, sma],
                [sma, sma, 1.0],
                [1.0, sma, sma],
                [sma, 1.0, sma],
                [sma, sma, 1.0],
            ]
        ], dtype="float32"))
        mask = Val(np.array([
            [1, 1, 1, 0, 0]
        ], dtype="float32"))
        obs.mask = mask
        trans = Val(np.array([
            [sma, 1.0, sma],
            [sma, sma, 1.0],
            [1.0, sma, sma]
        ], dtype="float32"))
        out = forward(obs, trans)
        outv = out.eval()
        print outv

        transval = trans.d.eval()
        obsval = obs.d.eval()[0]

        def sxy(y):
            acc = 0
            for i in range(1, len(y)):
                acc += transval[y[i - 1], y[i]]
                acc += obsval[i - 1, y[i]]
            return acc

        bigacc = []
        # enumerate all possible seqs of 4 with 3 vals
        for i in range(3 ** 3):
            k = i
            y = [0]
            for j in range(3):
                y.append(k % 3)
                k = k // 3
            bigacc.append(sxy(y))

        sumexp = np.log(np.sum(np.exp(bigacc)))
        print sumexp

        self.assertTrue(np.allclose([sumexp], outv))
        self.assertTrue(np.allclose(outv, [6.]))

    def test_forward_random_data(self):
        sma = -1e3
        seqlen = 4
        obs = Val(np.random.random((10, seqlen, 12)).astype("float32"))
        trans = Val(np.random.random((12, 12)).astype("float32"))
        out = forward(obs, trans)
        outv = out.eval()
        print outv

        transval = trans.d.eval()
        obsval = obs.d.eval()[0]

        def sxy(y):
            acc = 0
            for i in range(1, len(y)):
                acc += transval[y[i - 1], y[i]]
                acc += obsval[i - 1, y[i]]
            return acc

        bigacc = []
        # enumerate all possible seqs of 4 with 3 vals
        for i in xrange(12 ** seqlen):
            k = i
            y = [0]
            for j in range(seqlen):
                y.append(k % 12)
                k = k // 12
            bigacc.append(sxy(y))

        sumexp = np.log(np.sum(np.exp(bigacc)))
        print sumexp

        self.assertEqual(outv.shape, (10,))
        self.assertTrue(np.allclose(sumexp, outv[0]))

    def test_viterbi_fixed_data(self):
        sma = -1e3
        obs = Val(np.array([
            [
                [sma, 1.0, sma],
                [sma, sma, 1.0],
                [1.0, 0.5, sma]
            ]
        ], dtype="float32"))
        trans = Val(np.array([
            [sma, 1.0, sma],
            [sma, sma, 1.0],
            [1.0, 0.5, sma]
        ], dtype="float32"))
        out = forward(obs, trans, viterbi=True)
        outv = out.eval()
        print outv

        transval = trans.d.eval()
        obsval = obs.d.eval()[0]

        def sxy(y):
            acc = 0
            for i in range(1, len(y)):
                acc += transval[y[i - 1], y[i]]
                acc += obsval[i - 1, y[i]]
            return acc

        bigacc = []
        # enumerate all possible seqs of 4 with 3 vals
        for i in range(3 ** 3):
            k = i
            y = [0]
            for j in range(3):
                y.append(k % 3)
                k = k // 3
            bigacc.append(sxy(y))

        sumexp = np.log(np.sum(np.exp(bigacc)))
        sumexp = np.max(bigacc)
        print sumexp

        self.assertTrue(np.allclose([sumexp], outv))
        self.assertTrue(np.allclose(outv, [6.]))

    def test_viterbi_fixed_data_masked(self):
        sma = -1e3
        obs = Val(np.array([
            [
                [sma, 1.0, sma],
                [sma, sma, 1.0],
                [1.0, 0.5, sma],
                [sma, 1.0, sma],
                [sma, sma, 1.0],
            ]
        ], dtype="float32"))
        mask = Val(np.array([
            [1, 1, 1, 0, 0]
        ], dtype="float32"))
        obs.mask = mask
        trans = Val(np.array([
            [sma, 1.0, sma],
            [sma, sma, 1.0],
            [1.0, 0.5, sma]
        ], dtype="float32"))
        out = forward(obs, trans, viterbi=True)
        outv = out.eval()
        print outv

        transval = trans.d.eval()
        obsval = obs.d.eval()[0]

        def sxy(y):
            acc = 0
            for i in range(1, len(y)):
                acc += transval[y[i - 1], y[i]]
                acc += obsval[i - 1, y[i]]
            return acc

        bigacc = []
        # enumerate all possible seqs of 4 with 3 vals
        for i in range(3 ** 3):
            k = i
            y = [0]
            for j in range(3):
                y.append(k % 3)
                k = k // 3
            bigacc.append(sxy(y))

        sumexp = np.log(np.sum(np.exp(bigacc)))
        sumexp = np.max(bigacc)
        print sumexp

        self.assertTrue(np.allclose([sumexp], outv))
        self.assertTrue(np.allclose(outv, [6.]))

    def test_viterbi_fixed_data_best_seq(self):
        sma = -1e3
        obs = Val(np.array([
            [
                [sma, 1.0, sma],
                [sma, sma, 1.0],
                [1.0, sma, sma],
                [sma, 1.0, sma],
            ]
        ], dtype="float32"))
        trans = Val(np.array([
            [sma, 1.0, sma],
            [sma, sma, 1.0],
            [1.0, sma, sma]
        ], dtype="float32"))
        out = forward(obs, trans, viterbi=True, return_best_sequence=True)
        outv = out.eval()
        print outv

        transval = trans.d.eval()
        obsval = obs.d.eval()[0]

        def sxy(y):
            acc = 0
            for i in range(1, len(y)):
                acc += transval[y[i - 1], y[i]]
                acc += obsval[i - 1, y[i]]
            return acc

        bestscore = -1e6
        bestseq = []
        # enumerate all possible seqs of 4 with 3 vals
        for i in range(3 ** 4):
            k = i
            y = [0]
            for j in range(4):
                y.append(k % 3)
                k = k // 3
            score = sxy(y)
            if score > bestscore:
                bestscore = score
                bestseq = y

        print bestseq
        self.assertEqual(bestseq, list(outv[0]))

    def test_viterbi_fixed_data_best_seq_masked(self):
        sma = -1e3
        obs = Val(np.array([
            [
                [sma, 1.0, sma],
                [sma, sma, 1.0],
                [1.0, sma, sma],
                [sma, 1.0, sma],
                [sma, sma, 1.0],
                [1.0, sma, sma],
            ]
        ], dtype="float32"))
        mask = Val(np.array([
            [1, 1, 1, 1, 0, 0]
        ], dtype="int8"))
        obs.mask = mask
        trans = Val(np.array([
            [sma, 1.0, sma],
            [sma, sma, 1.0],
            [1.0, sma, sma]
        ], dtype="float32"))
        out = forward(obs, trans, viterbi=True, return_best_sequence=True)
        outv = out.eval()
        mask = out.mask.d.eval()
        print outv

        transval = trans.d.eval()
        obsval = obs.d.eval()[0]

        def sxy(y):
            acc = 0
            for i in range(1, len(y)):
                acc += transval[y[i - 1], y[i]]
                acc += obsval[i - 1, y[i]]
            return acc

        bestscore = -1e6
        bestseq = []
        # enumerate all possible seqs of 4 with 3 vals
        for i in range(3 ** 6):
            k = i
            y = [0]
            for j in range(6):
                y.append(k % 3)
                k = k // 3
            score = sxy(y)
            if score > bestscore:
                bestscore = score
                bestseq = y

        bestseq = np.array(bestseq)
        print bestseq
        outv = outv[0]
        mask = mask[0]
        print mask, outv, bestseq
        bestseq[mask == 0] = -1
        outv[mask == 0] = -1
        print bestseq, outv, mask
        self.assertTrue(np.allclose(bestseq, outv))


