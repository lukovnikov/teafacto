from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.attention import Attention, AttGen, WeightedSumAttCon
from teafacto.blocks.seq.rnn import SeqDecoder
from teafacto.blocks.seq.rnu import GRU
from teafacto.blocks.basic import IdxToOneHot
from teafacto.blocks.match import LinearDistance, BilinearDistance, DotDistance, \
    CosineDistance, LinearGateDistance


class TestBilinearAttGen(TestCase):
    def test_shapes(self):
        batsize, seqlen, datadim, critdim = 100, 7, 49, 47
        crit = np.random.random((batsize, critdim))
        data = np.random.random((batsize, seqlen, datadim))
        m = AttGen(BilinearDistance(critdim, datadim))
        pred = m.predict(crit, data)
        self.assertEqual(pred.shape, (batsize, seqlen))
        self.assertTrue(np.allclose(np.sum(pred, axis=1), np.ones((pred.shape[0],))))


class TestDotprodAttgen(TestCase):
    def test_with_mask(self):
        batsize, seqlen, dim = 6, 5, 10
        crit = np.random.random((batsize, dim))
        data = np.random.random((batsize, seqlen, dim))
        maskstarts = np.random.randint(1, 5, (batsize,))
        mask = np.ones((batsize, seqlen), dtype="int32")
        for i in range(batsize):
            mask[i, maskstarts[i]:] = 0
        m = AttGen(DotDistance())
        pred = m.predict(crit, data, mask=mask)
        self.assertTrue(np.allclose(mask, pred > 0))
        self.assertTrue(np.allclose(np.sum(pred, axis=1), np.ones((pred.shape[0],))))


class TestForwardAttGen(TestCase):
    def test_shapes(self):
        batsize, seqlen, datadim, critdim, attdim = 5, 3, 4, 3, 7
        crit = np.random.random((batsize, critdim))
        data = np.random.random((batsize, seqlen, datadim))
        m = AttGen(LinearDistance(critdim, datadim, attdim))
        pred = m.predict(crit, data)
        self.assertEqual(pred.shape, (batsize, seqlen))
        self.assertTrue(np.allclose(np.sum(pred, axis=1), np.ones((pred.shape[0],))))


class AttentionGenTest(TestCase):
    def test_shapes(self):
        batsize, seqlen = 100, 7
        criterionshape = (batsize, 10)
        datashape = (batsize, seqlen, 10)
        attgen = AttGen(CosineDistance())
        # generate data
        criterion = np.random.random(criterionshape)
        data = np.random.random(datashape)
        # predict and test
        pred = attgen.predict(criterion, data)
        self.assertEqual(pred.shape, (batsize, seqlen))
        self.assertTrue(np.allclose(pred.sum(axis=1), np.ones((pred.shape[0],))))

    def test_mask(self):
        batsize, seqlen = 100, 7
        criterionshape = (batsize, 10)
        datashape = (batsize, seqlen, 10)
        attgen = AttGen(CosineDistance())
        # generate data
        criterion = np.random.random(criterionshape)
        data = np.random.random(datashape)
        mask = np.ones((batsize, seqlen))
        maskids = np.random.randint(2, seqlen+1, (batsize,))
        for i in range(maskids.shape[0]):
            mask[i, maskids[i]:] = 0
        # predict and test
        pred = attgen.predict(criterion, data, mask)
        maskthrough = np.not_equal(pred, 0)
        self.assertTrue(np.all(maskthrough == mask))


class DummyAttentionGeneratorConsumerTest(TestCase):
    def setUp(self):
        criteriondim = 20
        datadim = 20
        innerdim = 30
        batsize = 33
        seqlen = 11
        self.attgenshape = (batsize, seqlen)
        self.attconshape = (batsize, datadim)
        self.attgen = self.getattgenc(critdim=criteriondim, datadim=datadim, attdim=innerdim)
        self.attgenparams = self.getattgenparams()
        self.attcon = WeightedSumAttCon()
        self.att = Attention(self.attgen, self.attcon)
        self.criterion_val = np.random.random((batsize, criteriondim)).astype("float32")
        self.data_val = np.random.random((batsize, seqlen, datadim)).astype("float32")

    def getattgenc(self, critdim=None, datadim=None, attdim=None):
        return AttGen(DotDistance())

    def getattgenparams(self):
        return set()

    def test_generator_shape(self):
        pred = self.attgen.predict(self.criterion_val, self.data_val)
        self.assertEqual(pred.shape, self.attgenshape)

    def test_generator_param_prop(self):
        _, outps = self.attgen.autobuild(self.criterion_val, self.data_val)
        allparams = outps[0].allparams
        self.assertSetEqual(allparams, self.attgenparams)

    def test_consumer_shape(self):
        pred = self.att.predict(self.criterion_val, self.data_val)
        self.assertEqual(pred.shape, self.attconshape)

    def test_consumer_param_prop(self):
        _, outps = self.att.autobuild(self.criterion_val, self.data_val)
        allparams = outps[0].allparams
        self.assertSetEqual(allparams, self.attgenparams)


class LinearGateAttentionGenTest(DummyAttentionGeneratorConsumerTest):
    def getattgenc(self, critdim=None, datadim=None, attdim=None):
        return AttGen(LinearGateDistance(critdim, datadim, attdim, nobias=False))

    def getattgenparams(self):
        return {self.attgen.dist.leftblock.W, self.attgen.dist.leftblock.b,
                self.attgen.dist.rightblock.W, self.attgen.dist.rightblock.b,
                self.attgen.dist.agg}


class TestAttentionRNNDecoder(TestCase):
    def setUp(self):
        vocsize = 10
        innerdim = 50
        encdim = 30
        seqlen = 5
        batsize = 77
        self.att = Attention(AttGen(BilinearDistance(innerdim, encdim)),
                             WeightedSumAttCon())
        self.decwatt = SeqDecoder(
            [IdxToOneHot(vocsize), GRU(dim=vocsize+encdim, innerdim=innerdim)],
            inconcat=True,
            attention=self.att,
            innerdim=innerdim
        )
        self.decwoatt = SeqDecoder(
            [IdxToOneHot(vocsize), GRU(dim=vocsize+encdim, innerdim=innerdim)],
            inconcat=True,
            innerdim=innerdim
        )
        self.attdata = np.random.random((batsize, seqlen, encdim)).astype("float32")
        self.data = np.random.random((batsize, encdim)).astype("float32")
        self.seqdata = np.random.randint(0, vocsize, (batsize, seqlen))
        self.predshape = (batsize, seqlen, vocsize)

    def test_shape(self):
        pred = self.decwatt.predict(self.attdata, self.seqdata)
        self.assertEqual(pred.shape, self.predshape)

    def test_shape_wo_att(self):
        pred = self.decwoatt.predict(self.data, self.seqdata)
        self.assertEqual(pred.shape, self.predshape)

    def test_attentiongenerator_param_in_allparams(self):
        inps, outps = self.decwatt.autobuild(self.attdata, self.seqdata)
        allparams = outps[0].allparams
        self.assertIn(self.att.attentiongenerator.dist.W, allparams)

    def test_attentiongenerator_param_not_in_params_of_dec_wo_att(self):
        _, outps = self.decwoatt.autobuild(self.data, self.seqdata)
        allparams = outps[0].allparams
        self.assertNotIn(self.att.attentiongenerator.dist.W, allparams)
