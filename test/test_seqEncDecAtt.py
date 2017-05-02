from unittest import TestCase
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.blocks.seq.rnu import LSTM, GRU
from teafacto.blocks.match import LinearDistance
from teafacto.core.base import Val
import numpy as np


class TestSimpleSeqEncDecAtt(TestCase):
    def test_unidir_shapes(self):
        self.do_test_shapes(bidir=False)

    def test_bidir_shapes(self):
        self.do_test_shapes(bidir=True)

    def test_bidir_with_lstm(self):
        self.do_test_shapes(bidir=True, rnu=LSTM)

    def do_test_shapes(self, bidir=False, rnu=GRU):
        inpvocsize = 100
        outvocsize = 13
        inpembdim = 10
        outembdim = 7
        encdim = [26, 14]
        decdim = [21, 15]
        batsize = 11
        inpseqlen = 6
        outseqlen = 5

        if bidir:
            encdim = [e / 2 for e in encdim]

        m = SimpleSeqEncDecAtt(inpvocsize=inpvocsize,
                               inpembdim=inpembdim,
                               outvocsize=outvocsize,
                               outembdim=outembdim,
                               encdim=encdim,
                               decdim=decdim,
                               bidir=bidir,
                               statetrans=True,
                               attdist=LinearDistance(15, 14, 17),
                               rnu=rnu)

        inpseq = np.random.randint(0, inpvocsize, (batsize, inpseqlen)).astype("int32")
        outseq = np.random.randint(0, outvocsize, (batsize, outseqlen)).astype("int32")

        predenco, enco, states = m.enc.predict(inpseq)
        self.assertEqual(predenco.shape, (batsize, encdim[-1] if not bidir else encdim[-1] * 2))
        if rnu == GRU:
            self.assertEqual(len(states), 2)
            for state, encdime in zip(states, encdim):
                self.assertEqual(state.shape, (batsize, inpseqlen, encdime if not bidir else encdime * 2))
        elif rnu == LSTM:
            self.assertEqual(len(states), 4)
            for state, encdime in zip(states, [encdim[0], encdim[0], encdim[1], encdim[1]]):
                self.assertEqual(state.shape, (batsize, inpseqlen, encdime if not bidir else encdime * 2))

        pred = m.predict(inpseq, outseq)
        self.assertEqual(pred.shape, (batsize, outseqlen, outvocsize))

        _, outvar = m.autobuild(inpseq, outseq)
        for p in sorted(outvar[0].allparams, key=lambda x: str(x)):
            print p


    def test_mask(self, rnu=GRU, bidir=False):
        inpvocsize = 100
        outvocsize = 13
        inpembdim = 10
        outembdim = 7
        encdim = [26, 14]
        decdim = [21, 15]
        batsize = 11
        inpseqlen = 6
        outseqlen = 7

        maskid = 0

        m = SimpleSeqEncDecAtt(inpvocsize=inpvocsize,
                               inpembdim=inpembdim,
                               outvocsize=outvocsize,
                               outembdim=outembdim,
                               encdim=encdim,
                               decdim=decdim,
                               bidir=bidir,
                               maskid=maskid,
                               statetrans=True,
                               attdist=LinearDistance(15, 14, 17),
                               rnu=rnu)

        inpseq = np.random.randint(1, inpvocsize, (batsize, inpseqlen)).astype("int32")
        outseq = np.random.randint(1, outvocsize, (batsize, outseqlen)).astype("int32")

        inpseq[:, -2:] = 0
        outseq[:, -3:] = 0

        from teafacto.core.base import asblock

        maskpred = asblock(lambda x, y: m(x, y).mask)

        maskprediction = maskpred.predict(inpseq, outseq)
        self.assertTrue(np.allclose(maskprediction, outseq != maskid))

        encmaskpred = asblock(lambda x: m.enc(x)[1].mask).predict(inpseq)
        self.assertTrue(np.allclose(encmaskpred, inpseq != maskid))

        decpred, extra = m.predict(inpseq, outseq, _extra_outs=["attention_weights"])
        weights = extra["attention_weights"]
        maskedweights = weights[:, :, -2:]
        self.assertTrue(np.allclose(maskedweights, np.zeros_like(maskedweights)))
        np.set_printoptions(precision=3)


class TestSimple2SeqEncDecAtt(TestCase):
    def test_vector_out(self):
        encdec = SimpleSeqEncDecAtt(inpvocsize=19, outvocsize=17, outconcat=False, encdim=110, decdim=110)
        encdata = np.random.randint(0, 19, (2, 5))
        decdata = np.random.randint(0, 17, (2, 5))
        pred = encdec.predict(encdata, decdata)
        self.assertEqual(pred.shape, (2, 5, 17))

        encdec = SimpleSeqEncDecAtt(inpvocsize=19, outvocsize=17, vecout=False, outconcat=False, encdim=110, decdim=110)
        pred = encdec.predict(encdata, decdata)
        print pred.shape
        self.assertEqual(pred.shape, (2, 5, 110))

    def test_get_params(self):
        attdist = LinearDistance(110, 110, 100, nobias=False)
        encdec = SimpleSeqEncDecAtt(inpvocsize=19, outvocsize=17, outconcat=False, encdim=(110, 100), decdim=100, attdist=attdist)
        enclayers = encdec.enc.block.layers
        params = set()
        for layer in enclayers:
            for paramname in "w wm whf u um uhf b bm bhf".split():      # GRU params
                params.add(getattr(layer, paramname))
        declayers = encdec.dec.block.layers
        for layer in declayers:
            for paramname in "w wm whf u um uhf b bm bhf".split():      # GRU params
                params.add(getattr(layer, paramname))
        declin = encdec.dec.softmaxoutblock.l
        params.update({declin.W, declin.b})

        params.update({encdec.dec.attention.attentiongenerator.dist.leftblock.W,
                       encdec.dec.attention.attentiongenerator.dist.leftblock.b,
                       encdec.dec.attention.attentiongenerator.dist.rightblock.W,
                       encdec.dec.attention.attentiongenerator.dist.rightblock.b,
                       encdec.dec.attention.attentiongenerator.dist.agg})
        self.assertEqual(params, encdec.get_params())

    def test_set_lr(self):
        attdist = LinearDistance(110, 110, 100)
        encdec = SimpleSeqEncDecAtt(inpvocsize=19, outvocsize=17, outconcat=False, encdim=110, decdim=110, attdist=attdist)
        encdec.dec.set_lr(0.1)
        encdec.dec.attention.set_lr(0.5)    # TODO
        encdata = np.random.randint(0, 19, (2, 5))
        decdata = np.random.randint(0, 17, (2, 5))
        o = encdec(Val(encdata), Val(decdata))
        #print "\n".join(["{}: {}".format(x, x.lrmul) for x in o.allparams])
        #print "\n".join(["{}: {}".format(x, x.lrmul) for x in o.allparams])
        encparams = encdec.enc.get_params()
        decparams = encdec.dec.get_params()
        attparams = encdec.dec.attention.get_params()
        print "\n".join(["{}: {}".format(x, x.lrmul) for x in encparams]) + "\n"
        print "\n".join(["{}: {}".format(x, x.lrmul) for x in decparams]) + "\n"
        for x in encparams:
            self.assertEqual(x.lrmul, 1.0)
        for x in decparams:
            if x not in attparams:
                self.assertEqual(x.lrmul, 0.1)
            else:
                self.assertEqual(x.lrmul, 0.5)

    def test_two_phase_training(self):
        encdec = SimpleSeqEncDecAtt(inpvocsize=19, inpembdim=50, outvocsize=17, outembdim=40, outconcat=False, encdim=110, decdim=110, statetrans=True)
        originaldecparams = encdec.dec.get_params()
        originalencparams = encdec.enc.get_params()
        originaldecparamvals = dict(zip(originaldecparams, [x.v for x in originaldecparams]))
        for x in originaldecparams:
            self.assertEqual(x.lrmul, 1.0)
        for x in originalencparams:
            self.assertEqual(x.lrmul, 1.0)

        inpseq = np.random.randint(0, 19, (10, 20))
        outseq = np.random.randint(0, 17, (10, 15))

        encdec.train([inpseq, outseq[:, :-1]], outseq[:, 1:])\
            .cross_entropy().rmsprop(lr=0.001).train(1, 5)

        traineddecparamvals = dict(zip(originaldecparams, [x.v for x in originaldecparams]))
        for k in originaldecparamvals:
            self.assertTrue(not np.allclose(originaldecparamvals[k], traineddecparamvals[k]))
            print "{} {}".format(k, np.linalg.norm(originaldecparamvals[k] - traineddecparamvals[k]))

        encdec.dec.set_lr(0.0)
        encdec.remake_encoder(inpvocsize=21, inpembdim=60, innerdim=110)
        for x in originaldecparams:
            self.assertEqual(x.lrmul, 0.0)
        newencparams = encdec.enc.get_params()
        self.assertEqual(newencparams.difference(originalencparams), newencparams)
        originalnewencparamvals = dict(zip(newencparams, [x.v for x in newencparams]))

        inpseq = np.random.randint(0, 21, (10, 16))
        outseq = np.random.randint(0, 17, (10, 14))
        encdec.train([inpseq, outseq[:, :-1]], outseq[:, 1:]).cross_entropy().rmsprop(lr=0.001).train(1, 5)

        trainednewencparamvals = dict(zip(newencparams, [x.v for x in newencparams]))
        newdecparamvals = dict(zip(originaldecparams, [x.v for x in originaldecparams]))
        print "\n"
        for k in originaldecparams:
            self.assertTrue(np.allclose(traineddecparamvals[k], newdecparamvals[k]))
            print "{} {}".format(k, np.linalg.norm(newdecparamvals[k] - traineddecparamvals[k]))
        print "\n"
        for k in newencparams:
            self.assertTrue(not np.allclose(trainednewencparamvals[k], originalnewencparamvals[k]))
            print "{} {}".format(k, np.linalg.norm(trainednewencparamvals[k] - originalnewencparamvals[k]))

        #print "\n".join(["{} {}".format(x, x.lrmul) for x in encdec.get_params()])
