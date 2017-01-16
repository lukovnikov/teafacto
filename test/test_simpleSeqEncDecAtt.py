from unittest import TestCase

import numpy as np

from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.blocks.match import LinearDistance
from teafacto.core.base import Val


class TestSimpleSeqEncDecAtt(TestCase):
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
        attdist = LinearDistance(110, 110, 100)
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
        params.update({encdec.dec.lin.W, encdec.dec.lin.b})

        params.update({encdec.dec.attention.attentiongenerator.dist.lin.W, encdec.dec.attention.attentiongenerator.dist.lin.b, encdec.dec.attention.attentiongenerator.dist.lin2.W, encdec.dec.attention.attentiongenerator.dist.lin2.b, encdec.dec.attention.attentiongenerator.dist.agg})
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
        encdec.train([inpseq, outseq[:, :-1]], outseq[:, 1:]).cross_entropy().rmsprop(lr=0.001).train(1, 5)

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