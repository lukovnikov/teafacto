from unittest import TestCase
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
from teafacto.blocks.match import LinearDistance
import numpy as np


class TestSimpleSeqEncDecAtt(TestCase):
    def test_unidir_shapes(self):
        self.do_test_shapes(False)

    def test_bidir_shapes(self):
        self.do_test_shapes(True)

    def test_sepatt_shapes(self):
        self.do_test_shapes(False, True)

    def do_test_shapes(self, bidir=False, sepatt=False):
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
                               sepatt=sepatt)

        inpseq = np.random.randint(0, inpvocsize, (batsize, inpseqlen)).astype("int32")
        outseq = np.random.randint(0, outvocsize, (batsize, outseqlen)).astype("int32")

        predenco, enco = m.enc.predict(inpseq)
        self.assertEqual(predenco.shape, (batsize, encdim[-1] if not bidir else encdim[-1] * 2))

        if sepatt:
            self.assertEqual(enco.shape, (batsize, inpseqlen, 2, encdim[-1] if not bidir else encdim[-1] * 2))

        pred = m.predict(inpseq, outseq)
        self.assertEqual(pred.shape, (batsize, outseqlen, outvocsize))

        _, outvar = m.autobuild(inpseq, outseq)
        for p in sorted(outvar[0].allparams, key=lambda x: str(x)):
            print p