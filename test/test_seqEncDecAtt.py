from unittest import TestCase
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
import numpy as np


class TestSimpleSeqEncDecAtt(TestCase):
    def test_shapes(self):
        inpvocsize = 1000
        outvocsize = 103
        inpembdim = 100
        outembdim = 50
        encdim = 97
        decdim = 100
        attdim = 80
        bidir = False
        batsize = 111
        inpseqlen = 7
        outseqlen = 5

        m = SimpleSeqEncDecAtt(inpvocsize=inpvocsize,
                               inpembdim=inpembdim,
                               outvocsize=outvocsize,
                               outembdim=outembdim,
                               encdim=encdim,
                               decdim=decdim,
                               attdim=attdim,
                               bidir=bidir)

        inpseq = np.random.randint(0, inpvocsize, (batsize, inpseqlen)).astype("int32")
        outseq = np.random.randint(0, outvocsize, (batsize, outseqlen)).astype("int32")

        predenco, _, _ = m.enc.predict(inpseq)
        self.assertEqual(predenco.shape, (batsize, encdim))

        pred = m.predict(inpseq, outseq)
        self.assertEqual(pred.shape, (batsize, outseqlen, outvocsize))

