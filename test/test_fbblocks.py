from unittest import TestCase
from teafacto.blocks.kgraph.fbencdec import FBBasicCompositeEncoder
import numpy as np


class TestFBBasicCompositeEncoder(TestCase):
    def test_output_shape(self):
        batsize = 100
        wordembdim = 50
        wordencdim = 20
        innerdim = 40
        datanuments = 77
        vocnumwords = 100
        numchars = 10
        wseqlen = 3
        cseqlen = 5

        m = FBBasicCompositeEncoder(
            wordembdim=wordembdim,
            wordencdim=wordencdim,
            innerdim=innerdim,
            outdim=datanuments,
            numchars=numchars,
            numwords=vocnumwords,
        )

        worddata = np.random.randint(0, vocnumwords, (batsize, wseqlen, 1))
        chardata = np.random.randint(0, numchars, (batsize, wseqlen, cseqlen))
        data = np.concatenate([worddata, chardata], axis=2)

        predshape = m.predict(data).shape
        self.assertEqual(predshape, (batsize, datanuments))
