from unittest import TestCase
from teafacto.scripts.atisslotfil import SimpleSeqTransducer
import numpy as np


class TestSeqTransducer(TestCase):
    def test_output_shape(self):
        # settings
        batsize = 10
        seqlen = 57
        invocsize = 500
        inembdim = 50
        innerdim = 113
        outvocsize = 179

        # data
        traindata = np.random.randint(0, invocsize, (batsize, seqlen))
        traingold = np.random.randint(0, outvocsize, (batsize, seqlen))

        # model
        m = SimpleSeqTransducer(indim=invocsize, embdim=inembdim, innerdim=innerdim, outdim=outvocsize)

        pred = m.predict(traindata)
        self.assertEqual(pred.shape, (batsize, seqlen, outvocsize))