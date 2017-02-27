from unittest import TestCase
import numpy as np
from IPython import embed
from teafacto.examples.dummy import Dummy
from teafacto.eval.extvalid import Accuracy, Perplexity
from teafacto.core.base import asblock
from teafacto.blocks.basic import SMO


class TestExternalValidators(TestCase):
    def test_external_validator(self):
        vocabsize = 1000
        m = Dummy(indim=vocabsize, dim=10, outdim=2000)
        numbats = 20
        lr = 0.8
        data = np.arange(0, vocabsize).astype("int32")
        gdata = np.random.randint(0, 2000, (vocabsize,))

        mpredf = m.predict

        def extacc(*sampleinp):
            pred = mpredf(*sampleinp[:-1])
            ret = np.sum(np.argmax(pred, axis=1) == sampleinp[-1])
            return ret * 1. / sampleinp[-1].shape[0]

        _, err, verr, _, _ = \
            m.train([data], gdata).adadelta(lr=lr).cross_entropy() \
             .autovalidate().cross_entropy().extvalid(extacc).accuracy() \
            .train(numbats=numbats, epochs=10, returnerrors=True)

        verr = np.asarray(verr)
        verr = verr[:, 1] + verr[:, 2]
        self.assertTrue(np.allclose(verr, np.ones_like(verr)))


class TestExternalAccuracy(TestCase):
    def test_accuracy(self):
        vocabsize = 1000
        m = Dummy(indim=vocabsize, dim=10, outdim=2000)
        numbats = 20
        lr = 0.8
        data = np.arange(0, vocabsize).astype("int32")
        gdata = np.random.randint(0, 2000, (vocabsize,))

        _, err, verr, _, _ = \
            m.train([data], gdata).adadelta(lr=lr).cross_entropy() \
             .autovalidate().cross_entropy().extvalid(Accuracy(m)).accuracy() \
            .train(numbats=numbats, epochs=10, returnerrors=True)

        verr = np.asarray(verr)
        verr = verr[:, 1] + verr[:, 2]
        self.assertTrue(np.allclose(verr, np.ones_like(verr)))


from teafacto.blocks.cnn import CNNSeqEncoder
from teafacto.blocks.seq.rnn import SeqEncoder
from teafacto.core.base import Block


class TestExternalPerplexity(TestCase):
    def test_perplexity(self):
        vocabsize = 100
        numbats = 5
        seqlen = 15
        lr = 0.1
        data = np.random.randint(0, vocabsize, (numbats*20, seqlen))

        enc = CNNSeqEncoder(inpvocsize=vocabsize, inpembdim=10, maskid=0,
                          innerdim=[20, 20, 30]).all_outputs()

        enc0 = SeqEncoder.fluent().vectorembedder(vocabsize, 10, maskid=0)\
            .addlayers([40], bidir=True).addlayers([30]).make().all_outputs()

        smo = SMO(30, vocabsize)

        m = asblock(lambda x: smo(enc(x)))

        _, err, verr, _, _ = \
            m.train([data[:, :-1]], data[:, 1:]).adadelta(lr=lr).cross_entropy() \
             .validate_on([data[:, :-1]], data[:, 1:]).cross_entropy().extvalid(Perplexity(m)) \
            .train(numbats=numbats, epochs=10, returnerrors=True)


