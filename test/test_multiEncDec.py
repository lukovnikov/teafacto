from unittest import TestCase
from teafacto.blocks.seq.encdec import MultiEncDec
from teafacto.blocks.seq.rnn import SeqEncoder
from teafacto.blocks.seq.attention import Attention
from teafacto.blocks.basic import SMO, VectorEmbed
from teafacto.core.base import asblock
from teafacto.blocks.cnn import CNNSeqEncoder
import numpy as np


class TestMultiEncDec(TestCase):
    def setUp(self):
        vocsize = 101
        outvocsize = 120
        embdim = 50
        maskid = 0

        encoder_one = SeqEncoder.fluent()\
                            .vectorembedder(vocsize, embdim, maskid)\
                            .addlayers([40], bidir=True).addlayers([50])\
                        .make().all_outputs()
        encoder_two = SeqEncoder.fluent()\
                            .vectorembedder(vocsize, embdim, maskid)\
                            .addlayers([40], bidir=True).addlayers([50])\
                        .make().all_outputs()

        splitters = (asblock(lambda x: x[:, :, :20]), asblock(lambda x: x[:, :, 20:]))
        attention_one = Attention(splitters=splitters)
        splitters = (asblock(lambda x: x[:, :, :20]), asblock(lambda x: x[:, :, 20:]))
        attention_two = Attention(splitters=splitters)

        smo = SMO(25, outvocsize)

        slices = (20, 20)

        self.m = MultiEncDec(encoders=[encoder_one, encoder_two],
                        indim=30+30+embdim,
                        slices=slices,
                        attentions=[attention_one, attention_two],
                        inpemb=VectorEmbed(outvocsize, embdim, maskid=maskid),
                        smo=smo,
                        innerdim=[99])

        self.vocsize = vocsize
        self.outvocsize = outvocsize

    def test_prediction_shape(self):
        batsize = 10
        seqlen = 11
        vocsize = self.vocsize
        outvocsize = self.outvocsize

        data_one = np.random.randint(1, vocsize, (batsize, seqlen))
        data_two = np.random.randint(1, vocsize, (batsize, seqlen))
        data_dec = np.random.randint(1, outvocsize, (batsize, seqlen))

        pred = self.m.predict(data_dec, data_one, data_two)
        self.assertEqual(pred.shape, (batsize, seqlen, outvocsize))

    def test_training(self):
        numsam = 1001
        seqlen = 11
        vocsize = self.vocsize
        outvocsize = self.outvocsize
        lr = 0.5

        data_one = np.random.randint(1, vocsize, (numsam, seqlen))
        data_one[:, -3:] = 0
        data_two = np.random.randint(1, vocsize, (numsam, seqlen))
        data_two[:, -4:] = 0
        data_dec = np.random.randint(1, outvocsize, (numsam, seqlen))
        data_dec[:, 3:] = 0

        self.m.train([data_dec[:, :-1], data_one, data_two], data_dec[:, 1:])\
            .adadelta(lr=lr).cross_entropy()\
            .train(numbats=100, epochs=5)

        pred = self.m.predict(data_dec, data_one, data_two)
        self.assertEqual(pred.shape, (numsam, seqlen, outvocsize))


class TestMultiEncDecCNNEncoders(TestCase):
    def setUp(self):
        vocsize = 101
        outvocsize = 120
        embdim = 50
        maskid = 0

        encoder_one = CNNSeqEncoder(inpvocsize=vocsize, inpembdim=embdim, maskid=maskid,
                                    window=[3, 4, 5], innerdim=[50, 40, 50]).all_outputs()
        encoder_two = CNNSeqEncoder(inpvocsize=vocsize, inpembdim=embdim, maskid=maskid,
                                    window=[3, 4, 5], innerdim=[50, 40, 50]).all_outputs()


        splitters = (asblock(lambda x: x[:, :, :20]), asblock(lambda x: x[:, :, 20:]))
        attention_one = Attention(splitters=splitters)
        splitters = (asblock(lambda x: x[:, :, :20]), asblock(lambda x: x[:, :, 20:]))
        attention_two = Attention(splitters=splitters)

        smo = SMO(25, outvocsize)

        slices = (20, 20)

        self.m = MultiEncDec(encoders=[encoder_one, encoder_two],
                        indim=30+30+embdim,
                        slices=slices,
                        attentions=[attention_one, attention_two],
                        inpemb=VectorEmbed(outvocsize, embdim, maskid=maskid),
                        smo=smo,
                        innerdim=[99])

        self.vocsize = vocsize
        self.outvocsize = outvocsize

    def test_prediction_shape(self):
        batsize = 10
        seqlen = 11
        vocsize = self.vocsize
        outvocsize = self.outvocsize

        data_one = np.random.randint(1, vocsize, (batsize, seqlen))
        data_two = np.random.randint(1, vocsize, (batsize, seqlen))
        data_dec = np.random.randint(1, outvocsize, (batsize, seqlen))

        pred = self.m.predict(data_dec, data_one, data_two)
        self.assertEqual(pred.shape, (batsize, seqlen, outvocsize))

    def test_training(self):
        numsam = 1000
        seqlen = 11
        vocsize = self.vocsize
        outvocsize = self.outvocsize
        lr = 0.5

        data_one = np.random.randint(1, vocsize, (numsam, seqlen))
        data_one[:, -3:] = 0
        data_two = np.random.randint(1, vocsize, (numsam, seqlen))
        data_two[:, -4:] = 0
        data_dec = np.random.randint(1, outvocsize, (numsam, seqlen))
        data_dec[:, 3:] = 0

        self.m.train([data_dec[:, :-1], data_one, data_two], data_dec[:, 1:])\
            .adadelta(lr=lr).cross_entropy().train(numbats=100, epochs=5)

        pred = self.m.predict(data_dec, data_one, data_two)
        self.assertEqual(pred.shape, (numsam, seqlen, outvocsize))





