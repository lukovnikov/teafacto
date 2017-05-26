from unittest import TestCase
from teafacto.blocks.seq.rnn import RiboRNN
from teafacto.blocks.basic import VectorEmbed, SMO
import numpy as np
from teafacto.core.base import Val


class TestRiboRNN(TestCase):
    def test_shapes(self):
        inpvocsize = 20
        inpembdim = 10
        outvocsize = 5
        nsteps = 17
        seqlen = 7
        batsize = 6
        innerdim = 30

        data = np.random.randint(0, inpvocsize, (batsize, seqlen))
        labels = np.random.randint(0, outvocsize, (batsize, seqlen))

        emb = VectorEmbed(inpvocsize, inpembdim)
        smo = SMO(innerdim, outvocsize)

        m = RiboRNN(inpemb=emb, innerdim=innerdim, smo=smo, nsteps=nsteps)

        pred = m.predict(data)

        print pred[0]

        self.assertEqual(pred.shape, (batsize, nsteps, outvocsize))

    def test__get_position_weights(self):
        inpvocsize = 20
        inpembdim = 10
        outvocsize = 5
        nsteps = 3
        seqlen = 7
        batsize = 6
        innerdim = 30

        data = np.random.randint(0, inpvocsize, (batsize, seqlen))

        emb = VectorEmbed(inpvocsize, inpembdim)
        smo = SMO(innerdim, outvocsize)
        crit = Val(np.asarray([1, 1]))
        maxlen = Val(5)
        m = RiboRNN(inpemb=emb, innerdim=innerdim, smo=smo, nsteps=5)

        pred = m._RiboRNN__get_position_weights(crit, maxlen).eval()
        print pred
        self.assertTrue(np.allclose(pred, np.asarray([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]])))

    def test__get_position_weights_maskout_future(self):
        inpvocsize = 20
        inpembdim = 10
        outvocsize = 5
        nsteps = 3
        seqlen = 7
        batsize = 6
        innerdim = 30

        data = np.random.randint(0, inpvocsize, (batsize, seqlen))

        emb = VectorEmbed(inpvocsize, inpembdim)
        smo = SMO(innerdim, outvocsize)
        crit = Val(np.asarray([1, 1]))
        maxlen = Val(5)
        m = RiboRNN(inpemb=emb, innerdim=innerdim, smo=smo, nsteps=5)
        m.position_maxhot_pred = False

        pred = m._RiboRNN__get_position_weights(crit, maxlen, maskout="future").eval()
        print pred
        self.assertTrue(np.allclose(pred > 0, np.asarray([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]])))

    def test__get_raw_position_distances(self):
        inpvocsize = 20
        inpembdim = 10
        outvocsize = 5
        nsteps = 3
        seqlen = 7
        batsize = 6
        innerdim = 30

        data = np.random.randint(0, inpvocsize, (batsize, seqlen))

        emb = VectorEmbed(inpvocsize, inpembdim)
        smo = SMO(innerdim, outvocsize)
        crit = Val(np.asarray([0, 0]))
        maxlen = Val(5)
        m = RiboRNN(inpemb=emb, innerdim=innerdim, smo=smo, nsteps=5)

        pred = m._RiboRNN__get_raw_position_distances(crit, maxlen).eval()
        print pred
        self.assertEqual(pred.shape, (2, 5))

    def test__get_position_time_mask(self):
        inpvocsize = 20
        inpembdim = 10
        outvocsize = 5
        nsteps = 3
        seqlen = 7
        batsize = 6
        innerdim = 30

        data = np.random.randint(0, inpvocsize, (batsize, seqlen))

        emb = VectorEmbed(inpvocsize, inpembdim)
        smo = SMO(innerdim, outvocsize)
        crit = Val(np.asarray([2, 2]))
        maxlen = Val(5)
        m = RiboRNN(inpemb=emb, innerdim=innerdim, smo=smo, nsteps=5)

        pred1 = m._RiboRNN__get_position_time_mask(crit, maxlen, mode="past").eval()
        pred2 = m._RiboRNN__get_position_time_mask(crit, maxlen, mode="future").eval()
        print pred1
        print pred2
        self.assertTrue(np.allclose(pred1, pred2[:, ::-1]))

