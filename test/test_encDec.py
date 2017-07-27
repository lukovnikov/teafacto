from unittest import TestCase
from teafacto.core.base import tensorops as T, Val
from teafacto.blocks.seq.encdec import EncDec
from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.blocks.seq.attention import Attention
from teafacto.blocks.basic import VectorEmbed, SMO
import numpy as np


class TestEncDec(TestCase):
    def setUp(self):
        vocsize = 10
        embdim = 5
        dim = 10
        encoder = RNNSeqEncoder.fluent()\
            .vectorembedder(vocsize, embdim, maskid=0)\
            .addlayers(dim=dim)\
            .make()\
            .all_outputs()
        decemb = VectorEmbed(vocsize, embdim, maskid=0)
        attention = Attention()
        attention.forward_gen(dim, dim, dim)
        smo = SMO(dim+dim+embdim, vocsize)
        encdec = EncDec(encoder=encoder,
                        attention=attention,
                        smo=smo,
                        inpemb=decemb,
                        indim=dim+embdim,
                        innerdim=dim,
                        outconcat=True,
                        concatdecinptoout=True,
                        return_attention_weights=True,
                        )
        self.encdec = encdec
        self.vocsize = vocsize
        self.dim = dim

    def test_shapes_and_attention_mask(self):
        batsize = 7
        seqlen = 6
        inpseq = np.random.randint(1, self.vocsize, (batsize, seqlen))
        inpseq[:, 4:] = 0
        inpseq[2, 3:] = 0
        inpseq = Val(inpseq)

        outseq = np.random.randint(1, self.vocsize, (batsize, seqlen))
        outseq = Val(outseq)

        predvar, attvar = self.encdec(outseq[:, :-1], inpseq)

        pred = predvar.eval()
        att = attvar.eval()
        self.assertEqual(pred.shape, (batsize, seqlen-1, self.dim))
        self.assertEqual(att.shape, (batsize, seqlen-1, seqlen))
        print pred.shape

        print att.shape
        print att[1:3, :, :]
        self.assertTrue(np.allclose(att[:, :, -2:], np.zeros_like(att[:, :, -2:])))
        self.assertTrue(np.allclose(att[2, :, -3:], np.zeros_like(att[2, :, -3:])))
