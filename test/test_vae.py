from teafacto.core.base import Block, param, tensorops as T, RVal
from teafacto.blocks.basic import Softmax
from unittest import TestCase
import numpy as np


class EmbeddingVAE(Block):
    def __init__(self, vocsize, embdim, seed=None, **kw):
        super(EmbeddingVAE, self).__init__(**kw)
        self.W_m = param((vocsize, embdim)).glorotuniform()
        self.W_s = param((vocsize, embdim)).glorotuniform()
        self.O = param((embdim, vocsize)).glorotuniform()
        self.embdim = embdim
        self.seed = seed

    def apply(self, x):     # (batsize,)
        m = self.W_m[x]
        s = self.W_s[x]
        z = RVal(seed=self.seed).normal(m.shape) * s + m
        o = T.dot(z, self.O)
        return Softmax()(o)


class EmbeddingVAETest(TestCase):
    def test_shape(self):
        vocsize = 12
        batsize = 3
        embdim = 2
        seed = 1337
        np.random.seed(1337)
        x = np.random.randint(0, vocsize, (batsize,))
        vae = EmbeddingVAE(vocsize, embdim, seed=seed)
        o1 = vae.predict(x)
        self.assertEqual(o1.shape, (batsize, vocsize))

    # TODO : KL div: 1/2 * (log(sqrt(d)) - log(|s|) + sum(s) - d + sum(m**2))

