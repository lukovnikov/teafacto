from unittest import TestCase
from teafacto.blocks.attention import vec2seq
import numpy as np


class TestVec2sec(TestCase):
    def setUp(self):
        self.block = vec2seq(indim=20, innerdim=50, seqlen=5, vocsize=10)
        self.data = np.random.random((100, 20))
        self.out = self.block.predict(self.data)

    def test_output_shape(self):
        self.assertEqual(self.out.shape, (100, 5, 10))

    def test_outputs_are_probabilities(self):
        for i in range(self.out.shape[0]):
            for j in range(self.out.shape[1]):
                self.assertTrue(np.isclose(np.sum(self.out[i, j, :]), 1.0))
