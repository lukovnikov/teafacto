from unittest import TestCase
from teafacto.core.base import RVal

class TestRVal(TestCase):
    def test_value_shape(self):
        rv = RVal().binomial((25,), p=0.5)
        self.assertEqual(rv.v.shape, (25,))
