from unittest import TestCase
from teafacto.blocks.basic import IdxToOneHot, MatDot, Linear, Softmax, Switch
from teafacto.core.base import Val
from teafacto.core.stack import stack
import numpy as np

class TestBasic(TestCase):
    def test_idx_to_one_hot(self):
        ioh = IdxToOneHot(25)
        data = np.arange(0, 25).astype("int32")
        expout = np.eye(25, 25)
        outioh = ioh.predict(data)
        self.assertEqual(np.linalg.norm(expout - outioh), 0)


class TestMatDot(TestCase):
    def setUp(self):
        self.matdot = MatDot(indim=10, dim=15)
        self.data = np.random.random((100, 10))
        self.matdotout = self.matdot.predict(self.data)

    def test_matdot_shapes(self):
        self.assertEqual(self.matdotout.shape, (100, 15))

    def test_matdot_output(self):
        self.assertTrue(np.allclose(self.matdotout, np.dot(self.data, self.matdot.W.d.get_value())))

    def test_set_lr(self):
        self.matdot = MatDot(indim=10, dim=15)




class TestLinear(TestCase):
    def setUp(self):
        self.linear = Linear(indim=10, dim=15)
        self.data = np.random.random((100, 10))
        self.out = self.linear.predict(self.data)

    def test_linear_shapes(self):
        self.assertEqual(self.out.shape, (100, 15))

    def test_linear_output(self):
        self.assertTrue(np.allclose(self.out, np.dot(self.data, self.linear.W.d.get_value()) + self.linear.b.d.get_value()))

    def test_set_lr(self):
        lin = Linear(indim=10, dim=15)
        lin.set_lr(0.123)
        o = lin(Val(0))
        #print ["{}: {}".format(x, x.lrmul) for x in o.allparams]
        for x in o.allparams:
            self.assertEqual(x.lrmul, 0.123)

    def test_get_params(self):
        lin = Linear(indim=10, dim=15)
        params = {lin.W, lin.b}
        self.assertEqual(params, lin.get_params())

    def test_multilevel_set_lr(self):
        l1 = Linear(10, 11)
        l2 = Linear(11, 12)
        l3 = Linear(12, 13)
        s = stack(l1, l2, l3)
        s[1].set_lr(0.5)
        s[2].set_lr(0.1)
        o = s(Val(0))
        l1o = s[0](Val(0))
        l2o = s[1](Val(0))
        l3o = s[2](Val(0))
        print ["{}: {}".format(x, x.lrmul) for x in o.allparams]
        for x in o.allparams:
            if x in l1o.allparams:
                self.assertEqual(x.lrmul, 1.0)
            elif x in l2o.allparams:
                self.assertEqual(x.lrmul, 0.5)
            elif x in l3o.allparams:
                self.assertEqual(x.lrmul, 0.1)
        s.set_lr(0.21)
        o = s(Val(0))
        print ["{}: {}".format(x, x.lrmul) for x in o.allparams]
        for x in o.allparams:
            self.assertEqual(x.lrmul, 0.21)


class TestCompoundVar(TestCase):
    def test_compound_var(self):
        aval = np.zeros((10, 10))
        bval = np.ones((10, 10))
        maskval = np.repeat(np.asarray([[0,1,0,0,1,0,0,1,1,1]]).T, 10, axis=1)
        print maskval
        print aval * maskval + bval * (1 - maskval)
        a = Val(aval)
        b = Val(bval)
        mask = Val(maskval)
        cv = Switch(a, b, mask)
        cvpred = cv().eval()
        print cvpred
        self.assertTrue(np.allclose(cvpred, aval * maskval + bval * (1 - maskval)))

