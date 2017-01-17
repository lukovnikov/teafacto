from unittest import TestCase, main
from teafacto.examples.dummy import Dummy
from teafacto.blocks.seq.encdec import SimpleSeqEncDecAtt
import numpy as np

class TestSaveable(TestCase):
    def test_continued_training_dummy(self):
        m = Dummy(20, 5)
        data = np.random.randint(0, 20, (50,))
        r = m.train([data], data).cross_entropy().adadelta().train(5, 10, returnerrors=True)
        terrs = r[1]
        print "\n".join(map(str, terrs))
        for i in range(0, len(terrs) - 1):
            self.assertTrue(terrs[i+1] < terrs[i])
        m.save("/tmp/testmodelsave")
        m = m.load("/tmp/testmodelsave")
        r = m.train([data], data).cross_entropy().adadelta().train(5, 10, returnerrors=True)
        nterrs = r[1]
        print "\n".join(map(str, nterrs))
        for i in range(0, len(nterrs) - 1):
            self.assertTrue(nterrs[i+1] < nterrs[i])
        self.assertTrue(nterrs[0] < terrs[-1])

    def test_continued_training_encdec(self):
        m = SimpleSeqEncDecAtt(inpvocsize=20, outvocsize=20, inpembdim=5, outembdim=5, encdim=10, decdim=10)
        data = np.random.randint(0, 20, (50, 7))
        r = m.train([data, data[:, :-1]], data[:, 1:]).cross_entropy().adadelta().train(5, 10, returnerrors=True)
        a = r[1]
        print "\n".join(map(str, a))
        for i in range(0, len(a) - 1):
            self.assertTrue(a[i+1] < a[i])
        m.get_params()
        m.save("/tmp/testmodelsave")
        m = m.load("/tmp/testmodelsave")
        r = m.train([data, data[:, :-1]], data[:, 1:]).cross_entropy().adadelta().train(5, 10, returnerrors=True)
        b = r[1]
        print "\n".join(map(str, b))
        for i in range(0, len(b) - 1):
            self.assertTrue(b[i+1] < b[i])
        self.assertTrue(b[0] < a[-1])


if __name__ == "__main__":
    main()
