from unittest import TestCase
from teafacto.blocks.linear import Perceptron, SVM, LogisticRegression
import numpy as np, matplotlib.pyplot as plt


class TestLinear(TestCase):
    def do_test_linear(self, lc, mean=(-0.2, 0.3), covfactor=0.1, shift=5):
        lr = 0.1
        numfeats = lc.dim
        numsam = 300
        posmean = [x + shift for x in mean]
        negmean = [-x + shift for x in mean]
        np.random.seed(789456)
        posdata = np.random.multivariate_normal(posmean, [[1*covfactor, 0.], [3*covfactor, 1*covfactor]], numsam).T
        negdata = np.random.multivariate_normal(negmean, [[1*covfactor, 0.], [3*covfactor, 1*covfactor]], numsam).T
        data = np.concatenate([posdata.T, negdata.T], axis=0).astype("float32")
        posgold = np.ones((posdata.shape[1],))
        neggold = -np.ones((negdata.shape[1],))
        gold = np.concatenate([posgold, neggold]).astype("int32")
        print data.shape, gold.shape

        lc.train([data], gold).sgd(lr).train(epochs=300, numbats=1)
        pred = lc.predict(data)
        print pred


        # plotting
        plt.plot(posdata[0], posdata[1], '.g')
        plt.plot(negdata[0], negdata[1], '.r')

        arrow_x, arrow_y = lc.w.d.get_value()
        bias = lc.b.d.get_value()[0]
        slope = arrow_y / arrow_x
        a_x = 0
        a_y = 0
        print arrow_x, arrow_y, bias, "\n"
        plt.arrow(a_x, a_y, arrow_x*1.+a_x, arrow_y*1.+a_y, fc="b", ec="b")
        slope = -arrow_x / arrow_y
        print plt.xlim(), slope
        plt.plot([a_x, plt.xlim()[1]+a_x], [a_y, plt.xlim()[1]*slope+a_y], "k-")
        plt.plot([a_x, plt.xlim()[0]+a_x], [a_y, plt.xlim()[0]*slope+a_y], "k-")
        plt.axis('equal')
        plt.show()

    def test_perceptron(self):
        self.do_test_linear(Perceptron(2))

    def test_svm(self):
        self.do_test_linear(SVM(2))

    def test_logreg(self):
        self.do_test_linear(LogisticRegression(2))
