__author__ = 'denis'
import collections, os, urllib2, zipfile, bz2, re, argparse, inspect

import numpy as np
#import theano
#from theano import tensor

#from teafacto.core.blocksparse import sparse_block_dot
from datetime import datetime as dt

class ticktock(object):
    def __init__(self, prefix="", verbose=True):
        self.prefix = prefix
        self.verbose = verbose
        self.state = None
        self.perc = None
        self.prevperc = None
        self._tick()

    def tick(self, state=None):
        if self.verbose and state is not None:
            print "%s: %s" % (self.prefix, state)
        self._tick()

    def _tick(self):
        self.ticktime = dt.now()

    def _tock(self):
        return (dt.now() - self.ticktime).total_seconds()

    def progress(self, x, of):
        self.perc = int(round(100.* x/of))
        if self.perc != self.prevperc:
            print "%s: %d" % (self.prefix, self.perc)  + "%"
            self.prevperc = self.perc

    def tock(self, action=None, prefix=None):
        duration = self._tock()
        if self.verbose:
            prefix = prefix if prefix is not None else self.prefix
            action = action if action is not None else self.state
            print "%s: %s in %s" % (prefix, action, self._getdurationstr(duration))
        return self

    def _getdurationstr(self, duration):
        if duration > 60:
            acc = ""
            seconds = int(round(duration))
            acc = "%d seconds" % (seconds % 60)
            minutes = seconds // 60
            acc = "%d minutes, %s" % (minutes % 60, acc)
            if minutes < 60:
                return acc
            hours = minutes // 60
            acc = "%d hours, %s" % (hours % 24, acc)
            if hours < 24:
                return acc
            days = hours % 24
            return "%d days, %s" % (days, acc)

        else:
            return "%f seconds" % duration


def argparsify(f):
    args, _, _, defaults = inspect.getargspec(f)
    assert(len(args) == len(defaults))
    parser = argparse.ArgumentParser()
    i = 0
    for arg in args:
        parser.add_argument("-%s"%arg, "--%s"%arg, type=type(defaults[i]))
        i += 1
    par = parser.parse_args()
    kwargs = {}
    for arg in args:
        if getattr(par, arg) is not None:
            kwargs[arg] = getattr(par, arg)
    return kwargs



def issequence(x):
    return isinstance(x, collections.Sequence) and not isinstance(x, basestring)


def isnumber(x):
    return isinstance(x, float) or isinstance(x, int)


def exprandint(shape, steepness, vocsize): # generate exponentially distributed integers between 0 and vocsize
    r = np.random.random(shape)
    s = -np.log(1- (1 - np.exp(-steepness)) * r) / steepness
    return np.round(s*vocsize).astype("int32")


class TodoException(NotImplementedError):
    pass


def h_softmax(x, batch_size, n_outputs, n_classes, n_outputs_per_class,
              W1, b1, W2, b2, target=None):
    """ Two-level hierarchical softmax.
    The architecture is composed of two softmax layers: the first predicts the
    class of the input x while the second predicts the output of the input x in
    the predicted class.
    More explanations can be found in the original paper [1]_.
    If target is specified, it will only compute the outputs of the
    corresponding targets. Otherwise, if target is None, it will compute all
    the outputs.
    The outputs are grouped in the same order as they are initially defined.
    .. versionadded:: 0.7.1
    Parameters
    ----------
    x: tensor of shape (batch_size, number of features)
        the minibatch input of the two-layer hierarchical softmax.
    batch_size: int
        the size of the minibatch input x.
    n_outputs: int
        the number of outputs.
    n_classes: int
        the number of classes of the two-layer hierarchical softmax. It
        corresponds to the number of outputs of the first softmax. See note at
        the end.
    n_outputs_per_class: int
        the number of outputs per class. See note at the end.
    W1: tensor of shape (number of features of the input x, n_classes)
        the weight matrix of the first softmax, which maps the input x to the
        probabilities of the classes.
    b1: tensor of shape (n_classes,)
        the bias vector of the first softmax layer.
    W2: tensor of shape (n_classes, number of features of the input x, n_outputs_per_class)
        the weight matrix of the second softmax, which maps the input x to
        the probabilities of the outputs.
    b2: tensor of shape (n_classes, n_outputs_per_class)
        the bias vector of the second softmax layer.
    target: tensor of shape either (batch_size,) or (batch_size, 1)
        (optional, default None)
        contains the indices of the targets for the minibatch
        input x. For each input, the function computes the output for its
        corresponding target. If target is None, then all the outputs are
        computed for each input.
    Returns
    -------
    output_probs: tensor of shape (batch_size, n_outputs) or (batch_size, 1)
        Output of the two-layer hierarchical softmax for input x. If target is
        not specified (None), then all the outputs are computed and the
        returned tensor has shape (batch_size, n_outputs). Otherwise, when
        target is specified, only the corresponding outputs are computed and
        the returned tensor has thus shape (batch_size, 1).
    Notes
    -----
    The product of n_outputs_per_class and n_classes has to be greater or equal
    to n_outputs. If it is strictly greater, then the irrelevant outputs will
    be ignored.
    n_outputs_per_class and n_classes have to be the same as the corresponding
    dimensions of the tensors of W1, b1, W2 and b2.
    The most computational efficient configuration is when n_outputs_per_class
    and n_classes are equal to the square root of n_outputs.
    References
    ----------
    .. [1] J. Goodman, "Classes for Fast Maximum Entropy Training,"
        ICASSP, 2001, <http://arxiv.org/abs/cs/0108006>`.


    # First softmax that computes the probabilities of belonging to each class
    class_probs = theano.tensor.nnet.softmax(tensor.dot(x, W1) + b1)

    if target is None:  # Computes the probabilites of all the outputs

        class_ids = tensor.tile(
            tensor.arange(n_classes, dtype="int32")[None, :], (batch_size, 1))

        # Second softmax that computes the output probabilities
        activations = sparse_block_dot(
            W2[None, :, :, :], x[:, None, :],
            tensor.zeros((batch_size, 1), dtype='int32'), b2, class_ids)

        output_probs = theano.tensor.nnet.softmax(
            activations.reshape((-1, n_outputs_per_class)))
        output_probs = output_probs.reshape((batch_size, n_classes, -1))
        output_probs = class_probs[:, :, None] * output_probs
        output_probs = output_probs.reshape((batch_size, -1))
        # output_probs.shape[1] is n_classes * n_outputs_per_class, which might
        # be greater than n_outputs, so we ignore the potential irrelevant
        # outputs with the next line:
        output_probs = output_probs[:, :n_outputs]

    else:  # Computes the probabilities of the outputs specified by the targets

        target = target.flatten()

        # Classes to which belong each target
        target_classes = target // n_outputs_per_class

        # Outputs to which belong each target inside a class
        target_outputs_in_class = target % n_outputs_per_class

        # Second softmax that computes the output probabilities
        activations = sparse_block_dot(
            W2[None, :, :, :], x[:, None, :],
            tensor.zeros((batch_size, 1), dtype='int32'), b2,
            target_classes[:, None])

        output_probs = theano.tensor.nnet.softmax(activations[:, 0, :])
        target_class_probs = class_probs[tensor.arange(batch_size),
                                         target_classes]
        output_probs = output_probs[tensor.arange(batch_size),
                                    target_outputs_in_class]
        output_probs = target_class_probs * output_probs

    return output_probs
    """


class FileHandler(object):
    @staticmethod
    def check(fp):
        return os.path.exists(fp)

    @staticmethod
    def download(url, fp):
        f = None
        CHUNK = 1*1024
        try:
            f = urllib2.urlopen(url)
            # create directories if they don't exist
            destdir = os.path.dirname(fp)
            if not os.path.isdir(destdir):
                os.makedirs(destdir)
            # wrap archive type
            decompressor = None
            if re.match(r'^.+\.bz2$', url):
                decompressor = bz2.BZ2Decompressor()
            # download and write to disk
            with open(fp, "w") as tf:
                while True:
                    chunk = f.read(CHUNK)
                    if not chunk:
                        break
                    if decompressor is not None:
                        chunk = decompressor.decompress(chunk)
                    tf.write(chunk)
        except Exception, e:
            print e
            print "could not download and save %s" % url
        finally:
            if f is not None:
                f.close()

    @staticmethod
    def ensurefile(fp, url, zippath=None):
        if not FileHandler.check(fp):
            FileHandler.download(url, fp)

    @staticmethod
    def ensuredir(p):
        if not os.path.isdir(p):
            os.makedirs(p)


if __name__ == "__main__":
    def f(a=1, b=0.1, c="qsdf"):
        print a, b, c
    p = argparsify(f)
    f(**p)
