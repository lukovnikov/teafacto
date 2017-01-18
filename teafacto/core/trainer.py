import sys, gc, inspect
from pympler import asizeof
from datetime import datetime as dt
from IPython import embed

import numpy as np
import theano
from lasagne.objectives import *
from lasagne.regularization import l1, l2
from lasagne.updates import *
from theano import tensor as tensor
from theano.compile.nanguardmode import NanGuardMode

#from core import Input
from teafacto.core.datafeed import DataFeeder, SplitIdxIterator
from teafacto.util import ticktock as TT


class DynamicLearningParam(object):
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, lr, epoch, maxiter, terrs, verrs): # get new learning rate based on old one, epoch, maxiter, training error, validation errors
        raise NotImplementedError("use subclass")


class thresh_lr(DynamicLearningParam):
    def __init__(self, lr, thresh=5):
        super(thresh_lr, self).__init__(lr)
        self.thresh = thresh

    def __call__(self, lr, epoch, maxiter, terrs, verrs):
        return lr if epoch < self.thresh else 0.


class ModelTrainer(object):
    def __init__(self, model, gold):
        self.model = model
        self.goldvar = gold
        self.validsetmode= False
        self.average_err = True # TODO: do we still need this?
        self._autosave = False
        self._autosavepath = None
        self._autosaveblock = None
        # training settings
        self.learning_rate = None
        self.dynamic_lr = None
        self.objective = None
        self.regularizer = None
        self._exp_mov_avg_decay = 0.0
        self.optimizer = None
        self.traindata = None
        self.traingold = None
        self.gradconstraints = []
        self._sampletransformers = []
        # validation settings
        self._validinter = 1
        self.trainstrategy = self._train_full
        self.validsplits = 0
        self.validrandom = False
        self.validata = None
        self.validgold = None
        self.validation = None
        self.validators = []
        self.tt = TT("FluentTrainer")
        # taking best
        self.besttaker = None
        self.bestmodel = None


    #region ====================== settings =============================

    #region ################### LOSSES ##########################

    def _set_objective(self, obj):
        if self.validsetmode is False:
            self.objective = obj
        else:
            self.validators.append(obj)

    def linear_objective(self): # multiplies prediction with gold, assumes prediction is already the loss
                                # (this is for negative sampling models where the training model already computes the loss)
        self._set_objective(lambda x, y: x * y)
        return self

    def cross_entropy(self):
        """ own implementation of categorical cross-entropy """
        self._set_objective(self._inner_cross_entropy)
        return self

    @classmethod
    def _inner_cross_entropy(cls, probs, gold, mask=None):
        if gold.ndim == 1:
            assert(mask is None)
            return tensor.nnet.categorical_crossentropy(probs, gold) #-tensor.log(probs[tensor.arange(gold.shape[0]), gold])
        elif gold.ndim == 2:    # sequences
            return cls._inner_seq_neg_log_prob(probs, gold, mask=mask)

    def seq_cross_entropy(self): # probs (batsize, seqlen, vocsize) + gold: (batsize, seqlen) ==> sum of neg log-probs of correct seq
        """ Own implementation of categorical cross-entropy, applied to a sequence of probabilities that should be multiplied """
        self._set_objective(self._inner_seq_neg_log_prob)
        return self

    @classmethod
    def _inner_seq_neg_log_prob(cls, probs, gold, mask=None):   # probs: (batsize, seqlen, vocsize) probs, gold: (batsize, seqlen) idxs
        #print "using inner seq neg log prob"
        def _f(probsmat, goldvec):      # probsmat: (seqlen, vocsize), goldvec: (seqlen,)
            ce = tensor.nnet.categorical_crossentropy(probsmat, goldvec) #-tensor.log(probsmat[tensor.arange(probsmat.shape[0]), goldvec])
            return ce       # (seqlen,) ==> (1,)
        o, _ = theano.scan(fn=_f, sequences=[probs, gold], outputs_info=None)      # out: (batsize, seqlen)
        #print "MASK!!" if mask is not None else "NO MASK!!!"
        o = o * mask if mask is not None else o     # (batsize, seqlen)
        o = tensor.sum(o, axis=1)
        return o        # (batsize,)

    def squared_error(self):
        self._set_objective(squared_error)
        return self

    def squared_loss(self):
        self._set_objective(lambda x, y: (1 - x * y) ** 2)        # [-1, +1](batsize, )
        return self

    def binary_cross_entropy(self): # theano binary cross entropy (through lasagne), probs: (batsize,) float, gold: (batsize,) float
        self._set_objective(binary_crossentropy)
        return self

    def bin_accuracy(self, sep=0):
        self._set_objective(lambda x, y: theano.tensor.eq(x > sep, y > sep))
        return self

    def accuracy(self, top_k=1):
        def categorical_accuracy(predictions, targets, top_k=1): # !!! copied from Lasagne # TODO: import properly
            if targets.ndim == predictions.ndim:
                targets = theano.tensor.argmax(targets, axis=-1)
            elif targets.ndim != predictions.ndim - 1:
                raise TypeError('rank mismatch between targets and predictions')

            if top_k == 1:
                # standard categorical accuracy
                top = theano.tensor.argmax(predictions, axis=-1)
                return theano.tensor.eq(top, targets)
            else:
                # top-k accuracy
                top = theano.tensor.argsort(predictions, axis=-1)
                # (Theano cannot index with [..., -top_k:], we need to simulate that)
                top = top[[slice(None) for _ in range(top.ndim - 1)] +
                          [slice(-top_k, None)]]
                targets = theano.tensor.shape_padaxis(targets, axis=-1)
                return theano.tensor.any(theano.tensor.eq(top, targets), axis=-1)
        self._set_objective(lambda x, y: 1-categorical_accuracy(x, y, top_k=top_k))
        return self

    def seq_accuracy(self): # sequences must be exactly the same
        def inner(probs, gold, mask=None):
            if gold.ndim == probs.ndim:
                gold = tensor.argmax(gold, axis=-1)
            elif gold.ndim != probs.ndim - 1:
                raise TypeError('rank mismatch between targets and predictions')
            top = tensor.argmax(probs, axis=-1)
            assert(gold.ndim == 2 and top.ndim == 2)
            assert(mask is None or mask.ndim == 2)
            if mask is not None:
                gold *= mask
                top *= mask
            diff = tensor.sum(abs(top - gold), axis=1)
            return tensor.eq(diff, tensor.zeros_like(diff))
        self._set_objective(inner)
        return self

    def hinge_loss(self, margin=1., labelbin=True): # gold must be -1 or 1 if labelbin if False, otherwise 0 or 1
        def inner(preds, gold):     # preds: (batsize,), gold: (batsize,)
            if labelbin is True:
                gold = 2 * gold - 1
            return tensor.nnet.relu(margin - gold * preds)
        self._set_objective(inner)
        return self

    def multiclass_hinge_loss(self, margin=1.):
        def inner(preds, gold):     # preds: (batsize, numclasses) scores, gold: int:(batsize)
            pass
        self._set_objective(inner)
        return self

    def log_loss(self):
        """ NOT cross-entropy, BUT log(1+e^(-t*y))"""
        def inner(preds, gold):     # preds: (batsize,) float, gold: (batsize,) float
            return tensor.nnet.softplus(-gold*preds)
        self._set_objective(inner)
        return self
    #endregion

    #region ################### GRADIENT CONSTRAINTS ############ --> applied in the order that they were added
    def grad_total_norm(self, max_norm, epsilon=1e-7):
        self.gradconstraints.append(lambda allgrads: total_norm_constraint(allgrads, max_norm, epsilon=epsilon))
        return self

    def grad_add_constraintf(self, f):
        self.gradconstraints.append(f)
        return self

    def _gradconstrain(self, allgrads):
        ret = allgrads
        for gcf in self.gradconstraints:
            ret = gcf(ret)
        return ret

    # !!! can add more
    #endregion

    #region #################### REGULARIZERS ####################
    def _regul(self, regf, amount, params):
        return amount * reduce(lambda x, y: x+y, [regf(x.d)*x.regmul for x in params], 0)

    def l2(self, amount):
        self.regularizer = lambda x: self._regul(l2, amount, x)
        return self

    def l1(self, amount):
        self.regularizer = lambda x: self._regul(l1, amount, x)
        return self

    def exp_mov_avg(self, decay=0.0):
        self._exp_mov_avg_decay = decay
        return self
    #endregion

    #region ###################  LEARNING RATE ###################
    def lr(self, lr):
        self._setlr(lr)
        return self

    def _setlr(self, lr):
        if isinstance(lr, DynamicLearningParam):
            self.dynamic_lr = lr
            lr = lr.lr
        self.learning_rate = theano.shared(np.cast[theano.config.floatX](lr))

    def _update_lr(self, epoch, maxepoch, terrs, verrs):
        if self.dynamic_lr is not None:
            self.learning_rate.set_value(
                np.cast[theano.config.floatX](
                    self.dynamic_lr(self.learning_rate.get_value(),
                                    epoch, maxepoch, terrs, verrs)))

    def dlr_thresh(self, thresh=5):
        self.dynamic_lr = thresh_lr(self.learning_rate, thresh=thresh)
        return self


    def dlr_exp_decay(self, decay=0.5):
        pass
    #endregion

    #region #################### OPTIMIZERS ######################
    def sgd(self, lr):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: sgd(x, y, learning_rate=l)
        return self

    def momentum(self, lr, mome=0.9):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: momentum(x, y, learning_rate=l, momentum=mome)
        return self

    def nesterov_momentum(self, lr, momentum=0.9):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: nesterov_momentum(x, y, learning_rate=l, momentum=momentum)
        return self

    def adagrad(self, lr=1.0, epsilon=1e-6):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: adagrad(x, y, learning_rate=l, epsilon=epsilon)
        return self

    def rmsprop(self, lr=1., rho=0.9, epsilon=1e-6):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: rmsprop(x, y, learning_rate=l, rho=rho, epsilon=epsilon)
        return self

    def adadelta(self, lr=1., rho=0.95, epsilon=1e-6):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: adadelta(x, y, learning_rate=l, rho=rho, epsilon=epsilon)
        return self

    def adam(self, lr=0.001, b1=0.9, b2=0.999, epsilon=1e-8):
        self._setlr(lr)
        self.optimizer = lambda x, y, l: adam(x, y, learning_rate=l, beta1=b1, beta2=b2, epsilon=epsilon)
        return self
    #endregion

    #region ################### VALIDATION ####################### --> use one of following

    def validinter(self, validinter=1):
        self._validinter = validinter
        return self

    def autovalidate(self, splits=5, random=True): # validates on the same data as training data
        self.validate_on(self.traindata, self.traingold, splits=splits, random=random)
        self.validsetmode = True
        return self

    def split_validate(self, splits=5, random=True):
        self.trainstrategy = self._train_split
        self.validsplits = splits
        self.validrandom = random
        self.validsetmode = True
        return self

    def validate_on(self, data, gold=None, splits=1, random=True):
        self.trainstrategy = self._train_validdata
        self.validdata = data
        self.validgold = gold
        self.validsplits = splits
        self.validrandom = random
        self.validsetmode = True
        return self

    def cross_validate(self, splits=5, random=False):
        self.trainstrategy = self._train_cross_valid
        self.validsplits = splits
        self.validrandom = random
        self.validsetmode = True
        return self
    #endregion

    #region ######################### SELECTING THE BEST ######################
    def takebest(self, f=None):
        if f is None:
            f = lambda x: x[1]   # pick the model with the best first validation score
        self.besttaker = f
        self.bestmodel = (None, float("inf"))
        return self
    #endregion
    #endregion

    #region ====================== execution ============================

    #region ######################### ACTUAL TRAINING #########################
    def traincheck(self):
        assert(self.optimizer is not None)
        assert(self.objective is not None)
        assert(self.traindata is not None)
        assert(self.traingold is not None)

    def train(self, numbats, epochs, returnerrors=False):
        self.traincheck()
        self.numbats = numbats
        self.maxiter = epochs
        errors = self.trainstrategy()       # trains according to chosen training strategy, returns errors
        if self.besttaker is not None:      # unfreezes best model if best choosing was chosen
            self.model = self.model.__class__.unfreeze(self.bestmodel[0])
            self.tt.tock("unfroze best model (%.3f) - " % self.bestmodel[1]).tick()
        ret = self.model
        if returnerrors:
            ret = (ret,) + errors
        return ret

    def get_learning_rate(self):
        return self.learning_rate

    def autobuild_model(self, model, *traindata, **kw):
        return model.autobuild(*traindata, **kw)

    def buildtrainfun(self, model, batsize):
        self.tt.tick("training - autobuilding")
        with model.trainmode(True):
            inps, outps = self.autobuild_model(model, *self.traindata, _trainmode=True, _batsize=batsize)
            assert(len(outps) == 1)
            outp = outps[0]
            self.tt.tock("training - autobuilt")
            self.tt.tick("compiling training function")
            params = outp.allparams
            nonparams = [p for p in params if not p.lrmul > 0]
            params = [p for p in params if p.lrmul > 0]
            scanupdates = outp.allupdates
            inputs = inps
            loss, newinp = self.buildlosses(outp, [self.objective])
            loss = loss[0]
            if newinp is not None:
                inputs = newinp
            if self.regularizer is not None:
                reg = self.regularizer(params)
                cost = loss+reg
            else:
                cost = loss
            # theano.printing.debugprint(cost)
            # theano.printing.pydotprint(cost, outfile="pics/debug.png")
            updates = []
            print "params:\n " + "".join(
                map(lambda x: "\t%s\n" % str(x),
                    sorted(params, key=lambda x: str(x))))
            if len(nonparams) > 0:
                print "non-params:\n " + "".join(
                    map(lambda x: "\t%s\n" % str(x),
                        sorted(nonparams, key=lambda x: str(x))))
            print "\n\t\t (in buildtrainfun(), trainer.py) \n"
            self.tt.msg("computing gradients")
            #grads = []
            #for x in params:
            #    self.tt.msg("computing gradient for %s" % str(x))
            #    grads.append(tensor.grad(cost, x.d))
            grads = tensor.grad(cost, [x.d for x in params])  # compute gradient
            self.tt.msg("computed gradients")
            grads = self._gradconstrain(grads)
            for param, grad in zip(params, grads):
                upds = self.optimizer([grad], [param.d], self.get_learning_rate() * param.lrmul)
                newparamval = None

                for upd in upds:
                    broken = False
                    for para in params:
                        if para.d == upd:
                            newparamval = upds[upd]
                            newparamval = para.constraintf()(newparamval)
                            updates.append((upd, newparamval))
                            broken = True
                            break
                    if not broken:
                        updates.append((upd, upds[upd]))
                if self._exp_mov_avg_decay > 0:
                    # initialize ema_value in params and add EMA updates
                        param.ema_value = theano.shared(param.value.get_value())
                        updates.append((param.ema_value,
                            param.ema_value * self._exp_mov_avg_decay + newparamval * (1 - self._exp_mov_avg_decay)))
            #print updates
            #embed()
            finputs = [x.d for x in inputs] + [self.goldvar]
            allupdates = updates + scanupdates.items()
            trainf = theano.function(
                inputs=finputs,
                outputs=[cost],
                updates=allupdates,
                #mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False)
                # TODO: enabling NanGuard with Dropout doesn't work --> see Theano.git/issues/4823
            )
            self.tt.tock("training function compiled")
        return trainf

    def buildlosses(self, output, objs):
        acc = []
        for objective in objs:
            if "mask" in inspect.getargspec(objective)[0]:
                mask = output.mask.d if output.mask is not None else None
                obj = objective(output.d, self.goldvar, mask=mask)
            else:
                assert(output.mask is None)
                obj = objective(output.d, self.goldvar)
            objagg = aggregate(obj, mode="mean" if self.average_err is True else "sum")
            acc.append(objagg)
        return acc, None

    def buildvalidfun(self, model, batsize):
        self.tt.tick("validation - autobuilding")
        inps, outps = self.autobuild_model(model, *self.traindata, _trainmode=False, _batsize=batsize)
        assert(len(outps) == 1)
        outp = outps[0]
        self.tt.tock("validation - autobuilt")
        self.tt.tick("compiling validation function")
        metrics, newinp = self.buildlosses(outp, self.validators)
        inputs = newinp if newinp is not None else inps
        ret = None
        if len(metrics) > 0:
            ret = theano.function(inputs=[x.d for x in inputs] + [self.goldvar],
                                  outputs=metrics,
                                  mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=True)
                                  )
        else:
            self.tt.msg("NO VALIDATION METRICS DEFINED, RETURNS NONE")
        self.tt.tock("validation function compiled")
        return ret
    #endregion

    #region ################## TRAINING STRATEGIES ############
    def _train_full(self): # on all data, no validation
        df = DataFeeder(*(self.traindata + [self.traingold])).numbats(self.numbats)
        trainf = self.buildtrainfun(self.model, df.batsize)
        err, _ = self.trainloop(
                trainf=self.getbatchloop(trainf, df, phase="TRAIN"))
        return err, None, None, None

    def _train_validdata(self):
        df = DataFeeder(*(self.traindata + [self.traingold])).numbats(self.numbats)
        vdf = DataFeeder(*(self.validdata + [self.validgold]))
        vdf.batsize = df.batsize
        trainf = self.buildtrainfun(self.model, df.batsize)
        validf = self.buildvalidfun(self.model, vdf.batsize)
        #embed()
        #dfvalid = df.osplit(split=self.validsplits, random=self.validrandom)
        err, verr = self.trainloop(
                trainf=self.getbatchloop(trainf, df, phase="TRAIN"),
                validf=self.getbatchloop(validf, vdf, phase="VALID"))
        return err, verr, None, None

    def _train_split(self):
        df = DataFeeder(*(self.traindata + [self.traingold]))
        dftrain, dfvalid = df.split(self.validsplits, self.validrandom)
        dftrain.numbats(self.numbats)
        dfvalid.batsize = dftrain.batsize
        trainf = self.buildtrainfun(self.model, dftrain.batsize)
        validf = self.buildvalidfun(self.model, dfvalid.batsize)
        err, verr = self.trainloop(
                trainf=self.getbatchloop(trainf, dftrain, phase="TRAIN"),
                validf=self.getbatchloop(validf, dfvalid, phase="VALID"))
        return err, verr, None, None

    def _train_cross_valid(self):
        df = DataFeeder(*(self.traindata + [self.traingold]))
        splitter = SplitIdxIterator(df.size, split=self.validsplits, random=self.validrandom, folds=self.validsplits)
        err = []
        verr = []
        c = 0
        for splitidxs in splitter:
            tf, vf = df.isplit(splitidxs)
            tf.numbats(self.numbats)
            vf.batsize = tf.batsize
            trainf = self.buildtrainfun(self.model, tf.batsize)
            validf = self.buildvalidfun(self.model, vf.batsize)
            serr, sverr = self.trainloop(
                trainf=self.getbatchloop(trainf, tf, phase="TRAIN"),
                validf=self.getbatchloop(validf, vf, phase="VALID"))
            err.append(serr)
            verr.append(sverr)
            self.resetmodel(self.model)
        err = np.asarray(err)
        avgerr = np.mean(err, axis=0)
        verr = np.asarray(verr)
        avgverr = np.mean(verr, axis=0)
        self.tt.tock("done")
        return avgerr, avgverr, err, verr
    #endregion

    def resetmodel(self, model):    # TODO: very hacky
        _, outs = model.autobuild(*self.traindata)
        params = outs[0].allparams
        for param in params:
            param.reset()

    #region ############# TRAINING LOOPS ##################
    def trainloop(self, trainf, validf=None):
        self.tt.tick("training")
        err = []
        verr = []
        stop = self.maxiter == 0
        self.currentiter = 1
        evalinter = self._validinter
        evalcount = evalinter
        tt = TT("iter")
        prevverre = [float("inf")] * len(self.validators)
        while not stop:
            tt.tick("%d/%d" % (self.currentiter, int(self.maxiter)))
            erre = trainf()
            if self.currentiter == self.maxiter:
                stop = True
            self.currentiter += 1
            err.append(erre)
            #print "done training"
            verre = prevverre
            if validf is not None and self.currentiter % evalinter == 0: # validate and print
                verre = validf()
                prevverre = verre
                verr.append(verre)
                ttmsg = "training error: %s \t validation error: %s" \
                       % ("%.4f" % erre[0],
                          " - ".join(map(lambda x: "%.4f" % x, verre)))
            else:
                ttmsg = "training error: %s" % " - ".join(map(lambda x: "%.4f" % x, erre))
            # retaining the best
            if self.besttaker is not None:
                modelscore = self.besttaker(([erre]+verre+[self.currentiter]))
                if modelscore < self.bestmodel[1]:
                    #tt.tock("freezing best with score %.3f (prev: %.3f)" % (modelscore, self.bestmodel[1]), prefix="-").tick()
                    self.bestmodel = (self.model.freeze(), modelscore)
            tt.tock(ttmsg + "\t", prefix="-")
            self._update_lr(self.currentiter, self.maxiter, err, verr)
            evalcount += 1
            #embed()
            if self._autosave:
                self.save()
        self.tt.tock("trained").tick()
        return err, verr

    def getbatchloop(self, trainf, datafeeder, verbose=True, phase="TEST"):
        '''
        returns the batch loop, loaded with the provided trainf training function and samplegen sample generator
        '''
        sampletransf = self._transformsamples
        this = self

        def batchloop():
            c = 0
            prevperc = -1.
            terr = [0.0]
            numdigs = 2
            tt = TT("iter progress", verbose=verbose)
            tt.tick()
            datafeeder.reset()
            while datafeeder.hasnextbatch():
                perc = round(c*100.*(10**numdigs)/datafeeder.getnumbats())/(10**numdigs)
                if perc > prevperc:
                    s = ("%."+str(numdigs)+"f%% \t error: %.3f") % (perc, terr[0])
                    tt.live(s)
                    prevperc = perc
                sampleinps = datafeeder.nextbatch()
                #embed()
                sampleinps = sampletransf(*sampleinps, phase=phase)
                try:
                    eterr = trainf(*sampleinps)
                    if len(terr) != len(eterr) and terr.count(0.0) == len(terr):
                        terr = [0.0]*len(eterr)
                except Exception, e:
                    raise e
                if self.average_err is True:
                    terr = [xterr*(1.0*(c)/(c+1)) + xeterr*(1.0/(c + 1)) for xterr, xeterr in zip(terr, eterr)]
                else:
                    terr = [xterr + xeterr for xterr, xeterr in zip(terr, eterr)]
                c += 1
            tt.stoplive()
            return terr
        return batchloop

    def _transformsamples(self, *s, **kw):
        phase = kw["phase"] if "phase" in kw else None
        if len(self._sampletransformers) == 0:
            return s
        else:
            for sampletransformer in self._sampletransformers:
                s = sampletransformer(*s, phase=phase)
            return s

    def sampletransform(self, *f):
        self._sampletransformers = f
        return self
    #endregion
    #endregion

    @property
    def autosave(self):
        self._autosave = True
        return self

    def autosavethis(self, block, p):
        self._autosave = True
        self._autosaveblock = block
        self._autosavepath = p
        return self

    def save(self, model=None, filepath=None):
        model = model if model is not None else \
            self.model if self._autosaveblock is None else \
                self._autosaveblock
        filepath = filepath if filepath is not None else self._autosavepath
        model.save(filepath=filepath)


class NSModelTrainer(ModelTrainer):
    """ Model trainer using negative sampling """
    def __init__(self, model, gold, nrate, nsamgen):
        super(NSModelTrainer, self).__init__(model, gold)
        self.ns_nrate = nrate
        self.ns_nsamgen = nsamgen

    def _transformsamples(self, *s, **kw):
        # phase in kw
        """ apply negative sampling function and neg sam rate """
        psams = s[:-1]
        acc = []
        for i in range(self.ns_nrate):
            nsams = self.ns_nsamgen(*psams)
            news = psams + nsams + (s[-1],)
            ret = []
            if len(acc) == 0:       # first one
                ret = news
            else:
                for x, y in zip(acc, news):
                    ret.append(np.concatenate([x, y], axis=0))
            acc = ret
        return acc

    def autobuild_model(self, model, *traindata, **kw):
        return model.autobuild(*(traindata + traindata))
