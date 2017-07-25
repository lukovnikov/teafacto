import sys, gc, inspect
from pympler import asizeof
from datetime import datetime as dt
from IPython import embed
from collections import OrderedDict

import numpy as np
import theano
from lasagne.objectives import *
from lasagne.regularization import l1, l2
from lasagne.updates import *
from theano import tensor as tensor
from theano.compile.nanguardmode import NanGuardMode

#from core import Input
from teafacto.core import asblock, Block
from teafacto.core.datafeed import DataFeeder, SplitIdxIterator
from teafacto.blocks.loss import *
from teafacto.util import ticktock as TT, issequence, handleSIGINT


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


class ProtoObjective(object):
    def __init__(self, aggmode="mean"):
        super(ProtoObjective, self).__init__()
        self.aggmode = aggmode
        self.agg_history = []
        self.current_agg_error = 0.
        self.current_agg_norma = 0.

    def get_agg_error(self):
        if self.aggmode == "mean":
            if self.current_agg_norma == 0.:
                return -0.
            return self.current_agg_error / self.current_agg_norma
        return self.current_agg_error

    def update_agg(self, err, numex):
        self.current_agg_norma += numex
        err = err * numex if self.aggmode == "mean" else err
        self.current_agg_error += err

    def _reset(self):  # full reset
        self.reset_agg()
        self.agg_history = []

    def get_agg_error_history(self):
        return self.agg_history

    def reset_agg(self):
        self.current_agg_error = 0.
        self.current_agg_norma = 0.

    def push_agg_to_history(self):
        self.agg_history.append(self.get_agg_error())


class Objective(ProtoObjective):
    def __init__(self, loss, aggmode="mean"):
        super(Objective, self).__init__(aggmode=aggmode)
        self.obj = loss


class ExternalObjective(Objective):
    def __init__(self):
        super(ExternalObjective, self).__init__(None)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("override this")


class ExternalFunctionObjective(ExternalObjective):
    def __init__(self, f):
        super(ExternalFunctionObjective, self).__init__()
        self.f = f

    def __call__(self, *args):
        return self.f(*args)


class LossesApplied(Block):
    def __init__(self, model, losses, **kw):
        super(LossesApplied, self).__init__(**kw)
        self.model = model
        self.losses = losses

    def apply(self, *data, **kw):
        gold = data[-1]
        inpd = data[:-1]
        modelout = self.model(*inpd, **kw)
        acc = []
        for loss in self.losses:
            if hasattr(loss, "apply") and \
                    "mask" in inspect.getargspec(loss.apply)[0]:
                mask = modelout.mask.d if modelout.mask is not None else None
                acc.append(loss(modelout, gold, mask=mask))
            else:
                assert(modelout.mask is None)
                acc.append(loss(modelout, gold))
        return tuple(acc)


class ModelTrainer(object):
    def __init__(self, model):
        self.model = model
        self.validsetmode= False
        self.average_err = True # TODO: do we still need this?
        self._autosave = False
        self._autosavepath = None
        self._autosaveblock = None
        # training settings
        self.numbats = None
        self.learning_rate = None
        self.dynamic_lr = None
        self.training_objectives = []
        self._model_gives_train_losses = False
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
        self.validdata = None
        self.validgold = None
        self.validation = None
        self.validation_objectives = []
        self._model_gives_valid_losses = False
        self.external_validators = []
        self.tt = TT("FluentTrainer")
        # taking best
        self.besttaker = None
        self.bestmodel = None
        self.savebest = None
        self.smallerbetter = True
        # writing
        self._writeresultspath = None
        # early stopping
        self._earlystop = False
        self._earlystop_criterium = None
        self._earlystop_selector = None
        self._earlystop_select_history = None


    #region ====================== settings =============================
    #region ################### GENERAL ###################
    def numbats(self, s):
        self.numbats = s
        return self
    #endregion

    #region ################### LOSSES ##########################

    def _set_objective(self, obj):
        if self.validsetmode is False:
            self.training_objectives.append(obj)
        else:
            self.validation_objectives.append(obj)

    def model_loss(self, aggmode="mean"):   # custom loss
        self._set_objective(ProtoObjective(aggmode=aggmode))
        if self.validsetmode is False:
            self._model_gives_train_losses = True
        else:
            self._model_gives_valid_losses = True
        return self

    def model_losses(self, number=1, aggmode="mean", aggmodes=None):
        # mtl loss, custom loss
        if aggmodes is None:
            aggmodes = [aggmode] * number
        for aggm in aggmodes:
            self.model_loss(aggm)
        return self

    def loss(self, loss, mode="mean"):      # custom loss block
        self._set_objective(Objective(loss, aggmode=mode))
        return self

    def linear_objective(self, mode="mean"):
        self._set_objective(Objective(LinearLoss(), aggmode=mode))
        return self

    def cross_entropy(self, mode="mean", cemode="sum"):
        ce = CrossEntropy(mode=cemode)
        self._set_objective(Objective(ce, aggmode=mode))
        return self

    def seq_cross_entropy(self, mode="mean", cemode="sum"): # probs (batsize, seqlen, vocsize) + gold: (batsize, seqlen) ==> sum of neg log-probs of correct seq
        ce = CrossEntropy(mode=cemode)
        self._set_objective(Objective(ce, aggmode=mode))
        return self

    def perplexity(self, mode="mean"):
        self._set_objective(Objective(Perplexity(), aggmode=mode))
        return self

    def bitspersym(self, mode="mean"):
        self._set_objective(Objective(BitsPerSym(), aggmode=mode))
        return self

    def squared_error(self, mode="mean"):
        self._set_objective(Objective(SquaredError(), aggmode=mode))
        return self

    def squared_loss(self, mode="mean"):
        self._set_objective(Objective(SquaredLoss(), aggmode=mode))        # [-1, +1](batsize, )
        return self

    def binary_cross_entropy(self, mode="mean"): # theano binary cross entropy (through lasagne), probs: (batsize,) float, gold: (batsize,) float
        self._set_objective(Objective(BinaryCrossEntropy(), aggmode=mode))
        return self

    def bin_accuracy(self, sep=0, mode="mean"):
        self._set_objective(Objective(BinaryAccuracy(sep=sep), aggmode=mode))
        return self

    def accuracy(self, top_k=1, mode="mean"):
        self._set_objective(Objective(Accuracy(top_k=top_k), aggmode=mode))
        return self

    def seq_accuracy(self, mode="mean"): # sequences must be exactly the same
        self._set_objective(Objective(SeqAccuracy(), aggmode=mode))
        return self

    def hinge_loss(self, margin=1., labelbin=True, mode="mean"): # gold must be -1 or 1 if labelbin if False, otherwise 0 or 1
        self._set_objective(Objective(HingeLoss(margin=margin, labelbin=labelbin), aggmode=mode))
        return self

    def log_loss(self, mode="mean"):
        self._set_objective(Objective(LogLoss(), aggmode=mode))
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
        self.validsetmode = True
        self.validdata = data
        #if gold is None:
        #    gold = np.ones((data[0].shape[0],), dtype="float32")
        #    self.linear_objective()
        self.validgold = gold
        self.validsplits = splits
        self.validrandom = random
        return self

    def cross_validate(self, splits=5, random=False):
        self.trainstrategy = self._train_cross_valid
        self.validsplits = splits
        self.validrandom = random
        self.validsetmode = True
        return self

    def extvalid(self, evaluator):
        if not isinstance(evaluator, ExternalObjective):
            evaluator = ExternalFunctionObjective(evaluator)
        self.validation_objectives.append(evaluator)
        return self

    #endregion

    #region ######################### SELECTING THE BEST ######################
    def takebest(self, f=None, save=False, smallerbetter=True, no=False):
        if no:              # disables besttaker
            print "no best taking"
            return self
        else:
            print "best taking"
        if f is None:
            f = lambda x: x[1]   # pick the model with the best first validation score
        self.besttaker = f
        self.bestmodel = (None, float("inf"))
        self.savebest = save
        self.smallerbetter = smallerbetter
        return self

    def earlystop(self, select=None, stopcrit=None):
        if select is None:
            select = lambda x: x[1]
        if stopcrit is None:
            stopcrit = lambda h: h[-2] < h[-1] if len(h) >= 2 else False
        elif isinstance(stopcrit, int):
            stopcrit_window = stopcrit

            def windowstopcrit(h):
                window = stopcrit_window
                minpos = 0
                minval = np.infty
                for i, he in enumerate(h):
                    if he < minval:
                        minval = he
                        minpos = i
                ret = minpos < len(h) - window
                return ret

            stopcrit = windowstopcrit
        self._earlystop_criterium = stopcrit
        self._earlystop_selector = select
        self._earlystop_select_history = []
        self._earlystop = True
        return self

    def earlystop_eval(self, scores):
        selected = self._earlystop_selector(scores)
        self._earlystop_select_history.append(selected)
        ret = self._earlystop_criterium(self._earlystop_select_history)
        return ret
    #endregion
    #endregion

    #region ====================== execution ============================

    #region ######################### ACTUAL TRAINING #########################
    def traincheck(self):
        assert(self.optimizer is not None)
        assert(len(self.training_objectives) > 0)
        assert(self.traindata is not None)

    def train(self, numbats, epochs, returnerrors=False, _skiptrain=False):
        #handleSIGINT()
        self.traincheck()
        self.numbats = numbats
        self.maxiter = epochs
        errors = self.trainstrategy(_skiptrain=_skiptrain)       # trains according to chosen training strategy, returns errors
        if self.besttaker is not None and self.savebest is not True:      # unfreezes best model if best choosing was chosen
            self.model = self.model.__class__.unfreeze(self.bestmodel[0])
            self.tt.tock("unfroze best model (%.3f) - " % self.bestmodel[1]).tick()
        ret = self.model
        if returnerrors:
            ret = (ret,) + errors
        return ret

    def train_lambda(self, numbats, batprop=1):     # TODO: _skiptrain???
        self.traincheck()
        self.numbats = numbats
        if self.trainstrategy == self._train_cross_valid:
            raise NotImplementedError("CV training not supported with lambda training yet")
        trainf, validf, traind, validd = self.trainstrategy(_lambda=True)
        return ProtoTrainer(trainf, validf, traind, validd, batprop, self)

    def get_learning_rate(self):
        return self.learning_rate

    def autobuild_model(self, model, *data, **kw):
        return model.autobuild(*data, **kw)

    def apply_losses(self, model, losses):
        lossblock = LossesApplied(model, losses)
        return lossblock

    def buildtrainfun(self, model, batsize):
        self.tt.tick("training - autobuilding")
        with model.trainmode(True):
            if self._model_gives_train_losses:
                lossblock = model
                assert(self.traingold is None)
            else:
                lossblock = self.apply_losses(model, [o.obj for o in self.training_objectives])
            concatdata = self.traindata + [self.traingold] if self.traingold is not None else self.traindata
            inps, lossouts = self.autobuild_model(lossblock, *concatdata, _trainmode=True, _batsize=batsize)
            if not issequence(lossouts):
                lossouts = [lossouts]
            primarylossout = lossouts[0]
            self.tt.tock("training - autobuilt")
            self.tt.tick("compiling training function")
            params = primarylossout.allparams
            nonparams = [p for p in params if not p.lrmul > 0]
            params = [p for p in params if p.lrmul > 0]
            scanupdates = primarylossout.allupdates
            inputs = inps
            losses, newinp = self.buildlosses(lossouts, self.training_objectives)
            primaryloss = losses[0]
            if newinp is not None:
                inputs = newinp
            if self.regularizer is not None:
                reg = self.regularizer(params)
                cost = primaryloss + reg
            else:
                cost = primaryloss
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
            totalgradnorm = sum(tensor.sum(grad ** 2) for grad in grads)
            originalgrads = grads
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

            finputs = [x.d for x in inputs]
            allupdates = updates + scanupdates.items()
            trainf = theano.function(
                inputs=finputs,
                outputs=[cost]+losses[1:]+[totalgradnorm]+originalgrads,
                updates=allupdates,
                on_unused_input="warn",
                #mode=theano.compile.MonitorMode(post_func=theano.compile.monitormode.detect_nan),
                #mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False)
                # TODO: enabling NanGuard with Dropout doesn't work --> see Theano.git/issues/4823
            )
            self.tt.tock("training function compiled")
        return trainf

    def buildlosses(self, losses, objs):    # every objective produces one number
        acc = []
        for loss, obj in zip(losses, objs):
            objagg = aggregate(loss.d, mode=obj.aggmode)
            acc.append(objagg)
        return acc, None

    def getvalidfun(self, model, batsize):
        symbolic_validfun = self.buildvalidfun(model, batsize)
        valids = []
        i = 0
        for validator in self.validation_objectives:
            if isinstance(validator, ExternalObjective):
                valids.append(validator)
            else:
                valids.append(i)
                i += 1
        if i == len(self.validation_objectives):
            return symbolic_validfun
        else:
            def validfun(*sampleinps):
                #embed()
                ret = []
                symret = []
                if symbolic_validfun is not None:
                    svf = symbolic_validfun(*sampleinps)
                    if not issequence(svf):
                        svf = [svf]
                    symret = svf
                for validf in valids:
                    if not isinstance(validf, ExternalObjective):
                        ret.append(symret[validf])
                    else:
                        a = validf(*sampleinps)
                        ret.append(a)
                return ret
            return validfun

    def buildvalidfun(self, model, batsize):
        self.tt.tick("validation - autobuilding")
        validators = filter(lambda x: not isinstance(x, ExternalObjective), self.validation_objectives)
        if self._model_gives_valid_losses:
            lossblock = model
            assert(self.validgold is None)
        else:
            lossblock = self.apply_losses(model, [o.obj for o in validators])
        if self.validdata is None:
            concatdata = self.traindata + [self.traingold] if self.traingold is not None else self.traindata
        else:
            concatdata = self.validdata + [self.validgold] if self.validgold is not None else self.validdata
        inps, lossouts = self.autobuild_model(lossblock, *concatdata, _trainmode=False, _batsize=batsize)
        if not issequence(lossouts):
            lossouts = [lossouts]
        self.tt.tock("validation - autobuilt")
        self.tt.tick("compiling validation function")

        losses, newinp = self.buildlosses(lossouts, validators)
        inputs = newinp if newinp is not None else inps
        ret = None
        if len(losses) > 0:
            ret = theano.function(inputs=[x.d for x in inputs],
                                  outputs=losses,
                                  updates=lossouts[0].allupdates,
                                  on_unused_input="warn",
                                  #mode=NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=True)  # same issue as training
                                  )
        else:
            self.tt.msg("NO VALIDATION METRICS DEFINED, RETURNS NONE")
        self.tt.tock("validation function compiled")
        return ret
    #endregion

    def _concat_data(self, *data):
        out = []
        for x in data:
            if isinstance(x, list):
                out += x
            elif x is None:
                pass
            else:
                out += [x]
        return out


    #region ################## TRAINING STRATEGIES ############
    def _train_full(self, _lambda=False, _skiptrain=False): # on all data, no validation
        df = DataFeeder(*self._concat_data(self.traindata, self.traingold)).numbats(self.numbats)
        trainf = self.buildtrainfun(self.model, df.batsize)
        if _lambda:
            return trainf, None, df, None
        else:
            err, _ = self.trainloop(
                    trainf=self.getbatchloop(trainf, df, phase="TRAIN"),
                    _skiptrain=_skiptrain)
            return err, None, None, None

    def _train_validdata(self, _lambda=False, _skiptrain=False):
        df = DataFeeder(*self._concat_data(self.traindata, self.traingold)).numbats(self.numbats)
        vdf = DataFeeder(*self._concat_data(self.validdata, self.validgold), random=False)
        vdf.batsize = df.batsize
        trainf = self.buildtrainfun(self.model, df.batsize)
        validf = self.getvalidfun(self.model, vdf.batsize)
        #embed()
        #dfvalid = df.osplit(split=self.validsplits, random=self.validrandom)
        if _lambda:
            return trainf, validf, df, vdf
        else:
            err, verr = self.trainloop(
                    trainf=self.getbatchloop(trainf, df, phase="TRAIN"),
                    validf=self.getbatchloop(validf, vdf, phase="VALID"),
                    _skiptrain=_skiptrain)
            return err, verr, None, None

    def _train_split(self, _lambda=False, _skiptrain=False):
        df = DataFeeder(*self._concat_data(self.traindata, self.traingold))
        dftrain, dfvalid = df.split(self.validsplits, self.validrandom, df_randoms=(True, False))
        dftrain.numbats(self.numbats)
        dfvalid.batsize = dftrain.batsize
        #embed()
        trainf = self.buildtrainfun(self.model, dftrain.batsize)
        validf = self.getvalidfun(self.model, dfvalid.batsize)
        if _lambda:
            return trainf, validf, dftrain, dfvalid
        else:
            err, verr = self.trainloop(
                    trainf=self.getbatchloop(trainf, dftrain, phase="TRAIN"),
                    validf=self.getbatchloop(validf, dfvalid, phase="VALID"),
                    _skiptrain=_skiptrain)
            return err, verr, None, None

    def _train_cross_valid(self, _skiptrain=False):
        df = DataFeeder(*self._concat_data(self.traindata, self.traingold))
        splitter = SplitIdxIterator(df.size, split=self.validsplits, random=self.validrandom, folds=self.validsplits)
        err = []
        verr = []
        c = 0
        for splitidxs in splitter:
            tf, vf = df.isplit(splitidxs, df_randoms=(True, False))
            tf.numbats(self.numbats)
            vf.batsize = tf.batsize
            trainf = self.buildtrainfun(self.model, tf.batsize)
            validf = self.getvalidfun(self.model, vf.batsize)
            serr, sverr = self.trainloop(
                trainf=self.getbatchloop(trainf, tf, phase="TRAIN"),
                validf=self.getbatchloop(validf, vf, phase="VALID"),
                _skiptrain=_skiptrain)
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
    def trainloop(self, trainf, validf=None, _skiptrain=False):
        self.tt.tick("training")
        stop = self.maxiter == 0
        self.currentiter = 1
        evalinter = self._validinter
        evalcount = evalinter
        tt = TT("iter")

        # resetting objectives
        tt.msg("resetting objectives")
        for obj in self.training_objectives:
            obj._reset()
        for obj in self.validation_objectives:
            obj._reset()

        writeresf = None
        if self._writeresultspath is not None:
            writeresf = open(self._writeresultspath, "w", 1)

        while not stop:
            tt.tick("%d/%d" % (self.currentiter, int(self.maxiter)))
            if _skiptrain:
                tt.msg("skipping training")
            else:
                trainf()
            if self.currentiter == self.maxiter:
                stop = True
            self.currentiter += 1
            restowrite = ""
            if self._autosave:
                self.save()
            epoch_train_errors = [obj.get_agg_error() for obj in self.training_objectives]
            epoch_valid_errors = []
            if validf is not None and self.currentiter % evalinter == 0: # validate and print
                validf()
                epoch_valid_errors = [validator.get_agg_error() for validator in self.validation_objectives]
                ttmsg = "training error: %s \t validation error: %s" \
                       % (" - ".join(map(lambda x: "%.4f" % x, epoch_train_errors)),
                          " - ".join(map(lambda x: "%.4f" % x, epoch_valid_errors)))
                restowrite = "\t".join(map(str, [epoch_train_errors] + epoch_valid_errors))
            else:
                ttmsg = "training error: %s" % " - ".join(map(lambda x: "%.4f" % x, epoch_train_errors))
                restowrite = str(epoch_train_errors)
            if writeresf is not None:
                writeresf.write("{}\t{}\n".format(self.currentiter - 1, restowrite))
            # retaining the best
            if self.besttaker is not None:
                modelscore = self.besttaker([epoch_train_errors[0]] + epoch_valid_errors + [self.currentiter])
                smallerbetter = 1 if self.smallerbetter else -1
                if smallerbetter * modelscore < smallerbetter * self.bestmodel[1]:
                    if self.savebest:
                        self.save(suffix=".best")
                        self.bestmodel = (None, modelscore)
                    else:
                        #tt.tock("freezing best with score %.3f (prev: %.3f)" % (modelscore, self.bestmodel[1]), prefix="-").tick()
                        self.bestmodel = (self.save(freeze=True, filepath=False), modelscore)
            if self._earlystop:
                doearlystop = self.earlystop_eval([epoch_train_errors[0]] + epoch_valid_errors + [self.currentiter])
                if doearlystop:
                    tt.msg("stopping early")
                stop = stop or doearlystop
            tt.tock(ttmsg + "\t", prefix="-")
            self._update_lr(self.currentiter,
                            self.maxiter,
                            [obj.get_agg_error_history() for obj in self.training_objectives],
                            [validator.get_agg_error_history() for validator in self.validation_objectives])
            evalcount += 1
            if writeresf is not None:
                writeresf.close()
        self.tt.tock("trained").tick()
        return np.asarray([obj.get_agg_error_history() for obj in self.training_objectives]), \
               np.asarray([validator.get_agg_error_history() for validator in self.validation_objectives]).T

    def getbatchloop(self, f, datafeeder, verbose=True, phase="TEST"):
        '''
        returns the batch loop, loaded with the provided trainf training function and samplegen sample generator
        '''
        sampletransf = self._transformsamples
        objectives = []
        if phase == "TRAIN":
            objectives = self.training_objectives
        elif phase == "VALID":
            objectives = self.validation_objectives
        numdigs = 2
        this = self

        def batchloop():
            thisthis = this
            c = 0
            number_examples = 0
            prevperc = -1.
            tt = TT("iter progress", verbose=verbose)
            tt.tick()
            datafeeder.reset()
            for obj in objectives:
                obj.reset_agg()
            tgn = 0.
            while datafeeder.hasnextbatch():
                perc = round(c*100.*(10**numdigs)/datafeeder.getnumbats())/(10**numdigs)
                if perc > prevperc:     # print errors
                    current_agg_errors = [obj.get_agg_error() for obj in objectives]
                    errorstr = " - ".join(["{:10.10s}".format("{:.10g}".format(current_agg_error)) for current_agg_error in current_agg_errors])
                    s = ("{:4.2f}%    errors: {}").format(perc, errorstr)
                    s += "    TGN: {:10.10s}    ".format("{:.10g}".format(tgn))
                    tt.live(s)
                    prevperc = perc
                sampleinps, batsize = datafeeder.nextbatch(withbatchsize=True)
                number_examples += batsize
                #embed()
                sampleinps = sampletransf(*sampleinps, phase=phase)
                train_f_out = f(*sampleinps)
                errors_current = train_f_out[:len(objectives)]
                other_outs = train_f_out[len(objectives):]
                if len(other_outs) > 0:
                    tgn = other_outs[0]
                    tgn = float(tgn)
                    tgn = np.sqrt(tgn)
                    if np.isnan(tgn):
                        print "NAN totalnorm"
                        embed()
                for current_error, objective in zip(errors_current, objectives):
                    objective.update_agg(current_error, batsize)
                c += 1
            tt.stoplive()
            # END OF EPOCH
            for obj in objectives:
                obj.push_agg_to_history()
            errors_agg = [obj.get_agg_error() for obj in objectives]
            return errors_agg
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
        self._autosavepath = self._gen_autosave_path()
        return self

    def autosaveit(self, p=None):
        self._autosave = True
        self._autosavepath = p if p is not None else self._gen_autosave_path()
        return self

    def _gen_autosave_path(self):
        import os, random
        path = None
        badpath = True
        while badpath:
            randid = "".join([str(random.sample(range(10), 1)[0]) for x in range(5)])
            path = "{}.model".format(randid)
            if os.path.exists(path):
                badpath = True
            else:
                badpath = False
        print "autosave path: {}".format(path)
        return path

    def autosavethis(self, block, p):
        self._autosave = True
        self._autosaveblock = block
        self._autosavepath = p
        return self

    def writeresultstofile(self, p):
        self._writeresultspath = p
        return self

    def save(self, model=None, filepath=None, suffix="", freeze=False):
        filepath = False if freeze else filepath
        model = model if model is not None else \
            self.model if self._autosaveblock is None else \
                self._autosaveblock
        if filepath is not False:
            filepath = filepath if filepath is not None else self._autosavepath
            model.save(filepath=filepath + suffix)
        else:
            return model.freeze()


class ProtoTrainer(object):
    def __init__(self, trainf, validf, traind, validd, batprop, trainer):
        self.trainf, self.validf, self.traind, self.validd = trainf, validf, traind, validd
        self.batprop = batprop
        self.original = trainer

    def interleave(self, *otherprototrainers):
        return InterleavedTrainer(self, *otherprototrainers)


class InterleavedTrainer(object):
    def __init__(self, maintrainer, *othertrainers):
        self.spts = [maintrainer] + list(othertrainers)
        self.tt = TT("InterleavedTrainer")
        self.currentiter = 0
        self._validinter = self.spts[0].original._validinter

    def train(self, epochs=10, verbose=True):
        self.maxiter = epochs
        tf = self.getbatchloop([spt.trainf for spt in self.spts],
                               [spt.traind for spt in self.spts],
                               verbose=verbose, phase="TRAIN")
        subvfs = []
        for spt in self.spts:
            if spt.validf is not None and spt.validd is not None:
                subvf = spt.original.getbatchloop(spt.validf, spt.validd, verbose=verbose, phase="TEST")
                subvfs.append(subvf)
            else:
                subvfs.append(None)
        def vf():
            return [subvf() if subvf is not None else None for subvf in subvfs]
        return self.trainloop(tf, vf)

    # region ############# TRAINING LOOPS ##################
    def trainloop(self, tf, vf):
        self.tt.tick("training")
        stop = self.maxiter == 0
        self.currentiter = 1
        evalinter = self._validinter
        evalcount = evalinter
        tt = TT("iter")
        err = []
        verr = []
        prevverre = [[float("inf")] * len(subt.original.validators)
                     for subt in self.spts]
        while not stop:
            tt.tick("%d/%d" % (self.currentiter, int(self.maxiter)))
            erre = tf()
            if self.currentiter == self.maxiter:
                stop = True
            self.currentiter += 1
            err.append(erre)
            # print "done training"
            verre = prevverre
            if self.currentiter % evalinter == 0:  # validate and print
                verre = vf()
                prevverre = verre
                verr.append(verre)
                #embed()     # TODO
            # retaining the best of main trainer
            if self.spts[0].original.besttaker is not None:
                modelscore = self.spts[0].original.besttaker(([erre[0]] + verre[0] + [self.currentiter]))
                if modelscore < self.spts[0].original.bestmodel[1]:
                    # tt.tock("freezing best with score %.3f (prev: %.3f)" % (modelscore, self.bestmodel[1]), prefix="-").tick()
                    self.spts[0].original.bestmodel = (self.spts[0].original.model.freeze(), modelscore)

            ttlines = []
            for i in range(len(erre)):
                if verre[i] is not None:
                    ttlines.append("\t%s:\ttraining error: %s \t validation error: %s" \
                            % (i+1, "%.4f" % erre[i][0],
                               " - ".join(map(lambda x: "%.4f" % x, verre[i]))))
                else:
                    ttlines.append("\t%s:\ttraining error: %s"
                            % (i+1, " - ".join(map(lambda x: "%.4f" % x, erre[i]))))
            tt.tock("\n".join(ttlines) + "\n", prefix="-")
            for i, subt in enumerate(self.spts):
                subt.original._update_lr(self.currentiter, self.maxiter,
                                         [errx[i] for errx in err],
                                         [verrx[i] for verrx in verr])
            evalcount += 1
            # embed()
            for subt in self.spts:
                if subt.original._autosave:
                    subt.original.save()
        self.tt.tock("trained").tick()
        return err, verr

    def getbatchloop(self, trainfs, datafeeders, verbose=True, phase="TEST"):
        '''
        returns the batch loop, loaded with the provided trainf training function and samplegen sample generator
        '''
        sampletransfs = [spt.original._transformsamples for spt in self.spts]
        this = self

        def batchloop():
            thisthis = this
            c = 0
            prevperc = -1.
            terrs = [[0.0] if tf is not None else None for tf in trainfs]
            numdigs = 2
            tt = TT("iter progress", verbose=verbose)
            tt.tick()
            for dataf in datafeeders:
                if dataf is not None:
                    dataf.reset()
            while datafeeders[0].hasnextbatch():
                perc = round(c * 100. * (10 ** numdigs) / datafeeders[0].getnumbats()) / (10 ** numdigs)
                if perc > prevperc:
                    s = ("%." + str(numdigs) + "f%% \t error: %s") \
                        % (perc, " - ".join(map(lambda x: "%.3f" % x[0], terrs)))
                    tt.live(s)
                    prevperc = perc
                for df in datafeeders:
                    if not df.hasnextbatch():
                        df.reset()
                sampleinps = [df.nextbatch() for df in datafeeders]
                # embed()
                sampleinps = [stf(*si, phase=phase) for (stf, si) in zip(sampletransfs, sampleinps)]
                try:
                    eterrs = [tf(*si) for (tf, si) in zip(trainfs, sampleinps)]
                    for i in range(len(terrs)):
                        if len(terrs[i]) != len(eterrs[i]) and terrs[i].count(0.0) == len(terrs[i]):
                            terrs[i] = [0.0] * len(eterrs[i])
                except Exception, e:
                    raise e
                for i, subt in enumerate(this.spts):
                    if subt.original.average_err is True:
                        terrs[i] = [xterr * (1.0 * (c) / (c + 1)) + xeterr * (1.0 / (c + 1))
                                for xterr, xeterr in zip(terrs[i], eterrs[i])]
                    else:
                        terrs[i] = [xterr + xeterr for xterr, xeterr in zip(terrs[i], eterrs[i])]
                c += 1
            tt.stoplive()
            return terrs

        return batchloop


class NSModelTrainer(ModelTrainer):
    """ Model trainer using negative sampling """
    def __init__(self, model, nrate, nsamgen, nrate_valid=None):
        super(NSModelTrainer, self).__init__(model)
        self.ns_nrate = nrate
        self.ns_nrate_valid = nrate if nrate_valid is None else nrate_valid
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
