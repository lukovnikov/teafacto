from teafacto.util import ticktock, argprun, tokenize, StringMatrix
from teafacto.procutil import *
from teafacto.blocks.seq.neurex import DGTN, DGTN_S, PWPointerLoss, KLPointerLoss, PointerRecall, PointerFscore, PointerPrecision
from teafacto.blocks.seq.encdec import EncDec
from teafacto.blocks.seq.rnn import SeqEncoder
from teafacto.blocks.seq.attention import Attention
from teafacto.core.base import Block, param, tensorops as T
from teafacto.blocks.activations import Softmax
from IPython import embed
import pickle, scipy.sparse as sp, numpy as np, re


def loadtensor(p="../../../data/got/tensor.dock"):
    dockobj = pickle.load(open(p))
    entrytensor = dockobj["entries"]
    entdic = dockobj["entdic"]
    reldic = dockobj["reldic"]
    numents = max(np.max(entrytensor[:, 1]), np.max(entrytensor[:, 2])) + 1
    numrels = np.max(entrytensor[:, 0]) + 1
    dense_tensor = np.zeros((numrels, numents, numents), dtype="int8")
    #print dense_tensor.shape
    #embed()
    for x, y, z in entrytensor:
        dense_tensor[x, y, z] = 1
    return GraphTensor(dense_tensor, entdic, reldic)

    embed()


class DictGetter():
    def __init__(self, dic):
        self._dic = dic

    def __getitem__(self, item):
        if item in self._dic:
            return self._dic[item]
        else:
            return None


class GraphTensor():
    def __init__(self, tensor, entdic, reldic):
        self._tensor = tensor
        self._entdic = entdic
        self._reldic = reldic
        self._red = {v: k for k, v in self._entdic.items()}
        self._rld = {v: k for k, v in self._reldic.items()}

    @property
    def tensor(self):
        return self._tensor     # stored tensor is dense

    @property
    def ed(self):
        return DictGetter(self._entdic)

    @property
    def rd(self):
        return DictGetter(self._reldic)

    @property
    def red(self):
        return DictGetter(self._red)

    @property
    def rrd(self):
        return DictGetter(self._red)


def loadquestions(graphtensor,
                  p="../../../data/got/generated_questions.tsv",
                  simple=False):
    answers = []
    numents = graphtensor.tensor.shape[1]
    sm = StringMatrix(freqcutoff=2, indicate_start_end=True)
    with open(p) as f:
        for line in f:
            qid, numrels, numbranch, ambi, question, startents, answerents\
                = map(lambda x: x.strip(), line.split("\t"))
            qid, numrels, numbranch, ambi = map(int, [qid, numrels, numbranch, ambi])
            # preprocess questions
            if simple and numrels != 1:
                continue
            else:
                sm.add(question)
                # answers
                answerentss = [graphtensor.ed[(answer if answer[0] == ":" else ":"+answer).strip()]
                               for answer in answerents.split(",:")]
                try:
                    answerents = np.asarray(answerentss).astype("int32")
                except TypeError, e:
                    print answer
                    print answerents
                answerpointer = np.zeros((1, numents,))
                answerpointer[0, answerents] = 1
                answers.append(answerpointer)
    sm.finalize()
    #embed()
    answers = np.concatenate(answers, axis=0)
    return sm, answers

def load_simple_questions(graphtensor,
                  p="../../../data/got/questions.simple.tsv",
                  simple=False):
    answers = []
    numents = graphtensor.tensor.shape[1]
    sm = StringMatrix(freqcutoff=2, indicate_start_end=True)
    tvt = []
    starts = []
    with open(p) as f:
        for line in f:
            tvti, numrels, numbranch, ambi, question, startents, answerents \
                = map(lambda x: x.strip(), line.split("\t"))
            tvti, numrels, numbranch, ambi = map(int, [tvti, numrels, numbranch, ambi])
            tvt.append(tvti)
            # preprocess questions
            sm.add(question)
            # answers
            answerentss = [graphtensor.ed[(answer if answer[0] == ":" else ":" + answer).strip()]
                           for answer in answerents.split(",:")]
            try:
                answerents = np.asarray(answerentss).astype("int32")
            except TypeError, e:
                print answer
                print answerents
            answerpointer = np.zeros((1, numents,), dtype="float32")
            answerpointer[0, answerents] = 1
            answers.append(answerpointer)
            # starts
            startentss = [graphtensor.ed[(startent if startent[0] == ":" else ":" + startent).strip()]
                           for startent in startents.split(",:")]
            try:
                startents = np.asarray(startentss).astype("int32")
            except TypeError, e:
                print startent
                print startents
            startpointer = np.zeros((1, numents,), dtype="float32")
            startpointer[0, startents] = 1
            starts.append(startpointer)
    sm.finalize()
    # embed()
    answers = np.concatenate(answers, axis=0)
    starts = np.concatenate(starts, axis=0)
    return sm, answers, tvt, starts


def loadentitylabels(graphtensor):
    sm = StringMatrix(indicate_start_end=False, freqcutoff=0)
    gold = []
    numents = graphtensor.tensor.shape[1]
    for id, idx in graphtensor._entdic.items():
        sm.add(id)
        ptr = np.zeros((1, numents), dtype="int32")
        ptr[0, idx] = 1
        gold.append(ptr)
    sm.finalize()
    gold = np.concatenate(gold, axis=0)
    return sm, gold


def run(lr=0.1,
        epochs=100,
        batsize=50,
        nsteps=7,
        innerdim=310,
        nlayers=2,
        wordembdim=64,
        encdim=100,
        nenclayers=2,
        dropout=0.1,
        inspectdata=False,
        testpred=False,
        trainfind=False,
        simple=False,
        withstart=False,
        actionoverride=False,
        smmode="sm",        # "sm" or "gumbel" or "maxhot"
        debug=False,
        loss="klp",         # "klp", "pwp", "bpwp"
        ):
    if debug:
        inspectdata = True
        simple = True
        withstart = True
        actionoverride = True
    if trainfind:
        run_trainfind(**locals())
    tt = ticktock("script")
    tt.tick("loading graph")
    graphtensor = loadtensor()
    tt.tock("graph loaded")
    tt.tick("loading questions")
    if simple:
        qsm, answers, tvt, startents = load_simple_questions(graphtensor)
        qmat = qsm.matrix
        tvt = np.asarray(tvt)
        trainmat = qmat[tvt==0, :]
        validmat = qmat[tvt==1, :]
        testmat = qmat[tvt==2, :]
        traingold = answers[tvt==0]
        validgold = answers[tvt==1]
        testgold = answers[tvt==2]
        trainstartents = startents[tvt==0]
        validstartents = startents[tvt==1]
        teststartents = startents[tvt==2]
        if withstart:
            traindata, validdata, testdata = [trainmat, trainstartents], [validmat, validstartents], [testmat, teststartents]
        else:
            traindata, validdata, testdata = [trainmat], [validmat], [testmat]
        tt.tock("{} questions loaded".format(len(qmat)))
    else:
        qsm, answers = loadquestions(graphtensor)
        qmat = qsm.matrix
        # split 80/10/10
        splita, splitb = int(round(len(qmat) * 0.8)), int(round(len(qmat) * 0.9))
        trainmat, validmat, testmat = qmat[:splita, :], qmat[splita:splitb, :], qmat[splitb:, :]
        traingold, validgold, testgold = answers[:splita, :], answers[splita:splitb, :], answers[splitb:, :]
        traindata, validdata, testdata = [trainmat], [validmat], [testmat]
        tt.tock("{} questions loaded".format(len(qmat)))
    if inspectdata:
        embed()

    if actionoverride:
        if simple:
            if not withstart:
                tt.msg("doing action override with find-hop template for simple questions")
                assert(nsteps >= 2)
                actionoverride = np.zeros((nsteps, DGTN_S.numacts), dtype="float32")
                actionoverride[0, 1] = 1.
                actionoverride[1, 2] = 1.
            else:
                tt.msg("doing action override with only a hop for simple questions")
                assert(nsteps >= 2)
                actionoverride = np.zeros((nsteps, DGTN_S.numacts), dtype="float32")
                actionoverride[0, 2] = 1.
                actionoverride[1, 0] = 1.
        else:
            raise Exception("don't know how to override non-simple")
    else:
        actionoverride = None


    # build model
    tt.tick("building model")
    dgtn = DGTN_S(reltensor=graphtensor.tensor, nsteps=nsteps,
                entembdim=200, actembdim=10, attentiondim=encdim,
                entitysummary=False, relationsummary=False,
                action_override=actionoverride,
                gumbel=smmode=="gumbel", maxhot=smmode=="maxhot")
    enc = SeqEncoder.fluent()\
        .vectorembedder(qsm.numwords, wordembdim, maskid=qsm.d("<MASK>"))\
        .addlayers([encdim]*nenclayers, dropout_in=dropout, zoneout=dropout)\
        .make().all_outputs()
    dec = EncDec(encoder=enc,
                 inconcat=True, outconcat=True, stateconcat=True, concatdecinp=False,
                 updatefirst=False,
                 inpemb=None, inpembdim=dgtn.get_indim(),
                 innerdim=[innerdim]*nlayers,
                 dropout_in=dropout,
                 zoneout=dropout,
                 attention=Attention())
    dgtn.set_core(dec)
    tt.tock("model built")

    dgtn._ret_actions = True
    dgtn._ret_entities = True
    dgtn._ret_relations = True
    predf = dgtn.predict
    testprediction, actions, entities, relations = predf(*[testdatamat[:5] for testdatamat in testdata])

    # test prediction
    if testpred:
        embed()

    # training
    numbats = (len(trainmat) // batsize) + 1

    tt.tick("training")
    dgtn._no_extra_ret()

    if loss=="klp":     trainloss = KLPointerLoss(softmaxnorm=False)
    elif loss=="pwp":   trainloss = PWPointerLoss()
    elif loss=="bpwp":  trainloss = PWPointerLoss(balanced=True)
    else:               raise Exception("unknown loss option")

    dgtn.train(traindata, traingold)\
        .adadelta(lr=lr).loss(trainloss).loss(PointerFscore()).grad_total_norm(5.)\
        .validate_on(validdata, validgold).loss(trainloss).loss(PointerFscore()).loss(PointerRecall()).loss(PointerPrecision())\
        .train(numbats, epochs)

    tt.tock("trained")
    embed()


class Vec2Ptr(Block):
    def __init__(self, enc, numents, **kw):
        super(Vec2Ptr, self).__init__(**kw)
        self.enc = enc
        self.W = param((enc.outdim, numents)).glorotuniform()

    def apply(self, x):
        vec = self.enc(x)   # (batsize, vecdim)
        scores = T.dot(vec, self.W)
        probs = Softmax()(scores)
        return probs


def run_trainfind(lr=0.1,
        epochs=100,
        batsize=50,
        nsteps=7,
        gradnorm=5,
        innerdim=310,
        nlayers=2,
        wordembdim=64,
        encdim=100,
        nenclayers=1,
        dropout=0.0,
        inspectdata=False,
        testpred=False,
        trainfind=False,
        dodummy=False,
        smmode="sm",            # "sm" or "maxhot" or "gumbel"
    ):
    tt = ticktock("script")
    tt.tick("loading graph")
    graphtensor = loadtensor()
    tt.tock("graph loaded")
    tt.tick("loading labels")
    lsm, gold = loadentitylabels(graphtensor)
    tt.tock("labels loaded")
    lmat = lsm.matrix
    if inspectdata:
        embed()

    # build model
    tt.tick("building model")
    enc = SeqEncoder.fluent() \
        .vectorembedder(lsm.numwords, wordembdim, maskid=lsm.d("<MASK>")) \
        .addlayers([encdim] * nenclayers, dropout_in=dropout, zoneout=dropout) \
        .make().all_outputs()
    if dodummy:
        m = Vec2Ptr(enc, len(graphtensor._entdic))
    else:
        m = DGTN_S(reltensor=graphtensor.tensor, nsteps=nsteps,
                   entembdim=200, actembdim=10,
                   attentiondim=encdim,
                   entitysummary=False, relationsummary=False,
                   gumbel=smmode=="gumbel", maxhot=smmode=="maxhot")
        dec = EncDec(encoder=enc,
                     inconcat=True, outconcat=True, stateconcat=True, concatdecinp=False,
                     updatefirst=False,
                     inpemb=None, inpembdim=m.get_indim(),
                     innerdim=[innerdim] * nlayers,
                     dropout_in=dropout,
                     zoneout=dropout,
                     attention=Attention(),
                     )
        m.set_core(dec)
    tt.tock("model built")

    # test prediction
    if testpred:
        tt.tick("doing test prediction")
        testprediction = m.predict(lmat[:5, :])
        tt.tock("test prediction done")
        embed()

    # training
    numbats = (len(lmat) // batsize) + 1

    tt.tick("training")

    m.train([lmat], gold) \
        .adadelta(lr=lr).loss(PWPointerLoss(balanced=True)).grad_total_norm(gradnorm) \
        .train(numbats, epochs)

    tt.tock("trained")


if __name__ == "__main__":
    argprun(run)