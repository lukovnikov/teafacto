from teafacto.util import ticktock, argprun, tokenize, StringMatrix, issequence
from teafacto.procutil import *
from teafacto.blocks.seq.neurex import DGTN, DGTN_S, PWPointerLoss, KLPointerLoss, PointerRecall, PointerFscore, PointerPrecision
from teafacto.blocks.seq.encdec import EncDec
from teafacto.blocks.seq.rnn import SeqEncoder
from teafacto.blocks.seq.attention import Attention
from teafacto.blocks.word.wordvec import Glove, WordEmb
from teafacto.core.base import Block, param, tensorops as T
from teafacto.blocks.activations import Softmax
from IPython import embed
import pickle, scipy.sparse as sp, numpy as np, re, random, unidecode


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
        return DictGetter(self._rld)


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


class DataLoader(object):
    def __init__(self, graphtensor, simplep="../../../data/got/questions.simple.tsv",
                 allp="../../../data/got/generated_questions.tsv",
                 freqcutoff=0, indicate_start_end=True, **kw):
        super(DataLoader, self).__init__(**kw)
        self.graphtensor = graphtensor
        self.numents = self.graphtensor.tensor.shape[1]
        self.simplep = simplep
        self.allp = allp
        self.qsm = StringMatrix(freqcutoff=freqcutoff, indicate_start_end=indicate_start_end)
        self.answerptrs = []
        self.startptrs = []
        self.tvx = []
        self._finalized = False
        self._src_dic = {"_default": 0}
        self._srcs = []

    def add_labels(self, usefortrain=True, useforvalid=False, usefortest=False, src="_default"):
        assert(not self._finalized)
        assert(usefortest or useforvalid or usefortrain)
        if src not in self._src_dic:    self._src_dic[src] = max(self._src_dic.values()) + 1
        for id, idx in self.graphtensor._entdic.items():
            id = id[1:]
            ptr = np.zeros((1, self.numents), dtype="float32")
            ptr[0, idx] = 1
            for use, v in {0: int(usefortrain), 1: int(useforvalid), 2: int(usefortest)}.items():
                if v:
                    self.qsm.add(id)
                    self.answerptrs.append(ptr)
                    self.startptrs.append(np.zeros_like(ptr))
                    self.tvx.append(use)
                    self._srcs.append(self._src_dic[src])

    def add_labels_with_random_contexts(self, freq=3, maxlen=10, nwords=5000, src="_default"):
        assert(not self._finalized)
        g = Glove(50, nwords)
        words = g.D.keys()
        if src not in self._src_dic:    self._src_dic[src] = max(self._src_dic.values())+1
        for id, idx in self.graphtensor._entdic.items():
            id = unidecode.unidecode(id[1:])
            ptr = np.zeros((1, self.numents), dtype="float32")
            ptr[0, idx] = 1
            for i in range(freq):
                # generate random sequence of words with id in it
                seqlen = random.choice(range(3, maxlen))
                wordseq = random.sample(words, seqlen)
                replacepos = random.choice(range(0, seqlen))
                wordseq[replacepos] = id
                q = " ".join(wordseq)
                for tvxi in [0, 1, 2]:
                    self.qsm.add(q)
                    self.answerptrs.append(ptr)
                    self.startptrs.append(np.zeros_like(ptr))
                    self.tvx.append(tvxi)
                    self._srcs.append(self._src_dic[src])

    def _add_to_src(self, src, _srcs=None):
        if src not in self._src_dic:
            self._src_dic[src] = max(self._src_dic.values()) + 1
        if _srcs is None:
            _srcs = self._srcs
        _srcs.append(self._src_dic[src])

    def add_simple_questions(self, hop_only=False, find_only=False, trainportion=1.0, src="_default"):
        assert(not self._finalized)
        if src not in self._src_dic:    self._src_dic[src] = max(self._src_dic.values()) + 1
        _tvx = []
        _questions = []
        _startptrs = []
        _answerptrs = []
        with open(self.simplep) as f:
            for line in f:
                tvxi, numrels, numbranch, ambi, question, startents, answerents \
                    = map(lambda x: x.strip(), line.split("\t"))
                tvxi, numrels, numbranch, ambi = map(int, [tvxi, numrels, numbranch, ambi])
                _tvx.append(tvxi)
                _questions.append(question)
                # answer pointer
                answerentss = [self.graphtensor.ed[(answer if answer[0] == ":" else ":" + answer).strip()]
                               for answer in answerents.split(",:")]
                answerents = np.asarray(answerentss).astype("int32")
                answerptr = np.zeros((1, self.numents), dtype="float32")
                answerptr[0, answerents] = 1
                # start pointer
                startentss = [self.graphtensor.ed[(startent if startent[0] == ":" else ":" + startent).strip()]
                              for startent in startents.split(",:")]
                assert(len(startentss) == 1)
                startent = int(startentss[0])
                startptr = np.zeros((1, self.numents), dtype="float32")
                startptr[0, startent] = 1
                # choose
                if hop_only:
                    _startptrs.append(startptr)
                    _answerptrs.append(answerptr)
                elif find_only:
                    _startptrs.append(np.zeros_like(startptr))
                    _answerptrs.append(startptr)
                else:
                    _startptrs.append(np.zeros_like(startptr))
                    _answerptrs.append(answerptr)
        number = _tvx.count(0)
        firstn = int(round(number * trainportion))
        i = 0
        for tvxi, question, startptr, answerptr \
                in zip(_tvx, _questions, _startptrs, _answerptrs):
            if tvxi == 0:
                if i > firstn:
                    continue
                else:
                    i += 1
            self.tvx.append(tvxi)
            self.qsm.add(question)
            self.startptrs.append(startptr)
            self.answerptrs.append(answerptr)
            self._srcs.append(self._src_dic[src])

    def add_chains(self, length=None, withstart=False, src="_default"):
        assert(not self._finalized)
        if src not in self._src_dic:    self._src_dic[src] = max(self._src_dic.values()) + 1
        with open(self.allp) as f:
            tvxic = 0
            for line in f:
                qid, numrels, numbranch, ambi, question, startents, answerents\
                    = map(lambda x: x.strip(), line.split("\t"))
                qid, numrels, numbranch, ambi = map(int, [qid, numrels, numbranch, ambi])
                # preprocess questions
                if numbranch == 0:
                    if length is None or length == numrels:
                        self.qsm.add(question)
                        # answers
                        answerentss = [self.graphtensor.ed[(answer if answer[0] == ":" else ":" + answer).strip()]
                                       for answer in answerents.split(",:")]
                        answerents = np.asarray(answerentss).astype("int32")
                        answerptr = np.zeros((1, self.numents), dtype="float32")
                        answerptr[0, answerents] = 1
                        self.answerptrs.append(answerptr)
                        # start pointer
                        if withstart:
                            startentss = [self.graphtensor.ed[(startent if startent[0] == ":" else ":" + startent).strip()]
                                          for startent in startents.split(",:")]
                            assert (len(startentss) == 1)
                            startent = int(startentss[0])
                            startptr = np.zeros((1, self.numents), dtype="float32")
                            startptr[0, startent] = 1
                        else:
                            startptr = np.zeros_like(answerptr)
                        self.startptrs.append(startptr)
                        self.tvx.append(0 if tvxic < 2 else 1 if tvxic == 2 else 2)
                        tvxic += 1
                        tvxic %= 4
                        self._srcs.append(self._src_dic[src])

    def finalize(self):
        self.qsm.finalize()
        self.tvx = np.asarray(self.tvx)
        self.answerptrs = np.concatenate(self.answerptrs, axis=0)
        self.startptrs = np.concatenate(self.startptrs, axis=0)
        qmat = self.qsm.matrix
        self._trainmat = qmat[self.tvx == 0]
        self._validmat = qmat[self.tvx == 1]
        self._testmat = qmat[self.tvx == 2]
        self._traingold = self.answerptrs[self.tvx == 0]
        self._validgold = self.answerptrs[self.tvx == 1]
        self._testgold = self.answerptrs[self.tvx == 2]
        self._trainstartptrs = self.startptrs[self.tvx == 0]
        self._validstartptrs = self.startptrs[self.tvx == 1]
        self._teststartptrs = self.startptrs[self.tvx == 2]
        self._srcs = np.asarray(self._srcs)
        self._trainsrcs = self._srcs[self.tvx == 0]
        self._validsrcs = self._srcs[self.tvx == 1]
        self._testsrcs = self._srcs[self.tvx == 2]
        self._finalized = True
        #return [trainmat, trainstartptrs], [validmat, validstartptrs], [testmat, teststartptrs], \
        #        traingold, validgold, testgold

    def get(self, src=None):
        if src is None:     # return all
            return [self._trainmat, self._trainstartptrs], \
                   [self._validmat, self._validstartptrs], \
                   [self._testmat, self._teststartptrs], \
                   self._traingold, self._validgold, self._testgold
        else:
            if not issequence(src):
                src = [src]
            trainidxs = np.zeros_like(self._trainsrcs, dtype="bool")
            valididxs = np.zeros_like(self._validsrcs, dtype="bool")
            testidxs = np.zeros_like(self._testsrcs, dtype="bool")
            for srce in src:
                srcesplits = srce.split(".")
                srceparts = ["train", "valid", "test"]
                srce = srcesplits[0]
                if len(srcesplits) > 1:
                    srceparts = srcesplits[1:]
                if "train" in srceparts:
                    trainidxs |= self._trainsrcs == self._src_dic[srce]
                if "valid" in srceparts:
                    valididxs |= self._validsrcs == self._src_dic[srce]
                if "test" in srceparts:
                    testidxs |= self._testsrcs == self._src_dic[srce]
            trainmat = self._trainmat[trainidxs]
            validmat = self._validmat[valididxs]
            testmat = self._testmat[testidxs]
            traingold = self._traingold[trainidxs]
            validgold = self._validgold[valididxs]
            testgold = self._testgold[testidxs]
            trainstartptrs = self._trainstartptrs[trainidxs]
            validstartptrs = self._validstartptrs[valididxs]
            teststartptrs = self._teststartptrs[testidxs]
            return [trainmat, trainstartptrs], [validmat, validstartptrs], [testmat, teststartptrs],\
                   traingold, validgold, testgold




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
        wordembdim=50,
        gloveoverride=False,
        encdim=100,
        nenclayers=2,
        dropout=0.1,
        inspectdata=False,
        testpred=False,
        trainlabels=False,
        simplefind=False,
        simplefindred=False,
        simplehop=False,
        twohops=False,
        simple=False,
        actionoverride=False,
        smmode="sm",        # "sm" or "gumbel" or "maxhot"
        debug=False,
        loss="klp",         # "klp", "pwp", "bpwp"
        temperature=1.,
        enttemp=1.,
        reltemp=1.,
        acttemp=1.,
        recipe="none",       # "none" or e.g. "trainfind+simplefind.train/epochs=30,nsteps=1/simple/10"
        ):
    if debug:
        #inspectdata = True
        recipe = "trainlabels.train+labelsincontext.train+simplefind.valid.test/epochs=30"
    tt = ticktock("script")
    tt.tick("loading graph")
    graphtensor = loadtensor()
    tt.tock("graph loaded")
    tt.tick("loading examples")
    # parse recipe
    train_recipe = [(None,), epochs]
    _load_sources = tuple()
    if recipe != "none":
        assert(not (trainlabels or simplefind or simplehop or simple))
        train_recipe = [tuple([source for source in x.split("+")])
                 if i % 2 == 0
                        else dict([(y.split("=")[0], y.split("=")[1]) for y in x.split(",") if len(y.split("=")) == 2])
                 for i, x in enumerate(recipe.split("/"))]
        i = 0
        while i < len(train_recipe):
            _load_sources += tuple([tri.split(".")[0] for tri in train_recipe[i]])
            i += 2

        tt.msg("using recipe {}".format(recipe))
    ql = DataLoader(graphtensor)
    if trainlabels or "trainlabels" in _load_sources:
        tt.msg("adding labels")
        ql.add_labels(usefortrain=True, useforvalid=True, usefortest=True, src="trainlabels")
    if "labelsincontext" in _load_sources:
        tt.msg("adding labels in randomly generated contexts")
        ql.add_labels_with_random_contexts(freq=6, src="labelsincontext")
    if twohops or "twohops" in _load_sources:
        ql.add_chains(length=2, src="twohops")
    if simplefind or "simplefind" in _load_sources:
        tt.msg("adding simple questions for finds")
        ql.add_simple_questions(find_only=True, src="simplefind")
    if simplefindred or "simplefindred" in _load_sources:
        tt.msg("adding simple questions for finds, reduced")
        ql.add_simple_questions(find_only=True, trainportion=0.2, src="simplefindred")
    if simplehop or "simplehop" in _load_sources:
        tt.msg("adding simple questions for hops")
        ql.add_simple_questions(hop_only=True, src="simplehop")
    if simple or "simple" in _load_sources:
        tt.msg("adding simple questions")
        ql.add_simple_questions(src="simple")

    if trainlabels or simplefind or simplehop or simple or recipe != "none":
        ql.finalize()
        qsm = ql.qsm
        traindata, validdata, testdata, traingold, validgold, testgold \
            = ql.get()
        tt.tock("{} examples loaded".format(len(qsm.matrix)))
    else:
        assert(False)       # integrate in question loader
        qsm, answers = loadquestions(graphtensor)
        qmat = qsm.matrix
        # split 80/10/10
        splita, splitb = int(round(len(qmat) * 0.8)), int(round(len(qmat) * 0.9))
        trainmat, validmat, testmat = qmat[:splita, :], qmat[splita:splitb, :], qmat[splitb:, :]
        traingold, validgold, testgold = answers[:splita, :], answers[splita:splitb, :], answers[splitb:, :]
        traindata, validdata, testdata = [trainmat, np.zeros_like(traingold)], [validmat, np.zeros_like(validgold)], [testmat, np.zeros_like(testgold)]
        tt.tock("{} questions loaded".format(len(qmat)))
    if inspectdata and False:
        embed()

    if actionoverride:
        assert(recipe == "none")
        assert(trainlabels + simple + simplehop + simplefind <= 1)
        if trainlabels or simplefind:
            tt.msg("doing action override with find-hop template for simple questions")
            assert(nsteps >= 2)
            actionoverride = np.zeros((nsteps, DGTN_S.numacts), dtype="float32")
            actionoverride[:, 0] = 1.
            actionoverride[0, 1] = 1.
        elif simplehop:
            tt.msg("doing action override with find-hop template for simple questions")
            assert(nsteps >= 2)
            actionoverride = np.zeros((nsteps, DGTN_S.numacts), dtype="float32")
            actionoverride[:, 0] = 1.
            actionoverride[0, 1] = 1.
            actionoverride[1, 2] = 1.
        elif simple:
            tt.msg("doing action override with only a hop for simple questions")
            assert(nsteps >= 2)
            actionoverride = np.zeros((nsteps, DGTN_S.numacts), dtype="float32")
            actionoverride[:, 0] = 1.
            actionoverride[0, 2] = 1.
        else:
            raise Exception("don't know how to override this")
    else:
        actionoverride = None

    # build model
    tt.tick("building model")
    dgtn = DGTN_S(reltensor=graphtensor.tensor, nsteps=nsteps,
                entembdim=200, actembdim=10, attentiondim=encdim,
                entitysummary=False, relationsummary=False,
                action_override=actionoverride,
                gumbel=smmode=="gumbel", maxhot=smmode=="maxhot",
                )
    dgtn.disable("difference")
    dgtn.disable("swap")
    dgtn.disable("union")
    dgtn._ent_temp = temperature * enttemp
    dgtn._act_temp = temperature * acttemp
    dgtn._rel_temp = temperature * reltemp
    wordemb = WordEmb(dim=wordembdim, indim=qsm.numwords, worddic=qsm._dictionary, maskid=qsm.d("<MASK>"))
    if gloveoverride:
        wordemb = wordemb.override(Glove(wordembdim))
    enc = SeqEncoder.fluent()\
        .setembedder(wordemb)\
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

    if False:
        # prediction function
        dgtn._ret_actions = True
        dgtn._ret_entities = True
        dgtn._ret_relations = True
        predf = dgtn.predict
        testprediction, actions, entities, relations = predf(*[testdatamat[:5] for testdatamat in testdata])
        def tpred():
            return predf(*[testdatamat[:15] for testdatamat in testdata])
        # inspect prediction
        if testpred:
            embed()

    # choose loss
    if loss == "klp":
        trainloss = KLPointerLoss(softmaxnorm=False)
    elif loss == "pwp":
        trainloss = PWPointerLoss()
    elif loss == "bpwp":
        trainloss = PWPointerLoss(balanced=True)
    else:
        raise Exception("unknown loss option")

    # training

    i = 0
    while i < len(train_recipe):
        source, settings = train_recipe[i:i+2]
        local_epochs = int(settings["epochs"]) if "epochs" in settings else epochs
        local_nsteps = int(settings["nsteps"]) if "nsteps" in settings else nsteps
        traindata, validdata, testdata, traingold, validgold, testgold = ql.get(source)

        numbats = (len(traindata[0]) // batsize) + 1
        dgtn.nsteps = local_nsteps
        if "action" in settings:
            tt.msg("enabled only {}".format(settings["action"]))
            dgtn.disable("all")
            dgtn.disable("_nop")
            dgtn.enable(settings["action"])
            print dgtn._act_sel_mask

        dgtn._predictor = None
        dgtn._ret_actions = True
        dgtn._ret_entities = True
        dgtn._ret_relations = True
        predf = dgtn.predict
        testprediction = predf(*[testdatamat[:5] for testdatamat in testdata])
        if inspectdata:
            embed()

        tt.tick("training on {} ({} train examples, {} valid examples), nsteps={}".format("+".join(source), len(traindata[0]), len(validdata[0]), local_nsteps))
        dgtn._no_extra_ret()

        if not debug:
            dgtn.train(traindata, traingold)\
                .adadelta(lr=lr).loss(trainloss).loss(PointerFscore()).grad_total_norm(5.)\
                .validate_on(validdata, validgold).loss(trainloss).loss(PointerFscore()).loss(PointerRecall()).loss(PointerPrecision())\
                .train(numbats, local_epochs)

        tt.tock("trained")
        i += 2
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