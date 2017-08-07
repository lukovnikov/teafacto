import pickle, re, numpy as np
from teafacto.util import argprun, ticktock, StringMatrix
from IPython import embed
from teafacto.blocks.word.wordvec import WordEmb, Glove
from teafacto.core.base import Val, asblock, tensorops as T, Block
from teafacto.blocks.basic import SMO, Forward
from teafacto.blocks.seq import RNNSeqEncoder
from teafacto.blocks.seq.encdec import EncDec
from teafacto.blocks.seq.attention import Attention
from teafacto.blocks.seq.rnu import GRU


def loaddata(p="webqa.data.loaded.pkl"):
    tt = ticktock("loader")
    # region 1. load data
    tt.tick("loading data")
    data = pickle.load(open(p))
    textsm, formsm, validnexttokenses, exampleids, t_ub_ents, splitid = \
        [data[k] for k in "textsm formsm validnexttokenses exampleids t_ub_ents traintestsplit".split()]
    tt.tock("data loaded")
    # endregion
    # region 2. split train test
    train_nl_mat, test_nl_mat = textsm.matrix[:splitid], textsm.matrix[splitid:]
    train_fl_mat, test_fl_mat = formsm.matrix[:splitid], formsm.matrix[splitid:]
    train_vts, test_vts = validnexttokenses[:splitid], validnexttokenses[splitid:]
    nl_dic = textsm._dictionary
    fl_dic = formsm._dictionary
    #embed()
    # endregion
    # region 3. test missing stats
    tt.tick("missing stats")
    test_allrels = set()
    for test_vt in test_vts:
        for test_vte in test_vt:
            test_allrels.update(set([x for x in test_vte if x[0] == ":"]))
    train_allrels = set()
    for train_vt in train_vts:
        for train_vte in train_vt:
            train_allrels.update(set([x for x in train_vte if x[0] == ":"]))
    tt.msg("{}/{} rels in test not in train"
           .format(len(test_allrels.difference(train_allrels)), len(test_allrels)))
    tt.tock("missing stats computed")
    # endregion
    # region 4. relation representations preparing
    # update fl_dic
    allrels = train_allrels | test_allrels
    i = max(fl_dic.values()) + 1
    for rel in allrels:
        if rel not in fl_dic:
            fl_dic[rel] = i
            i += 1
    vtn_mat = np.zeros((len(validnexttokenses), formsm.matrix.shape[1], max(fl_dic.values()) + 1), dtype="int8")
    vtn_mat[:, :, fl_dic["<MASK>"]] = 1
    for i, validnexttokens in enumerate(validnexttokenses):
        for j, validnexttokense in enumerate(validnexttokens[1:]):
            vtn_mat[i, j, fl_dic["<MASK>"]] = 0
            for validnexttoken in validnexttokense:
                if validnexttoken not in fl_dic:
                    pass #print i, j, validnexttoken
                else:
                    vtn_mat[i, j, fl_dic[validnexttoken]] = 1
    train_vtn_mat = vtn_mat[:splitid]
    test_vtn_mat = vtn_mat[splitid:]
    # get list of all relations and make reldic and relmat
    allrels = set([x for x in fl_dic.keys() if x[0] == ":"])
    rel_dic = {}

    def rel_tokenizer(uri):
        pre = ":forward:"
        if uri[:8] == ":reverse":
            pre = ":reverse:"
            uri = uri[8:]
        uri = uri[1:]
        uri = uri.replace(".", " :dot: ").replace("_", " ").split()
        uri = [pre] + uri
        return uri

    sm = StringMatrix(freqcutoff=0)
    sm.tokenize = rel_tokenizer
    for k, rel in enumerate(allrels):
        rel_dic[rel] = k
        sm.add(rel)
    sm.finalize()
    rel_mat = sm.matrix
    rel_mat_dic = sm._dictionary
    # endregion
    return (train_nl_mat, train_fl_mat, train_vtn_mat), (test_nl_mat, test_fl_mat, test_vtn_mat), \
           (nl_dic, fl_dic, rel_dic), (rel_mat, rel_mat_dic)


# TODO: initialize decoder state to encoding at first <E0> position
# TODO: try bypass connections in RNNs
# TODO: incorporate given entity linker results

def run(p="webqa.data.loaded.pkl",
        worddim=50,
        fldim=50,
        encdim=50,
        decdim=50,
        numenclayers=1,
        numdeclayers=1,
        testpred=False,
        inspectdata=False,
        lr=0.1,
        dropout=0.1,
        wreg=0.00000001,
        gradnorm=1.0,
        epochs=20,
        numbats=100,
        mode="s2s",         # "s2s", "disappointer" # TODO
        usezeropos=False,
        notrainmask=False,  # don't use the dec out symbol mask during training
        attention="forward",    # forward or dot
        layernorm=False,
        ):
    GRU.layernormalize = layernorm
    tt = ticktock("script")
    (train_nl_mat, train_fl_mat, train_vtn), (test_nl_mat, test_fl_mat, test_vtn), \
    (nl_dic, fl_dic, rel_dic), (rel_mat, rel_mat_dic) = loaddata(p)
    # add starts
    train_fl_mat = np.concatenate([
        np.ones_like(train_fl_mat[:, 0:1]) * fl_dic["<START>"],
        train_fl_mat], axis=1)
    test_fl_mat = np.concatenate([
        np.ones_like(test_fl_mat[:, 0:1]) * fl_dic["<START>"],
        test_fl_mat], axis=1)
    vtn_e0_vec = np.zeros((1, 1, train_vtn.shape[2]), dtype=train_vtn.dtype)
    vtn_e0_vec[0, 0, fl_dic["<E0>"]] = 1
    train_vtn = np.concatenate([vtn_e0_vec.repeat(train_vtn.shape[0], axis=0),
                                train_vtn], axis=1)
    test_vtn = np.concatenate([vtn_e0_vec.repeat(test_vtn.shape[0], axis=0),
                               test_vtn], axis=1)
    # find first position of <E0> in question
    def find_e0_pos(x):
        pos = np.zeros((x.shape[0],), dtype="int32")
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] == nl_dic["<E0>"]:
                    pos[i] = j
                    break
        return pos
    train_e0_pos = find_e0_pos(train_nl_mat)
    test_e0_pos = find_e0_pos(test_nl_mat)

    if inspectdata:
        embed()

    glove = Glove(worddim)

    tt.tick("building nl reps")
    nl_emb = get_nl_emb(nl_dic, glove=glove, dim=worddim)
    tt.tock("built nl reps")

    tt.tick("building fl reps")
    fl_emb = get_fl_emb(fl_dic, rel_dic, rel_mat, rel_mat_dic,
                        glove=glove, dim=fldim, worddim=worddim, dropout=dropout)
    fl_smo = get_fl_smo_from_emb(fl_emb)
    tt.tock("built fl reps")

    # test_rel_smo(train_nl_mat, train_fl_mat, test_nl_mat, test_fl_mat,
    #              nl_emb, fl_dic, fl_smo, encdim=encdim, dropout=dropout,
    #              testpred=testpred, numbats=numbats, epochs=epochs, lr=lr)

    # MODEL :::::::::::::::::::::::::::::::::::::::::::::
    # encoder
    encoder = RNNSeqEncoder.fluent().setembedder(nl_emb)\
        .addlayers([encdim]*numenclayers, bidir=True, dropout_in=dropout, zoneout=dropout)\
        .addlayers(encdim, bidir=False, dropout_in=dropout, zoneout=dropout)\
        .make().all_outputs()

    if attention == "forward":
        attention = Attention().forward_gen(encdim, decdim, decdim)
    elif attention == "dot":
        assert(encdim == decdim)
        attention = Attention().dot_gen()

    if not usezeropos:
        dec_state_init_block = Forward(encdim, decdim)
        dec_state_init = asblock(lambda x: dec_state_init_block(x[:, -1, :]))
    else:
        class DecStateInit(Block):
            def set_positions(self, x):
                self.positions = x
            def apply(self, enco):
                ret = enco[T.arange(self.positions.shape[0]), self.positions]
                return ret
        dec_state_init = DecStateInit()

    trans_y_t_from_dim = encdim + decdim + fl_emb.outdim
    transform_y_t = Forward(trans_y_t_from_dim, fl_emb.outdim)

    encdec = EncDec(encoder=encoder,
                    attention=attention,
                    init_state_gen=dec_state_init,
                    transform_y_t=transform_y_t,
                    inconcat=True,
                    outconcat=True,
                    stateconcat=True,
                    concatdecinptoout=True,
                    inpemb=fl_emb,
                    innerdim=[decdim]*numdeclayers,
                    smo=None,
                    dropout_in=dropout,
                    zoneout=dropout,
                    return_attention_weights=True,
                    )

    def m_fun(inpseq, outseq, outmask, e0_pos, _trainmode=False):
        #print "trainmode: {}".format(_trainmode)
        if usezeropos:
            dec_state_init.set_positions(e0_pos)
        deco, attention_weights = encdec(outseq, inpseq)
        #deco += T.sum(e0_pos) * 0.      # avoid unused input
        seqmask = deco.mask     # save seqmask from dec out
        if notrainmask:
            outmask = T.ones_like(outmask)
        deco.mask = outmask     # set forced action mask
        out = fl_smo(deco)     # compute output probs
        out.mask = seqmask      # restore seqmask
        return out

    m = asblock(m_fun)

    def runtest(total=1639, inspect=False, step=20):        # see http://anthology.aclweb.org/P16-2033
        tt.tick("running test")
        acc = 0
        tot = test_nl_mat.shape[0]
        i = 0
        j = 0
        rev_nl_dic = {v: k for k, v in nl_dic.items()}
        rev_fl_dic = {v: k for k, v in fl_dic.items()}
        while j < test_nl_mat.shape[0]:
            tt.progress(i, tot, live=True)
            j = min(i + step, test_nl_mat.shape[0])
            testpred = m.predict(test_nl_mat[i:j], test_fl_mat[i:j, :-1], test_vtn[i:j, :-1], test_e0_pos[i:j])
            testpred = np.argmax(testpred, axis=2)
            eqs = np.all(test_fl_mat[i:j, 1:] == testpred, axis=1)
            acc += eqs.sum()
            if inspect:
                def pp(q, g, a, p):
                    for qe, ge, ae, pe in zip(q, g, a, p):
                        print "Question:\t {}".format(" ".join([rev_nl_dic[qex] for qex in qe if qex != 0]))
                        print "\tGolden:\t {}".format(" ".join([rev_fl_dic[gex] for gex in ge if gex != 0]))
                        print "\tPredicted:\t {}".format(" ".join([rev_fl_dic[aex] for aex in ae if aex != 0]))
                        print "Positions: {}".format(pe)
                pp(test_nl_mat[i:j], test_fl_mat[i:j], testpred, pos)
                k = raw_input("press to continue, (s) to stop\n>>")
                if k == "s":
                    inspect = False
            i += step

        tt.tock("test run")
        tt.msg("Accuracy on test: {} ({}/{}); [all:] {} ({}/{})".format(acc * 1.0 / tot, acc, tot, acc * 1.0 / total, acc, total))

    def testpredictions():        # do some predictions for debugging
        dev_inpseq = Val(train_nl_mat[:5])
        dev_outseq = Val(train_fl_mat[:5])
        dev_outmsk = Val(train_vtn[:5])
        dev_e0pos = Val(train_e0_pos[:5])
        # outputs
        dev_inpenc = encoder(dev_inpseq)
        if usezeropos:
            encdec.init_state_gen.set_positions(dev_e0pos)
        dev_dec_init = encdec.init_state_gen(dev_inpenc)
        dev_decout, dev_m_att = encdec(dev_outseq[:, :-1], dev_inpseq)
        dev_m_out = m(dev_inpseq, dev_outseq[:, :-1], dev_outmsk[:, :-1], dev_e0pos)
        r = runtest
        embed()

    if testpred:
        testpredictions()

    m.train([train_nl_mat, train_fl_mat[:, :-1], train_vtn[:, :-1], train_e0_pos], train_fl_mat[:, 1:])\
        .adadelta(lr=lr).seq_cross_entropy().seq_accuracy().grad_total_norm(gradnorm).l2(wreg)\
        .split_validate(splits=10)\
        .seq_cross_entropy().seq_accuracy()\
        .train(numbats=numbats, epochs=epochs)

    embed()


def get_nl_emb(nl_dic, glove=None, dim=50, dropout=None):
    emb = WordEmb(worddic=nl_dic, dim=dim)
    if glove is not None:
        emb = emb.override(glove)
    return emb


def get_fl_emb(fl_dic, rel_dic, rel_mat, rel_mat_dic,
               dim=50, worddim=50, dropout=None, glove=None):
    # !!! fl_dic and rel_dic have different indices for same rel
    # 1. make normal vector reps from fl_dic
    base_emb = WordEmb(worddic=fl_dic, dim=dim)
    # 2. build overriding block
    # 2.1 emb words in rel uris
    relwordemb = WordEmb(worddic=rel_mat_dic, dim=worddim)
    if glove is not None:   # override with glove
        relwordemb = relwordemb.override(glove)
    # 2.2 encode rel mat
    relenc = RNNSeqEncoder.fluent().setembedder(relwordemb)\
        .addlayers(dim=dim, dropout_in=dropout, zoneout=dropout).make()
    rel_mat_val = Val(rel_mat)
    rel_mat_enc = relenc(rel_mat_val)
    # 2.3 use encs in overriding wordemb
    rel_emb = WordEmb(worddic=rel_dic, dim=dim, value=rel_mat_enc)
    # 3. override
    fl_emb = base_emb.override(rel_emb)
    return fl_emb


def get_fl_smo_from_emb(fl_emb, dropout=None):
    smo = SMO(fl_emb.indim, fl_emb.outdim, dropout=dropout, nobias=True)
    smo.l.W = fl_emb.W.T
    return smo


def test_rel_smo(train_nl_mat, train_fl_mat, test_nl_mat, test_fl_mat,
                 nl_emb, fl_dic, fl_smo, encdim=50, dropout=None,
                 testpred=False, numbats=None, epochs=None, lr=None):
    # learn to predict one relation from sentence
    rev_fl_dic = {v: k for k, v in fl_dic.items()}
    # region get gold
    traingold = np.zeros((train_fl_mat.shape[0],), dtype="int32")
    testgold = np.zeros((test_fl_mat.shape[0],), dtype="int32")
    def _get_gold(fl_mat, gold):
        for i in range(len(fl_mat)):
            for j in range(fl_mat.shape[1]):
                if rev_fl_dic[fl_mat[i, j]][0] == ":":
                    gold[i] = fl_mat[i, j]
                    break
    _get_gold(train_fl_mat, traingold)      # in fl dic domain
    _get_gold(test_fl_mat, testgold)
    # endregion

    # encode question
    encoder = RNNSeqEncoder.fluent().setembedder(nl_emb)\
        .addlayers(dim=encdim, bidir=True, dropout_in=dropout, zoneout=dropout)\
        .addlayers(dim=encdim, dropout_in=dropout, zoneout=dropout)\
        .make()

    def block_apply(x):     # (batsize, seqlen)^wordids
        enco = encoder(x)   # (batsize, encdim)
        out = fl_smo(enco)  # (batsize, fl_dic_size)
        return out

    m = asblock(block_apply)

    if testpred:
        testpred = m.predict(train_nl_mat[:5])

    m.train([train_nl_mat], traingold).cross_entropy().accuracy()\
        .adadelta(lr=lr).grad_total_norm(5.)\
        .validate_on([test_nl_mat], testgold).cross_entropy().accuracy()\
        .train(numbats=numbats, epochs=epochs)
    embed()

if __name__ == "__main__":
    argprun(run)
