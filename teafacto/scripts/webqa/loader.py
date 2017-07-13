from teafacto.util import argprun, ticktock, StringMatrix, tokenize
import re, json
from collections import OrderedDict
from teafacto.scripts.webqa.traversal import Traverser
from IPython import embed


def run(p="../../../data/WebQSP/data/"):
    #q = "<E0> :film.actor.film <E1> :reverse:film.performance.film <JOIN> :film.performance.character <RETURN>"
    #vnt = _get_valid_next_tokens("<E0> :film.actor.film <E1> :reverse:film.performance.film <JOIN> :film.performance.character <RETURN>")
    #for qe, vnte in zip(q.split(), vnt):
    #    print qe, vnte
    loaded = load_lin_question("WebQTrn-1641	what is the name of the main train station in san francisco	0	4	1	0	m.0d6lp[san francisco]* travel.travel_destination.how_to_get_here var1 ; OUT location.location.containedby m.0d6lp[san francisco] ; var1 travel.transportation.transport_terminus OUT ; var1 travel.transportation.mode_of_transportation m.07jdr[train] ;")
    qid, question, answer, (nldic, lfdic), info = loaded
    answer, ub_ents = relinearize(answer)
    result, validrels = enrich_lin_q(answer, lfdic)
    print answer
    print _get_valid_next_tokens(answer, validrels=validrels, ub_entities=ub_ents)
    #print ""
    #load_lin(p=p)


def load_lin(p="../../../data/WebQSP/data/"):
    trainexamplespath = p+"WebQSP.train.lin"
    testexamplespath = p+"WebQSP.test.lin"
    trainexamples, t_ub_ents = load_lin_dataset(trainexamplespath)
    testexamples, x_ub_ents = load_lin_dataset(testexamplespath)
    ub_ents = t_ub_ents | x_ub_ents
    textsm = StringMatrix(indicate_start_end=True, freqcutoff=2)
    textsm.tokenize = lambda x: tokenize(x, preserve_patterns=["<E\d>"])
    formsm = StringMatrix(indicate_start_end=False, freqcutoff=2)
    formsm.tokenize = lambda x: x.split()
    exampleids = []
    validrelses = []
    validnexttokenses = []
    for trainexample in trainexamples:
        exampleids.append(trainexample[0])
        textsm.add(trainexample[1])
        formsm.add(trainexample[2])
        validrelses.append(trainexample[5])
        validnexttokenses.append(_get_valid_next_tokens(trainexample[2], trainexample[5], ub_ents))
    for testexample in testexamples:
        exampleids.append(testexample[0])
        textsm.add(testexample[1])
        formsm.add(testexample[2])
        validrelses.append(testexample[5])
        validnexttokenses.append(_get_valid_next_tokens(testexample[2], testexample[5], ub_ents))
    allvalidrelses = set()
    for validrels_i in validrelses:
        for validrels_e in validrels_i:
            allvalidrelses.update(validrels_e)
    embed()
    textsm.finalize()
    formsm.finalize()
    print textsm.matrix[:5]
    print textsm.pp(textsm.matrix[:5])
    print textsm.matrix.shape[1]
    print formsm.pp(formsm.matrix[:5])
    print formsm.matrix.shape[1]
    for k, v in sorted(formsm._wordcounts_original.items(), key=lambda (x, y): y, reverse=True):
        print "{}: {}".format(k, v)
    print len(formsm._dictionary)
    print len(textsm._dictionary)


def _get_valid_next_tokens(tree, validrels=None, ub_entities=set()):
    ent_placeholders = {"<E0>", "<E1>", "<E2>", "<E3>", "<E4>"}
    tokens = tree.split()
    if validrels is None:
        validrels = []
        for _ in tokens:
            validrels.append(set())
    assert(len(tokens) == len(validrels))
    validtokens = []
    branching = 0
    argmaxer = False
    i = 0
    valid_next_tokens = {"<E0>"}
    for token, validrels_for_token in zip(tokens, validrels):
        assert((token[1:] if token[0] == ":" else token) in valid_next_tokens)
        valid_next_tokens = set()
        if i < len(tokens) - 1:
            #valid_next_tokens.update({tokens[i + 1]})       # ensure next rel is there
            pass
        if re.match("<E\d>", token) or token in ub_entities:    # entity placeholder or unbound entity
            pass                        # only relations
            branching += 1
            argmaxer = False            # resets argmaxer by assumption of just one-hop argmaxes
            valid_next_tokens.update(validrels_for_token)   # only relations
            #assert(tokens[i+1] in validrels_for_token)
        elif token[0] == ":":       # relation
            if argmaxer is True:    # if this hop is inside argmax
                valid_next_tokens.update({"<JOIN>"})    # can only join
                valid_next_tokens.update(ent_placeholders | ub_entities)    # or start a new branch
            elif branching > 1:    # if already two branches, must join or follow can't start new branch or return
                valid_next_tokens.update({"<JOIN>"})
                valid_next_tokens.update({"ARGMAX", "ARGMIN"})
                valid_next_tokens.update(ent_placeholders | ub_entities)
                valid_next_tokens.update(validrels_for_token)
            elif branching == 1:    # if just one branch, can't join
                valid_next_tokens.update({"<RETURN>", "ARGMAX", "ARGMIN"})
                valid_next_tokens.update(ent_placeholders | ub_entities)
                valid_next_tokens.update(validrels_for_token)
            else:
                raise Exception("invalid branching {} in {}".format(branching, tree))
        elif token == "<JOIN>":
            valid_next_tokens.update({"<RETURN>"})
            valid_next_tokens.update(ent_placeholders | ub_entities)
            valid_next_tokens.update({"<RETURN>", "ARGMAX", "ARGMIN"})
            valid_next_tokens.update(validrels_for_token)
            branching -= 1      # merges two branches in one
            argmaxer = False    # resets argmaxer
        elif token == "ARGMAX" or token == "ARGMIN":    # a relation must follow
            valid_next_tokens.update(validrels_for_token)
            argmaxer = True     # entering argmax
            branching += 1      # creates new branch
        elif token == "<RETURN>":
            valid_next_tokens.update({"<MASK>"})
        else:
            raise Exception("unsupported token {} in {}".format(token, tree))
        i += 1
        validtokens.append(valid_next_tokens)
    return validtokens


def load_lin_dataset(p):
    ret = []
    c = 0
    all_ub_ents = set()
    with open(p) as f:
        maxlen = 0
        for line in f:
            loaded = load_lin_question(line)
            if loaded is not None:
                qid, question, answer, (nldic, lfdic), info = loaded
                #print c, question
                #print question
                if "what was the name of" in question:
                    pass
                answer, ub_ents = relinearize(answer)
                all_ub_ents.update(ub_ents)
                #print answer
                maxlen = max(maxlen, len(answer.split()))
                result, validrels = enrich_lin_q(answer, lfdic)
                print c, answer, lfdic
                print len(answer.split()), len(validrels), [len(x) for x in validrels]
                ret.append((qid, question, answer, (nldic, lfdic), info, validrels))
            else:
                print c, "not loaded"
            c += 1
            #break
        print "{} max len".format(maxlen)
    print "done"
    return ret, all_ub_ents


def enrich_lin_q(query, lfdic):
    t = Traverser()
    res, validrels = t.traverse_tree(query, lfdic)
    #assert([len(x) for x in validrels].count(0) == 1)
    return res, validrels


def relinearize(q, binary_tree=True):
    triples = [tuple(x.strip().split()) for x in q.strip().split(";") if len(x) > 0]
    lin, ub_ents = _relin_rec(triples, "OUT", binary_tree=binary_tree)
    if len(lin) == 1:
        lin = lin[0]
    else:
        if binary_tree:
            lin = _join_branches_binary(lin)
        else:
            lin = " and ".join(lin)
    lin += " <RETURN>"
    return lin, ub_ents


def _relin_rec(triples, root, binary_tree=False):
    roottriples = []    # triples resulting in root node
    redtriples = []     # other triples
    ub_ents = set()
    if not (re.match("var\d", root) or root == "OUT"):
        if not re.match("<E\d>", root):
            ub_ents.add(root)
        return [root], ub_ents
    for s, p, o in triples:
        if s == root:
            roottriples.append((o, ":reverse:"+p, s))
        elif o == root:
            roottriples.append((s, ":"+p, o))
        else:
            redtriples.append((s, p, o))
    sublins = []
    for s, p, o in roottriples:
        sublin, l_ub_ents = _relin_rec(redtriples, s, binary_tree=binary_tree)
        ub_ents.update(l_ub_ents)
        if len(sublin) == 1:
            sublin = sublin[0]
        else:
            if binary_tree:
                sublin = _join_branches_binary(sublin)
            else:
                sublin = "( {} )".format(" and ".join(sublin))
        sublin = "{} {}".format(sublin, p)
        sublins.append(sublin)
    return sublins, ub_ents
    # orient triples right way


def _join_branches_binary(branches):
    # sort branches
    def _sorter_cmp(x, y):
        x = x.split()[0]
        y = y.split()[0]
        if x == "ARGMIN" or y == "ARGMIN" or x == "ARGMAX" or y == "ARGMAX":
            pass
        if re.match(r'<E\d{1,2}>', x) and re.match(r'<E\d{1,2}>', y):
            return 1 if x > y else -1 if x < y else 0
        elif re.match(r'<E\d{1,2}>', x):
            return -1
        elif re.match(r'<E\d{1,2}>', y):
            return 1
        else:
            if (x == "ARGMAX" or x == "ARGMIN") and (y == "ARGMAX" or y == "ARGMIN"):
                return 0
            elif x == "ARGMAX" or x == "ARGMIN":
                return 1
            elif y == "ARGMAX" or y == "ARGMIN":
                return -1
            else:
                if re.match(r'[a-z]{1,2}\..+', x) and re.match(r'[a-z]{1,2}\..+', y):
                    return 1 if x > y else -1 if x < y else 0
                elif re.match(r'[a-z]{1,2}\..+', x):
                    return -1
                elif re.match(r'[a-z]{1,2}\..+', y):
                    return 1
                else:
                    return 1 if x > y else -1 if x < y else 0
    branches = sorted(branches, cmp=_sorter_cmp)
    # join branches
    acc = branches[0]
    for branch in branches[1:]:
        acc = "{} {} <JOIN>".format(acc, branch)
    return acc


def load_lin_question(line):
    splits = line.split("\t")
    if len(splits) > 3:
        qid, question, unlinkedents, numrels, numvars, valconstraints, query = splits
        unlinkedents, numrels, numvars, valconstraints = map(int, (unlinkedents, numrels, numvars, valconstraints))
        # replace entities by placeholders
        entitymatches = re.findall(r"([a-z]+\.[^\s\[]+)\[([^\]]+)\](\*?)", query)
        topicmid = None
        fbid2str = OrderedDict()
        if qid == "WebQTest-16":
            pass
        for mid, sff, topic in entitymatches:
            if topic == "*":
                fbid2str[mid] = sff
                topicmid = mid
        for mid, sff, topic in entitymatches:
            if topic != "*":
                fbid2str[mid] = sff
        if len(set(fbid2str.values())) != len(fbid2str.values()):
            print qid
        nl_emdic = {}
        i = 0
        for _, strr in fbid2str.items():
            if strr in question:
                if strr not in nl_emdic:
                    nl_emdic[strr] = "<E{}>".format(i)
                    i += 1
        #nl_emdic = dict(zip(fbid2str.values(), ["E{}".format(i) for i in range(len(fbid2str.values()))]))
        lf_emdic = {}
        for fbid, strr in fbid2str.items():
            if strr in nl_emdic:
                evar = nl_emdic[strr]
                if evar in lf_emdic.values():
                    pass
                else:
                    lf_emdic[fbid] = evar
            else:
                pass

        for entmatch, eid in nl_emdic.items():
            if entmatch in question:
                question = question.replace(entmatch, eid)
        for entmatch, eid in lf_emdic.items():
            if entmatch in query:
                query = query.replace(entmatch, eid)
        query = re.sub(r'\[[^\]]+\]\*?', "", query)
        rev_nl_emdic = {v: k for k, v in nl_emdic.items()}
        rev_lf_emdic = {v: k for k, v in lf_emdic.items()}
        return qid, question, query, (rev_nl_emdic, rev_lf_emdic), {"qid": qid, "unlinkedents": unlinkedents, "numrels": numrels, "numvars": numvars, "valconstraints": valconstraints}
    else:
        return None




if __name__ == "__main__":
    argprun(run)