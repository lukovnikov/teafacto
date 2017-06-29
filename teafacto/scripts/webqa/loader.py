from teafacto.util import argprun, ticktock, StringMatrix
import re


def run(p="../../../data/WebQSP/data/"):
    load_lin(p=p)


def load_lin(p="../../../data/WebQSP/data/"):
    trainexamplespath = p+"WebQSP.train.lin"
    testexamplespath = p+"WebQSP.test.lin"
    trainexamples = load_lin_dataset(trainexamplespath)
    for trainexample in trainexamples:
        #print trainexample
        pass
    testexamples = load_lin_dataset(testexamplespath)


def load_lin_dataset(p):
    ret = []
    with open(p) as f:
        maxlen = 0
        for line in f:
            loaded = load_lin_question(line)
            if loaded is not None:
                question, answer, (nldic, lfdic), info = loaded
                print question
                if "what is the <E1> of <E0>" in question:
                    pass
                answer = relinearize(answer)
                maxlen = max(maxlen, len(answer.split()))
                print answer
                ret.append((question, answer, (nldic, lfdic), info))
        print "{} max len".format(maxlen)
    print "done"
    return ret


def relinearize(q, binary_tree=True):
    triples = [tuple(x.strip().split()) for x in q.strip().split(";") if len(x) > 0]
    lin = _relin_rec(triples, "OUT", binary_tree=binary_tree)
    if len(lin) == 1:
        lin = lin[0]
    else:
        if binary_tree:
            lin = _join_branches_binary(lin)
        else:
            lin = " and ".join(lin)
    lin += " <RETURN>"
    return lin


def _relin_rec(triples, root, binary_tree=False):
    roottriples = []    # triples resulting in root node
    redtriples = []     # other triples
    if not (re.match("var\d", root) or root == "OUT"):
        return [root]
    for s, p, o in triples:
        if s == root:
            roottriples.append((o, "reverse:"+p, s))
        elif o == root:
            roottriples.append((s, p, o))
        else:
            redtriples.append((s, p, o))
    sublins = []
    for s, p, o in roottriples:
        sublin = _relin_rec(redtriples, s)
        if len(sublin) == 1:
            sublin = sublin[0]
        else:
            if binary_tree:
                sublin = _join_branches_binary(sublin)
            else:
                sublin = "( {} )".format(" and ".join(sublin))
        sublin = "{} {}".format(sublin, p)
        sublins.append(sublin)
    return sublins
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
        entitymatches = re.findall(r"[a-z]\.[^\s\[]+\[[^\]]+\]", query)
        fbid2str = dict([tuple(em[:-1].split("[")) for em in entitymatches])
        if len(set(fbid2str.values())) != len(fbid2str.values()):
            print qid
        nl_emdic = {}
        i = 0
        for strr in fbid2str.values():
            if strr in question:
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
                    lf_emdic["{}[{}]".format(fbid, strr)] = evar
            else:
                pass

        for entmatch, eid in nl_emdic.items():
            if entmatch in question:
                question = question.replace(entmatch, eid)
        for entmatch, eid in lf_emdic.items():
            if entmatch in query:
                query = query.replace(entmatch, eid)
        query = re.sub(r'\[[^\]]+\]', "", query)
        rev_nl_emdic = {v: k for k, v in nl_emdic.items()}
        rev_lf_emdic = {v: k for k, v in lf_emdic.items()}
        return question, query, (rev_nl_emdic, rev_lf_emdic), {"qid": qid, "unlinkedents": unlinkedents, "numrels": numrels, "numvars": numvars, "valconstraints": valconstraints}
    else:
        return None




if __name__ == "__main__":
    argprun(run)