from IPython import embed
import re, random, sys, numpy as np, scipy.sparse as sp
from collections import OrderedDict


def run():
    return loadtriples()


def loadtriples(tripf="got.f.triples", inspect=False):
    triples = []
    labels = {}
    aliases = {}
    entities = {}
    relations = {}
    with open(tripf) as f:
        for line in f:
            try:
                s, p, o = map(lambda x: x.strip().decode("utf8"), line.split("\t"))
            except Exception:
                print "exception!!!"
                embed()
            entities[s] = entities[s] + 1 if s in entities else 1
            relations[p] = relations[p] + 1 if p in relations else 1
            if p == ":label":
                labels[s] = o
            elif p == ":alias":
                if s not in aliases:
                    aliases[s] = []
                aliases[s].append(o)
            elif re.match(r':.+', o):
                triples.append((s, p, o))
                entities[o] = entities[o] + 1 if o in entities else 1
    names = {}
    for entity in entities:
        if re.match(r':Type:.+', entity):
            names[entity] = re.match(r':Type:(.+)', entity).group(1).replace('_', ' ')
        elif re.match(r':.+', entity):
            names[entity] = re.match(r':(.+)', entity).group(1).replace('_', ' ')
            if re.match(r':.+\(.+\)', entity):
                names[entity] = re.match(r':(.+)\(.+\)', entity).group(1).replace("_", " ").strip()
    for rel in relations:
        names[":rel"+rel] = re.match(r':(.+)', rel).group(1).replace("_", " ")
    if inspect:
        embed()
    entities = OrderedDict(sorted(entities.items(), key=lambda (x,y): y, reverse=True))
    relations = OrderedDict(sorted(relations.items(), key=lambda (x, y): y, reverse=True))
    rellex = loadrellex(p="rellex.tsv")
    return Graph(triples, names, labels, aliases, entities, relations, rellex=rellex)


def loadrellex(p="rellex.tsv"):
    lex = {}
    with open(p) as f:
        for line in f:
            splits = [split.strip() for split in line.split(",")]
            if len(splits) < 2:
                continue
            relid = splits[0]
            lexes = splits[1:]
            lex[relid] = lexes
    return lex



class Graph(object):
    def __init__(self, triples, names, labels, aliases, entities, relations, rellex=None):
        self._triples = triples
        self._names = names
        self._labels = labels
        self._aliases = aliases
        self._entcounts = entities
        self._relcounts = relations
        self._types = {s: o for (s, p, o) in self._triples if p == ":type"}
        #self._types.update({s: ":Type:None" for s in filter(lambda x: x not in self._types, self._entcounts.keys())})
        self._tensor, self._entdic, self._reldic = self._to_tensor()
        self._dense_tensor = None
        self._red = {v: k for k, v in self._entdic.items()}
        self._rld = {v: k for k, v in self._reldic.items()}
        self._rellex = rellex

    def types_of(self, *entities):
        ret = []
        for entity in entities:
            ret.append(self._types[entity] if entity in self._types else ":Type:None")
        return ret

    def get_lex(self, entity):
        name = label = None
        aliases = []
        if entity in self._names:
            name = self._names[entity]
        if entity in self._labels:
            label = self._labels[entity]
        if entity in self._aliases:
            aliases = self._aliases[entity]
        return name, label, aliases

    def triples(self, s=None, p=None, o=None):
        for st, pt, ot in self._triples:
            if (st == s if s is not None else True) and \
                (pt == p if p is not None else True) and \
                (ot == o if o is not None else True):
                yield (st, pt, ot)

    @staticmethod
    def from_triples(p="got.f.triples"):
        return loadtriples(p)

    def _to_tensor(self):
        entdic = OrderedDict(zip(self._entcounts.keys(), range(len(self._entcounts))))
        reldic = OrderedDict(zip(self._relcounts.keys(), range(len(self._relcounts))))
        tensor = [sp.dok_matrix((len(entdic), len(entdic)), dtype="int8") for _ in range(len(reldic))]
        acc = 0
        for s, p, o in self._triples:
            pid = reldic[p]
            sid = entdic[s]
            oid = entdic[o]
            tensor[pid][sid, oid] = 1
        tensor = [t.tocsr() for t in tensor]
        if True:        # add reverse
            reldic.update(OrderedDict(zip(["-" + k for k in self._relcounts.keys()],
                                          range(len(self._relcounts), len(self._relcounts) * 2))))
            for i in range(len(tensor)):
                tensor.append(tensor[i].transpose())
        return tensor, entdic, reldic

    def to_tensor(self, addreverse=True, dense=False):
        tensor, entdic, reldic = self._tensor, self._entdic, self._reldic
        if dense:
            if self._dense_tensor is None:
                dense_tensor = np.zeros((len(reldic), len(entdic), len(entdic)), dtype="int8")
                for i, relslice in enumerate(tensor):
                    dense_tensor[i, :, :] = relslice.todense()
                tensor = dense_tensor
                self._dense_tensor = dense_tensor
            else:
                tensor = self._dense_tensor     # cache
        if not addreverse:
            tensor = tensor[:len(tensor)//2]
            reldickeys = reldic.keys()[:len(tensor)//2]
            reldic = {k: reldic[k] for k in reldickeys}
        return tensor, entdic, reldic

    def save_tensor(self, outp="tensor.dock"):      # save tensor and dictionaries
        # saves tensor as dok
        with open(outp, "w") as dokfile:
            relmats = []
            for ridx, relmat in enumerate(self._tensor):
                entries = np.argwhere(relmat)
                entries = np.concatenate([
                                          ridx * np.ones_like(entries[:, 0:1], dtype="int32"),
                                          entries[:, 0:1],
                                          entries[:, 1:2]],
                                         axis=1)
                relmats.append(entries)
            doktensor = np.concatenate(relmats, axis=0)
            import pickle
            pickle.dump({"entries": doktensor, "entdic": self._entdic, "reldic": self._reldic},
                        dokfile)


class QuestionGenerator(object):
    ent_blacklist = {":Type:None"}
    rel_blacklist = {":type", ":result", "-:result", ":Type", "-:Type"}

    def __init__(self, graph):
        self._graph = graph
        self.rel_blacklist = set(map(lambda x: self._graph._reldic[x], self.rel_blacklist))

    def generateChain(self, seednode=None, max_hops=3, min_hops=1):
        tensor, entdic, reldic = self._graph.to_tensor(addreverse=True, dense=False)
        if seednode is None:
            seedtriple = random.choice(self._graph._triples)
            seednode = entdic[seedtriple[0]]
        hasrels = []
        for relslice in tensor:
            s = (relslice.sum(axis=1) > 0) * 1
            hasrels.append(s)
        hasrels = np.concatenate(hasrels, axis=1)
        n_hops = random.choice(range(min_hops, max_hops+1))
        curnodes = [seednode]
        path = []
        # sample forward
        for i in range(n_hops):
            curnode = random.choice(curnodes)
            #print red[curnode]
            relsofnode = set(np.argwhere(hasrels[curnode])[:, 1])
            rel = random.choice(list(relsofnode.difference(self.rel_blacklist)))
            #print "->{}->".format(rld[rel])
            curnodes = list(np.argwhere(tensor[rel][curnode, :])[:, 1])
            path.append((rel + len(reldic)//2) % (len(reldic)))    # append to path

        #print ":---------------------"
        # go back
        startnode = random.choice(curnodes)
        pointer = np.zeros((1, len(entdic),), dtype="int8")
        pointer[0, startnode] = 1
        res, pp = self._execute_chain(pointer, path)
        assert(seednode in res)
        return startnode, path, res, pp

    def _execute_chain(self, startpointer, path):
        tensor, entdic, reldic = self._graph.to_tensor(addreverse=True, dense=True)
        red = self._graph._red
        rld = self._graph._rld
        def pp_p(p): return ", ".join(map(lambda x: red[x], list(np.argwhere(p)[:, 1])))
        pointer = startpointer
        pp = pp_p(pointer) + "\n"
        for hoprel in path:
            pp += "->{}->\n".format(rld[hoprel])
            pointer = np.dot(pointer, tensor[hoprel])
            pp += pp_p(pointer) + "\n"
        endnodes = list(np.argwhere(pointer)[:, 1])
        return endnodes, pp

    def _generateConjTree(self, seednode=None, relcred=5, max_branch=2):
        nodes = {seednode: []}
        def isleaf(node): return len(nodes[node]) == 0
        curnode = seednode
        while relcred > 0:
            retry = True
            while retry:
                curnode = random.choice(nodes.keys())
                retry = False
                if not isleaf(curnode):
                    if max_branch > 0:
                        max_branch -= 1
                    else:
                        retry = True
            start = curnode; path = []
            while start in nodes:
                start, path, _, _ = self.generateChain(curnode, min_hops=1, max_hops=1)
            nodes[start] = []
            nodes[curnode].append((start, path))
            relcred -= 1

        def _treedic_to_tree(node):
            if isleaf(node):
                return node, []
            else:
                rets = []
                for s, p in nodes[node]:
                    a, b = _treedic_to_tree(s)
                    rets.append((a, b + p))
                if len(rets) == 1:
                    return rets[0]
                else:
                    return tuple(rets), []

        tree = _treedic_to_tree(seednode)
        #print tree
        return tree

    def generateConjTree(self, seednode=None, relcred=5, max_branch=2):
        tensor, entdic, reldic = self._graph.to_tensor(addreverse=True, dense=False)
        if seednode is None:
            seedtriple = random.choice(self._graph._triples)
            seednode = entdic[seedtriple[0]]
        tree = self._generateConjTree(seednode=seednode, relcred=relcred, max_branch=max_branch)
        res, pp = self._executeConjTree(tree)
        assert(seednode in res)
        return tree, res, pp

    def _pointer_from_indices(self, ind):
        pointer = np.zeros((1, len(self._graph._entdic),), dtype="int8")
        pointer[0, ind] = 1
        return pointer

    def _executeConjTree(self, tree):
        start, path = tree
        if isinstance(start, tuple):    # junction
            lres, lpp = self._executeConjTree(start[0])
            rres, rpp = self._executeConjTree(start[1])
            lpointer = self._pointer_from_indices(lres)
            rpointer = self._pointer_from_indices(rres)
            pointer = lpointer * rpointer       # AND
            pp = "LEFT:\n{} \nRIGHT:\n{} \nJOIN =>\n ".format(lpp, rpp)
        else:
            pointer = self._pointer_from_indices(start)
            pp = ""
        res, ppo = self._execute_chain(pointer, path)
        pp = pp + ppo
        return res, pp

    def _verbalizeConjTree(self, tree, use_aliases=True, indicate_branch=True):
        start, path = tree
        if isinstance(start, tuple):
            lv = self._verbalizeConjTree(start[0])
            rv = self._verbalizeConjTree(start[1])
            if indicate_branch:
                ptrn = "( {} and {} )"
            else:
                ptrn = "{} and {}"
            v = ptrn.format(lv, rv)
        else:
            # get verbalizations of start entity
            name, label, aliases = self._graph.get_lex(self._graph._red[start])
            if use_aliases:
                lex = {name, label}.union(set(aliases))
            else:
                lex = {name, label}
            lex = lex.difference({None})
            v = random.choice(list(lex))
        # verbalize path
        for rel in path:
            lex = self._graph._rellex[self._graph._rld[rel]]
            lex = random.choice(lex)
            v = lex.format(v)
        return v

    def _ptr2set(self, ptr):
        return set(np.argwhere(ptr)[0, :])

    def _idx2ptr(self, idx):
        pointer = np.zeros((1, len(self._graph._entdic),), dtype="int8")
        pointer[0, idx] = 1
        return pointer

    def _idxs2ids(self, *idxs):
        return [self._graph._red[idx] for idx in idxs]

    def _analyzeConjTree(self, tree, debug_replay=False):
        if debug_replay:
            print "debug replaying"
        tensor, entdic, reldic = self._graph.to_tensor(addreverse=True, dense=True)
        start, path = tree
        numbranches = 0
        ambiguous_branches = 0
        if isinstance(start, tuple):
            lptr, ltypes, lnumbranches, lambbr, lstartentities = self._analyzeConjTree(start[0])
            rptr, rtypes, rnumbranches, rambbr, rstartentities = self._analyzeConjTree(start[1])
            #lrestypes = set(self._graph.types_of(*self._idxs2ids(*self._ptr2set(lptr))))
            rrestypes = set(self._graph.types_of(*self._idxs2ids(*self._ptr2set(rptr))))
            ambiguous_branches += lambbr + rambbr
            rettypes = ltypes | rtypes
            ptr = lptr * rptr
            restypes = set(self._graph.types_of(*self._idxs2ids(*self._ptr2set(ptr))))
            if len(rrestypes & ltypes) > 0 or \
                    (lnumbranches > 0 and len(restypes & ltypes) > 0):  # right branch result compatible with some intermediate types from left branch
                ambiguous_branches += 1
            numbranches += lnumbranches + rnumbranches + 1
            startentities = lstartentities | rstartentities
        else:
            ptr = self._idx2ptr(start)
            rettypes = set()
            restypes = set()
            startentities = set(self._idxs2ids(start))
        for rel in path:
            rettypes = rettypes | restypes
            ptr = np.dot(ptr, tensor[rel])
            restypes = set(self._graph.types_of(*self._idxs2ids(*self._ptr2set(ptr))))
        return ptr, rettypes, numbranches, ambiguous_branches, startentities


def gen_questions(number=10000, outp="generated_questions.tsv", verbose=False, write=True):
    g = Graph.from_triples()
    qg = QuestionGenerator(g)
    with open(outp, "w") as f:
        for i in range(1, number+1):
            if i % 100 == 0:
                print "{}\r".format(i)
            relcred = random.choice([1, 2, 3])
            tree, res, pp = qg.generateConjTree(relcred=relcred, max_branch=1)
            vb = qg._verbalizeConjTree(tree)
            _, _, numbranch, ambi, startentities = qg._analyzeConjTree(tree)
            line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i, relcred, numbranch, ambi, vb, ",".join(startentities), ",".join([g._red[x] for x in res]))
            if verbose: print line
            if write: f.write(line)




if __name__ == "__main__":
    gen_questions(10000, verbose=False, write=True)
    #sys.exit()
    g = Graph.from_triples()
    g.save_tensor()
    sys.exit()
    qg = QuestionGenerator(g)
    embed()
    qg._analyzeConjTree(tree)
