from IPython import embed
import re
from collections import OrderedDict
import numpy as np
import scipy.sparse as sp
import random


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
    return Graph(triples, names, labels, aliases, entities, relations)


class Graph(object):
    def __init__(self, triples, names, labels, aliases, entities, relations):
        self._triples = triples
        self._names = names
        self._labels = labels
        self._aliases = aliases
        self._entcounts = entities
        self._relcounts = relations
        self._tensor, self._entdic, self._reldic = self._to_tensor()
        self._dense_tensor = None
        self._red = {v: k for k, v in self._entdic.items()}
        self._rld = {v: k for k, v in self._reldic.items()}

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


class QuestionGenerator(object):
    ent_blacklist = {":Type:None"}

    def __init__(self, graph):
        self._graph = graph

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
            relsofnode = list(set(np.argwhere(hasrels[curnode])[:, 1]))
            rel = random.choice(relsofnode)
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
        for hoprel in path[::-1]:
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
                    rets.append((a, p + b))
                if len(rets) == 1:
                    return rets[0]
                else:
                    return tuple(rets), []

        tree = _treedic_to_tree(seednode)
        print tree
        return tree

    def generateConjTree(self, seednode=None, relcred=5):
        tensor, entdic, reldic = self._graph.to_tensor(addreverse=True, dense=False)
        if seednode is None:
            seedtriple = random.choice(self._graph._triples)
            seednode = entdic[seedtriple[0]]
        tree = self._generateConjTree(seednode=seednode, relcred=relcred)
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


if __name__ == "__main__":
    import sys
    g = Graph.from_triples()
    #tensor, entdic, reldic = g.to_tensor(addreverse=True, dense=True)
    qg = QuestionGenerator(g)
    #start, path, result, pp = qg.generateChain(min_hops=1, max_hops=2)
    tree, res, pp = qg.generateConjTree()
    print pp
    sys.exit()
    embed()
