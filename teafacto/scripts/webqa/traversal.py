import rdflib, re, time
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError, EndPointNotFound
from teafacto.util import iscollection


class Traverser(object):

    rel_blacklist = ["http://rdf\.freebase\.com/key/.+",
                     "http://www\.w3\.org/.+",
                     "http://rdf\.freebase\.com/ns/common\.topic\..+",
                     "http://rdf\.freebase\.com/ns/type\.object\..+",]
    rel_unblacklist = ["http://rdf\.freebase\.com/ns/common\.topic\.notable_types",
                       "http://rdf\.freebase\.com/ns/common\.topic\.notable_for",
                        "http://rdf\.freebase\.com/ns/type\.object\.name"]

    limit = 50

    def __init__(self, address="http://drogon:9890/sparql", **kw):
        super(Traverser, self).__init__(**kw)
        self.sparql = SPARQLWrapper(address)
        self.sparql.setReturnFormat(JSON)

    def name(self, mid):
        q = "SELECT ?o WHERE {{ <{}> <http://rdf.freebase.com/ns/type.object.name> ?o  FILTER (lang(?o) = 'en')}}"\
            .format("http://rdf.freebase.com/ns/{}".format(mid))
        #print q
        ret = set()
        res = self._exec_query(q)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["o"]["value"]
            ret.add(rete)
        return list(ret)[0] if len(ret) > 0 else None

    def get_relations_of(self, mid, only_reverse=False, incl_reverse=False):
        revrels = set()
        if not iscollection(mid):
            mid = (mid,)
        for mide in mid:        # check if mid input is valid
            if not re.match("[a-z]{1,2}\..+", mide):
                return set()
        if incl_reverse:
            revrels = self.get_relations_of(mid, only_reverse=True, incl_reverse=False)
            revrels = {"reverse:{}".format(revrel) for revrel in revrels}
        if only_reverse:
            q = "SELECT DISTINCT(?p) WHERE {{ ?o ?p ?s VALUES ?s {{ {} }} }}"\
                .format(" ".join(["<http://rdf.freebase.com/ns/{}>".format(srce) for srce in mid]))
        else:
            q = "SELECT DISTINCT(?p) WHERE {{ ?s ?p ?o VALUES ?s {{ {} }} }}"\
                .format(" ".join(["<http://rdf.freebase.com/ns/{}>".format(srce) for srce in mid]))
        #print q
        ret = set()
        res = self._exec_query(q)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["p"]["value"]
            toadd = True
            for rel_blacklister in self.rel_blacklist:
                if re.match(rel_blacklister, rete):
                    toadd = False
                    break
            for rel_unblacklister in self.rel_unblacklist:
                if re.match(rel_unblacklister, rete):
                    toadd = True
                    break
            if toadd:
                rete = re.match("http://rdf\.freebase\.com/ns/(.+)", rete).group(1)
                ret.add(rete)
        ret.update(revrels)
        return ret

    def hop(self, src, rel):
        if not iscollection(src):
            src = (src,)
        reverse = False
        if re.match("^reverse:(.+)$", rel):
            reverse = True
            rel = re.match("^reverse:(.+)$", rel).group(1)
        q = "SELECT DISTINCT({}) WHERE {{ ?s <{}> ?o VALUES {} {{ {} }} }} LIMIT {}"\
            .format("?o" if reverse == False else "?s",
                    "http://rdf.freebase.com/ns/{}".format(rel),
                    "?s" if reverse == False else "?o",
                    " ".join(["<http://rdf.freebase.com/ns/{}>".format(srce) for srce in src]),
                    self.limit)
        #print q
        ret = set()
        res = self._exec_query(q)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["o" if reverse is False else "s"]["value"]
            retem = re.match("http://rdf\.freebase\.com/ns/(.+)", rete)
            if not retem:
                ret.add(rete)
            else:
                rete = retem.group(1)
                ret.add(rete)
        return ret

    def _exec_query(self, q):
        self.sparql.setQuery(q)
        retries = 10
        while True:
            try:
                res = self.sparql.query().convert()
                return res
            except (EndPointInternalError, EndPointNotFound), e:
                print "retrying {}".format(retries)
                retries -= 1
                if retries < 0:
                    raise e
                time.sleep(1)

    def join(self, a, b):
        return a & b

    def argmaxmin(self, src, rel, mode="max"):
        assert(iscollection(src))
        reverse = False
        if re.match("^reverse:(.+)$", rel):
            reverse = True
            rel = re.match("^reverse:(.+)$", rel).group(1)
        assert(reverse is True)
        q = "SELECT ?x WHERE {{ ?x <{}> ?v VALUES ?x {{ {} }} }} ORDER BY {}(?v) LIMIT 1"\
            .format("http://rdf.freebase.com/ns/{}".format(rel),
                    " ".join(["<http://rdf.freebase.com/ns/{}>".format(srce) for srce in src]),
                    "DESC" if mode == "max" else "ASC")
        #print q
        ret = set()
        res = self._exec_query(q)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["x"]["value"]
            rete = re.match("http://rdf\.freebase\.com/ns/(.+)", rete).group(1)
            ret.add(rete)
        return ret

    def argmax(self, src, rel):
        return self.argmaxmin(src, rel, mode="max")

    def argmin(self, src, rel):
        return self.argmaxmin(src, rel, mode="min")

    def traverse_tree(self, tree, entdic=None):
        tokens = tree.split()
        ptrstack = []
        i = 0
        argmaxer = None
        result = set()
        validrels = []
        while i < len(tokens):
            token = tokens[i]
            token = entdic[token] if token in entdic else token
            validrelses = set()
            if token[0] == ":":     # relation ==> hop
                if argmaxer is None:
                    mainpointer = self.hop(ptrstack[-1], token[1:])
                    ptrstack[-1] = mainpointer
                    validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
                else:
                    validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
            elif re.match("[a-z]+\..+", token):     # entity ==> make ptr
                ptrstack.append({token})
                validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
            elif token == "ARGMAX":     # do argmax
                assert(argmaxer is None)
                argmaxer = (tokens[i+1][1:], "max")
                #i += 1
                validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
                validrelses = self._revert_rels(validrelses, only_reverse=True)
            elif token == "ARGMIN":
                assert(argmaxer is None)
                argmaxer = (tokens[i+1][1:], "min")
                #i += 1
                validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
                validrelses = self._revert_rels(validrelses, only_reverse=True)
            elif token == "<JOIN>":     # join, execute argmaxers
                if argmaxer is not None:    # ignore auxptr, do argmax
                    mainpointer = self.argmaxmin(ptrstack[-1], argmaxer[0], mode=argmaxer[1])
                    ptrstack[-1] = mainpointer  # replace top of stack
                    argmaxer = None
                else:
                    if len(ptrstack[-1]) == self.limit == len(ptrstack[-2]):
                        pass
                    if len(ptrstack[-1]) == self.limit:
                        ptrstack[-1] = ptrstack[-2]
                    elif len(ptrstack[-2]) == self.limit:
                        pass
                    else:
                        mainpointer = self.join(ptrstack[-1], ptrstack[-2])
                        ptrstack[-1] = mainpointer
                    del ptrstack[-2]
                validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
            elif token == "<RETURN>":
                result = ptrstack[-1]
            else:
                raise Exception("unsupported token: {}".format(token))
            #print [(mptr, self.name(mptr)) for mptr in mainptr]
            validrels.append(validrelses)
            i += 1
        return result, validrels

    def _revert_rels(self, rels, only_reverse=False):
        ret = set()
        for rel in rels:
            m = re.match('^reverse:(.+)$', rel)
            if m:
                if not only_reverse:
                    ret.add(m.group(1))
            else:
                ret.add("reverse:"+rel)
        return ret


if __name__ == "__main__":
    import sys
    t = Traverser()
    #print t.name("m.0gmm518")
    #sys.exit()
    #for x in  t.get_relations_of("m.01vsl3_", incl_reverse=True):
    #    print x
    #print " "
    #for x in t.hop(["m.01vsl3_", "m.06mt91"], "people.person.place_of_birth"):
    #    print x, t.name(x)
    #print " "
    #print t.get_relations_of(t.hop(["m.01vsl3_", "m.06mt91"], "people.person.place_of_birth"))
    #for x in t.argmax(["m.01vsl3_", "m.06mt91"], "people.person.date_of_birth"):
    #    print x, t.name(x)
    #sys.exit()
    #res = t.traverse_tree("<E0> :reverse:film.performance.film <E1> :film.actor.film <JOIN> :film.performance.character <RETURN>", {'<E0>': 'm.017gm7', '<E1>': 'm.02fgm7'})
    #res = t.traverse_tree("<E0> :location.country.currency_used <RETURN>", {'<E0>': 'm.0160w'})
    res, validrels = t.traverse_tree("<E0> :sports.sports_team.championships ARGMAX :reverse:time.event.start_date <JOIN> <RETURN>", {'<E0>': 'm.01yjl'})
    #res, validrels = t.traverse_tree("<E0> :people.person.sibling_s :people.sibling_relationship.sibling m.05zppz :reverse:people.person.gender <JOIN> <RETURN>", {'<E0>': 'm.06w2sn5'})
    for r in res:
        print r, t.name(r)
    for vr in validrels:
        print len(vr), vr
