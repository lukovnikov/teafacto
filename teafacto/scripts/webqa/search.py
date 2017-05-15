from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
import re, nltk, numpy as np
from teafacto.util import ticktock
from SPARQLWrapper import SPARQLWrapper, JSON
from IPython import embed


class Searcher(object):
    def __init__(self, host="drogon", port=9200, index="fb_labels", verbose=False):
        self.client = Elasticsearch(["{}:{}".format(host, port)])
        self.index_name = index
        self.tt = ticktock("searcher", verbose=verbose)

    def search_term(self, x, _fuzziness=0, leastfuzzy=True,
                    maxnumres=np.infty, minnumres=0, maxfuzziness=1,
                    _res=None):
        self.tt.tick("searching for \"{}\"~{} | {}".format(x, _fuzziness, maxfuzziness))
        if maxfuzziness < 0:
            self.tt.tock("not searching because maxfuzziness is {}".format(maxfuzziness))
            return {} if _res is None else _res
        assert(maxnumres >= minnumres)
        assert(_fuzziness <= maxfuzziness)
        x = " ".join(self.get_words(x))
        q = Q({
            "fuzzy": {
                "name": {
                    "value": x,
                    "fuzziness": _fuzziness,
                }
            }
        })
        s = Search(using=self.client, index=self.index_name).query(q)
        res = {} if _res is None else _res
        for hit in s.scan():
            if len(res) >= maxnumres:
                break
            if hit.mid in res:
                continue
            res[hit.mid] = (hit.name, hit.oname, _fuzziness)
        localnumres = len(res)
        searchnextfuzzy = (_fuzziness < maxfuzziness) and \
                          (
                              (not (leastfuzzy and len(res) > 0))
                              or
                              (len(res) < minnumres)
                          )
        if searchnextfuzzy:
            res = self.search_term(x, _fuzziness=_fuzziness+1, maxfuzziness=maxfuzziness,
                                   maxnumres=maxnumres, minnumres=minnumres, leastfuzzy=leastfuzzy,
                                   _res=res)
        self.tt.tock("done searching for \"{}\"~{}, {}({}) results".format(x, _fuzziness, localnumres, len(res)))
        return res

    def search_sentence(self, x, uptongram=4, return_positions=True,
                        leastfuzzy=True, maxnumres=np.infty, minnumres=0,
                        maxfuzziness=2, discard_subgrams=False):
        x = self.get_words(x)
        BIG = 9999
        burns = np.ones((len(x),), dtype="int32") * BIG
        ret = {}
        fro = 0
        discard_offset = 0
        if discard_subgrams:
            discard_offset = 1
        while fro < len(x):
            to = min(len(x), fro + uptongram)
            while to > fro:
                _maxfuzziness = min(maxfuzziness, int(np.max(burns[fro:to]))-discard_offset)
                termquery = " ".join(x[fro:to])
                results = self.search_term(termquery, leastfuzzy=leastfuzzy, maxnumres=maxnumres, minnumres=minnumres,
                                           maxfuzziness=_maxfuzziness)
                leastfuzziness = BIG
                for mid, val in results.items():
                    leastfuzziness = min(leastfuzziness, val[2])
                    if not return_positions:
                        ret[mid] = val
                    else:
                        try:
                            existing_res = ret[mid]
                        except KeyError, e:
                            ret[mid] = val + ([],)
                            existing_res = ret[mid]
                        existing_res[3].append((fro, to))
                if len(results) > 0:   # burn range
                    burns[fro:to] = np.minimum([leastfuzziness]*(to-fro), burns[fro:to])
                    #print burns
                to -= 1
            fro += 1
        return ret

    def search_mid(self, mid):
        q = Q("term", mid=mid)
        s = Search(using=self.client, index=self.index_name).query(q)
        res = set()
        for hit in s.scan():
            res.add((hit.name, hit.type))
        return res

    def get_words(self, x):
        return nltk.word_tokenize(re.sub("[^\w\d]", " ", x.lower()))


class EntityInfoGetter(object):
    def __init__(self, host="drogon", es_port=9200, index="fb_labels",
                 sparql_port=9890, **kw):
        super(EntityInfoGetter, self).__init__(**kw)
        self.searcher = Searcher(host=host, port=es_port, index=index)
        self.sparqler = SPARQLWrapper("http://{}:{}/sparql".format(host, sparql_port))
        self.sparqler.setReturnFormat(JSON)

    def get_info(self, *mids):
        entities = ",".join(
                        map(lambda x: "<http://rdf.freebase.com/ns/{}>".format(x),
                            mids))
        qstring = """SELECT * {
                        {SELECT ?s ?p ?o {
                            ?s ?p ?o.
                            FILTER (?p IN (<http://rdf.freebase.com/ns/type.object.type>,
                                            <http://rdf.freebase.com/ns/common.topic.notable_types>,
                                            <http://rdf.freebase.com/ns/common.topic.description>)) }}
                        UNION
                        {SELECT ?s, ?p, count(?r) as ?o {
                            ?s ?r ?o.
                            BIND("outedgecount" AS ?p).             }}
                        UNION
                        {SELECT ?s, ?p, count(?r) as ?o {
                            ?o ?r ?s.
                            BIND("inedgecount" AS ?p).              }}
                        UNION
                        {SELECT DISTINCT ?s, ?p, ?r as ?o {
                            ?s ?r ?x.
                            BIND("outrel" AS ?p)                    }}
                        UNION
                        {SELECT DISTINCT ?s, ?p, ?r as ?o {
                            ?x ?r ?s.
                            BIND("inrel" AS ?p)                     }}
                        FILTER (lang(?o) = 'en' OR isIri(?o) OR isNumeric(?o)).
                        FILTER (?s IN (""" + entities + """)).
                    }"""
        self.sparqler.setQuery(qstring)
        results = self.sparqler.query().convert()

        ret = dict(zip(mids, [EntityInfo()] * len(mids)))
        entre = re.compile("^http://rdf\.freebase\.com/ns/(.+)$")

        for result in results["results"]["bindings"]:
            s, p, o = map(lambda x: entre.match(x["value"]).group(1) if x["type"] == "uri" and entre.match(x["value"]) else x["value"],
                          (result["s"], result["p"], result["o"]))
            ret[s][p].append(o)
        for mid in mids:
            name = None
            aliases = []
            for res in self.searcher.search_mid(mid):
                if res[1] == "alias":
                    aliases.append(res[0])
                elif res[1] == "name":
                    #assert(name == None)
                    name = res[0]
                else:
                    raise Exception("unknown label type")
            ret[mid].name = name
            ret[mid].aliases = aliases
            ret[mid].mid = mid
        return ret


class EntityInfo(object):
    __slots__ = ("mid", "types", "notable_types", "aliases", "name", "description",
                 "numoutedges", "numinedges", "outrels", "inrels")

    def __init__(self):
        for slot in self.__slots__:
            setattr(self, slot, [])

    mapper = {
            "type.object.type": "types",
            "common.topic.notable_types": "notable_types",
            "common.topic.description": "description",
            "mid": "mid",
            "name": "name",
            "aliases": "aliases",
            "inrel": "inrels",
            "outrel": "outrels",
            "outedgecount": "numoutedges",
            "inedgecount": "numinedges",
        }

    def __setitem__(self, key, value):
        setattr(self, EntityInfo.mapper[key], value)

    def __getitem__(self, item):
        return getattr(self, EntityInfo.mapper[item])

    def finalize(self):
        for slot in self.__slots__:
            slotvalue = getattr(self, slot)
            if len(slotvalue) > 1:
                setattr(self, slot, tuple(slotvalue))
            elif len(slotvalue) == 1:
                setattr(self, slot, slotvalue[0])
            else:
                setattr(self, slot, None)
        return self


if __name__ == "__main__":
    info = EntityInfoGetter()
    ret = info.get_info("m.0kpv1m")
    embed()


    s = Searcher(verbose=False)
    tt = ticktock("script")
    def printres(x):
        mids = set()
        for ret in x:
            print "{}:\t {}".format(ret, x[ret][0])
            mids.add(ret)
        print "{} results".format(len(x))
        return mids

    """res = s.search_sentence("who is barack husseini obama", leastfuzzy=True, maxfuzziness=1)
    printres(res)

    res = s.search_term("barack obama")
    mids1 = printres(res)
    res = s.search_term("barack hussein obama")
    mids2 = printres(res)
    print mids1.intersection(mids2)"""
    tt.tick("searching")
    res = s.search_sentence("who does amy stiller play on dodgeball",
                            maxnumres=100, leastfuzzy=True)
    printres(res)
    tt.tock("found {} results".format(len(res)))

