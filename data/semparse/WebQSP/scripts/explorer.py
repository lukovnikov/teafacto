import json
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError
from IPython import embed
import re

def gothrough(p="../data/WebQSP.train.json"):
    with open(p) as f:
        data = json.load(f)
        print "{} questions".format(len(data["Questions"]))
        morethanoneparse = []
        noinfchain = []
        manualsparql = []
        mentionnotfound = []
        toomanymentions = []
        nullmention = []
        cleanqs = []
        for q in data["Questions"]:
            clean = True
            if len(q["Parses"]) > 1:
                morethanoneparse.append(q)
                clean = False
            if q["Parses"][0]["InferentialChain"] == None:
                noinfchain.append(q)
                clean = False
            if q["Parses"][0]["Sparql"][:len("#MANUAL SPARQL")] == "#MANUAL SPARQL":
                manualsparql.append(q)
                clean = False
            mention = q["Parses"][0]["PotentialTopicEntityMention"]
            if mention is not None:
                #mention = re.sub("(\w)\s\.", lambda x: x.group(1)+".", mention)
                if q["ProcessedQuestion"].count(mention) > 1:
                    toomanymentions.append(q)
                    clean = False
                elif q["ProcessedQuestion"].count(mention) < 1:
                    mentionnotfound.append(q)
                    clean = False
            else:
                nullmention.append(q)
                clean = False
            if clean:
                cleanqs.append(q)

        print "more than one parse: {}".format(len(morethanoneparse))
        print "no inferential chain: {}".format(len(noinfchain))
        print "manual sparql: {}".format(len(manualsparql))
        print "too many mentions: {}".format(len(toomanymentions))
        print "null mention: {}".format(len(nullmention))
        print "mention not found: {}".format(len(mentionnotfound))
        print "clean: {}".format(len(cleanqs))
        embed()


def testresponses(p="../data/WebQSP.train.json", endpoint="http://drogon:9890/sparql"):
    query = SPARQLWrapper(endpoint)
    query.setReturnFormat(JSON)
    recall = 0
    precision = 0
    c = 0
    with open(p) as f:
        data = json.load(f)
        morethanoneparse = []
        for q in data["Questions"]:
            if len(q["Parses"]) > 1:
                morethanoneparse.append(q)
            else:
                if c % 100 == 0:
                    print c
                try:
                    query.setQuery(q["Parses"][0]["Sparql"])
                    results = query.query().convert()
                    results = results["results"]["bindings"]
                    retrieved = set([x['x']['value'][len("http://rdf.freebase.com/ns/"):] for x in results])
                    given = set([x["AnswerArgument"] for x in q["Parses"][0]["Answers"]])
                    recall += len(given.intersection(retrieved)) * 1. / len(given)
                    precision += len(retrieved.intersection(given)) * 1. / len(retrieved)
                    c += 1
                except ZeroDivisionError:
                    print c, "ZeroDivision", q["RawQuestion"], q["Parses"][0]
                except KeyError:
                    print c, "KeyError", q["RawQuestion"]
                except EndPointInternalError, e:
                    print c, "EndPointInternalError"
        recall /= c
        precision /= c
        print "recall {}\n precision {}".format(recall, precision)


if __name__ == "__main__":
    #testresponses()
    gothrough()