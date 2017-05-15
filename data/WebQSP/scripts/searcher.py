from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

client = Elasticsearch(["drogon:9200"])

q = Q({
    "fuzzy": {
        "name": {
            "value": "liv tyler",
            "fuzziness": 0,
        }
    }
})

s = Search(using=client, index="fb_names") \
           .query(q)


i = 0
for hit in s.scan():
    print "{}: [{}]\t - {}".format(hit.meta, hit.mid, hit.name)
    i += 1

print "{} results".format(i)