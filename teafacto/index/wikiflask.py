from flask import Flask, jsonify
from teafacto.index.wikipedia import WikipediaIndex

def run():
    wi = WikipediaIndex()
    app = Flask(__name__)

    @app.route("/search/<q>")
    def searchd(q):
        return search(q, 20)

    @app.route("/search/<q>/<l>")
    def search(q, l):
        r = wi.search(q, limit=int(l))
        return jsonify({"q": q, "r": r, "n": len(r)})

    app.run()


if __name__ == "__main__":
    run()