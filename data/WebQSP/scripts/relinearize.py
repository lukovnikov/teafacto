from teafacto.util import argprun
import json


def run(trainp="../data/WebQSP.train.json", testp="../data/WebQSP.test.json"):
    traind = json.load(open(trainp))
    testd = json.load(open(testp))
    traingraphs = buildgraphs(traind)
    testgraphs  = buildgraphs(testd)


def buildgraphs(d):
    # iterate over questions and their parses, output dictionary from question id to parse ids and dictionary of parses
    q2p = {}
    parsegraphs = {}
    multipleparsescount = 0
    for q in d["Questions"]:
        qid = q["QuestionId"]
        parses = q["Parses"]
        if len(parses) > 1:
            multipleparsescount += 1
        parseids = []
        for parse in parses:
            parseid = parse["ParseId"]
            parsegraph = buildgraph(parse)
            parsegraphs[parseid] = parsegraph
            parseids.append(parseid)
        q2p[qid] = parseids
    return q2p, parsegraphs


def buildgraph(parse):
    ret = buildgraph_from_fish(parse)
    return ret


def buildgraph_from_fish(parse):
    # fish head and tail
    qnode = OutputNode()        # tail
    topicentity = EntityNode(parse["TopicEntityMid"], parse["TopicEntityName"]) #head
    # fish spine
    cnode = topicentity
    spinenodes = []
    for i, rel in enumerate(parse["InferentialChain"]):
        tonode = VariableNode() if i < len(parse["InferentialChain"]) - 1 else qnode
        spinenodes.append(tonode)
        cnode.add_edge(rel, tonode)
        cnode = tonode
    # constraints
    for constraint in parse["Constraints"]:
        operator, argtype, arg, name, pos, pred, valtype = constraint["Operator"], constraint["ArgumentType"], constraint["Argument"], constraint["EntityName"], constraint["SourceNodeIndex"], constraint["NodePredicate"], constraint["ValueType"]
        if argtype == "Entity":
            assert(operator == "Equal")
            assert(valtype == "String")
            ent = EntityNode(arg, name)
            edge = RelationEdge(spinenodes[pos], ent, pred)
            spinenodes[pos].append_edge(edge)
            ent.append_edge(edge)
        elif argtype == "Value":
            assert(name == "" or name == None)
            intervar = VariableNode()
            edge = RelationEdge(spinenodes[pos], intervar, pred)
            spinenodes[pos].append_edge(edge)
            intervar.append_edge(intervar)
            if operator == "LessOrEqual":
                rel = "<="
            elif operator == "GreaterOrEqual":
                rel = ">="
            elif operator == "Equal":
                rel = "=="
            else:
                raise Exception("unknown operator")
            val = ValueNode(arg, valuetype=valtype)
            edge = MathEdge(intervar, val, rel)
            intervar.append_edge(edge)
            val.append_edge(edge)
    # order



    return qnode




class Graph(object):
    def __init__(self):
        self._nodes = []


class Node(object):
    def __init__(self):
        self._edges = []

    @property
    def edges(self):
        return self._edges

    def add_edge(self, rel, tonode):
        edg = Edge(self, tonode, rel)
        self._edges.append(edg)
        tonode._edges.append(edg)

    def append_edge(self, edge):
        self._edges.append(edge)

    @property
    def value(self):
        raise NotImplementedError()


class VariableNode(Node):
    def __init__(self, name=None):
        super(VariableNode, self).__init__()
        self._name = name

    @property
    def value(self):
        return self._name


class OrderNode(Node):
    def __init__(self, sort, start, count):
        super(OrderNode, self).__init__()
        self._sort = sort
        self._start = start
        self._count = count

    @property
    def value(self):
        return None


class OutputNode(VariableNode):
    pass


class EntityNode(Node):
    def __init__(self, id, name):
        super(EntityNode, self).__init__()
        self._id = id
        self._name = name

    @property
    def value(self):
        return self._id


class ValueNode(Node):
    def __init__(self, value, valuetype=None):
        super(ValueNode, self).__init__()
        self._value = value
        self._valuetype = valuetype

    @property
    def value(self):
        return self._value


class Edge(object):
    def __init__(self, src, tgt, label):
        self._src = src
        self._tgt = tgt
        self._lbl = label

    @property
    def src(self):
        return self._src

    @property
    def tgt(self):
        return self._tgt

    @property
    def lbl(self):
        return self._lbl


class RelationEdge(Edge):
    pass


class MathEdge(Edge):
    pass


class CountEdge(Edge):
    pass


if __name__ == "__main__":
    argprun(run)