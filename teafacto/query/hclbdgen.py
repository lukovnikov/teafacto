import re, random


class LbdGen(object):
    def __init__(self):
        self.types = {}
        self.symbols = {}
        self._entities = set()
        self._relations = set()
        self._relspertyp_subj_super = {}
        self._relspertyp_obj_super = {}
        self._relspertyp_subj_sub = {}
        self._relspertyp_obj_sub = {}

    def _post_proc(self):
        # separate symbols in entities and relations
        for sym, sim in self.symbols.items():
            if isinstance(sim.typ, EntityType):
                self._entities.add(sym)
            elif isinstance(sim.typ, RelationType):
                self._relations.add(sym)
        # from type name to suitable relations with type as subj
        for typ, tip in self.types.items():
            if not isinstance(tip, EntityType):
                continue
            supertyps = set()
            supertyps.add(tip)
            subtyps = set()
            subtyps.add(tip)
            cet = tip
            while cet.supertype is not None:
                supertyps.add(cet.supertype)
                cet = cet.supertype
            change = True
            while change:
                change = False
                for typ2, tip2 in self.types.items():
                    if not isinstance(tip2, EntityType):
                        continue
                    if tip2.supertype in subtyps:
                        if tip2 not in subtyps:
                            subtyps.add(tip2)
                            change = True
            acc_s_super = set()
            acc_o_super = set()
            acc_s_sub = set()
            acc_o_sub = set()
            for rn in self._relations:
                r = self.symbols[rn]
                if len(set(r.typ.fro).intersection(supertyps)) > 0:
                    acc_s_super.add(r.name)
                if len(set(r.typ.to).intersection(supertyps)) > 0:
                    acc_o_super.add(r.name)
                if len(set(r.typ.fro).intersection(subtyps)) > 0:
                    acc_s_sub.add(r.name)
                if len(set(r.typ.to).intersection(subtyps)) > 0:
                    acc_o_sub.add(r.name)
            self._relspertyp_subj_super[typ] = acc_s_super
            self._relspertyp_obj_super[typ] = acc_o_super
            self._relspertyp_subj_sub[typ] = acc_s_sub
            self._relspertyp_obj_sub[typ] = acc_o_sub

    def _relsbytyp(self, sts, ots):
        if isinstance(sts, Super):
            srs = self._relspertyp_subj_super[sts.typename]
        elif isinstance(sts, Sub):
            srs = self._relspertyp_subj_sub[sts.typename]
        else:
            srs = self._relspertyp_subj_super[sts].intersection(self._relspertyp_subj_sub[sts])
        if isinstance(ots, Super):
            ors = self._relspertyp_obj_super[ots.typename]
        elif isinstance(ots, Sub):
            ors = self._relspertyp_obj_sub[ots.typename]
        else:
            ors = self._relspertyp_obj_super[ots].intersection(self._relspertyp_obj_sub[ots])
        return srs.intersection(ors)

    def _parse_loop(self, d, f):
        # rules must be of form HEAD -> BODY | BODY | ... ;
        ss = d.split("\n")
        accu = ""
        for split in ss:
            m = re.match("^([^#]*)(#.+)?$", split)
            t = m.group(1)
            t = t.rstrip(" \t").lstrip(" \t")
            accu += t
            if len(t) > 0 and t[-1] == ";":
                f(accu)
                accu = ""

    def parse_info(self, s):
        g = self
        def f(accu):
            accu = accu.rstrip(" ;\t").lstrip(" \t")
            splits = accu.split("<")
            head = splits[0].rstrip(" \t").lstrip(" \t")
            body = splits[1]
            if body[0] == "=":  # type expression
                body = body[1:]
                headsym = g[head]
                t = g.get_type(body)
                assert(isinstance(headsym, Symbol))
                headsym.typ = t
            else:               # subtype expression
                t = g.get_type(body)
                h = g.get_type(head)
                h.supertype = t
        self._parse_loop(s, f)
        self._post_proc()

    def __getitem__(self, item):  # item is string, symbol name
        if item not in self.symbols:
            ret = Symbol(item)
            self.add_symbol(ret)
        return self.symbols[item]

    def add_symbol(self, sym):
        if sym.name not in self.symbols:
            self.symbols[sym.name] = sym
        else:
            raise KeyError("symbol already in grammar")

    def get_type(self, s):
        s = s.lstrip(" \t").rstrip(" \t")
        ss = s.split("->")
        if len(ss) == 1:  # unary type
            if s not in self.types:
                self.types[s] = EntityType(s)
            return self.types[s]
        elif len(ss) == 2:  # binary type
            fro = map(lambda x: self.get_type(x.rstrip(" ").lstrip(" ")), ss[0].lstrip(" ").rstrip(" ").split("|"))
            to = map(lambda x: self.get_type(x.rstrip(" ").lstrip(" ")), ss[1].lstrip(" ").rstrip(" ").split("|"))
            if s not in self.types:
                self.types[s] = RelationType(fro, to)
            return self.types[s]
        else:
            raise NotImplementedError("higher-order types not supported")

    def gen_prop_sel(self):     # generates property selection expression
        # sample entity
        en = random.sample(self._entities, 1)[0]
        e = self.symbols[en]
        # get entity type
        et = e.typ
        # sample matching property
        rns = self._relsbytyp(Super(et.name), Sub("lit"))
        rn = random.sample(rns, 1)[0]
        r = self.symbols[rn]
        return "( {} {} )".format(rn, en)

    def gen_spl_lbd(self):
        # sample entity
        en = random.sample(self._entities, 1)[0]
        e = self.symbols[en]
        et = e.typ
        if random.getrandbits(1) == True:
            rns = self._relsbytyp(Super(et.name), Sub("ent"))
            rn = random.sample(rns, 1)[0]
            r = self.symbols[rn]
            return "( the ${0} ( {1} {2} ${0} ) )".format(0, rn, en)
        else:
            rns = self._relsbytyp(Sub("ent"), Super(et.name))
            rn = random.sample(rns, 1)[0]
            r = self.symbols[rn]
            return "( the ${0} ( {1} ${0} {2}) )".format(0, rn, en)

    def gen_tpd_spl_lbd(self):  #TODO: swap subj and obj randomly
        en = random.sample(self._entities, 1)[0]
        e = self.symbols[en]
        et = e.typ
        rns = self._relsbytyp(Super(et.name), Sub("ent"))
        rn = random.sample(rns, 1)[0]
        r = self.symbols[rn]
        vartyp = random.sample(r.typ.to, 1)[0]
        ret = "( the ${0} ( and ( {1} ${0} ) ( {2} {3} ${0} ) ) )"\
            .format(0, vartyp.name, rn, en)
        return ret

    def gen_spl_cnt(self):  # TODO: swap subj and obj randomly
        en = random.sample(self._entities, 1)[0]
        e = self.symbols[en]
        et = e.typ
        rns = self._relsbytyp(Super(et.name), Sub("ent"))
        rn = random.sample(rns, 1)[0]
        r = self.symbols[rn]
        vartyp = random.sample(r.typ.to, 1)[0]
        ret = "( count ${0} ( and ( {1} ${0} ) ( {2} {3} ${0} ) ) )"\
            .format(0, vartyp.name, rn, en)
        return ret

    def gen_spl_ent_argopt(self):
        argwopt = "argmax" if random.getrandbits(1) == True else "argmin"
        en = random.sample(self._entities, 1)[0]
        e = self.symbols[en]
        et = e.typ
        rns = self._relsbytyp(Super(et.name), Sub("ent"))
        rn = random.sample(rns, 1)[0]
        r = self.symbols[rn]
        vartyp = random.sample(r.typ.to, 1)[0]
        rnsi = self._relsbytyp(Super(vartyp.name), Sub("lit"))
        rni = random.sample(rnsi, 1)[0]
        ri = self.symbols[rni]
        ret = "( {5} ${0} ( and ( {1} ${0} ) ( {2} {3} ${0} ) ) ( {4} ${0} ) )"\
            .format(0, vartyp.name, rn, en, rni, argwopt)
        return ret

    def gen_spl_typ_argopt(self):
        argwopt = "argmax" if random.getrandbits(1) == True else "argmin"
        vartypn = random.sample(self.types.keys(), 1)[0]
        vartyp = self.types[vartypn]
        rns = self._relsbytyp(Super(vartypn), Sub("lit"))
        rn = random.sample(rns, 1)[0]
        r = self.symbols[rn]
        ret = "( {3} ${0} ( {1} ${0} ) ( {2} ${0} ) )"\
            .format(0, vartyp.name, rn, argwopt)
        return ret

    def gen_ent_singleseed_chain(self):
        # sample seed entity at random
        en = random.sample(self.gen_tpd_spl_lbd())
        e = self.symbols[en]
        et = e.typ
        if random.getrandbits(1) == True:
            rn = self._relsbytyp(Super(et.name), Sub("lit"))
        else:
            rn = self._relsbytyp(Super(et.name), Sub("ent"))




class TypeSel(object):
    def __init__(self, typename):
        self.typename = typename

class Sub(TypeSel): pass
class Super(TypeSel): pass


class Symbol(object):
    def __init__(self, name):
        self.name = name
        self.typ = None

    def __repr__(self):
        return self.name + "-"

    def __str__(self):
        return self.name


class SymbolType(object):
    pass


class EntityType(SymbolType):
    def __init__(self, name):
        self.name = name
        self.supertype = None

    def __repr__(self):
        return self.name


class RelationType(SymbolType):
    def __init__(self, fro, to):
        self.fro = fro
        self.to = to

    def __repr__(self):
        return "|".join(map(str, self.fro)) + " -> " + "|".join(map(str, self.to))


if __name__ == "__main__":
    typeinfo = """
        # entities
        ohio:s <= state:t ;
        washington:s <= state:t ;
        seattle:c <= city:t ;
        usa:co <= country:t ;
        # relations
        capital:c <= state:t | country:t -> city:t ;
        population:i <= country:t | state:t | city:t -> num ;
        next_to:t <= state:t -> state:t ;
        loc:t <= place:t -> place:t ;
        size:i <= state:t | city:t -> num ;
        # subtypes
        city:t < place:t ;
        state:t < place:t ;
        country:t < place:t ;
        place:t < ent ;
        int < num ;
        float < num ;
        num < lit ;
        ent < any ;
        lit < any ;
    """
    # generate lambda expressions from type info
    g = LbdGen()
    g.parse_info(typeinfo)
    print g.symbols
    print g.types
    print g._entities

    print g.gen_prop_sel()
    print g.gen_spl_lbd()
    print g.gen_tpd_spl_lbd()
    print g.gen_spl_cnt()
    print g.gen_spl_ent_argopt()
    print g.gen_spl_typ_argopt()
