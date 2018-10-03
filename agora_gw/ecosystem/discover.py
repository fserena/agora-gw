"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2018 Fernando Serena
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
"""

import logging
from collections import defaultdict

import networkx as nx
from agora.engine.plan import AGP, TP, find_root_types
from agora.engine.plan.agp import extend_uri
from rdflib import RDF, Variable, BNode

from agora_gw.ecosystem.description import build_component, build_TED

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.gateway.discover')


def extract_bgps(q, cache=None, init_ns={}):
    from agora.graph.evaluate import traverse_part
    from rdflib.plugins.sparql.algebra import translateQuery
    from rdflib.plugins.sparql.parser import parseQuery
    from rdflib.plugins.sparql.sparql import Query

    if cache is not None and q in cache:
        return cache[q]

    if not isinstance(q, Query):
        parsetree = parseQuery(q)
        query = translateQuery(parsetree, None, init_ns)
    else:
        query = q

    part = query.algebra
    filters = {}
    bgps = []

    for p in traverse_part(part, filters):
        bgps.append(p)

    if cache is not None:
        cache[q] = bgps, filters
    return bgps, filters


def build_agp(fountain, bgp):
    agp = AGP([TP(*tp) for tp in bgp.triples], prefixes=fountain.prefixes)
    return agp


def bgp_root_types(fountain, bgp):
    agp = build_agp(fountain, bgp)
    graph = agp.graph
    roots = agp.roots
    str_roots = map(lambda x: str(x), roots)
    root_types = set()
    for c in graph.contexts():
        try:
            root_tps = filter(lambda (s, pr, o): str(s).replace('?', '') in str_roots, c.triples((None, None, None)))
            context_root_types = find_root_types(fountain, root_tps, c, extend=False).values()
            root_types.update(reduce(lambda x, y: x.union(y), context_root_types, set()))
        except TypeError:
            pass

    return root_types


def query_root_types(fountain, q, bgp_cache=None):
    types = reduce(lambda x, y: x.union(bgp_root_types(fountain, y)),
                   extract_bgps(q, cache=bgp_cache, init_ns=fountain.prefixes)[0], set())
    desc_types = describe_types(fountain, types)
    return keep_general(desc_types)


def keep_general(types):
    ids = set(types.keys())
    filtered_ids = filter(lambda x: not set.intersection(set(types[x]['super']), ids), types)
    return {fid: types[fid] for fid in filtered_ids}


def keep_specific(types):
    ids = set(types.keys())
    filtered_ids = filter(lambda x: (types[x]['super'] and not set.intersection(set(types[x]['super']), ids))
                                    or not set.intersection(set(types[x]['sub']), ids), types)
    return {fid: types[fid] for fid in filtered_ids}


def describe_type(fountain, t):
    desc = fountain.get_type(t)
    desc['id'] = t
    return desc


def describe_types(fountain, types):
    return {t: describe_type(fountain, t) for t in types}


def is_subclass_of_thing(t):
    return t['id'] == 'wot:Thing' or 'wot:Thing' in t['super']


def tuple_from_result_row(row):
    return tuple([row[var]['value'] for var in row])


def generate_dict(n3, ns, res):
    d = defaultdict(set)
    for k, v in map(lambda x: tuple_from_result_row(x), res):
        d[k].add(n3(v, ns))
    return d


def contains_solutions(R, id, query, bgp_cache=None):
    result = True
    queries = list(transform_into_specific_queries(R, id, query, bgp_cache=bgp_cache))
    for sub_query in queries:
        result = result and bool(map(lambda r: r, R.query(sub_query, cache=True, expire=300)))
        if not result:
            break
    return result


def is_semantically_reachable(fountain, source_types, target, cache=None):
    for st in source_types:
        if cache and (st, target) in cache:
            return cache[(st, target)]

        connected = fountain.connected(st, target)
        if cache is not None:
            cache[(st, target)] = connected
        if connected:
            return True

    return False


def is_described_reachable(fountain, R, td_network, seed, type):
    res = R.query("""
    PREFIX core: <http://iot.linkeddata.es/def/core#>
    SELECT DISTINCT ?g ?id WHERE {
        GRAPH ?g {
           [] a core:ThingDescription ;
              core:identifier ?id ;
              core:describes <%s>
        }
    }
    """ % seed, cache=True, expire=300, infer=True)
    rd = generate_dict(R.n3, R.ns(fountain=fountain), res)
    try:
        td_id = rd.keys().pop()
    except IndexError:
        return True  # what if the seed does not correspond to a described (TD-based) thing?

    type_uri = extend_uri(type, fountain.prefixes)
    res = R.query("""
    PREFIX core: <http://iot.linkeddata.es/def/core#>
    SELECT DISTINCT ?id WHERE {    
        [] a core:ThingDescription ;
           core:identifier ?id ;
           core:describes [
            a <%s>
        ]                
    }
    """ % type_uri, cache=True, expire=300, infer=True)
    target_tds = map(lambda x: x['id']['value'], res)
    for target in target_tds:
        try:
            if nx.shortest_path(td_network, td_id, target):
                return True
        except nx.NetworkXNoPath:
            pass

    return False


def infer_types(types, fountain):
    return set.union(*map(lambda t: set.union(set(fountain.get_type(t)['super']), types), types))


def get_type_dicts(types, fountain, infer=True):
    if infer:
        type_ids = infer_types(types, fountain)
    return {t: fountain.get_type(t) for t in type_ids}


def search_things(R, type, q, td_network, reachability=True, reachability_cache=None, bgp_cache=None, fountain=None):
    res = R.query("""
       prefix core: <http://iot.linkeddata.es/def/core#>
       prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> 
       SELECT DISTINCT * WHERE {
           {
              [] a core:Ecosystem ;
                 core:hasComponent ?s
           }
           UNION
           {
              [] a core:ThingDescription ;
                 core:describes ?s
           }
           ?s a ?type             
           FILTER(isURI(?type) && isURI(?s) && ?type != rdfs:Resource)
       }
       """, cache=True, expire=300, infer=True)

    if not fountain:
        fountain = R.fountain

    rd = generate_dict(R.n3, R.ns(fountain=fountain), res)
    type_n3 = type['id']
    all_types = fountain.types

    if reachability_cache is None:
        reachability_cache = {}

    for seed, type_ids in rd.items():
        try:
            types = get_type_dicts([t for t in type_ids if t in all_types], fountain, infer=True)
            if types and (type_n3 in types or is_semantically_reachable(fountain, types.keys(), type_n3,
                                                                        cache=reachability_cache) and is_described_reachable(
                fountain, R, td_network, seed, type_n3)):
                rd[seed] = types
            else:
                del rd[seed]
        except TypeError:
            del rd[seed]

    final_rd = {}
    for seed in rd:
        if reachability or contains_solutions(R, seed, q, bgp_cache=bgp_cache):
            final_rd[seed] = rd[seed]

    return final_rd


def fix_tp_serialization(tp, tp_str, bnode_map={}):
    s, p, o = tp

    fix = tp_str
    for node in [s, o]:
        if isinstance(node, BNode) and node not in bnode_map:
            bnode_map[node] = 'b' + str(len(bnode_map))

        node_str = str(node)
        fix = fix.replace(node_str, bnode_map.get(node, node_str))
    return fix


def make_up_bgp_query(q, predicate_mask, bgp_cache=None):
    bgps, filters = extract_bgps(q, cache=bgp_cache)
    tp_fix_map = {}
    for bgp in bgps:
        desc_tps = filter(lambda tp: tp[1] in predicate_mask or tp[1] == RDF.type, bgp.triples)
        bgp_vars = filter(lambda part: isinstance(part, Variable),
                          reduce(lambda x, y: x.union(list(y)), bgp.triples, set()))

        bgp_filters = reduce(lambda x, y: x.union(filters[y]), [v for v in bgp_vars if v in filters], set())
        filter_clause = 'FILTER(%s)' % ' && '.join(bgp_filters) if bgp_filters else ''

        if desc_tps:
            tps_str = ' .\n'.join([fix_tp_serialization(tp, str(TP(*tp)), tp_fix_map) for tp in desc_tps])
            yield tps_str, filter_clause


def transform_into_specific_queries(R, id, q, bgp_cache=None):
    desc_predicates = R.thing_describing_predicates(id)

    td_q = """SELECT DISTINCT * WHERE { GRAPH <%s> { %s  %s } }"""
    for tps_str, filter_clause in make_up_bgp_query(q, desc_predicates, bgp_cache=bgp_cache):
        yield td_q % (id, tps_str, filter_clause)


def transform_into_graph_td_queries(R, q, bgp_cache=None):
    desc_predicates = R.describing_predicates

    td_q = """SELECT DISTINCT ?g { GRAPH ?g { %s  %s } }"""
    for tps_str, filter_clause in make_up_bgp_query(q, desc_predicates, bgp_cache=bgp_cache):
        yield td_q % (tps_str, filter_clause)


def discover_ecosystem(R, VTED, q, reachability=False, lazy=False):
    bgp_cache = {}

    fountain = R.fountain

    # 1. Get all BPG root types
    root_types = query_root_types(fountain, q, bgp_cache=bgp_cache)

    if not root_types:
        raise AttributeError('Could not understand the given query')

    log.debug('Triggered discovery for \n{}'.format(q))
    log.debug('Query root types: {}'.format(root_types.keys()))

    # 2. Find relevant things for identified root types
    log.debug('Searching for relevant things...')

    reachability_cache = {}
    typed_things = {
        type['id']: search_things(R, type, q, VTED.network,
                                  reachability=reachability,
                                  fountain=fountain,
                                  reachability_cache=reachability_cache,
                                  bgp_cache=bgp_cache) for type in root_types.values()}
    log.debug('Found things of different types: {}'.format(typed_things.keys()))

    # 2b. Filter seeds
    log.debug('Analyzing relevant things...')
    graph_td_queries = list(transform_into_graph_td_queries(R, q, bgp_cache=bgp_cache))
    query_matching_things = set()
    for q in graph_td_queries:
        graphs = map(lambda r: r['g']['value'], R.query(q, cache=True, expire=300))
        query_matching_things.update(set(graphs))

    root_thing_ids = reduce(lambda x, y: x.union(y), typed_things.values(), set())
    root_things = root_thing_ids
    if graph_td_queries:
        root_things = set.intersection(query_matching_things, root_thing_ids)

    log.debug('Discovered {} root things!'.format(len(root_things)))

    # 3. Retrieve/Build ecosystem TDs
    log.debug('Preparing TDs for the discovered ecosystem...')
    node_map = {}
    components = {root: list(build_component(R, VTED, root, node_map=node_map, lazy=lazy)) for root in root_things}

    # 4. Compose ecosystem description
    log.debug('Building TED of the discovered ecosystem...')
    ted = build_TED(components.values())

    return ted
