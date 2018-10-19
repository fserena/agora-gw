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

import calendar
import logging
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Lock

import networkx as nx
from agora.engine.plan.agp import extend_uri
from agora_gw.data.graph import canonize_node
from agora_gw.data.repository import CORE
from agora_wot.blocks.eco import request_loader, Ecosystem
from agora_wot.blocks.resource import Resource
from agora_wot.blocks.td import TD
from agora_wot.blocks.ted import TED
from rdflib import BNode, URIRef, Graph, RDF, RDFS
from redis_cache import cache_it, SimpleCache

__author__ = 'Fernando Serena'

META = {'network': None, 'roots': None, 'ts': None}
lock = Lock()

log = logging.getLogger('agora.gateway.description')

QUERY_CACHE_HOST = os.environ.get('QUERY_CACHE_HOST', 'localhost')
QUERY_CACHE_NUMBER = int(os.environ.get('QUERY_CACHE_NUMBER', 8))

cache = SimpleCache(limit=10000, expire=60 * 60, hashkeys=True, host=QUERY_CACHE_HOST, port=6379,
                    db=QUERY_CACHE_NUMBER, namespace='desc')


def get_context(R):
    def wrapper(uri):
        g = R.pull(uri, cache=True, infer=False, expire=300)
        if not g:
            g = request_loader(uri)
        return g

    return wrapper


def get_td_nodes(R, cache=True):
    res = R.query("""
    PREFIX core: <http://iot.linkeddata.es/def/core#>
    SELECT DISTINCT ?td WHERE {
        ?td a core:ThingDescription
    }""", cache=cache, infer=False, expire=300)

    return map(lambda r: r['td']['value'], res)


def get_td_node(R, id):
    res = R.query("""
    PREFIX core: <http://iot.linkeddata.es/def/core#>
    SELECT ?td WHERE {
        ?td a core:ThingDescription ;
            core:identifier ?id
        FILTER(STR(?id)="%s")
    }""" % str(id))

    try:
        return URIRef(res.pop()['td']['value'])
    except IndexError:
        log.warn('No TD for identifier {}'.format(id))
        raise AttributeError(id)


def get_th_node(R, id):
    res = R.query("""
    PREFIX core: <http://iot.linkeddata.es/def/core#>
    SELECT ?th WHERE {
        [] a core:ThingDescription ;
            core:identifier ?id ;
            core:describes ?th
        FILTER(STR(?id)="%s")
    }""" % str(id))

    try:
        return URIRef(res.pop()['th']['value'])
    except IndexError:
        log.warn('No thing for TD identifier {}'.format(id))
        raise AttributeError(id)


def get_th_nodes(R, cache=True):
    res = R.query("""
       PREFIX core: <http://iot.linkeddata.es/def/core#>
       SELECT DISTINCT ?th WHERE {
           [] a core:ThingDescription ;
               core:describes ?th
       }""", cache=cache, infer=False, expire=300)

    return map(lambda r: r['th']['value'], res)


def is_root(R, th_uri):
    res = R.query("""
    PREFIX core: <http://iot.linkeddata.es/def/core#>
    ASK {
        [] a core:ThingEcosystemDescription ;
           core:describes [
              core:hasComponent <%s>
           ]               
    }""" % th_uri, cache=True, infer=False, expire=300)
    return res


def create_TD_from(R, td_uri, node_map, lazy=True, **kwargs):
    td_uri = URIRef(td_uri)

    if td_uri in node_map:
        return node_map[td_uri]

    log.debug('Creating TD for {}...'.format(td_uri))
    th_uri = get_td_thing(R, td_uri)
    g = R.pull(th_uri, cache=True, infer=False, expire=300)
    g.__iadd__(R.pull(td_uri, cache=True, infer=False, expire=300))

    return TD.from_graph(g, node=URIRef(td_uri), node_map=node_map, fetch=not lazy, **kwargs)


def get_matching_TD(R, th_uri, node_map={}, **kwargs):
    res = R.query("""
    PREFIX core: <http://iot.linkeddata.es/def/core#>
    SELECT DISTINCT ?g WHERE {
        GRAPH ?g {
            [] a core:ThingDescription ;
               core:describes <%s>
        }
    }""" % th_uri, cache=True, infer=False, expire=300)
    td_uri = res.pop()['g']['value']
    return create_TD_from(R, td_uri, node_map, **kwargs)


def build_component(R, VTED, id, node_map=None, lazy=True):
    if node_map is None:
        node_map = {}
    uri = URIRef(id)
    suc_tds = []

    loader = None if lazy else R.pull

    try:
        matching_td = get_matching_TD(R, uri, node_map, loader=loader, lazy=lazy)
        network = VTED.network
        if not is_root(R, id):
            roots = filter(lambda (th, td): th and td, VTED.roots)
            for th_uri, td_uri in roots:
                root = create_TD_from(R, td_uri, node_map=node_map, lazy=lazy, loader=loader)
                try:
                    root_paths = nx.all_simple_paths(network, root.id, matching_td.id)
                    for root_path in root_paths:
                        root_path = root_path[1:]
                        suc_tds = []
                        for suc_td_id in root_path:
                            if suc_td_id not in node_map:
                                node_map[suc_td_id] = get_td_node(R, suc_td_id)
                            suc_td = create_TD_from(R, node_map[suc_td_id], node_map=node_map, lazy=lazy,
                                                    loader=loader)
                            if suc_td not in suc_tds:
                                suc_tds.append(suc_td)
                        yield root, suc_tds
                except nx.NetworkXNoPath:
                    pass
                except nx.NodeNotFound:
                    pass
        else:
            if not lazy:
                suc_td_ids = reduce(lambda x, y: x.union(set(y)), list(nx.dfs_edges(network, matching_td.id)), set())
                for suc_td_id in suc_td_ids:
                    if suc_td_id not in node_map:
                        node_map[suc_td_id] = get_td_node(R, suc_td_id)
                    suc_td = create_TD_from(R, node_map[suc_td_id], node_map=node_map, lazy=lazy, loader=loader)
                    if suc_td not in suc_tds:
                        suc_tds.append(suc_td)
            yield matching_td, suc_tds
    except IndexError:
        graph = R.pull(uri)
        resource = Resource.from_graph(graph, uri, node_map=node_map)
        yield resource, []


def build_TED(root_paths):
    ted = TED()
    for root_path in root_paths:
        for base, sub_tds in root_path:
            if isinstance(base, TD):
                ted.ecosystem.add_root_from_td(base)
                for std in sub_tds:
                    ted.ecosystem.add_td(std)
            else:
                ted.ecosystem.add_root(base)

    return ted


def learn_descriptions_from(R, desc_g):
    virtual_eco_node = BNode()
    td_nodes = list(desc_g.subjects(RDF.type, CORE.ThingDescription))
    for td_node in filter(lambda t: not list(desc_g.triples((None, None, t))), td_nodes):
        th_node = list(desc_g.objects(td_node, CORE.describes)).pop()
        desc_g.add((virtual_eco_node, CORE.hasComponent, th_node))

    candidate_th_nodes = set(desc_g.subjects(RDF.type)).difference(td_nodes)
    for cand_th_node in candidate_th_nodes:
        if not list(desc_g.triples((None, None, cand_th_node))):
            desc_g.add((virtual_eco_node, CORE.hasComponent, cand_th_node))

    desc_g.add((virtual_eco_node, RDF.type, CORE.Ecosystem))
    eco = Ecosystem.from_graph(desc_g, loader=get_context(R))
    g = eco.to_graph()

    node_map = {}
    sub_eco = Ecosystem()
    td_nodes = g.subjects(RDF.type, CORE.ThingDescription)
    for td_node in td_nodes:
        try:
            skolem_id = list(g.objects(td_node, CORE.identifier)).pop()
        except IndexError:
            skolem_id = None
        g = canonize_node(g, td_node, R.base, id='descriptions/{}'.format(skolem_id))

    tdh_nodes = g.subject_objects(predicate=CORE.describes)
    for td_node, th_node in tdh_nodes:
        try:
            skolem_id = list(g.objects(td_node, CORE.identifier)).pop()
        except IndexError:
            skolem_id = None
        g = canonize_node(g, th_node, R.base, id='things/{}'.format(skolem_id))

    td_nodes = g.subjects(RDF.type, CORE.ThingDescription)
    for node in td_nodes:
        td = TD.from_graph(g, node, node_map)
        sub_eco.add_td(td)

    network = sub_eco.network()

    root_ids = filter(lambda x: network.in_degree(x) == 0, network.nodes())
    root_tds = filter(lambda td: td.id in root_ids, sub_eco.tds)
    for td in root_tds:
        sub_eco.add_root_from_td(td)

    all_types = R.agora.fountain.types
    ns = R.ns()

    non_td_resources = defaultdict(set)
    for elm, _, cl in desc_g.triples((None, RDF.type, None)):
        if isinstance(elm, URIRef) and (None, CORE.hasComponent, elm) in g:
            if cl.n3(ns) in all_types:
                non_td_resources[elm].add(cl)

    for r_uri, types in non_td_resources.items():
        sub_eco.add_root(Resource(uri=r_uri, types=types))

    ted = TED()
    ted.ecosystem = sub_eco

    return ted


def now():
    return calendar.timegm(datetime.utcnow().timetuple())


@cache_it(cache=cache, expire=300)
def _sync_VTED():
    return now()


def get_td_ids(R, cache=True):
    res = R.query("""
    PREFIX core: <http://iot.linkeddata.es/def/core#>
    SELECT DISTINCT ?g ?id ?th WHERE {
        GRAPH ?g {
           [] a core:ThingDescription ;
              core:identifier ?id ;
              core:describes ?th
        }
    }""", cache=cache, infer=False, expire=300)

    return map(lambda r: (r['g']['value'], r['id']['value'], r['th']['value']), res)


def get_resource_transforms(R, td, cache=True):
    res = R.query("""
    PREFIX map: <http://iot.linkeddata.es/def/wot-mappings#>
    SELECT DISTINCT ?t FROM <%s> WHERE {                        
       [] map:valuesTransformedBy ?t            
    }""" % td, cache=cache, infer=False, expire=300, namespace='network')
    return map(lambda r: r['t']['value'], res)


def get_thing_links(R, th, cache=True):
    res = R.query("""
    SELECT DISTINCT ?o FROM <%s> WHERE {
      [] ?p ?o
      FILTER(isURI(?o))
    }
    """ % th, cache=cache, namespace='network')
    return map(lambda r: r['o']['value'], res)


def get_td_thing(R, td_uri):
    res = R.query("""
            PREFIX core: <http://iot.linkeddata.es/def/core#>
            SELECT DISTINCT ?th WHERE {
              <%s> a core:ThingDescription ;
                 core:describes ?th                 
            }""" % td_uri, cache=True, infer=False)
    try:
        return res.pop()['th']['value']
    except IndexError:
        log.warn('No described thing for TD {}'.format(td_uri))


def get_th_types(R, th_uri, **kwargs):
    res = R.query("""
    PREFIX core: <http://iot.linkeddata.es/def/core#>
    SELECT DISTINCT ?type WHERE {
      <%s> a ?type                 
    }""" % th_uri, cache=True, expire=300, **kwargs)
    return [URIRef(r['type']['value']) for r in res if r['type']['value'] != str(RDFS.Resource)]


def materialize_th_types(R, ids):
    for th_id in ids:
        g = R.pull(th_id)
        prev_len = len(g)
        prefixes = R.agora.fountain.prefixes
        for s, p, o in g.triples((None, RDF.type, None)):
            try:
                type_n3 = g.qname(o)
                type_super = R.agora.fountain.get_type(type_n3)['super']
                for ts in type_super:
                    ts_triple = (s, p, URIRef(extend_uri(ts, prefixes)))
                    if ts_triple not in g:
                        g.add(ts_triple)
            except Exception:
                pass
        if len(g) > prev_len:
            R.push(g)


class VTED(object):
    def __init__(self, R):
        self.R = R

    def __sync_VTED(self, force=False):
        ts = now()
        if force or ts - _sync_VTED() > 300 or META['ts'] is None:
            log.info('[{}] Syncing VTED...'.format(ts))
            META['network'] = self._network(cache=not force)
            META['roots'] = self._roots(cache=not force)
            td_th_ids = set(map(lambda x: x[2], get_td_ids(self.R, cache=not force)))
            root_ids = set(map(lambda x: x[0], META['roots']))
            materialize_th_types(self.R, td_th_ids.union(root_ids))
            META['ts'] = ts
            log.info('[{}] Syncing completed'.format(ts))

    def __getattr__(self, key):
        self.__sync_VTED()
        if key in META:
            return META[key]
        else:
            return self.__getattribute__(key)

    def sync(self, force=False):
        self.__sync_VTED(force=force)

    def add_component(self, ted, eco, uri):
        with lock:
            g = Graph(identifier=ted)
            g.add((URIRef(eco), CORE.hasComponent, URIRef(uri)))
            self.R.insert(g)

    def remove_component(self, ted, uri):
        with lock:
            self.R.update(u"""
            PREFIX core: <http://iot.linkeddata.es/def/core#>
            DELETE { GRAPH <%s> { ?s ?p ?o }} WHERE { ?s core:hasComponent <%s> }
            """ % (ted, uri))

    def _network(self, cache=True):
        network = nx.DiGraph()
        td_th_ids = get_td_ids(self.R, cache=cache)
        td_ids_dict = dict(map(lambda x: (x[0], x[1]), td_th_ids))
        td_th_dict = dict(map(lambda x: (x[0], x[2]), td_th_ids))
        th_td_dict = dict(map(lambda x: (x[2], x[0]), td_th_ids))
        all_tds = map(lambda x: x[0], td_th_ids)
        for td, id in td_ids_dict.items():
            network.add_node(id)

            td_transforms = filter(lambda x: x in all_tds, get_resource_transforms(self.R, td, cache=cache))
            for linked_td in td_transforms:
                network.add_edge(id, td_ids_dict[linked_td])

            th = td_th_dict[td]
            thing_td_links = map(lambda x: th_td_dict[x],
                                 filter(lambda x: x in th_td_dict, get_thing_links(self.R, th, cache=cache)))
            for linked_td in thing_td_links:
                network.add_edge(id, td_ids_dict[linked_td])

        return network

    def _roots(self, cache=True):
        res = self.R.query("""
        PREFIX core: <http://iot.linkeddata.es/def/core#>
        SELECT DISTINCT ?root ?td WHERE {
            [] a core:ThingEcosystemDescription ;
               core:describes [
                  core:hasComponent ?root
               ] .
               OPTIONAL { ?td core:describes ?root }               
        }""", cache=cache, infer=False, expire=300, namespace='eco')
        roots = map(lambda r: (r['root']['value'], r.get('td', {}).get('value', None)), res)
        return roots

    def ted_eco(self):
        try:
            res = self.R.query("""
            PREFIX core: <http://iot.linkeddata.es/def/core#>                
            SELECT ?g ?eco WHERE {
               GRAPH ?g {
                  [] a core:ThingEcosystemDescription ;
                     core:describes ?eco
               }
            }""", cache=False, namespace='eco').pop()
            eco = res['eco']['value']
            ted_uri = res['g']['value']
            return ted_uri, eco
        except IndexError:
            raise EnvironmentError

    def update(self, ted, th_graph_builder, eco_uri):
        td_nodes = {td: td.node for td in ted.ecosystem.tds}
        last_td_based_roots = set([URIRef(root_uri) for (root_uri, td) in self._roots(cache=False) if td and root_uri])

        for td in ted.ecosystem.tds:
            self.R.push(td.to_graph(td_nodes=td_nodes))
            self.R.push(th_graph_builder(td))

        try:
            ted_uri, eco = self.ted_eco()
        except EnvironmentError:
            self.R.push(ted.to_graph(node=eco_uri, abstract=True))
        else:
            self.sync(force=True)
            network_roots = set(map(lambda (n, _): URIRef(get_th_node(self.R, n)),
                                    filter(lambda (n, degree): degree == 0, dict(self.network.in_degree()).items())))
            obsolete_td_based_roots = set.difference(last_td_based_roots, network_roots)
            ted_components = ted.ecosystem.roots
            for root in ted_components:
                if isinstance(root, TD):
                    resource = root.resource
                    if resource.node in network_roots and resource.node not in last_td_based_roots:
                        self.add_component(ted_uri, eco, resource.node)
                else:
                    self.R.push(root.to_graph())
                    self.add_component(ted_uri, eco, root.node)

            promoted_td_based_root_uris = set.difference(network_roots, last_td_based_roots)
            for root in promoted_td_based_root_uris:
                self.add_component(ted_uri, eco, root)

            for root in obsolete_td_based_roots:
                self.remove_component(ted_uri, root)

        self.sync(force=True)
        self.R.expire_cache()
