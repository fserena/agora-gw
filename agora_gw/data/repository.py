"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Ontology Engineering Group
        http://www.oeg-upm.net/
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2017 Ontology Engineering Group.
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
import json
import logging
import os
import traceback
from StringIO import StringIO
from multiprocessing import Lock
from urllib import urlencode
from urlparse import urlparse

import requests
from SPARQLWrapper import SPARQLWrapper, JSON, N3
from agora import Agora, Planner
from agora.engine.fountain.onto import DuplicateVocabulary
from agora.engine.utils import Wrapper
from agora.server.fountain import client as fc
from rdflib import Graph, RDF, URIRef, ConjunctiveGraph
from rdflib.namespace import Namespace, OWL, DC
from rdflib.term import BNode, Literal
from redis_cache import SimpleCache, cache_it, DEFAULT_EXPIRY
from shortuuid import uuid

__author__ = 'Fernando Serena'

sparql_url = 'localhost'
objects_url = 'localhost'
update_url = 'localhost'

SPARQL_HOST = os.environ.get('SPARQL_HOST', sparql_url)
UPDATE_HOST = os.environ.get('UPDATE_HOST', update_url)
SCHEMA_GRAPH = os.environ.get('SCHEMA_GRAPH', 'http://example.org/schema')
EXTENSION_BASE = os.environ.get('EXTENSION_BASE', 'http://example.org/extensions/')

QUERY_CACHE_HOST = os.environ.get('QUERY_CACHE_HOST', 'localhost')
QUERY_CACHE_NUMBER = int(os.environ.get('QUERY_CACHE_NUMBER', 8))

WOT = Namespace('http://iot.linkeddata.es/def/wot#')
CORE = Namespace('http://iot.linkeddata.es/def/core#')
EXT = Namespace(EXTENSION_BASE)

GEO = Namespace('http://www.w3.org/2003/01/geo/wgs84_pos#')

FOUNTAIN_HOST = os.environ.get('FOUNTAIN_HOST', 'localhost')
FOUNTAIN_PORT = os.environ.get('FOUNTAIN_PORT', 8001)

log = logging.getLogger('agora.gateway.repository')
_lock = Lock()

query_cache = SimpleCache(limit=10000, expire=60 * 60, hashkeys=True, host=QUERY_CACHE_HOST, port=6379,
                          db=QUERY_CACHE_NUMBER, namespace='gateway')

REPOSITORY_BASE = unicode(os.environ.get('REPOSITORY_BASE', 'http://descriptions').rstrip('/'))
BNODE_SKOLEM_BASE = os.environ.get('BNODE_SKOLEM_BASE', 'http://bnodes').rstrip('/')


def get_agora():
    fountain = fc(host=FOUNTAIN_HOST, port=FOUNTAIN_PORT)
    planner = Planner(fountain)

    agora = Agora()
    agora.planner = planner
    log.info('The knowledge graph contains {} types and {} properties'.format(len(agora.fountain.types),
                                                                              len(agora.fountain.properties)))
    return agora


def update(q):
    res = requests.post(UPDATE_HOST,
                        headers={
                            'Accept': 'text/plain,*/*;q=0.9',
                            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
                        },
                        data=urlencode({'update': q.encode('utf-8')}))
    return res


def query(q, cache=True, infer=True, expire=DEFAULT_EXPIRY, namespace=None):
    def wrapper(q):
        sparql = SPARQLWrapper(SPARQL_HOST)
        sparql.setRequestMethod("postdirectly")
        sparql.setMethod('POST')

        log.debug(u'Querying: {}'.format(q))
        sparql.setQuery(q)

        sparql.addCustomParameter('infer', str(infer).lower())
        if not ('construct' in q.lower()):
            sparql.setReturnFormat(JSON)
        else:
            sparql.setReturnFormat(N3)

        results = sparql.query().convert()
        if isinstance(results, str):
            return results.decode('utf-8')
        else:
            if 'results' in results:
                return json.dumps(results["results"]["bindings"]).decode('utf-8')
            else:
                return json.dumps(results['boolean']).decode('utf-8')

    if cache:
        try:
            ret = cache_it(cache=query_cache, expire=expire, namespace=namespace)(wrapper)(q)
        except UnicodeDecodeError:
            traceback.print_exc()
            return []
    else:
        ret = wrapper(q)

    try:
        return json.loads(ret)
    except ValueError:
        return ret


def learn_thing_describing_predicates(id):
    res = query("""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT DISTINCT ?p WHERE {            
        GRAPH <%s> { [] ?p [] }
    }
    """ % id, cache=True, expire=300)
    all_predicates = set([URIRef(x['p']['value']) for x in res])

    res = query("""
            prefix core: <http://iot.linkeddata.es/def/core#>
            SELECT DISTINCT ?p WHERE {
                [] a core:ThingDescription ;
                   core:describes ?thing .
                <%s> ?p ?thing
            }
            """ % id, cache=True, infer=False, expire=300)
    bound_predicates = set([URIRef(x['p']['value']) for x in res])
    return all_predicates.difference(bound_predicates)


def learn_describing_predicates():
    res = query("""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT DISTINCT ?p WHERE {    
        {
            SELECT DISTINCT * WHERE {
                {
                    ?p a owl:DatatypeProperty
                } UNION {
                    ?p a owl:ObjectProperty
                }
            }
        }
        [] ?p [] .
    }
    """, cache=True, expire=300)
    all_predicates = set([URIRef(x['p']['value']) for x in res])

    res = query("""
        prefix map: <http://iot.linkeddata.es/def/wot-mappings#>
        SELECT DISTINCT ?p WHERE {                
            [] map:predicate ?p
        }
        """, cache=True, infer=False, expire=300)
    mapped_predicates = set([URIRef(x['p']['value']) for x in res])

    res = query("""
        prefix core: <http://iot.linkeddata.es/def/core#>
        SELECT DISTINCT ?p WHERE {
            [] a core:ThingDescription ;
               core:describes ?r .
            [] ?p ?r
          FILTER(?p != core:describes)
        }
        """, cache=True, infer=False, expire=300)
    td_bound_predicates = set([URIRef(x['p']['value']) for x in res])
    return all_predicates.difference(mapped_predicates).difference(td_bound_predicates)


def learn_describing_types():
    res = query("""
    SELECT DISTINCT ?type WHERE {
        GRAPH ?g { [] a ?type }    	
        FILTER(?g != <%s>)
    }
    """ % SCHEMA_GRAPH, cache=True, expire=300)
    types = set([URIRef(x['type']['value']) for x in res])
    return types


def get_graph(gid, **kwargs):
    data_gid = gid
    g_n3 = query("""
        CONSTRUCT { ?s ?p ?o }
        WHERE
        {
            GRAPH <%s> {
                ?s ?p ?o
            } .
        }
        """ % gid, **kwargs)

    g = Graph(identifier=gid)
    g.parse(StringIO(g_n3), format='n3')
    if g:
        res = query("""
        SELECT DISTINCT ?t WHERE {
          <%s> a ?t
          FILTER(isURI(?t))
        }
        """ % data_gid, **kwargs)
        for trow in res:
            g.add((URIRef(data_gid), RDF.type, URIRef(trow['t']['value'])))

    bn_map = {}
    deskolem = Graph(identifier=gid)
    for s, p, o in g:
        if BNODE_SKOLEM_BASE in s:
            if s not in bn_map:
                bn_map[s] = BNode()
            s = bn_map[s]
        if BNODE_SKOLEM_BASE in o:
            if o not in bn_map:
                bn_map[o] = BNode()
            o = bn_map[o]
        deskolem.add((s, p, o))
    return deskolem


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    index = 0
    chunk = []
    for t in l.triples((None, None, None)):
        if index < n:
            chunk.append(t)
            index += 1
        else:
            yield chunk[:]
            chunk = []
            index = 0

    if chunk:
        yield chunk[:]


def store_graph(g, gid=None, delete=True):
    q_tmpl = u"""    
    INSERT DATA
    { GRAPH <%s> { %s } }
    """
    gid = gid or g.identifier
    log.debug('Storing graph {}...'.format(gid))
    if delete:
        update(u"""
        DELETE { GRAPH <%s> { ?s ?p ?o }} WHERE { ?s ?p ?o }
        """ % gid)

    skolem = skolemize(g)
    for chunk in chunks(skolem, 50):
        all_triples_str = u' . '.join(map(lambda (s, p, o): u'{} {} {}'.format(s.n3(), p.n3(), o.n3()), chunk))
        query = q_tmpl % (gid, all_triples_str)
        try:
            update(query)
        except Exception as e:
            print query
            log.warn(e.message)




def delete_graph(gid):
    update(u"""
    DELETE { GRAPH <%s> { ?s ?p ?o }} WHERE { ?s ?p ?o }
    """ % gid)


def skolemize(g):
    bn_map = {}
    skolem = ConjunctiveGraph()
    for prefix, ns in g.namespaces():
        skolem.bind(prefix, ns)
    for s, p, o in g:
        if isinstance(s, BNode):
            if s not in bn_map:
                bn_map[s] = URIRef('/'.join([BNODE_SKOLEM_BASE, str(s)]))
            s = bn_map[s]
        if isinstance(o, BNode):
            if o not in bn_map:
                bn_map[o] = URIRef('/'.join([BNODE_SKOLEM_BASE, str(o)]))
            o = bn_map[o]
        skolem.add((s, p, o))
    return skolem


def deskolemize(g):
    bn_map = {}
    deskolem = ConjunctiveGraph()
    for prefix, ns in g.namespaces():
        deskolem.bind(prefix, ns)

    for s, p, o in g:
        if isinstance(s, URIRef) and s.startswith(BNODE_SKOLEM_BASE):
            if s not in bn_map:
                bn_map[s] = BNode(s.replace(BNODE_SKOLEM_BASE + '/', ''))
            s = bn_map[s]
        if isinstance(o, URIRef) and o.startswith(BNODE_SKOLEM_BASE):
            if o not in bn_map:
                bn_map[o] = BNode(o.replace(BNODE_SKOLEM_BASE + '/', ''))
            o = bn_map[o]
        deskolem.add((s, p, o))
    return deskolem


def canonize_node(g, node, authority=REPOSITORY_BASE, id=None):
    skolem = ConjunctiveGraph()

    if isinstance(node, URIRef):
        node_parse = urlparse(node)
        node_id = node_parse.path.lstrip('/')
        if node_id != id:
            return g

    if not id:
        id = str(node)

    skolem_uri = URIRef('/'.join([authority, str(id)]))
    for s, p, o in g:
        if s == node:
            s = skolem_uri
        if o == node:
            o = skolem_uri
        skolem.add((s, p, o))
    return skolem


class Repository(object):
    def __init__(self):
        self.agora = get_agora()
        self.expire_cache()

        fountain = self.fountain
        prefixes = fountain.prefixes

        extension_prefixes = self.extensions
        extension_vocabs = set([prefixes.get(ext, EXT[ext]) for ext in extension_prefixes])
        rev_prefixes = {prefixes[prefix]: prefix for prefix in prefixes}

        res = self.query("""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?g ?gid WHERE {
            GRAPH ?g {
                {
                    [] a owl:Class 
                } UNION {
                    [] a rdfs:Class
                } UNION {
                    [] a owl:DatatypeProperty
                } UNION {
                    [] a owl:ObjectProperty
                }
            }
        }
        """)
        remote_vocabs = set([URIRef(r['g']['value']) for r in res])
        remote_delta = remote_vocabs.difference(extension_vocabs)
        for rv in remote_delta:
            rg = self.pull(rv)
            try:
                ext_id = list(rg.objects(URIRef(rv), DC.identifier)).pop()
            except IndexError:
                ext_id = rev_prefixes.get(rv, None)

            if ext_id is None:
                try:
                    ext_id = [prefix for (prefix, ns) in rg.namespaces() if ns == rv].pop()
                except IndexError:
                    if EXTENSION_BASE in rv:
                        ext_id = rv.replace(EXTENSION_BASE, '').lstrip('/').lstrip('#')

            if ext_id is not None:
                self.learn(rg, ext_ns=rv, ext_id=ext_id, push=False)

        local_delta = extension_vocabs.difference(remote_vocabs)
        for lv in local_delta:
            ttl = self.get_extension(rev_prefixes.get(lv, lv.replace(EXT, '')))
            g = Graph(identifier=lv)
            g.parse(StringIO(ttl), format='turtle')
            store_graph(g, delete=True)

    @property
    def describing_types(self):
        return learn_describing_types()

    @property
    def describing_predicates(self):
        return learn_describing_predicates()

    def thing_describing_predicates(self, id):
        return learn_thing_describing_predicates(id)

    def query(self, q, **kwargs):
        return query(u'{}'.format(q), **kwargs)

    def update(self, q):
        return update(u'{}'.format(q))

    def pull(self, uri, **kwargs):
        return get_graph(u'{}'.format(uri), **kwargs)

    def push(self, g):
        store_graph(g)

    def insert(self, g):
        store_graph(g, delete=False)

    @property
    def extensions(self):
        return self.agora.fountain.vocabularies

    def learn(self, g, ext_ns=None, ext_id=None, push=True):
        if ext_id is None:
            ext_id = 'ext_' + uuid()

        agora_prefixes = self.agora.fountain.prefixes
        ext_prefixes = dict(g.namespaces())
        agora_ns = agora_prefixes.get(ext_id, None)

        if ext_ns is None:
            ext_ns = ext_prefixes.get(ext_id, EXT[ext_id]) if agora_ns is None else agora_ns

            if '' in ext_prefixes:
                ext_ns = ext_prefixes['']

        if agora_ns is not None and agora_ns != ext_ns:
            raise DuplicateVocabulary('Conflict with prefixes')

        g.namespace_manager.bind(ext_id, ext_ns, replace=True, override=True)
        g.set((ext_ns, RDF.type, OWL.Ontology))
        g.set((ext_ns, DC.identifier, Literal(ext_id)))
        g_ttl = g.serialize(format='turtle')
        if ext_id not in self.agora.fountain.vocabularies:
            self.agora.fountain.add_vocabulary(g_ttl)
        else:
            self.agora.fountain.update_vocabulary(ext_id, g_ttl)

        if push:
            store_graph(g, gid=ext_ns, delete=True)
        query_cache.connection.flushdb()

        return ext_id

    def get_extension(self, id):
        def match_ns(term):
            filter_ns = [ns for ns in rev_ns if ns in term]
            if filter_ns:
                ns = filter_ns.pop()
                res_g.bind(rev_ns[ns], ns)
                del rev_ns[ns]

        ttl = self.agora.fountain.get_vocabulary(id)
        g = Graph()
        g.parse(StringIO(ttl), format='turtle')
        res_g = Graph()
        rev_ns = {ns: prefix for prefix, ns in g.namespaces()}
        for s, p, o in g:
            if o == OWL.Ontology:
                continue

            match_ns(s)
            match_ns(p)
            match_ns(o)
            res_g.add((s, p, o))
        return res_g.serialize(format='turtle')

    def delete_extension(self, id):
        prefixes = self.agora.fountain.prefixes
        ext_ns = prefixes.get(id, EXT[id])
        self.agora.fountain.delete_vocabulary(id)
        delete_graph(ext_ns)

    def shutdown(self):
        try:
            Agora.close()
        except Exception:
            pass

    def ns(self):
        g = Graph()
        for prefix, ns in self.agora.fountain.prefixes.items():
            g.bind(prefix, URIRef(ns))
        return g.namespace_manager

    def n3(self, uri, ns):
        qname = URIRef(uri).n3(ns)
        qname_strip = qname.lstrip('<').rstrip('>')
        if qname_strip == uri:
            return uri

        return qname

    def link_path(self, source, target, fountain=None):
        if fountain is None:
            fountain = self.agora.fountain
        return fountain.connected(source, target)

    def expire_cache(self, namespace=None):
        if namespace is not None:
            query_cache.flush_namespace(namespace)
        else:
            query_cache.connection.flushdb()

    @property
    def fountain(self):
        return Wrapper(self.agora.fountain)
