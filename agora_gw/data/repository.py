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
import os
from StringIO import StringIO
from multiprocessing import Lock

from agora import Agora
from agora.engine.fountain.onto import DuplicateVocabulary
from agora.engine.utils import Wrapper
from rdflib import Graph, RDF, URIRef
from rdflib.namespace import Namespace, OWL, DC
from rdflib.term import Literal
from shortuuid import uuid

from agora_gw.data.graph import push as push_g, pull, delete as delete_g
from agora_gw.data.sparql import SPARQL

__author__ = 'Fernando Serena'

SCHEMA_GRAPH = os.environ.get('SCHEMA_GRAPH', 'http://agora.org/schema')
EXTENSION_BASE = os.environ.get('EXTENSION_BASE', 'http://agora.org/extensions/')

WOT = Namespace('http://iot.linkeddata.es/def/wot#')
CORE = Namespace('http://iot.linkeddata.es/def/core#')
EXT = Namespace(EXTENSION_BASE)

FOUNTAIN_HOST = os.environ.get('FOUNTAIN_HOST', None)
FOUNTAIN_PORT = os.environ.get('FOUNTAIN_PORT', None)

log = logging.getLogger('agora.gateway.data.repository')
_lock = Lock()

REPOSITORY_BASE = unicode(os.environ.get('REPOSITORY_BASE', 'http://agora.org/data/').rstrip('/'))


def _learn_thing_describing_predicates(id, sparql):
    # type: (str, SPARQL, dict) -> set

    res = sparql.query("""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT DISTINCT ?p WHERE {            
        GRAPH <%s> { [] ?p [] }
    }
    """ % id, cache=True, expire=300)
    all_predicates = set([URIRef(x['p']['value']) for x in res])

    res = sparql.query("""
            prefix core: <http://iot.linkeddata.es/def/core#>
            SELECT DISTINCT ?p WHERE {
                [] a core:ThingDescription ;
                   core:describes ?thing .
                <%s> ?p ?thing
            }
            """ % id, cache=True, infer=False, expire=300)
    bound_predicates = set([URIRef(x['p']['value']) for x in res])
    return all_predicates.difference(bound_predicates)


def _learn_describing_predicates(sparql):
    # type: (SPARQL, dict) -> set

    res = sparql.query("""
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

    res = sparql.query("""
        prefix map: <http://iot.linkeddata.es/def/wot-mappings#>
        SELECT DISTINCT ?p WHERE {                
            [] map:predicate ?p
        }
        """, cache=True, infer=False, expire=300)
    mapped_predicates = set([URIRef(x['p']['value']) for x in res])

    res = sparql.query("""
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


def _learn_describing_types(sparql, schema_graph=SCHEMA_GRAPH):
    # type: (sparql, dict) -> set

    res = sparql.query("""
    SELECT DISTINCT ?type WHERE {
        GRAPH ?g { [] a ?type }    	
        FILTER(?g != <%s>)
    }
    """ % schema_graph, cache=True, expire=300)
    types = set([URIRef(x['type']['value']) for x in res])
    return types


class Repository(object):
    def __init__(self, **kwargs):
        pass

    @property
    def ext_base(self):
        return self.__ext_base or EXTENSION_BASE

    @ext_base.setter
    def ext_base(self, eb):
        self.__ext_base = eb

    @property
    def sparql(self):
        return self._sparql

    @property
    def agora(self):
        return self._agora

    @agora.setter
    def agora(self, a):
        self._agora = a

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
        remote_ext_vocabs = set(filter(lambda v: v.startswith(self.ext_base), remote_vocabs))
        remote_delta = remote_ext_vocabs.difference(extension_vocabs)
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
                    if self.ext_base in rv:
                        ext_id = rv.replace(self.ext_base, '').lstrip('/').lstrip('#')

            if ext_id is not None and ext_id not in extension_prefixes:
                self.learn(rg, ext_ns=rv, ext_id=ext_id, push=False)

        local_delta = extension_vocabs.difference(remote_ext_vocabs)
        for lv in local_delta:
            lv_prefix = rev_prefixes.get(lv, None)
            if lv_prefix and lv_prefix not in extension_prefixes:
                ext_g = self.get_extension(rev_prefixes.get(lv, lv.replace(EXT, '')))
                g = Graph(identifier=lv)
                g.__iadd__(ext_g)
                push_g(self.sparql, g)

    @property
    def base(self):
        return self.__repository_base or REPOSITORY_BASE

    @base.setter
    def base(self, b):
        self.__repository_base = b

    @property
    def describing_types(self):
        return _learn_describing_types(self.sparql)

    @property
    def describing_predicates(self):
        return _learn_describing_predicates(self.sparql)

    def thing_describing_predicates(self, id):
        return _learn_thing_describing_predicates(id, self.sparql)

    def query(self, q, **kwargs):
        return self.sparql.query(u'{}'.format(q), **kwargs)

    def update(self, q):
        return self.sparql.update(u'{}'.format(q))

    def pull(self, uri, **kwargs):
        if not uri:
            raise AttributeError(u'Cannot pull {} from repository'.format(uri))
        g = pull(self.sparql, u'{}'.format(uri), **kwargs)
        for prefix, uri in self.fountain.prefixes.items():
            g.bind(prefix, uri)
        return g

    def push(self, g):
        push_g(self.sparql, g)

    def insert(self, g):
        push_g(self.sparql, g, delete=False)

    def delete(self, gid):
        delete_g(self.sparql, gid)

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
            push_g(self.sparql, g, gid=ext_ns, delete=True)
        self.sparql.expire_cache()

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
        res_g = Graph(identifier=id)
        rev_ns = {ns: prefix for prefix, ns in g.namespaces()}
        for s, p, o in g:
            if o == OWL.Ontology:
                continue

            match_ns(s)
            match_ns(p)
            match_ns(o)
            res_g.add((s, p, o))
        return res_g

    def delete_extension(self, id):
        prefixes = self.agora.fountain.prefixes
        ext_ns = prefixes.get(id, EXT[id])
        self.agora.fountain.delete_vocabulary(id)
        delete_g(self.sparql, ext_ns)
        self.sparql.expire_cache()

    def shutdown(self):
        try:
            Agora.close()
        except Exception:
            pass

    def ns(self, fountain=None):
        if fountain is None:
            fountain = self.agora.fountain
        g = Graph()
        for prefix, ns in fountain.prefixes.items():
            g.bind(prefix, URIRef(ns))
        return g.namespace_manager

    def n3(self, uri, ns=None):
        if ns is None:
            ns = self.ns()
        qname = URIRef(uri).n3(ns)
        qname_strip = qname.lstrip('<').rstrip('>')
        if qname_strip == uri:
            return uri

        return qname

    def expire_cache(self, namespace=None):
        self.sparql.expire_cache(namespace=namespace)

    @property
    def fountain(self):
        return Wrapper(self.agora.fountain)

    def __new__(cls, *args, **kwargs):
        r = super(Repository, cls).__new__(cls)
        r.__init__()
        r_base = kwargs.get('repository_base', None)
        r.base = r_base
        ext_base = kwargs.get('extension_base', None)
        r.ext_base = ext_base
        # if r_base is not None:
        #     del kwargs['repository_base']
        # if ext_base is not None:
        #     del kwargs['extension_base']
        r._sparql = SPARQL(**kwargs.get('data', {}))

        return r
