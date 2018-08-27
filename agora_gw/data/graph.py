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
from urlparse import urlparse

from rdflib import Graph, URIRef, RDF, BNode, ConjunctiveGraph

from agora_gw.data.sparql import SPARQL

__author__ = 'Fernando Serena'


BNODE_SKOLEM_BASE = os.environ.get('BNODE_SKOLEM_BASE', 'http://bnodes').rstrip('/')

log = logging.getLogger('agora.gateway.data.graph')


def _get_graph(gid, sparql, de_skolemize=True, **kwargs):
    # type: (str, SPARQL, bool, any) -> Graph
    data_gid = gid

    if not isinstance(sparql.sparql_host, basestring):
        g = sparql.sparql_host.get_context(gid)
    else:
        g_n3 = sparql.query("""
            CONSTRUCT { ?s ?p ?o }
            WHERE
            {
                GRAPH <%s> {
                    ?s ?p ?o
                } .
            }
            """ % gid, **kwargs)

        g = Graph(identifier=URIRef(gid))
        g.parse(StringIO(g_n3), format='n3')

    if g:
        res = sparql.query("""
        SELECT DISTINCT ?t WHERE {
          <%s> a ?t
          FILTER(isURI(?t))
        }
        """ % data_gid, **kwargs)
        for trow in res:
            type_value = trow['t']['value']
            if BNODE_SKOLEM_BASE not in type_value:
                g.add((URIRef(data_gid), RDF.type, URIRef(type_value)))

    if not de_skolemize:
        return g

    bn_map = {}
    deskolem = Graph(identifier=URIRef(gid))
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


def _chunks(l, n):
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


def _store_graph(g, sparql, gid=None, delete=True, do_skolem=True):
    # type: (Graph, SPARQL, str, bool, bool) -> None

    if not isinstance(sparql.update_host, basestring):
        if delete:
            sparql.update_host.remove_context(sparql.update_host.get_context(gid))
        sparql.update_host.get_context(gid).__iadd__(g)
    else:
        q_tmpl = u"""    
        INSERT DATA
        { GRAPH <%s> { %s } }
        """
        gid = gid or g.identifier
        log.debug('Storing graph {}...'.format(gid))
        if delete:
            sparql.update(u"""
            DELETE { GRAPH <%s> { ?s ?p ?o }} WHERE { ?s ?p ?o }
            """ % gid)

        skolem = skolemize(g) if do_skolem else g
        for chunk in _chunks(skolem, 1000):
            all_triples_str = u' . '.join(map(lambda (s, p, o): u'{} {} {}'.format(s.n3(), p.n3(), o.n3()), chunk))
            query = q_tmpl % (gid, all_triples_str)
            try:
                sparql.update(query)
            except Exception as e:
                print query
                log.warn(e.message)


def _delete_graph(gid, sparql):
    sparql.update(u"""
    DELETE { GRAPH <%s> { ?s ?p ?o }} WHERE { ?s ?p ?o }
    """ % gid)
    sparql.update(u"""
    DELETE WHERE { ?s ?p <%s> }
    """ % gid)


def skolemize(g, skolem_base=BNODE_SKOLEM_BASE):
    bn_map = {}
    skolem = ConjunctiveGraph()
    for prefix, ns in g.namespaces():
        skolem.bind(prefix, ns)
    for s, p, o in g:
        if isinstance(s, BNode):
            if s not in bn_map:
                bn_map[s] = URIRef('/'.join([skolem_base, str(s)]))
            s = bn_map[s]
        if isinstance(o, BNode):
            if o not in bn_map:
                bn_map[o] = URIRef('/'.join([skolem_base, str(o)]))
            o = bn_map[o]
        skolem.add((s, p, o))
    return skolem


def deskolemize(g, skolem_base=BNODE_SKOLEM_BASE):
    bn_map = {}
    deskolem = ConjunctiveGraph()
    for prefix, ns in g.namespaces():
        deskolem.bind(prefix, ns)

    for s, p, o in g:
        if isinstance(s, URIRef) and s.startswith(skolem_base):
            if s not in bn_map:
                bn_map[s] = BNode(s.replace(skolem_base + '/', ''))
            s = bn_map[s]
        if isinstance(o, URIRef) and o.startswith(skolem_base):
            if o not in bn_map:
                bn_map[o] = BNode(o.replace(skolem_base + '/', ''))
            o = bn_map[o]
        deskolem.add((s, p, o))
    return deskolem


def canonize_node(g, node, authority, id=None):
    skolem = ConjunctiveGraph()

    if isinstance(node, URIRef):
        node_parse = urlparse(node)
        node_id = node_parse.path.lstrip('/')
        if node_id != id:
            return g

    if not id:
        id = str(node)

    authority = authority.rstrip('/')
    skolem_uri = URIRef('/'.join([authority, str(id)]))
    for s, p, o in g:
        if s == node:
            s = skolem_uri
        if o == node:
            o = skolem_uri
        skolem.add((s, p, o))
    return skolem


class GraphHandler(object):
    def __init__(self, gid, sparql):
        self.gid = gid
        self.sparql = sparql

    def get(self, **kwargs):
        return _get_graph(self.gid, self.sparql, **kwargs)

    def store(self, g, delete=True):
        retry = True
        skolem = skolemize(g)
        while retry:
            retry = False
            _store_graph(skolem, self.sparql, gid=self.gid, delete=delete, do_skolem=False)

            if not delete:
                break

            stored_g = _get_graph(self.gid, self.sparql, cache=None, infer=False, de_skolemize=False)
            if len(skolem) != len(stored_g):
                retry = True

    def delete(self):
        _delete_graph(self.gid, self.sparql)


def push(sparql, g, gid=None, delete=True):
    GraphHandler(gid or g.identifier, sparql).store(g, delete=delete)


def pull(sparql, gid, **kwargs):
    return GraphHandler(gid, sparql).get(**kwargs)


def delete(sparql, gid):
    return GraphHandler(gid, sparql).delete()
