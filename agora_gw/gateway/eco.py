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

from multiprocessing import Lock

from agora import Agora
from agora.engine.plan.agp import extend_uri
from agora_wot.blocks.resource import Resource
from agora_wot.blocks.td import TD
from agora_wot.blocks.ted import TED
from agora_wot.gateway import DataGateway
from rdflib import ConjunctiveGraph, URIRef, Graph, BNode, RDF

from agora_gw.data.repository import CORE, Repository
from agora_gw.ecosystem.description import learn_descriptions_from, get_td_node, get_th_node, get_th_types, VTED
from agora_gw.ecosystem.discover import discover_ecosystem
from agora_gw.ecosystem.serialize import deserialize, JSONLD
from agora_gw.gateway.abstract import AbstractEcoGateway
from agora_gw.server.client import EcoGatewayClient

__author__ = 'Fernando Serena'


class EcoGateway(AbstractEcoGateway):

    def __init__(self, **kwargs):
        self.__lock = Lock()

    @property
    def agora(self):
        return self.repository.agora

    @property
    def repository(self):
        return self.__repository

    @repository.setter
    def repository(self, r):
        self.__repository = r

    @property
    def VTED(self):
        return self.__VTED

    @VTED.setter
    def VTED(self, vted):
        self.__VTED = vted

    def add_extension(self, eid, g):
        if g:
            self.__repository.learn(g, ext_id=eid)
            self.__VTED.sync(force=True)
        else:
            raise AttributeError('no vocabulary provided')

    def update_extension(self, eid, g):
        if g:
            self.__repository.learn(g, ext_id=eid)
            self.__VTED.sync(force=True)
        else:
            raise AttributeError('no vocabulary provided')

    def delete_extension(self, eid):
        self.__repository.delete_extension(eid)

    def get_extension(self, eid):
        ttl = self.__repository.get_extension(eid)
        return ttl

    @property
    def extensions(self):
        return self.__repository.extensions

    def _get_thing_graph(self, td):
        g = td.resource.to_graph()

        def_g = ConjunctiveGraph(identifier=td.resource.node)
        for ns, uri in self.__repository.agora.fountain.prefixes.items():
            def_g.bind(ns, uri)

        for s, p, o in g:
            def_g.add((s, p, o))

        td_node = td.node

        if not list(def_g.objects(td.resource.node, CORE.describedBy)):
            def_g.add((td.resource.node, CORE.describedBy, td_node))
        return def_g

    def learn_descriptions(self, g, ted_path='/ted'):
        if not g:
            raise AttributeError('no description/s provided')

        ted = learn_descriptions_from(self.__repository, g)

        if self.__repository.base[-1] == '/':
            ted_path = ted_path.lstrip('/')
        eco_node = URIRef(self.__repository.base + ted_path)
        self.__VTED.update(ted, self._get_thing_graph, eco_node)

        return ted

    def delete_description(self, tdid, ted_path='/ted'):
        td_node = get_td_node(self.__repository, tdid)
        if td_node:
            try:
                ted_uri, eco = self.__VTED.ted_eco()
            except EnvironmentError:
                pass
            else:
                th_node = get_th_node(self.__repository, tdid)
                self.__VTED.remove_component(ted_uri, th_node)
                self.__repository.delete(td_node)
                self.__repository.delete(th_node)

            eco_node = URIRef(self.__repository.base + ted_path)
            self.__VTED.update(TED(), None, eco_node)

    def add_resource(self, uri, types):
        ted = self.ted
        matching_resources = filter(lambda r: r.node.toPython() == uri, self.ted.ecosystem.non_td_root_resources)
        if not matching_resources:
            agora = self.agora
            if not all([t in agora.fountain.types for t in types]):
                raise TypeError('Unknown type')

            g = Graph()
            prefixes = agora.fountain.prefixes

            uri_ref = URIRef(uri)
            if not types:
                dgw = DataGateway(agora.agora, ted, cache=None)
                rg, headers = dgw.loader(uri)

                type_uris = set([extend_uri(t, prefixes) for t in agora.fountain.types])

                resource_types = set(rg.objects(uri_ref, RDF.type))
                types = tuple(set.intersection(type_uris, resource_types))

            for t in types:
                g.add((uri_ref, RDF.type, URIRef(extend_uri(t, prefixes))))

            ted = self.add_descriptions(g)
            return ted
        else:
            raise AttributeError(uri)

    @property
    def descriptions(self):
        return self.ted.ecosystem.tds

    def add_description(self, id, types):
        # type: (basestring, iter) -> TED
        agora = self.agora
        if not all([t in agora.fountain.types for t in types]):
            raise TypeError('Unknown type')

        prefixes = agora.fountain.prefixes
        type_uris = [extend_uri(t, prefixes) for t in types]
        td = TD.from_types(types=type_uris, id=id)
        g = td.to_graph(th_nodes={})

        try:
            self.get_description(id)
        except AttributeError:
            ted = self.add_descriptions(g)
            tds = ted.ecosystem.tds
            if len(tds):
                return tds.pop()
        else:
            raise AttributeError(id)

    def delete_resource(self, uri):
        matching_resources = filter(lambda r: r.node.toPython() == uri, self.ted.ecosystem.non_td_root_resources)
        if matching_resources:
            self.repository.delete(uri)
        else:
            raise AttributeError(uri)

    def get_resource(self, uri):
        matching_resources = filter(lambda r: r.node.toPython() == uri, self.ted.ecosystem.non_td_root_resources)
        if matching_resources:
            return matching_resources.pop()
        else:
            raise AttributeError(uri)

    @property
    def resources(self):
        return self.ted.ecosystem.non_td_root_resources

    def __loader(self):
        def wrapper(*args, **kwargs):
            try:
                g = self.repository.pull(*args, **kwargs)
            except Exception:
                g = None

            return g if g else Graph().load(args[0], format='application/ld+json')

        return wrapper

    def get_description(self, tdid, fetch=False):
        td_node = get_td_node(self.__repository, tdid)
        g = self.__repository.pull(td_node, cache=True, infer=False, expire=300)
        for ns, uri in self.__repository.fountain.prefixes.items():
            g.bind(ns, uri)

        return TD.from_graph(g, td_node, {}, fetch=fetch, loader=self.__loader())

    def update_description(self, td, mediatype=JSONLD, ted_path='/ted'):
        if not td:
            raise AttributeError('no description/s provided')

        g = deserialize(td, mediatype)
        ted = learn_descriptions_from(self.__repository, g)

        if self.__repository.base[-1] == '/':
            ted_path = ted_path.lstrip('/')

        eco_node = URIRef(self.__repository.base + ted_path)
        self.__VTED.update(ted, self._get_thing_graph, eco_node)

        return ted

    def get_thing(self, tid, lazy=False):
        th_node = get_th_node(self.__repository, tid)
        g = self.__repository.pull(th_node, cache=True, infer=False, expire=300)

        for prefix, ns in self.__repository.fountain.prefixes.items():
            g.bind(prefix, ns)

        if not list(g.objects(th_node, CORE.describedBy)):
            td_node = get_td_node(self.__repository, tid)
            g.add((th_node, CORE.describedBy, td_node))

        return Resource.from_graph(g, th_node, {}, fetch=not lazy, loader=None if lazy else self.repository.pull)

    def discover(self, query, strict=False, **kwargs):
        # type: (str, bool, iter) -> TED
        if not query:
            raise AttributeError('no query provided')

        ted = discover_ecosystem(self.repository, self.VTED, query, reachability=not strict, **kwargs)
        return ted

    @property
    def ted(self):
        return self.get_ted()

    def get_ted(self, ted_uri=BNode(), fountain=None, lazy=False):
        local_node = URIRef(ted_uri)
        if fountain is None:
            fountain = self.repository.fountain
        known_types = fountain.types
        ns = self.repository.ns(fountain=fountain)
        ted = TED()
        g = ted.to_graph(node=local_node, abstract=True, fetch=False)
        for root_uri, td_uri in self.VTED.roots:
            root_uri = URIRef(root_uri)
            types = get_th_types(self.repository, root_uri, infer=True)
            valid_types = filter(lambda t: t.n3(ns) in known_types, types)
            if valid_types:
                r = Resource(root_uri, types=valid_types)
                if td_uri is None:
                    g.__iadd__(r.to_graph(abstract=True, fetch=False))
                g.add((ted.ecosystem.node, CORE.hasComponent, root_uri))

        for prefix, ns in fountain.prefixes.items():
            g.bind(prefix, ns)

        ted = TED.from_graph(g, fetch=not lazy, loader=None if lazy else self.repository.pull)
        return ted

    def __new__(cls, **kwargs):
        host = kwargs.get('host', None)
        if host:
            client_args = {'host': host}
            port = kwargs.get('port', None)
            if port:
                client_args['port'] = port
            gw = EcoGatewayClient(**client_args)
            return gw

        gw = super(EcoGateway, cls).__new__(cls)
        gw.__init__()

        engine_kwargs = kwargs.get('engine', {})
        agora = Agora(**engine_kwargs)
        repository_kwargs = kwargs.get('repository', {})
        gw.repository = Repository(**repository_kwargs)
        gw.repository.agora = agora
        gw.VTED = VTED(gw.repository)

        return gw
