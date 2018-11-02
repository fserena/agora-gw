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

from StringIO import StringIO

from agora import Agora
from agora.server import Client
from agora_wot.blocks.eco import request_loader
from agora_wot.blocks.resource import Resource
from agora_wot.blocks.td import TD
from agora_wot.blocks.ted import TED

from agora_gw.data.graph import deskolemize
from agora_gw.data.repository import CORE
from agora_gw.gateway.abstract import AbstractEcoGateway
from rdflib import Graph, RDF, URIRef

__author__ = 'Fernando Serena'


class EcoGatewayClient(Client, AbstractEcoGateway):

    def __loader(self, uri):
        g = request_loader(uri)
        if g:
            return deskolemize(g)

    def add_extension(self, eid, g):
        response = self._put_request('extensions/{}'.format(eid), g.serialize(format='turtle'))
        return response

    def update_extension(self, eid, g):
        response = self._put_request('extensions/{}'.format(eid), g.serialize(format='turtle'))
        return response

    def delete_extension(self, eid):
        try:
            self._delete_request('extensions/{}'.format(eid))
        except IOError as e:
            raise AttributeError(e.message['text'])

    def get_extension(self, eid):
        g = Graph()
        response = self._get_request('extensions/{}'.format(eid), accept='text/turtle')
        g.parse(StringIO(response), format='turtle')
        return g

    @property
    def extensions(self):
        return self._get_request('extensions')

    @property
    def agora(self):
        return self._agora

    @property
    def descriptions(self):
        response = self._get_request('descriptions', accept='text/turtle')
        g = Graph()
        g.parse(StringIO(response), format='turtle')
        all_tds = set()
        for td_uri in g.subjects(RDF.type, CORE.ThingDescription):
            try:
                td_g = self.__loader(td_uri)
                g.__iadd__(td_g)
                th_uri = list(g.objects(td_uri, CORE.describes)).pop()
                th_g = self.__loader(th_uri)
                g.__iadd__(th_g)
                all_tds.add(TD.from_graph(g, td_uri, {}, fetch=True))
            except Exception:
                pass
        return all_tds

    def learn_descriptions(self, g):
        try:
            response = self._post_request('descriptions', g.serialize(format='turtle'), content_type='text/turtle',
                                          accept='text/turtle')
            g = Graph()
            g.parse(StringIO(response), format='turtle')
            ted = TED.from_graph(g, loader=self.__loader)
            return ted
        except IOError as e:
            raise AttributeError(e.message['text'])

    def add_description(self, id, types):
        td = TD.from_types(id, types)
        try:
            response = self._post_request('descriptions', td.to_graph().serialize(format='turtle'),
                                          content_type='text/turtle',
                                          accept='text/turtle')

            g = Graph()
            g.parse(StringIO(response), format='turtle')
            ted = TED.from_graph(g, loader=self.__loader)
            tds = ted.ecosystem.tds
            if len(tds):
                return list(tds).pop()
        except IOError:
            pass

        raise AttributeError(id)

    def get_description(self, tdid, fetch=True):
        response = self._get_request('descriptions/{}'.format(tdid), accept='text/turtle')
        g = Graph()
        g.parse(StringIO(response), format='turtle')
        g = deskolemize(g)
        return TD.from_graph(g, list(g.subjects(RDF.type, CORE.ThingDescription)).pop(), {}, fetch=fetch)

    def update_description(self, td):
        pass

    def delete_description(self, tdid):
        try:
            self._delete_request('descriptions/{}'.format(tdid))
        except IOError as e:
            raise AttributeError(e.message['text'])

    def add_resource(self, uri, types):
        r = Resource(URIRef(uri), types)
        try:
            response = self._post_request('descriptions', r.to_graph().serialize(format='turtle'),
                                          content_type='text/turtle',
                                          accept='text/turtle')
            g = Graph()
            g.parse(StringIO(response), format='turtle')
            ted = TED.from_graph(g, loader=self.__loader)
            all_resources = ted.ecosystem.non_td_root_resources
            if all_resources:
                return list(all_resources).pop()
        except IOError as e:
            raise AttributeError(e.message['text'])

        raise AttributeError(uri)

    def delete_resource(self, uri):
        try:
            self._delete_request(u'resources/{}'.format(uri))
        except IOError as e:
            raise AttributeError(e.message['text'])

    def get_resource(self, uri):
        response = self._get_request(u'resources/{}'.format(uri), accept='text/turtle')
        g = Graph()
        g.parse(StringIO(response), format='turtle')
        return Resource(URIRef(uri), types=list(g.objects(URIRef(uri), RDF.type)))

    @property
    def resources(self):
        response = self._get_request('resources', accept='text/turtle')
        g = Graph()
        g.parse(StringIO(response), format='turtle')
        g = deskolemize(g)
        all_resources = set()
        for uri in set(g.subjects()):
            r_types = list(g.objects(predicate=RDF.type))
            all_resources.add(Resource(URIRef(uri), r_types))
        return all_resources

    def get_thing(self, tid):
        response = self._get_request('things/{}'.format(tid), accept='text/turtle')
        g = Graph()
        g.parse(StringIO(response), format='turtle')
        g = deskolemize(g)

        try:
            node = list(g.subjects(predicate=CORE.describedBy)).pop()
            return Resource.from_graph(g, node, {})
        except IndexError:
            raise AttributeError(tid)

    def discover(self, query, strict=False, lazy=True, **kwargs):
        path = 'discover'
        lazy_arg = 'min' if lazy else ''
        strict_arg = 'strict' if strict else ''
        args = '&'.join([lazy_arg, strict_arg])
        if args:
            path += '?' + args
        try:
            response = self._post_request(path, query, content_type='text/plain', accept='text/turtle')
            g = Graph()
            g.parse(StringIO(response), format='turtle')
            g = deskolemize(g)
            ted = TED.from_graph(g, fetch=lazy, loader=self.__loader)
            return ted
        except IOError as e:
            raise AttributeError(e.message)

    @property
    def ted(self):
        response = self._get_request('ted', accept='text/turtle')
        g = Graph()
        g.parse(StringIO(response), format='turtle')
        g = deskolemize(g)
        ted = TED.from_graph(g)
        return ted

    def __init__(self, host='localhost', port=8000):
        super(EcoGatewayClient, self).__init__(host, port)
        self._agora = Agora(planner_host=host, planner_port=port)


def client(host='localhost', port=8000):
    # type: (str, int) -> GatewayClient
    return EcoGatewayClient(host, port)
