from StringIO import StringIO

from agora import Agora
from agora.server import Client
from agora_wot.blocks.eco import request_loader
from agora_wot.blocks.resource import Resource
from agora_wot.blocks.td import TD
from agora_wot.blocks.ted import TED

from agora_gw.data.graph import deskolemize
from agora_gw.data.repository import CORE
from agora_gw.gateway.abstract import AbstractGateway
from rdflib import Graph, RDF


class GatewayClient(Client, AbstractGateway):

    def __loader(self, uri):
        g = request_loader(uri)
        return deskolemize(g)

    def add_extension(self, eid, g):
        response = self._put_request('extensions/{}'.format(eid), g.serialize(format='turtle'))
        return response

    def update_extension(self, eid, g):
        response = self._put_request('extensions/{}'.format(eid), g.serialize(format='turtle'))
        return response

    def delete_extension(self, eid):
        return self._delete_request('extensions/{}'.format(eid))

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

    def add_description(self, g):
        response = self._post_request('descriptions', g.serialize(format='turtle'), content_type='text/turtle',
                                      accept='text/turtle')
        g = Graph()
        g.parse(StringIO(response), format='turtle')
        ted = TED.from_graph(g, loader=self.__loader)
        return ted

    def get_description(self, tdid, fetch=True):
        response = self._get_request('descriptions/{}'.format(tdid), accept='text/turtle')
        g = Graph()
        g.parse(StringIO(response), format='turtle')
        return TD.from_graph(g, list(g.subjects(RDF.type, CORE.ThingDescription)).pop(), {}, fetch=fetch)

    def update_description(self, td):
        pass

    def delete_description(self, tdid):
        pass

    def get_thing(self, tid):
        response = self._get_request('things/{}'.format(tid), accept='text/turtle')
        g = Graph()
        g.parse(StringIO(response), format='turtle')
        g = deskolemize(g)
        return Resource.from_graph(g)

    def discover(self, query, strict=False, **kwargs):
        response = self._post_request('discover?min', query, content_type='text/plain', accept='text/turtle')
        g = Graph()
        g.parse(StringIO(response), format='turtle')
        g = deskolemize(g)
        ted = TED.from_graph(g, fetch=True, loader=self.__loader)
        return ted

    @property
    def ted(self):
        response = self._get_request('ted', accept='text/turtle')
        g = Graph()
        g.parse(StringIO(response), format='turtle')
        g = deskolemize(g)
        ted = TED.from_graph(g)
        return ted

    def __init__(self, host='localhost', port=8000):
        super(GatewayClient, self).__init__(host, port)
        self._agora = Agora(planner_host=host, planner_port=port)


def client(host='localhost', port=8000):
    # type: (str, int) -> GatewayClient
    return GatewayClient(host, port)