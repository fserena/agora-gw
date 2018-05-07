from abc import abstractmethod

from agora_wot.blocks.resource import Resource
from agora_wot.blocks.td import TD
from agora_wot.blocks.ted import TED
from rdflib import Graph


class AbstractGateway(object):
    @abstractmethod
    def add_extension(self, eid, g):
        # type: (Graph) -> iter
        raise NotImplementedError

    @abstractmethod
    def update_extension(self, eid, g):
        # type: (str, Graph) -> None
        raise NotImplementedError

    @abstractmethod
    def delete_extension(self, eid):
        # type: (str) -> None
        raise NotImplementedError

    @abstractmethod
    def get_extension(self, eid):
        # type: (str) -> Graph
        raise NotImplementedError

    @property
    @abstractmethod
    def extensions(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def agora(self):
        raise NotImplementedError

    @abstractmethod
    def add_description(self, g):
        # type: (Graph) -> TED
        raise NotImplementedError

    @abstractmethod
    def get_description(self, tdid):
        # type: (str) -> TD
        raise NotImplementedError

    @abstractmethod
    def update_description(self, td):
        # type: (str) -> None
        raise NotImplementedError

    @abstractmethod
    def delete_description(self, tdid):
        raise NotImplementedError

    @abstractmethod
    def get_thing(self, tid):
        # type: (str) -> Resource
        raise NotImplementedError

    @abstractmethod
    def discover(self, query):
        # type: (str) -> str
        raise NotImplementedError
