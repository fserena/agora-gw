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

from abc import abstractmethod

from agora_wot.blocks.resource import Resource
from agora_wot.blocks.td import TD
from agora_wot.blocks.ted import TED
from rdflib import Graph

__author__ = 'Fernando Serena'


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

    @property
    @abstractmethod
    def ted(self):
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
