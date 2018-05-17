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
import logging
import os

from agora import RedisCache
from agora_wot.gateway import Gateway as DataGateway, AbstractGateway as AbstractDataGateway

from agora_gw.gateway import Gateway as EcoGateway, AbstractGateway as AbstractEcoGateway

__author__ = 'Fernando Serena'

LOG_LEVEL = int(os.environ.get('LOG_LEVEL', logging.DEBUG))


class EcoGatewayAdapter(object):
    def __init__(self, gw):
        self.__gw = gw

    def __getattribute__(self, item):
        if item.startswith('_'):
            return super(EcoGatewayAdapter, self).__getattribute__(item)

        if item == 'get_description':
            return lambda *args, **kwargs: self.__gw.get_description(*args, lazy=False)
        elif item == 'get_thing':
            return lambda *args, **kwargs: self.__gw.get_thing(*args, lazy=False)

        return self.__gw.__getattribute__(item)

    def get_description(self):
        def wrapper(*args, **kwargs):
            self.__gw.get_description(*args, lazy=False, **kwargs)

        return wrapper

    def get_thing(self):
        def wrapper(*args, **kwargs):
            self.__gw.get_thing(*args, lazy=False, **kwargs)

        return wrapper


class Gateway(AbstractEcoGateway, AbstractDataGateway):
    def __init__(self, **kwargs):
        self.__eco = EcoGateway(**kwargs)
        if isinstance(self.__eco, EcoGateway):
            self.__eco = EcoGatewayAdapter(self.__eco)
        if 'cache' in kwargs:
            self.__cache = RedisCache(**kwargs['cache'])
        else:
            self.__cache = None

    def __data_proxy(self, item):
        def wrapper(*args, **kwargs):
            if item == 'query' or item == 'fragment':
                query = args[0]
                ted = self.__eco.discover(query, strict=False, lazy=False)
                dgw = DataGateway(self.__eco.agora, ted, cache=self.__cache, static_fountain=True)
                return dgw.__getattribute__(item)(*args, **kwargs)

        return wrapper

    def __getattribute__(self, item):
        if item.startswith('_'):
            return super(Gateway, self).__getattribute__(item)
        elif hasattr(self.__eco, item):
            return self.__eco.__getattribute__(item)
        else:
            if hasattr(AbstractDataGateway, item):
                return self.__data_proxy(item)

            return super(Gateway, self).__getattribute__(item)

    def data(self, query, **kwargs):
        ted = self.__eco.discover(query, strict=False, lazy=False)
        dgw = DataGateway(self.__eco.agora, ted, cache=self.__cache, **kwargs)
        return dgw