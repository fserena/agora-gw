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

from agora import RedisCache, Agora
from agora.engine.plan.agp import extend_uri
from agora_wot.blocks.td import TD
from agora_wot.gateway import DataGateway

from agora_gw.gateway.add import add_access_mapping, add_mapping
from agora_gw.gateway.discover import discover_seeds
from agora_gw.gateway.eco import EcoGateway, AbstractEcoGateway
from agora_gw.gateway.errors import GatewayError, NotFoundError, ConflictError

__author__ = 'Fernando Serena'


class Gateway(object):
    def __init__(self, **kwargs):
        self.__eco = EcoGateway(**kwargs)
        if 'data_cache' in kwargs:
            self.__cache = RedisCache(**kwargs['data_cache'])
        else:
            self.__cache = None

    @property
    def eco(self):
        # type: () -> AbstractEcoGateway
        return self.__eco

    @property
    def ted(self):
        # type: () -> TED
        try:
            return self.__eco.ted
        except GatewayError as e:
            raise e
        except Exception as e:
            raise GatewayError(e.message)

    @property
    def data_cache(self):
        return self.__cache

    @property
    def agora(self):
        return self.__eco.agora

    @data_cache.setter
    def data_cache(self, c):
        self.__cache = c

    def __data_proxy(self, item):
        def wrapper(*args, **kwargs):
            if item == 'query' or item == 'fragment':
                query = args[0]
                ted = self.__eco.discover(query, strict=False, lazy=False)
                if 'cache' in kwargs:
                    self.__cache = kwargs['cache']
                    del kwargs['cache']
                dgw = DataGateway(self.__eco.agora, ted, cache=self.__cache, static_fountain=True)
                return dgw.__getattribute__(item)(*args, **kwargs)

        return wrapper

    def data(self, query, strict=False, lazy=False, **kwargs):
        ted = self.__eco.discover(query, strict=strict, lazy=lazy)
        if 'cache' in kwargs:
            self.__cache = kwargs['cache']
            del kwargs['cache']
        dgw = DataGateway(self.agora, ted, cache=self.__cache, **kwargs)
        return dgw

    def seeds(self, query, host='agora', port=80, **kwargs):
        try:
            return discover_seeds(self.__eco, query, host, port, **kwargs)
        except GatewayError as e:
            raise e
        except Exception as e:
            raise GatewayError(e.message)

    def add_resource(self, uri, types):
        prefixes = self.agora.fountain.prefixes
        try:
            types = map(lambda t: extend_uri(t, prefixes), types)
            return self.__eco.add_resource(uri, types)
        except TypeError as e:
            raise NotFoundError(e.message)
        except AttributeError as e:
            raise ConflictError(e.message)
        except Exception as e:
            raise GatewayError(e.message)

    def add_description(self, id, types):
        prefixes = self.agora.fountain.prefixes
        try:
            types = map(lambda t: extend_uri(t, prefixes), types)
            return self.__eco.add_description(id, types)
        except TypeError as e:
            raise NotFoundError(e.message)
        except AttributeError as e:
            raise ConflictError(e.message)
        except Exception as e:
            raise GatewayError(e.message)

    def update_description(self, td):
        try:
            ted = self.__eco.add_descriptions(td.to_graph())
            return filter(lambda t: t.id == td.id, ted.ecosystem.tds).pop()
        except Exception as e:
            raise GatewayError(e.message)

    def learn_descriptions(self, g):
        try:
            return self.__eco.learn_descriptions(g)
        except Exception as e:
            raise GatewayError(e.message)

    def get_extension(self, eid):
        if eid in self.__eco.extensions:
            try:
                return self.__eco.get_extension(eid)
            except GatewayError as e:
                raise e
            except Exception as e:
                raise GatewayError(e.message)

        raise NotFoundError('{}?'.format(eid))

    @property
    def extensions(self):
        return self.__eco.extensions

    def get_type(self, ty):
        try:
            return self.__eco.agora.fountain.get_type(ty)
        except TypeError:
            raise NotFoundError(ty)
        except Exception as e:
            raise GatewayError(e.message)

    def get_property(self, pr):
        try:
            return self.__eco.agora.fountain.get_property(pr)
        except TypeError:
            raise NotFoundError(pr)
        except Exception as e:
            raise GatewayError(e.message)

    def paths(self, source, dest):
        types = self.__eco.agora.fountain.types
        if dest not in types:
            raise NotFoundError('{}'.format(dest))
        if source not in types:
            raise NotFoundError('{}'.format(source))
        try:
            return self.__eco.agora.fountain.get_paths(dest, force_seed=[
                ('<{}-uri>'.format(source.lower()).replace(':', '-'), source)])
        except Exception as e:
            raise GatewayError(e.message)

    def get_description(self, id):
        try:
            return self.__eco.get_description(id)
        except AttributeError:
            raise NotFoundError(id)
        except Exception as e:
            raise GatewayError(e.message)

    @property
    def resources(self):
        return self.__eco.resources

    def get_resource(self, uri):
        try:
            return self.__eco.get_resource(uri)
        except AttributeError:
            raise NotFoundError(id)
        except Exception as e:
            raise GatewayError(e.message)

    def delete_description(self, id):
        try:
            td = self.__eco.get_description(id)
        except AttributeError:
            td = None
        if td is None:
            raise NotFoundError('Unknown description: {}'.format(id))

        try:
            self.__eco.delete_description(id)
        except Exception as e:
            raise GatewayError(e.message)

    def get_thing(self, id):
        try:
            return self.__eco.get_thing(id)
        except AttributeError:
            raise NotFoundError(id)
        except Exception as e:
            raise GatewayError(e.message)

    def delete_resource(self, uri):
        try:
            self.__eco.delete_resource(uri)
        except AttributeError:
            raise NotFoundError(uri)
        except Exception as e:
            raise GatewayError(e.message)

    @property
    def args(self):
        try:
            ted = self.__eco.ted

            td_roots = list(filter(lambda r: isinstance(r, TD), ted.ecosystem.roots))
            res = {}
            for td in td_roots:
                td_root_vars = ted.ecosystem.root_vars(td)
                if td_root_vars:
                    res[td.id] = list(td_root_vars)
            return res
        except Exception as e:
            raise GatewayError(e.message)

    def learn_extension(self, name, g, replace=True):
        if not replace and name in self.__eco.extensions:
            raise ConflictError(name)

        try:
            self.__eco.add_extension(name, g)
        except Exception as e:
            raise GatewayError(e.message)

    def forget_extension(self, name):
        if name not in self.__eco.extensions:
            raise NotFoundError(name)

        try:
            self.__eco.delete_extension(name)
        except Exception as e:
            raise GatewayError(e.message)

    def discover(self, query, strict=False, **kwargs):
        try:
            return self.__eco.discover(query, strict, **kwargs)
        except Exception as e:
            raise GatewayError(e.message)

    def add_access_mapping(self, td_id, link):
        try:
            return add_access_mapping(self.__eco, td_id, link)
        except GatewayError:
            raise
        except Exception as e:
            raise GatewayError(e.message)

    def add_mapping(self, id, amid, predicate, key, jsonpath=None, root=False, transformed_by=None):
        try:
            return add_mapping(self.__eco, id, amid, predicate, key, jsonpath=jsonpath, root=root, transformed_by=transformed_by)
        except GatewayError:
            raise
        except Exception as e:
            raise GatewayError(e.message)

    @property
    def descriptions(self):
        return self.__eco.descriptions

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        Agora.close()
