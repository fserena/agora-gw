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

import json
import logging
import os
import re
import traceback
from urllib import urlencode

import requests
from SPARQLWrapper import SPARQLWrapper, JSON, N3
from agora import get_kv
from rdflib import ConjunctiveGraph, URIRef
from redis_cache import cache_it, DEFAULT_EXPIRY, SimpleCache as BaseSimpleCache, RedisNoConnException, \
    RedisConnect as BaseRedisConnect, redis

from agora.engine.utils.graph import get_triple_store

__author__ = 'Fernando Serena'


SPARQL_HOST = os.environ.get('SPARQL_HOST')
UPDATE_HOST = os.environ.get('UPDATE_HOST')

QUERY_CACHE_HOST = os.environ.get('QUERY_CACHE_HOST', 'localhost')
QUERY_CACHE_PORT = int(os.environ.get('QUERY_CACHE_PORT', 6379))
QUERY_CACHE_NUMBER = int(os.environ.get('QUERY_CACHE_NUMBER', 8))

log = logging.getLogger('agora.gateway.data.sparql')


def _update(q, update_host=UPDATE_HOST):
    def remote():
        res = requests.post(update_host,
                            headers={
                                'Accept': 'text/plain,*/*;q=0.9',
                                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
                            },
                            data=urlencode({'update': q.encode('utf-8')}))
        return res

    def local():
        from rdflib.plugins.sparql.parser import parseUpdate
        from rdflib.plugins.sparql.algebra import translateUpdate

        graph = update_host
        update_str = q

        parsetree = parseUpdate(update_str)
        query = translateUpdate(parsetree)
        try:
            context_aware = query[0].get('delete', {}).get('quads', {}).keys()
        except AttributeError:
            context_aware = None
        if context_aware:
            update_str = re.sub(r'{(.*)GRAPH(.*)WHERE', 'WHERE', update_str)
            delete_graph = graph.get_context(URIRef(context_aware[0]))
        else:
            delete_graph = graph

        delete_graph.update(update_str)

    query_fn = local if isinstance(update_host, ConjunctiveGraph) else remote
    return query_fn()


def _query(q, cache=None, infer=True, expire=DEFAULT_EXPIRY, namespace=None, sparql_host=SPARQL_HOST):
    def remote(q):
        sparql = SPARQLWrapper(sparql_host)
        sparql.setRequestMethod("postdirectly")
        sparql.setMethod('POST')

        log.debug(u'Querying: {}'.format(q))
        sparql.setQuery(q)

        sparql.addCustomParameter('infer', str(infer).lower())
        if not ('construct' in q.lower()):
            sparql.setReturnFormat(JSON)
        else:
            sparql.setReturnFormat(N3)

        try:
            results = sparql.query().convert()
        except Exception as e:
            print q, e.message
            raise e

        if isinstance(results, str):
            return results.decode('utf-8')
        else:
            if 'results' in results:
                return json.dumps(results["results"]["bindings"]).decode('utf-8')
            else:
                return json.dumps(results['boolean']).decode('utf-8')

    def local(q):
        from rdflib.plugins.sparql.parser import parseQuery
        from rdflib.plugins.sparql.algebra import translateQuery

        graph = sparql_host
        query_str = q
        parsetree = parseQuery(query_str)
        query = translateQuery(parsetree)
        dataset = query.algebra['datasetClause']
        if dataset is not None and len(dataset) == 1:
            graph = graph.get_context(dataset.pop()['default'])
            query_str = re.sub(r'FROM(.*)>', '', query_str)

        results = json.loads(graph.query(query_str).serialize(format='json'))

        if 'results' in results:
            return json.dumps(results["results"]["bindings"]).decode('utf-8')
        else:
            return json.dumps(results['boolean']).decode('utf-8')

    query_fn = local if isinstance(sparql_host, ConjunctiveGraph) else remote

    if cache is not None:
        try:
            ret = cache_it(cache=cache, expire=expire, namespace=namespace)(query_fn)(q)
        except UnicodeDecodeError:
            traceback.print_exc()
            return []
    else:
        ret = query_fn(q)

    try:
        return json.loads(ret)
    except ValueError:
        return ret


class SPARQL(object):
    def __init__(self, sparql_host=None, update_host=None, cache={}, base=None, path=None, persist_mode=False):
        self.sparql_host = sparql_host or SPARQL_HOST
        self.update_host = update_host or UPDATE_HOST

        if not self.sparql_host:
            self.sparql_host = get_triple_store(persist_mode=persist_mode, base=base, path=path)
            self.update_host = self.sparql_host

        self.cache = SimpleCache(limit=10000, expire=60 * 60, hashkeys=True, **cache)

    def query(self, q, cache=True, infer=True, expire=DEFAULT_EXPIRY, namespace=None):
        cache = self.cache if cache else None

        return _query(q, cache=cache, infer=infer, expire=expire, namespace=namespace, sparql_host=self.sparql_host)

    def update(self, q):
        _update(q, update_host=self.update_host)
        self.expire_cache()

    def expire_cache(self, namespace=None):
        if self.cache.connection:
            if namespace is not None:
                self.cache.flush_namespace(namespace)
            else:
                self.cache.connection.flushdb()


class SimpleCache(BaseSimpleCache):
    def __init__(self,
                 limit=10000,
                 expire=DEFAULT_EXPIRY,
                 hashkeys=False,
                 host=None,
                 port=None,
                 db=None,
                 password=None,
                 file=None,
                 base=None,
                 namespace="agora"):

        self.limit = limit  # No of json encoded strings to cache
        self.expire = expire  # Time to keys to expire in seconds
        self.prefix = namespace
        self.host = host
        self.port = port
        self.db = db
        self.file = file
        self.base = base

        try:
            self.connection = RedisConnect(host=self.host,
                                           port=self.port,
                                           db=self.db,
                                           password=password,
                                           file=self.file,
                                           base=self.base).connect()
        except RedisNoConnException, e:
            self.connection = None
            pass

        # Should we hash keys? There is a very small risk of collision invloved.
        self.hashkeys = hashkeys


class RedisConnect(BaseRedisConnect):
    def __init__(self, host=None, port=None, db=None, password=None, file=None, base=None):
        self.host = host if host else 'localhost'
        self.port = port if port else 6379
        self.db = db if db else 0
        self.base = base
        self.file = file
        self.password = password

    def connect(self):
        if self.file is None:
            try:
                redis.StrictRedis(host=self.host, port=self.port, password=self.password).ping()
            except redis.ConnectionError as e:
                raise RedisNoConnException("Failed to create connection to redis",
                                           (self.host,
                                            self.port)
                                           )
            return redis.StrictRedis(host=self.host,
                                     port=self.port,
                                     db=self.db,
                                     password=self.password)
        else:
            return get_kv(persist_mode=True, redis_file=self.file, redis_db=self.db, base=self.base)
