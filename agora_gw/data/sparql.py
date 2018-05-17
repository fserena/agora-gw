import json
import logging
import os
import traceback
from urllib import urlencode

import requests
from SPARQLWrapper import SPARQLWrapper, JSON, N3
from redis_cache import cache_it, DEFAULT_EXPIRY, SimpleCache


SPARQL_HOST = os.environ.get('SPARQL_HOST')
UPDATE_HOST = os.environ.get('UPDATE_HOST')

QUERY_CACHE_HOST = os.environ.get('QUERY_CACHE_HOST', 'localhost')
QUERY_CACHE_NUMBER = int(os.environ.get('QUERY_CACHE_NUMBER', 8))

log = logging.getLogger('agora.gateway.data.sparql')


def _update(q, update_host=UPDATE_HOST):
    res = requests.post(update_host,
                        headers={
                            'Accept': 'text/plain,*/*;q=0.9',
                            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
                        },
                        data=urlencode({'update': q.encode('utf-8')}))
    return res


def _query(q, cache=None, infer=True, expire=DEFAULT_EXPIRY, namespace=None, sparql_host=SPARQL_HOST):
    def wrapper(q):
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

    if cache is not None:
        try:
            ret = cache_it(cache=cache, expire=expire, namespace=namespace)(wrapper)(q)
        except UnicodeDecodeError:
            traceback.print_exc()
            return []
    else:
        ret = wrapper(q)

    try:
        return json.loads(ret)
    except ValueError:
        return ret


class SPARQL(object):
    def __init__(self, sparql_host=None, update_host=None, namespace='gateway', **kwargs):
        self.sparql_host = sparql_host or SPARQL_HOST
        self.update_host = update_host or UPDATE_HOST
        self.cache = SimpleCache(limit=10000, expire=60 * 60, hashkeys=True, host=QUERY_CACHE_HOST, port=6379,
                                 db=QUERY_CACHE_NUMBER, namespace=namespace)

    def query(self, q, cache=True, infer=True, expire=DEFAULT_EXPIRY, namespace=None):
        cache = self.cache if cache else None
        return _query(q, cache=cache, infer=infer, expire=expire, namespace=namespace, sparql_host=self.sparql_host)

    def update(self, q):
        return _update(q, update_host=self.update_host)

    def expire_cache(self, namespace=None):
        if namespace is not None:
            self.cache.flush_namespace(namespace)
        else:
            self.cache.connection.flushdb()
