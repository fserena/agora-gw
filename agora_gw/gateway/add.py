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
from agora.engine.plan.agp import extend_uri
from agora_wot.blocks.endpoint import Endpoint
from agora_wot.blocks.td import AccessMapping, ResourceTransform, Mapping
from rdflib import URIRef

from agora_gw.gateway.eco import EcoGateway
from agora_gw.gateway.errors import NotFoundError, ConflictError

__author__ = 'Fernando Serena'


def add_access_mapping(eco_gw, td_id, link):
    # type: (EcoGateway, basestring, basestring) -> AccessMapping
    td = eco_gw.get_description(td_id)
    if not td:
        raise NotFoundError(td_id)

    endpoint_hrefs = map(lambda e: u'{}'.format(e.href), td.endpoints)
    if link in endpoint_hrefs:
        raise ConflictError('Link already mapped')

    e = Endpoint(href=link)
    am = AccessMapping(e)
    td.add_access_mapping(am)
    g = td.to_graph(th_nodes={})

    eco_gw.learn_descriptions(g)
    return am


def add_mapping(eco_gw, id, amid, predicate, key, jsonpath=None, root=False, transformed_by=None):
    td = eco_gw.get_description(id)
    if not td:
        raise NotFoundError(id)

    transform_td = None
    if transformed_by:
        try:
            transform_td = eco_gw.get_description(transformed_by)
        except AttributeError:
            raise NotFoundError(transformed_by)

    target_am = [am for am in td.access_mappings if str(am.id) == amid or am.endpoint.href.toPython() == amid]
    if not target_am:
        raise NotFoundError(amid)

    target_am = target_am.pop()

    m = Mapping(key=key, predicate=URIRef(extend_uri(predicate, eco_gw.agora.fountain.prefixes)), root=root,
                path=jsonpath,
                transform=ResourceTransform(transform_td) if transform_td else None)
    target_am.mappings.add(m)
    g = td.to_graph(th_nodes={})

    eco_gw.learn_descriptions(g)
    return m
