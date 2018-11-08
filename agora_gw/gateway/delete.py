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
from agora_gw.gateway.errors import NotFoundError

__author__ = 'Fernando Serena'


def delete_access_mapping(eco_gw, id, amid):
    try:
        td = eco_gw.get_description(id)
    except AttributeError:
        raise NotFoundError(id)

    target_am = [am for am in td.access_mappings if str(am.id) == amid or am.endpoint.href.toPython() == amid]
    if not target_am:
        raise NotFoundError(amid)

    target_am = target_am.pop()
    td.remove_access_mapping(target_am)
    g = td.to_graph(th_nodes={})

    eco_gw.learn_descriptions(g)


def delete_mapping(eco_gw, id, mid):
    try:
        td = eco_gw.get_description(id)
    except AttributeError:
        raise NotFoundError(id)

    target_m = None
    target_am = None
    for am in td.access_mappings:
        m = filter(lambda m: str(m.id) == mid, am.mappings)
        if m:
            target_m = m.pop()
            target_am = am
            break

    if not target_m:
        raise NotFoundError(mid)

    target_am.mappings.remove(target_m)
    g = td.to_graph(th_nodes={})

    eco_gw.learn_descriptions(g)
