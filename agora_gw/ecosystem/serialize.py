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

import copy
import json
import logging
from StringIO import StringIO

from agora_wot.blocks.td import TD
from agora_wot.gateway.proxy import path_data, ld_triples
from agora_wot.gateway.publish import build_graph_context
from agora_wot.ns import MAP, CORE, WOT
from pyld import jsonld
from rdflib import XSD, Graph, ConjunctiveGraph
from rdflib.plugins.parsers.notation3 import BadSyntax

# from agora_gw.data.repository import GEO
from agora_gw.data.graph import skolemize, deskolemize
from agora_gw.misc.utils import merge_two_dicts

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.gateway.serialize')

JSONLD = 'application/ld+json'
TURTLE = 'text/turtle'

CTX_NAMESPACES = {
    CORE: '@vocab',
    WOT: 'wot',
    # MAP: 'map',
    # GEO: 'geo'
}

CTX_MAPPINGS = {
    MAP.predicate: {
        '@type': '@id',
        'alias': 'predicate'
    },
    WOT.href: 'href',
    WOT.thingName: 'oid',
    WOT.interactionName: 'interaction',
    WOT.propertyName: 'pid',
    WOT.actionName: 'aid',
    WOT.eventName: 'eid',
    MAP.key: 'key',
    CORE.describes: {
        '@type': '@id',
        'alias': 'describes'
    },
    WOT.providesInteractionPattern: {
        '@type': '@id',
        'alias': 'interactions'
    },
    WOT.isAccessibleThrough: {
        '@type': '@id',
        'alias': 'links'
    },
    WOT.isReadableThrough: {
        '@type': '@id',
        'alias': 'read_links'
    },
    WOT.isWritableThrough: {
        '@type': '@id',
        'alias': 'write_links'
    },
    WOT.mediaType: 'mediaType',
    MAP.valuesTransformedBy: {
        '@type': '@id',
        'alias': 'transformed_by'
    },
    MAP.hasMapping: {
        '@type': '@id',
        'alias': 'mappings'
    },
    MAP.hasAccessMapping: {
        '@type': '@id',
        'alias': 'access_mappings'
    },
    MAP.mapsResourcesFrom: {
        '@type': '@id',
        'alias': 'source'
    },
    CORE.hasComponent: {
        '@type': '@id',
        'alias': 'components'
    },
    WOT.hasInputData: {
        '@type': '@id',
        'alias': 'input'
    },
    WOT.hasOutputData: {
        '@type': '@id',
        'alias': 'output'
    },
    WOT.isMeasuredIn: {
        '@type': '@id',
        'alias': 'units'
    },
    WOT.hasValueType: {
        '@type': '@id',
        'alias': 'datatype'
    },
    CORE.monitors: {
        '@type': '@id',
        'alias': 'monitors'
    },
    CORE.actsOn: {
        '@type': '@id',
        'alias': 'affects'
    },
    WOT.writable: {
        '@type': XSD.boolean,
        'alias': 'writable'
    },
    CORE.hasValue: {
        '@type': '@id',
        'alias': 'value'
    },
    MAP.rootMode: {
        '@type': XSD.boolean,
        'alias': 'root_mode'
    },
    MAP.order: {
        '@type': XSD.int,
        'alias': 'order'
    },
    CORE.ThingEcosystemDescription: 'TED',
    CORE.Ecosystem: 'Ecosystem',
    WOT.Thing: 'Thing',
    WOT.Link: 'Link',
    WOT.Property: 'Property',
    WOT.Action: 'Action',
    WOT.Event: 'Event',
    CORE.ThingDescription: 'TD',
    MAP.Mapping: 'Mapping',
    MAP.AccessMapping: 'AccessMapping'
}


def build_context(g):
    ctx = {}
    for term, map in CTX_MAPPINGS.items():
        if (None, term, None) in g or (None, None, term) in g:
            if isinstance(map, dict):
                alias = map['alias']
                ctx[alias] = {
                    '@type': str(map['@type']),
                    '@id': str(term)
                }
            else:
                ctx[map] = str(term)

    for s, p, o in g:
        try:
            matching_ns = filter(lambda ns: str(ns) in s or str(ns) in o, CTX_NAMESPACES).pop()
            ctx[CTX_NAMESPACES[matching_ns]] = str(matching_ns)
        except IndexError:
            pass

    return ctx


def replace_interaction_name(interaction_dict):
    if 'wot:propertyName' in interaction_dict:
        interaction_dict['pid'] = interaction_dict['wot:propertyName']
        del interaction_dict['wot:propertyName']
    elif 'wot:actionName' in interaction_dict:
        interaction_dict['aid'] = interaction_dict['wot:actionName']
        del interaction_dict['wot:actionName']
    elif 'wot:eventName' in interaction_dict:
        interaction_dict['eid'] = interaction_dict['wot:eventName']
        del interaction_dict['wot:eventName']


def graph_TED(ted, td_nodes=None, th_nodes=None, min=False, abstract=False, prefixes=None):
    g = ted.to_graph(td_nodes=td_nodes, th_nodes=th_nodes, abstract=abstract)
    if prefixes:
        for prefix, ns in prefixes.items():
            g.namespace_manager.bind(prefix, ns, replace=True, override=True)

    if not min:
        for root in filter(lambda x: isinstance(x, TD), ted.ecosystem.roots):
            root.resource.to_graph(g)

    return g


def serialize_TED(ted, format, td_nodes=None, th_nodes=None, min=False, abstract=False, prefixes=None):
    g = graph_TED(ted, td_nodes=td_nodes, th_nodes=th_nodes, min=min, abstract=abstract, prefixes=prefixes)
    return serialize_graph(g, format, frame=CORE.ThingEcosystemDescription)


def serialize_graph(g, format=TURTLE, frame=None, skolem=True):
    if skolem:
        cg = skolemize(g)
    else:
        cg = ConjunctiveGraph()
        cg.__iadd__(g)

    context = build_graph_context(g)

    if format == TURTLE:
        for prefix, uri in g.namespaces():
            if prefix in context:
                cg.bind(prefix, uri)

        return cg.serialize(format='turtle')

    ted_nquads = cg.serialize(format='nquads')
    ld = jsonld.from_rdf(ted_nquads)
    if frame is not None:
        ld = jsonld.frame(ld, {'context': context, '@type': str(frame)})
    ld = jsonld.compact(ld, context)

    return json.dumps(ld, indent=3, sort_keys=True)


def _ted_as_json_ld(sg):
    g = ConjunctiveGraph()
    g.__iadd__(sg)

    for res in g.query("""SELECT ?p ?name WHERE { ?p a <%s> ; <%s> ?name}""" % (WOT.Property, WOT.interactionName)):
        g.remove((res.p, WOT.interactionName, res.name))
        g.add((res.p, WOT.propertyName, res.name))

    for res in g.query("""SELECT ?p ?name WHERE { ?p a <%s> ; <%s> ?name}""" % (WOT.Action, WOT.interactionName)):
        g.remove((res.p, WOT.interactionName, res.name))
        g.add((res.p, WOT.actionName, res.name))

    for res in g.query("""SELECT ?p ?name WHERE { ?p a <%s> ; <%s> ?name}""" % (WOT.Event, WOT.interactionName)):
        g.remove((res.p, WOT.interactionName, res.name))
        g.add((res.p, WOT.eventName, res.name))

    context = build_context(g)

    if 'pid' in context:
        context['pid'] = str(WOT.interactionName)
    if 'aid' in context:
        context['aid'] = str(WOT.interactionName)
    if 'eid' in context:
        context['eid'] = str(WOT.interactionName)

    cg = skolemize(g)
    ted_nquads = cg.serialize(format='nquads')
    ld = jsonld.from_rdf(ted_nquads)

    td_frame = jsonld.compact(jsonld.frame(ld, {'context': context, '@type': str(CORE.ThingDescription)}),
                              context)

    td_context = td_frame['@context']
    del td_frame['@context']
    ted_frame = jsonld.compact(jsonld.frame(ld, {'context': context, '@type': str(CORE.ThingEcosystemDescription)}),
                               context)
    ted_context = ted_frame['@context']
    del ted_frame['@context']

    component_ids = []
    ted_components = ted_frame.get('describes', {}).get('components', [])
    if isinstance(ted_components, dict) or isinstance(ted_components, str):
        ted_components = [ted_components]
    for component in ted_components:
        # if it does not contain 'describedBy' it is a resource
        cid = component['@id'] if isinstance(component, dict) and 'describedBy' in component else component
        component_ids.append(cid)
    if component_ids:
        ted_frame['describes']['components'] = component_ids
    if '@graph' not in td_frame:
        source_td_frame = copy.deepcopy(td_frame)
        td_frame = {'@graph': []}
        if source_td_frame:
            td_frame['@graph'].append(source_td_frame)

    td_frame['@graph'].append(ted_frame)
    td_frame['@context'] = merge_two_dicts(td_context, ted_context)
    try:
        for pdata in path_data("$..interactions", td_frame['@graph']):
            if isinstance(pdata, list):
                for int_dict in pdata:
                    replace_interaction_name(int_dict)
            else:
                replace_interaction_name(pdata)
    except TypeError:
        pass

    return json.dumps(td_frame, indent=3, sort_keys=True)


def deserialize(data, format):
    g = Graph()
    if format == TURTLE:
        try:
            g.parse(StringIO(data), format='turtle')
        except BadSyntax as e:
            raise ValueError(e.message)
    else:
        try:
            ld_triples(json.loads(data), g)
        except Exception as e:
            raise ValueError(e.message)
    return deskolemize(g)
