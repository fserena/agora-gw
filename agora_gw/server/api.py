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
from agora.engine.fountain.onto import DuplicateVocabulary
from agora.server import HTML, JSON
from agora_wot.blocks.resource import Resource
from agora_wot.blocks.ted import TED
from flask import Flask, request, jsonify, make_response, url_for
from flask_negotiate import produces, consumes
from rdflib import URIRef, ConjunctiveGraph, RDF

from agora_gw.data import R
from agora_gw.data.repository import CORE, REPOSITORY_BASE
from agora_gw.ecosystem.description import learn_descriptions_from, VTED, get_td_node, \
    get_th_node, get_th_types
from agora_gw.ecosystem.discover import discover_ecosystem
from agora_gw.ecosystem.serialize import serialize_TED, JSONLD, TURTLE, deserialize, serialize_graph

__author__ = 'Fernando Serena'

DISCOVERY_MIMES = [JSONLD, TURTLE]
DESCRIPTION_MIMES = [JSONLD, TURTLE]


def request_wants_turtle():
    best = request.accept_mimetypes \
        .best_match(DISCOVERY_MIMES)
    return best == TURTLE and \
           request.accept_mimetypes[best] > \
           request.accept_mimetypes[JSONLD]


@consumes(TURTLE)
def learn():
    vocabulary = request.data
    try:
        if vocabulary:
            g = deserialize(vocabulary, request.content_type)
            ext_id = R.learn(g)
            VTED.sync(force=True)
            response = make_response()
            response.headers['Location'] = url_for('get_extension', id=ext_id, _external=True)
            response.status_code = 201
            return response
        else:
            reason = 'no vocabulary provided'
    except DuplicateVocabulary as e:
        reason = e.message
    except ValueError as e:
        reason = e.message

    response = jsonify({'status': 'error', 'reason': reason})
    response.status_code = 400
    return response


@consumes(TURTLE)
def learn_with_id(id):
    vocabulary = request.data
    try:
        if vocabulary:
            g = deserialize(vocabulary, request.content_type)
            R.learn(g, ext_id=id)
            VTED.sync(force=True)
            response = make_response()
            response.headers['Location'] = url_for('get_extension', id=id, _external=True)
            response.status_code = 201
            return response
        else:
            reason = 'no vocabulary provided'
    except DuplicateVocabulary as e:
        reason = e.message
    except ValueError as e:
        reason = e.message

    response = jsonify({'status': 'error', 'reason': reason})
    response.status_code = 400
    return response


@produces(TURTLE, HTML)
def get_extension(id):
    ttl = R.get_extension(id)
    response = make_response(ttl)
    response.headers['Content-Type'] = 'text/turtle'
    return response


def get_extensions():
    extensions = R.extensions
    return jsonify(extensions)


def delete_extension(id):
    R.delete_extension(id)
    response = make_response()
    return response


@produces(*DISCOVERY_MIMES)
def discover():
    query = request.data
    try:
        if query:
            reachability = request.args.get('strict')
            reachability = False if reachability is not None else True
            ted = discover_ecosystem(query, reachability=reachability)
            format = TURTLE if request_wants_turtle() else JSONLD

            min = request.args.get('min')
            min = True if min is not None else False
            ted_str = serialize_TED(ted, format, min=min, abstract=min, prefixes=R.fountain.prefixes)

            own_base = unicode(request.url_root)
            ted_str = ted_str.decode('utf-8').replace(REPOSITORY_BASE + u'/', own_base)

            response = make_response(ted_str)
            response.headers['Content-Type'] = format
            return response
        else:
            reason = 'no query provided'
    except AttributeError as e:
        reason = e.message

    response = jsonify({'status': 'error', 'reason': reason})
    response.status_code = 400
    return response


@consumes(*DESCRIPTION_MIMES)
@produces(*DISCOVERY_MIMES)
def add_descriptions():
    descriptions = request.data
    if not descriptions:
        reason = 'no description/s provided'
    else:
        try:
            g = deserialize(descriptions, request.content_type)
            ted = learn_descriptions_from(g)
            format = TURTLE if request_wants_turtle() else JSONLD
            ted_str = serialize_TED(ted, format, prefixes=R.fountain.prefixes)

            eco_node = URIRef(REPOSITORY_BASE + url_for('get_ted'))
            VTED.update(ted, _get_thing_graph, eco_node)

            own_base = unicode(request.url_root)
            ted_str = ted_str.decode('utf-8').replace(REPOSITORY_BASE + u'/', own_base)
            response = make_response(ted_str)
            response.headers['Content-Type'] = format
            return response
        except ValueError as e:
            reason = e.message

    response = jsonify({'status': 'error', 'reason': reason})
    response.status_code = 400
    return response


def _url_for(endpoint):
    def wrapper(id):
        return url_for(endpoint, _external=True, id=id)

    return wrapper


def _get_thing_graph(td):
    g = td.resource.to_graph()

    def_g = ConjunctiveGraph(identifier=td.resource.node)
    for ns, uri in R.agora.fountain.prefixes.items():
        def_g.bind(ns, uri)

    for s, p, o in g:
        def_g.add((s, p, o))

    td_node = td.node

    if not list(def_g.objects(td.resource.node, CORE.describedBy)):
        def_g.add((td.resource.node, CORE.describedBy, td_node))
    return def_g


def get_td(id):
    try:
        td_node = get_td_node(id)
        g = R.pull(td_node, cache=True, infer=False, expire=300)
        for ns, uri in R.fountain.prefixes.items():
            g.bind(ns, uri)

        format = TURTLE if request_wants_turtle() else JSONLD
        ttl = serialize_graph(g, format, frame=CORE.ThingDescription)

        own_base = unicode(request.url_root)
        ttl = ttl.decode('utf-8').replace(REPOSITORY_BASE + u'/', own_base)
        response = make_response(ttl)
        response.headers['Content-Type'] = 'text/turtle'
        return response
    except IndexError:
        pass

    response = make_response()
    response.status_code = 404

    return response


def get_ted():
    try:
        local_node = URIRef(url_for('get_ted', _external=True))
        fountain = R.fountain
        known_types = fountain.types
        ns = R.ns()
        ted = TED()
        g = ted.to_graph(node=local_node, abstract=True)
        for root_uri, td_uri in VTED.roots:
            root_uri = URIRef(root_uri)
            types = get_th_types(root_uri, infer=True)
            valid_types = filter(lambda t: t.n3(ns) in known_types, types)
            if valid_types:
                r = Resource(root_uri, types=valid_types)
                if td_uri is None:
                    g.__iadd__(r.to_graph(abstract=True))
                g.add((ted.ecosystem.node, CORE.hasComponent, root_uri))

        format = TURTLE if request_wants_turtle() else JSONLD
        # g = ted.to_graph(node=local_node, abstract=True)
        for prefix, ns in fountain.prefixes.items():
            g.bind(prefix, ns)

        ted_str = serialize_graph(g, format, frame=CORE.ThingEcosystemDescription)
        own_base = unicode(request.url_root)
        ted_str = ted_str.decode('utf-8')
        ted_str = ted_str.replace(REPOSITORY_BASE + u'/', own_base)

        response = make_response(ted_str)
        response.headers['Content-Type'] = format
        return response
    except (EnvironmentError, IndexError):
        pass

    response = make_response()
    response.status_code = 404

    return response


def get_thing(id):
    try:
        th_node = get_th_node(id)
        g = R.pull(th_node, cache=True, infer=False, expire=300)

        for prefix, ns in R.fountain.prefixes.items():
            g.bind(prefix, ns)

        if not list(g.objects(th_node, CORE.describedBy)):
            td_node = get_td_node(id)
            g.add((th_node, CORE.describedBy, td_node))

        th_types = list(g.objects(URIRef(th_node), RDF.type))
        th_type = th_types.pop() if th_types else None

        format = TURTLE if request_wants_turtle() else JSONLD
        ttl = serialize_graph(g, format, frame=th_type)

        own_base = unicode(request.url_root)
        ttl = ttl.decode('utf-8').replace(REPOSITORY_BASE + u'/', own_base)
        response = make_response(ttl)
        response.headers['Content-Type'] = 'text/turtle'
        return response
    except IndexError:
        pass

    response = make_response()
    response.status_code = 404

    return response


@produces(JSON, HTML)
def get_namespaces():
    return jsonify(R.namespaces)


@consumes(JSON)
def add_namespaces():
    namespaces = request.json()
    R.add_namespaces(namespaces)
    return make_response()


def build(name):
    app = Flask(name)
    app.route('/namespaces')(get_namespaces)
    app.route('/namespaces', methods=['PATCH'])(add_namespaces)
    app.route('/things/<id>')(get_thing)
    app.route('/descriptions/<id>')(get_td)
    app.route('/ted')(get_ted)
    app.route('/discover', methods=['POST'])(discover)
    app.route('/descriptions', methods=['POST'])(add_descriptions)
    app.route('/extensions', methods=['POST'])(learn)
    app.route('/extensions')(get_extensions)
    app.route('/extensions/<id>')(get_extension)
    app.route('/extensions/<id>', methods=['PUT'])(learn_with_id)
    app.route('/extensions/<id>', methods=['DELETE'])(delete_extension)
    return app
