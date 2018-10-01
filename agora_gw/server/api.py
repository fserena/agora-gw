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

import traceback

from agora.engine.fountain.onto import DuplicateVocabulary
from agora.server import HTML
from flask import Flask, request, jsonify, make_response, url_for
from flask_negotiate import produces, consumes
from rdflib import URIRef, RDF

from agora_gw.data.repository import CORE
from agora_gw.ecosystem.serialize import serialize_TED, JSONLD, TURTLE, serialize_graph, deserialize
from agora_gw.gateway import Gateway

__author__ = 'Fernando Serena'

DISCOVERY_MIMES = [JSONLD, TURTLE]
DESCRIPTION_MIMES = [JSONLD, TURTLE]


def request_wants_turtle():
    best = request.accept_mimetypes \
        .best_match(DISCOVERY_MIMES)
    return best == TURTLE and \
           request.accept_mimetypes[best] > \
           request.accept_mimetypes[JSONLD]


def learn_with_id(gw):
    @consumes(TURTLE)
    def _learn_with_id(id):
        vocabulary = request.data
        try:
            g = deserialize(vocabulary, format=request.content_type)
            gw.add_extension(id, g)

            response = make_response()
            response.headers['Location'] = url_for('_get_extension', id=id, _external=True)
            response.status_code = 201
            return response
        except (AttributeError, DuplicateVocabulary, ValueError) as e:
            reason = e.message

        response = jsonify({'status': 'error', 'reason': reason})
        response.status_code = 400
        return response

    return _learn_with_id


def get_extension(gw):
    @produces(TURTLE, HTML)
    def _get_extension(id):
        g = gw.get_extension(id)
        response = make_response(g.serialize(format='turtle'))
        response.headers['Content-Type'] = 'text/turtle'
        return response

    return _get_extension


def get_extensions(gw):
    def _get_extensions():
        extensions = gw.repository.extensions
        return jsonify(extensions)

    return _get_extensions


def delete_extension(gw):
    def _delete_extension(id):
        gw.repository.delete_extension(id)
        response = make_response()
        return response

    return _delete_extension


def discover(gw):
    @produces(*DISCOVERY_MIMES)
    def _discover():
        query = request.data
        try:
            strict = request.args.get('strict')
            strict = True if strict is not None else False

            min = request.args.get('min')
            min = True if min is not None else False
            ted = gw.discover(query, strict=strict, lazy=min)

            format = TURTLE if request_wants_turtle() else JSONLD
            own_base = unicode(request.url_root)
            ted_str = serialize_TED(ted, format, min=min, abstract=min, prefixes=gw.repository.fountain.prefixes)
            ted_str = ted_str.decode('utf-8').replace(gw.repository.base.rstrip('/') + u'/', own_base)

            response = make_response(ted_str)
            response.headers['Content-Type'] = format
            return response
        except AttributeError as e:
            reason = e.message

        response = jsonify({'status': 'error', 'reason': reason})
        response.status_code = 400
        return response

    return _discover


def add_descriptions(gw):
    @consumes(*DESCRIPTION_MIMES)
    @produces(*DISCOVERY_MIMES)
    def _add_descriptions():
        descriptions = request.data
        try:
            g = deserialize(descriptions, format=request.content_type)
            ted = gw.add_description(g, ted_path=url_for('_get_ted'))
            format = TURTLE if request_wants_turtle() else JSONLD
            ted_str = serialize_TED(ted, format, prefixes=gw.repository.fountain.prefixes)

            own_base = unicode(request.url_root)
            ted_str = ted_str.decode('utf-8').replace(gw.repository.base.rstrip('/') + u'/', own_base)
            response = make_response(ted_str)
            response.headers['Content-Type'] = format
            return response
        except (AttributeError, ValueError) as e:
            reason = e.message

        response = jsonify({'status': 'error', 'reason': reason})
        response.status_code = 400
        return response

    return _add_descriptions


def get_td(gw):
    def _get_td(id):
        try:
            td = gw.get_description(id, fetch=False)
            g = td.to_graph()
            format = TURTLE if request_wants_turtle() else JSONLD
            ttl = serialize_graph(g, format, frame=CORE.ThingDescription)

            own_base = unicode(request.url_root)
            ttl = ttl.decode('utf-8').replace(gw.repository.base.rstrip('/') + u'/', own_base)
            response = make_response(ttl)
            response.headers['Content-Type'] = format
            return response
        except IndexError:
            pass

        response = make_response()
        response.status_code = 404

        return response

    return _get_td


def delete_td(gw):
    def _delete_td(id):
        try:
            gw.delete_description(id)
            response = make_response()
            return response
        except IndexError:
            pass

        response = make_response()
        response.status_code = 404

        return response

    return _delete_td


def get_ted(gw):
    def _get_ted():
        fountain = gw.repository.fountain
        try:
            local_node = URIRef(url_for('_get_ted', _external=True))
            ted = gw.get_ted(ted_uri=local_node, fountain=fountain, lazy=True)
            g = ted.to_graph(node=local_node, fetch=False)

            format = TURTLE if request_wants_turtle() else JSONLD

            ted_str = serialize_graph(g, format, frame=CORE.ThingEcosystemDescription)
            own_base = unicode(request.url_root)
            ted_str = ted_str.decode('utf-8')
            ted_str = ted_str.replace(gw.repository.base.rstrip('/') + u'/', own_base)

            response = make_response(ted_str)
            response.headers['Content-Type'] = format
            return response
        except (EnvironmentError, IndexError, AttributeError):
            traceback.print_exc()
            pass

        response = make_response()
        response.status_code = 404

        return response

    return _get_ted


def get_thing(gw):
    def _get_thing(id):
        try:
            g = gw.get_thing(id, lazy=True).to_graph()
            th_node = g.identifier
            th_types = list(g.objects(URIRef(th_node), RDF.type))
            th_type = th_types.pop() if th_types else None

            format = TURTLE if request_wants_turtle() else JSONLD
            ttl = serialize_graph(g, format, frame=th_type)

            own_base = unicode(request.url_root)
            ttl = ttl.decode('utf-8').replace(gw.repository.base.rstrip('/') + u'/', own_base)
            response = make_response(ttl)
            response.headers['Content-Type'] = format
            return response
        except (IndexError, AttributeError):
            traceback.print_exc()
            pass

        response = make_response()
        response.status_code = 404

        return response

    return _get_thing


def build(name, gw=None, **kwargs):
    if gw is None:
        gw = Gateway(**kwargs)
    app = Flask(name)
    app.route('/things/<id>')(get_thing(gw))
    app.route('/descriptions/<id>')(get_td(gw))
    app.route('/descriptions/<id>', methods=['DELETE'])(delete_td(gw))
    app.route('/ted')(get_ted(gw))
    app.route('/discover', methods=['POST'])(discover(gw))
    app.route('/descriptions', methods=['POST'])(add_descriptions(gw))
    app.route('/extensions')(get_extensions(gw))
    app.route('/extensions/<id>')(get_extension(gw))
    app.route('/extensions/<id>', methods=['PUT'])(learn_with_id(gw))
    app.route('/extensions/<id>', methods=['DELETE'])(delete_extension(gw))
    return app, gw
