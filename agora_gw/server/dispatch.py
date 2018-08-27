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

import logging
import os
import traceback
from multiprocessing import Lock

from gunicorn.http.wsgi import Response
from werkzeug.exceptions import MethodNotAllowed

__author__ = 'Fernando Serena'

log = logging.getLogger('gateway.server.dispatcher')

PUBLIC_NAME = os.environ.get('PUBLIC_NAME', 'localhost')


class Dispatcher(object):
    def __init__(self, app, create_app):
        self.create_app = create_app
        self.app = app
        self.lock = Lock()

    def get_sub_app(self, environ):
        try:
            app = self.create_app(environ)
            if app is None:
                environ['PATH_INFO'] = '/not-found'
                log.debug('Forcing 404')
            return app
        except AttributeError as e:
            traceback.print_exc()
            raise e

    def get_application(self, environ):
        host = environ['HTTP_HOST']
        method = environ['REQUEST_METHOD']
        hostname = host.split(':')[0]

        log.debug('Request environment: host={}, method={}'.format(hostname, method))

        # Path dispatching
        app = self.app.test_default(environ)
        if not app:
            # environ['PATH_INFO'] = '/gateways' + environ['PATH_INFO']
            # environ['SCRIPT_NAME'] = environ['PATH_INFO']
            app = self.get_sub_app(environ)
        return app

    def __call__(self, environ, start_response):
        try:
            app = self.get_application(environ)
            if app is None:
                app = self.app.default_app
            response = app(environ, start_response)
        except MethodNotAllowed as e:
            response = self.app.default_app(environ, start_response)
            response.status_code = e.code

        return response
