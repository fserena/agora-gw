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

from multiprocessing import Lock

from agora import setup_logging
from gunicorn.app.base import BaseApplication
from gunicorn.six import iteritems
from werkzeug.exceptions import NotFound

from agora_gw import LOG_LEVEL
from agora_gw.server.api import build

__author__ = 'Fernando Serena'


class Application(BaseApplication):
    def init(self, **kwargs):
        if self.default_app is None:
            self.default_app, self.gw = build(__name__, **kwargs)

    def __init__(self, options=None):
        self.default_app = None
        self.gw = None
        self.lock = Lock()
        self.options = options or {}
        with self.lock:
            self.init(**self.options)
        super(Application, self).__init__()

    def load_config(self):
        config = dict([(key, value) for key, value in iteritems(self.options)
                       if key in self.cfg.settings and value is not None])
        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)

        self.cfg.set('worker_class', 'agora_gw.server.worker.Worker')

    def load(self):
        return self.default_app

    def test_default(self, environ):
        adapter = self.default_app.url_map.bind_to_environ(environ)
        try:
            res = adapter.match()
        except NotFound:
            res = ()
        return self.default_app if res else None
