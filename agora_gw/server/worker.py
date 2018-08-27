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

import multiprocessing
from multiprocessing import Lock

from agora import Agora
from agora.server.fountain import build as bn
from agora.server.planner import build as bp
from gunicorn.workers.sync import SyncWorker
from rdflib import Graph

from agora_gw.server.dispatch import Dispatcher

__author__ = 'Fernando Serena'


def number_of_workers():
    return (multiprocessing.cpu_count() * 2) + 1


_lock = Lock()


class Worker(SyncWorker):
    def __init__(self, *args, **kwargs):
        self.agora_app = None
        list(Graph().query('SELECT * WHERE {}'))
        super(Worker, self).__init__(*args, **kwargs)

    def init_process(self):
        super(Worker, self).init_process()

    def run(self):
        with _lock:
            self.wsgi = self.create()
        super(Worker, self).run()

    def handle_quit(self, sig, frame):
        try:
            Agora.close()
        except Exception:
            pass
        super(Worker, self).handle_quit(sig, frame)

    def create(self):
        return Dispatcher(self.app, self.create_sub_app())

    def create_sub_app(self):
        def creator(environ):
            with _lock:
                if self.agora_app is None:
                    self.agora_app = bn(self.app.gw.repository.agora.fountain, import_name=__name__)
                    bp(self.app.gw.repository.agora.planner, server=self.agora_app)
                return self.agora_app

        return creator
