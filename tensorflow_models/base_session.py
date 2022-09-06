"""
/* Copyright 2018 The Enflame Tech Company. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
"""
# !/usr/bin/python
# -*- coding: utf-8 -*-

from functools import partial
from common.session import BaseSession, register_session
import tensorflow as tf


class TFSession(BaseSession):
    def __init__(self, target='', graph=None, config=None):
        self.sess = tf.Session(target, graph, config)

    def reset(self, target, containers=None, config=None):
        self.sess.reset(target, containers, config)

    def list_devices(self):
        return self.sess.list_devices()

    def close(self):
        self.sess.close()

    @property
    def graph(self):
        return self.sess.graph

    @property
    def graph_def(self):
        return self.sess.graph_def

    @property
    def sess_str(self):
        return self.sess.sess_str

    def as_default(self):
        return self.sess.as_default()

    def run(self, fetches, device_str, feed_dict=None, options=None,
            run_metadata=None):
        self.sess.graph.device(device_str)
        output = self.sess.run(fetches, feed_dict, options, run_metadata)
        return output

    def partial_run(self, handle, fetches, feed_dict=None):
        return self.sess.partial_run(handle, fetches, feed_dict)

    def partial_run_setup(self, fetches, feeds=None):
        return self.sess.partial_run_setup(fetches, feeds)


@register_session
class TFCPUSession(TFSession):
    def __init__(self, device_id, target='', graph=None, config=None):
        super(TFCPUSession, self).__init__(target, graph, config)
        self.device_str = '/cpu:' + device_id

    def name(self):
        return "tf-cpu"

    @partial
    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return partial(super().run, device_str=self.device_str)


@register_session
class TFGPUSession(TFSession):
    def __init__(self, device_id, target='', graph=None, config=None):
        super(TFCPUSession, self).__init__(target, graph, config)
        self.device_str = '/gpu:' + device_id

    def name(self):
        return "tf-gpu"

    @partial
    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return partial(super().run, device_str=self.device_str)


@register_session
class TFGCUSession(TFSession):
    def __init__(self, device_id, target='', graph=None, config=None):
        super(TFCPUSession, self).__init__(target, graph, config)
        self.device_str = '/device:XLA_DTU:' + device_id

    def name(self):
        return "tf-gcu"

    @partial
    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return partial(super().run, device_str=self.device_str)
