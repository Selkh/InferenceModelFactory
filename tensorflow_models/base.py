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

from abc import abstractmethod
from common.model_factory import *
from common.session import *
from common.model import Model
from common.device import Device
from common.options import Options
import tensorflow_models.base_session


class TFModelFactory(ModelFactory):
    name = 'tf'
    model = 'undefined'


class TFModel(Model):

    @abstractmethod
    def construct_graph(self):
        # This method must be overrided in concrete model implement, whether load from a freezed file or construct with Tensorflow API and thus output of this function could be either a 'Graph' or a 'Tensor'.
        pass

    @abstractmethod
    def run_internal(self, session: BaseSession):
        pass

    def create_session(self):
        options = self.get_options()

        graph_or_tensor = self.construct_graph()
        if type(graph_or_tensor).__name__ == 'Graph':
            graph = graph_or_tensor
        elif type(graph_or_tensor).__name__ == 'Tensor':
            graph = graph_or_tensor.graph
        else:
            raise TypeError(
                "Output of 'construct_graph' must be either 'Graph' or 'Tensor', but got {}".format(
                    type(graph_or_tensor).__name__))

        device_name = options.get_device() if hasattr(options,
                                                      'get_device') else 'gcu'
        device = Device.parse(device_name)
        self.set_device(device)

        target = options.get_target() if hasattr(options, 'get_target') else ''

        config = tf.ConfigProto()
        config.allow_soft_placement = options.get_allow_soft_placement() if hasattr(
            options, 'get_allow_soft_placement') else True
        config.log_soft_placement = options.get_log_soft_placement() if hasattr(
            options, 'get_log_soft_placement') else False

        sess = SESSION_FACTORY['tf-' + device.type](device.id, target=target,
                                                    graph=graph, config=config)
        return sess

    def run(self, *args, **kwargs):
        sess = self.create_session()
        output = self.run_internal(sess, args, kwargs)
        return output
