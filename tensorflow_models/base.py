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
import tensorflow as tf
import ray
from common.model_factory import ModelFactory
from common.model import Model
from common.device import Device
from common.session import SESSION_FACTORY
from tensorflow_models.base_session import TFSession


class TFModelFactory(ModelFactory):
    name = 'tf'
    model = 'undefined'


class TFModel(Model):

    @abstractmethod
    def construct_graph(self):
        # This method must be overrided in concrete model
        # implement, whether load from a freezed file or
        # construct with Tensorflow API and thus output of
        # this function could be either a 'Graph' or a 'Tensor'.
        pass

    @abstractmethod
    def run_internal(self, sess: TFSession):
        pass

    def set_config(self, config):
        return config

    def create_session(self):
        options = self.get_options()

        graph_or_tensor = self.construct_graph()
        if type(graph_or_tensor).__name__ == 'Graph':
            graph = graph_or_tensor
        elif type(graph_or_tensor).__name__ == 'Tensor':
            graph = graph_or_tensor.graph
        else:
            raise TypeError(
                "Expected output of 'construct_graph' must be with type "
                "of either 'tf.Graph' or 'tf.Tensor', but got error type: "
                "'{}'".format(type(graph_or_tensor)))

        device_name = options.get_device() if hasattr(options,
                                                      'get_device') else 'gcu'
        device = Device.parse(device_name)
        self.set_device(device)

        target = options.get_target() if hasattr(options, 'get_target') else ''

        config = self.set_config(tf.ConfigProto())

        sess = SESSION_FACTORY['tf-' + device.type](device.id, target=target,
                                                    graph=graph, config=config)
        return sess

    def run(self, *args, **kwargs):
        self.sanity_check()
        options = self.get_options()
        batch_size = self.get_batch_size(options)  # default as 1

        self.dataset = self.create_dataset()

        import os
        if os.environ.get('inference_models_internal_debug'):
            # disable pipeline, serialize execution, only for internal debug
            # hopefully never use
            self.dataset = self.dataset.repartition(num_blocks=1)

        pipe = self.dataset.window(blocks_per_window=1)
        pipe = pipe.map(self.load_data)
        pipe = pipe.map(self.preprocess)

        sess = self.create_session()
        ray_sess = ray.put(sess)

        class BatchInfer:
            def __init__(self, fn):
                self.fn = fn

            def __call__(self, items):
                return self.fn(ray.get(ray_sess), items)

        pipe = pipe.map_batches(
            BatchInfer(self.run_internal),
            compute="actors",
            batch_size=batch_size,
            drop_last=True)
        pipe = pipe.map(self.postprocess)

        collections = pipe.take()
        return self.eval(collections)
