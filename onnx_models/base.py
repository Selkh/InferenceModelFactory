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


import os
import time
from abc import abstractmethod
from functools import partial
from common.model_factory import ModelFactory
from common.session import BaseSession, SESSION_FACTORY, register_session
from common.model import Model
from common.device import Device
import onnxruntime as rt
import ray


class OnnxModelPathNotSetException(Exception):
    def __init__(self):
        print("argument: '--model_path' is necessary for onnxruntime ")


class OnnxModelFactory(ModelFactory):
    name = 'onnx'
    model = 'undefined'


class OrtSession(BaseSession):
    _loop = 0
    _time = []

    def __init__(self,
                 path_or_bytes,
                 sess_options=None,
                 providers=None,
                 provider_options=None, **kwargs):
        # if isinstance(path_or_bytes, str):
        #     import onnx
        #     self.onnx_bytes = onnx.load(path_or_bytes)
        # elif isinstance(path_or_bytes, bytes):
        #     self.onnx_bytes = path_or_bytes
        # else:
        # raise TypeError(
        #     "Not supported type '{}' for onnx session".format(
        #         type(path_or_bytes)))

        # self.sess = rt.InferenceSession(self.onnx_bytes.SerializeToString())
        # self._version = rt.__version__
        if isinstance(path_or_bytes, str):
            self.path = path_or_bytes
        else:
            raise TypeError(
                "Not supported type '{}' by now".format(type(path_or_bytes)))

        self.sess_options = sess_options
        self.providers = providers
        self.provider_options = provider_options

        if 'disabled_optimizers' in kwargs:
            self.kwargs = {
                'disabled_optimizers': kwargs['disabled_optimizers']}
        else:
            self.kwargs = {}

        self.sess = rt.InferenceSession(
            self.path, sess_options, providers, provider_options, **kwargs)

    def __getstate__(self):
        # return {'onnx_bytes': self.onnx_bytes}
        return {'path': self.path,
                'sess_options': self.sess_options,
                'providers': self.providers,
                'provider_options': self.provider_options,
                'kwargs': self.kwargs}

    def __setstate__(self, values):
        # self.onnx_bytes = values['onnx_bytes']
        # self.sess = rt.InferenceSession(self.onnx_bytes.SerializeToString())
        self.path = values['path']
        self.sess_options = values['sess_options']
        self.providers = values['providers']
        self.provider_options = values['provider_options']
        self.kwargs = values['kwargs']
        self.sess = rt.InferenceSession(
            self.path,
            self.sess_options,
            self.providers,
            self.provider_options,
            **self.kwargs)

    @property
    def version(self):
        # return "1.9.1"
        # return self._version
        return rt.__version__

    def get_session_options(self):
        return self.sess.get_session_options()

    def get_inputs(self):
        return self.sess.get_inputs()

    def get_outputs(self):
        return self.sess.get_outputs()

    def get_overridable_initializers(self):
        return self.sess.get_overridable_initializers()

    def get_modelmeta(self):
        return self.sess.get_modelmeta()

    def get_providers(self):
        return self.sess.get_providers()

    def get_provider_options(self):
        return self.sess.get_provider_options()

    def set_providers(self, providers=None, provider_options=None):
        self.sess.set_providers(providers, provider_options)

    def disable_fallback(self):
        self.sess.disable_fallback()

    def enable_fallback(self):
        self.sess.enable_fallback()

    def run(self, output_names, input_feed, run_options=None):
        self._loop += 1
        start = time.time()
        output = self.sess.run(output_names, input_feed, run_options=None)
        end = time.time()
        self._time.append(end - start)
        return output

    def run_with_ort_values(self, output_names, input_dict_ort_values,
                            run_options=None):
        return self.sess.run_with_ort_values(output_names,
                                             input_dict_ort_values,
                                             run_options=None)

    def end_profiling(self):
        self.sess.end_profiling()

    def get_profiling_start_time_ns(self):
        return self.sess.get_profiling_start_time_ns()

    def io_binding(self):
        return self.sess.io_binding()

    def run_with_iobinding(self, iobinding, run_options=None):
        return self.sess.run_with_iobinding(iobinding, run_options=None)

    def generate_provider_options(self, options, device):
        return [{}], {}


@register_session
class OrtCPUSession(OrtSession):
    def __init__(self, path_or_bytes, sess_options=None, **kwargs):
        if 'model_options' in kwargs:
            model_options = kwargs['model_options']
        else:
            model_options = None
        provider_options, kwargs = self.generate_provider_options(
            model_options)

        super().__init__(
            path_or_bytes,
            sess_options=sess_options,
            providers=['CPUExecutionProvider'],
            provider_options=provider_options,
            **kwargs)

    def name(self):
        return "onnx-cpu"

    def generate_provider_options(self, options=None):
        key = 'disabled_optimizers'
        if options and options.get(key):
            return None, {key: options.get(key)}
        else:
            return None, {}


@register_session
class OrtGPUSession(OrtSession):
    def __init__(self, path_or_bytes, sess_options=None, **kwargs):
        if 'model_options' in kwargs:
            model_options = kwargs['model_options']
        else:
            model_options = None
        provider_options, kwargs = self.generate_provider_options(
            model_options)

        super().__init__(
            path_or_bytes,
            sess_options=sess_options,
            providers=['CUDAExecutionProvider'],
            provider_options=provider_options,
            **kwargs)

    def name(self):
        return "onnx-gpu"

    def generate_provider_options(self, options=None):
        key = 'disabled_optimizers'
        if options and options.get(key):
            return None, {key: options.get(key)}
        else:
            return None, {}


@register_session
class OrtGCUSession(OrtSession):
    def __init__(self, path_or_bytes, sess_options=None, **kwargs):
        if 'model_options' in kwargs:
            model_options = kwargs['model_options']
        else:
            model_options = None
        if 'model_device' in kwargs:
            model_device = kwargs['model_device']
        else:
            model_device = None

        provider_options, kwargs = self.generate_provider_options(
            model_options, model_device)

        super().__init__(
            path_or_bytes,
            sess_options=sess_options,
            providers=['TopsInferenceExecutionProvider'],
            provider_options=provider_options,
            **kwargs
        )

    def name(self):
        return "onnx-gcu"

    def generate_provider_options(self, options=None, device=None):

        provider_options = [{}]
        kwargs = {}
        key_list = ['output_names', 'compiled_batchsize',
                    'export_executable', 'load_executable']

        if options:
            for key in key_list:
                value = options.get(key)
                if value:
                    provider_options[0].update({key: value})
            if options.get('disabled_optimizers'):
                kwargs['disabled_optimizers'] = options.get(
                    'disabled_optimizers')
        if device:
            provider_options[0].update({'device': device.id})
            if len(device.cluster_ids):
                provider_options[0].update({'cluster': device.cluster_ids})

        return provider_options, kwargs


class OnnxModel(Model):

    @abstractmethod
    def run_internal(self, session: BaseSession, datas):
        return []

    def run(self):
        self.sanity_check()
        options = self.get_options()

        self.dataset = self.create_dataset()
        count = self.dataset.count()

        batch_size = self.get_batch_size(options)  # default as 1

        step = options.get_step()
        epoch = options.get_epoch()
        if step > 0:
            if epoch > 1:
                raise ValueError("Multiple epochs not supported when only tend"
                                 " to perform several step not full dataset")
            assert step * batch_size <= count, \
                'Dataset has {} items, not enough for specidied ' \
                'step {} and batch size {}'.format(count, step, batch_size)
            self.dataset, _ = self.dataset.split_at_indices(
                [step * batch_size])
        else:
            step = count // batch_size
            if hasattr(options, 'get_drop_last') and options.get_drop_last():
                self.dataset, _ = self.dataset.split_at_indices(
                    [count - count % batch_size])

        if os.environ.get('inference_models_internal_debug'):
            # disable pipeline, serialize execution, only for internal debug
            # hopefully never use
            self.dataset = self.dataset.repartition(num_blocks=1)
        else:
            self.dataset = self.dataset.repartition(num_blocks=step)

        pipe = self.dataset.window(blocks_per_window=8).repeat(epoch)
        
        pipe = pipe.map(self.load_data)
        pipe = pipe.map(self.preprocess)

        sess = self.create_session_by_options(options)
        ray_sess = ray.put(sess)

        class BatchInfer:
            def __init__(self, fn):
                self.fn = fn

            def __call__(self, items):
                return self.fn(ray.get(ray_sess), items)

        pipe = pipe.map_batches(
            BatchInfer(self.run_internal),
            compute="actors",
            batch_size=batch_size)
        pipe = pipe.map(self.postprocess)

        collections = pipe.take_all(self.dataset.count() * epoch)
        return self.eval(collections)

    def create_session(self, device_name: str, model_path: str) -> BaseSession:
        device = Device.parse(device_name)
        self.set_device(device)

        options = self.get_options()

        sess = SESSION_FACTORY['onnx-' +
                               device.type](model_path,
                                            model_options=options,
                                            model_device=device)

        return sess

    def create_session_by_options(self, options) -> BaseSession:
        if hasattr(options, 'get_device'):
            device_name = options.get_device()
        else:
            device_name = 'gcu'

        if hasattr(options, 'get_model_path'):
            model_path = options.get_model_path()
        else:
            raise OnnxModelPathNotSetException()

        return self.create_session(device_name, model_path)

    def create_session_func_by_device(self, device_name: str):
        # create multi-session on the same device with differente model paths
        return partial(self.create_session, device_name=device_name)

    def create_session_func_by_model(self, model_path: str):
        # create multi-session on different devices with the same path
        return partial(self.create_session, model_path=model_path)
