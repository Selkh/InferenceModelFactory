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


import time
from abc import abstractmethod
from functools import partial
from common.model_factory import ModelFactory
from common.session import BaseSession, SESSION_FACTORY, register_session
from common.model import Model
from common.device import Device
from common.dataset import Item
import onnxruntime as rt


class OnnxModelPathNotSetException(Exception):
    def __init__(self):
        print("argument: '--model_path' is necessary for onnxruntime ")


class OnnxModelFactory(ModelFactory):
    name = 'onnx'
    model = 'undefined'


class OrtSession(BaseSession):
    def __init__(self, path_or_bytes, sess_options=None, **kwargs):
        if isinstance(path_or_bytes, str):
            import onnx
            self.onnx_bytes = onnx.load(path_or_bytes)
        elif isinstance(path_or_bytes, bytes)
            self.onnx_bytes = path_or_bytes
        else:
            raise TypeError("Not supported type '' for onnx session created".format(type(path_or_bytes)))

        self.sess = rt.InferenceSession(self.onnx_bytes.SerializeToString())
        # self._version = rt.__version__

        self.loop = 0
        self.time = []

    def __getstate__(self):
        return {'onnx_bytes': self.onnx_bytes}

    def __setstate__(self, values):
        self.onnx_bytes = value['onnx_bytes']
        self.sess = rt.InferenceSession(self.onnx_bytes.SerializeToString())

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
        self.loop += 1
        start = time.time()
        output = self.sess.run(output_names, input_feed, run_options=None)
        end = time.time()
        self.time.append(end - start)
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
        return [{}]


@register_session
class OrtCPUSession(OrtSession):

    def name(self):
        return "onnx-cpu"

    def set_providers(self, provider_options=None):
        providers = ['CPUExecutionProvider']
        super().set_providers(providers, provider_options)


@register_session
class OrtGPUSession(OrtSession):

    def name(self):
        return "onnx-gpu"

    def set_providers(self, provider_options=None):
        providers = ['CUDAExecutionProvider']
        super().set_providers(providers, provider_options)


@register_session
class OrtGCUSession(OrtSession):

    def name(self):
        return "onnx-gcu"

    def set_providers(self, provider_options=None):
        providers = ['TopsInferenceExecutionProvider']
        super().set_providers(providers, provider_options)

    def generate_provider_options(self, options, device):
        provider_options = [{}]
        key_list = ['output_names', 'compiled_batchsize',
                    'export_executable', 'load_executable']

        for key in key_list:
            value = options.get(key)
        if value:
            provider_options[0].update({key: value})

        provider_options[0].update({'device': device.id})
        provider_options[0].update({'cluster': device.cluster_ids})
        return provider_options


class OnnxModel(Model):

    @abstractmethod
    def run_internal(self, session: BaseSession, datas):
        return []

    def run(self):
        self.sanity_check()
        options = self.get_options()

        batch_size = 3
        if hasattr(options, '_batch_size'):
            batch_size = options.get_batch_size()
        elif hasattr(options, '_batchsize'):
            batch_size = options.get_batchsize()
        elif hasattr(options, '_bs'):
            batch_size = options.get_bs()

        self.dataset = self.create_dataset()

        pipe = self.dataset.window(blocks_per_window=1)
        pipe = pipe.map(self.load_data)
        pipe = pipe.map(self.preprocess)

        sess = self.create_session_by_options(options)

        class BatchInfer:
            def __init__(self, fn):
                self.fn = fn

            def __call__(self, items):
                if isinstance(items[0], Item):
                    # Derived class from common.dataset.Item
                    batch = Model.make_batch([item.data for item in items])
                else:
                    # Not identified data
                    batch = Model.make_batch(items)

                outputs = self.fn(sess, batch)

                def assignment(item, value):
                    if hasattr(item, 'final_result'):
                        raise AttributeError(
                            "attribute 'final_result' has already be used,"
                            "please modify your derived Item class")
                    try:
                        setattr(item, "final_result", value)
                    except AttributeError:
                        print("item itself is data with builtin type")
                        return False
                    return True

                result = set([assignment(*z) for z in zip(items, outputs)])
                if len(result) > 1:
                    raise RuntimeError("Partial error during run_internal")

                return items if result.pop() else outputs

        pipe = pipe.map_batches(
            BatchInfer(self.run_internal),
            compute="actors",
            batch_size=batch_size,
            drop_last=True)
        pipe = pipe.map(self.postprocess)

        correct = 0
        for row in pipe.iter_rows():
            # print(batch[0].data.shape)
            correct += row

        print(correct)

    def create_session(self, device_name: str, model_path: str) -> BaseSession:
        device = Device.parse(device_name)
        self.set_device(device)

        sess = SESSION_FACTORY['onnx-' + device.type](model_path)

        options = self.get_options()
        provider_options = sess.generate_provider_options(options, device)

        sess.set_providers(provider_options)
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
