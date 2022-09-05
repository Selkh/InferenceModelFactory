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
#!/usr/bin/python
# -*- coding: utf-8 -*-

# try:
#     import onnxruntime as rt
# except ModuleNotFoundError as ex:
#     print(ex)

import time
from abc import ABC, abstractmethod
from common.model_factory import *
from common.session import *
from common.model import Model
from common.device import Device
from common.options import Options
import onnxruntime as rt


class OnnxModelPathNotSetException(Exception):
    def __init__(self):
        print("argument: '--model_path' is necessary for onnxruntime ")

class OnnxModelArgumentException(Exception):
    def __init__(self, args: str):
        print("method 'run_internal' can have only one positional argument with type 'BaseSession', but additionally got: {}".format(args))

class OnnxModelFactory(ModelFactory):
   name = 'onnx'
   model = 'undefined'


class OrtSession(BaseSession):
    def __init__(self, path_or_bytes, sess_options=None, **kwargs):
        self.sess = rt.InferenceSession(path_or_bytes, sess_options, **kwargs)
        # self.__version = rt.__version__

        self.loop = 0
        self.time = []

    @property
    def version(self):
        return "1.9.1"
        # return self.__version

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

    def run_with_ort_values(self, output_names, input_dict_ort_values, run_options=None):
        return self.sess.run_with_ort_values(output_names, input_dict_ort_values, run_options=None)

    def end_profiling(self):
        self.sess.end_profiling()

    def get_profiling_start_time_ns(self):
        return self.sess.get_profiling_start_time_ns()

    def io_binding(self):
        return self.sess.io_binding()

    def run_with_iobinding(self, iobinding, run_options=None):
        return self.sess.run_with_iobinding(iobinding, run_options=None)

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


class OnnxModel(Model):

    @abstractmethod
    def run_internal(self, session: BaseSession):
        return []

    def run(self):
        options = self.get_options()

        argcount = self.run_internal.__code__.co_argcount
        if argcount > 2:
            raise OnnxModelArgumentException(str(self.run_internal.__code__.co_varnames[2: argcount])[1:-1])

        sess = self.create_session_by_options(options)
        output = self.run_internal(sess)
        return output

    def create_session(self, device_name: str, model_path: str) -> BaseSession:
        device = Device.parse(device_name)
        self.set_device(device)

        sess = SESSION_FACTORY['onnx-' + device.type](model_path)

        provider_options = [{}]
        if device.name == 'gcu':
            key_list = ['output_names', 'compiled_batchsize', 'export_executable', 'load_executable']

            for key in key_list:
                value = self.options.get(key)
                if value:
                    provider_options[0].update({key: value})

            provider_options[0].update({'device': device.id})
            provider_options[0].update({'cluster': device.cluster_ids})

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

    def create_session_func_by_device(self, device: str):
        # Used to create multi-session on the same device with differente model paths
        return partial(self.create_session, device = device)

    def create_session_func_by_model(self, model_path: str):
        # Used to create multi-session on different devices with the same path
        return partial(self.create_session, model_path = model_path)

    # def __new__(cls, *args, **kwargs):
    #     out_cls = super(OnnxModel, cls).__new__(cls, *args, **kwargs)
    #     if cls.__dict__['run_internal'].__code__.co_argcount > 2:
    #         raise OnnxModelArgumentException(cls.__dict__['run_internal'].__code__.co_varnames[2:])


    # def create_session(self, options) -> BaseSession:

    #     if hasattr(options, 'get_device'):
    #         device_name = options.get_device()
    #         Device.parse(device_name)
    #     else:
    #         Device.parse('gcu')
    #     device = Device()
    #     self.set_device(device)

    #     try:
    #         model_path = options.get_model_path()
    #     except AttributeError as ex:
    #         raise OnnxModelPathNotSetException()

    #     # sess_options is ignored as rarely used
    #     sess = SESSION_FACTORY['onnx-' + device.type](model_path)

    #     provider_options = [{}]
    #     if device.name == 'gcu':
    #         key_list = ['output_names', 'compiled_batchsize', 'export_executable', 'load_executable']

    #         for key in key_list:
    #             value = options.get(key)
    #             if value:
    #                 provider_options[0].update({key: value})

    #         provider_options[0].update({'device': device.id})
    #         provider_options[0].update({'cluster': device.cluster_ids})

    #     sess.set_providers(provider_options)
    #     return sess


   # def __init__(cls):
   #     pass
      # print("================================")
      # if len(cls.__abstractmethods__) > 0:
      #     for name in cls.__abstractmethods__:
      #         if name in cls.__dict__.keys():
      #             cls.__abstractmethods__.clear()
      #         else:
      #              raise TypeError()

   # def new_model() -> BaseEngine:
   #     raise NotImplementedError("Base onnx model factory has no concrete implement for new model. Each model registered should fill specific one.")
