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

from abc import ABC, abstractmethod
from common.model_factory import *
from common.session import *
from common.model import Model
from common.device import Device
from common.options import Options


class OnnxModelPathNotSetException(Exception):
    def __init__(self):
        print("argument: '--model_path' is necessary for onnxruntime ")

class OnnxModelFactory(ModelFactory):
   name = 'onnx'
   model = 'undefined'


class OrtSession(BaseSession):
    def __init__(self, path_or_bytes, sess_options=None, **kwargs):
        import onnxruntime as rt
        self.sess = rt.InferenceSession(path_or_bytes, sess_options, kwargs)
        # self.__version = rt.__version__

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
        return self.sess.run(output_names, input_feed, run_options=None)

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
        pass

    def run(self, *args, **kwargs):
        sess = self.create_session()
        output = self.run_internal(sess, args, kwargs)
        return output

    def create_session(self) -> BaseSession:

        options = self.get_options()

        if hasattr(options, 'get_device'):
            device_name = options.get_device()
            Device.parse(device_name)
        else:
            Device.parse('gcu')
        device = Device()
        self.set_device(device)

        try:
            model_path = options.get_model_path()
        except AttributeError as ex:
            raise OnnxModelPathNotSetException()

        # sess_options is ignored as rarely used
        sess = SESSION_FACTORY['onnx-' + device.type](model_path)

        provider_options = [{}]
        if device.name == 'gcu':
            key_list = ['output_names', 'compiled_batchsize', 'export_executable', 'load_executable']

            for key in key_list:
                value = options.get(key)
                if value:
                    provider_options[0].update({key: value})

            provider_options[0].update({device: device.id})
            provider_options[0].update({cluster: device.cluster_ids})

        sess.set_providers(provider_options)
        return sess


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
