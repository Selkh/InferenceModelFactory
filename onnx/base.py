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


from common.model_factory import *
from common.session import optional_sessions
from engine import BaseEngine

ONNXSession = optional_sessions['onnx']

class OnnxModelFactory(ModelFactory):
   name = 'onnx'
   model = 'undefined'

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
